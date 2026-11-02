
"""ML-053: tokenizer-only selection and sealing of a fresh paraphrase test.

No model forward passes, log probabilities, or experimental result files are read.
Do not overwrite the ML-053 manifest after sealing. Commit script + manifest
before creating any calibration or protected output.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/ml053_design_manifest.json"

TRAIN_NAMES = (
    "Abigail", "Andrew", "Brooke", "Brandon", "Celia", "Dmitri",
    "Eleanor", "Felix", "Georgia", "Hector", "Indira", "Julian",
    "Kelsey", "Landon", "Megan", "Nicholas", "Opal", "Preston",
    "Rosa", "Sebastian", "Teresa", "Uriel", "Valerie", "William",
    "Yara", "Zander", "Alicia", "Colton", "Daisy", "Emmett",
    "Hannah", "Noel",
)

TEST_NAMES = (
    "Adeline", "Bennett", "Camille", "Damian", "Eloise", "Franklin",
    "Genevieve", "Holden", "Imogen", "Jasper", "Katherine", "Levi",
    "Madeline", "Nelson", "Octavia", "Parker", "Rosalie", "Sterling",
    "Tabitha", "Ulric", "Vanessa", "Wyatt", "Yvette", "Zane",
    "Anita", "Bruno", "Clara", "Dorian", "Esme", "Frederick",
    "Greta", "Howard",
)

TRAIN_VERBS = (
    "confronted", "motivated", "teased", "instructed", "challenged",
    "approached", "mentored", "encountered", "persuaded", "startled",
    "interrupted", "notified",
)

TEST_VERBS = (
    "escorted", "observed", "reminded", "coached", "inspected",
    "consulted", "entertained", "criticized", "questioned",
    "comforted", "supported", "advised",
)

CONSTRUCTIONS = ("active", "object_cleft", "passive_cleft")
STYLES = ("generic", "paraphrase", "verb_explicit")
ROLES = ("agent", "patient")
N_GROUPS = {"calibration": 6, "protected": 8}
MODEL_NAMES = ("qwen7b", "llama3b")

SPEC = {
    "experiment": "ML-053",
    "status": "prospective behavior test; no causal claims",
    "hypothesis": (
        "Qwen passive-cleft role-query reversal persists when the role "
        "question is paraphrased without copying the event verb"
    ),
    "primary_unit": "independent name-pair/verb group",
    "primary_gate": (
        "Across eight protected lexical groups, Qwen has mean D < 0 for "
        "passive_cleft/paraphrase in >=6 groups and mean D > 0 for "
        "active/paraphrase in >=6 groups. Groups average both role-swapped "
        "event directions. Report exact magnitudes and all failures."
    ),
    "D_definition": (
        "[logP(agent|agent-question)-logP(patient|agent-question)] "
        "- [logP(agent|patient-question)-logP(patient|patient-question)]"
    ),
    "secondary": (
        "generic and verb-explicit prompt sensitivity, actual answer "
        "switches, object-cleft control, Llama replication/contrast"
    ),
    "no_posthoc_gate_changes": True,
    "calibration_groups": N_GROUPS["calibration"],
    "protected_groups": N_GROUPS["protected"],
    "constructions": CONSTRUCTIONS,
    "styles": STYLES,
    "roles": ROLES,
    "two_agent_patient_directions": True,
    "selection_rule": (
        "Visit verbs in declared order. For each verb, choose the first "
        "pair of as-yet unused names in listed order whose complete-name "
        "continuations have equal, nonzero token counts in both models "
        "for all ML-053 prompts. Stop after 6/8 groups, respectively. "
        "No model logits or outcomes may enter selection."
    ),
    "train_names": TRAIN_NAMES,
    "test_names": TEST_NAMES,
    "train_verbs": TRAIN_VERBS,
    "test_verbs": TEST_VERBS,
}


def passage(agent, patient, verb, construction):
    if construction == "active":
        return f"At dawn, {agent} {verb} {patient}."
    if construction == "object_cleft":
        return f"At dawn, it was {patient} that {agent} {verb}."
    if construction == "passive_cleft":
        return f"At dawn, it was {patient} who was {verb} by {agent}."
    raise ValueError(construction)


def question(verb, role, style):
    if style == "generic":
        return (
            "Who performed the action at dawn?" if role == "agent"
            else "Who received the action at dawn?"
        )

    if style == "paraphrase":
        return (
            "At dawn, which person did something to the other person?"
            if role == "agent"
            else "At dawn, which person had something done to them?"
        )

    if style == "verb_explicit":
        return (
            f"At dawn, who {verb} the other person?"
            if role == "agent"
            else f"At dawn, who was {verb}?"
        )

    raise ValueError(style)


def generate(group, split):
    first, second, verb = group

    for direction in (0, 1):
        agent, patient = (
            (first, second) if direction == 0 else (second, first)
        )

        for construction, style, role in itertools.product(
            CONSTRUCTIONS, STYLES, ROLES
        ):
            yield {
                "id": (
                    f"{split}_{first}_{second}_{verb}_{direction}_"
                    f"{construction}_{style}_{role}"
                ),
                "split": split,
                "pair": [first, second],
                "verb": verb,
                "direction": direction,
                "construction": construction,
                "style": style,
                "role": role,
                "agent": agent,
                "patient": patient,
                "gold": agent if role == "agent" else patient,
                "passage": passage(agent, patient, verb, construction),
                "question": question(verb, role, style),
            }


def render(tokenizer, case):
    content = (
        "Answer with exactly one person's name.\n"
        f"Passage: {case['passage']}\n"
        f"Question: {case['question']}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def answer_tokens(tokenizer, prompt, answer):
    bos = tokenizer.bos_token
    add_special = bos is None or not prompt.startswith(bos)

    prefix = tokenizer.encode(
        prompt,
        add_special_tokens=add_special,
    )

    full = tokenizer.encode(
        prompt + answer,
        add_special_tokens=add_special,
    )

    if full[:len(prefix)] != prefix:
        return None

    return len(full) - len(prefix)


def static_audit():
    names = TRAIN_NAMES + TEST_NAMES
    verbs = TRAIN_VERBS + TEST_VERBS

    assert len(names) == len(set(names))
    assert len(verbs) == len(set(verbs))
    assert len(CONSTRUCTIONS) == 3 and len(STYLES) == 3
    assert set(ROLES) == {"agent", "patient"}

    for split, group in (
        ("calibration", (TRAIN_NAMES[0], TRAIN_NAMES[1], TRAIN_VERBS[0])),
        ("protected", (TEST_NAMES[0], TEST_NAMES[1], TEST_VERBS[0])),
    ):
        records = list(generate(group, split))

        assert len(records) == 36
        assert len(set(r["id"] for r in records)) == 36

        for r in records:
            assert r["passage"].count(r["agent"]) == 1
            assert r["passage"].count(r["patient"]) == 1
            assert r["gold"] in r["pair"]

    print("STATIC AUDIT PASS: fresh split, cases, roles and questions")
    print("Expected cases: 216 calibration, 288 protected per model")


def seal():
    static_audit()

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    from transformers import AutoTokenizer

    model_root = (
        Path.home() / "Documents/ChatGPT/project null"
        / "llm-relational-mechanisms/.local_models"
    )

    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            str(model_root / name),
            local_files_only=True,
        )
        for name in MODEL_NAMES
    }

    selection = {}
    excluded_counts = {}

    for split, name_bank, verbs in (
        ("calibration", TRAIN_NAMES, TRAIN_VERBS),
        ("protected", TEST_NAMES, TEST_VERBS),
    ):
        remaining = list(name_bank)
        chosen = []
        candidates_checked = 0

        for verb in verbs:
            match = None

            for a, b in itertools.combinations(remaining, 2):
                candidates_checked += 1
                group = (a, b, verb)
                eligible = True

                for row in generate(group, split):
                    for tokenizer in tokenizers.values():
                        p = render(tokenizer, row)
                        x = answer_tokens(tokenizer, p, a)
                        y = answer_tokens(tokenizer, p, b)

                        if x is None or x == 0 or x != y:
                            eligible = False
                            break

                    if not eligible:
                        break

                if eligible:
                    match = group
                    break

            if match is not None:
                chosen.append(match)
                remaining.remove(match[0])
                remaining.remove(match[1])

            if len(chosen) == N_GROUPS[split]:
                break

        if len(chosen) != N_GROUPS[split]:
            raise RuntimeError(
                f"Only {len(chosen)}/{N_GROUPS[split]} groups eligible "
                f"for {split}. Do not collect inference results."
            )

        selection[split] = chosen
        excluded_counts[split] = candidates_checked

        print(
            f"{split}: {len(chosen)} groups, "
            f"checked {candidates_checked} pairs"
        )

    cases = {
        split: [
            case
            for group in groups
            for case in generate(group, split)
        ]
        for split, groups in selection.items()
    }

    assert len(cases["calibration"]) == 216
    assert len(cases["protected"]) == 288

    manifest = {
        "spec": SPEC,
        "selection": selection,
        "pair_checks": excluded_counts,
        "cases": cases,
    }

    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("SEALED", OUTPUT)
    print("MANIFEST SHA256", manifest["manifest_sha256"])
    print("No model inference performed. Commit this manifest before scoring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("audit", "seal"),
        required=True,
    )

    args = parser.parse_args()

    if args.phase == "audit":
        static_audit()
    else:
        seal()
