
"""ML-052: preregistered design and tokenizer eligibility.

No model inference. Select lexical groups using tokenizers only.
Commit the manifest before collecting any model scores.
"""

import argparse
import hashlib
import json
from pathlib import Path

TRAIN_BANK = (
    ("Harper", "Quinn", "praised"),
    ("Elena", "Victor", "helped"),
    ("Hazel", "Caleb", "admired"),
    ("Freya", "Isaac", "thanked"),
    ("Martha", "Jonah", "followed"),
    ("Piper", "Louis", "greeted"),
    ("Evelyn", "Adrian", "supported"),
    ("Phoebe", "Arthur", "rescued"),
    ("Amelia", "Samuel", "encouraged"),
    ("Chloe", "Jacob", "comforted"),
    ("Miriam", "George", "welcomed"),
    ("Sylvia", "Martin", "guided"),
    ("Lydia", "Marcus", "advised"),
    ("Ruth", "Thomas", "visited"),
    ("Adele", "Vincent", "assisted"),
    ("Rose", "Elliot", "alerted"),
)

PROTECTED_BANK = (
    ("Cora", "Elias", "congratulated"),
    ("Fiona", "Miles", "applauded"),
    ("Beatrice", "Rowan", "recognized"),
    ("Nadia", "Hugo", "reassured"),
    ("Daphne", "Colin", "hugged"),
    ("Naomi", "Ezra", "invited"),
    ("Willa", "Gavin", "chased"),
    ("Vera", "Leon", "respected"),
    ("Ivy", "Nathan", "photographed"),
    ("Selena", "Tobias", "surprised"),
    ("Bianca", "Rafael", "consoled"),
    ("Mabel", "Owen", "trusted"),
    ("Leila", "Dominic", "consulted"),
    ("Olive", "Calvin", "observed"),
    ("Jade", "Stanley", "criticized"),
    ("Esther", "Lawrence", "questioned"),
)

TRAIN_CONSTRUCTIONS = (
    "active",
    "passive",
    "object_cleft",
    "subject_cleft",
)

HOLDOUT = "passive_cleft"
REQUIRED_GROUPS = 6
MANIFEST = Path("results/ml052_design_manifest.json")

SPEC = {
    "experiment": "ML-052",
    "train_bank": TRAIN_BANK,
    "protected_bank": PROTECTED_BANK,
    "required_groups_per_split": REQUIRED_GROUPS,
    "train_constructions": TRAIN_CONSTRUCTIONS,
    "compositional_holdout": HOLDOUT,
    "directions": (0, 1),
    "roles": ("agent", "patient"),
    "selection": (
        "First six tokenizer-eligible groups "
        "from each fixed bank; no output scores"
    ),
    "eligible": (
        "Both candidate continuations have equal "
        "token counts in both tokenizers for "
        "every evaluated prompt"
    ),
    "primary": (
        "Correct-minus-other conditional name "
        "log-likelihood"
    ),
    "primary_test": (
        "Unseen passive-cleft composition "
        "with new names and verbs"
    ),
    "baselines": (
        "Role-only, position, subjecthood, "
        "construction-conditioned, additive cues"
    ),
    "independent_unit": (
        "Name-pair/verb group; grammatical "
        "variants within a group are dependent"
    ),
    "interpretation": (
        "Behavioral prediction; causal evidence "
        "requires a separate intervention"
    ),
}


def sentence(agent, patient, verb, construction):
    return {
        "active":
            f"At dawn, {agent} {verb} {patient}.",
        "passive":
            f"At dawn, {patient} was {verb} by {agent}.",
        "object_cleft":
            f"At dawn, it was {patient} that "
            f"{agent} {verb}.",
        "subject_cleft":
            f"At dawn, it was {agent} who "
            f"{verb} {patient}.",
        "passive_cleft":
            f"At dawn, it was {patient} who "
            f"was {verb} by {agent}.",
    }[construction]


def cases(group, split):
    left, right, verb = group

    constructions = TRAIN_CONSTRUCTIONS + (
        (HOLDOUT,)
        if split == "protected" else ()
    )

    for direction in (0, 1):
        agent, patient = (
            (left, right)
            if direction == 0
            else (right, left)
        )

        for construction in constructions:
            passage = sentence(
                agent, patient, verb, construction
            )

            for role in ("agent", "patient"):
                question = (
                    "Who performed the action at dawn?"
                    if role == "agent"
                    else
                    "Who received the action at dawn?"
                )

                yield {
                    "id": (
                        f"{split}_{left}_{right}_"
                        f"{verb}_{direction}_"
                        f"{construction}_{role}"
                    ),
                    "split": split,
                    "pair": (left, right),
                    "verb": verb,
                    "direction": direction,
                    "construction": construction,
                    "role": role,
                    "agent": agent,
                    "patient": patient,
                    "gold": (
                        agent
                        if role == "agent"
                        else patient
                    ),
                    "passage": passage,
                    "question": question,
                }


def prompt(tok, case):
    content = (
        "Answer with exactly one person's name.\n"
        f"Passage: {case['passage']}\n"
        f"Question: {case['question']}\n"
        "Answer:"
    )

    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def suffix_count(tok, rendered, name):
    bos = tok.bos_token
    special = (
        bos is None
        or not rendered.startswith(bos)
    )

    prefix = tok.encode(
        rendered,
        add_special_tokens=special,
    )

    full = tok.encode(
        rendered + name,
        add_special_tokens=special,
    )

    if full[:len(prefix)] != prefix:
        return None

    return len(full) - len(prefix)


def audit_static():
    assert len(TRAIN_BANK) == 16
    assert len(PROTECTED_BANK) == 16

    names = [
        name
        for bank in (TRAIN_BANK, PROTECTED_BANK)
        for group in bank
        for name in group[:2]
    ]

    verbs = [
        group[2]
        for bank in (TRAIN_BANK, PROTECTED_BANK)
        for group in bank
    ]

    assert len(names) == len(set(names))
    assert len(verbs) == len(set(verbs))

    for bank, split in (
        (TRAIN_BANK, "calibration"),
        (PROTECTED_BANK, "protected"),
    ):
        for group in bank:
            rows = list(cases(group, split))

            assert len(rows) == (
                20 if split == "protected"
                else 16
            )

            for row in rows:
                assert row["gold"] in row["pair"]

                assert all(
                    row["passage"].count(name) == 1
                    for name in row["pair"]
                )

                if split == "calibration":
                    assert (
                        row["construction"] != HOLDOUT
                    )

    print("STATIC AUDIT: PASS")
    print("Distinct names and verbs: PASS")
    print("Passive cleft excluded from calibration: PASS")
    print("Factorial directions and roles: PASS")
    print(
        "Planned selected contexts per model:",
        "96 calibration + 120 protected"
    )


def seal():
    audit_static()

    if MANIFEST.exists():
        raise FileExistsError(
            f"Refusing to overwrite {MANIFEST}"
        )

    from transformers import AutoTokenizer

    model_dir = (
        Path.home()
        / "Documents/ChatGPT/project null"
        / "llm-relational-mechanisms/.local_models"
    )

    tokenizers = {}

    for model in ("qwen7b", "llama3b"):
        tokenizers[model] = (
            AutoTokenizer.from_pretrained(
                str(model_dir / model),
                local_files_only=True,
            )
        )

    selected = {}
    excluded = {}

    for bank, split in (
        (TRAIN_BANK, "calibration"),
        (PROTECTED_BANK, "protected"),
    ):
        eligible = []
        ineligible = []

        for group in bank:
            failure = None

            for model, tok in tokenizers.items():
                for case in cases(group, split):
                    rendered = prompt(tok, case)

                    n1, n2 = [
                        suffix_count(tok, rendered, name)
                        for name in group[:2]
                    ]

                    if not n1 or n1 != n2:
                        failure = (
                            f"{model}:"
                            f"{case['construction']}:"
                            f"{case['role']}:"
                            f"{n1}/{n2}"
                        )
                        break

                if failure:
                    break

            if failure:
                ineligible.append({
                    "group": group,
                    "reason": failure,
                })
            else:
                eligible.append(group)

        print(
            f"{split}: "
            f"{len(eligible)}/{len(bank)} "
            "tokenizer-eligible groups"
        )

        if len(eligible) < REQUIRED_GROUPS:
            for item in ineligible:
                print("Excluded:", item)

            raise RuntimeError(
                "Eligibility gate failed. "
                "Do not collect model scores."
            )

        selected[split] = (
            eligible[:REQUIRED_GROUPS]
        )

        excluded[split] = ineligible

    content = {
        "spec": SPEC,
        "selection": selected,
        "excluded_for_tokenization": excluded,
        "cases": {
            split: [
                row
                for group in selected[split]
                for row in cases(group, split)
            ]
            for split in selected
        },
    }

    content["manifest_sha256"] = (
        hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )

    MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST.write_text(
        json.dumps(content, indent=2)
    )

    print(
        "SEALED:",
        MANIFEST,
        content["manifest_sha256"],
    )

    print(
        "Commit the design manifest "
        "before model inference."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("audit", "seal"),
        required=True,
    )

    args = parser.parse_args()

    if args.phase == "audit":
        audit_static()
    else:
        seal()
