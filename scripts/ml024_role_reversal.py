
"""ML-024: Role Reversal and Causal Binding."""

import argparse
import hashlib
import json

from datetime import datetime
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path


NAMES = ("Leo", "Felix", "Nora", "Clara")

# Selected before observing ML-024 outcomes.
LAYERS = (0, 7, 13)

QUESTIONS = {
    "agent": "Who helped the other person?",
    "patient": "Who was helped?",
}

EDIT = {
    "base": ("wrote", "folded"),
    "letter": ("carried", "folded"),
    "map": ("wrote", "painted"),
    "both": ("carried", "painted"),
}

ASSIGNMENTS = (
    {
        "writer": "Leo",
        "carrier": "Felix",
        "folder": "Nora",
        "painter": "Clara",
    },
    {
        "writer": "Felix",
        "carrier": "Leo",
        "folder": "Clara",
        "painter": "Nora",
    },
)


def role_family(mapping, role):
    if mapping == "letter_agent":
        return "letter" if role == "agent" else "map"

    return "map" if role == "agent" else "letter"


def passage(mapping, assignment, letter_word, map_word):
    letter = f"the person who {letter_word} the letter"
    map_desc = f"the person who {map_word} the map"

    if mapping == "letter_agent":
        agent, patient = letter, map_desc
    else:
        agent, patient = map_desc, letter

    facts = " ".join((
        f"{assignment['writer']} wrote the letter.",
        f"{assignment['carrier']} carried the letter.",
        f"{assignment['folder']} folded the map.",
        f"{assignment['painter']} painted the map.",
    ))

    return f"At dusk, {agent} helped {patient}. {facts}"


def gold(assignment, mapping, question, letter_word, map_word):
    family = role_family(mapping, question)

    if family == "letter":
        return (
            assignment["writer"]
            if letter_word == "wrote"
            else assignment["carrier"]
        )

    return (
        assignment["folder"]
        if map_word == "folded"
        else assignment["painter"]
    )


def make_design():
    trials = []

    for assignment_id, mapping, question, edit_name in product(
        range(len(ASSIGNMENTS)),
        ("letter_agent", "map_agent"),
        QUESTIONS,
        EDIT,
    ):
        assignment = ASSIGNMENTS[assignment_id]
        letter_word, map_word = EDIT[edit_name]

        trials.append({
            "assignment_id": assignment_id,
            "mapping": mapping,
            "question": question,
            "edit": edit_name,
            "passage": passage(
                mapping, assignment, letter_word, map_word
            ),
            "gold": gold(
                assignment, mapping, question,
                letter_word, map_word
            ),
        })

    assert len(trials) == 32

    assert len({
        (
            t["assignment_id"],
            t["mapping"],
            t["question"],
            t["edit"],
        )
        for t in trials
    }) == 32

    for t in trials:
        assignment = ASSIGNMENTS[t["assignment_id"]]

        assert all(
            t["passage"].count(name) == 1
            for name in NAMES
        )

        assert t["gold"] in NAMES

        family = role_family(
            t["mapping"], t["question"]
        )

        letter_word, map_word = EDIT[t["edit"]]

        if family == "letter":
            expected = (
                assignment["writer"]
                if letter_word == "wrote"
                else assignment["carrier"]
            )
        else:
            expected = (
                assignment["folder"]
                if map_word == "folded"
                else assignment["painter"]
            )

        assert expected == t["gold"]

    return trials


def make_prompt(tokenizer, statement, question):
    content = (
        "Answer with exactly one person's name.\n\n"
        f"Passage: {statement}\n"
        f"Question: {QUESTIONS[question]}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def one_token_position(a, b, label):
    edits = [
        e for e in SequenceMatcher(
            None, a, b, autojunk=False
        ).get_opcodes()
        if e[0] != "equal"
    ]

    if len(a) != len(b) or len(edits) != 1:
        raise ValueError(
            f"{label}: invalid alignment: {edits}"
        )

    tag, i, j, k, l = edits[0]

    if (
        tag != "replace"
        or j - i != 1
        or l - k != 1
        or i != k
    ):
        raise ValueError(
            f"{label}: expected aligned 1:1 edit: {edits}"
        )

    return i


def audit_prompts(tokenizer, prompts, encode_prompt):
    ids = {
        key: encode_prompt(tokenizer, value)
        for key, value in prompts.items()
    }

    base = ids["base"]

    letter_pos = one_token_position(
        base, ids["letter"], "letter"
    )

    map_pos = one_token_position(
        base, ids["map"], "map"
    )

    if letter_pos == map_pos:
        raise ValueError("Cue positions collide")

    if len(ids["both"]) != len(base):
        raise ValueError(
            "Combined edit changed token count"
        )

    differences = [
        i
        for i, (a, b) in enumerate(
            zip(base, ids["both"])
        )
        if a != b
    ]

    if differences != sorted((letter_pos, map_pos)):
        raise ValueError(
            f"Combined edit positions: {differences}"
        )

    if (
        ids["both"][letter_pos]
        != ids["letter"][letter_pos]
        or ids["both"][map_pos]
        != ids["map"][map_pos]
    ):
        raise ValueError("Combined edit token mismatch")

    decoded = (
        tokenizer.decode([base[letter_pos]]).strip(),
        tokenizer.decode(
            [ids["letter"][letter_pos]]
        ).strip(),
        tokenizer.decode([base[map_pos]]).strip(),
        tokenizer.decode(
            [ids["map"][map_pos]]
        ).strip(),
    )

    if decoded != (
        "wrote", "carried", "folded", "painted"
    ):
        raise ValueError(
            f"Unexpected cue tokens: {decoded}"
        )

    prefix = min(letter_pos, map_pos) - 2

    if prefix < 0:
        raise ValueError(
            "No causal-prefix control available"
        )

    return {
        "letter": letter_pos,
        "map": map_pos,
        "prefix": prefix,
        "token_count": len(base),
    }


def score(
    model, tokenizer, prompt,
    assignment, mapping, question,
    log_likelihood,
):
    scores = {}
    counts = {}

    for name in NAMES:
        lp, count = log_likelihood(
            model, tokenizer, prompt, name
        )

        scores[name] = float(lp)
        counts[name] = int(count)

    letter_margin = (
        scores[assignment["writer"]]
        - scores[assignment["carrier"]]
    )

    map_margin = (
        scores[assignment["folder"]]
        - scores[assignment["painter"]]
    )

    family = role_family(mapping, question)

    if family == "letter":
        selected_name = assignment["writer"]
        other_name = assignment["folder"]
    else:
        selected_name = assignment["folder"]
        other_name = assignment["writer"]

    # Fixed baseline referents, including after edits.
    result = {
        "letter": letter_margin,
        "map": map_margin,
        "requested_vs_other": (
            scores[selected_name] - scores[other_name]
        ),
    }

    return result, scores, counts


def diff(after, before):
    return {
        key: after[key] - before[key]
        for key in before
    }


def capture_states(
    model, tokenizer, prompt,
    layer_idx, positions, encode_prompt,
):
    import mlx.core as mx
    import mlx.nn as nn

    class Capture(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.use_sliding = getattr(
                layer, "use_sliding", False
            )
            self.value = None

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)

            self.value = {
                key: output[0, position, :]
                for key, position in positions.items()
                if key != "token_count"
            }

            return output

    layers = model.model.layers
    original = layers[layer_idx]
    wrapper = Capture(original)

    layers[layer_idx] = wrapper

    try:
        token_ids = encode_prompt(
            tokenizer, prompt
        )

        mx.eval(
            model(
                mx.array(
                    [token_ids],
                    dtype=mx.int32,
                )
            )
        )

        if wrapper.value is None:
            raise RuntimeError(
                "No activation captured"
            )

        mx.eval(*wrapper.value.values())

        return dict(wrapper.value)

    finally:
        layers[layer_idx] = original


def patched_score(
    model, tokenizer, prompt,
    assignment, mapping, question,
    layer_idx, replacement,
    log_likelihood,
):
    import mlx.core as mx
    import mlx.nn as nn

    class Patch(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.use_sliding = getattr(
                layer, "use_sliding", False
            )

        def __call__(self, *args, **kwargs):
            output = self.layer(*args, **kwargs)

            parts = []
            cursor = 0

            for position, state in sorted(
                replacement.items()
            ):
                parts.append(
                    output[:, cursor:position, :]
                )

                parts.append(
                    state[None, None, :].astype(
                        output.dtype
                    )
                )

                cursor = position + 1

            parts.append(
                output[:, cursor:, :]
            )

            return mx.concatenate(
                parts, axis=1
            )

    layers = model.model.layers
    original = layers[layer_idx]
    layers[layer_idx] = Patch(original)

    try:
        return score(
            model, tokenizer, prompt,
            assignment, mapping, question,
            log_likelihood,
        )[0]

    finally:
        layers[layer_idx] = original


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "behavior",
            "patch",
        ),
        default="design",
    )

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    parser.add_argument(
        "--assignment",
        choices=(0, 1),
        type=int,
        default=0,
        help=(
            "Assignment used for patching; "
            "behavior uses both assignments"
        ),
    )

    args = parser.parse_args()

    trials = make_design()

    design_hash = hashlib.sha256(
        json.dumps(
            trials, sort_keys=True
        ).encode()
    ).hexdigest()

    print(
        f"ML-024 | phase={args.phase} "
        f"| model={args.model}"
    )

    print("Design SHA-256:", design_hash)
    print("Behavioral conditions:", len(trials))
    print("Prespecified patch layers:", LAYERS)

    print("\nPredictions:")
    print(
        "1. Letter edits affect the letter candidate pair."
    )
    print(
        "2. Map edits affect the map candidate pair."
    )
    print(
        "3. Requested grammatical role changes "
        "under construction reversal."
    )
    print(
        "4. Activation effects may depend on "
        "role, question, and context."
    )
    print(
        "5. Lexical and positional explanations "
        "remain competing hypotheses."
    )

    if args.phase == "design":
        for mapping in (
            "letter_agent",
            "map_agent",
        ):
            print("\nMAPPING:", mapping)

            for edit_name, words in EDIT.items():
                print(
                    edit_name,
                    passage(
                        mapping,
                        ASSIGNMENTS[0],
                        *words,
                    ),
                )

        print("\nDesign checked.")
        print("No model loaded or inference run.")
        return

    from mlx_lm import load
    from ml003_role_margins import (
        MODELS, log_likelihood
    )
    from ml012_cue_patch import encode_prompt

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    print("\nLoaded model:", args.model)

    cases = {}

    for assignment_id, mapping, question in product(
        range(len(ASSIGNMENTS)),
        ("letter_agent", "map_agent"),
        QUESTIONS,
    ):
        prompts = {
            label: make_prompt(
                tokenizer,
                passage(
                    mapping,
                    ASSIGNMENTS[assignment_id],
                    *words,
                ),
                question,
            )
            for label, words in EDIT.items()
        }

        positions = audit_prompts(
            tokenizer, prompts, encode_prompt
        )

        cases[
            (assignment_id, mapping, question)
        ] = (prompts, positions)

        print(
            f"ALIGN a={assignment_id} "
            f"{mapping:12s} {question:7s}: "
            f"letter={positions['letter']} "
            f"map={positions['map']} "
            f"prefix={positions['prefix']}"
        )

    if args.phase == "audit":
        print("\nAll token audits passed.")
        print("No forward passes performed.")
        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = results_dir / (
        f"ml024_{args.model}_{args.phase}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "experiment": "ML-024",
        "model": args.model,
        "phase": args.phase,
        "assignment": (
            args.assignment
            if args.phase == "patch"
            else "both"
        ),
        "design_sha256": design_hash,
        "complete": False,
        "records": [],
    }

    def save():
        output.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    for key, (prompts, positions) in cases.items():
        assignment_id, mapping, question = key

        if (
            args.phase == "patch"
            and assignment_id != args.assignment
        ):
            continue

        assignment = ASSIGNMENTS[assignment_id]

        full = {}
        counts = None

        for label, prompt in prompts.items():
            values, raw_scores, token_counts = score(
                model, tokenizer, prompt,
                assignment, mapping, question,
                log_likelihood,
            )

            full[label] = values

            if counts is None:
                counts = token_counts

        baseline = full["base"]

        effects = {
            label: diff(values, baseline)
            for label, values in full.items()
            if label != "base"
        }

        requested_family = role_family(
            mapping, question
        )

        print(
            f"\nassignment={assignment_id} "
            f"mapping={mapping} "
            f"question={question} "
            f"requested={requested_family}"
        )

        for label in ("letter", "map", "both"):
            effect = effects[label]

            print(
                f"FULL {label:6s}: "
                f"letter={effect['letter']:+.3f} "
                f"map={effect['map']:+.3f} "
                f"competition="
                f"{effect['requested_vs_other']:+.3f}"
            )

        record = {
            "assignment_id": assignment_id,
            "mapping": mapping,
            "question": question,
            "requested_family": requested_family,
            "gold_base": gold(
                assignment, mapping,
                question, *EDIT["base"]
            ),
            "token_counts": counts,
            "positions": positions,
            "full_margins": full,
            "full_effects": effects,
            "patches": {},
        }

        data["records"].append(record)
        save()

        if args.phase != "patch":
            continue

        for layer_idx in LAYERS:
            if layer_idx >= len(model.model.layers):
                raise RuntimeError(
                    f"Layer {layer_idx} unavailable"
                )

            captured = {
                label: capture_states(
                    model,
                    tokenizer,
                    prompt,
                    layer_idx,
                    positions,
                    encode_prompt,
                )
                for label, prompt in prompts.items()
            }

            p = positions

            patch_configs = {
                "self": {
                    p["letter"]:
                        captured["base"]["letter"],
                    p["map"]:
                        captured["base"]["map"],
                },

                "prefix": {
                    p["prefix"]:
                        captured["letter"]["prefix"],
                },

                "letter": {
                    p["letter"]:
                        captured["letter"]["letter"],
                },

                "map": {
                    p["map"]:
                        captured["map"]["map"],
                },

                "both_independent": {
                    p["letter"]:
                        captured["letter"]["letter"],
                    p["map"]:
                        captured["map"]["map"],
                },

                "both_joint": {
                    p["letter"]:
                        captured["both"]["letter"],
                    p["map"]:
                        captured["both"]["map"],
                },
            }

            layer_results = {}

            for label, patch_map in patch_configs.items():
                measured = patched_score(
                    model,
                    tokenizer,
                    prompts["base"],
                    assignment,
                    mapping,
                    question,
                    layer_idx,
                    patch_map,
                    log_likelihood,
                )

                effect = diff(
                    measured, baseline
                )

                layer_results[label] = {
                    "margins": measured,
                    "effects": effect,
                }

                print(
                    f"L{layer_idx:02d} "
                    f"{label:16s} "
                    f"letter={effect['letter']:+.3f} "
                    f"map={effect['map']:+.3f} "
                    f"competition="
                    f"{effect['requested_vs_other']:+.3f}"
                )

                if label in ("self", "prefix"):
                    error = max(
                        abs(v)
                        for v in effect.values()
                    )

                    if error > 0.05:
                        record["patches"][
                            str(layer_idx)
                        ] = layer_results

                        save()

                        raise RuntimeError(
                            f"Control {label} failed "
                            f"at layer {layer_idx}"
                        )

            record["patches"][
                str(layer_idx)
            ] = layer_results

            save()

    data["complete"] = True
    save()

    print("\nSaved:", output)


if __name__ == "__main__":
    run()
