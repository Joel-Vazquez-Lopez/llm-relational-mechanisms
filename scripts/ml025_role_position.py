
"""ML-025: Grammatical Role x Surface Position.

Behavioral 2x2 factorial with two name assignments,
two questions, and four counterfactual dependency states.

Causal intervention: frozen layer 7.

Run locally. No inference occurs in design or audit phases.
"""

import argparse
import hashlib
import json

from datetime import datetime
from itertools import product
from pathlib import Path

from ml024_role_reversal import (
    ASSIGNMENTS,
    EDIT,
    QUESTIONS,
    audit_prompts,
    capture_states,
    diff,
    gold,
    make_prompt,
    patched_score,
    score,
)


PATCH_LAYER = 7

ORDERS = ("letter_first", "map_first")
VOICES = ("active", "passive")


def letter_role(order, voice):
    if (order == "letter_first") == (voice == "active"):
        return "agent"
    return "patient"


def mapping_for(order, voice):
    return (
        "letter_agent"
        if letter_role(order, voice) == "agent"
        else "map_agent"
    )


def make_passage(
    assignment, order, voice,
    letter_word, map_word
):
    letter = (
        f"the person who {letter_word} the letter"
    )
    map_desc = (
        f"the person who {map_word} the map"
    )

    if order == "letter_first":
        first, second = letter, map_desc
    else:
        first, second = map_desc, letter

    if voice == "active":
        event = (
            f"At dusk, {first} helped {second}."
        )
    else:
        event = (
            f"At dusk, {first} was helped by {second}."
        )

    facts = " ".join((
        f"{assignment['writer']} wrote the letter.",
        f"{assignment['carrier']} carried the letter.",
        f"{assignment['folder']} folded the map.",
        f"{assignment['painter']} painted the map.",
    ))

    return f"{event} {facts}"


def build_design():
    trials = []

    for aid, order, voice, question, edit_name in product(
        range(2),
        ORDERS,
        VOICES,
        QUESTIONS,
        EDIT,
    ):
        assignment = ASSIGNMENTS[aid]
        words = EDIT[edit_name]
        mapping = mapping_for(order, voice)

        trials.append({
            "assignment": aid,
            "order": order,
            "voice": voice,
            "question": question,
            "edit": edit_name,
            "letter_role": letter_role(order, voice),
            "gold": gold(
                assignment,
                mapping,
                question,
                *words,
            ),
            "passage": make_passage(
                assignment, order, voice, *words
            ),
        })

    assert len(trials) == 64

    assert len({
        (
            t["assignment"],
            t["order"],
            t["voice"],
            t["question"],
            t["edit"],
        )
        for t in trials
    }) == 64

    # Role and position must form a balanced 2x2.
    cells = {
        (letter_role(order, voice), order)
        for order, voice in product(
            ORDERS, VOICES
        )
    }

    assert len(cells) == 4

    for t in trials:
        assert t["gold"] in (
            "Leo", "Felix", "Nora", "Clara"
        )

    return trials


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design", "audit",
            "behavior", "patch"
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
        type=int,
        choices=(0, 1),
        default=0,
    )

    args = parser.parse_args()

    trials = build_design()

    frozen_design = {
        "trials": trials,
        "patch_layer": PATCH_LAYER,
        "patch_conditions": (
            "self", "prefix",
            "letter", "map",
            "both_independent", "both_joint"
        ),
    }

    digest = hashlib.sha256(
        json.dumps(
            frozen_design, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    print("ML-025 | Role x Position")
    print("Phase:", args.phase)
    print("Model:", args.model)
    print("Design SHA-256:", digest)
    print("Behavioral trials:", len(trials))
    print("Frozen causal layer:", PATCH_LAYER)

    print("\nRole-position cells:")

    for order, voice in product(
        ORDERS, VOICES
    ):
        print(
            f"{order:13s} {voice:7s} "
            f"letter_role={letter_role(order, voice)}"
        )

    if args.phase == "design":
        print("\n=== SAMPLE PASSAGES ===")

        for order, voice in product(
            ORDERS, VOICES
        ):
            print(
                f"\n{order} / {voice}\n"
                + make_passage(
                    ASSIGNMENTS[0],
                    order,
                    voice,
                    *EDIT["base"],
                )
            )

        print("\nDesign checks passed.")
        print("No model loaded.")
        return

    from mlx_lm import load
    from ml003_role_margins import (
        MODELS, log_likelihood
    )
    from ml012_cue_patch import encode_prompt

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    cases = {}

    for aid, order, voice, question in product(
        range(2),
        ORDERS,
        VOICES,
        QUESTIONS,
    ):
        assignment = ASSIGNMENTS[aid]

        prompts = {
            label: make_prompt(
                tokenizer,
                make_passage(
                    assignment,
                    order,
                    voice,
                    *words,
                ),
                question,
            )
            for label, words in EDIT.items()
        }

        # Reuses ML-024's proven single-token
        # alignment and combined-edit checks.
        positions = audit_prompts(
            tokenizer, prompts, encode_prompt
        )

        cases[(aid, order, voice, question)] = (
            prompts, positions
        )

        print(
            f"AUDIT a={aid} {order:13s} "
            f"{voice:7s} q={question:7s} "
            f"letter={positions['letter']} "
            f"map={positions['map']}"
        )

    if args.phase == "audit":
        print("\nAll 16 token audits passed.")
        print("No forward passes performed.")
        return

    if (
        args.phase == "patch"
        and PATCH_LAYER >= len(model.model.layers)
    ):
        raise RuntimeError(
            "Prespecified layer unavailable"
        )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml025_{args.model}_{args.phase}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "experiment": "ML-025",
        "model": args.model,
        "phase": args.phase,
        "assignment": (
            args.assignment
            if args.phase == "patch"
            else "both"
        ),
        "design_hash": digest,
        "patch_layer": PATCH_LAYER,
        "complete": False,
        "records": [],
    }

    def save():
        output.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    for key, (prompts, positions) in cases.items():
        aid, order, voice, question = key

        if (
            args.phase == "patch"
            and aid != args.assignment
        ):
            continue

        assignment = ASSIGNMENTS[aid]
        mapping = mapping_for(order, voice)

        full = {}
        forced = {}
        expected = {}

        for label, prompt in prompts.items():
            values, raw_scores, _ = score(
                model, tokenizer, prompt,
                assignment, mapping, question,
                log_likelihood,
            )

            full[label] = values

            forced[label] = max(
                raw_scores, key=raw_scores.get
            )

            expected[label] = gold(
                assignment, mapping,
                question, *EDIT[label]
            )

        baseline = full["base"]

        full_effects = {
            label: diff(values, baseline)
            for label, values in full.items()
            if label != "base"
        }

        current_role = letter_role(order, voice)

        print(
            f"\nA={aid} {order}/{voice} "
            f"q={question} letter_role={current_role}"
        )

        print(
            "BASE predicted:",
            forced["base"],
            "gold:",
            expected["base"],
        )

        for label in ("letter", "map", "both"):
            e = full_effects[label]

            print(
                f"FULL {label:6s} "
                f"dLetter={e['letter']:+.3f} "
                f"dMap={e['map']:+.3f} "
                f"dCompetition="
                f"{e['requested_vs_other']:+.3f} "
                f"pred={forced[label]} "
                f"gold={expected[label]}"
            )

        record = {
            "assignment": aid,
            "order": order,
            "voice": voice,
            "question": question,
            "letter_role": current_role,
            "map_role": (
                "patient"
                if current_role == "agent"
                else "agent"
            ),
            "letter_relevant": (
                current_role == question
            ),
            "positions": positions,
            "forced": forced,
            "expected": expected,
            "full": full,
            "full_effects": full_effects,
            "patches": {},
        }

        data["records"].append(record)
        save()

        if args.phase != "patch":
            continue

        states = {
            label: capture_states(
                model,
                tokenizer,
                prompt,
                PATCH_LAYER,
                positions,
                encode_prompt,
            )
            for label, prompt in prompts.items()
        }

        p = positions

        configs = {
            "self": {
                p["letter"]: states["base"]["letter"],
                p["map"]: states["base"]["map"],
            },
            "prefix": {
                p["prefix"]: states["letter"]["prefix"],
            },
            "letter": {
                p["letter"]:
                    states["letter"]["letter"],
            },
            "map": {
                p["map"]:
                    states["map"]["map"],
            },
            "both_independent": {
                p["letter"]:
                    states["letter"]["letter"],
                p["map"]:
                    states["map"]["map"],
            },
            "both_joint": {
                p["letter"]:
                    states["both"]["letter"],
                p["map"]:
                    states["both"]["map"],
            },
        }

        for label, patch_map in configs.items():
            patched = patched_score(
                model, tokenizer,
                prompts["base"],
                assignment, mapping, question,
                PATCH_LAYER,
                patch_map,
                log_likelihood,
            )

            effect = diff(patched, baseline)

            record["patches"][label] = {
                "margins": patched,
                "effects": effect,
            }

            print(
                f"L07 {label:16s} "
                f"dLetter={effect['letter']:+.3f} "
                f"dMap={effect['map']:+.3f} "
                f"dCompetition="
                f"{effect['requested_vs_other']:+.3f}"
            )

            save()

            if label in ("self", "prefix"):
                error = max(
                    abs(v) for v in effect.values()
                )

                if error > 0.05:
                    raise RuntimeError(
                        f"Integrity check failed: "
                        f"{label}, A={aid}, "
                        f"{order}/{voice}/{question}"
                    )

    data["complete"] = True
    save()

    correct = sum(
        predicted == expected
        for r in data["records"]
        for label, predicted in r["forced"].items()
        for expected in [r["expected"][label]]
    )

    total = sum(
        len(r["forced"])
        for r in data["records"]
    )

    print(
        f"\nForced-choice accuracy: {correct}/{total}"
    )
    print("Saved:", output)


if __name__ == "__main__":
    run()
