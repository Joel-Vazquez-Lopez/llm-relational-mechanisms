
"""ML-029: Event meaning versus prepared-answer transfer.

Exploratory factorial intervention at L20 readout.
Uses ML-028 stimuli. Fresh replication is required.
"""

import argparse
import hashlib
import json

from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from ml028_event_interchange import (
    PAIRS,
    VERBS,
    VOICES,
    QUESTIONS,
    gold,
    make_prompt,
    positions,
    answer_scores,
    gold_margin,
    capture,
    patched_scores,
)

LAYER = 20

DESIGN = {
    "experiment": "ML-029",
    "status": "exploratory follow-up",
    "pairs": PAIRS,
    "verbs": VERBS,
    "voices": VOICES,
    "questions": QUESTIONS,
    "layer": LAYER,
    "site": "readout",
    "donor_voice": "opposite recipient voice",
    "factors": (
        "event reversal",
        "question reversal",
    ),
    "primary": (
        "event contrast",
        "answer-mismatch contrast",
        "question contrast",
    ),
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(
        DESIGN,
        sort_keys=True,
    ).encode()
).hexdigest()


def opposite(value, values):
    return next(
        x for x in values
        if x != value
    )


def summarize(records):
    grouped = {}

    for row in records:
        key = (
            row["pair_id"],
            row["verb"],
            row["recipient_direction"],
            row["recipient_voice"],
            row["recipient_question"],
        )

        donor_condition = (
            row["event_reversed"],
            row["question_reversed"],
        )

        group = grouped.setdefault(
            key, {}
        )

        if donor_condition in group:
            raise ValueError(
                f"Duplicate condition: {key}"
            )

        group[donor_condition] = row["effect"]

    metrics = {
        name: []
        for name in (
            "event",
            "answer",
            "question",
        )
    }

    by_role = {
        role: {
            name: []
            for name in metrics
        }
        for role in QUESTIONS
    }

    for key, effects in grouped.items():
        required = set(
            product(
                (False, True),
                repeat=2,
            )
        )

        if set(effects) != required:
            raise ValueError(
                f"Incomplete comparison: {key}"
            )

        E00 = effects[False, False]
        E10 = effects[True, False]
        E01 = effects[False, True]
        E11 = effects[True, True]

        contrasts = {
            "event": (
                E10 + E11 - E00 - E01
            ) / 2,

            "answer": (
                E10 + E01 - E00 - E11
            ) / 2,

            "question": (
                E01 + E11 - E00 - E10
            ) / 2,
        }

        for name, value in contrasts.items():
            metrics[name].append(value)
            by_role[key[-1]][name].append(value)

    print(
        "\n=== ML-029 FACTORIAL "
        "INTERCHANGE ==="
    )

    print(
        "Recipient conditions:",
        len(grouped),
    )

    print(
        "Negative effects move away "
        "from recipient's original answer."
    )

    for name, values in metrics.items():
        print(
            f"\n{name.upper()} CONTRAST "
            f"mean={mean(values):+.4f} "
            f"negative={sum(x < 0 for x in values)}"
            f"/{len(values)}"
        )

        for role in QUESTIONS:
            subset = by_role[role][name]

            print(
                f"  {role}: "
                f"mean={mean(subset):+.4f} "
                f"negative="
                f"{sum(x < 0 for x in subset)}"
                f"/{len(subset)}"
            )

    print(
        "\nPure answer-copy account: "
        "large negative answer contrast, "
        "small event contrast."
    )

    print(
        "Event-sensitive account: "
        "negative event contrast even "
        "when answer identity is balanced."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design",
            "audit",
            "collect",
            "summarize",
        ),
        default="design",
    )

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    args = parser.parse_args()

    print(
        "ML-029",
        args.phase,
        args.model,
        DESIGN_HASH,
    )

    if args.phase == "design":
        print(
            json.dumps(
                DESIGN,
                indent=2,
            )
        )

        print(
            "32 recipients x 4 crossed donors "
            "= 128 interchange conditions."
        )

        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.phase == "summarize":
        paths = sorted(
            results_dir.glob(
                f"ml029_{args.model}_*.json"
            )
        )

        if not paths:
            raise FileNotFoundError(
                "No ML-029 results"
            )

        source = paths[-1]

        data = json.loads(
            source.read_text()
        )

        if (
            not data.get("complete")
            or data.get("design_hash")
            != DESIGN_HASH
        ):
            raise ValueError(
                f"Incomplete or mismatched: {source}"
            )

        summarize(data["records"])
        return

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    from ml012_cue_patch import (
        encode_prompt,
    )

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    for pair in PAIRS:
        lengths = [
            len(
                tokenizer.encode(
                    name,
                    add_special_tokens=False,
                )
            )
            for name in pair
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"Unequal name lengths: {pair}"
            )

    prepared = {}

    for key in product(
        range(len(PAIRS)),
        VERBS,
        (0, 1),
        VOICES,
        QUESTIONS,
    ):
        (
            pair_id,
            verb,
            direction,
            voice,
            question,
        ) = key

        prompt = make_prompt(
            tokenizer,
            PAIRS[pair_id],
            verb,
            direction,
            voice,
            question,
        )

        pos = positions(
            tokenizer,
            prompt,
            encode_prompt,
        )

        prepared[key] = (
            prompt,
            pos["readout"],
        )

    print(
        "Audited prompts:",
        len(prepared),
    )

    if args.phase == "audit":
        return

    output = results_dir / (
        f"ml029_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}"
        ".json"
    )

    data = {
        "experiment": "ML-029",
        "design_hash": DESIGN_HASH,
        "complete": False,
        "records": [],
    }

    def save():
        output.write_text(
            json.dumps(
                data,
                indent=2,
            )
        )

    for pair_id, verb in product(
        range(len(PAIRS)),
        VERBS,
    ):
        pair = PAIRS[pair_id]

        states = {}

        for direction, voice, question in product(
            (0, 1),
            VOICES,
            QUESTIONS,
        ):
            key = (
                pair_id,
                verb,
                direction,
                voice,
                question,
            )

            prompt, position = prepared[key]

            states[
                direction,
                voice,
                question,
            ] = capture(
                model,
                tokenizer,
                prompt,
                LAYER,
                position,
                encode_prompt,
            )

        for direction, voice, question in product(
            (0, 1),
            VOICES,
            QUESTIONS,
        ):
            key = (
                pair_id,
                verb,
                direction,
                voice,
                question,
            )

            prompt, position = prepared[key]

            recipient_gold = gold(
                pair,
                direction,
                question,
            )

            baseline_scores = answer_scores(
                model,
                tokenizer,
                prompt,
                log_likelihood,
                pair,
            )

            baseline = gold_margin(
                baseline_scores,
                recipient_gold,
                pair,
            )

            self_scores = patched_scores(
                model,
                tokenizer,
                prompt,
                pair,
                LAYER,
                position,
                states[
                    direction,
                    voice,
                    question,
                ],
                log_likelihood,
            )

            self_effect = (
                gold_margin(
                    self_scores,
                    recipient_gold,
                    pair,
                )
                - baseline
            )

            if abs(self_effect) > 0.05:
                raise RuntimeError(
                    f"Self-control failed: "
                    f"{key}, {self_effect}"
                )

            donor_voice = opposite(
                voice,
                VOICES,
            )

            for event_rev, question_rev in product(
                (False, True),
                repeat=2,
            ):
                donor_direction = (
                    1 - direction
                    if event_rev
                    else direction
                )

                donor_question = (
                    opposite(question, QUESTIONS)
                    if question_rev
                    else question
                )

                donor_answer = gold(
                    pair,
                    donor_direction,
                    donor_question,
                )

                answer_mismatch = (
                    donor_answer != recipient_gold
                )

                assert answer_mismatch == (
                    event_rev != question_rev
                )

                donor_state = states[
                    donor_direction,
                    donor_voice,
                    donor_question,
                ]

                scores = patched_scores(
                    model,
                    tokenizer,
                    prompt,
                    pair,
                    LAYER,
                    position,
                    donor_state,
                    log_likelihood,
                )

                patched_margin = gold_margin(
                    scores,
                    recipient_gold,
                    pair,
                )

                data["records"].append({
                    "pair_id": pair_id,
                    "verb": verb,
                    "recipient_direction": direction,
                    "recipient_voice": voice,
                    "recipient_question": question,
                    "event_reversed": event_rev,
                    "question_reversed": question_rev,
                    "donor_voice": donor_voice,
                    "donor_answer": donor_answer,
                    "donor_answer_mismatch": answer_mismatch,
                    "baseline_margin": baseline,
                    "patched_margin": patched_margin,
                    "effect": patched_margin - baseline,
                    "self_control_effect": self_effect,
                })

            save()

            print(
                "DONE",
                key,
                "records:",
                len(data["records"]),
            )

    data["complete"] = True
    save()

    print("Saved:", output)

    summarize(data["records"])


if __name__ == "__main__":
    main()
