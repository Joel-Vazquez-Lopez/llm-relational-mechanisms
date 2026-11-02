
"""ML-049: Prospective Qwen behavioral test of semantic vs positional routing.

English active, passive, and object-cleft constructions with new names
and verbs. Compare neutral role questions against explicit relational
questions.

This is a behavioral diagnostic, not a causal intervention.
Commit this script BEFORE --phase run.
All outcomes, including failures, are retained.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXAMPLES = (
    ("Alice", "Bob", "admired", "admire"),
    ("Grace", "Henry", "rescued", "rescue"),
    ("Sophie", "Daniel", "encouraged", "encourage"),
)

CONSTRUCTIONS = (
    "active",
    "passive",
    "object_cleft",
)

ROLES = ("agent", "patient")
QUESTION_TYPES = ("neutral", "explicit")

OUT = Path("results/ml049_qwen7b_behavior.json")


SPEC = {
    "id": "ML-049",
    "model": "qwen7b",
    "examples": EXAMPLES,
    "directions": (0, 1),
    "constructions": CONSTRUCTIONS,
    "roles": ROLES,
    "question_types": QUESTION_TYPES,
    "primary": (
        "Neutral questions: compare semantic and "
        "first/second-mention predictions"
    ),
    "prospective_prediction": (
        "Neutral object-cleft position matches >=9/12"
    ),
    "interpretability_gate": (
        "Neutral active semantic accuracy >=10/12"
    ),
    "predictions": {
        "semantic": (
            "Agent question selects actual agent; "
            "patient question selects actual patient"
        ),
        "positional": (
            "Agent question selects first named person; "
            "patient question selects second"
        ),
    },
    "secondary": (
        "Explicit relational questions may rescue "
        "grammatical errors"
    ),
    "status": (
        "Prospective new names and verbs; "
        "no training and no causal claim"
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def make_cases():
    rows = []

    for left, right, past, base in EXAMPLES:
        for direction in (0, 1):
            agent, patient = (
                (left, right)
                if direction == 0
                else (right, left)
            )

            for construction in CONSTRUCTIONS:
                if construction == "active":
                    sentence = (
                        f"At dusk, {agent} {past} {patient}."
                    )
                    order = (agent, patient)

                elif construction == "passive":
                    sentence = (
                        f"At dusk, {patient} was "
                        f"{past} by {agent}."
                    )
                    order = (patient, agent)

                else:
                    sentence = (
                        f"At dusk, it was {patient} "
                        f"that {agent} {past}."
                    )
                    order = (patient, agent)

                for question_type in QUESTION_TYPES:
                    for role in ROLES:
                        if question_type == "neutral":
                            question = (
                                "Who performed the action?"
                                if role == "agent"
                                else "Who received the action?"
                            )

                        elif role == "agent":
                            question = (
                                f"Who {past} {patient}?"
                            )

                        else:
                            question = (
                                f"Who did {agent} {base}?"
                            )

                        rows.append({
                            "pair": [left, right],
                            "direction": direction,
                            "verb": past,
                            "construction": construction,
                            "question_type": question_type,
                            "role": role,
                            "agent": agent,
                            "patient": patient,
                            "first": order[0],
                            "second": order[1],
                            "sentence": sentence,
                            "question": question,
                            "gold": (
                                agent
                                if role == "agent"
                                else patient
                            ),
                            "positional_prediction": (
                                order[0]
                                if role == "agent"
                                else order[1]
                            ),
                        })

    return rows


def audit():
    rows = make_cases()

    assert len(rows) == 72

    assert len({
        name
        for x in EXAMPLES
        for name in x[:2]
    }) == 6

    for row in rows:
        for name in row["pair"]:
            assert (
                row["sentence"].count(name) == 1
            ), row

        assert row["gold"] in row["pair"]
        assert (
            row["positional_prediction"]
            in row["pair"]
        )

    for construction in CONSTRUCTIONS:
        item = next(
            r for r in rows
            if r["construction"] == construction
            and r["question_type"] == "neutral"
            and r["role"] == "agent"
        )

        print(
            f"{construction:13} "
            f"{item['sentence']}"
        )

    print("DESIGN SHA256", HASH)
    print(
        "AUDIT PASS: 72 cases, "
        "3 name pairs, 3 verbs, "
        "3 constructions"
    )
    print(
        "Primary: 36 neutral-question cases; "
        "explicit questions secondary"
    )


def ensure_script_committed():
    script = (
        "scripts/ml049_position_counterfactual.py"
    )

    commands = (
        [
            "git",
            "ls-files",
            "--error-unmatch",
            script,
        ],
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            script,
        ],
    )

    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Commit ML-049 script before "
                "--phase run: " + script
            )


def summary(rows):
    print(
        "\n=== ML-049 SEMANTIC VS "
        "POSITIONAL SELECTION ==="
    )

    print(
        "Semantic = correct grammatical role; "
        "positional = first for agent / "
        "second for patient"
    )

    for question_type in QUESTION_TYPES:
        for construction in CONSTRUCTIONS:
            group = [
                r for r in rows
                if r["question_type"] == question_type
                and r["construction"] == construction
            ]

            if not group:
                continue

            n = len(group)

            correct = sum(
                r["chosen"] == r["gold"]
                for r in group
            )

            positional = sum(
                r["chosen"]
                == r["positional_prediction"]
                for r in group
            )

            print(
                f"{question_type:8} "
                f"{construction:13} "
                f"n={n:2} "
                f"semantic={correct}/{n} "
                f"positional={positional}/{n}"
            )

            mismatches = sum(
                r["token_counts"][0]
                != r["token_counts"][1]
                for r in group
            )

            print(
                "  token-count mismatches:",
                mismatches,
            )

    if len(rows) == 72:
        cleft = [
            r for r in rows
            if r["question_type"] == "neutral"
            and r["construction"] == "object_cleft"
        ]

        active = [
            r for r in rows
            if r["question_type"] == "neutral"
            and r["construction"] == "active"
        ]

        cleft_pos = sum(
            r["chosen"] == r["positional_prediction"]
            for r in cleft
        )

        active_correct = sum(
            r["chosen"] == r["gold"]
            for r in active
        )

        print(
            f"FROZEN primary: "
            f"cleft positional matches="
            f"{cleft_pos}/12; "
            f"gate >=9: {cleft_pos >= 9}"
        )

        print(
            f"FROZEN interpretability: "
            f"active semantic accuracy="
            f"{active_correct}/12; "
            f"gate >=10: {active_correct >= 10}"
        )

        print(
            "COMPLETE; archive all outcomes, "
            "including disconfirmations"
        )

    else:
        print(f"PARTIAL {len(rows)}/72")


def run():
    ensure_script_committed()

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    cases = make_cases()

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    if OUT.exists():
        previous = json.loads(
            OUT.read_text()
        )

        if (
            previous.get("design_hash") != HASH
            or previous.get("complete")
        ):
            raise RuntimeError(
                "Existing result complete or "
                "design mismatch; refusing overwrite"
            )

        results = previous.get(
            "records", []
        )

        if len(results) > len(cases):
            raise RuntimeError(
                "Invalid resume length"
            )

        for i, item in enumerate(results):
            for key, value in cases[i].items():
                if item.get(key) != value:
                    raise RuntimeError(
                        f"Resume mismatch at "
                        f"case {i}: {key}"
                    )

    model, tokenizer = load(
        str(MODELS / "qwen7b")
    )

    for i in range(len(results), len(cases)):
        case = cases[i]

        content = (
            "Answer with exactly one person's name.\n"
            f"Event: {case['sentence']}\n"
            f"Question: {case['question']}\n"
            "Answer:"
        )

        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        left, right = case["pair"]

        lp_left, n_left = log_likelihood(
            model,
            tokenizer,
            prompt,
            left,
        )

        lp_right, n_right = log_likelihood(
            model,
            tokenizer,
            prompt,
            right,
        )

        chosen = (
            left
            if lp_left > lp_right
            else right
            if lp_right > lp_left
            else "TIE"
        )

        item = {
            **case,
            "chosen": chosen,
            "token_counts": [
                n_left,
                n_right,
            ],
            "logprobs": {
                left: lp_left,
                right: lp_right,
            },
            "gold_minus_other": (
                (lp_left - lp_right)
                if case["gold"] == left
                else (lp_right - lp_left)
            ),
        }

        results.append(item)

        OUT.write_text(
            json.dumps(
                {
                    "design_hash": HASH,
                    "complete": False,
                    "records": results,
                },
                indent=2,
            )
        )

        print(
            f"{i+1:02}/{len(cases)} "
            f"{case['construction']:13} "
            f"{case['question_type']:8} "
            f"{case['role']:7} "
            f"gold={case['gold']:7} "
            f"chosen={chosen:7} "
            f"positional="
            f"{case['positional_prediction']:7} "
            f"margin="
            f"{item['gold_minus_other']:+.2f}",
            flush=True,
        )

    OUT.write_text(
        json.dumps(
            {
                "design_hash": HASH,
                "complete": True,
                "records": results,
            },
            indent=2,
        )
    )

    summary(results)

    print("SAVED", OUT)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "audit",
            "run",
            "summarize",
        ),
    )

    args = parser.parse_args()

    if args.phase == "audit":
        audit()

    elif args.phase == "run":
        run()

    else:
        data = json.loads(
            OUT.read_text()
        )

        if data.get("design_hash") != HASH:
            raise RuntimeError(
                "Design mismatch"
            )

        summary(data["records"])


if __name__ == "__main__":
    main()
