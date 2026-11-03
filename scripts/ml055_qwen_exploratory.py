
"""Exploratory Qwen 7B follow-up using the exact frozen ML-055 cases.

Phases: audit -> commit script -> run -> summarize.

Does not modify ML-055's freeze or Llama results.
Qwen results are exploratory because Llama outcomes are already known.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import ml055_experiment as ml055


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ml055_qwen_exploratory.py"
OUTPUT = ROOT / "results/ml055_qwen7b_exploratory.json"
LLAMA = ROOT / "results/ml055_llama3b_results.json"
MODEL_NAME = "qwen7b"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    # Reuse the original frozen design and code-integrity checks.
    cases, frozen = ml055.read_frozen()
    ml055.tokenize_check(cases)

    assert len(cases) == 192

    print("AUDIT PASS: exact ML-055 frozen design and tokenizers")
    print("CASE HASH:", frozen["design_hash"])
    print("FROZEN SHA256:", sha256(ml055.FROZEN))
    print("No model inference performed")

    return cases


def validate_records(records, cases):
    if len(records) > len(cases):
        raise RuntimeError("Too many output records")

    for old, expected in zip(records, cases):
        if any(old.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                f"Case mismatch: {expected['id']}"
            )

        for field in ("name_score", "turn_score"):
            if field not in old:
                raise RuntimeError(
                    f"Missing {field}: {expected['id']}"
                )

            if set(old[field]) != {
                old["agent"], old["patient"]
            }:
                raise RuntimeError(
                    f"Candidate mismatch: {expected['id']}"
                )

            if not all(
                math.isfinite(v)
                for v in old[field].values()
            ):
                raise RuntimeError(
                    f"Nonfinite score: {expected['id']}"
                )


def load_report(cases):
    frozen_sha = sha256(ml055.FROZEN)
    script_sha = sha256(SCRIPT)

    if OUTPUT.exists():
        report = json.loads(OUTPUT.read_text())

        if (
            report.get("model") != MODEL_NAME
            or report.get("frozen_sha256") != frozen_sha
            or report.get("runner_sha256") != script_sha
        ):
            raise RuntimeError(
                "Output provenance differs from current code/freeze"
            )

        validate_records(report.get("records", []), cases)
        return report

    return {
        "experiment": "ML-055-QWEN-EXPLORATORY",
        "model": MODEL_NAME,
        "frozen_sha256": frozen_sha,
        "runner_sha256": script_sha,
        "complete": False,
        "records": [],
    }


def run():
    cases = audit()

    # Require the original frozen files and this runner
    # to be committed and unchanged.
    for path in (
        ml055.ENGINE,
        ml055.DESIGN,
        ml055.SCORE,
        ml055.RENDER,
        ml055.BASE_SCORER,
        ml055.FROZEN,
        SCRIPT,
    ):
        ml055.committed(path)

    report = load_report(cases)

    if report["complete"]:
        raise RuntimeError(
            "Qwen run already complete; refusing overwrite"
        )

    from mlx_lm import load, generate
    from ml003_role_margins import MODELS
    from ml053_design import render
    from ml054_measurement_audit import (
        score_name_and_turn,
        exact_generated_name,
    )

    model, tokenizer = load(
        str(MODELS / MODEL_NAME)
    )

    start = len(report["records"])

    print(
        f"Resuming Qwen from prompt {start + 1}/192",
        flush=True,
    )

    for i in range(start, len(cases)):
        row = cases[i]

        prompt = render(tokenizer, row)

        name_scores = {}
        turn_scores = {}

        for name in (
            row["agent"],
            row["patient"],
        ):
            name_score, turn_score, _ = (
                score_name_and_turn(
                    model,
                    tokenizer,
                    prompt,
                    row,
                    name,
                )
            )

            name_scores[name] = name_score
            turn_scores[name] = turn_score

        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=14,
            verbose=False,
        )

        pair = (
            row["agent"],
            row["patient"],
        )

        report["records"].append({
            **row,
            "name_score": name_scores,
            "turn_score": turn_scores,
            "generated_text": generated,
            "generated_name": exact_generated_name(
                generated,
                pair,
            ),
        })

        # Resumable checkpoint after each prompt.
        ml055.checkpoint(OUTPUT, report)

        if (i + 1) % 12 == 0:
            print(
                f"Qwen progress: {i + 1}/192",
                flush=True,
            )

    validate_records(report["records"], cases)

    report["summary"] = ml055.summarize(
        report["records"]
    )

    report["complete"] = True

    ml055.checkpoint(OUTPUT, report)

    print("Qwen evaluation complete:", OUTPUT)

    show_summary(cases)


def show_summary(cases):
    if not OUTPUT.exists():
        raise FileNotFoundError(
            "No Qwen results yet. Run --phase run first."
        )

    report = load_report(cases)

    if (
        not report["complete"]
        or len(report["records"]) != 192
    ):
        raise RuntimeError(
            "Qwen results are incomplete"
        )

    qwen = ml055.summarize(
        report["records"]
    )

    print("\nQWEN FULL SUMMARY")
    print(json.dumps(qwen, indent=2))

    if not LLAMA.exists():
        print(
            "\nLlama results not found locally; "
            "skipping comparison"
        )
        return

    llama_report = json.loads(
        LLAMA.read_text()
    )

    if not llama_report.get("complete"):
        raise RuntimeError(
            "Llama results are incomplete"
        )

    if (
        llama_report.get("frozen_sha256")
        != sha256(ml055.FROZEN)
    ):
        raise RuntimeError(
            "Llama and Qwen do not share the same freeze"
        )

    if len(llama_report.get("records", [])) != 192:
        raise RuntimeError(
            "Unexpected Llama record count"
        )

    validate_records(
        llama_report["records"],
        cases,
    )

    llama = ml055.summarize(
        llama_report["records"]
    )

    print(
        "\nSIDE-BY-SIDE "
        "(COUNTS OUT OF 32 EVENTS PER STYLE)"
    )

    for measurement in (
        "name_score",
        "turn_score",
        "generated_name",
    ):
        print(f"\n{measurement}:")

        for style in (
            "original",
            "parallel_active",
            "generic",
        ):
            a = llama[measurement]["by_style"][style]
            b = qwen[measurement]["by_style"][style]

            print(
                f"  {style:16s}  "
                f"Llama correct/wrong/same/unresolved="
                f"{a['correct_switch']}/"
                f"{a['wrong_switch']}/"
                f"{a['same_choice']}/"
                f"{a['unresolved']}  "
                f"Qwen="
                f"{b['correct_switch']}/"
                f"{b['wrong_switch']}/"
                f"{b['same_choice']}/"
                f"{b['unresolved']}"
            )

    print(
        "\nExploratory comparison only: "
        "no new preregistered Qwen gate."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "audit",
            "run",
            "summarize",
        ),
        required=True,
    )

    args = parser.parse_args()

    if args.phase == "audit":
        audit()

    elif args.phase == "run":
        run()

    else:
        show_summary(audit())


if __name__ == "__main__":
    main()
