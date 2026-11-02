
"""ML-055: frozen, prospective Llama 3B question-wording replication.

Phases: audit -> seal -> git commit -> run -> summarize.
No protected model inference is permitted before the freeze is committed.
Existing data are never overwritten; interrupted run is resumable.
"""

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "scripts/ml055_experiment.py"
DESIGN = ROOT / "scripts/ml055_design.py"
SCORE = ROOT / "scripts/ml054_measurement_audit.py"
RENDER = ROOT / "scripts/ml053_design.py"
BASE_SCORER = ROOT / "scripts/ml003_role_margins.py"
FROZEN = ROOT / "results/ml055_frozen.json"
OUTPUT = ROOT / "results/ml055_llama3b_results.json"

EXPECTED_CASES = (
    "beacbe0b65d26ac518d60a432e35c724e8f54fdc7da934c7956d2dc36783d618"
)

HYPOTHESIS = {
    "experiment": "ML-055",
    "kind": "prospective question-wording replication and comparison",
    "population": "One Llama 3B checkpoint, 16 new lexical blocks, 2 name-order directions",
    "primary_metric": "completed-answer likelihood preferred-name switches",
    "primary_unit": "name-pair/verb lexical block (NOT individual prompts)",
    "primary_contrast": "wrong-direction switches in original minus parallel_active questions",
    "primary_success_requires_all": [
        "original wrong-direction switches >= 16/32 paired events",
        "original - parallel_active wrong-direction switches >= 8/32 events",
        "at least 12/16 lexical blocks have strictly more original than parallel_active wrong switches",
    ],
    "secondary": [
        "original generic and parallel_active per-role accuracy",
        "original and parallel_active candidate-name likelihood switches",
        "greedy generated name switches; unresolved answers remain unresolved",
        "per-block and per-verb variation and potential answer-format effects",
    ],
    "interpretation": (
        "A behavioral wording effect, not proof of a separate "
        "event representation or question-processing module"
    ),
    "selection_rule": "All 192 static rows in manifest, no exclusions after scoring",
    "no_posthoc_changes": True,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_and_hash():
    from ml055_design import design

    rows = design()

    if len(rows) != 192 or len({r["id"] for r in rows}) != 192:
        raise RuntimeError("Expected exactly 192 unique cases")

    if len({r["group"] for r in rows}) != 16:
        raise RuntimeError("Expected exactly 16 lexical groups")

    if len({(r["group"], r["direction"]) for r in rows}) != 32:
        raise RuntimeError("Expected exactly 32 events")

    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()

    if digest != EXPECTED_CASES:
        raise RuntimeError(
            f"Unexpected design hash: {digest}"
        )

    return rows, digest


def tokenize_check(rows):
    from transformers import AutoTokenizer
    from ml003_role_margins import MODELS
    from ml053_design import render, answer_tokens

    for model_name in ("llama3b", "qwen7b"):
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODELS / model_name),
            local_files_only=True,
        )

        for row in rows:
            prompt = render(tokenizer, row)

            a = answer_tokens(
                tokenizer, prompt, row["agent"]
            )

            p = answer_tokens(
                tokenizer, prompt, row["patient"]
            )

            if a is None or p is None or a == 0 or a != p:
                raise RuntimeError(
                    f"Tokenizer mismatch {model_name}: "
                    f"{row['id']} {a}, {p}"
                )

        print(
            f"TOKENIZER PASS: {model_name} "
            f"all {len(rows)} rows"
        )


def write_once(path, data):
    if path.exists():
        raise FileExistsError(
            f"Refusing overwrite of {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2) + "\n"
    )


def checkpoint(path, data):
    tmp = path.with_suffix(".json.tmp")

    tmp.write_text(
        json.dumps(data, indent=2) + "\n"
    )

    tmp.replace(path)


def committed(path):
    relative = str(path.relative_to(ROOT))

    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            relative,
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            relative,
        ],
        cwd=ROOT,
        check=True,
    )


def source_provenance():
    paths = (
        ENGINE,
        DESIGN,
        SCORE,
        RENDER,
        BASE_SCORER,
    )

    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in paths
    }


def read_frozen():
    if not FROZEN.exists():
        raise FileNotFoundError(
            "No ML-055 freeze. Seal and commit before running."
        )

    frozen = json.loads(FROZEN.read_text())

    rows, cases_sha = rows_and_hash()

    if frozen.get("design_hash") != cases_sha:
        raise RuntimeError(
            "Frozen design mismatch"
        )

    if frozen.get("source_sha256") != source_provenance():
        raise RuntimeError(
            "Frozen code changed; do not run inference"
        )

    if frozen.get("hypothesis") != HYPOTHESIS:
        raise RuntimeError(
            "Frozen prediction mismatch"
        )

    return rows, frozen


def preferred(row, field):
    if field == "generated_name":
        return row.get("generated_name")

    scores = row[field]

    agent = row["agent"]
    patient = row["patient"]

    if not (
        math.isfinite(scores[agent])
        and math.isfinite(scores[patient])
    ):
        return None

    if scores[agent] == scores[patient]:
        return None

    if scores[agent] > scores[patient]:
        return agent

    return patient


def summarize(records):
    if len(records) != 192:
        raise RuntimeError(
            "Incomplete output"
        )

    by_event = defaultdict(dict)

    for r in records:
        key = (
            r["group"],
            r["direction"],
            r["style"],
        )

        if r["role"] in by_event[key]:
            raise RuntimeError(
                "Duplicate role in event"
            )

        by_event[key][r["role"]] = r

    if len(by_event) != 96:
        raise RuntimeError(
            "Expected 96 role-question event pairs"
        )

    result = {}

    for measurement in (
        "name_score",
        "turn_score",
        "generated_name",
    ):
        stats = {
            style: {
                "correct_switch": 0,
                "wrong_switch": 0,
                "same_choice": 0,
                "unresolved": 0,
                "agent_correct": 0,
                "patient_correct": 0,
            }
            for style in (
                "original",
                "parallel_active",
                "generic",
            )
        }

        block_wrong = {
            style: {
                str(i): 0
                for i in range(16)
            }
            for style in stats
        }

        for (group, direction, style), pair in by_event.items():
            if set(pair) != {"agent", "patient"}:
                raise RuntimeError(
                    "Incomplete role pair"
                )

            agent_row = pair["agent"]
            patient_row = pair["patient"]

            a = preferred(
                agent_row, measurement
            )

            p = preferred(
                patient_row, measurement
            )

            st = stats[style]

            st["agent_correct"] += (
                a == agent_row["agent"]
            )

            st["patient_correct"] += (
                p == patient_row["patient"]
            )

            if a is None or p is None:
                st["unresolved"] += 1

            elif (
                a == agent_row["agent"]
                and p == patient_row["patient"]
            ):
                st["correct_switch"] += 1

            elif (
                a == agent_row["patient"]
                and p == patient_row["agent"]
            ):
                st["wrong_switch"] += 1

                block_wrong[style][str(group)] += 1

            elif a == p:
                st["same_choice"] += 1

            else:
                raise RuntimeError(
                    "Unclassified event"
                )

        differences = {
            str(i): (
                block_wrong["original"][str(i)]
                - block_wrong["parallel_active"][str(i)]
            )
            for i in range(16)
        }

        orig = stats["original"]["wrong_switch"]

        parallel = stats["parallel_active"]["wrong_switch"]

        positive = sum(
            d > 0
            for d in differences.values()
        )

        result[measurement] = {
            "by_style": stats,
            "wrong_switches_by_lexical_block": block_wrong,
            "original_minus_parallel_wrong_switches": (
                orig - parallel
            ),
            "blocks_positive": positive,
            "blocks_tied": sum(
                d == 0
                for d in differences.values()
            ),
            "blocks_negative": sum(
                d < 0
                for d in differences.values()
            ),
            "primary_gate_pass": (
                orig >= 16
                and orig - parallel >= 8
                and positive >= 12
            )
            if measurement == "turn_score"
            else None,
        }

    result["exact_name_generations"] = sum(
        r.get("generated_name") is not None
        for r in records
    )

    result["total_prompts"] = len(records)

    return result


def run():
    rows, frozen = read_frozen()

    for path in (
        ENGINE,
        DESIGN,
        SCORE,
        RENDER,
        BASE_SCORER,
        FROZEN,
    ):
        committed(path)

    frozen_hash = sha(FROZEN)

    report = {
        "experiment": "ML-055",
        "model": "llama3b",
        "frozen_sha256": frozen_hash,
        "complete": False,
        "records": [],
    }

    if OUTPUT.exists():
        report = json.loads(
            OUTPUT.read_text()
        )

        if report.get("complete"):
            raise RuntimeError(
                "ML-055 completed; refusing re-run"
            )

        if (
            report.get("frozen_sha256") != frozen_hash
            or report.get("model") != "llama3b"
        ):
            raise RuntimeError(
                "Partial results incompatible with freeze"
            )

        if len(report["records"]) > len(rows):
            raise RuntimeError(
                "Too many partial records"
            )

        for old, row in zip(report["records"], rows):
            if any(
                old.get(k) != v
                for k, v in row.items()
            ):
                raise RuntimeError(
                    "Partial row mismatch"
                )

    from mlx_lm import load, generate
    from ml003_role_margins import MODELS
    from ml053_design import render

    from ml054_measurement_audit import (
        score_name_and_turn,
        exact_generated_name,
    )

    model, tokenizer = load(
        str(MODELS / "llama3b")
    )

    for i in range(
        len(report["records"]),
        len(rows),
    ):
        row = rows[i]

        prompt = render(
            tokenizer, row
        )

        name_scores = {}
        turn_scores = {}

        for name in (
            row["agent"],
            row["patient"],
        ):
            (
                name_scores[name],
                turn_scores[name],
                _,
            ) = score_name_and_turn(
                model,
                tokenizer,
                prompt,
                row,
                name,
            )

        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=14,
            verbose=False,
        )

        report["records"].append({
            **row,
            "name_score": name_scores,
            "turn_score": turn_scores,
            "generated_text": generated,
            "generated_name": exact_generated_name(
                generated,
                (
                    row["agent"],
                    row["patient"],
                ),
            ),
        })

        checkpoint(
            OUTPUT, report
        )

        if (i + 1) % 12 == 0:
            print(
                f"ML-055 progress: {i+1}/192",
                flush=True,
            )

    report["complete"] = True

    report["summary"] = summarize(
        report["records"]
    )

    checkpoint(
        OUTPUT, report
    )

    print(
        json.dumps(
            report["summary"],
            indent=2,
        )
    )

    print(
        "RESULTS:", OUTPUT
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "audit",
            "seal",
            "run",
            "summarize",
        ),
        required=True,
    )

    args = parser.parse_args()

    if args.phase == "audit":
        rows, digest = rows_and_hash()

        tokenize_check(rows)

        print(
            "CASE HASH:", digest
        )

        print(
            "AUDIT PASS: no inference, no files written"
        )

    elif args.phase == "seal":
        rows, digest = rows_and_hash()

        tokenize_check(rows)

        write_once(
            FROZEN,
            {
                "hypothesis": HYPOTHESIS,
                "design_hash": digest,
                "source_sha256": source_provenance(),
                "cases": rows,
            },
        )

        print(
            "FROZEN:", FROZEN
        )

        print(
            "SHA256:", sha(FROZEN)
        )

        print(
            "Commit freeze and code before running"
        )

    elif args.phase == "run":
        run()

    else:
        rows, frozen = read_frozen()

        report = json.loads(
            OUTPUT.read_text()
        )

        if (
            not report["complete"]
            or report["frozen_sha256"] != sha(FROZEN)
        ):
            raise RuntimeError(
                "Incomplete/mismatched results"
            )

        for old, row in zip(
            report["records"], rows
        ):
            if any(
                old.get(k) != v
                for k, v in row.items()
            ):
                raise RuntimeError(
                    "Output design changed"
                )

        print(
            json.dumps(
                summarize(report["records"]),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
