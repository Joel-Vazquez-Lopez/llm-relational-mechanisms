
"""ML-053: prospective question-paraphrase experiment.

Workflow:
  audit -> calibrate both models -> freeze -> commit -> test -> summarize

Uses complete-name conditional log likelihoods, not generated answers.
Protected inference is blocked until the frozen record is committed.
"""

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

from ml053_design import generate, render


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ml053_predict.py"
DESIGN_SCRIPT = ROOT / "scripts/ml053_design.py"
MANIFEST = ROOT / "results/ml053_design_manifest.json"
FROZEN = ROOT / "results/ml053_frozen.json"

MODELS = ("qwen7b", "llama3b")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(model, split):
    return ROOT / f"results/ml053_{model}_{split}.json"


def committed(path):
    relative = str(path.relative_to(ROOT))

    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=ROOT,
        check=True,
    )


def write_once(path, data):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def checkpoint(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def design():
    data = json.loads(MANIFEST.read_text())

    sealed = data.pop("manifest_sha256")
    computed = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()

    if sealed != computed:
        raise RuntimeError("Design manifest SHA256 mismatch")

    assert data["spec"]["experiment"] == "ML-053"
    assert len(data["cases"]["calibration"]) == 216
    assert len(data["cases"]["protected"]) == 288
    assert len(data["selection"]["calibration"]) == 6
    assert len(data["selection"]["protected"]) == 8

    for split in ("calibration", "protected"):
        expected = [
            row
            for group in data["selection"][split]
            for row in generate(group, split)
        ]

        if expected != data["cases"][split]:
            raise RuntimeError(
                "Manifest cases disagree with design source"
            )

        ids = [row["id"] for row in expected]

        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate case IDs")

    return data, sealed


def verify_records(records, cases):
    if len(records) > len(cases):
        raise RuntimeError("Too many recorded cases")

    for row, original in zip(records, cases):
        for key, value in original.items():
            if row.get(key) != value:
                raise RuntimeError(
                    f"Case mismatch: {original['id']} {key}"
                )

        if set(row.get("scores", {})) != set(original["pair"]):
            raise RuntimeError("Score candidate mismatch")

        counts = row.get("token_counts", {})

        if set(counts) != set(original["pair"]):
            raise RuntimeError("Token-count candidate mismatch")

        if any(
            type(count) is not int or count <= 0
            for count in counts.values()
        ):
            raise RuntimeError("Invalid token counts")

        if len(set(counts.values())) != 1:
            raise RuntimeError("Unequal candidate token counts")

        if not all(
            math.isfinite(value)
            for value in row["scores"].values()
        ):
            raise RuntimeError("Nonfinite log likelihood")


def read_data(model, split, cases, sealed):
    path = output(model, split)
    data = json.loads(path.read_text())

    if (
        data["model"] != model
        or data["split"] != split
        or data["manifest_sha256"] != sealed
        or data["script_sha256"] != digest(SCRIPT)
    ):
        raise RuntimeError(f"Result metadata mismatch: {path}")

    verify_records(data["records"], cases)

    if data["complete"] and len(data["records"]) != len(cases):
        raise RuntimeError(
            f"Incomplete file marked complete: {path}"
        )

    return data


def read_frozen(sealed):
    frozen = json.loads(FROZEN.read_text())

    if (
        frozen["manifest_sha256"] != sealed
        or frozen["design_script_sha256"] != digest(DESIGN_SCRIPT)
        or frozen["script_sha256"] != digest(SCRIPT)
    ):
        raise RuntimeError(
            "Frozen metadata or source mismatch"
        )

    expected_primary = {
        "model": "qwen7b",
        "style": "paraphrase",
        "passive_construction": "passive_cleft",
        "passive_sign": "negative",
        "active_construction": "active",
        "active_sign": "positive",
        "minimum_groups_each": 6,
        "total_groups": 8,
    }

    if frozen["primary"] != expected_primary:
        raise RuntimeError("Unexpected primary gate")

    for model in MODELS:
        if frozen["calibration_sha256"][model] != digest(
            output(model, "calibration")
        ):
            raise RuntimeError(
                "Calibration changed after freeze"
            )

    return frozen


def infer(model, split):
    committed(SCRIPT)
    committed(DESIGN_SCRIPT)
    committed(MANIFEST)

    source, sealed = design()
    cases = source["cases"][split]

    if split == "protected":
        committed(FROZEN)

        for m in MODELS:
            committed(output(m, "calibration"))

        read_frozen(sealed)

        for m in MODELS:
            cal = read_data(
                m,
                "calibration",
                source["cases"]["calibration"],
                sealed,
            )

            if not cal["complete"]:
                raise RuntimeError(
                    "Calibration incomplete"
                )

    path = output(model, split)

    if path.exists():
        old = read_data(
            model,
            split,
            cases,
            sealed,
        )

        if old["complete"]:
            raise RuntimeError(
                "Existing results complete; use summarize"
            )

        records = old["records"]

        print(
            "RESUMING",
            model,
            split,
            len(records),
            "/",
            len(cases),
        )
    else:
        records = []

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS as MODEL_ROOT,
        log_likelihood,
    )

    model_path = MODEL_ROOT / model

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    llm, tokenizer = load(str(model_path))

    for idx in range(len(records), len(cases)):
        case = cases[idx]
        prompt = render(tokenizer, case)

        scores = {}
        counts = {}

        for candidate in case["pair"]:
            lp, n_tokens = log_likelihood(
                llm,
                tokenizer,
                prompt,
                candidate,
            )

            scores[candidate] = float(lp)
            counts[candidate] = int(n_tokens)

        row = {
            **case,
            "scores": scores,
            "token_counts": counts,
        }

        verify_records([row], [case])
        records.append(row)

        checkpoint(
            path,
            {
                "model": model,
                "split": split,
                "manifest_sha256": sealed,
                "script_sha256": digest(SCRIPT),
                "complete": False,
                "records": records,
            },
        )

        if (
            (idx + 1) % 12 == 0
            or idx + 1 == len(cases)
        ):
            print(
                f"{model} {split}: "
                f"{idx + 1}/{len(cases)}",
                flush=True,
            )

    checkpoint(
        path,
        {
            "model": model,
            "split": split,
            "manifest_sha256": sealed,
            "script_sha256": digest(SCRIPT),
            "complete": True,
            "records": records,
        },
    )

    print("COMPLETE", path)


def pairs(records):
    grouped = defaultdict(dict)

    for row in records:
        key = (
            tuple(row["pair"]),
            row["verb"],
            row["direction"],
            row["construction"],
            row["style"],
        )

        if row["role"] in grouped[key]:
            raise RuntimeError("Duplicate question role")

        grouped[key][row["role"]] = row

    output_pairs = []

    for key, roles in grouped.items():
        if set(roles) != {"agent", "patient"}:
            raise RuntimeError("Unpaired questions")

        agent_q = roles["agent"]
        patient_q = roles["patient"]

        if (
            agent_q["passage"] != patient_q["passage"]
            or agent_q["agent"] != patient_q["agent"]
        ):
            raise RuntimeError(
                "Mismatched paired passage"
            )

        agent = agent_q["agent"]
        patient = agent_q["patient"]

        agent_margin = (
            agent_q["scores"][agent]
            - agent_q["scores"][patient]
        )

        patient_margin = (
            patient_q["scores"][agent]
            - patient_q["scores"][patient]
        )

        if agent_margin == 0 or patient_margin == 0:
            raise RuntimeError(
                "Exact participant tie: inspect manually"
            )

        output_pairs.append({
            "lexical_group": (*key[0], key[1]),
            "direction": key[2],
            "construction": key[3],
            "style": key[4],
            "D": agent_margin - patient_margin,
            "correct_switch": (
                agent_margin > 0 and patient_margin < 0
            ),
            "wrong_switch": (
                agent_margin < 0 and patient_margin > 0
            ),
            "same_choice": (
                (agent_margin > 0) == (patient_margin > 0)
            ),
            "agent_margin_at_agent_question": agent_margin,
            "agent_margin_at_patient_question": patient_margin,
        })

    if len(output_pairs) * 2 != len(records):
        raise RuntimeError("Incorrect pair count")

    return output_pairs


def summary(model, split, emit=True):
    source, sealed = design()

    data = read_data(
        model,
        split,
        source["cases"][split],
        sealed,
    )

    if not data["complete"]:
        raise RuntimeError(
            "Summarize requires complete results"
        )

    observations = pairs(data["records"])
    selected = source["selection"][split]
    n_groups = len(selected)

    rows_by_condition = {}

    for construction in source["spec"]["constructions"]:
        for style in source["spec"]["styles"]:
            subset = [
                row
                for row in observations
                if row["construction"] == construction
                and row["style"] == style
            ]

            if len(subset) != n_groups * 2:
                raise RuntimeError("Incomplete condition")

            group_means = {}

            for group in selected:
                grouped = [
                    row["D"]
                    for row in subset
                    if row["lexical_group"] == tuple(group)
                ]

                if len(grouped) != 2:
                    raise RuntimeError(
                        "Missing name-swap direction"
                    )

                group_means["|".join(group)] = mean(grouped)

            key = f"{construction}/{style}"

            rows_by_condition[key] = {
                "n_pairs": len(subset),
                "mean_D": mean(
                    row["D"] for row in subset
                ),
                "positive_D_pairs": sum(
                    row["D"] > 0 for row in subset
                ),
                "negative_D_pairs": sum(
                    row["D"] < 0 for row in subset
                ),
                "correct_switches": sum(
                    row["correct_switch"] for row in subset
                ),
                "wrong_switches": sum(
                    row["wrong_switch"] for row in subset
                ),
                "same_choice": sum(
                    row["same_choice"] for row in subset
                ),
                "lexical_group_mean_D": group_means,
            }

    if emit:
        print(
            f"\n{model.upper()} {split.upper()}: "
            f"{len(observations)} paired sentences"
        )

        print(
            "Condition                    "
            "mean D   D>0  correct  wrong  same"
        )

        for name, metrics in rows_by_condition.items():
            print(
                f"{name:28s} "
                f"{metrics['mean_D']:+8.3f} "
                f"{metrics['positive_D_pairs']:>3}/"
                f"{metrics['n_pairs']:<3} "
                f"{metrics['correct_switches']:>7} "
                f"{metrics['wrong_switches']:>6} "
                f"{metrics['same_choice']:>5}"
            )

    if split == "protected" and model == "qwen7b":
        negative = rows_by_condition[
            "passive_cleft/paraphrase"
        ]

        positive = rows_by_condition[
            "active/paraphrase"
        ]

        n_negative = sum(
            value < 0
            for value in negative[
                "lexical_group_mean_D"
            ].values()
        )

        n_positive = sum(
            value > 0
            for value in positive[
                "lexical_group_mean_D"
            ].values()
        )

        passed = (
            n_negative >= 6
            and n_positive >= 6
        )

        if emit:
            print(
                "\nFROZEN PRIMARY: "
                f"passive-negative {n_negative}/8; "
                f"active-positive {n_positive}/8 => "
                f"{'PASS' if passed else 'FAIL'}"
            )

            print(
                "All passive-cleft/paraphrase group means:"
            )

            for group, value in negative[
                "lexical_group_mean_D"
            ].items():
                print(
                    f"  {group:39s} "
                    f"D={value:+9.3f}"
                )

    return rows_by_condition


def freeze():
    for path in (
        SCRIPT,
        DESIGN_SCRIPT,
        MANIFEST,
    ):
        committed(path)

    source, sealed = design()

    for model in MODELS:
        path = output(model, "calibration")
        committed(path)

        data = read_data(
            model,
            "calibration",
            source["cases"]["calibration"],
            sealed,
        )

        if not data["complete"]:
            raise RuntimeError(
                "All calibrations must be complete"
            )

    if (
        FROZEN.exists()
        or any(
            output(model, "protected").exists()
            for model in MODELS
        )
    ):
        raise RuntimeError(
            "Freeze cannot overwrite or follow "
            "protected results"
        )

    for model in MODELS:
        summary(model, "calibration")

    write_once(
        FROZEN,
        {
            "experiment": "ML-053",
            "manifest_sha256": sealed,
            "design_script_sha256": digest(
                DESIGN_SCRIPT
            ),
            "script_sha256": digest(SCRIPT),
            "calibration_sha256": {
                model: digest(
                    output(model, "calibration")
                )
                for model in MODELS
            },
            "primary": {
                "model": "qwen7b",
                "style": "paraphrase",
                "passive_construction": "passive_cleft",
                "passive_sign": "negative",
                "active_construction": "active",
                "active_sign": "positive",
                "minimum_groups_each": 6,
                "total_groups": 8,
            },
            "notes": (
                "Predeclared in sealed manifest; "
                "no fitting; secondary comparisons exploratory."
            ),
        },
    )

    print(
        "FROZEN",
        FROZEN,
        "SHA256",
        digest(FROZEN),
    )

    print(
        "Commit frozen record and both calibration "
        "files before any protected inference"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "audit",
            "calibrate",
            "freeze",
            "test",
            "summarize",
        ),
    )

    parser.add_argument(
        "--model",
        choices=MODELS,
        default="qwen7b",
    )

    args = parser.parse_args()

    if args.phase == "audit":
        source, sealed = design()

        print(
            "AUDIT PASS",
            sealed,
            "calibration=216 protected=288",
        )

        print(
            "Primary: Qwen passive-cleft/paraphrase "
            "<0 in >=6/8 groups, "
            "active/paraphrase >0 in >=6/8 groups"
        )

    elif args.phase == "calibrate":
        infer(args.model, "calibration")

    elif args.phase == "freeze":
        freeze()

    elif args.phase == "test":
        infer(args.model, "protected")
        summary(args.model, "protected")

    elif args.phase == "summarize":
        summary(args.model, "protected")


if __name__ == "__main__":
    main()
