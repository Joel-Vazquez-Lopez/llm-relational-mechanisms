
"""ML-052: calibration, frozen cue predictions, protected structural holdout.

Requires the already committed ML-052 design manifest. Calibrates only on its
96 calibration prompts; tests on its 120 protected prompts after a Git freeze.
Scores are complete-name conditional log likelihoods, not generated answers.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


SCRIPT = Path("scripts/ml052_predict.py")
MANIFEST = Path("results/ml052_design_manifest.json")

MODELS = ("qwen7b", "llama3b")
LAMBDA = 8.0

FAMILIES = {
    "role_only": (
        "const",
        "agent_question",
    ),
    "position": (
        "const",
        "agent_question",
        "first_correct",
        "subject_correct",
    ),
    "construction": (
        "const",
        "agent_question",
        "passive",
        "cleft",
        "agent_passive",
        "agent_cleft",
    ),
    "cue_additive": (
        "const",
        "agent_question",
        "passive",
        "cleft",
        "first_correct",
        "subject_correct",
        "focus_correct",
    ),
    "cue_contextual": (
        "const",
        "agent_question",
        "passive",
        "cleft",
        "first_correct",
        "subject_correct",
        "focus_correct",
        "focus_agent_question",
    ),
}

BASELINES = (
    "role_only",
    "position",
    "construction",
)

PRIMARY = "cue_additive"


def digest(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def output_path(model, stage):
    return Path(
        f"results/ml052_{model}_{stage}.json"
    )


def git_clean(path):
    commands = (
        (
            "git",
            "ls-files",
            "--error-unmatch",
            str(path),
        ),
        (
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            str(path),
        ),
    )

    for command in commands:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
        )


def manifest_data():
    raw = json.loads(
        MANIFEST.read_text()
    )

    claimed = raw.pop(
        "manifest_sha256"
    )

    actual = hashlib.sha256(
        json.dumps(
            raw,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if actual != claimed:
        raise RuntimeError(
            "Manifest contents do not match sealed SHA"
        )

    calibration = raw["cases"]["calibration"]
    protected = raw["cases"]["protected"]

    if (
        len(calibration) != 96
        or len(protected) != 120
    ):
        raise RuntimeError(
            "Unexpected manifest case counts"
        )

    if any(
        r["construction"] == "passive_cleft"
        for r in calibration
    ):
        raise RuntimeError(
            "Structural holdout leaked into calibration"
        )

    n_holdout = sum(
        r["construction"] == "passive_cleft"
        for r in protected
    )

    if n_holdout != 24:
        raise RuntimeError(
            "Unexpected held-out case count"
        )

    return raw, claimed


def features(case):
    construction = case["construction"]

    agent = case["agent"]
    patient = case["patient"]
    gold = case["gold"]

    if construction in (
        "passive",
        "object_cleft",
        "passive_cleft",
    ):
        first = patient
    else:
        first = agent

    if construction in (
        "passive",
        "passive_cleft",
    ):
        subject = patient
    else:
        subject = agent

    focus = {
        "object_cleft": patient,
        "subject_cleft": agent,
        "passive_cleft": patient,
    }.get(construction)

    def signed(person):
        if person is None:
            return 0.0

        return (
            1.0
            if person == gold
            else -1.0
        )

    agent_question = (
        1.0
        if case["role"] == "agent"
        else -1.0
    )

    passive = float(
        construction in (
            "passive",
            "passive_cleft",
        )
    )

    cleft = float(
        "cleft" in construction
    )

    x = {
        "const": 1.0,
        "agent_question": agent_question,
        "passive": passive,
        "cleft": cleft,
        "agent_passive": (
            agent_question * passive
        ),
        "agent_cleft": (
            agent_question * cleft
        ),
        "first_correct": signed(first),
        "subject_correct": signed(subject),
        "focus_correct": signed(focus),
    }

    x["focus_agent_question"] = (
        x["focus_correct"]
        * agent_question
    )

    return x


def vector(row, family):
    x = features(row)

    return np.array(
        [
            x[name]
            for name in FAMILIES[family]
        ],
        dtype=np.float64,
    )


def margin(row):
    gold = row["gold"]

    other = next(
        name
        for name in row["pair"]
        if name != gold
    )

    return float(
        row["scores"][gold]
        - row["scores"][other]
    )


def fit(rows, family):
    X = np.stack([
        vector(row, family)
        for row in rows
    ])

    y = np.array(
        [
            margin(row)
            for row in rows
        ],
        dtype=np.float64,
    )

    penalty = (
        np.eye(X.shape[1])
        * LAMBDA
    )

    # Do not penalize the intercept.
    penalty[0, 0] = 0.0

    beta = np.linalg.solve(
        X.T @ X + penalty,
        X.T @ y,
    )

    return beta


def estimates(rows, family, beta):
    return np.array(
        [
            vector(row, family) @ beta
            for row in rows
        ]
    )


def model_stats(rows, predicted):
    observed = np.array(
        [
            margin(row)
            for row in rows
        ]
    )

    mse = float(
        np.mean(
            (observed - predicted) ** 2
        )
    )

    sign_matches = int(
        np.sum(
            (observed < 0)
            == (predicted < 0)
        )
    )

    return {
        "n": len(rows),
        "mse": mse,
        "error_sign_matches": sign_matches,
        "observed_wrong": int(
            np.sum(observed < 0)
        ),
        "predicted_wrong": int(
            np.sum(predicted < 0)
        ),
    }


def case_keys(rows):
    return [
        row["id"]
        for row in rows
    ]


def check_rows(rows, target):
    if case_keys(rows) != case_keys(target):
        raise RuntimeError(
            "Record IDs are not exactly "
            "in sealed manifest order"
        )

    for row, case in zip(
        rows,
        target,
    ):
        for key, value in case.items():
            if row.get(key) != value:
                raise RuntimeError(
                    f"Case mismatch: "
                    f"{row.get('id')} {key}"
                )

        if set(row["scores"]) != set(
            case["pair"]
        ):
            raise RuntimeError(
                "Candidate names mismatch"
            )

        token_counts = (
            row["token_counts"].values()
        )

        if len(set(token_counts)) != 1:
            raise RuntimeError(
                "Token eligibility not met "
                "during inference"
            )

        if not all(
            np.isfinite(value)
            for value in row["scores"].values()
        ):
            raise RuntimeError(
                "Nonfinite model log likelihood"
            )


def save(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        ".json.tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )

    temp.replace(path)


def audit():
    manifest, claimed = manifest_data()

    calibration = manifest[
        "cases"
    ]["calibration"]

    protected = manifest[
        "cases"
    ]["protected"]

    print(
        "DESIGN SHA256",
        claimed,
    )

    print(
        "Cases:",
        len(calibration),
        "calibration,",
        len(protected),
        "protected",
    )

    assert len(set(
        case_keys(calibration)
    )) == len(calibration)

    assert len(set(
        case_keys(protected)
    )) == len(protected)

    for family in FAMILIES:
        X = np.stack([
            vector(case, family)
            for case in calibration
        ])

        rank = int(
            np.linalg.matrix_rank(X)
        )

        print(
            f"{family:16s}: "
            f"columns={X.shape[1]} "
            f"rank={rank}"
        )

        if rank != X.shape[1]:
            raise RuntimeError(
                "Non-identifiable feature family: "
                + family
            )

    print(
        "AUDIT PASS: case integrity, "
        "disjoint holdout, "
        "identifiable factors"
    )


def infer(model_name, split):
    git_clean(SCRIPT)
    git_clean(MANIFEST)

    manifest, design_hash = (
        manifest_data()
    )

    cases = manifest["cases"][split]

    out = output_path(
        model_name,
        split,
    )

    recorded = []

    if split == "protected":
        for stage in (
            "calibration",
            "frozen",
        ):
            git_clean(
                output_path(
                    model_name,
                    stage,
                )
            )

        frozen = json.loads(
            output_path(
                model_name,
                "frozen",
            ).read_text()
        )

        if (
            frozen["manifest_sha256"]
            != design_hash
            or frozen["calibration_sha256"]
            != digest(
                output_path(
                    model_name,
                    "calibration",
                )
            )
            or frozen["script_sha256"]
            != digest(SCRIPT)
        ):
            raise RuntimeError(
                "Protected inference blocked: "
                "freeze integrity failed"
            )

    if out.exists():
        existing = json.loads(
            out.read_text()
        )

        if (
            existing["complete"]
            or existing["model"] != model_name
            or existing["manifest_sha256"]
            != design_hash
            or existing["script_sha256"]
            != digest(SCRIPT)
        ):
            raise RuntimeError(
                "Existing results complete "
                "or incompatible"
            )

        recorded = existing["records"]

        check_rows(
            recorded,
            cases[:len(recorded)],
        )

        print(
            "RESUMING",
            len(recorded),
            "/",
            len(cases),
        )

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS as ROOT,
        log_likelihood,
    )

    from ml052_design import prompt

    model_path = (
        ROOT / model_name
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    model, tokenizer = load(
        str(model_path)
    )

    for i in range(
        len(recorded),
        len(cases),
    ):
        case = cases[i]

        rendered = prompt(
            tokenizer,
            case,
        )

        scores = {}
        tokens = {}

        for candidate in case["pair"]:
            value, count = log_likelihood(
                model,
                tokenizer,
                rendered,
                candidate,
            )

            scores[candidate] = float(
                value
            )

            tokens[candidate] = int(
                count
            )

        row = {
            **case,
            "scores": scores,
            "token_counts": tokens,
        }

        check_rows(
            [row],
            [case],
        )

        recorded.append(row)

        save(
            out,
            {
                "model": model_name,
                "split": split,
                "manifest_sha256": design_hash,
                "script_sha256": digest(
                    SCRIPT
                ),
                "complete": False,
                "records": recorded,
            },
        )

        print(
            f"{i+1:3}/{len(cases)} "
            f"{case['construction']:14s} "
            f"{case['role']:7s} "
            f"margin={margin(row):+.3f}",
            flush=True,
        )

    save(
        out,
        {
            "model": model_name,
            "split": split,
            "manifest_sha256": design_hash,
            "script_sha256": digest(
                SCRIPT
            ),
            "complete": True,
            "records": recorded,
        },
    )

    print(
        "SAVED",
        out,
    )


def freeze(model_name):
    git_clean(SCRIPT)
    git_clean(MANIFEST)

    manifest, design_hash = (
        manifest_data()
    )

    calibration_path = output_path(
        model_name,
        "calibration",
    )

    frozen_path = output_path(
        model_name,
        "frozen",
    )

    protected_path = output_path(
        model_name,
        "protected",
    )

    if (
        frozen_path.exists()
        or protected_path.exists()
    ):
        raise RuntimeError(
            "Refusing to freeze after "
            "frozen/protected file exists"
        )

    calibration = json.loads(
        calibration_path.read_text()
    )

    if (
        not calibration["complete"]
        or calibration["model"] != model_name
        or calibration["manifest_sha256"]
        != design_hash
        or calibration["script_sha256"]
        != digest(SCRIPT)
    ):
        raise RuntimeError(
            "Calibration integrity failed"
        )

    check_rows(
        calibration["records"],
        manifest["cases"]["calibration"],
    )

    protected = manifest[
        "cases"
    ]["protected"]

    frozen = {
        "model": model_name,
        "manifest_sha256": design_hash,
        "script_sha256": digest(
            SCRIPT
        ),
        "calibration_sha256": digest(
            calibration_path
        ),
        "lambda": LAMBDA,
        "families": {},
        "protected_predictions": {},
    }

    for family in FAMILIES:
        beta = fit(
            calibration["records"],
            family,
        )

        projected = estimates(
            protected,
            family,
            beta,
        )

        fitted = estimates(
            calibration["records"],
            family,
            beta,
        )

        frozen["families"][family] = {
            "coefficients": {
                name: float(value)
                for name, value in zip(
                    FAMILIES[family],
                    beta,
                )
            },
            "calibration": model_stats(
                calibration["records"],
                fitted,
            ),
        }

        frozen[
            "protected_predictions"
        ][family] = {
            row["id"]: float(value)
            for row, value in zip(
                protected,
                projected,
            )
        }

        stats = frozen[
            "families"
        ][family]["calibration"]

        print(
            f"FREEZE {family:16s} "
            f"train_mse="
            f"{stats['mse']:.4f}"
        )

    save(
        frozen_path,
        frozen,
    )

    print(
        "FROZEN",
        frozen_path,
        "SHA256",
        digest(frozen_path),
    )

    print(
        "Commit calibration and frozen "
        "predictions BEFORE --phase test."
    )


def summarize(model_name):
    manifest, design_hash = (
        manifest_data()
    )

    frozen = json.loads(
        output_path(
            model_name,
            "frozen",
        ).read_text()
    )

    protected = json.loads(
        output_path(
            model_name,
            "protected",
        ).read_text()
    )

    if (
        not protected["complete"]
        or protected["manifest_sha256"]
        != design_hash
        or frozen["manifest_sha256"]
        != design_hash
        or frozen["calibration_sha256"]
        != digest(
            output_path(
                model_name,
                "calibration",
            )
        )
        or frozen["script_sha256"]
        != digest(SCRIPT)
    ):
        raise RuntimeError(
            "Result integrity failed"
        )

    rows = protected["records"]

    check_rows(
        rows,
        manifest["cases"]["protected"],
    )

    groups = (
        (
            "NEW LEXICAL GROUPS / FAMILIAR STRUCTURES",
            [
                r for r in rows
                if r["construction"]
                != "passive_cleft"
            ],
        ),
        (
            "PRIMARY: UNSEEN PASSIVE-CLEFT COMPOSITION",
            [
                r for r in rows
                if r["construction"]
                == "passive_cleft"
            ],
        ),
    )

    for label, group in groups:
        print(
            "\n",
            label,
            "n=",
            len(group),
        )

        stats = {}

        for family in FAMILIES:
            predicted = np.array(
                [
                    frozen[
                        "protected_predictions"
                    ][family][r["id"]]
                    for r in group
                ]
            )

            stats[family] = model_stats(
                group,
                predicted,
            )

            result = stats[family]

            print(
                f" {family:16s} "
                f"MSE={result['mse']:9.4f} "
                f"sign_matches="
                f"{result['error_sign_matches']}"
                f"/{result['n']} "
                f"predicted_wrong="
                f"{result['predicted_wrong']} "
                f"observed_wrong="
                f"{result['observed_wrong']}"
            )

        if "PRIMARY" in label:
            baseline = min(
                BASELINES,
                key=lambda name: (
                    stats[name]["mse"]
                ),
            )

            baseline_mse = (
                stats[baseline]["mse"]
            )

            primary_mse = (
                stats[PRIMARY]["mse"]
            )

            improvement = (
                1.0
                - primary_mse / baseline_mse
                if baseline_mse > 0
                else float("-inf")
            )

            group_wins = 0

            for pair in manifest[
                "selection"
            ]["protected"]:
                subset = [
                    r for r in group
                    if r["pair"] == pair[:2]
                ]

                if not subset:
                    raise RuntimeError(
                        "Missing protected lexical group"
                    )

                p = np.array(
                    [
                        frozen[
                            "protected_predictions"
                        ][PRIMARY][r["id"]]
                        for r in subset
                    ]
                )

                q = np.array(
                    [
                        frozen[
                            "protected_predictions"
                        ][baseline][r["id"]]
                        for r in subset
                    ]
                )

                primary_error = model_stats(
                    subset,
                    p,
                )["mse"]

                baseline_error = model_stats(
                    subset,
                    q,
                )["mse"]

                if primary_error < baseline_error:
                    group_wins += 1

            passed = (
                improvement >= 0.10
                and group_wins >= 4
            )

            print(
                f" BEST BASELINE: {baseline}, "
                f"primary gain="
                f"{100 * improvement:+.2f}%, "
                f"lexical wins={group_wins}/6"
            )

            print(
                " PRESPECIFIED PRIMARY GATE:",
                "PASS" if passed else "FAIL",
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
        audit()

    elif args.phase == "calibrate":
        infer(
            args.model,
            "calibration",
        )

    elif args.phase == "freeze":
        freeze(args.model)

    elif args.phase == "test":
        infer(
            args.model,
            "protected",
        )
        summarize(args.model)

    else:
        summarize(args.model)


if __name__ == "__main__":
    main()
