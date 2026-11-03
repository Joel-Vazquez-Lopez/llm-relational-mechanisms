
"""ML-051: retrospective cross-model factor-weight transfer.

Uses already frozen ML-050 coefficients and recorded candidate scores.
No new inference. No modification of protected data.

This is exploratory: both protected datasets were inspected before
this analysis was designed. A future fresh-data test is required for
confirmatory evidence.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ml050_factor_model import (
    FAMILIES,
    features,
    labels,
)

MODELS = ("qwen7b", "llama3b")
FAMILIES_TO_TEST = ("additive", "contextual")


def read_model(model):
    root = Path("results")

    paths = {
        phase: root / f"ml050_{model}_{phase}.json"
        for phase in ("calibration", "frozen", "protected")
    }

    data = {
        phase: json.loads(path.read_text())
        for phase, path in paths.items()
    }

    cal = data["calibration"]
    frozen = data["frozen"]
    protected = data["protected"]

    assert cal["complete"] and protected["complete"]
    assert len(cal["records"]) == 48
    assert len(protected["records"]) == 48

    assert cal["hash"] == frozen["hash"]
    assert cal["hash"] == protected["hash"]

    actual_hash = hashlib.sha256(
        paths["calibration"].read_bytes()
    ).hexdigest()

    assert frozen["calibration_sha256"] == actual_hash

    assert [
        row["id"] for row in cal["records"]
    ] == [
        row["id"] for row in data["calibration"]["records"]
    ]

    return data


def coefficient_vector(frozen, family):
    weights = frozen["coefficients"][family]
    names = FAMILIES[family]

    return np.array(
        [weights[name] for name in names],
        dtype=np.float64,
    )


def calibrate_scale(source_weights, calibration, names):
    """Fit only a scalar on recipient calibration records."""

    x = np.concatenate([
        features(row, names) @ source_weights
        for row in calibration
    ])

    y = np.concatenate([
        labels(row)
        for row in calibration
    ])

    denominator = float(x @ x)

    if denominator < 1e-12:
        return 0.0

    return max(
        0.0,
        float((x @ y) / denominator),
    )


def evaluate(records, family, frozen, transfer_weights):
    names = FAMILIES[family]
    native_preds = frozen["protected_predictions"][family]

    groups = {
        "all": [],
        "equal_tokens": [],
        "unequal_tokens": [],
    }

    for row in records:
        candidates = row["candidates"]
        observed = labels(row)

        native = np.array([
            native_preds[row["id"]][
                "candidate_centered_scores"
            ][candidate]
            for candidate in candidates
        ])

        transferred = (
            features(row, names) @ transfer_weights
        )

        observed_choice = int(np.argmax(observed))
        observed_correct = (
            candidates[observed_choice] == row["gold"]
        )

        result = {
            "native_mse": float(
                np.mean((native - observed) ** 2)
            ),
            "transfer_mse": float(
                np.mean((transferred - observed) ** 2)
            ),
            "native_match": int(
                np.argmax(native) == observed_choice
            ),
            "transfer_match": int(
                np.argmax(transferred) == observed_choice
            ),
            "observed_correct": observed_correct,
        }

        equal = (
            len(set(row["token_counts"].values())) == 1
        )

        groups["all"].append(result)

        groups[
            "equal_tokens" if equal else "unequal_tokens"
        ].append(result)

    return groups


def display_group(name, rows):
    native_mse = float(np.mean([
        r["native_mse"] for r in rows
    ]))

    transfer_mse = float(np.mean([
        r["transfer_mse"] for r in rows
    ]))

    native_match = sum(
        r["native_match"] for r in rows
    )

    transfer_match = sum(
        r["transfer_match"] for r in rows
    )

    ratio = (
        transfer_mse / native_mse
        if native_mse > 0 else float("inf")
    )

    wrong = [
        r for r in rows
        if not r["observed_correct"]
    ]

    native_wrong = sum(
        r["native_match"] for r in wrong
    )

    transfer_wrong = sum(
        r["transfer_match"] for r in wrong
    )

    print(
        f"  {name:14s} n={len(rows):2d} "
        f"native_MSE={native_mse:8.3f} "
        f"transfer_MSE={transfer_mse:8.3f} "
        f"transfer/native={ratio:6.3f}"
    )

    print(
        f"    choice matches: "
        f"native={native_match}/{len(rows)} "
        f"transfer={transfer_match}/{len(rows)}"
    )

    print(
        f"    error matches: "
        f"native={native_wrong}/{len(wrong)} "
        f"transfer={transfer_wrong}/{len(wrong)}"
    )


def main():
    data = {
        model: read_model(model)
        for model in MODELS
    }

    for family in FAMILIES_TO_TEST:
        names = FAMILIES[family]

        print("\n" + "=" * 76)
        print(f"FACTOR FAMILY: {family.upper()}")
        print("=" * 76)

        for target in MODELS:
            source = next(
                m for m in MODELS
                if m != target
            )

            donor_weights = coefficient_vector(
                data[source]["frozen"],
                family,
            )

            scale = calibrate_scale(
                donor_weights,
                data[target]["calibration"]["records"],
                names,
            )

            transferred = donor_weights * scale

            native_weights = coefficient_vector(
                data[target]["frozen"],
                family,
            )

            print(
                f"\n{source.upper()} --> {target.upper()}"
            )
            print(
                f"Recipient-calibrated donor scale: "
                f"{scale:.4f}"
            )

            print("\nFactor coefficients:")
            for name, native, donor in zip(
                names,
                native_weights,
                transferred,
            ):
                print(
                    f"  {name:24s} "
                    f"native={native:+9.3f} "
                    f"transferred={donor:+9.3f}"
                )

            groups = evaluate(
                data[target]["protected"]["records"],
                family,
                data[target]["frozen"],
                transferred,
            )

            print("\nProtected comparisons:")
            for group_name, rows in groups.items():
                display_group(group_name, rows)

    print("\nDIAGNOSTIC COMPLETE")
    print(
        "INTERPRETATION: exploratory retrospective analysis."
    )
    print(
        "A protected fresh-data replication is needed."
    )


if __name__ == "__main__":
    main()
