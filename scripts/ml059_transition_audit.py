
"""ML-059: Audit of apparently shared developmental transitions.

Uses ML-057 trajectories only. No training or protected evaluation.

Tests:
1. Same learning stage versus same training step.
2. Similarity after removing additive name-pair and construction effects.
3. Exact name-pair block permutation control.

Exploratory: two seeds and four calibration name pairs.
"""

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


STAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def mean(xs):
    return sum(xs) / len(xs)


def corr(x, y):
    x, y = list(x), list(y)

    if len(x) != len(y) or len(x) < 3:
        return None

    mx, my = mean(x), mean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]

    denom = math.sqrt(
        sum(v * v for v in dx) *
        sum(v * v for v in dy)
    )

    if denom <= 1e-12:
        return None

    return sum(a * b for a, b in zip(dx, dy)) / denom


def pair(key):
    return "|".join(key.split("|")[:2])


def construction(key):
    return "|".join(key.split("|")[2:])


def cases(snapshot):
    rows = snapshot["calibration"]["cases"]

    values = {
        row["case"]: float(row["signed_margin"])
        for row in rows
    }

    assert len(values) == len(rows), "Duplicate case keys"

    return values


def interpolate(history, target, coordinate):
    """Interpolate behavior, not neural activations."""

    if coordinate == "step":
        get_coordinate = lambda h: h["step"]
    else:
        get_coordinate = lambda h: (
            h["train"]["mean_signed_margin"]
        )

    for before, after in zip(history, history[1:]):
        left = get_coordinate(before)
        right = get_coordinate(after)

        if left <= target <= right and right > left:
            alpha = (target - left) / (right - left)

            x = cases(before)
            y = cases(after)

            assert x.keys() == y.keys()

            return {
                "step": (
                    before["step"] +
                    alpha * (after["step"] - before["step"])
                ),
                "values": {
                    key: x[key] + alpha * (y[key] - x[key])
                    for key in x
                },
            }

    return None


def two_way_residual(values):
    """Remove additive name-pair and construction effects."""

    by_pair = defaultdict(list)
    by_construction = defaultdict(list)

    for key, value in values.items():
        by_pair[pair(key)].append(value)
        by_construction[construction(key)].append(value)

    pair_mean = {
        k: mean(v) for k, v in by_pair.items()
    }

    construction_mean = {
        k: mean(v) for k, v in by_construction.items()
    }

    grand_mean = mean(list(values.values()))

    return {
        key: (
            value
            - pair_mean[pair(key)]
            - construction_mean[construction(key)]
            + grand_mean
        )
        for key, value in values.items()
    }


def compare(x, y, keys, permutations=False):
    raw = corr(
        [x[k] for k in keys],
        [y[k] for k in keys],
    )

    rx = two_way_residual(x)
    ry = two_way_residual(y)

    adjusted = corr(
        [rx[k] for k in keys],
        [ry[k] for k in keys],
    )

    result = {
        "raw_r": raw,
        "pair_x_construction_residual_r": adjusted,
    }

    if permutations and adjusted is not None:
        pairs = sorted({pair(k) for k in keys})

        assert len(pairs) == 4

        null_values = []

        for permutation in itertools.permutations(pairs):
            mapping = dict(zip(pairs, permutation))

            shuffled_y = [
                ry[
                    mapping[pair(key)] +
                    "|" +
                    construction(key)
                ]
                for key in keys
            ]

            value = corr(
                [rx[k] for k in keys],
                shuffled_y,
            )

            assert value is not None
            null_values.append(value)

        result["pair_block_null_count"] = len(null_values)

        result["pair_block_permutation_tail_fraction"] = (
            sum(
                value >= adjusted - 1e-12
                for value in null_values
            ) / len(null_values)
        )

        sorted_null = sorted(null_values)

        result["pair_block_95pct"] = sorted_null[
            int(0.95 * (len(sorted_null) - 1))
        ]

    return result


def run(root):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    expected_hash = manifest.pop("sha256")

    actual_hash = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    assert actual_hash == expected_hash, (
        "Manifest hash mismatch"
    )

    keys = sorted(manifest["calibration_case_keys"])

    assert len(keys) == 128
    assert len(set(keys)) == 128
    assert len({pair(k) for k in keys}) == 4
    assert len({construction(k) for k in keys}) == 32

    history = {}

    for seed in (7, 17):
        path = (
            root /
            f"seed_{seed}" /
            "trajectory.json"
        )

        report = json.loads(path.read_text())

        assert report["seed"] == seed
        assert report["split_hash"] == expected_hash

        snapshots = report["history"]

        assert [
            row["step"] for row in snapshots
        ] == manifest["checkpoints"]

        for snapshot in snapshots:
            assert sorted(cases(snapshot)) == keys

        history[seed] = snapshots

    results = []

    for stage in STAGES:
        row = {"train_margin": stage}

        for anchor, other in ((7, 17), (17, 7)):
            a = interpolate(
                history[anchor],
                stage,
                "train",
            )

            b = interpolate(
                history[other],
                stage,
                "train",
            )

            if a is None or b is None:
                row[f"anchor_{anchor}"] = {
                    "available": False
                }
                continue

            b_same_step = interpolate(
                history[other],
                a["step"],
                "step",
            )

            if b_same_step is None:
                row[f"anchor_{anchor}"] = {
                    "available": False
                }
                continue

            aligned = compare(
                a["values"],
                b["values"],
                keys,
                permutations=True,
            )

            same_step = compare(
                a["values"],
                b_same_step["values"],
                keys,
            )

            aligned_r = aligned[
                "pair_x_construction_residual_r"
            ]

            same_step_r = same_step[
                "pair_x_construction_residual_r"
            ]

            gain = (
                aligned_r - same_step_r
                if aligned_r is not None
                and same_step_r is not None
                else None
            )

            row[f"anchor_{anchor}"] = {
                "available": True,
                "anchor_interpolated_step": round(
                    a["step"], 3
                ),
                "other_aligned_interpolated_step": round(
                    b["step"], 3
                ),
                "matched_stage": aligned,
                "matched_step": same_step,
                "aligned_minus_step_residual_r": gain,
            }

        results.append(row)

    return {
        "experiment": "ML-059",
        "manifest_sha256": expected_hash,
        "input": "ML-057 trajectories",
        "analyses": results,
        "interpretation_limit": (
            "Exploratory comparison of two initializations "
            "and four calibration name-pair clusters. "
            "Behavioral correlation does not establish "
            "shared neural algorithms or sudden transitions. "
            "Block permutations are descriptive diagnostics."
        ),
    }


def show(result):
    print()
    print(
        "ML-059 | matched-stage vs same-step | "
        "two-way residual correlation"
    )

    print(
        "stage anchor   stage_r  step_r    gain   block-tail"
    )

    def fmt(value):
        return (
            "n/a"
            if value is None
            else f"{value:+.3f}"
        )

    for row in result["analyses"]:
        for anchor in (7, 17):
            x = row[f"anchor_{anchor}"]

            if not x["available"]:
                print(
                    f"{row['train_margin']:>5.1f} "
                    f"{anchor:>6}  unavailable"
                )
                continue

            print(
                f"{row['train_margin']:>5.1f} "
                f"{anchor:>6} "
                f"{fmt(x['matched_stage']['pair_x_construction_residual_r']):>9} "
                f"{fmt(x['matched_step']['pair_x_construction_residual_r']):>8} "
                f"{fmt(x['aligned_minus_step_residual_r']):>7} "
                f"{x['matched_stage']['pair_block_permutation_tail_fraction']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--out",
        default="results/ml059_transition_audit.json",
    )

    args = parser.parse_args()

    result = run(Path(args.root))

    output = Path(args.out)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    show(result)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
