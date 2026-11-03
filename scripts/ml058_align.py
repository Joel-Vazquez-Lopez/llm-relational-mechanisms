
"""ML-058: Are the learning transitions shared but time-shifted?

Read-only analysis of ML-057 JSON files.
No new training, causal patches, or protected evaluation.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

STAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def corr(x, y):
    if len(x) < 3:
        return None

    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    x = [v - mx for v in x]
    y = [v - my for v in y]

    denom = math.sqrt(
        sum(v * v for v in x) *
        sum(v * v for v in y)
    )

    if denom < 1e-10:
        return None

    return sum(a * b for a, b in zip(x, y)) / denom


def ranks(values):
    order = sorted(
        range(len(values)),
        key=lambda i: values[i]
    )
    result = [0.0] * len(values)

    i = 0
    while i < len(values):
        j = i + 1
        while (
            j < len(values)
            and values[order[j]] == values[order[i]]
        ):
            j += 1

        rank = (i + j - 1) / 2
        for k in order[i:j]:
            result[k] = rank
        i = j

    return result


def margins(snapshot):
    cases = snapshot["calibration"]["cases"]
    output = {
        row["case"]: float(row["signed_margin"])
        for row in cases
    }
    assert len(output) == len(cases)
    return output


def group(key):
    return tuple(key.split("|")[-2:])


def name_pair(key):
    return tuple(key.split("|")[:2])


def residualize(values, keys):
    groups = defaultdict(list)

    for key in keys:
        groups[group(key)].append(key)

    result = {}
    for members in groups.values():
        mean = sum(values[k] for k in members) / len(members)

        for k in members:
            result[k] = values[k] - mean

    return result


def at_stage(history, target, keys):
    """Interpolate calibration behavior at a TRAIN margin.

    This is behavioral interpolation, not a neural checkpoint.
    """

    for before, after in zip(history, history[1:]):
        left = before["train"]["mean_signed_margin"]
        right = after["train"]["mean_signed_margin"]

        if left <= target <= right and right > left:
            alpha = (target - left) / (right - left)

            x = margins(before)
            y = margins(after)

            values = {
                k: x[k] + alpha * (y[k] - x[k])
                for k in keys
            }

            return values, [
                before["step"],
                after["step"],
            ]

    return None, None


def shuffled_control(x, y, keys, rng):
    groups = defaultdict(list)

    for key in keys:
        groups[group(key)].append(key)

    shuffled = dict(y)

    for members in groups.values():
        values = [y[k] for k in members]
        rng.shuffle(values)

        for key, value in zip(members, values):
            shuffled[key] = value

    return corr(
        [x[k] for k in keys],
        [shuffled[k] for k in keys],
    )


def stable_acquisition(history, keys, threshold=0.5):
    """First checkpoint after which a margin stays positive
    and above the specified threshold.
    """

    snapshots = [margins(h) for h in history]
    acquired = {}

    for key in keys:
        times = [
            history[i]["step"]
            for i in range(len(history))
            if all(
                state[key] > threshold
                for state in snapshots[i:]
            )
        ]

        acquired[key] = times[0] if times else None

    return acquired


def run(root):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    expected_hash = manifest.pop("sha256")

    actual_hash = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True
        ).encode()
    ).hexdigest()

    assert actual_hash == expected_hash

    keys = sorted(manifest["calibration_case_keys"])
    assert len(keys) == len(set(keys)) == 128

    histories = []

    for seed in (7, 17):
        path = root / f"seed_{seed}" / "trajectory.json"
        report = json.loads(path.read_text())

        assert report["seed"] == seed
        assert report["split_hash"] == expected_hash

        history = report["history"]

        assert (
            [h["step"] for h in history]
            == manifest["checkpoints"]
        )

        for snapshot in history:
            assert sorted(margins(snapshot)) == keys

        histories.append(history)

    a, b = histories
    rng = random.Random(58058)

    stage_results = []

    for stage in STAGES:
        x, steps_a = at_stage(a, stage, keys)
        y, steps_b = at_stage(b, stage, keys)

        if x is None or y is None:
            stage_results.append({
                "train_margin": stage,
                "available": False,
            })
            continue

        x = residualize(x, keys)
        y = residualize(y, keys)

        observed = corr(
            [x[k] for k in keys],
            [y[k] for k in keys],
        )

        controls = [
            shuffled_control(x, y, keys, rng)
            for _ in range(1000)
        ]
        controls = sorted(
            v for v in controls if v is not None
        )

        leave_one_out = []

        for pair in sorted(
            set(name_pair(k) for k in keys)
        ):
            subset = [
                k for k in keys
                if name_pair(k) != pair
            ]

            value = corr(
                [x[k] for k in subset],
                [y[k] for k in subset],
            )

            if value is not None:
                leave_one_out.append(value)

        stage_results.append({
            "train_margin": stage,
            "available": True,
            "seed7_steps": steps_a,
            "seed17_steps": steps_b,
            "case_correlation": observed,
            "shuffle_95pct": (
                controls[int(.95 * (len(controls) - 1))]
                if controls else None
            ),
            "leave_one_pair_out_range": (
                [min(leave_one_out), max(leave_one_out)]
                if leave_one_out else None
            ),
        })

    first_a = stable_acquisition(a, keys)
    first_b = stable_acquisition(b, keys)

    common = [
        k for k in keys
        if first_a[k] is not None
        and first_b[k] is not None
    ]

    acquisition = {
        "threshold": 0.5,
        "seed7_acquired": sum(
            t is not None for t in first_a.values()
        ),
        "seed17_acquired": sum(
            t is not None for t in first_b.values()
        ),
        "acquired_by_both": len(common),
        "order_rank_correlation": (
            corr(
                ranks([first_a[k] for k in common]),
                ranks([first_b[k] for k in common]),
            ) if len(common) > 2 else None
        ),
    }

    return {
        "experiment": "ML-058",
        "manifest_sha256": expected_hash,
        "alignment_variable": "training mean signed margin",
        "stage_comparison": stage_results,
        "stable_acquisition": acquisition,
        "limitations": (
            "Exploratory: two seeds, four name-pair clusters, "
            "interpolated behavioral outputs. "
            "Similarity cannot prove shared neural algorithms "
            "or a discontinuous learning transition."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--out",
        default="results/ml058_alignment.json",
    )

    args = parser.parse_args()

    result = run(Path(args.root))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    def fmt(value):
        return "n/a" if value is None else f"{value:+.3f}"

    print("\nML-058: Developmental Alignment")
    print("Training stage | Correlation | Shuffle 95%")

    for row in result["stage_comparison"]:
        if not row["available"]:
            print(row["train_margin"], "| unavailable")
            continue

        print(
            f"{row['train_margin']:>5} | "
            f"{fmt(row['case_correlation'])} | "
            f"{fmt(row['shuffle_95pct'])} | "
            f"steps {row['seed7_steps']} vs "
            f"{row['seed17_steps']}"
        )

    print("\nStable acquisition:")
    print(result["stable_acquisition"])

    print("\nSaved:", out)


if __name__ == "__main__":
    main()
