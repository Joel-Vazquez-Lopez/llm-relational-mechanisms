
"""Reproduce and save ML-083 untrained event-role interaction geometry.

Run from repo root:
  python3 scripts/ml083_untrained_geometry.py

This script NEVER loads trained checkpoints, generates protected ML-080
cases, or changes a frozen prior experiment. All data are ML-080
familiar calibration.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch
import ml080_learning_curve as ml
import ml081_layerwise_event_role as m81
import ml083_compositional_layer_patching as m83

SITES = (
    (0, "attn"),
    (1, "attn"),
    (2, "attn"),
    (2, "mlp"),
)

OUT = Path("reports/ml083/untrained_geometry.json")


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Refusing overwrite: {OUT}"
        )

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    doc = m81.verify_ml080(
        Path("reports/ml080/frozen_protocol.json")
    )

    grid = m83.sentence_grid(doc)

    # Revalidate the frozen ML-083 experiment.
    m83.freeze(
        Path("reports/ml083/frozen_protocol.json"),
        doc,
        grid,
        False,
    )

    prepared = {}
    for corner in ("00", "01", "10", "11"):
        x, r, _, _ = ml.encode(grid[corner])
        prepared[corner] = x, r

    samples = []

    with torch.inference_mode():
        for seed in ml.SEEDS:
            torch.manual_seed(seed)

            # Initialization only. No checkpoint loaded.
            model = ml.Model().eval()

            traces = {}

            for corner, (x, r) in prepared.items():
                _, traces[corner] = m81.forward_trace(
                    model, x, r
                )

            for layer, site in SITES:
                a, b, c, d = [
                    traces[k][layer][site]
                    for k in ("00", "01", "10", "11")
                ]

                interaction = (
                    d - c - b + a
                ).norm(dim=1)

                joint = (d - a).norm(dim=1)

                relative = (
                    interaction
                    / joint.clamp_min(1e-12)
                )

                samples.append({
                    "seed": seed,
                    "layer": layer + 1,
                    "site": site,
                    "median_relative_interaction":
                        float(relative.median()),
                    "median_absolute_interaction":
                        float(interaction.median()),
                    "median_joint_change":
                        float(joint.median()),
                    "small_joint_rate":
                        float(
                            (joint < 1e-8).float().mean()
                        ),
                })

    summary = []

    print(
        "ML-083 UNTRAINED GEOMETRY — "
        "eight initialization seeds"
    )

    for layer, site in SITES:
        group = [
            r for r in samples
            if r["layer"] == layer + 1
            and r["site"] == site
        ]

        assert len(group) == 8

        item = {
            "layer": layer + 1,
            "site": site,
            "mean_seed_median_relative_interaction":
                mean(
                    r["median_relative_interaction"]
                    for r in group
                ),
            "mean_seed_median_absolute_interaction":
                mean(
                    r["median_absolute_interaction"]
                    for r in group
                ),
            "mean_seed_median_joint_change":
                mean(
                    r["median_joint_change"]
                    for r in group
                ),
        }

        summary.append(item)

        print(
            f"L{layer+1} {site:<6} "
            f"R={item['mean_seed_median_relative_interaction']:.4f} "
            f"abs={item['mean_seed_median_absolute_interaction']:.4f} "
            f"joint={item['mean_seed_median_joint_change']:.4f}"
        )

    payload = {
        "experiment": "ML-083 initialization control",
        "source_protocol_sha256": doc["sha256"],
        "ml083_source_sha256": sha(m83.__file__),
        "script_sha256": sha(__file__),
        "source":
            "active/active familiar CALIBRATION only; "
            "same 64 sentences as ML-083",
        "trained_models_loaded": False,
        "protected_cases_generated": False,
        "n_sentences": 64,
        "per_seed": samples,
        "summary": summary,
        "interpretation":
            "Ratios are descriptive; denominator effects "
            "must be checked using absolute and joint norms. "
            "Untrained does not isolate grammar-specific learning.",
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    print("Saved", OUT)


if __name__ == "__main__":
    main()
