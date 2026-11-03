
"""ML-065: Transfer of ML-064 routing signatures.

Two explicit phases:
  freeze   - lock predictions from calibration results
  evaluate - test historically examined ML-037 holdout pairs

No new training or checkpoint modification.

The holdout is independent of ML-057 training/calibration, but
was exposed in earlier ML-037 research. Not pristine confirmation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    split_examples,
)
from ml057_developmental_checkpoints import (
    case_key,
    make_split,
)
from ml063_write_read_mediation import evaluate


SEEDS = (23, 29, 31, 37, 41, 43)
STEP = 300


def sha_json(obj):
    serialized = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def signature(result):
    sites = result["sites"]

    q = sites["readout"]["repairs"]["Q"][
        "mean_ce_rescued"
    ]

    v = sites["role_cue"]["repairs"]["V"][
        "mean_ce_rescued"
    ]

    return {
        "answer_Q_rescue": float(q),
        "role_V_rescue": float(v),
        "Q_minus_V": float(q - v),
        "readout_harm": float(
            sites["readout"]["mean_ce_increase"]
        ),
        "role_cue_harm": float(
            sites["role_cue"]["mean_ce_increase"]
        ),
    }


def direction(d, tolerance=0.2):
    if d > tolerance:
        return "Q"
    if d < -tolerance:
        return "V"
    return "indeterminate"


def frozen_protocol(root, calibration_path):
    manifest = json.loads(
        (root / "manifest.json").read_text()
    )

    claimed_hash = manifest.pop("sha256")

    assert sha_json(manifest) == claimed_hash, (
        "ML-057 manifest hash mismatch"
    )

    previous = json.loads(
        calibration_path.read_text()
    )

    assert previous["manifest_sha256"] == claimed_hash
    assert previous["source"] == "ML-057 calibration only"

    selected = [
        r for r in previous["runs"]
        if r["step"] == STEP
    ]

    assert len(selected) == len(SEEDS)
    assert {
        r["seed"] for r in selected
    } == set(SEEDS)

    predictions = {}

    for row in sorted(
        selected, key=lambda r: r["seed"]
    ):
        sig = signature(row)

        predictions[str(row["seed"])] = {
            "calibration_accuracy": row[
                "baseline_accuracy"
            ],
            "calibration_signature": sig,
            "prediction": direction(
                sig["Q_minus_V"]
            ),
        }

    return {
        "experiment": "ML-065",
        "source_manifest_hash": claimed_hash,
        "calibration_results_sha256": hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
        "seeds": list(SEEDS),
        "checkpoint_step": STEP,
        "prediction_rule": (
            "Answer-position Q rescue minus role-cue "
            "V rescue; threshold +/-0.20 CE units"
        ),
        "predictions": predictions,
        "evaluation": (
            "Historical ML-037 heldout pairs, not used "
            "in ML-057 training or calibration"
        ),
        "limitation": (
            "Historical exposure; not pristine "
            "independent confirmation. Site/scale "
            "confounds remain possible."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("freeze", "evaluate"),
        required=True,
    )

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--calibration-results",
        default="results/ml064_fresh_seed_mediation.json",
    )

    parser.add_argument(
        "--frozen",
        default="reports/ml065/frozen_predictions.json",
    )

    parser.add_argument(
        "--out",
        default="results/ml065_transfer",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    frozen_path = Path(args.frozen)
    frozen_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    expected = frozen_protocol(
        root,
        Path(args.calibration_results),
    )

    if frozen_path.exists():
        assert (
            json.loads(frozen_path.read_text())
            == expected
        ), "Frozen predictions have changed"

    elif args.phase == "freeze":
        frozen_path.write_text(
            json.dumps(expected, indent=2) + "\n"
        )

    else:
        raise RuntimeError(
            "Freeze predictions before evaluating."
        )

    if args.phase == "freeze":
        print(
            "ML-065 frozen predictions "
            "(no holdout evaluation):"
        )

        for seed, row in expected[
            "predictions"
        ].items():
            print(
                f"seed={seed} "
                f"prediction={row['prediction']} "
                f"D={row['calibration_signature']['Q_minus_V']:+.3f}"
            )

        print("Saved:", frozen_path)
        return

    result_path = output / "transfer_results.json"

    if result_path.exists():
        raise RuntimeError(
            "Results already exist. "
            "Refusing to overwrite the evaluation."
        )

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    device = args.device

    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    if device == "cuda":
        assert torch.cuda.is_available()

    if device == "mps":
        assert torch.backends.mps.is_available()

    train, calibration, _ = make_split()

    _, historical_holdout, heldout_pairs = (
        split_examples()
    )

    assert len(historical_holdout) == 224
    assert len(heldout_pairs) == 7

    train_keys = set(map(case_key, train))
    cal_keys = set(map(case_key, calibration))
    test_keys = set(
        map(case_key, historical_holdout)
    )

    assert not (train_keys & test_keys)
    assert not (cal_keys & test_keys)
    assert len(test_keys) == len(
        historical_holdout
    )

    report = {
        "experiment": "ML-065",
        "frozen_protocol_sha256": sha_json(
            expected
        ),
        "holdout_keys_sha256": sha_json(
            sorted(test_keys)
        ),
        "holdout_pairs": len(heldout_pairs),
        "holdout_questions": len(
            historical_holdout
        ),
        "status": (
            "Historical-exposure transfer check; "
            "not independent confirmation"
        ),
        "runs": [],
    }

    with torch.inference_mode():
        for seed in SEEDS:
            path = (
                root
                / f"seed_{seed}"
                / f"checkpoint_{STEP:04d}.pt"
            )

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            assert checkpoint["seed"] == seed
            assert checkpoint["step"] == STEP

            assert (
                checkpoint["split_hash"]
                == expected["source_manifest_hash"]
            )

            model = ScratchTransformer(
                width=96,
                heads=4,
                layers=2,
            ).to(device)

            model.load_state_dict(
                checkpoint["model"],
                strict=True,
            )

            tested = evaluate(
                model,
                historical_holdout,
                device,
            )

            observed = signature(tested)

            predicted_direction = expected[
                "predictions"
            ][str(seed)]["prediction"]

            observed_direction = direction(
                observed["Q_minus_V"]
            )

            row = {
                "seed": seed,
                "predicted_direction": (
                    predicted_direction
                ),
                "observed_direction": (
                    observed_direction
                ),
                "direction_matches": (
                    observed_direction
                    == predicted_direction
                ),
                "holdout_accuracy": tested[
                    "baseline_accuracy"
                ],
                "holdout_signature": observed,
                "details": tested,
            }

            report["runs"].append(row)

            print(
                f"seed={seed} "
                f"accuracy={tested['baseline_accuracy']:.3f} "
                f"pred={predicted_direction} "
                f"observed={observed_direction} "
                f"Q={observed['answer_Q_rescue']:+.3f} "
                f"V={observed['role_V_rescue']:+.3f}",
                flush=True,
            )

    report["direction_match_count"] = sum(
        r["direction_matches"]
        for r in report["runs"]
    )

    result_path.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(
        "\nDirection matches:",
        report["direction_match_count"],
        "/",
        len(SEEDS),
    )

    print("Saved:", result_path)


if __name__ == "__main__":
    main()
