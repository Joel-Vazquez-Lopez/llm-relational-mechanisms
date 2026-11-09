
"""ML-086 post-hoc representation audit.

Reconstructs the original role/event subspace overlap and
factorial main-effect / interaction decomposition.

Uses frozen calibration data, frozen subspaces and saved checkpoints.
Reads existing protected results only to summarize them.
Never generates protected examples or runs protected inference.

This analysis is EXPLORATORY, not a preregistered ML-086 test.
"""

import json
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, "scripts")

import ml080_learning_curve as ml
import ml081_layerwise_event_role as m81
import ml086_role_subspace_necessity as m86


ROOT = Path("results/ml086_fresh_models")
REPORTS = Path("reports/ml086")
OUTPUT = REPORTS / "posthoc_audit.json"
NOTES = REPORTS / "interpretation.md"

torch.set_num_threads(min(4, torch.get_num_threads()))


def overlap(a, b):
    a = a.double()
    b = b.double()
    singular = torch.linalg.svdvals(a.T @ b)
    return float((singular ** 2).mean())


def top8(x):
    _, _, vh = torch.linalg.svd(
        x.double(), full_matrices=False
    )
    return vh[:8].T.contiguous()


def arm_mean(rows, arm, metric):
    return mean(
        x[metric] for x in rows if x["arm"] == arm
    )


def main():
    if OUTPUT.exists() or NOTES.exists():
        raise FileExistsError(
            "Audit outputs already exist; refusing overwrite."
        )

    # Reconstruct original ML-086 calibration data and verify
    # the frozen protocol against the unchanged source.
    vocab, tr, ca, held, base, cal, dose, grid = m86.setup()

    h = m86.frozen(
        REPORTS / "frozen_protocol.json",
        m86.protocol(vocab, tr, ca, held, grid),
        False,
    )

    # Verify frozen calibration files and their hashes.
    bank, calibration_manifest = m86.load_bases(
        REPORTS / "calibration_bases.pt",
        REPORTS / "calibration_manifest.json",
        h,
    )

    rows = []

    for seed in m86.SEEDS:
        for arm in ml.ARMS:
            key = f"{seed}/{arm}"

            checkpoint = (
                ROOT / f"seed_{seed}" / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint, seed, arm, h, "cpu"
            )

            states = {}

            with torch.inference_mode():
                for corner in ("00", "01", "10", "11"):
                    x, ro, _, _ = m86.enc(
                        grid[corner], "cpu"
                    )

                    _, traces = m81.forward_trace(
                        model, x, ro
                    )

                    states[corner] = (
                        traces[1]["attn"]
                        .detach()
                        .cpu()
                        .double()
                    )

            a, b, c, d = (
                states[k]
                for k in ("00", "01", "10", "11")
            )

            # Factorial decomposition.
            role_main = ((b - a) + (d - c)) / 2
            event_main = ((c - a) + (d - b)) / 2
            gamma = d - c - b + a

            role_main_basis = top8(role_main)
            event_main_basis = top8(event_main)
            gamma_basis = top8(gamma)

            raw_role = bank[key]["role"]
            raw_event = bank[key]["event"]

            raw_overlap = overlap(raw_role, raw_event)
            main_overlap = overlap(
                role_main_basis, event_main_basis
            )

            gamma_role = overlap(
                raw_role, gamma_basis
            )
            gamma_event = overlap(
                raw_event, gamma_basis
            )

            # Exact interaction-energy fractions for the
            # original uncentered difference matrices.
            gamma_energy = 0.5 * gamma.square().sum()

            role_total = (
                (b - a).square().sum()
                + (d - c).square().sum()
            )

            event_total = (
                (c - a).square().sum()
                + (d - b).square().sum()
            )

            role_energy = float(
                gamma_energy
                / role_total.clamp_min(1e-12)
            )

            event_energy = float(
                gamma_energy
                / event_total.clamp_min(1e-12)
            )

            record = {
                "seed": seed,
                "arm": arm,
                "raw_overlap": raw_overlap,
                "main_overlap": main_overlap,
                "gamma_role": gamma_role,
                "gamma_event": gamma_event,
                "role_interaction_energy": role_energy,
                "event_interaction_energy": event_energy,
            }

            rows.append(record)

            print(
                f"{key:<16} "
                f"raw={raw_overlap:.4f} "
                f"main={main_overlap:.4f} "
                f"gamma_role={gamma_role:.4f} "
                f"gamma_event={gamma_event:.4f}",
                flush=True,
            )

    metrics = (
        "raw_overlap",
        "main_overlap",
        "gamma_role",
        "gamma_event",
        "role_interaction_energy",
        "event_interaction_energy",
    )

    arm_summary = {
        arm: {
            metric: arm_mean(rows, arm, metric)
            for metric in metrics
        }
        for arm in ml.ARMS
    }

    # Read, but DO NOT rerun, the existing protected evaluation.
    protected = json.loads(
        (REPORTS / "protected_evaluation.json").read_text()
    )

    assert protected["protocol_sha256"] == h

    control_summary = {}

    for arm in ml.ARMS:
        control_summary[arm] = {}

        for construction in ("target", "familiar"):
            run_rows = [
                r["patches"][construction]
                for r in protected["runs"]
                if r["arm"] == arm
            ]

            assert len(run_rows) == 8

            control_summary[arm][construction] = {
                metric: mean(r[metric] for r in run_rows)
                for metric in (
                    "native", "role", "event", "random",
                    "native_full", "role_full",
                    "event_full", "random_full",
                    "role_delta_norm", "event_delta_norm",
                )
            }

    report = {
        "experiment": "ML-086",
        "analysis": "post-hoc factorial subspace audit",
        "status": "EXPLORATORY_NOT_PREREGISTERED",
        "protocol_sha256": h,
        "calibration_manifest_sha256":
            calibration_manifest["sha256"],
        "protected_results_recomputed": False,
        "protected_questions_generated": False,
        "site": "L2 attention final readout",
        "rank": 8,
        "independent_random_overlap_expectation": 8 / 96,
        "models": rows,
        "arm_summary": arm_summary,
        "existing_protected_summary": protected["summary"],
        "existing_control_summary": control_summary,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    familiar = arm_summary["familiar"]
    target = arm_summary["target"]

    def fmt(key):
        return (
            f"{familiar[key]:.4f} | "
            f"{target[key]:.4f}"
        )

    primary = protected["summary"]
    ci = primary["primary_ci"]

    notes = f"""# ML-086 — Post-hoc representation audit

**Status:** Exploratory analysis performed after the
ML-086 protected evaluation. Not a predeclared primary test.

## Frozen experiment

- Protocol hash: `{h}`
- Intervention site: layer 2 attention, final answer token
- Subspace rank: 8
- Eight paired seeds, two training conditions
- Original protected results remain unchanged

## Original prospective results

- Mean behavioral exposure benefit:
  {primary["behavioral_mean"]:.4f}
- Positive behavioral pairs:
  {primary["behavioral_positive_pairs"]}/8
- Mean primary mechanistic contrast:
  {primary["primary_mean"]:.4f}
- Positive mechanistic comparisons:
  {primary["primary_positive_seeds"]}/8
- Bootstrap 95% interval:
  [{ci[0]:.4f}, {ci[1]:.4f}]
- Frozen primary success rule:
  {primary["primary_rule_met"]}

The mechanistic rule was met narrowly, but its bootstrap
interval includes zero. This is suggestive rather than
conclusive evidence of selective dependence.

## Calibration-only geometry

| Measurement | Familiar | Target |
|---|---:|---:|
| Raw role/event overlap | {fmt("raw_overlap")} |
| Main-effect overlap | {fmt("main_overlap")} |
| Interaction/role-basis overlap | {fmt("gamma_role")} |
| Interaction/event-basis overlap | {fmt("gamma_event")} |
| Role-difference interaction energy | {fmt("role_interaction_energy")} |
| Event-difference interaction energy | {fmt("event_interaction_energy")} |

Expected overlap of independent random 8D subspaces
in 96D: {8 / 96:.4f}.

## Interpretation

1. Role and event subspaces overlap strongly.
2. Separating factorial main effects reduces the measured
   overlap, but substantial alignment remains.
3. The factorial interaction contributes considerable
   activation-difference energy in many models.
4. The pattern is heterogeneous; it is not a universal
   large-interaction mechanism.
5. Role- and event-subspace erasures show similar
   behavioral effects, including on familiar grammar.
6. The intervention is therefore not yet uniquely
   attributable to role binding, nor is a specific
   mediation pathway identified.

The energy fractions describe squared activation
differences, not explained behavioral variance.

The factorial interaction also need not uniquely encode
binding: nonlinear responses to question cues can
produce the same mathematical structure.

## Reproducibility

Run:

    python3 scripts/ml086_posthoc_audit.py

Uses only frozen familiar calibration data and existing
checkpoints to recompute geometry.

The existing protected JSON is read for aggregation only;
no protected questions or model inference are generated.

Machine-readable results:
`reports/ml086/posthoc_audit.json`

Frozen prospective results:
`reports/ml086/protected_evaluation.json`

## Next hypothesis

A future independent cohort should compare the causal
effects of interaction-associated directions with
main-effect and matched-random directions.

Such an experiment must be frozen independently before
evaluating unfamiliar grammar.
"""

    NOTES.write_text(notes)

    print("\nARM MEANS")
    print(json.dumps(arm_summary, indent=2))
    print("\nSaved:", OUTPUT)
    print("Saved:", NOTES)
    print("No protected evaluation performed.")


if __name__ == "__main__":
    main()
