"""ML-090-CORR-001: retrospective correction of the cross-example logit indexing.

This is a SEPARATE VERSIONED REANALYSIS. Never edits ML-090 source, frozen
protocol, discovery_evaluation.json or any ML-088 model. No protected cases.

From repository root:
  python3 scripts/ml090_role_cue_path_v2_corrected.py --phase preflight
  python3 scripts/ml090_role_cue_path_v2_corrected.py --phase run --device auto

The run requires all original source/checkpoint and plan hashes to agree;
it refuses to overwrite an existing correction. Treat as retrospective only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch
import ml090_role_cue_path as old

OUTPUT = Path("reports/ml090/correction_001_reanalysis.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rowwise_role_evidence(logits, cases):
    """One paired, agent-question-vs-patient-question contrast per sentence."""
    assert logits.ndim == 2 and logits.shape[0] == len(cases)
    assert len(cases) % 2 == 0
    ta, tp, _, _ = old.semantic_ids(cases)
    ix0 = torch.arange(0, len(cases), 2, dtype=torch.long)
    ix1 = ix0 + 1
    assert torch.equal(ta[ix0], ta[ix1])
    assert torch.equal(tp[ix0], tp[ix1])
    assert all(cases[i].role == "agent" and cases[i + 1].role == "patient"
               for i in range(0, len(cases), 2))
    return .5 * (
        (logits[ix0, ta[ix0]] - logits[ix0, tp[ix0]])
        - (logits[ix1, ta[ix1]] - logits[ix1, tp[ix1]])
    )


def preflight():
    # Uses no model checkpoint, and does not initialize the historical generator.
    x = torch.zeros(4, 10)
    x[0, 1], x[0, 2], x[0, 3] = 6, 1, 50
    x[1, 1], x[1, 2] = 1, 6
    x[2, 3], x[2, 4], x[2, 1] = 7, 0, 100
    x[3, 3], x[3, 4] = 0, 7
    a = torch.tensor([1, 1, 3, 3])
    p = torch.tensor([2, 2, 4, 4])
    legacy = x[::2, a[::2]]
    ix0 = torch.tensor([0, 2])
    ix1 = ix0 + 1
    corrected = .5 * ((x[ix0, a[ix0]] - x[ix0, p[ix0]])
                      - (x[ix1, a[ix1]] - x[ix1, p[ix1]]))
    assert legacy.shape == (2, 2), legacy.shape
    assert corrected.shape == (2,)
    assert torch.allclose(corrected, torch.tensor([5., 7.]))
    clone = x.clone()
    clone[1] = clone[0]
    clone[3] = clone[2]
    invariant = .5 * ((clone[ix0, a[ix0]] - clone[ix0, p[ix0]])
                      - (clone[ix1, a[ix1]] - clone[ix1, p[ix1]]))
    assert torch.count_nonzero(invariant) == 0
    print("PREFLIGHT PASS: original selection cross-indexes examples (2x2); "
          "corrected selection is paired (2,). No models or reports loaded.")


def forward_all(model, cases, device, key=None, start_layer=0):
    result = []
    for k in range(0, len(cases), old.BATCH):
        z, _, _ = old.forward(model, cases[k:k + old.BATCH],
                              device, key, start_layer)
        result.append(z)
    return torch.cat(result)


def corrected_summary(rows):
    def cell(seed, arm, construction):
        return next(x for x in rows
                    if (x["seed"], x["arm"], x["construction"])
                    == (seed, arm, construction))
    overview = {}
    for construction in old.CONSTRUCTIONS:
        for arm in old.ml.ARMS:
            subset = [cell(seed, arm, construction) for seed in old.SEEDS]
            sel = [r["interventions"]["who_cut_from_L2"]
                   - r["interventions"]["role_cut_from_L2"] for r in subset]
            overview[f"{arm}/{construction}"] = {
                "corrected_native_role_evidence_mean":
                    mean(r["native"] for r in subset),
                "corrected_who_minus_role_L2_by_seed":
                    dict(zip(map(str, old.SEEDS), sel)),
                "corrected_who_minus_role_L2_mean": mean(sel),
                "positive_seeds": sum(v > 0 for v in sel),
            }
    effects = []
    for seed in old.SEEDS:
        def selective(arm):
            r = cell(seed, arm, "second_passive")
            return (r["interventions"]["who_cut_from_L2"]
                    - r["interventions"]["role_cut_from_L2"])
        effects.append(selective("target") - selective("familiar"))
    return {
        "cells": overview,
        "corrected_exposure_effect_by_seed":
            dict(zip(map(str, old.SEEDS), effects)),
        "corrected_exposure_effect_mean": mean(effects),
        "corrected_exposure_effect_seed_bootstrap_95pct": old.bootstrap(effects),
        "corrected_exposure_effect_positive_seeds": sum(v > 0 for v in effects),
        "interpretation": "RETROSPECTIVE exploratory reanalysis; not a new "
                          "causal identification or independent confirmation.",
    }


@torch.inference_mode()
def run(device):
    if OUTPUT.exists():
        raise FileExistsError(f"Correction already exists; refusing overwrite: {OUTPUT}")
    if not old.OUTPUT.is_file() or not old.PLAN.is_file():
        raise FileNotFoundError("Original ML-090 plan/report must remain present")
    original = json.loads(old.OUTPUT.read_text())
    old_hash, cases_by_key = old.sample_cases()
    plan_hash = old.freeze(old.spec(old_hash, cases_by_key), create=False)
    assert original["experiment"] == "ML-090"
    assert original["plan_sha256"] == plan_hash
    assert original["ml088_protocol_sha256"] == old_hash
    assert original["protected_examples_generated"] is False
    assert len(original["rows"]) == len(old.SEEDS) * len(old.ml.ARMS) * len(old.CONSTRUCTIONS)
    gate, counts = old.m88.model_gate(old_hash)
    if not gate:
        raise PermissionError(f"Original ML-088 calibration competence gate failed: {counts}")

    rows = []
    for seed in old.SEEDS:
        for arm in old.ml.ARMS:
            model = old.ml.load_checkpoint(
                old.ROOT / f"seed_{seed}" / arm / "model_step_12800.pt",
                seed, arm, old_hash, device)
            for construction in old.CONSTRUCTIONS:
                cases = cases_by_key[construction]
                legacy = next(row for row in original["rows"]
                              if (row["seed"], row["arm"], row["construction"])
                              == (seed, arm, construction))
                native = forward_all(model, cases, device)
                legacy_native = old.scores(native, cases)
                # Confirm exact historical reproduction BEFORE issuing a correction.
                for key in ("choice_accuracy", "full_vocab_accuracy",
                            "mean_role_antisymmetric_evidence",
                            "mean_within_target_role_margin"):
                    delta = abs(legacy_native[key] - legacy["native"][key])
                    if delta > 1e-4:
                        raise AssertionError(("Original baseline parity failure",
                                              seed, arm, construction, key, delta))
                entry = {
                    "seed": seed, "arm": arm, "construction": construction,
                    "legacy_native_role_evidence":
                        legacy["native"]["mean_role_antisymmetric_evidence"],
                    "native": float(rowwise_role_evidence(native, cases).mean()),
                    "interventions": {},
                    "legacy_interventions": {},
                }
                for key in old.KEYS:
                    for layer in old.START_LAYERS:
                        name = f"{key}_cut_from_L{layer+1}"
                        patched = forward_all(model, cases, device, key, layer)
                        reproduced = old.scores(patched, cases)
                        stored = legacy["interventions"][name]
                        for metric in ("choice_accuracy",
                                       "mean_role_antisymmetric_evidence"):
                            delta = abs(reproduced[metric] - stored[metric])
                            if delta > 1e-4:
                                raise AssertionError(("Historical intervention parity failure",
                                                      seed, arm, construction, name,
                                                      metric, delta))
                        entry["interventions"][name] = float(
                            rowwise_role_evidence(patched, cases).mean())
                        entry["legacy_interventions"][name] = stored[
                            "mean_role_antisymmetric_evidence"]
                assert abs(entry["interventions"]["role_cut_from_L1"]) < 1e-4
                rows.append(entry)
            print(f"Validated historical parity and corrected role-evidence "
                  f"seed={seed} arm={arm}", flush=True)
    report = {
        "experiment": "ML-090-CORR-001",
        "status": "RETROSPECTIVE_CORRECTION_NOT_CONFIRMATORY",
        "original_report_sha256": sha(old.OUTPUT),
        "original_plan_sha256": sha(old.PLAN),
        "original_script_sha256": sha(old.__file__),
        "corrected_script_sha256": sha(__file__),
        "ml088_protocol_sha256": old_hash,
        "original_report_unchanged": True,
        "protected_cases_generated": False,
        "metric_replaced": "mean_role_antisymmetric_evidence",
        "derived_quantities_replaced": [
            "delta_role_evidence",
            "L2plus_role_vs_who_role_evidence_by_seed",
            "L2plus_role_vs_who_mean",
            "paired_exposure_effect_on_L2plus_role_specific_evidence",
            "its bootstrap confidence interval and positive-seed count",
        ],
        "rows": rows,
        "summary": corrected_summary(rows),
        "limitations": "Previously examined 16 ML-088 checkpoints and ML-090 "
                       "calibration cases; corrections require exact historical "
                       "parity; no new protected evaluation, claims of necessity, "
                       "or independent validation.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved correction only:", OUTPUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("preflight", "run"))
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "mps", "cuda"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.phase == "preflight":
        preflight()
    else:
        run(old.ml.select_device(args.device))


if __name__ == "__main__":
    main()
