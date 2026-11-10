"""ML-093: equal-magnitude source-state-transfer audit (EXPLORATORY).

Reuses *exactly* the frozen ML-092 discovery cases and ML-088 checkpoints.
Does not rerun ML-092, change its artifacts, train models, or access ML-088
protected evaluation cases. Do not interpret this as independent confirmation.

Research question: does the L1 aligned-active donor outperform the
role-reversed-active donor after their SOURCE RESIDUAL CHANGES have the same
per-example, per-participant Euclidean magnitude?

The common per-token magnitude is min(||delta_aligned||, ||delta_reversed||).
Thus neither direction is enlarged. Four independent, fixed spherical random
source directions are given that SAME magnitude on each token. All four
questions about a sentence receive the same pre-question source perturbation.

This audits magnitude versus direction; it does not isolate semantics from
source position, causal history, or grammatical-form differences.

Run from repository root:
  python3 scripts/ml093_equal_norm_source_audit.py --phase preflight
  python3 scripts/ml093_equal_norm_source_audit.py --phase plan
  # REVIEW + COMMIT this script + reports/ml093/frozen_discovery_plan.json
  python3 scripts/ml093_equal_norm_source_audit.py --phase run
  # REVIEW + COMMIT reports/ml093/discovery_evaluation.json
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ("familiar", "target")
SITES = (0,)  # L1 is the primary mechanistic site in ML-092; no layer search.
BATCH_TEMPLATES = 12
RANDOM_REPLICAS = 4
EPS = 1e-12
ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml093/frozen_discovery_plan.json")
REPORT = Path("reports/ml093/discovery_evaluation.json")
MODES = ("native", "original_aligned", "original_reversed",
         "equalnorm_aligned", "equalnorm_reversed") + tuple(
             f"equalnorm_random_{i}" for i in range(RANDOM_REPLICAS)
         )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def bootstrap(values, n=20000, seed=93093):
    assert len(values) == len(SEEDS)
    rng = random.Random(seed)
    dist = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                  for _ in range(n))
    return [dist[int(.025 * n)], dist[int(.975 * n) - 1]]


def second_accuracy(metrics):
    q = metrics["per_question_four_choice_accuracy"]
    return .5 * (q["second_agent"] + q["second_patient"])


def prepare():
    # m92.select_cases() initializes ML globals once; no other prepare call.
    old_hash, groups, ml091_hash = m92.select_cases()
    assert tuple(ml.SEEDS) == SEEDS
    assert tuple(ml.ARMS) == ARMS
    assert len(groups["passive"]) == len(groups["active"]) == 96
    ml092_spec = m92.specification(old_hash, groups, ml091_hash)
    previous_plan_hash = m92.freeze(ml092_spec, False)
    if not m92.REPORT.is_file():
        raise FileNotFoundError("ML-092 discovery report must be preserved first")
    previous = json.loads(m92.REPORT.read_text())
    assert previous["experiment"] == "ML-092"
    assert previous["plan_sha256"] == previous_plan_hash
    assert previous["ml088_protocol_sha256"] == old_hash
    assert previous["protected_cases_generated"] is False
    assert len(previous["rows"]) == len(SEEDS) * len(ARMS)
    return old_hash, groups, previous_plan_hash, previous


def specification(old_hash, groups, previous_plan_hash):
    return {
        "experiment": "ML-093",
        "status": "EXPLORATORY_MAGNITUDE_AUDIT_NO_NEW_MODELS_OR_CASES",
        "script_sha256": sha(__file__),
        "ml080_sha256": sha(ml.__file__),
        "ml088_sha256": sha(m88.__file__),
        "ml091_sha256": sha(m91.__file__),
        "ml092_sha256": sha(m92.__file__),
        "ml088_protocol_sha256": old_hash,
        "ml092_frozen_plan_sha256": previous_plan_hash,
        "ml092_report_file_sha256": sha(m92.REPORT),
        "cohort": "Same 16 frozen ML-088 step-12800 checkpoints, eight paired seeds",
        "cases": "Exactly ML-092's 96 already examined calibration sentence templates; no newly held-out cases",
        "passive_prompts_sha256": digest([
            ml.prompt(c) for group in groups["passive"] for c in group
        ]),
        "seeds": SEEDS,
        "arms": ARMS,
        "patch_site": "Post-L1 residual source-token positions, aligned by participant NAME, not answer query",
        "modes": MODES,
        "matched_donor": "Second-active source; same agent/patient semantics as passive receiver",
        "reversed_donor": "Second-active source; agent/patient assignments swapped, names unchanged",
        "delta": "Donor source state minus receiver source state; two participant positions before question",
        "normalization": "For each source example and each of its two participant tokens, common norm = min(aligned delta norm, reversed delta norm); equalnorm deltas preserve each original direction but never increase magnitude",
        "random_control": "Four deterministic isotropic unit directions in model-width space, seeded independently per paired seed and replicate, reused across exposure arms and four role/event queries for each sentence; scaled to identical common norms",
        "random_seed_formula": "93093000 + 1000 * model_seed + replicate_index; sample shape [96,2,width] on CPU",
        "sites_1indexed": [1],
        "primary_exploratory_contrast": "Within the familiar arm at L1, equalnorm_aligned minus equalnorm_reversed in second_event_role_gating, paired across all 8 seeds",
        "secondary": "Equalnorm aligned-minus-reversed second-question accuracy; each intervention minus native; direction controls versus equalnorm random mean; target arm; full-vocabulary, wrong-event/role, four-query all-correct, per-token patch norms and clipping ratios",
        "predeclared_controls": "Original aligned and reversed reproduce ML-092 descriptively; four common-norm random source edits; native exact no-op; all four queries per source share identical intervention; frozen ML-092 report checked without mutation",
        "scope": "Previously examined models AND cases, exploratory audit only, not confirmation; equal norms do not equate positions/semantic contents; patch transfers potential answer-relevant information",
        "protected_safety": "Do not access/generate ML-088 protected examples or rerun ML-092; do not repeat ML-093 after output exists",
    }


def freeze(document, create):
    canonical = json.loads(json.dumps({**document, "sha256": digest(document)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(canonical, indent=2) + "\n")
    if not PLAN.exists():
        raise FileNotFoundError("Run --phase plan after preflight")
    assert json.loads(PLAN.read_text()) == canonical, "ML-093 frozen script/inputs changed"
    return canonical["sha256"]


def get_vectors(model, passive_groups, aligned_trace, reversed_trace,
                device, layer, total_templates):
    """All four question versions share a fixed perturbation for their sentence."""
    cases = [q for g in passive_groups for q in g]
    ids, readout, _, _ = ml.encode(cases)
    ids = ids.to(device)
    readout = readout.to(device)
    b, n = ids.shape
    assert b == 4 * len(passive_groups)
    owner = torch.arange(b, device=device) // 4
    ix = torch.arange(b, device=device)
    original_receiver = [g[2] for g in passive_groups]
    pool = m92.donors(passive_groups)
    aligned_src, dst = m92.indexes(passive_groups, pool["aligned"], "aligned_both", device)
    reversed_src, reversed_dst = m92.indexes(passive_groups, pool["reversed"], "reversed_both", device)
    assert torch.equal(dst, reversed_dst)
    # Standard causal forward up to post-L1 source site.
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, device=device, dtype=torch.bool).triu(1)
    for j in range(layer + 1):
        x = model.blocks[j](x, mask)
    recv_state = x[ix[:, None], dst[owner]]
    a_state = aligned_trace[layer][owner][ix[:, None], aligned_src[owner]]
    r_state = reversed_trace[layer][owner][ix[:, None], reversed_src[owner]]
    da = a_state - recv_state
    dr = r_state - recv_state
    na = da.norm(dim=-1)
    nr = dr.norm(dim=-1)
    common = torch.minimum(na, nr)
    assert torch.isfinite(na).all() and torch.isfinite(nr).all()
    assert bool((common >= 0).all())
    # Causal source representations must be identical across four query variants.
    for j in range(1, 4):
        assert torch.allclose(da[::4], da[j::4], atol=2e-5, rtol=1e-5)
        assert torch.allclose(dr[::4], dr[j::4], atol=2e-5, rtol=1e-5)
    data = {
        "ids": ids, "readout": readout, "x": x, "mask": mask,
        "dst": dst[owner], "da": da, "dr": dr,
        "na": na, "nr": nr, "common": common,
    }
    return data


def random_unit_vectors(seed, replicate, total_templates, width, device):
    rng = torch.Generator(device="cpu").manual_seed(93093000 + 1000 * seed + replicate)
    noise = torch.randn((total_templates, 2, width), generator=rng)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return noise.to(device)


def intervention_logits(model, vectors, mode, random_noise=None, template_offset=0):
    """Run remainder of model from a precisely controlled L1 patch."""
    x = vectors["x"].clone()
    da, dr = vectors["da"], vectors["dr"]
    na, nr = vectors["na"], vectors["nr"]
    common = vectors["common"]
    if mode == "native":
        delta = torch.zeros_like(da)
    elif mode == "original_aligned":
        delta = da
    elif mode == "original_reversed":
        delta = dr
    elif mode == "equalnorm_aligned":
        delta = da * (common / na.clamp_min(EPS)).unsqueeze(-1)
    elif mode == "equalnorm_reversed":
        delta = dr * (common / nr.clamp_min(EPS)).unsqueeze(-1)
    elif mode.startswith("equalnorm_random_"):
        assert random_noise is not None
        n_templates = x.shape[0] // 4
        assert random_noise.shape[0] >= template_offset + n_templates
        owner = torch.arange(x.shape[0], device=x.device) // 4
        unit = random_noise[template_offset:template_offset + n_templates][owner]
        delta = unit * common.unsqueeze(-1)
    else:
        raise ValueError(mode)
    applied_norm = delta.norm(dim=-1)
    if mode.startswith("equalnorm_"):
        tol = 5e-5 + 2e-5 * common
        assert bool(((applied_norm - common).abs() <= tol).all()), mode
    if mode == "native":
        assert float(applied_norm.max()) == 0.0
    ix = torch.arange(x.shape[0], device=x.device)
    x[ix[:, None], vectors["dst"]] = x[ix[:, None], vectors["dst"]] + delta
    # The only remaining layers run normally. No downstream answer patch.
    for layer in range(1, len(model.blocks)):
        x = model.blocks[layer](x, vectors["mask"])
    output = model.out(model.norm(x[ix, vectors["readout"]])).float().cpu()
    return output, {
        "applied_norm_mean": float(applied_norm.mean()),
        "applied_norm_max": float(applied_norm.max()),
    }


def native_model_logits(model, groups, device):
    return m92.native_logits(model, groups, device)


def check_previous(report, seed, arm, native, outputs):
    prior = next(r for r in report["rows"] if r["seed"] == seed and r["arm"] == arm)
    checks = [
        ("native", native, prior["native_passive"]),
        ("original_aligned", outputs["original_aligned"],
         prior["interventions"]["L1/aligned_both"]["metrics"]),
        ("original_reversed", outputs["original_reversed"],
         prior["interventions"]["L1/reversed_both"]["metrics"]),
    ]
    deltas = {}
    for name, fresh, old in checks:
        d = {
            "second_event_role_gating": fresh["second_event_role_gating"] - old["second_event_role_gating"],
            "four_choice_accuracy": fresh["four_choice_accuracy"] - old["four_choice_accuracy"],
        }
        deltas[name] = d
        # Should reproduce exactly when using same checkpoints and sentences.
        assert abs(d["second_event_role_gating"]) < .04, (seed, arm, name, d)
        assert abs(d["four_choice_accuracy"]) < .012, (seed, arm, name, d)
    return deltas


@torch.inference_mode()
def preflight(groups):
    torch.manual_seed(93093)
    model = ml.Model().eval()
    sub = groups["passive"][:4]
    pool = m92.donors(sub)
    aligned = m92.trace(model, pool["aligned"], "cpu")
    reverse = m92.trace(model, pool["reversed"], "cpu")
    a = get_vectors(model, sub, aligned, reverse, "cpu", 0, len(groups["passive"]))
    direct = native_model_logits(model, sub, "cpu")
    matched = {}
    for mode in MODES:
        random = None
        if mode.startswith("equalnorm_random_"):
            rep = int(mode.rsplit("_", 1)[1])
            random = random_unit_vectors(SEEDS[0], rep, 96,
                                         a["da"].shape[-1], "cpu")
        z, stat = intervention_logits(model, a, mode, random)
        assert torch.isfinite(z).all() and stat["applied_norm_mean"] >= 0
        matched[mode] = z
    err = float((matched["native"] - direct).abs().max())
    assert err < 1e-4, err
    # The unnormalized ML-093 patches must exactly reproduce the original
    # ML-092 implementation, independent of our new equal-norm machinery.
    ref_aligned, _ = m92.patched_logits(model, sub, pool["aligned"], aligned,
                                       "cpu", 0, "aligned_both")
    ref_reversed, _ = m92.patched_logits(model, sub, pool["reversed"], reverse,
                                        "cpu", 0, "reversed_both")
    orig_a_err = float((matched["original_aligned"] - ref_aligned).abs().max())
    orig_r_err = float((matched["original_reversed"] - ref_reversed).abs().max())
    assert orig_a_err < 1e-4, orig_a_err
    assert orig_r_err < 1e-4, orig_r_err
    assert float((matched["original_aligned"] - direct).abs().max()) > 1e-6
    # Numeric construct validation: zero/no-op, equal source-token norm.
    for mode in ("equalnorm_aligned", "equalnorm_reversed", "equalnorm_random_0"):
        assert not torch.isnan(matched[mode]).any()
    assert abs(m91.analyze_logits(matched["native"], sub)["four_choice_accuracy"] -
               m91.analyze_logits(direct, sub)["four_choice_accuracy"]) < 1e-7
    # Per-example cap must prevent either direction being enlarged.
    assert bool((a["common"] <= a["na"] + 1e-5).all())
    assert bool((a["common"] <= a["nr"] + 1e-5).all())
    # Original source deltas are invariant across event/role query variants.
    assert torch.allclose(a["da"][::4], a["da"][1::4], atol=2e-5)
    # Fixed random control: repeat yields bitwise-identical logits on same CPU model.
    rand = random_unit_vectors(SEEDS[0], 0, 96, a["da"].shape[-1], "cpu")
    again, _ = intervention_logits(model, a, "equalnorm_random_0", rand)
    assert torch.equal(again, matched["equalnorm_random_0"])
    print("PREFLIGHT PASS: original checkpoint interface, four-query cases, same-example source invariance")
    print(f"Native parity={err:.2e}; old-patch parity aligned={orig_a_err:.2e} "
          f"reversed={orig_r_err:.2e}; exact matched-norm + random controls pass")
    print("No trained checkpoints loaded, no protected examples accessed, no discovery results generated.")


@torch.inference_mode()
def run(old_hash, plan_hash, groups, previous, device):
    if REPORT.exists():
        raise FileExistsError(f"ML-093 already evaluated: {REPORT}; do not rerun")
    gate, counts = m88.model_gate(old_hash)
    if not gate:
        raise PermissionError(f"ML-088 competence gate failed: {counts}")
    results = []
    for seed in SEEDS:
        random_vectors = {
            i: random_unit_vectors(seed, i, len(groups["passive"]), ml.WIDTH, device)
            for i in range(RANDOM_REPLICAS)
        }
        for arm in ARMS:
            model = ml.load_checkpoint(
                ROOT / f"seed_{seed}" / arm / "model_step_12800.pt",
                seed, arm, old_hash, device,
            )
            output_chunks = {mode: [] for mode in MODES}
            norm_chunks = {mode: [] for mode in MODES}
            input_norms = {"aligned": [], "reversed": [], "common": [],
                           "aligned_clip_fraction": [], "reversed_clip_fraction": []}
            for start in range(0, len(groups["passive"]), BATCH_TEMPLATES):
                sub = groups["passive"][start:start+BATCH_TEMPLATES]
                pool = m92.donors(sub)
                a_trace = m92.trace(model, pool["aligned"], device)
                r_trace = m92.trace(model, pool["reversed"], device)
                v = get_vectors(model, sub, a_trace, r_trace, device, 0, len(groups["passive"]))
                input_norms["aligned"].append(float(v["na"].mean()))
                input_norms["reversed"].append(float(v["nr"].mean()))
                input_norms["common"].append(float(v["common"].mean()))
                input_norms["aligned_clip_fraction"].append(float((v["na"] > v["nr"]).float().mean()))
                input_norms["reversed_clip_fraction"].append(float((v["nr"] > v["na"]).float().mean()))
                for mode in MODES:
                    random = (random_vectors[int(mode.rsplit("_", 1)[1])]
                              if mode.startswith("equalnorm_random_") else None)
                    out, stats = intervention_logits(model, v, mode, random, template_offset=start)
                    output_chunks[mode].append(out)
                    norm_chunks[mode].append(stats["applied_norm_mean"])
            metrics = {mode: m91.analyze_logits(torch.cat(chunks), groups["passive"])
                       for mode, chunks in output_chunks.items()}
            reproduction = check_previous(previous, seed, arm,
                                           metrics["native"], metrics)
            changes = {}
            baseline = metrics["native"]
            for mode in MODES:
                m = metrics[mode]
                changes[mode] = {
                    "second_event_role_gating": m["second_event_role_gating"],
                    "delta_gating": m["second_event_role_gating"] - baseline["second_event_role_gating"],
                    "second_accuracy": second_accuracy(m),
                    "delta_second_accuracy": second_accuracy(m) - second_accuracy(baseline),
                    "four_choice_accuracy": m["four_choice_accuracy"],
                    "full_vocab_accuracy": m["full_vocab_accuracy"],
                    "wrong_event_rate": m["wrong_event_rate"],
                    "correct_event_wrong_role_rate": m["correct_event_wrong_role_rate"],
                    "all_four_queries_correct_rate": m["all_four_queries_correct_rate"],
                    "mean_applied_norm_per_participant": mean(norm_chunks[mode]),
                }
            info = {k: mean(values) for k, values in input_norms.items()}
            results.append({
                "seed": seed, "arm": arm,
                "interventions": changes,
                "source_norms": info,
                "original_ml092_reproduction_deviations": reproduction,
            })
            a = changes["equalnorm_aligned"]
            b = changes["equalnorm_reversed"]
            rr = mean(changes[f"equalnorm_random_{j}"]["second_event_role_gating"]
                      for j in range(RANDOM_REPLICAS))
            print(f"seed={seed} arm={arm:8s} "
                  f"native2={second_accuracy(baseline):.3f} "
                  f"equalA_dacc={a['delta_second_accuracy']:+.3f} "
                  f"equalR_dacc={b['delta_second_accuracy']:+.3f} "
                  f"AminusR_gating={a['second_event_role_gating']-b['second_event_role_gating']:+.3f} "
                  f"AminusRand_gating={a['second_event_role_gating']-rr:+.3f}",
                  flush=True)
    assert len(results) == len(SEEDS) * len(ARMS)
    def get(seed, arm):
        return next(r for r in results if r["seed"] == seed and r["arm"] == arm)
    summary = {}
    for arm in ARMS:
        rows = [get(seed, arm) for seed in SEEDS]
        def vals(fn):
            return [fn(r) for r in rows]
        def mode(r, name):
            return r["interventions"][name]
        gating = vals(lambda r: mode(r,"equalnorm_aligned")["second_event_role_gating"] -
                                 mode(r,"equalnorm_reversed")["second_event_role_gating"])
        accuracy = vals(lambda r: mode(r,"equalnorm_aligned")["second_accuracy"] -
                                   mode(r,"equalnorm_reversed")["second_accuracy"])
        random_control = vals(lambda r: mode(r,"equalnorm_aligned")["second_event_role_gating"] -
                              mean(mode(r, f"equalnorm_random_{j}")["second_event_role_gating"]
                                   for j in range(RANDOM_REPLICAS)))
        summary[arm] = {
            "equalnorm_aligned_minus_reversed_gating_by_seed": dict(zip(map(str,SEEDS),gating)),
            "equalnorm_aligned_minus_reversed_gating_mean": mean(gating),
            "equalnorm_aligned_minus_reversed_gating_bootstrap_ci": bootstrap(gating),
            "equalnorm_aligned_minus_reversed_gating_positive_seeds": sum(x > 0 for x in gating),
            "equalnorm_aligned_minus_reversed_second_accuracy_mean": mean(accuracy),
            "equalnorm_aligned_minus_random_gating_mean": mean(random_control),
            "equalnorm_aligned_mean_second_accuracy_change": mean(vals(lambda r:mode(r,"equalnorm_aligned")["delta_second_accuracy"])),
            "equalnorm_reversed_mean_second_accuracy_change": mean(vals(lambda r:mode(r,"equalnorm_reversed")["delta_second_accuracy"])),
            "original_aligned_mean_second_accuracy_change": mean(vals(lambda r:mode(r,"original_aligned")["delta_second_accuracy"])),
            "original_reversed_mean_second_accuracy_change": mean(vals(lambda r:mode(r,"original_reversed")["delta_second_accuracy"])),
            "mean_common_per_participant_delta_norm": mean(r["source_norms"]["common"] for r in rows),
            "mean_aligned_clipped_fraction": mean(r["source_norms"]["aligned_clip_fraction"] for r in rows),
            "mean_reversed_clipped_fraction": mean(r["source_norms"]["reversed_clip_fraction"] for r in rows),
        }
    doc = {
        "experiment":"ML-093", "status":"EXPLORATORY_MAGNITUDE_AUDIT",
        "plan_sha256":plan_hash, "ml088_protocol_sha256":old_hash,
        "ml092_frozen_plan_sha256": previous["plan_sha256"],
        "protected_examples_generated":False,
        "n_existing_checkpoints": len(SEEDS) * len(ARMS),
        "rows":results,
        "summary":summary,
        "interpretation_warning": (
            "Same models and ML-092 data; post-ML-092 audit, not independent confirmation. "
            "Per-example per-participant min-norm scaling equalizes Euclidean delta size "
            "but not donor source positions, causal histories, semantics or directions. "
            "Pre-question transplant may introduce answer-relevant semantic information. "
            "Do not interpret positive result as a native necessary binding circuit."
        ),
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(doc,indent=2)+"\n")
    print("\nML-093 EXPLORATORY SUMMARY\n"+json.dumps(summary,indent=2))
    print("Saved", REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("preflight","plan","run"))
    ap.add_argument("--device", default="auto", choices=("auto","cpu","mps","cuda"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    old_hash, groups, prior_hash, previous = prepare()
    spec = specification(old_hash, groups, prior_hash)
    if args.phase == "preflight":
        if PLAN.exists():
            freeze(spec, False)
        preflight(groups)
    else:
        h = freeze(spec, create=args.phase == "plan")
        if args.phase == "plan":
            print("ML-093 frozen exploratory plan:", PLAN, "sha256", h)
            print("No trained checkpoints loaded; no protected examples accessed.")
        else:
            run(old_hash, h, groups, previous, ml.select_device(args.device))


if __name__ == "__main__":
    main()
