"""ML-099: frozen, fresh-cohort replication of ML-098 L2 role-Q interaction.

Fresh participant names, matching splits and eight paired training seeds. Reuses
EXACT ML-098 Q-swap implementation. Do not modify ML-080/084/091/098.

Protected examples are generated ONLY by --phase evaluate --confirm-protected
when all 16 models are trained and both arms pass the frozen competence gate.

From repository root:
    python scripts/ml099_fresh_role_q_replication.py --phase preflight
    python scripts/ml099_fresh_role_q_replication.py --phase plan
    # Commit script and reports/ml099_role_q/frozen_protocol.json.
    python scripts/ml099_fresh_role_q_replication.py --phase train --until 3200
    python scripts/ml099_fresh_role_q_replication.py --phase train --until 12800
    python scripts/ml099_fresh_role_q_replication.py --phase gate
    # Deliberate one-time decision to open protected test set:
    python scripts/ml099_fresh_role_q_replication.py --phase evaluate --confirm-protected
"""
import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml084_prospective_geometry as m84
import ml091_counterfactual_binding as m91
import ml098_question_q_path as m98

# All of these choices are fixed before seeing any ML-099 trained-model output.
NAMES = (
    "Adele", "Barton", "Celia", "Darwin", "Eloise", "Fabian", "Greta", "Hugo",
    "Ingrid", "Jasper", "Kira", "Leon", "Marta", "Noel", "Opal", "Quentin",
)
SEEDS = (809, 811, 821, 823, 827, 829, 839, 853)
MATCHING_SEED = 99099
ARMS = ("familiar", "target")
VOICES = ("active", "passive")
ROOT = Path("results/ml099_fresh_role_q")
PLAN = Path("reports/ml099_role_q/frozen_protocol.json")
REPORT = Path("reports/ml099_role_q/protected_evaluation.json")
OPENED = Path("reports/ml099_role_q/protected_opened_once.json")
FINAL_STEP = 12800
BATCH_GROUPS = 12
PER_HELD_PAIR = 8  # 24 disjoint held pairs x 8 = 192 sources per voice


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def prepare():
    # m84.prepare reuses the frozen ML-080 training code and configures its globals.
    # Only call ONCE per Python process; no old-cohort prepare() afterwards.
    m84.NEW_NAMES = NAMES
    m84.NEW_SEEDS = SEEDS
    m84.MATCHING_SEED = MATCHING_SEED
    vocab, train, cal_rounds, held, base, cal, dose, grid = m84.prepare()
    assert tuple(ml.NAMES) == NAMES and tuple(ml.SEEDS) == SEEDS
    assert len(base) == 20736 and len(cal) == 6912
    assert all(len(dose[arm]) == 64 for arm in ARMS)
    assert len(grid["10"]) == 64
    assert len({p for rd in held for p in rd}) == 24
    return vocab, train, cal_rounds, held, base, cal, dose, grid


def protocol(inputs):
    vocab, train, cal_rounds, held, base, cal, dose, grid = inputs
    return {
        "experiment": "ML-099",
        "status": "PROSPECTIVE_FRESH_COHORT_REPLICATION",
        "script_sha256": sha(__file__),
        "dependencies_sha256": {
            "ml080": sha(ml.__file__), "ml084": sha(m84.__file__),
            "ml091": sha(m91.__file__), "ml098": sha(m98.__file__),
        },
        "names": NAMES,
        "seeds": SEEDS,
        "matching_seed": MATCHING_SEED,
        "vocab_sha256": digest(vocab),
        "train_pair_ids": train,
        "cal_pair_ids": cal_rounds,
        "heldout_pair_ids_sealed": held,
        "train_prompts_sha256": digest([ml.prompt(c) for c in base]),
        "calibration_prompts_sha256": digest([ml.prompt(c) for c in cal]),
        "exposure_sha256": {a: digest([ml.prompt(c) for c in dose[a]]) for a in ARMS},
        "model": "Original ML-080 3-layer 4-head width-96 Transformer, unchanged",
        "training": "Original ML-080 12800-step optimizer, batch, data, paired-seed schedule and calibration-gate code; target/familiar 64 repeated exposure cases",
        "test_generation": "Only after protected unlock: ML-080 protected_target_cases(held); agent cases sorted by SHA256 prompt independently within each held pair; select first 8/32; construct four event x role questions per source, for active and passive second-clause voice",
        "test_per_held_pair": PER_HELD_PAIR,
        "test_sources_per_voice": 24 * PER_HELD_PAIR,
        "intervention": "Exactly ML-098 answer-token attention Q transplant at L2, from opposite-role question about the identical source sentence; receiver K/V/residual native; no donor training or extra answer token",
        "PRIMARY": {
            "metric": "L2 role-swap SECOND-event donor-minus-receiver logit-margin shift",
            "contrast": "(target_passive - familiar_passive) - (target_active - familiar_active), computed within each seed, then averaged across 8 paired seeds",
            "direction": "positive, following the exploratory ML-098 finding",
            "success_rule": "paired-seed mean > 0, >=6/8 positive, and lower limit of 20k seed bootstrap 95% CI > 0",
            "unit_of_independence": "eight paired training seeds, not the 192 heldout source sentences",
            "no_peeking": "No protected examples, predictions, or Q effects accessed before final competence gate and explicit unlock",
        },
        "secondary_frozen": [
            "Same binary second-event donor-answer gain interaction (ML-098 metric)",
            "Native choice4 and full-vocabulary accuracy for both voices",
            "Native donor-answer rate and new/lost donor predictions",
            "L2 Q-difference norm and output-logit perturbation norm",
            "Native participant-name logit scale, and scaled-margin sensitivity analysis",
            "L2 event-Q swap control, L3 role-Q swap control, L1 role-Q structural null",
            "Protected behavioral target-minus-familiar gap on the same held-out pair split",
        ],
        "competence_gate": "At final 12800 steps, >=6/8 models PER ARM must have familiar-calibration 4-choice accuracy>=0.95 and both-role correctness>=0.90; failing gate does not authorize test or seed exclusions",
        "limitations": "Same synthetic task and 3-layer architecture as discovery; not pretrained-model or cross-architecture validation. Q donor may encode answer content, so confirmation tests prediction, not native necessity or a unique binding circuit.",
        "one_time_protected": True,
    }


def freeze(doc, create=False):
    # JSON-roundtrip is necessary: Python tuples become JSON lists.
    expected = json.loads(json.dumps({**doc, "sha256": digest(doc)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        with PLAN.open("x") as stream:
            json.dump(expected, stream, indent=2)
            stream.write("\n")
    if not PLAN.exists():
        raise FileNotFoundError(f"Freeze the protocol first: {PLAN}")
    saved = json.loads(PLAN.read_text())
    if saved != expected:
        differing = [k for k in expected if expected[k] != saved.get(k)]
        raise RuntimeError(f"FROZEN PROTOCOL MISMATCH: {differing}; do not overwrite")
    return expected["sha256"]


def four_question_groups(cases, voice):
    result = []
    for case in cases:
        assert case.target_pos == 1 and case.role == "agent"
        second = replace(case, target_voice=voice)
        group = m91.four_questions(second)
        assert len({m91.sentence(q) for q in group}) == 1
        assert len({ml.answer(q) for q in group}) == 4
        result.append(group)
    return result


def preflight(inputs):
    grid = inputs[-1]
    sample = grid["10"][:3]
    groups = four_question_groups(sample, "passive")
    torch.manual_seed(99099)
    model = ml.Model().eval()
    cases = [q for group in groups for q in group]
    x, ro, _, _ = ml.encode(cases)
    with torch.inference_mode():
        direct = model(x, ro).float()
        reference = m98.forward(model, groups, "cpu")
        own = m98.forward(model, groups, "cpu", 1, "self")
        l1 = m98.forward(model, groups, "cpu", 0, "role")
        l2 = m98.forward(model, groups, "cpu", 1, "role")
    err = max(float((direct - a).abs().max()) for a in (reference, own, l1))
    assert err < 3e-4, err
    change = float((l2 - direct).abs().max())
    assert change > 1e-6, change
    print("PREFLIGHT PASS: fresh names/splits, four questions/source, native parity, L1 null, L2 intervention")
    print(f"names={len(NAMES)} fresh_seeds={len(SEEDS)} calibration_sources=64; model_max_err={err:.3g}; L2_synthetic_change={change:.3g}")
    print("No trained model loaded. No protected examples generated.")


def model_gate(h):
    rows = ml.all_metrics(ROOT, h, FINAL_STEP, complete=True)
    counts = {}
    for arm in ARMS:
        counts[arm] = sum(
            rows[seed, arm]["calibration"]["choice_accuracy"] >= 0.95
            and rows[seed, arm]["calibration"]["correct_both_rate"] >= 0.90
            for seed in SEEDS
        )
    passed = all(counts[a] >= 6 for a in ARMS)
    print("ML-099 COMPETENCE GATE:", counts, "PASS" if passed else "FAIL")
    return passed


def protected_sources(held):
    # This function MUST NOT be called in plan/preflight/train/gate.
    target_cases = ml.protected_target_cases(held)
    agents = [c for c in target_cases if c.role == "agent"]
    held_pairs = sorted({p for rd in held for p in rd})
    selected = []
    for pair in held_pairs:
        candidates = [c for c in agents if c.target_pair == pair]
        assert len(candidates) == 32, (pair, len(candidates))
        chosen = sorted(candidates, key=lambda c: digest(ml.prompt(c)))[:PER_HELD_PAIR]
        selected.extend(chosen)
    assert len(selected) == 24 * PER_HELD_PAIR
    assert len({tuple(ml.prompt(c)) for c in selected}) == len(selected)
    return selected, target_cases


def bootstrap(values, n=20000):
    rng = random.Random(990990)
    draws = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(n))
    return [draws[int(.025*n)], draws[int(.975*n)-1]]


@torch.inference_mode()
def q_difference_norm(model, groups, device):
    """At L2, measure receiver-vs-opposite-role donor Q difference, second queries."""
    values = []
    for start in range(0, len(groups), BATCH_GROUPS):
        subset = groups[start:start+BATCH_GROUPS]
        cases = [q for g in subset for q in g]
        ids, readout, _, _ = ml.encode(cases)
        ids, readout = ids.to(device), readout.to(device)
        b, length = ids.shape
        ix = torch.arange(b, device=device)
        x = model.embed(ids) + model.pos(torch.arange(length, device=device))
        causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        x = model.blocks[0](x, causal)
        blk = model.blocks[1]
        normed = blk.n1(x)
        width = normed.shape[-1]
        q = F.linear(normed[ix, readout], blk.attn.in_proj_weight[:width],
                     blk.attn.in_proj_bias[:width]).reshape(len(subset), 4, width)
        distance = (q[:, 2, :] - q[:, 3, :]).norm(dim=-1)
        values.extend(distance.cpu().tolist())
    return mean(values)


@torch.inference_mode()
def voice_metrics(model, groups, device):
    all_native = []
    edited = {key: [] for key in ("L2/role", "L2/event", "L3/role", "L1/role")}
    for start in range(0, len(groups), BATCH_GROUPS):
        subset = groups[start:start+BATCH_GROUPS]
        all_native.append(m98.forward(model, subset, device))
        for key in edited:
            layer, swap = key.split("/")
            edited[key].append(m98.forward(model, subset, device, int(layer[1:])-1, swap))
    native = torch.cat(all_native)
    patched = {k: torch.cat(v) for k, v in edited.items()}
    metrics = {
        k: m98.outcomes(native, z, groups, k.split("/")[1]) for k, z in patched.items()
    }
    metrics["native"] = m98.outcomes(native, native, groups, "self")
    ids = torch.tensor([ml.VOCAB[n] for n in NAMES], dtype=torch.long)
    # Descriptive scale for sensitivity analysis; no hypothesis tuning.
    native_name_logit_std = float(native[:, ids].std(dim=1).mean())
    q_norm = q_difference_norm(model, groups, device)
    metrics["audit"] = {
        "mean_native_name_logit_sd": native_name_logit_std,
        "mean_second_role_q_difference_norm": q_norm,
        "role_L2_second_margin_per_name_logit_sd":
            metrics["L2/role"]["second_margin_change"] / max(native_name_logit_std, 1e-6),
        "role_L2_second_margin_per_q_delta_norm":
            metrics["L2/role"]["second_margin_change"] / max(q_norm, 1e-6),
    }
    assert metrics["L1/role"]["mean_output_logit_delta_l2"] < 0.01
    return metrics


def evaluate(h, held, device, unlock):
    if not unlock:
        raise PermissionError("Explicit --confirm-protected required")
    if REPORT.exists() or OPENED.exists():
        raise FileExistsError("ML-099 protected evaluation already opened; cannot repeat")
    if not model_gate(h):
        raise PermissionError("Competence gate failed; protected test remains sealed")
    for seed in SEEDS:
        for arm in ARMS:
            chk = ROOT / f"seed_{seed}" / arm / f"model_step_{FINAL_STEP:05d}.pt"
            if not chk.is_file():
                raise FileNotFoundError(chk)
    OPENED.parent.mkdir(parents=True, exist_ok=True)
    with OPENED.open("x") as fp:
        json.dump({"experiment": "ML-099", "protocol_sha256": h,
                   "status": "PROTECTED EVALUATION OPENED ONCE"}, fp, indent=2)
    # Protected cases are first instantiated HERE, after gate + one-time lock.
    sources, protected_questions = protected_sources(held)
    cases_by_voice = {v: four_question_groups(sources, v) for v in VOICES}
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            chk = ROOT / f"seed_{seed}" / arm / f"model_step_{FINAL_STEP:05d}.pt"
            model = ml.load_checkpoint(chk, seed, arm, h, device)
            behavior = ml.measure(model, protected_questions, device)
            measures = {v: voice_metrics(model, cases_by_voice[v], device) for v in VOICES}
            results.append({"seed": seed, "arm": arm, "protected_behavior": behavior,
                            "voices": measures})
            print(f"seed={seed} arm={arm:8s} target4={behavior['choice_accuracy']:.3f} "
                  f"active_L2role_margin={measures['active']['L2/role']['second_margin_change']:+.2f} "
                  f"passive_L2role_margin={measures['passive']['L2/role']['second_margin_change']:+.2f}", flush=True)
    lookup = {(x["seed"], x["arm"]): x for x in results}
    def score(seed, arm, voice, name):
        return lookup[seed, arm]["voices"][voice]["L2/role"][name]
    def contrast(name):
        return [
            (score(s, "target", "passive", name)-score(s, "familiar", "passive", name))
            -(score(s, "target", "active", name)-score(s, "familiar", "active", name))
            for s in SEEDS
        ]
    primary = contrast("second_margin_change")
    binary = contrast("second_donor_gold_gain4")
    normalized = [
        (lookup[s, "target"]["voices"]["passive"]["audit"]["role_L2_second_margin_per_name_logit_sd"]
         -lookup[s, "familiar"]["voices"]["passive"]["audit"]["role_L2_second_margin_per_name_logit_sd"])
        -(lookup[s, "target"]["voices"]["active"]["audit"]["role_L2_second_margin_per_name_logit_sd"]
         -lookup[s, "familiar"]["voices"]["active"]["audit"]["role_L2_second_margin_per_name_logit_sd"])
        for s in SEEDS
    ]
    ci = bootstrap(primary)
    gaps = [lookup[s, "target"]["protected_behavior"]["choice_accuracy"]
            -lookup[s, "familiar"]["protected_behavior"]["choice_accuracy"] for s in SEEDS]
    summary = {
        "primary_predeclared": {
            "per_seed": dict(zip(map(str, SEEDS), primary)),
            "mean": mean(primary), "seed_bootstrap95": ci,
            "positive_seed_pairs": sum(v > 0 for v in primary),
            "success_rule_met": mean(primary) > 0 and sum(v > 0 for v in primary) >= 6 and ci[0] > 0,
        },
        "secondary": {
            "binary_donor_gain_interaction_mean": mean(binary),
            "binary_donor_gain_by_seed": dict(zip(map(str, SEEDS), binary)),
            "scaled_margin_interaction_mean": mean(normalized),
            "scaled_margin_by_seed": dict(zip(map(str, SEEDS), normalized)),
            "behavioral_target_minus_familiar_mean": mean(gaps),
            "behavioral_gap_by_seed": dict(zip(map(str, SEEDS), gaps)),
        },
        "warning": "Independent fresh names/splits/seeds within SAME synthetic grammar and architecture. Replication of selected metric is not unique circuit identification or proof of mediation.",
    }
    with REPORT.open("x") as fp:
        json.dump({"experiment": "ML-099", "protocol_sha256": h,
                   "protected_evaluation_opened": True, "sources_per_voice": len(sources),
                   "models": len(results), "summary": summary, "rows": results}, fp, indent=2)
        fp.write("\n")
    print("\nML-099 FROZEN PRIMARY RESULT\n" + json.dumps(summary["primary_predeclared"], indent=2))
    print("Saved:", REPORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("preflight", "plan", "train", "gate", "evaluate"), required=True)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--until", type=int, choices=(3200, 12800), default=3200)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--confirm-protected", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    inputs = prepare()
    doc = protocol(inputs)
    if args.phase == "preflight":
        preflight(inputs)
        print("Prospective primary: L2 role-Q SECOND-event margin, target/familiar x passive/active DID")
        return
    h = freeze(doc, create=args.phase == "plan")
    if args.phase == "plan":
        print("ML-099 frozen protocol SHA256:", h)
        print("COMMIT the script and frozen protocol before training")
        return
    if args.phase == "train":
        _, _, _, _, base, cal, dose, _ = inputs
        for seed in (SEEDS if args.seed is None else (args.seed,)):
            for arm in (ARMS if args.arm is None else (args.arm,)):
                ml.train_one(seed, arm, args.until, args.save_every, ROOT,
                             h, base, cal, dose, ml.select_device(args.device))
        print("TRAINING DONE for requested seeds/arms; held-out test remains sealed")
        return
    if args.phase == "gate":
        model_gate(h)
        return
    evaluate(h, inputs[3], ml.select_device(args.device), args.confirm_protected)


if __name__ == "__main__":
    main()
