"""
 ML-082: preregistered donor-directed patch specificity on unused ML-080 calibration.

Run from repository root after placing this beside ml080 and ml081 scripts:
 python3 scripts/ml082_directional_layer_patching.py --phase plan
 python3 scripts/ml082_directional_layer_patching.py --phase preflight
 # Freeze+commit the code and generated protocol BEFORE evaluating real checkpoints.
 python3 scripts/ml082_directional_layer_patching.py --phase evaluate

No protected cases are ever instantiated. The original 16 trained checkpoints are used
only after the protocol is frozen. This is a same-model held-aside calibration test,
NOT validation on independent models / grammar or held-out ML-080 protected data.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch
import ml080_learning_curve as ml
import ml081_layerwise_event_role as ml081

N_SENTENCES = 64
REPS = 5
SITES = ((0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp"),
         (2, "attn"), (2, "mlp"))
DONORS = ("role", "event")
SEEDS = tuple(ml.SEEDS)
ARMS = tuple(ml.ARMS)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def case_sets(doc):
    cal = ml.base_cases(doc["splits"]["calibration_pairs"])
    second = [c for c in cal if c.target_pos == 1 and
              c.target_voice == "active" and
              c.distractor_voice == "active" and c.role == "agent"]
    assert len(second) == 576
    second.sort(key=lambda c: hashlib.sha256(
        " ".join(ml.prompt(c)).encode()).hexdigest())
    # ML-081 used the first 48 entries of the very same deterministic sort.
    previously_used = set(" ".join(ml.prompt(c)) for c in second[:48])
    picked = second[48:48+N_SENTENCES]
    assert len(picked) == N_SENTENCES
    assert not any(" ".join(ml.prompt(c)) in previously_used for c in picked)
    questions = []
    for c in picked:
        first = ml081.swap_event(c)
        assert ml081.changed_tokens(first, c) == [("first", "second")]
        questions.extend((first, replace(first, role="patient"),
                          c, replace(c, role="patient")))
    role = [replace(c, role="patient" if c.role == "agent" else "agent")
            for c in questions]
    event = [ml081.swap_event(c) for c in questions]
    lexical = [ml081.lexical_donor(c) for c in questions]
    for c, r, e, l in zip(questions, role, event, lexical):
        assert len(ml081.changed_tokens(c, r)) == 1
        assert len(ml081.changed_tokens(c, e)) == 1
        assert len(ml081.changed_tokens(c, l)) == 1
        assert len({ml.answer(c), ml.answer(r), ml.answer(e)}) == 3
        assert ml.answer(c) == ml.answer(l)
    return questions, {"role": role, "event": event, "lexical": lexical}


def freeze_doc(doc, cases):
    return {
        "experiment": "ML-082",
        "script_sha256": ml.sha_file(__file__),
        "ml080_protocol_sha256": doc["sha256"],
        "ml081_source_sha256": ml.sha_file(ml081.__file__),
        "case_selection": "SHA256-sorted active/active calibration second-agent entries 48:112; ML-081 used 0:48",
        "n_sentences": N_SENTENCES,
        "n_questions": len(cases),
        "sample_digest": digest([ml.prompt(c) for c in cases]),
        "seeds": SEEDS, "arms": ARMS, "step": 12800,
        "sites": [list(s) for s in SITES],
        "donors": DONORS, "random_repetitions": REPS,
        "primary_metric": "donor-hit excess = P(patched predicts counterfactual gold) - average P(norm-matched random patch predicts counterfactual gold)",
        "primary_contrast": "paired seed average donor-hit excess for L1/L2 attention versus L1/L2 MLP; report both arms and donor types separately",
        "secondary_metrics": ["wrong-other flip (not donor)", "native accuracy", "patched flip rate", "lexical corruption", "donor change in logit advantage", "median patch delta norm", "random-control flip rate"],
        "design": "same 16 ML-080 models, disjoint ML-081 calibration instances only, no ML-080 protected examples",
        "interpretation": "Directional donor switching above isotropic norm-matched baselines supports answer-specific information transfer at a site, not native necessity, a unique circuit, or explanation of protected generalization; intervention magnitude remains site/arm dependent",
        "evaluation_rule": "after plan/preflight, freeze and commit script/protocol before running evaluate; no overwrite or reruns for tuning",
    }


def check_freeze(path, doc, cases, create):
    base = freeze_doc(doc, cases)
    complete = json.loads(json.dumps({**base, "sha256": digest(base)}))
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(complete, indent=2) + "\n")
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen protocol {path}: run --phase plan")
    assert json.loads(path.read_text()) == complete, "ML-082 script/protocol mismatch"
    return complete["sha256"]


def get_preds(logits, choices):
    return choices.gather(1, logits.gather(1, choices).argmax(1, keepdim=True)).flatten()


@torch.inference_mode()
def measure_one(model, cases, donors, device, seed, arm):
    prepared = {}
    for label, batch in {"native": cases, **donors}.items():
        x, r, y, choices = ml.encode(batch)
        prepared[label] = (x.to(device), r.to(device), y.to(device), choices.to(device))
    x, readout, gold, choices = prepared["native"]
    original, orig_sites = ml081.forward_trace(model, x, readout)
    err = float((original - model(x, readout).float()).abs().max())
    assert err < 1e-3, f"Forward mismatch for {seed}/{arm}: {err}"
    native_pred = get_preds(original, choices)
    native_acc = float((native_pred == gold).float().mean())
    donor_sites, donor_gold = {}, {}
    for name, (di, dr, dg, _) in prepared.items():
        if name == "native":
            continue
        _, donor_sites[name] = ml081.forward_trace(model, di, dr)
        donor_gold[name] = dg
    output = []
    for layer, site in SITES:
        origin = orig_sites[layer][site]
        for donor in DONORS:
            target = donor_gold[donor]
            assert bool(torch.all(gold != target))
            delta = donor_sites[donor][layer][site] - origin
            patched, _ = ml081.forward_trace(
                model, x, readout, patch=(layer, site, origin + delta))
            pred = get_preds(patched, choices)
            hit = (pred == target).float()
            error_other = ((pred != target) & (pred != gold)).float()
            flip = (pred != gold).float()
            native_donor_adv = ml081.donor_advantage(original, target, gold)
            shifted_donor_adv = ml081.donor_advantage(patched, target, gold) - native_donor_adv
            random_hit = []
            random_flip = []
            random_shift = []
            for rep in range(REPS):
                # Separate independent, reproducible random streams by model/site/donor/rep.
                key = (seed * 1000000 + (0 if arm == "familiar" else 1) * 100000
                       + layer * 10000 + (0 if site == "attn" else 1) * 1000
                       + (0 if donor == "role" else 1) * 100 + rep)
                generator = torch.Generator(device="cpu").manual_seed(key)
                rd = ml081.random_like_delta(delta, generator)
                control, _ = ml081.forward_trace(
                    model, x, readout, patch=(layer, site, origin + rd))
                p = get_preds(control, choices)
                random_hit.append(float((p == target).float().mean()))
                random_flip.append(float((p != gold).float().mean()))
                random_shift.append(float((
                    ml081.donor_advantage(control, target, gold) - native_donor_adv
                ).mean()))
            output.append({
                "seed": seed, "arm": arm, "layer": layer+1, "site": site,
                "donor": donor, "n": len(cases), "native_acc": native_acc,
                "donor_hit": float(hit.mean()),
                "random_donor_hit": mean(random_hit),
                "donor_hit_excess": float(hit.mean()) - mean(random_hit),
                "flip_rate": float(flip.mean()),
                "random_flip_rate": mean(random_flip),
                "non_donor_flip_rate": float(error_other.mean()),
                "donor_fraction_among_flips": float(hit.sum()/flip.sum()) if bool(flip.sum() > 0) else None,
                "donor_advantage_shift": float(shifted_donor_adv.mean()),
                "random_advantage_shift": mean(random_shift),
                "median_delta_norm": float(delta.norm(dim=1).median()),
            })
        # Lexical negative control: same gold; no claims of norm equivalence.
        lexdelta = donor_sites["lexical"][layer][site] - origin
        lexpatched, _ = ml081.forward_trace(
            model, x, readout, patch=(layer, site, origin + lexdelta))
        output.append({
            "seed": seed, "arm": arm, "layer": layer+1, "site": site,
            "donor": "lexical", "n": len(cases), "native_acc": native_acc,
            "lexical_flip_rate": float((get_preds(lexpatched, choices) != gold).float().mean()),
            "median_delta_norm": float(lexdelta.norm(dim=1).median()),
        })
    return err, output


def overview(rows):
    print("\nML-082 directional specificity: excess donor-hit over same-norm random controls")
    print(f"{'Layer':<7}{'Site':<10}{'Donor':<9}{'Familiar':>11}{'Target':>11}{'Gap T-F':>11}")
    for layer, site in SITES:
        for donor in DONORS:
            f = [r["donor_hit_excess"] for r in rows if r["layer"] == layer+1 and r["site"] == site and r["donor"] == donor and r["arm"] == "familiar"]
            t = [r["donor_hit_excess"] for r in rows if r["layer"] == layer+1 and r["site"] == site and r["donor"] == donor and r["arm"] == "target"]
            assert len(f) == len(t) == 8
            print(f"{layer+1:<7}{site:<10}{donor:<9}{mean(f):>+11.3f}{mean(t):>+11.3f}{mean(t)-mean(f):>+11.3f}")
    print("\nPrimary within-model early-attention minus early-MLP excess, reported by arm")
    for arm in ARMS:
        vals = []
        for seed in SEEDS:
            att = [r["donor_hit_excess"] for r in rows if r["seed"] == seed and r["arm"] == arm and r["layer"] in (1,2) and r["site"] == "attn" and r["donor"] in DONORS]
            mlp = [r["donor_hit_excess"] for r in rows if r["seed"] == seed and r["arm"] == arm and r["layer"] in (1,2) and r["site"] == "mlp" and r["donor"] in DONORS]
            assert len(att) == len(mlp) == 4
            vals.append(mean(att)-mean(mlp))
        print(f"  {arm:8s}: mean={mean(vals):+.3f}; paired seeds=" + ", ".join(f"{v:+.3f}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("plan", "preflight", "evaluate"))
    ap.add_argument("--ml080-protocol", default="reports/ml080/frozen_protocol.json")
    ap.add_argument("--ml081-protocol", default="reports/ml081/frozen_protocol.json")
    ap.add_argument("--protocol", default="reports/ml082/frozen_protocol.json")
    ap.add_argument("--root", default="results/ml080_learning_curve")
    ap.add_argument("--output", default="reports/ml082/directional_specificity.json")
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    doc = ml081.verify_ml080(Path(args.ml080_protocol))
    old_doc = json.loads(Path(args.ml081_protocol).read_text())
    assert old_doc["experiment"] == "ML-081"
    assert old_doc["code_sha256"] == ml.sha_file(ml081.__file__)
    questions, donors = case_sets(doc)
    h = check_freeze(Path(args.protocol), doc, questions, args.phase == "plan")
    if args.phase == "plan":
        print(f"ML-082 frozen: {args.protocol}; sha256={h[:12]}; {len(questions)} calibration questions")
        return
    # preflight checks files, never loads trained model predictions or any protected cases.
    ml081.check_runs(Path(args.root), doc)
    print("ML-082: 16 checkpoints found; distinct ML-081 calibration examples; no protected cases")
    torch.manual_seed(82082)
    demo = ml.Model().eval()
    small = questions[:4]
    ids, ro, _, _ = ml.encode(small)
    with torch.inference_mode():
        orig, tr = ml081.forward_trace(demo, ids, ro)
        assert float((orig-demo(ids,ro)).abs().max()) < 1e-3
        unchanged, _ = ml081.forward_trace(demo, ids, ro, patch=(0, "attn", tr[0]["attn"]))
        assert float((orig-unchanged).abs().max()) < 1e-3
    if args.phase == "preflight":
        print("Preflight PASS: forward reconstruction, identity patch, checkpoint presence, case isolation")
        return
    result_path = Path(args.output)
    if result_path.exists():
        raise FileExistsError(f"Refusing overwrite: {result_path}")
    device = ml.select_device(args.device)
    result = {"experiment":"ML-082", "protocol_sha256":h, "validation_scope":"new calibration prompts, same trained models; not independent replication", "n_questions":len(questions), "rows":[]}
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint = Path(args.root)/f"seed_{seed}"/arm/"model_step_12800.pt"
            model = ml.load_checkpoint(checkpoint, seed, arm, doc["sha256"], device)
            parity, scored = measure_one(model, questions, donors, device, seed, arm)
            result["rows"].extend(scored)
            first = next(r for r in scored if r["layer"] == 1 and r["site"] == "attn" and r["donor"] == "role")
            print(f"seed={seed} arm={arm:8s} native={first['native_acc']:.3f} L1_attn_role_donor_excess={first['donor_hit_excess']:+.3f} parity={parity:.2e}", flush=True)
    overview(result["rows"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    print(f"Saved: {result_path}; exploratory new calibration only; NOT independent models")


if __name__ == "__main__":
    main()
