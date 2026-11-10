"""ML-089: exploratory receiver-only attention-routing study.

Uses frozen ML-088 checkpoints and *calibration pair* counterfactuals.
Never calls protected_target_cases, never accesses the ML-088 protected set.

Phases (from repository root):
    python3 scripts/ml089_receiver_only_routing.py --phase plan
    python3 scripts/ml089_receiver_only_routing.py --phase preflight
    # Review + commit this script and frozen discovery plan.
    python3 scripts/ml089_receiver_only_routing.py --phase run

Causal intervention: add log(4) to answer-query attention logits for
TWO SOURCE TOKENS, without donor activations and without selecting the gold
participant. The oracle queried-event selector (from case metadata) is an
external intervention and is NOT claimed to be the model's native mechanism.
All evaluations are EXPLORATORY; independent confirmation requires new models.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88

ROOT = Path("results/ml088_fresh_models")
PLAN = Path("reports/ml089/frozen_discovery_plan.json")
REPORT = Path("reports/ml089/discovery_evaluation.json")
SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
LAYERS = (0, 1, 2)
MODES = ("target_pair", "distractor_pair", "both_verbs", "all_names")
LOG_GAIN = math.log(4.0)
N_PER_PAIR = 4
BATCH = 48


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare():
    vocab, tr, ca, held, base, cal, dose, fit, valid = m88.prepare()
    old_hash = m88.frozen(m88.specification(vocab, tr, ca, held, fit, valid), False)
    assert tuple(ml.SEEDS) == SEEDS
    # Model gate is checked only in --phase run; plan/preflight require no checkpoints.

    # Use previously unexposed calibration PAIRS, not the opened protected pairs.
    cal_pairs = sorted({p for rd in ca for p in rd})
    train_pairs = {p for rd in tr for p in rd}
    held_pairs = {p for rd in held for p in rd}
    assert len(cal_pairs) == 24
    assert not (set(cal_pairs) & train_pairs or set(cal_pairs) & held_pairs)

    candidates = defaultdict(list)
    for c in cal:
        if (c.target_pos == 0 and c.target_voice == "passive"
                and c.role == "agent" and c.target_verb in ml.VERBS[2:]):
            candidates[c.target_pair].append(c)

    chosen = []
    for pair in cal_pairs:
        options = sorted(candidates[pair], key=lambda c: digest(ml.prompt(c)))
        assert len(options) >= N_PER_PAIR
        chosen.extend(options[:N_PER_PAIR])
    assert len(chosen) == 24 * N_PER_PAIR

    examples = {}
    for construction in ("familiar", "target"):
        rows = []
        for agent in chosen:
            c = agent if construction == "familiar" else replace(agent, target_pos=1)
            rows.extend((c, replace(c, role="patient")))
        assert len(rows) == 192
        assert all(c.target_pair in set(cal_pairs) for c in rows)
        assert all(c.target_voice == "passive" for c in rows)
        assert all((c.target_pos == 0) == (construction == "familiar") for c in rows)
        assert all(c.target_verb in ml.VERBS[2:] for c in rows)
        examples[construction] = rows

    # Paired questions differ by construction, NOT gold participant identity.
    for a, b in zip(examples["familiar"], examples["target"]):
        assert ml.answer(a) == ml.answer(b)
        assert set(a.target_pair) == set(b.target_pair)
        assert a.target_pos == 0 and b.target_pos == 1
        assert ml.prompt(a) != ml.prompt(b)

    # Target construction is not in any familiar-calibration or training case.
    calibration_prompts = {tuple(ml.prompt(c)) for c in cal}
    assert not any(tuple(ml.prompt(c)) in calibration_prompts for c in examples["target"])
    return old_hash, examples


def source_positions(c):
    tokens = ml.prompt(c)
    def one(tok):
        hits = [i for i, t in enumerate(tokens) if t == tok]
        assert len(hits) == 1, (tok, hits, tokens)
        return hits[0]
    target = tuple(sorted(one(n) for n in c.target_pair))
    distractor = tuple(sorted(one(n) for n in c.distractor_pair))
    verbs = tuple(sorted((one(c.target_verb), one(c.distractor_verb))))
    all_names = tuple(sorted(target + distractor))
    assert len(set(all_names)) == 4
    assert len(set(verbs)) == 2
    return {"target_pair": target, "distractor_pair": distractor,
            "both_verbs": verbs, "all_names": all_names}


def specification(old_hash, examples):
    return {
        "experiment": "ML-089",
        "status": "EXPLORATORY_DISCOVERY_NO_PROTECTED_CASES",
        "script_sha256": sha(__file__),
        "ml080_sha256": sha(ml.__file__),
        "ml088_sha256": sha(m88.__file__),
        "ml088_protocol_sha256": old_hash,
        "cohort": "Existing ML-088 16 frozen step-12800 checkpoints, eight paired seeds",
        "seeds": SEEDS,
        "selection": "Four SHA256-sorted first-passive, exposure-unseen-verb agent events for each of 24 calibration target pairs; append agent+patient",
        "per_construction_questions": 192,
        "familiar_prompt_sha256": digest([ml.prompt(c) for c in examples["familiar"]]),
        "target_prompt_sha256": digest([ml.prompt(c) for c in examples["target"]]),
        "intervention": {
            "mechanism": "Within the same receiver, add fixed log(4) to attention logits of answer query for source-token keys only, renormalizing normally",
            "source_token_selection": "Using queried event metadata, choose BOTH participant names, never choose the gold answer name alone",
            "source_positions": MODES,
            "layer_sites_zero_indexed": LAYERS,
            "primary_site": 1,
            "gain_log": LOG_GAIN,
            "controls": "Distractor pair: two names in the nonqueried event; both verbs: matched count, no names; all names: nonselective participant boost (different count)",
            "no_donor": "No activation from any other prompt or model is imported; avoids direct donor-answer injection",
        },
        "primary_exploratory_contrast": "Mean across 8 familiar-arm seeds of (target-pair minus distractor-pair) change in gold-vs-best-other four-choice margin at L2, on target-construction calibration pairs",
        "secondary": "Same contrast on familiar construction, target-exposure models, L1/L3 sites; continuous margins, 4-choice and full-vocabulary accuracy, correct-both-event, native-error rescue and native-correct damage, source attention mass, realized attention-output perturbation norms",
        "limitations": "External oracle selection of queried event; can supply event location but not role identity. Source positions contain names, not verified grammatical binding. Matched logit bias and token counts do not ensure matched attention-output change magnitudes; log those norms as descriptive covariates. No independent nonrelational retrieval task or native-circuit proof. All results exploratory, ML-088 models previously studied.",
        "safety": "Never instantiate previously protected ML-088 cases; use calibration pairs only. One-time output file, no overwrite.",
    }


def freeze(spec, create):
    final = json.loads(json.dumps({**spec, "sha256": digest(spec)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(final, indent=2) + "\n")
    if not PLAN.is_file():
        raise FileNotFoundError("Run --phase plan first")
    assert json.loads(PLAN.read_text()) == final, "Frozen ML-089 discovery design changed"
    return final["sha256"]


def attention_mask(cases, ro, n, device, dtype, mode=None, gain=LOG_GAIN):
    """Native causal mask or source-key bias at the final answer query."""
    b = len(cases)
    future = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
    if mode is None:
        return future
    bias = torch.zeros((b, n), device=device, dtype=dtype)
    for j, c in enumerate(cases):
        positions = source_positions(c)[mode]
        assert all(p <= int(ro[j]) for p in positions)
        bias[j, list(positions)] = gain
    additive = torch.zeros((b, n, n), device=device, dtype=dtype)
    additive.masked_fill_(future.unsqueeze(0), float("-inf"))
    additive[torch.arange(b, device=device), ro, :] += bias
    return additive.repeat_interleave(ml.HEADS, dim=0)


def forward(model, cases, device, site=None, mode=None, gain=LOG_GAIN, report_delta=False):
    """Use PyTorch's ORIGINAL MHA, modifying only an additive mask.

    No other-model states, extra answer tokens, or manual QKV changes.
    """
    ids, ro, gold, choices = ml.encode(cases)
    ids, ro = ids.to(device), ro.to(device)
    b, n = ids.shape
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    ix = torch.arange(b, device=device)
    delta = None
    for layer, block in enumerate(model.blocks):
        z = block.n1(x)
        bias_mode = mode if layer == site else None
        mask = attention_mask(cases, ro, n, device, z.dtype, bias_mode, gain)
        attn = block.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        if report_delta and bias_mode is not None:
            baseline_mask = attention_mask(cases, ro, n, device, z.dtype)
            baseline = block.attn(z, z, z, attn_mask=baseline_mask, need_weights=False)[0]
            delta = (attn[ix, ro] - baseline[ix, ro]).norm(dim=1).detach().cpu()
        x = x + attn
        x = x + block.mlp(block.n2(x))
    logits = model.out(model.norm(x[ix, ro])).float()
    if report_delta:
        assert delta is not None
        return logits.cpu(), gold, choices, delta
    return logits.cpu(), gold, choices


def stats(logits, gold, choices):
    logits = logits.float()
    native_cand = logits.gather(1, choices)
    col = (choices == gold[:, None]).long().argmax(1)
    assert bool((choices == gold[:, None]).sum(1).eq(1).all())
    correct = native_cand.gather(1, col[:, None]).flatten()
    runnerup = native_cand.masked_fill(choices == gold[:, None], -float("inf")).max(1).values
    chosen = choices.gather(1, native_cand.argmax(1, keepdim=True)).flatten()
    return {
        "margin": float((correct - runnerup).mean()),
        "choice_accuracy": float((chosen == gold).float().mean()),
        "full_accuracy": float((logits.argmax(1) == gold).float().mean()),
        "both_roles_accuracy": float((((chosen[::2] == gold[::2]) &
                                         (chosen[1::2] == gold[1::2])).float().mean())),
        "pred": chosen,
        "correct": chosen == gold,
    }


def collapse(scores):
    return {k: v for k, v in scores.items() if k not in ("pred", "correct")}


@torch.inference_mode()
def attention_mass(model, cases, device):
    """Descriptive attention mass; attention weight alone is not causal influence."""
    data = {layer: {key: [] for key in ("target_pair", "distractor_pair", "both_verbs")}
            for layer in LAYERS}
    for st in range(0, len(cases), BATCH):
        chunk = cases[st:st+BATCH]
        ids, ro, _, _ = ml.encode(chunk)
        ids, ro = ids.to(device), ro.to(device)
        b, n = ids.shape
        x = model.embed(ids) + model.pos(torch.arange(n, device=device))
        future = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
        ix = torch.arange(b, device=device)
        for layer, block in enumerate(model.blocks):
            z = block.n1(x)
            attn, weights = block.attn(z,z,z,attn_mask=future,
                                       need_weights=True, average_attn_weights=False)
            rows = weights[ix, :, ro, :]  # [batch,heads,source_position]
            for key in data[layer]:
                values = []
                for i, c in enumerate(chunk):
                    values.append(float(rows[i, :, list(source_positions(c)[key])].sum(0).mean()))
                data[layer][key].extend(values)
            x = x + attn
            x = x + block.mlp(block.n2(x))
    return {f"L{layer+1}": {key: mean(vals) for key, vals in d.items()}
            for layer,d in data.items()}


@torch.inference_mode()
def run(old_hash, h, examples, device):
    if REPORT.exists():
        raise FileExistsError(f"Refuse to repeat ML-089 discovery: {REPORT}")
    ok, counts = m88.model_gate(old_hash)
    if not ok:
        raise PermissionError(f"ML-088 model gate failed: {counts}")
    rows = []
    for seed in SEEDS:
        for arm in ml.ARMS:
            path = ROOT / f"seed_{seed}" / arm / "model_step_12800.pt"
            model = ml.load_checkpoint(path, seed, arm, old_hash, device)
            for construction, cases in examples.items():
                native_logits = []
                for start in range(0, len(cases), BATCH):
                    a = cases[start:start+BATCH]
                    n, _, _ = forward(model,a,device)
                    native_logits.append(n)
                ids, ro, gold, choice = ml.encode(cases)
                native_logits = torch.cat(native_logits)
                native = stats(native_logits, gold, choice)
                record = {"seed":seed,"arm":arm,"construction":construction,
                          "native":collapse(native),
                          "native_attention_mass": attention_mass(model,cases,device),
                          "interventions":{}}
                for layer in LAYERS:
                    layer_rec={}
                    for mode in MODES:
                        altered=[]
                        delta_norms=[]
                        for start in range(0,len(cases),BATCH):
                            logits, _, _, dnorm = forward(model,cases[start:start+BATCH],device,
                                                   site=layer,mode=mode,report_delta=True)
                            altered.append(logits)
                            delta_norms.extend(dnorm.tolist())
                        score=stats(torch.cat(altered),gold,choice)
                        before=native["correct"]
                        after=score["correct"]
                        rescued=int((~before & after).sum())
                        harmed=int((before & ~after).sum())
                        layer_rec[mode]={**collapse(score),
                            "mean_attn_output_delta_norm":mean(delta_norms),
                            "delta_margin":score["margin"]-native["margin"],
                            "delta_choice_accuracy":score["choice_accuracy"]-native["choice_accuracy"],
                            "rescued_from_native_errors":rescued,
                            "harmed_native_correct":harmed,
                            "native_error_count":int((~before).sum()),
                            "native_correct_count":int(before.sum())}
                    record["interventions"][f"L{layer+1}"]=layer_rec
                rows.append(record)
                p=record["interventions"]["L2"]
                contrast=p["target_pair"]["delta_margin"]-p["distractor_pair"]["delta_margin"]
                print(f"seed={seed} {arm:8s} {construction:8s} native={native['choice_accuracy']:.3f} "
                      f"L2_target={p['target_pair']['choice_accuracy']:.3f} "
                      f"L2_wrong={p['distractor_pair']['choice_accuracy']:.3f} "
                      f"selective_margin={contrast:+.3f}", flush=True)

    def get(seed,arm,construction):
        return next(r for r in rows if (r["seed"],r["arm"],r["construction"]) ==
                    (seed,arm,construction))
    summary={}
    for arm in ml.ARMS:
        for construction in ("target","familiar"):
            vals=[]
            for seed in SEEDS:
                r=get(seed,arm,construction)
                z=r["interventions"]["L2"]
                vals.append(z["target_pair"]["delta_margin"] -
                            z["distractor_pair"]["delta_margin"])
            summary[f"{arm}/{construction}"]={
                "L2_target_minus_wrong_event_margin_by_seed":dict(zip(map(str,SEEDS),vals)),
                "L2_mean_selective_margin":mean(vals),
                "positive_seeds":sum(v>0 for v in vals),
                "native_choice_accuracy_mean":mean(get(s,arm,construction)["native"]["choice_accuracy"] for s in SEEDS)}
    report={"experiment":"ML-089","status":"EXPLORATORY_DISCOVERY",
            "discovery_plan_sha256":h,"ml088_protocol_sha256":old_hash,
            "protected_cases_generated":False,"rows":rows,"summary":summary,
            "warning":"Queried-event selection is supplied by intervention; effect is not proof of native role-binding/routing."}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2)+"\n")
    print("\nML-089 EXPLORATORY SUMMARY\n"+json.dumps(summary,indent=2))
    print("Saved",REPORT)


def preflight(examples):
    torch.manual_seed(89089)
    model=ml.Model().eval()
    batch=examples["target"][:12]
    ids,ro,_,_=ml.encode(batch)
    with torch.inference_mode():
        direct=model(ids,ro).float()
        native, gold, choices=forward(model,batch,"cpu")
        zero,_,_=forward(model,batch,"cpu",site=1,mode="target_pair",gain=0.0)
        target,_,_=forward(model,batch,"cpu",site=1,mode="target_pair")
        wrong,_,_=forward(model,batch,"cpu",site=1,mode="distractor_pair")
        mass=attention_mass(model,batch,"cpu")
        # Verify that the intervention actually increases targeted attention
        # mass at the answer query. Check all attention heads and examples.
        b,n=ids.shape
        x=model.embed(ids)+model.pos(torch.arange(n))
        block0=model.blocks[0]
        z=block0.n1(x)
        a=block0.attn(z,z,z,attn_mask=attention_mask(batch,ro,n,"cpu",z.dtype),
                      need_weights=False)[0]
        x=x+a
        x=x+block0.mlp(block0.n2(x))
        block1=model.blocks[1]
        z=block1.n1(x)
        base_mask=attention_mask(batch,ro,n,"cpu",z.dtype)
        boost_mask=attention_mask(batch,ro,n,"cpu",z.dtype,"target_pair")
        _,w0=block1.attn(z,z,z,attn_mask=base_mask,need_weights=True,
                         average_attn_weights=False)
        _,w1=block1.attn(z,z,z,attn_mask=boost_mask,need_weights=True,
                         average_attn_weights=False)
        attention_mass_gains=[]
        for i,c in enumerate(batch):
            pos=list(source_positions(c)["target_pair"])
            old_mass=w0[i,:,int(ro[i]),pos].sum(0)
            new_mass=w1[i,:,int(ro[i]),pos].sum(0)
            # Depending on indexing, both values remain [heads].
            assert bool((new_mass > old_mass).all()), (old_mass,new_mass)
            attention_mass_gains.append(float((new_mass-old_mass).mean()))
    err=float((native-direct).abs().max())
    zero_err=float((zero-direct).abs().max())
    assert err < 1e-4 and zero_err < 1e-4,(err,zero_err)
    assert bool(torch.isfinite(target).all() and torch.isfinite(wrong).all())
    assert all(0 <= mass[layer][key] <= 1 for layer in mass for key in mass[layer])
    assert len(set(tuple(ml.prompt(c)) for c in batch))==len(batch)
    print(f"PREFLIGHT PASS: native parity={err:.2e}; zero-bias parity={zero_err:.2e}; "
          f"steered attention-mass gain={mean(attention_mass_gains):.3f}; "
          "96 paired events; 192 questions/construction")
    print("No trained-model inference, no protected-case generation, no discovery result observed.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("plan","preflight","run"),required=True)
    p.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    args=p.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old_hash,examples=prepare()
    h=freeze(specification(old_hash,examples),args.phase=="plan")
    if args.phase=="plan":
        print("Frozen exploratory plan",PLAN,"hash",h)
    elif args.phase=="preflight":
        preflight(examples)
    else:
        run(old_hash,h,examples,ml.select_device(args.device))


if __name__=="__main__":
    main()
