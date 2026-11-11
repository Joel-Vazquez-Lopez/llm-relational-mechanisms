"""ML-100: bounded source/access/output factorial diagnostic, exploratory.

Uses only existing ML-088 checkpoints and the already-explored ML-092
CALIBRATION sentence family; no new training, no protected-case generation.

From repo root:
  cp ~/Downloads/ml100_source_access_factorial.py scripts/
  python scripts/ml100_source_access_factorial.py --phase preflight
  python scripts/ml100_source_access_factorial.py --phase plan
  # Commit script + frozen plan BEFORE trained inference.
  python scripts/ml100_source_access_factorial.py --phase run --seed 701
  python scripts/ml100_source_access_factorial.py --phase run
  python scripts/ml100_source_access_factorial.py --phase summarize

Question: Under an active/passive same-meaning counterfactual, do transplanted
second-clause source K/V (source carrier), answer-position Q (access), or both
change the queried second-event answer more selectively than matched controls?

IMPORTANT: These cross-voice interventions are sufficiency / information-transfer
tests; they cannot by themselves locate the native computation, show necessity,
or rule out answer import (especially answer-position Q donors).
"""

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92

SEEDS = tuple(m88.SEEDS)
ARMS = ("familiar", "target")
VOICES = ("active", "passive")
LAYERS = (1, 2)  # zero-indexed, L2 and L3, only two prespecified sites
MODES = (
    "identity", "source_aligned", "source_reversed", "source_random",
    "query_aligned", "query_reversed", "query_random",
    "joint_aligned", "joint_reversed", "joint_random",
)
PLAN = Path("reports/ml100_source_access/frozen_plan.json")
ROOT = Path("results/ml088_fresh_models")
RUN_DIR = Path("reports/ml100_source_access/runs")
FINAL = Path("reports/ml100_source_access/exploratory_summary.json")
FIT_INDEX_MOD4 = 0  # 24 sentence templates, one per calibration pair
EVAL_INDEX_MOD4 = 1  # 24 different templates, one per calibration pair
BATCH_GROUPS = 6
BIAS_STEPS = 120
BIAS_LR = 0.10
BIAS_L2 = 0.05
RANDOM_SEED = 100100
MAX_PATCHED_RECEIVER_EVALUATIONS = 60000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def canonical(doc):
    return json.loads(json.dumps({**doc, "sha256": digest(doc)}))


def groups_for_study():
    prior_protocol, groups, previous = m92.select_cases()
    assert tuple(ml.SEEDS) == SEEDS and tuple(ml.ARMS) == ARMS
    assert all(len(groups[v]) == 96 for v in VOICES)
    fit = {v: [g for i, g in enumerate(groups[v]) if i % 4 == FIT_INDEX_MOD4]
           for v in VOICES}
    evaluation = {v: [g for i, g in enumerate(groups[v]) if i % 4 == EVAL_INDEX_MOD4]
                  for v in VOICES}
    assert all(len(fit[v]) == len(evaluation[v]) == 24 for v in VOICES)
    for voice in VOICES:
        sf = {m91.sentence(g[0]) for g in fit[voice]}
        se = {m91.sentence(g[0]) for g in evaluation[voice]}
        assert len(sf) == len(se) == 24 and not sf & se
        for g in fit[voice] + evaluation[voice]:
            assert len(g) == 4
            assert [q.target_pos for q in g] == [0, 0, 1, 1]
            assert [q.role for q in g] == ["agent", "patient", "agent", "patient"]
            assert len(set(m91.sentence(c) for c in g)) == 1
            assert len(set(ml.answer(c) for c in g)) == 4
    for i in range(24):
        assert [ml.answer(q) for q in evaluation["active"][i]] == [
            ml.answer(q) for q in evaluation["passive"][i]
        ]
    # Source family is reused from ML-092, not an untouched confirmation set.
    return prior_protocol, previous, fit, evaluation


def other_voice_groups(groups, reverse=False):
    result = []
    for g in groups:
        c = g[2]
        opposite = "passive" if c.target_voice == "active" else "active"
        d = replace(c, target_voice=opposite,
                    target_dir=1 - c.target_dir if reverse else c.target_dir)
        donor = m91.four_questions(d)
        if not reverse:
            assert [ml.answer(q) for q in donor] == [ml.answer(q) for q in g]
        else:
            assert [ml.answer(q) for q in donor[:2]] == [ml.answer(q) for q in g[:2]]
            assert [ml.answer(q) for q in donor[2:]] == [ml.answer(q) for q in g[2:]][::-1]
        assert len(set(m91.sentence(q) for q in donor)) == 1
        result.append(donor)
    return result


def specification(protocol, previous, fit, evaluation):
    expected_eval = len(SEEDS) * len(ARMS) * len(VOICES) * len(LAYERS) * (len(MODES)-1) * 24 * 4
    assert expected_eval <= MAX_PATCHED_RECEIVER_EVALUATIONS
    return {
        "experiment": "ML-100",
        "status": "BOUNDED_EXPLORATORY_EXISTING_ML088_COHORT",
        "script_sha256": sha(__file__),
        "dependencies_sha256": {
            name: sha(mod.__file__) for name, mod in
            (("ml080", ml), ("ml088", m88), ("ml091", m91), ("ml092", m92))
        },
        "model_protocol_sha256": protocol,
        "ml091_prior_plan_sha256": previous,
        "seeds": SEEDS, "arms": ARMS, "voices": VOICES,
        "layers_one_indexed": [x + 1 for x in LAYERS],
        "modes": MODES,
        "fit_template_offsets_mod4": [FIT_INDEX_MOD4],
        "evaluation_template_offsets_mod4": [EVAL_INDEX_MOD4],
        "unused_offsets_mod4": [2, 3],
        "n_fit_templates_per_voice": 24,
        "n_evaluation_templates_per_voice": 24,
        "prompt_hashes": {
            v: {"fit": digest([ml.prompt(q) for g in fit[v] for q in g]),
                "evaluation": digest([ml.prompt(q) for g in evaluation[v] for q in g])}
            for v in VOICES
        },
        "donor_rule": (
            "For each same-sentence four-question receiver packet, use the opposite "
            "voice with the same names, verb, direction, event/question labels, or "
            "a role-reversed opposite-voice donor. Align source name-token sites by name."
        ),
        "intervention": (
            "At a single block's pre-attention normalized Q/K/V projections, "
            "replace only K/V at second-event participant NAME tokens, "
            "only Q at answer position, or both. No other Q/K/V or residual changes "
            "at the injection site. Later blocks are computed normally."
        ),
        "norm_controls": (
            "Reversed donor and deterministic random displacement directions "
            "are matched to the aligned donor change norm per sample/site/head, "
            "separately for K, V and Q. When reverse-donor displacement is exactly "
            "zero at a site/head, use a seeded random fallback direction and report "
            "fallback prevalence. Never zero the residual stream."
        ),
        "output_comparator": (
            "Question-type-by-vocabulary additive logit bias, fitted with cross-entropy "
            "plus L2 on 24 separate calibration sentences/voice/model. No test labels "
            "used in fitting; 120 Adam steps lr=0.1 L2=0.05."
        ),
        "native_and_safety_controls": (
            "MHA exact-forward parity; identity donor no-op; same-clause first-event "
            "K/V control no-op; nontrivial projected perturbations; norm matching; "
            "four question types; four-choice and full-vocab results."
        ),
        "primary_descriptive_scope": (
            "Second-event passive questions, all 8 familiar/target paired seeds; "
            "report exact wrong-event and wrong-role to correct, harm, donor-vs-control "
            "mean gold-margin effects; preserve first-event accuracy."
        ),
        "candidate_rule_exploratory_not_confirmatory": (
            "For each of source-only, Q-only and joint at either L2/L3 in target/passive, "
            "aligned second-event accuracy must exceed reversed, matched-random and "
            "output-bias controls (and for joint, each aligned single-site) by >=0.05 "
            "mean; advantage positive in >=6/8 seeds and seed-bootstrap 95pct "
            "lower >0; aligned wrong-to-correct rate >=0.05; native-correct first-event "
            "harm <=0.02. Even if a candidate "
            "passes, its result is exploratory and donor answer import remains possible."
        ),
        "stopping_rule": (
            "Inspect exactly 2 layers x 3 mechanism families; do not sweep additional "
            "sites, donors, doses, examples or seeds after seeing results. If no clean "
            "candidate, stop mechanistic search for this paper."
        ),
        "patched_receiver_evaluations_upper_bound": expected_eval,
        "limitations": (
            "Reused ML-088 cohort and previously examined ML-092 calibration family; "
            "cross-voice donors differ in length/position; Q donor can import "
            "answer information; null result doesn't exclude information elsewhere. "
            "No claim of native necessity or independent confirmation."
        ),
        "protected": "NEVER READ/GENERATE ML-088 OR ML-099 PROTECTED EXAMPLES",
    }


def frozen(doc, create):
    expected = canonical(doc)  # JSON normalization avoids ML-098 tuple/list bug.
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        with PLAN.open("x") as f:
            json.dump(expected, f, indent=2)
            f.write("\n")
    if not PLAN.is_file():
        raise FileNotFoundError(f"First freeze protocol: {PLAN}")
    assert json.loads(PLAN.read_text()) == expected, "Frozen source/data/protocol mismatch"
    return expected["sha256"]


def encode_groups(groups, device):
    cases = [q for g in groups for q in g]
    x, ro, _, _ = ml.encode(cases)
    return x.to(device), ro.to(device)


def starting_residual(model, ids):
    _, n = ids.shape
    return model.embed(ids) + model.pos(torch.arange(n, device=ids.device))


def projections(block, x):
    b, n, w = x.shape
    heads = block.attn.num_heads
    assert w % heads == 0
    z = block.n1(x)
    q, k, v = F.linear(z, block.attn.in_proj_weight,
                        block.attn.in_proj_bias).chunk(3, dim=-1)
    return tuple(t.reshape(b, n, heads, w // heads).contiguous() for t in (q,k,v))


def token_sites(groups, donor_groups, event, device):
    """Return participant positions [4*n_groups,2], aligned by NAME."""
    dest, src = [], []
    for g, dg in zip(groups, donor_groups):
        c, d = g[2], dg[2]
        names = m92.participants(c, event)
        assert set(names) == set(m92.participants(d, event))
        dest.append([m92.source_token_pos(c, name) for name in names])
        src.append([m92.source_token_pos(d, name) for name in names])
    return (torch.tensor(dest, dtype=torch.long, device=device).repeat_interleave(4, 0),
            torch.tensor(src, dtype=torch.long, device=device).repeat_interleave(4, 0))


def deterministic_direction(shape, device, salt):
    # Explicit CPU PRNG gives reproducible controls across CPU/MPS/CUDA.
    rng = torch.Generator(device="cpu").manual_seed(RANDOM_SEED + salt)
    return torch.randn(shape, generator=rng, dtype=torch.float32).to(device)


def direction_with_norm(current, reference_delta, donor_delta=None, random_salt=None):
    """Match aligned displacement norms PER HEAD (last tensor dimension)."""
    target = reference_delta.float().norm(dim=-1, keepdim=True)
    if random_salt is not None:
        vector = deterministic_direction(current.shape, current.device, random_salt)
    else:
        assert donor_delta is not None
        vector = donor_delta.float()
    vn = vector.norm(dim=-1, keepdim=True)
    # Sometimes the reversed donor is exactly identical at an early causal
    # name token. There is no semantic direction to match. Substitute a
    # deterministic random direction at those individual site/head cells;
    # report fallback fractions, not an illusory semantic contrast.
    if donor_delta is not None:
        missing = (vn < 1e-9) & (target > 1e-7)
        if bool(missing.any()):
            fallback = deterministic_direction(current.shape, current.device, 501)
            vector = torch.where(missing, fallback, vector)
            vn = vector.norm(dim=-1, keepdim=True)
    step = vector * (target / vn.clamp_min(1e-12))
    return current + step.to(current.dtype)


def modified_block(block, x, xa, xr, ro, roa, ror,
                   dst, srca, srcr, mode,
                   first_sites=None):
    q,k,v = [t.clone() for t in projections(block,x)]
    qa,ka,va = projections(block,xa)
    qr,kr,vr = projections(block,xr)
    b,n,h,d = q.shape
    ix = torch.arange(b,device=x.device)
    info={"source_K":0.,"source_V":0.,"answer_Q":0.,"reverse_fallback_fraction":0.}
    source_modes={"source_aligned","source_reversed","source_random",
                  "joint_aligned","joint_reversed","joint_random","first_control"}
    query_modes={"query_aligned","query_reversed","query_random",
                 "joint_aligned","joint_reversed","joint_random"}
    if mode in source_modes:
        if mode=="first_control":
            assert first_sites is not None
            dest, a, r = first_sites
        else:
            dest,a,r=dst,srca,srcr
        fallbacks=[]
        for key, dg, dr, metric, salt in (
                (k,ka,kr,"source_K",11), (v,va,vr,"source_V",12)):
            old=key[ix[:,None],dest]
            good=dg[ix[:,None],a]
            reversed_=dr[ix[:,None],r]
            delta=good-old
            if mode in ("source_reversed","joint_reversed"):
                fallbacks.append(float((((reversed_-old).float().norm(dim=-1) < 1e-9)
                         & (delta.float().norm(dim=-1) > 1e-7)).float().mean()))
            if mode in ("source_aligned","joint_aligned","first_control"):
                new=good
            elif mode in ("source_reversed","joint_reversed"):
                new=direction_with_norm(old,delta,reversed_-old)
            else:
                new=direction_with_norm(old,delta,random_salt=salt)
            key[ix[:,None],dest]=new
            info[metric]=float((new-old).float().norm(dim=-1).mean())
            if mode in ("source_reversed","source_random","joint_reversed","joint_random"):
                assert float(((new-old).float().norm(dim=-1)-
                              delta.float().norm(dim=-1)).abs().max())<2e-4
        if fallbacks:
            info["reverse_fallback_fraction"] = mean(fallbacks)
    if mode in query_modes:
        old=q[ix,ro]
        good=qa[ix,roa]
        reversed_=qr[ix,ror]
        delta=good-old
        if mode in ("query_reversed","joint_reversed"):
            qfallback=float((((reversed_-old).float().norm(dim=-1)<1e-9)
                     & (delta.float().norm(dim=-1)>1e-7)).float().mean())
            info["reverse_fallback_fraction"] = max(info["reverse_fallback_fraction"],qfallback)
        if mode in ("query_aligned","joint_aligned"):
            new=good
        elif mode in ("query_reversed","joint_reversed"):
            new=direction_with_norm(old,delta,reversed_-old)
        else:
            new=direction_with_norm(old,delta,random_salt=13)
        q[ix,ro]=new
        info["answer_Q"]=float((new-old).float().norm(dim=-1).mean())
        if mode in ("query_reversed","query_random","joint_reversed","joint_random"):
            assert float(((new-old).float().norm(dim=-1)-
                          delta.float().norm(dim=-1)).abs().max())<2e-4
    if mode not in MODES and mode!="first_control":
        raise ValueError(mode)
    attn=F.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        dropout_p=0.,is_causal=True)
    out=attn.transpose(1,2).contiguous().reshape(b,n,h*d)
    out=F.linear(out,block.attn.out_proj.weight,block.attn.out_proj.bias)
    y=x+out
    return y+block.mlp(block.n2(y)),info


@torch.inference_mode()
def forward_modes(model, groups, layer, device, modes=MODES):
    """Every mode sees identical receiver input; donor prior states are native."""
    aligned=other_voice_groups(groups,False)
    reversed_=other_voice_groups(groups,True)
    ids,ro=encode_groups(groups,device)
    ida,roa=encode_groups(aligned,device)
    idr,ror=encode_groups(reversed_,device)
    x=starting_residual(model,ids)
    xa=starting_residual(model,ida)
    xr=starting_residual(model,idr)
    for j in range(layer):
        x=model.blocks[j](x,torch.ones(x.shape[1],x.shape[1],device=device,dtype=torch.bool).triu(1))
        xa=model.blocks[j](xa,torch.ones(xa.shape[1],xa.shape[1],device=device,dtype=torch.bool).triu(1))
        xr=model.blocks[j](xr,torch.ones(xr.shape[1],xr.shape[1],device=device,dtype=torch.bool).triu(1))
    dst,srca=token_sites(groups,aligned,"second",device)
    _,srcr=token_sites(groups,reversed_,"second",device)
    first_dst,first_srca=token_sites(groups,aligned,"first",device)
    _,first_srcr=token_sites(groups,reversed_,"first",device)
    output={}
    for mode in modes:
        y,info=modified_block(model.blocks[layer],x,xa,xr,ro,roa,ror,
                              dst,srca,srcr,mode,
                              first_sites=(first_dst,first_srca,first_srcr))
        for j in range(layer+1,len(model.blocks)):
            mask=torch.ones(y.shape[1],y.shape[1],device=device,dtype=torch.bool).triu(1)
            y=model.blocks[j](y,mask)
        ix=torch.arange(y.shape[0],device=device)
        logits=model.out(model.norm(y[ix,ro])).float().cpu()
        output[mode]=(logits,info)
    return output


@torch.inference_mode()
def native(model,groups,device):
    ids,ro=encode_groups(groups,device)
    return model(ids,ro).float().cpu()


def fit_output_bias(fit_logits,groups):
    """Fit only on independent fit cases; freeze before evaluation logits."""
    n=len(groups)
    vocab=fit_logits.shape[1]
    names=torch.tensor([m91.name_ids(g) for g in groups],dtype=torch.long)
    choices=names[:,None,:].expand(n,4,4).reshape(n*4,4)
    queries=torch.arange(n*4)%4
    bias=torch.nn.Parameter(torch.zeros(4,vocab,dtype=torch.float32))
    optim=torch.optim.Adam([bias],lr=BIAS_LR)
    fit_logits=fit_logits.detach().cpu()
    torch.manual_seed(RANDOM_SEED)
    for _ in range(BIAS_STEPS):
        scores=(fit_logits+bias[queries]).gather(1,choices)
        loss=F.cross_entropy(scores,queries)+BIAS_L2*bias.square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    return bias.detach().clone()


def apply_output_bias(logits,bias):
    query=torch.arange(logits.shape[0])%4
    return logits+bias[query]


def evaluate_metric(native_logits,edited,groups,info=None):
    n=len(groups)
    vocab=native_logits.shape[1]
    gold=torch.tensor([m91.name_ids(g) for g in groups],dtype=torch.long)
    choices=gold[:,None,:].expand(n,4,4).reshape(n*4,4)
    query=torch.arange(n*4)
    label=query%4
    a=native_logits.gather(1,choices)
    b=edited.gather(1,choices)
    pred0=a.argmax(1)
    pred1=b.argmax(1)
    success0=pred0==label
    success1=pred1==label
    second=label>=2
    first=~second
    wrong_event0=(pred0//2)!=(label//2)
    wrong_role0=((pred0%2)!=(label%2))&~wrong_event0
    new_correct=(~success0)&success1
    lost_correct=success0&~success1
    def ratio(mask,sel=None):
        if sel is None: sel=torch.ones_like(second)
        return float((mask&sel).float().sum()/sel.float().sum())
    idx=torch.arange(n*4)
    margin0=a[idx,label]-a.masked_fill(F.one_hot(label,4).bool(),float("-inf")).max(1).values
    margin1=b[idx,label]-b.masked_fill(F.one_hot(label,4).bool(),float("-inf")).max(1).values
    full_gold=choices[idx,label]
    return {
        "four_accuracy":ratio(success1),
        "full_vocab_accuracy":float((edited.argmax(1)==full_gold).float().mean()),
        "first_accuracy":ratio(success1,first),
        "second_accuracy":ratio(success1,second),
        "first_correct_to_wrong":ratio(lost_correct,first),
        "second_wrong_to_correct":ratio(new_correct,second),
        "second_correct_to_wrong":ratio(lost_correct,second),
        "second_wrong_event_to_correct":ratio(wrong_event0&success1,second),
        "second_wrong_role_to_correct":ratio(wrong_role0&success1,second),
        "second_correct_to_wrong_event":ratio(success0&((pred1//2)!=(label//2)),second),
        "second_correct_to_wrong_role":ratio(success0&((pred1%2)!=(label%2))&((pred1//2)==(label//2)),second),
        "second_margin_shift":float((margin1[second]-margin0[second]).mean()),
        "first_margin_shift":float((margin1[first]-margin0[first]).mean()),
        "logit_change_l2":float((edited-native_logits).norm(dim=-1).mean()),
        "first_both_correct":float(success1.reshape(n,4)[:,:2].all(dim=1).float().mean()),
        "second_both_correct":float(success1.reshape(n,4)[:,2:].all(dim=1).float().mean()),
        "all_four_correct":float(success1.reshape(n,4).all(dim=1).float().mean()),
        "source_K_change":(info or {}).get("source_K",0.),
        "source_V_change":(info or {}).get("source_V",0.),
        "answer_Q_change":(info or {}).get("answer_Q",0.),
        "reverse_fallback_fraction":(info or {}).get("reverse_fallback_fraction",0.),
    }


def bootstrap(xs):
    r=random.Random(RANDOM_SEED)
    ds=sorted(mean(xs[r.randrange(len(xs))] for _ in xs) for _ in range(20000))
    return [ds[500],ds[19499]]


@torch.inference_mode()
def preflight(fit,evaluation):
    torch.manual_seed(RANDOM_SEED)
    model=ml.Model().eval()
    for voice in VOICES:
        groups=evaluation[voice][:2]
        direct=native(model,groups,"cpu")
        for layer in LAYERS:
            outputs=forward_modes(model,groups,layer,"cpu",
                                  modes=MODES+("first_control",))
            assert float((outputs["identity"][0]-direct).abs().max())<3e-4
            assert float((outputs["first_control"][0]-direct).abs().max())<3e-4
            for mode in MODES:
                logits,info=outputs[mode]
                assert bool(torch.isfinite(logits).all()),(voice,layer,mode)
                assert all(np.isfinite(v) for v in info.values())
            delta=max(float((outputs[m][0]-direct).abs().max())
                      for m in ("source_aligned","query_aligned","joint_aligned"))
            assert delta>1e-6,("Intervention not reaching output",voice,layer)
    # Metric sanity on a synthetic 4-query packet: no oracle mapping ambiguity.
    g=evaluation["passive"][:2]
    z=native(model,g,"cpu")
    res=evaluate_metric(z,z,g)
    assert res["second_wrong_to_correct"]==res["second_correct_to_wrong"]==0
    assert res["logit_change_l2"]==0
    print("ML-100 PREFLIGHT PASS: native parity, identity and first-clause null,")
    print("  nontrivial L2/L3 Q and K/V interventions, norm-match, finite logits,")
    print("  four-question alignment, 24 fit/24 evaluation templates per voice.")
    print("  Protected examples and trained checkpoints NOT loaded.")


@torch.inference_mode()
def evaluate_model(model,fit,evaluation,device):
    results={}
    arrays={}
    for voice in VOICES:
        groups=fit[voice]
        fitlog=torch.cat([native(model,groups[i:i+BATCH_GROUPS],device)
                          for i in range(0,len(groups),BATCH_GROUPS)])
        # fit_output_bias must run with autograd enabled, not under inference_mode.
        with torch.inference_mode(False),torch.enable_grad():
            fitted_bias=fit_output_bias(fitlog.clone(),groups)
        evalgroups=evaluation[voice]
        base=torch.cat([native(model,evalgroups[i:i+BATCH_GROUPS],device)
                        for i in range(0,len(evalgroups),BATCH_GROUPS)])
        arrays[f"{voice}/native"]=base.numpy()
        biaslog=apply_output_bias(base,fitted_bias)
        arrays[f"{voice}/output_bias"]=biaslog.numpy()
        base_metrics=evaluate_metric(base,base,evalgroups)
        bias_metrics=evaluate_metric(base,biaslog,evalgroups)
        results[voice]={"native":base_metrics,"output_bias":bias_metrics,"layers":{}}
        for layer in LAYERS:
            pieces={m:[] for m in MODES}
            infos={m:[] for m in MODES}
            for i in range(0,len(evalgroups),BATCH_GROUPS):
                chunk=evalgroups[i:i+BATCH_GROUPS]
                outputs=forward_modes(model,chunk,layer,device)
                for mode,(logits,info) in outputs.items():
                    pieces[mode].append(logits)
                    infos[mode].append(info)
            metrics={}
            for mode in MODES:
                logits=torch.cat(pieces[mode])
                assert logits.shape==base.shape
                inf={k:mean(x[k] for x in infos[mode]) for k in infos[mode][0]}
                metrics[mode]=evaluate_metric(base,logits,evalgroups,inf)
                arrays[f"{voice}/L{layer+1}/{mode}"]=logits.numpy()
            assert metrics["identity"]["logit_change_l2"]<0.001
            results[voice]["layers"][f"L{layer+1}"]=metrics
    return results,arrays


def run(protocol_hash,model_protocol,fit,evaluation,device,seed=None,arm=None):
    gate,counts=m88.model_gate(model_protocol)
    if not gate:
        raise RuntimeError(f"ML-088 checkpoint competence gate failed: {counts}")
    RUN_DIR.mkdir(parents=True,exist_ok=True)
    for s in (SEEDS if seed is None else (seed,)):
        for a in (ARMS if arm is None else (arm,)):
            path=RUN_DIR/f"seed_{s}_{a}.json"
            archive=RUN_DIR/f"seed_{s}_{a}_logits.npz"
            if path.exists() and archive.exists():
                existing=json.loads(path.read_text())
                assert existing["plan_sha256"]==protocol_hash
                print(f"seed={s} arm={a} verified existing record; skipped")
                continue
            if path.exists() or archive.exists():
                raise RuntimeError(f"Incomplete existing pair: {path} / {archive}; inspect manually")
            checkpoint=ROOT/f"seed_{s}"/a/"model_step_12800.pt"
            model=ml.load_checkpoint(checkpoint,s,a,model_protocol,device)
            metrics,arrays=evaluate_model(model,fit,evaluation,device)
            # Raw per-example logits are archived to permit independent analysis.
            with archive.open("xb") as fh:
                np.savez_compressed(fh,**arrays)
            doc={"experiment":"ML-100","status":"EXPLORATORY","plan_sha256":protocol_hash,
                 "seed":s,"arm":a,"raw_logits_file":str(archive),"metrics":metrics,
                 "raw_logits_sha256":sha(archive)}
            with path.open("x") as fh:
                json.dump(doc,fh,indent=2);fh.write("\n")
            print(f"seed={s} arm={a:8} passive_native_second="
                  f"{metrics['passive']['native']['second_accuracy']:.3f} "
                  f"source_L2="
                  f"{metrics['passive']['layers']['L2']['source_aligned']['second_accuracy']:.3f} "
                  f"query_L2="
                  f"{metrics['passive']['layers']['L2']['query_aligned']['second_accuracy']:.3f}",
                  flush=True)


def summary_one(rows,voice,layer,mode,key):
    return [r["metrics"][voice]["layers"][layer][mode][key] for r in rows]


def summarize(plan_hash):
    if FINAL.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {FINAL}")
    rows=[]
    for s in SEEDS:
        for a in ARMS:
            path=RUN_DIR/f"seed_{s}_{a}.json"
            doc=json.loads(path.read_text())
            assert (doc["plan_sha256"],doc["seed"],doc["arm"])==(plan_hash,s,a)
            assert sha(doc["raw_logits_file"])==doc["raw_logits_sha256"]
            rows.append(doc)
    def by_arm(arm): return [r for r in rows if r["arm"]==arm]
    metrics={}
    discoveries=[]
    for arm in ARMS:
        subset=by_arm(arm)
        for voice in VOICES:
            for layer in ("L2","L3"):
                groupkey=f"{arm}/{voice}/{layer}"
                stats={}
                for mode in MODES:
                    stats[mode]={key:mean(summary_one(subset,voice,layer,mode,key))
                      for key in ("second_accuracy","first_accuracy","second_wrong_to_correct",
                                  "second_correct_to_wrong","second_wrong_event_to_correct",
                                  "second_wrong_role_to_correct","first_correct_to_wrong",
                                  "second_margin_shift","source_K_change","source_V_change",
                                  "answer_Q_change","reverse_fallback_fraction", "logit_change_l2")}
                metrics[groupkey]=stats
    # Fixed exploratory decision rule: ONLY these six candidate site/family pairs.
    for layer in ("L2","L3"):
        for family in ("source","query","joint"):
            subset=by_arm("target")
            key="second_accuracy"
            mode=f"{family}_aligned"
            reversed_mode=f"{family}_reversed"
            aligned=summary_one(subset,"passive",layer,mode,key)
            rev=summary_one(subset,"passive",layer,reversed_mode,key)
            native_acc=[r["metrics"]["passive"]["native"][key] for r in subset]
            rand=summary_one(subset,"passive",layer,f"{family}_random",key)
            output_bias=[r["metrics"]["passive"]["output_bias"][key] for r in subset]
            if family != "joint":
                advantage=[x-max(y,z,o) for x,y,z,o in zip(aligned,rev,rand,output_bias)]
            else:
                source=summary_one(subset,"passive",layer,"source_aligned",key)
                query=summary_one(subset,"passive",layer,"query_aligned",key)
                advantage=[x-max(y,z,o,u,w) for x,y,z,o,u,w in
                           zip(aligned,rev,rand,output_bias,source,query)]
            rescue=summary_one(subset,"passive",layer,mode,"second_wrong_to_correct")
            harm=summary_one(subset,"passive",layer,mode,"first_correct_to_wrong")
            ci=bootstrap(advantage)
            passed=(mean(advantage)>=0.05 and sum(x>0 for x in advantage)>=6
                    and ci[0]>0 and mean(rescue)>=0.05 and mean(harm)<=0.02)
            discoveries.append({"site":layer,"family":family,"mean_advantage_vs_controls":mean(advantage),
                                "seed_advantages":dict(zip(map(str,SEEDS),advantage)),
                                "bootstrap_95":ci,"positive_seeds":sum(x>0 for x in advantage),
                                "mean_second_wrong_to_correct":mean(rescue),
                                "mean_first_correct_to_wrong":mean(harm),
                                "mean_native_second_accuracy":mean(native_acc),
                                "candidate_rule_passed":passed})
    document={"experiment":"ML-100","status":"EXPLORATORY_DIAGNOSTIC",
              "plan_sha256":plan_hash,"n_models":len(rows),
              "n_eval_sentences_per_voice":24,"n_fit_sentences_per_voice":24,
              "metrics":metrics,"predeclared_candidate_tests":discoveries,
              "any_candidate_passed":any(d["candidate_rule_passed"] for d in discoveries),
              "interpretation":(
                  "A passing exploratory screen is NOT replication or causal necessity. "
                  "Q donors may import answer information and source donors differ in "
                  "token history. If no screen passes, STOP searching this mechanistic "
                  "branch for this paper; do not add ML-101 variants."
              )}
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    with FINAL.open("x") as fh:
        json.dump(document,fh,indent=2);fh.write("\n")
    print("\nML-100 FIXED EXPLORATORY CANDIDATE SCREEN")
    for d in discoveries:
        print(f"  {d['site']} {d['family']:6} advantage={d['mean_advantage_vs_controls']:+.3f} "
              f"positive={d['positive_seeds']}/8 ci={d['bootstrap_95']} "
              f"first_harm={d['mean_first_correct_to_wrong']:.3f} "
              f"screen={'PASS' if d['candidate_rule_passed'] else 'FAIL'}")
    print("At least one exploratory candidate:",document["any_candidate_passed"])
    print("Saved:",FINAL)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,choices=("preflight","plan","run","summarize"))
    ap.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    ap.add_argument("--seed",type=int,choices=SEEDS)
    ap.add_argument("--arm",choices=ARMS)
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    previous,ml091_plan,fit,evaluation=groups_for_study()
    if args.phase=="preflight":
        preflight(fit,evaluation)
        return
    doc=specification(previous,ml091_plan,fit,evaluation)
    h=frozen(doc,args.phase=="plan")
    if args.phase=="plan":
        print("ML-100 frozen protocol SHA256:",h)
        print("Commit source and frozen_plan.json before trained inference.")
        return
    if args.phase=="run":
        run(h,previous,fit,evaluation,ml.select_device(args.device),args.seed,args.arm)
        return
    summarize(h)


if __name__=="__main__":
    main()
