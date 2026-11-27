#!/usr/bin/env python3
"""ROLE-ANCHOR-PROBE-v0: read-only post-hoc diagnostic of trained-role anchoring.

Reuse the frozen TRAIN-SURFACE-FACTORIAL-v0 48 checkpoints and frozen
CROSS-EVAL source sample. NEVER train, alter checkpoints, open original
development/reserved/protected/L+I, or select seeds/checkpoints.
"""
from __future__ import annotations
import argparse,json,math,subprocess
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean,stdev

import torch

import train_surface_factorial_v0 as parent
import train_surface_cross_eval_v0 as crossed
import train_surface_cross_eval_v0_preflight as pre

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/train_role_anchor_probe_v0"
SCRIPT=Path(__file__).resolve()
PROTOCOL=ROOT/"paper/ROLE_ANCHOR_PROBE_V0_PROTOCOL.md"
SEEDS=parent.SEEDS
ARMS=parent.ARMS
VOICES=pre.VOICE
SURFACES=pre.CONDITIONS
SOURCE_POP="unseen_roles"
T95=2.200985160082949
assert len(SEEDS)==12 and len(ARMS)==4

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def save_once(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(record,f,indent=2,sort_keys=True);f.write("\n")

def fixtures():
    sources,meta,vocab,parent_study=crossed.fixtures()
    train_ids,_held,original,_info=pre.sample_ids()
    assert len(train_ids)==pre.N_TRAIN_QUARTETS==24
    assert len(sources[SOURCE_POP]["bare"][VOICES[0]])==24*16
    gold_orig={}
    for ix in train_ids:
        events=original[ix]
        gold_orig[ix]=tuple(n for e in events for n in (e.agent,e.patient))
        assert len(set(gold_orig[ix]))==4
    orig_role_ids=[]
    gold_role_ids=[]
    for ix in train_ids:
        original_tuple=tuple(vocab[name] for name in gold_orig[ix])
        _train8,never_train=pre.old.previous.all_roles(gold_orig[ix])
        for roles in never_train:
            orig_role_ids.append(original_tuple)
            gold_role_ids.append(tuple(vocab[name] for name in roles))
    for v in VOICES:
        for s in SURFACES:
            original_sources=sources[SOURCE_POP][s][v]
            assert len(original_sources)==len(orig_role_ids)
            assert all(
                tuple(vocab[name] for name in src.gold)==gold_role_ids[k]
                for k,src in enumerate(original_sources))
    original_tensor=torch.tensor(orig_role_ids,dtype=torch.long)
    gold_tensor=torch.tensor(gold_role_ids,dtype=torch.long)
    assert not torch.any(torch.all(original_tensor==gold_tensor,dim=1))
    # Include the 16 withheld role assignment permutations; no exposure contamination.
    baseline={
      "original_mapping_never_equals_true_four_roles":True,
      "original_mapping_single_answer_accuracy":float((original_tensor==gold_tensor).float().mean()),
      "original_mapping_cquery":0.,
      "original_mapping_event_name_membership":None
    }
    return sources,meta,vocab,parent_study,original_tensor,gold_tensor,baseline

def frozen_inputs():
    frozen_paths=(
      "scripts/train_role_anchor_probe_v0.py",
      "paper/ROLE_ANCHOR_PROBE_V0_PROTOCOL.md")
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        for relative in frozen_paths:
            committed=subprocess.check_output(
                ["git","show",head+":"+relative],cwd=ROOT,stderr=subprocess.DEVNULL)
            if committed!=(ROOT/relative).read_bytes():
                raise RuntimeError("Uncommitted role anchor input: "+relative)
        return head
    except (FileNotFoundError,subprocess.CalledProcessError) as e:
        raise RuntimeError("Commit source and protocol before any old checkpoint scoring") from e

def analyze(pred,truth,anchor):
    assert pred.shape==truth.shape==anchor.shape
    assert pred.ndim==2 and pred.shape[1]==4 and pred.shape[0]==24*16
    exact=(pred==truth).all(1)
    anchored=(pred==anchor).all(1)
    assert not torch.any(exact & anchored)
    # Which ORIGINAL EVENT does the predicted participant belong to,
    # ignoring its original agent/patient role within that event?
    event_membership=[]
    for i in (0,2):
        first_pair=anchor[:,0:2] if i==0 else anchor[:,2:4]
        for q in (i,i+1):
            event_membership.append((pred[:,q:q+1]==first_pair).any(1))
    membership=torch.stack(event_membership,dim=1)
    modal=[];unique=[];anchored_mode=[]
    for k in range(24):
        sub=pred[16*k:16*(k+1)]
        counts=Counter(tuple(map(int,tuple(row))) for row in sub)
        value,count=sorted(counts.items(),key=lambda item:(-item[1],item[0]))[0]
        modal.append(count/16)
        unique.append(len(counts))
        anchored_mode.append(value==tuple(map(int,anchor[16*k])))
    return {
      "n_sources":len(pred),
      "strict_cquery":float(exact.float().mean()),
      "accuracy4":float((pred==truth).float().mean()),
      "matches_original_role_tuple_exact":float(anchored.float().mean()),
      "matches_original_role_slots":float((pred==anchor).float().mean()),
      "answers_in_original_event_pair":float(membership.float().mean()),
      "per_quartet_modal_prediction_fraction":float(mean(modal)),
      "per_quartet_unique_prediction_tuples":float(mean(unique)),
      "quartets_whose_modal_prediction_is_original_tuple":int(sum(anchored_mode)),
      "number_exactly_original":int(anchored.int().sum()),
      "number_exactly_gold":int(exact.int().sum())
    }

def smoke(device):
    sources,meta,vocab,study,anchor,gold,baseline=fixtures()
    _=meta,study,baseline
    torch.manual_seed(17002026)  # NOT any of the frozen 12 model seeds.
    net=parent.Model(len(vocab)).to(device)
    items=sources[SOURCE_POP]["bare"][VOICES[0]]
    _,_,pred=crossed.score(net,items,vocab,device,chunk_size=128)
    result=analyze(pred,gold,anchor)
    assert result["n_sources"]==24*16
    assert 0<=result["matches_original_role_slots"]<=1
    assert math.isfinite(result["per_quartet_modal_prediction_fraction"])
    print("SMOKE PASS: random model only; no checkpoints or original DEV/reserved/protected data")

def run(seed,device):
    if seed not in SEEDS:raise ValueError("Not an original frozen seed")
    out=OUT/("seed%d_role_anchor.json"%seed)
    if out.exists():raise FileExistsError("Existing immutable seed audit: "+str(out))
    sources,source_meta,vocab,study,anchor,gold,baseline=fixtures()
    diagnostic_commit=frozen_inputs()
    checkpoint=parent.WEIGHTS/("seed%d_mps_step6400.pt"%seed)
    if not checkpoint.exists():
        raise FileNotFoundError("Read-only original checkpoint missing; DO NOT retrain "+str(checkpoint))
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["study"]=="TRAIN-SURFACE-FACTORIAL-v0"
    assert ck["seed"]==seed and ck["steps"]==6400
    assert ck["protocol_digest"]==study["protocol_digest"]
    assert set(ck["models"])==set(ARMS)
    stats={}
    for arm in ARMS:
        model=parent.Model(len(vocab)).to(device)
        model.load_state_dict(ck["models"][arm],strict=True)
        model.eval()
        stats[arm]={}
        for surface in SURFACES:
            stats[arm][surface]={}
            for voice in VOICES:
                _metrics,_flags,pred=crossed.score(
                   model,sources[SOURCE_POP][surface][voice],vocab,device)
                stats[arm][surface][voice]=analyze(pred,gold,anchor)
        del model
        print("READ-ONLY",seed,arm,flush=True)
    save_once(out,{
       "study":"ROLE-ANCHOR-PROBE-v0",
       "status":"SEALED_READ_ONLY_EXPLORATORY",
       "seed":seed,
       "diagnostic_commit":diagnostic_commit,
       "source_manifest_sha256":sha(pre.MANIFEST),
       "parent_protocol_digest":study["protocol_digest"],
       "original_checkpoint_sha256":sha(checkpoint),
       "original_training_mapping_baseline":baseline,
       "no_new_training":True,
       "no_original_dev_reserved_protected_LI":True,
       "results":stats
    })
    print("SEALED",out,"No intermediate aggregation")

def descriptive(values):
    assert len(values)==12
    avg=mean(values)
    rad=T95*stdev(values)/math.sqrt(12)
    return {"mean":avg,"exploratory_unadjusted_t95":[avg-rad,avg+rad],
            "min":min(values),"max":max(values),
            "per_seed":dict(zip(map(str,SEEDS),values))}

def summary():
    sources,meta,vocab,study,anchor,gold,baseline=fixtures()
    _=sources,meta,vocab,anchor,gold
    raw={}
    for seed in SEEDS:
        path=OUT/("seed%d_role_anchor.json"%seed)
        if not path.exists():
            print("INCOMPLETE: missing",seed,"; no partial analysis")
            return
        report=json.loads(path.read_text())
        assert report["study"]=="ROLE-ANCHOR-PROBE-v0"
        assert report["seed"]==seed and report["no_new_training"]
        assert report["no_original_dev_reserved_protected_LI"]
        assert report["source_manifest_sha256"]==sha(pre.MANIFEST)
        assert report["parent_protocol_digest"]==study["protocol_digest"]
        assert set(report["results"])==set(ARMS)
        raw[seed]=report
    keys=("strict_cquery","accuracy4","matches_original_role_tuple_exact",
          "matches_original_role_slots","answers_in_original_event_pair",
          "per_quartet_modal_prediction_fraction",
          "per_quartet_unique_prediction_tuples",
          "quartets_whose_modal_prediction_is_original_tuple")
    values={}
    for arm in ARMS:
        values[arm]={}
        for surface in SURFACES:
            values[arm][surface]={}
            for key in keys:
                vals=[mean(raw[seed]["results"][arm][surface][v][key] for v in VOICES)
                      for seed in SEEDS]
                values[arm][surface][key]=descriptive(vals)
    save_once(OUT/"analysis_summary.json",{
       "study":"ROLE-ANCHOR-PROBE-v0",
       "status":"COMPLETE_12_OLD_SEEDS_48_READ_ONLY_CHECKPOINTS",
       "exploratory_only":True,
       "no_new_training":True,
       "seeds":list(SEEDS),
       "source_manifest_sha256":sha(pre.MANIFEST),
       "parent_protocol_digest":study["protocol_digest"],
       "original_mapping_baseline":baseline,
       "comparisons":values
    })
    print("SUMMARY PASS: 12 old seeds / 48 checkpoints (read-only)")
    print("Baseline original-role slot agreement",baseline["original_mapping_single_answer_accuracy"])
    print("Interpretation exploratory only; original train mappings are NOT yet proven to be model predictions")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("smoke","run","summary"))
    p.add_argument("--seed",type=int)
    p.add_argument("--device",choices=("cpu","mps"),default="cpu")
    a=p.parse_args()
    if a.phase=="smoke":smoke(torch.device(a.device))
    elif a.phase=="run":
        if a.seed is None:p.error("run requires --seed")
        run(a.seed,torch.device(a.device))
    else:summary()

if __name__=="__main__":main()
