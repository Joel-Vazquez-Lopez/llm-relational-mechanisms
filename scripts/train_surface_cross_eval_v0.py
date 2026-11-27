#!/usr/bin/env python3
"""TRAIN-SURFACE-CROSS-EVAL-v0 read-only checkpoint diagnostics.

This is an EXPLORATORY analysis on the already trained 12-seed cohort.
No gradient steps, checkpoint writes, source modifications, original
DEV/reserved/protected/L+I use, or new-model training are permitted.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from hashlib import sha256
from pathlib import Path
from statistics import mean,stdev
import torch
from torch import nn

import train_surface_factorial_v0 as parent
import train_surface_cross_eval_v0_preflight as pre

ROOT=Path(__file__).resolve().parents[1]
OUT=pre.OUT
MANIFEST=pre.MANIFEST
PROTOCOL=ROOT/"paper/TRAIN_SURFACE_CROSS_EVAL_V0_PROTOCOL.md"
SCRIPT=Path(__file__).resolve()
SEEDS=parent.SEEDS
ARMS=parent.ARMS
VOICES=pre.VOICE
CONDITIONS=pre.CONDITIONS
POPS=pre.POPS
T95=2.200985160082949
assert len(SEEDS)==12

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(data):return pre.digest(data)
def save_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write("\n")

def fixtures():
    if not MANIFEST.exists():
        raise RuntimeError("Run source preflight --phase prepare first")
    expected_sources,expected=pre.build()
    actual=json.loads(MANIFEST.read_text())
    if actual!=expected:
        raise RuntimeError("Cross-surface source manifest drift")
    original_study=json.loads(parent.STUDY.read_text())
    if expected["parent_protocol_digest"]!=original_study["protocol_digest"]:
        raise RuntimeError("Parent scientific cohort protocol mismatch")
    _train,_held,_tables,vocab,_parent_src=parent.fixtures()
    if parent.digest(sorted(vocab.items()))!=original_study["protocol"]["vocab_sha256"]:
        raise RuntimeError("Parent vocabulary has changed")
    if sha(parent.SCRIPT)!=original_study["protocol"]["script_sha256"]:
        raise RuntimeError("Parent scientific training script has changed")
    if sha(parent.SOURCE)!=original_study["protocol"]["source_manifest_sha256"]:
        raise RuntimeError("Parent frozen source manifest has changed")
    if sha(parent.LAUNCH)!=original_study["protocol"]["launcher_sha256"]:
        raise RuntimeError("Parent launcher has changed")
    return expected_sources,expected,vocab,original_study

def freeze_inputs():
    paths=("scripts/train_surface_cross_eval_v0_preflight.py",
           "scripts/train_surface_cross_eval_v0.py",
           "paper/TRAIN_SURFACE_CROSS_EVAL_V0_PROTOCOL.md",
           "reports/train_surface_cross_eval_v0/source_manifest.json")
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],
                  cwd=ROOT,text=True).strip()
        for relative in paths:
            committed=subprocess.check_output(["git","show",head+":"+relative],
                  cwd=ROOT,stderr=subprocess.DEVNULL)
            if committed!=(ROOT/relative).read_bytes():
                raise RuntimeError("Cross-surface audit inputs not frozen: "+relative)
        return head
    except (subprocess.CalledProcessError,FileNotFoundError) as exc:
        raise RuntimeError("Commit diagnostic code, protocol and source manifest before scoring") from exc

@torch.inference_mode()
def score(model,sources,vocab,device,chunk_size=128):
    """Return strict accuracy, pair classes and item-aligned predictions.

    The prediction signatures remain in RAM for paired source transformations.
    Only summary metrics and paired transition counts are archived.
    """
    if not sources:raise ValueError("empty source list")
    model.eval()
    ids,readout,labels,candidates=parent.encode_sources(sources,vocab,check=True)
    pred=[];full=[]
    for first in range(0,len(ids),chunk_size):
        last=min(first+chunk_size,len(ids))
        logits=model(ids[first:last].to(device),
                     readout[first:last].to(device)).float().cpu()
        c=candidates[first:last]
        pred.append(c.gather(1,logits.gather(1,c).argmax(dim=1,keepdim=True)).flatten())
        full.append(logits.argmax(1))
    pred=torch.cat(pred).view(-1,4)
    gold=labels.view(-1,4)
    assert len(pred)==len(sources)
    correctness=pred.eq(gold)
    joint=correctness.all(1)
    outcome={
       "n_sources":len(sources),
       "C_query":float(joint.float().mean()),
       "accuracy4":float(correctness.float().mean()),
       "full_vocab_accuracy":float(torch.cat(full).eq(labels).float().mean()),
       "pair_classes":{}
    }
    for event in (1,2):
        off=(event-1)*2
        p0,p1=pred[:,off],pred[:,off+1]
        g0,g1=gold[:,off],gold[:,off+1]
        bad_event=~(((p0==g0)|(p0==g1))&((p1==g0)|(p1==g1)))
        masks={
            "correct":(p0==g0)&(p1==g1)&~bad_event,
            "reversed":(p0==g1)&(p1==g0)&~bad_event,
            "collapsed":(p0==p1)&~bad_event,
            "wrong_event_name":bad_event
        }
        assert torch.stack(tuple(masks.values())).int().sum(0).eq(1).all()
        outcome["pair_classes"][str(event)]={name:float(mask.float().mean())
                                             for name,mask in masks.items()}
    return outcome,joint,pred

def transitions(baseline,decorated):
    """Both scores use exactly identical semantic sources and item ordering."""
    b_correct,b_pred=baseline
    d_correct,d_pred=decorated
    assert len(b_correct)==len(d_correct) and b_pred.shape==d_pred.shape
    a=~b_correct;d=d_correct
    counts={
       "both_correct":int((b_correct&d_correct).sum()),
       "both_wrong":int((a&~d_correct).sum()),
       "rescued_on_decorated":int((a&d_correct).sum()),
       "lost_on_decorated":int((b_correct&~d_correct).sum()),
       "same_four_predictions":int((b_pred==d_pred).all(1).sum())
    }
    n=len(b_correct)
    assert sum(counts[k] for k in (
        "both_correct","both_wrong","rescued_on_decorated",
        "lost_on_decorated"))==n
    wrong=int(a.sum())
    right=int(b_correct.sum())
    return {"n":n,**counts,
            "rescued_fraction_of_bare_errors":(counts["rescued_on_decorated"]/wrong
                                               if wrong else None),
            "lost_fraction_of_bare_successes":(counts["lost_on_decorated"]/right
                                               if right else None),
            "source_level_prediction_invariance":counts["same_four_predictions"]/n}

def smoke(device):
    """Random-weight, training-core-only computational check; no checkpoints."""
    sources,meta,vocab,_study=fixtures()
    _=meta
    torch.manual_seed(17012026)
    model=parent.Model(len(vocab)).to(device)
    tiny=sources["train_core"]["bare"][VOICES[0]][:3]
    met,joint,pred=score(model,tiny,vocab,device,chunk_size=4)
    assert len(joint)==3 and pred.shape==(3,4)
    assert 0<=met["C_query"]<=1
    assert math.isclose(sum(met["pair_classes"]["1"].values()),1,abs_tol=1e-5)
    repeat=transitions((joint,pred),(joint,pred))
    assert repeat["same_four_predictions"]==3
    assert repeat["rescued_on_decorated"]==repeat["lost_on_decorated"]==0
    print("TRAIN-CORE-ONLY SMOKE PASS; no checkpoint loading or role-holdout inference")

def run(seed,device):
    if seed not in SEEDS:raise ValueError("Unknown frozen training seed")
    out=OUT/("seed%d_cross_eval.json"%seed)
    if out.exists():raise FileExistsError("Cross-surface report exists; never overwrite")
    sources,source_meta,vocab,study=fixtures()
    diagnostic_head=freeze_inputs()
    checkpoint=parent.WEIGHTS/("seed%d_mps_step6400.pt"%seed)
    if not checkpoint.exists():
        raise FileNotFoundError("Original checkpoint missing; DO NOT retrain: "+str(checkpoint))
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["seed"]==seed and ck["steps"]==parent.STEPS
    assert ck["study"]=="TRAIN-SURFACE-FACTORIAL-v0"
    assert ck["protocol_digest"]==study["protocol_digest"]
    assert set(ck["models"])==set(ARMS)
    outcomes={};paired={}
    for arm in ARMS:
        model=parent.Model(len(vocab)).to(device)
        model.load_state_dict(ck["models"][arm],strict=True)
        model.eval()
        outcomes[arm]={};paired[arm]={}
        for pop in POPS:
            outcomes[arm][pop]={};paired[arm][pop]={}
            raw={}
            for condition in CONDITIONS:
                outcomes[arm][pop][condition]={}
                raw[condition]={}
                for voice in VOICES:
                    met,flags,pred=score(model,sources[pop][condition][voice],vocab,device)
                    outcomes[arm][pop][condition][voice]=met
                    raw[condition][voice]=(flags,pred)
            if pop!="train_core":
                for condition in CONDITIONS:
                    if condition=="bare":continue
                    paired[arm][pop][condition]={}
                    for voice in VOICES:
                        paired[arm][pop][condition][voice]=transitions(
                            raw["bare"][voice],raw[condition][voice])
        print("SCORED",seed,arm,flush=True)
        del model
    save_once(out,{
        "study":"TRAIN-SURFACE-CROSS-EVAL-v0",
        "status":"SEALED_READ_ONLY_DIAGNOSTIC",
        "seed":seed,
        "source_manifest_digest":digest(source_meta),
        "parent_protocol_digest":study["protocol_digest"],
        "diagnostic_head":diagnostic_head,
        "checkpoint_file":checkpoint.name,
        "checkpoint_file_sha256":sha(checkpoint),
        "no_new_training":True,
        "no_original_dev_reserved_protected_LI":True,
        "outcomes":outcomes,
        "paired_transitions":paired})
    print("SEALED",out)
    print("Do not aggregate until all twelve seed reports are complete")

def sample_stats(values):
    assert len(values)==12
    average=mean(values)
    margin=T95*stdev(values)/math.sqrt(12)
    return {"mean":average,"t95_descriptive":[average-margin,average+margin],
            "min":min(values),"max":max(values),
            "per_seed":dict(zip(map(str,SEEDS),values))}

def summary():
    sources,meta,vocab,study=fixtures()
    _=(sources,vocab)
    reports={}
    for seed in SEEDS:
        path=OUT/("seed%d_cross_eval.json"%seed)
        if not path.exists():
            print("INCOMPLETE. Missing seed",seed,"; no partial aggregation")
            return
        r=json.loads(path.read_text())
        assert r["seed"]==seed and r["study"]=="TRAIN-SURFACE-CROSS-EVAL-v0"
        assert r["source_manifest_digest"]==digest(meta)
        assert r["parent_protocol_digest"]==study["protocol_digest"]
        assert r["no_new_training"] and r["no_original_dev_reserved_protected_LI"]
        assert set(r["outcomes"])==set(ARMS)
        reports[seed]=r
    result={}
    for arm in ARMS:
        result[arm]={}
        for pop in POPS:
            result[arm][pop]={}
            for condition in CONDITIONS:
                result[arm][pop][condition]={}
                for name in ("C_query","accuracy4","full_vocab_accuracy"):
                    vals=[mean(reports[s]["outcomes"][arm][pop][condition][v][name]
                               for v in VOICES) for s in SEEDS]
                    result[arm][pop][condition][name]=sample_stats(vals)
                if pop!="train_core" and condition!="bare":
                    vals=[]
                    for s in SEEDS:
                        voice_trans=[reports[s]["paired_transitions"][arm][pop]
                              [condition][v] for v in VOICES]
                        rescued=sum(t["rescued_on_decorated"] for t in voice_trans)
                        lost=sum(t["lost_on_decorated"] for t in voice_trans)
                        same=sum(t["same_four_predictions"] for t in voice_trans)
                        n=sum(t["n"] for t in voice_trans)
                        vals.append({"rescued":rescued/n,"lost":lost/n,
                                     "same_predictions":same/n})
                    result[arm][pop][condition]["paired_transition_rates"]={
                         k:sample_stats([a[k] for a in vals])
                         for k in ("rescued","lost","same_predictions")}
    finished={
       "study":"TRAIN-SURFACE-CROSS-EVAL-v0",
       "status":"DESCRIPTIVE_POSTHOC_12_SEED_CHECKPOINT_DIAGNOSTIC",
       "no_confirmatory_tests":True,
       "parent_protocol_digest":study["protocol_digest"],
       "source_manifest_digest":digest(meta),
       "seeds":list(SEEDS),"arm_population_surface_results":result}
    save_once(OUT/"analysis_summary.json",finished)
    print("SUMMARY PASS: 12 old independent seeds, 48 read-only model checkpoints")
    print("Exploratory diagnostic only; NO independent replication, NO new training")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("smoke","score","summary"))
    p.add_argument("--seed",type=int)
    p.add_argument("--device",choices=("cpu","mps"),default="cpu")
    args=p.parse_args()
    if args.phase=="smoke":smoke(torch.device(args.device))
    elif args.phase=="score":
        if args.seed is None:p.error("Specify a frozen training seed")
        run(args.seed,torch.device(args.device))
    else:summary()

if __name__=="__main__":main()
