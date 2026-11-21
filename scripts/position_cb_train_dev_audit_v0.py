#!/usr/bin/env python3
"""Exploratory checkpoint-only audit of POSITION-CB-v0 transfer reliability.

For each of the 32 frozen final models compare accuracy on:
  (1) exact source variants selected by its training schedule (144 cores)
  (2) source variants from the original TRAIN pool NEVER selected during that
      seed's entire 6,400-update schedule (up to 144 cores)
  (3) its already sealed familiar new-development score (read from original
      per-process JSON; no fresh dev predictions)
Only familiar voice combinations for each arm are evaluated in (1) and (2).

No training, checkpoint modifications, protected/reserved/L+I evaluation,
or new grammar generation. Post-hoc descriptive only.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, stdev
import torch
import position_counterbalance_v0 as cb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/position_cb_train_dev_audit_v0"
ARMS = ("same_voice", "mixed_voice")
N_VARIANTS = 144
TC95_7 = 2.364624251

def report_path(seed, repeat):
    return OUT / f"seed{seed}_repeat{repeat}_audit.json"

def indices_by_exposure(seed, dose, dp):
    used, sampled = set(), []
    sampled_set = set()
    for _step, indices, _cell in cb.schedule(seed, dp):
        used.update(indices)
        if len(sampled) < N_VARIANTS:
            for ix in indices[:3]:
                if ix not in sampled_set:
                    sampled_set.add(ix)
                    sampled.append(ix)
                    if len(sampled) == N_VARIANTS:
                        break
    assert len(sampled) == N_VARIANTS
    assert set(sampled).issubset(used)
    untouched = sorted(set(range(len(dose))) - used)
    # "Not sampled" means this EXACT pool index did not enter any update.
    # It DOES NOT mean its source names or event cores were never encountered.
    assert len(untouched) >= 64, (
        "Too few untouched train variants for this diagnostic", seed, len(untouched))
    rng = Random(seed + 127432)
    chosen_unused = rng.sample(untouched, min(N_VARIANTS, len(untouched)))
    assert not set(chosen_unused) & used
    return sampled, chosen_unused, {
        "unique_sampled_pool_variants":len(used),
        "never_selected_pool_variants":len(untouched),
        "selected_eval_variants":len(sampled),
        "unused_eval_variants":len(chosen_unused),
        "selected_rule":"first three indices per schedule step until 144 unique",
        "unused_rule":"seeded random sample from never-selected indices",
    }

def score_subset(net, dose, selected_indices, arm, vocab, device):
    voice_pair = cb.label_sets(arm)[0]
    sources = [dose[ix][v] for ix in selected_indices for v in voice_pair]
    assert len(sources) == 2*len(selected_indices)
    encoded = cb.pilot.encode_sources(sources, vocab, check=False)
    logits = cb.exp113.no_touch_identity(net.eval(), encoded, device)
    metrics = cb.exp112.metrics(logits, encoded)
    return {"C_query":metrics["C_query"],
            "accuracy4":metrics["accuracy4"],
            "full_vocab_accuracy":metrics["full_vocab_accuracy"],
            "n_sources":len(sources),
            "n_unique_pool_indices":len(selected_indices)}

def load_original(seed, repeat):
    p = ROOT / "reports/position_cb_v0" / (
        f"seed{seed}_repeat{repeat}_mps_step6400.json")
    data = json.loads(p.read_text(encoding="utf8"))
    assert (data["study"],data["seed"],data["repeat"],data["steps"]) == (
        "POSITION-CB-v0",seed,repeat,cb.STEPS)
    return data

def get_familiar_from_existing_record(rec, arm):
    seen,_unseen = cb.label_sets(arm)
    return {"C_query":mean(rec["outcomes"][arm][v]["C_query"] for v in seen),
            "accuracy4":mean(rec["outcomes"][arm][v]["accuracy4"] for v in seen),
            "full_vocab_accuracy":mean(rec["outcomes"][arm][v]["full_vocab_accuracy"] for v in seen)}

def preflight():
    man,(dose,dp,_dev,_vocab),_groups=cb.load_and_verify()
    counts={}
    for seed in cb.SEEDS:
        a,b,c=indices_by_exposure(seed,dose,dp)
        assert len(a)==N_VARIANTS and len(b)>=64
        counts[str(seed)]=c
    print("PREFLIGHT PASS: existing frozen manifest",man["evaluation_audit"]["evaluation_sha256"])
    print("New data generated: NONE; new model inference: NONE; protected/L+I: CLOSED")
    for seed,c in counts.items():
        print(seed,"never-selected train variants",c["never_selected_pool_variants"],
              "diagnostic",c["unused_eval_variants"])

def run(seed,repeat,device):
    if seed not in cb.SEEDS or repeat not in cb.REPEATS:
        raise ValueError("Unknown frozen seed/repetition")
    output=report_path(seed,repeat)
    if output.exists():
        print("SKIP existing immutable diagnostic",output)
        return
    man,(dose,dp,_dev,vocab),_groups=cb.load_and_verify()
    original=load_original(seed,repeat)
    assert original["protocol_digest"] == man["protocol_digest"]
    assert original["evaluation_sha256"] == man["evaluation_audit"]["evaluation_sha256"]
    selected,never,exposure=indices_by_exposure(seed,dose,dp)
    checkpoint = ROOT / "results/position_cb_v0" / (
        f"seed{seed}_repeat{repeat}_mps_step6400.pt")
    if not checkpoint.exists():
        raise FileNotFoundError("Local checkpoint missing: "+str(checkpoint))
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    assert saved["study"]=="POSITION-CB-v0"
    assert (saved["seed"],saved["repeat"],saved["step"])==(seed,repeat,cb.STEPS)
    assert saved["protocol_digest"]==man["protocol_digest"]
    assert set(saved["models"])==set(ARMS)
    results={}
    for arm in ARMS:
        net=cb.pilot.Model(len(vocab)).to(device)
        net.load_state_dict(saved["models"][arm],strict=True)
        net.eval()
        results[arm]={
            "exact_training_presentations":score_subset(net,dose,selected,arm,vocab,device),
            "not_selected_training_pool_variants":score_subset(net,dose,never,arm,vocab,device),
            "existing_dev_familiar":get_familiar_from_existing_record(original,arm),
            "final_recorded_minibatch_loss":original["loss_history"][-1]["losses"][arm],
        }
        del net
    record={"study":"POSITION-CB-TRAIN-DEV-AUDIT-v0",
            "status":"POSTHOC_CHECKPOINT_ONLY",
            "seed":seed,"repeat":repeat,"device":str(device),
            "parent_protocol_digest":man["protocol_digest"],
            "parent_evaluation_sha256":man["evaluation_audit"]["evaluation_sha256"],
            "no_training":True,"no_new_dev_scoring":True,
            "no_protected_or_LI":True,"exposure_counts":exposure,
            "arms":results,
            "caution":"Exact never-selected *pool variants* may share names and cores with training; not external held-out examples."}
    cb.write_once(output,record)
    print("AUDIT SAVED",output,"(no individual sentence text or predictions saved)")

def seed_ci(vals):
    assert len(vals)==8
    m=mean(vals);radius=TC95_7*stdev(vals)/math.sqrt(len(vals))
    return {"mean":m,"ci95":[m-radius,m+radius],"per_seed":
            dict(zip(map(str,cb.SEEDS),vals))}

def summary():
    _man,_r,_g=cb.load_and_verify()
    records={}
    for seed in cb.SEEDS:
        for rep in cb.REPEATS:
            p=report_path(seed,rep)
            if not p.exists():
                print("INCOMPLETE: missing",p,"— do not aggregate a subset")
                return
            x=json.loads(p.read_text())
            assert (x["seed"],x["repeat"])==(seed,rep)
            assert x["no_training"] and x["no_new_dev_scoring"] and x["no_protected_or_LI"]
            records[(seed,rep)]=x
    categories=("exact_training_presentations",
                "not_selected_training_pool_variants","existing_dev_familiar")
    metrics=("C_query","accuracy4","full_vocab_accuracy")
    results={}
    for arm in ARMS:
        results[arm]={}
        for cat in categories:
            results[arm][cat]={}
            for metric in metrics:
                values=[mean(records[(seed,rep)]["arms"][arm][cat][metric]
                             for rep in cb.REPEATS) for seed in cb.SEEDS]
                results[arm][cat][metric]=seed_ci(values)
    failures={}
    for arm in ARMS:
        failures[arm]={
            "n_model_states_dev_familiar_joint_below_10pct":
                sum(records[(s,r)]["arms"][arm]["existing_dev_familiar"]["C_query"]<.10
                    for s in cb.SEEDS for r in cb.REPEATS),
            "n_model_states_low_dev_with_final_train_batch_loss_below_001":
                sum(records[(s,r)]["arms"][arm]["existing_dev_familiar"]["C_query"]<.10
                    and records[(s,r)]["arms"][arm]["final_recorded_minibatch_loss"]<.01
                    for s in cb.SEEDS for r in cb.REPEATS),
        }
    result={"study":"POSITION-CB-TRAIN-DEV-AUDIT-v0",
            "status":"COMPLETE_POSTHOC_DIAGNOSTIC",
            "source_reports":16,"model_states":32,
            "independent_seeds":8,"n_repeats_per_seed":2,
            "no_new_training_or_dev_scoring":True,
            "analysis_unit":"seed; MPS repeats averaged before t interval",
            "results":results,"weak_model_counts":failures,
            "limitations":"Diagnostic train partition samples; non-selected pool variants are not guaranteed new cores/names; low training minibatch loss is not a test of generalization.",
            "no_protected_or_LI":True}
    cb.write_once(OUT/"analysis_summary.json",result)
    print("POSTHOC AUDIT COMPLETE: all 16 original checkpoints / 32 model states")
    for arm in ARMS:
        for cat in categories:
            o=results[arm][cat]
            print(arm,cat,"joint",round(o["C_query"]["mean"],4),
                  "question",round(o["accuracy4"]["mean"],4))
        print("weak models",failures[arm])
    print("No new confirmatory evidence; do not select or remove weak seeds.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("preflight","run","summary"),required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--repeat",type=int)
    p.add_argument("--device",choices=("cpu","mps"),default="mps")
    a=p.parse_args()
    if a.phase=="preflight":preflight()
    elif a.phase=="run":
        if a.seed is None or a.repeat is None:p.error("run needs --seed and --repeat")
        run(a.seed,a.repeat,torch.device(a.device))
    else:summary()

if __name__=="__main__":
    main()
