#!/usr/bin/env python3
"""ROLE-DIVERSITY-COMPLEX-v0: prospective 1/2/4/8 role exposure experiment.

Original TRAIN-derived 240/60 quartet partition only. Fresh seeds/models.
Training and test role assignment sets disjoint across EVERY arm.
The study MUST be source- and protocol-frozen before scientific inference.
"""
from __future__ import annotations
import argparse,json,math,subprocess
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import permutations,product
from pathlib import Path
from random import Random
from statistics import mean,stdev

import torch

import train_surface_factorial_v0 as old
import role_diversity_complex_v0_preflight as pf

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
PROTOCOL=ROOT/"paper/ROLE_DIVERSITY_COMPLEX_V0_PROTOCOL.md"
LAUNCH=ROOT/"scripts/run_role_diversity_complex_v0.sh"
SOURCE=pf.MANIFEST
STUDY=pf.OUT/"study_manifest.json"
OUT=pf.OUT
WEIGHTS=ROOT/"results/role_diversity_complex_v0"
SEEDS=(16101,16103,16107,16109,16113,16119,16121,16127,
       16131,16133,16139,16143,16149,16157,16161,16167)
ARMS=pf.ARMS
VOICES=pf.VOICES
STEPS=6400
CELLS=tuple(product(range(5),range(9)))
MILESTONES=(800,1600,2400,3200,4000,4800,5600,6400)
T95=2.131449545559323
POPS=("original_bare","original_rich","train_seen_rich",
      "unseen_roles_bare","unseen_roles_rich",
      "unseen_quartets_bare","unseen_quartets_rich")
assert len(SEEDS)==16 and len(ARMS)==4 and len(CELLS)==45

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return pf.digest(x)
def save_once(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(doc,f,sort_keys=True,indent=2);f.write("\n")

def role_ids(events):
    original,eligible,heldout,subsets=pf.role_orders(events)
    assert len(eligible)==8 and len(heldout)==16 and original==eligible[0]
    return eligible,heldout,subsets

def rich_cell(ix,index,n):
    """Deterministic frame+adjunct CELL, not selected using model results.

    n=16 for never-trained roles, n=24 for withheld quartets; for
    original/seen roles use the common 16-role indexing to avoid leakage.
    """
    k=ix*n+index
    return 1+(k%4),1+((k//4)%8)

def fixtures():
    if not SOURCE.exists():
        raise FileNotFoundError("Prepare and freeze source-only design manifest")
    actual=json.loads(SOURCE.read_text())
    expected=pf.audit()
    if actual!=expected:raise RuntimeError("Source-only feasibility manifest drift")
    train,held,orig,_=pf.prev.partition_cores()
    # Same vocabulary and word IDs as original factorial. No new lexical items.
    ptrain,pheld,_pool,vocab,parent_src=old.fixtures()
    assert train==ptrain and held==pheld
    parent_study=json.loads(old.STUDY.read_text())
    assert parent_study["protocol_digest"]==actual["parent_factorial_digest"]
    assert digest(sorted(vocab.items()))==parent_study["protocol"]["vocab_sha256"]
    assert sha(old.SCRIPT)==parent_study["protocol"]["script_sha256"]
    train_roles={};hold_roles={}
    for ix in train:
        events=orig[ix]
        eligible,heldout,subsets=role_ids(events)
        train_roles[ix]=(events,eligible,heldout)
        assert len(eligible)==8 and len(heldout)==16
    for ix in held:
        events=orig[ix]
        orig_roles=tuple(n for e in events for n in (e.agent,e.patient))
        hold_roles[ix]=(events,tuple(permutations(orig_roles)))
    return train,held,train_roles,hold_roles,vocab,actual,parent_study

def schedule(seed,train):
    """Identical quartet and 45-surface-cell schedule across all four arms.

    Every quartet's t-th exposure uses role(t)=t mod n for arm roles_n.
    Cell(t)=permuted_cell[t mod 45]. Because gcd(45,n)==1,
    every quartet has a balanced n×45 joint exposure matrix (+/-1)
    for n=1,2,4,8. Training starts at the original mapping (ordinal 0)
    in every arm. No test examples can appear in this schedule.
    """
    assert seed in SEEDS
    rng=Random(seed+16131013)
    perm={}
    for ix in train:
        cells=list(CELLS);rng.shuffle(cells)
        perm[ix]=tuple(cells)
    draw=Counter()
    for step in range(1,STEPS+1):
        chosen=tuple(rng.sample(train,8))
        event=[]
        for ix in chosen:
            count=draw[ix]
            fi,mi=perm[ix][count%45]
            event.append((ix,fi,mi,count))
            draw[ix]+=1
        yield step,chosen,tuple(event)

def schedule_fingerprint(seed,train,audit=True):
    h=sha256();draw_counts=Counter();joints={n:Counter() for n in pf.DIVERSITIES}
    for step,quartets,events in schedule(seed,train):
        assert len(set(quartets))==8 and len(events)==8
        for (ix,fi,mi,k) in events:
            assert k==draw_counts[ix]
            draw_counts[ix]+=1
            for n in pf.DIVERSITIES:
                joints[n][(ix,k%n,fi,mi)]+=1
        h.update((json.dumps((step,quartets,events),separators=(",",":"))+"\n").encode())
    assert sum(draw_counts.values())==STEPS*8
    if audit:
        assert set(draw_counts)==set(train)
        assert min(draw_counts.values())>=1
        for ix in train:
            t=draw_counts[ix]
            for n in pf.DIVERSITIES:
                c=joints[n]
                counts=[c[(ix,j,fi,mi)]
                        for j in range(n) for fi,mi in CELLS]
                assert sum(counts)==t
                assert max(counts)-min(counts)<=1,(seed,ix,n)
                # The marginal frequency of every allowed role order is balanced.
                role=[sum(c[(ix,j,fi,mi)] for fi,mi in CELLS)
                      for j in range(n)]
                assert max(role)-min(role)<=1
    return h.hexdigest(),{"min_quartet_exposures":min(draw_counts.values()),
                          "max_quartet_exposures":max(draw_counts.values()),
                          "all_role_by_surface_cell_counts_maxdiff":1}

def expected(train,held,vocab,src,parent_study):
    dep=("train_surface_factorial_v0.py",
         "train_surface_factorial_v0_preflight.py",
         "role_diversity_complex_v0_preflight.py",
         "core_diversity_v1_preflight.py",
         "exp105_generator_preflight.py",
         "exp105_position_control_preflight_v1.py")
    digests={};audit={}
    for seed in SEEDS:
        digests[str(seed)],audit[str(seed)]=schedule_fingerprint(seed,train)
    p={
      "study":"ROLE-DIVERSITY-COMPLEX-v0",
      "status":"READY_TO_FREEZE_BEFORE_ANY_NEW_TRAINING",
      "parent_protocol_digest":parent_study["protocol_digest"],
      "source_manifest_sha256":sha(SOURCE),
      "train_quartet_ids_sha256":digest(train),
      "heldout_quartet_ids_sha256":digest(held),
      "seeds":list(SEEDS),"steps":STEPS,"arms":list(ARMS),
      "train_eligible_role_counts":list(pf.DIVERSITIES),
      "role_selection":"nested role indices 0..n-1 from original 8 (4 cyclic +4 reverse)",
      "shared_heldout_roles":"same 16 remaining permutations never trained under ANY arm",
      "shared_heldout_quartets":"60 previously TRAIN-derived quartet ids; all 24 role permutations",
      "training_quartets_per_step":8,
      "two_voices_per_quartet":list(VOICES),
      "four_queries_per_source":True,
      "labels_per_model_update":64,
      "matched_initialization_per_seed":True,
      "source_cells":"5 prefix frames x 9 named-modifier pairs, including unmodified",
      "matched_quartet_and_cell_schedule":True,
      "per_quartet_role_by_cell_balance":"joint cell x allowed role frequencies differ <=1; gcd(45,n)=1",
      "per_quartet_schedule_audit":audit,
      "model":"original three-layer width-96 four-head causal Transformer, 48-position",
      "optimizer":"AdamW lr=.002 weight_decay=.01 6400 fixed updates",
      "rich_eval_cell":"fi=1+(k%4), mi=1+((k//4)%8), k=quartet_id*n + assignment_ordinal",
      "primary_population":"all 16 never-trained role permutations per each of 240 training quartets, AP and PA, same rich sources in each arm",
      "primary_contrast":"roles_8 - roles_1 strict native four-query C_query",
      "primary_success":"16 independent seed paired t 95% CI lower>0 AND mean >= +0.10 AND >=12/16 positive seed differences",
      "gated_secondary":"roles_2 - roles_1 same rich never-trained roles; tested only when primary passes; CI lower>0 AND mean>=0.10",
      "exploratory":"roles_4, dose shape, original-core competence, error taxonomy, bare test, withheld quartets, per-seed anchoring",
      "no_posthoc_seed_checkpoint_selection":True,
      "one_shot_learning_not_tested":True,
      "no_old_dev_reserved_protected_LI":True,
      "vocab_sha256":digest(sorted(vocab.items())),
      "source_hashes":src["source_hashes"],
      "script_sha256":sha(SCRIPT),
      "launcher_sha256":sha(LAUNCH),
      "protocol_sha256":sha(PROTOCOL),
      "dependency_sha256":{x:sha(ROOT/"scripts"/x) for x in dep},
      "schedule_sha256":digests
    }
    return {"protocol":p,"protocol_digest":digest(p)}

def prepare():
    train,held,training,withheld,vocab,src,parent_study=fixtures()
    _=training,withheld
    manifest=expected(train,held,vocab,src,parent_study)
    if STUDY.exists():
        if json.loads(STUDY.read_text())!=manifest:
            raise RuntimeError("Frozen study manifest changed, never overwrite")
        print("PREPARE VERIFIED")
    else:
        save_once(STUDY,manifest)
        print("PREPARE PASS")
    print("16 new independent seeds ×4 =64 new models; no training",
          manifest["protocol_digest"])

def verified():
    train,held,training,withheld,vocab,src,parent_study=fixtures()
    manifest=expected(train,held,vocab,src,parent_study)
    if not STUDY.exists() or json.loads(STUDY.read_text())!=manifest:
        raise RuntimeError("Study manifest unfrozen or changed")
    return train,held,training,withheld,vocab,manifest

def require_frozen():
    paths=(
      "scripts/role_diversity_complex_v0_preflight.py",
      "scripts/role_diversity_complex_v0.py",
      "scripts/run_role_diversity_complex_v0.sh",
      "paper/ROLE_DIVERSITY_COMPLEX_V0_PROTOCOL.md",
      "reports/role_diversity_complex_v0/design_source_audit.json",
      "reports/role_diversity_complex_v0/study_manifest.json"
    )
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        for relative in paths:
            blob=subprocess.check_output(["git","show",head+":"+relative],
                      cwd=ROOT,stderr=subprocess.DEVNULL)
            if blob!=(ROOT/relative).read_bytes():
                raise RuntimeError("Uncommitted / changed study input "+relative)
        return head
    except (FileNotFoundError,subprocess.CalledProcessError) as e:
        raise RuntimeError("Commit source, code, protocol and manifests BEFORE scientific training") from e

def source(events,roles,voice,fi,mi,group):
    s=pf.prev.source(events,roles,voice,fi,mi,group=group)
    assert s.gold==roles
    return s

def make_batch(events,training):
    batch={a:[] for a in ARMS}
    for ix,fi,mi,k in events:
        event,eligible,_unused=training[ix]
        for n in pf.DIVERSITIES:
            arm="roles_%d"%n
            role=eligible[k%n]
            batch[arm].extend(source(event,role,v,fi,mi,"rolediv_train")
                              for v in VOICES)
    assert all(len(batch[a])==16 for a in ARMS)
    return batch

def evaluations(train,held,training,withheld,vocab):
    """Shared rich/bare heldout eval, plus arm-specific TRAIN seen competence."""
    groups={p:{v:[] for v in VOICES} for p in POPS if p!="train_seen_rich"}
    seen={a:{v:[] for v in VOICES} for a in ARMS}
    def append(p,events,roles,ix,ordinal,n,fi=None,mi=None):
        if fi is None or mi is None:fi,mi=rich_cell(ix,ordinal,n)
        for v in VOICES:
            groups[p][v].append(source(events,roles,v,fi,mi,"rolediv_"+p))
    for ix in train:
        events,eligible,never=training[ix]
        original=eligible[0]
        append("original_bare",events,original,ix,0,16,0,0)
        append("original_rich",events,original,ix,0,16)
        for n in pf.DIVERSITIES:
            arm="roles_%d"%n
            for ordinal in range(n):
                fi,mi=rich_cell(ix,ordinal,16)
                for v in VOICES:
                    seen[arm][v].append(source(
                       events,eligible[ordinal],v,fi,mi,"rolediv_train_seen"))
        for ordinal,roles in enumerate(never):
            append("unseen_roles_bare",events,roles,ix,ordinal,16,0,0)
            append("unseen_roles_rich",events,roles,ix,ordinal,16)
    for ix in held:
        events,all24=withheld[ix]
        for ordinal,roles in enumerate(all24):
            append("unseen_quartets_bare",events,roles,ix,ordinal,24,0,0)
            append("unseen_quartets_rich",events,roles,ix,ordinal,24)
    expected_counts={
      "original_bare":240,"original_rich":240,
      "unseen_roles_bare":240*16,"unseen_roles_rich":240*16,
      "unseen_quartets_bare":60*24,"unseen_quartets_rich":60*24}
    for p,n in expected_counts.items():
        assert all(len(groups[p][v])==n for v in VOICES)
        # Every rich heldout evaluation has nonempty prefix and modifier pair.
        if p.endswith("rich"):
            assert all(s.mods!=( "bare","bare") for v in VOICES for s in groups[p][v])
    data={p:{v:old.encode_sources(groups[p][v],vocab,check=False)
             for v in VOICES} for p in groups}
    data["train_seen_rich"]={
       arm:{v:old.encode_sources(seen[arm][v],vocab,check=False)
            for v in VOICES} for arm in ARMS}
    return data

def eval_preflight():
    """Exhaustively encode frozen TRAIN-derived evaluation sources, NO model.

    Catch shape/token/gold errors before any expensive prospective training.
    Source-only check must never load checkpoints or call a model.
    """
    train,held,training,withheld,vocab,manifest=verified()
    _=manifest
    tests=evaluations(train,held,training,withheld,vocab)
    expected={"original_bare":240,"original_rich":240,
              "unseen_roles_bare":240*16,"unseen_roles_rich":240*16,
              "unseen_quartets_bare":60*24,"unseen_quartets_rich":60*24}
    for pop,n in expected.items():
        assert set(tests[pop])==set(VOICES)
        for voice in VOICES:
            ids,readout,labels,choices=tests[pop][voice]
            assert ids.shape[0]==readout.numel()==labels.numel()==4*n
            assert len(choices)==4*n and choices.shape[1]==4
            assert ids.shape[1]<=old.MAX_TOKENS
            assert int(readout.max())<old.MAX_TOKENS
    for arm,n in zip(ARMS,pf.DIVERSITIES):
        assert set(tests["train_seen_rich"][arm])==set(VOICES)
        for voice in VOICES:
            ids,readout,labels,choices=tests["train_seen_rich"][arm][voice]
            assert ids.shape[0]==readout.numel()==labels.numel()==240*n*4
            assert len(choices)==240*n*4
            assert ids.shape[1]<=old.MAX_TOKENS
    # Each heldout semantic role label is identical across bare and rich
    # expressions, including full candidate pools and label ordering.
    for pop in ("unseen_roles","unseen_quartets"):
        for voice in VOICES:
            bare=tests[pop+"_bare"][voice]
            rich=tests[pop+"_rich"][voice]
            assert torch.equal(bare[2],rich[2])
            assert torch.equal(bare[3],rich[3])
    print("EVAL-SOURCE-ONLY PREFLIGHT PASS: all populations, 4 role arms, both voices")
    print("No model training or inference; original DEV/reserved/protected/LI untouched")

def smoke(device):
    train,held,training,withheld,vocab,manifest=verified()
    _=(held,withheld,manifest)
    _,quartets,events=next(schedule(SEEDS[0],train))
    rows=make_batch(events,training)
    torch.manual_seed(1610026)
    for arm in ARMS:
        model=old.Model(len(vocab)).to(device)
        encoded=old.encode_sources(rows[arm],vocab,check=True)
        assert encoded[0].shape[0]==64
        with torch.inference_mode():
            assert model(encoded[0].to(device),
                         encoded[1].to(device)).shape==(64,len(vocab))
        opt=torch.optim.AdamW(model.parameters(),lr=old.LR,weight_decay=old.WD)
        loss=old.one_update(model,opt,encoded,device)
        assert math.isfinite(loss)
        print("SMOKE TRAIN-ONLY",arm,"loss",loss)
    print("SMOKE PASS: only current TRAIN-eligible source roles; no heldout inference")

def run(seed,device):
    if seed not in SEEDS:raise ValueError("Not a frozen new seed")
    if device.type!="mps":raise ValueError("Scientific training requires Apple MPS")
    report=OUT/("seed%d_mps_step6400.json"%seed)
    weights=WEIGHTS/("seed%d_mps_step6400.pt"%seed)
    if report.exists() or weights.exists():
        raise FileExistsError("Already sealed or partial model: refuse to overwrite")
    train,held,training,withheld,vocab,m=verified()
    frozen_commit=require_frozen()
    tests=evaluations(train,held,training,withheld,vocab)
    torch.manual_seed(seed)
    initial=old.Model(len(vocab)).to(device)
    nets={}
    for arm in ARMS:
        net=deepcopy(initial).to(device)
        assert all(torch.equal(p,q) for p,q in zip(net.parameters(),initial.parameters()))
        nets[arm]=(net,torch.optim.AdamW(net.parameters(),
                                  lr=old.LR,weight_decay=old.WD))
    history=[];trace=sha256()
    for step,chosen,events in schedule(seed,train):
        rows=make_batch(events,training)
        losses={}
        for arm in ARMS:
            encoded=old.encode_sources(rows[arm],vocab,check=False)
            losses[arm]=old.one_update(*nets[arm],encoded,device)
        trace.update((json.dumps((step,chosen,events),separators=(",",":"))+"\n").encode())
        if step in MILESTONES:
            history.append({"step":step,"losses":losses})
            print("STEP",step,losses,flush=True)
    assert trace.hexdigest()==m["protocol"]["schedule_sha256"][str(seed)]
    outcomes={}
    for arm in ARMS:
        model=nets[arm][0].eval()
        outcomes[arm]={}
        for pop in POPS:
            data=(tests[pop][arm] if pop=="train_seen_rich" else tests[pop])
            outcomes[arm][pop]=old.score_native(model,data,device)
        print("SCORED",arm,flush=True)
    OUT.mkdir(parents=True,exist_ok=True);WEIGHTS.mkdir(parents=True,exist_ok=True)
    torch.save({
       "study":"ROLE-DIVERSITY-COMPLEX-v0","seed":seed,
       "steps":STEPS,"protocol_digest":m["protocol_digest"],
       "models":{arm:nets[arm][0].state_dict() for arm in ARMS}
    },weights)
    save_once(report,{
       "study":"ROLE-DIVERSITY-COMPLEX-v0",
       "seed":seed,"steps":STEPS,"status":"SEALED_NEW_SEED",
       "protocol_digest":m["protocol_digest"],
       "schedule_sha256":trace.hexdigest(),
       "frozen_commit":frozen_commit,
       "no_old_dev_reserved_protected_LI":True,
       "loss_history":history,
       "outcomes":outcomes
    })
    print("SEALED",report,"No aggregation until all 16 seeds complete")

def score(report,arm,pop):
    return mean(report["outcomes"][arm][pop][v]["C_query"] for v in VOICES)

def ci(a):
    assert len(a)==len(SEEDS)
    m=mean(a);d=T95*stdev(a)/math.sqrt(len(a))
    return {"mean":m,"ci95":[m-d,m+d],
            "positive_seeds":sum(x>0 for x in a),
            "per_seed":dict(zip(map(str,SEEDS),a))}

def summary():
    train,held,training,withheld,vocab,m=verified()
    _=(train,held,training,withheld,vocab)
    reports={}
    for seed in SEEDS:
        path=OUT/("seed%d_mps_step6400.json"%seed)
        if not path.exists():
            print("INCOMPLETE: missing seed",seed,"No partial aggregation")
            return
        r=json.loads(path.read_text())
        assert r["study"]=="ROLE-DIVERSITY-COMPLEX-v0"
        assert r["seed"]==seed and r["steps"]==STEPS
        assert r["protocol_digest"]==m["protocol_digest"]
        assert r["schedule_sha256"]==m["protocol"]["schedule_sha256"][str(seed)]
        assert r["no_old_dev_reserved_protected_LI"]
        assert set(r["outcomes"])==set(ARMS)
        assert all(set(r["outcomes"][a])==set(POPS) for a in ARMS)
        reports[seed]=r
    def contrast(hi,lo,pop):
        return ci([score(reports[s],hi,pop)-score(reports[s],lo,pop) for s in SEEDS])
    primary=contrast("roles_8","roles_1","unseen_roles_rich")
    passes=(primary["ci95"][0]>0 and primary["mean"]>=.10
            and primary["positive_seeds"]>=12)
    gated=contrast("roles_2","roles_1","unseen_roles_rich") if passes else None
    gated_pass=(gated is not None and gated["ci95"][0]>0 and gated["mean"]>=.10)
    metrics=("C_query","accuracy4","full_vocab_accuracy")
    descriptive={
      p:{a:{k:ci([mean(reports[s]["outcomes"][a][p][v][k] for v in VOICES)
                          for s in SEEDS]) for k in metrics} for a in ARMS}
      for p in POPS}
    events={
      p:{a:{str(event):{name:ci([
          mean(reports[s]["outcomes"][a][p][v]
               ["event%d_pair_classes"%event][name] for v in VOICES)
          for s in SEEDS]) for name in
          ("correct","reversed","collapsed","wrong_event_name")}
          for event in (1,2)} for a in ARMS} for p in POPS}
    save_once(OUT/"analysis_summary.json",{
      "study":"ROLE-DIVERSITY-COMPLEX-v0",
      "status":"COMPLETE_FROZEN_PROSPECTIVE",
      "protocol_digest":m["protocol_digest"],
      "seeds":list(SEEDS),
      "primary":{"contrast":"roles_8 - roles_1, rich unseen role permutations",
                 "estimate":primary,"passes":passes},
      "gated_secondary":{
           "confirmatory_gate_open":passes,
           "contrast":"roles_2 - roles_1, rich unseen role permutations",
           "estimate":gated,"passes":gated_pass if passes else None},
      "exploratory":{"roles_4_vs_1":contrast("roles_4","roles_1","unseen_roles_rich"),
                     "all_arm_outcomes":descriptive,
                     "per_event_pair_classes":events}})
    print("SUMMARY PASS: 16 independent prospective seeds / 64 new models")
    print("PRIMARY",primary,"passes",passes)
    print("GATED SECONDARY",gated if passes else "not opened")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,
                    choices=("prepare","eval_preflight","smoke","run","summary"))
    ap.add_argument("--seed",type=int)
    ap.add_argument("--device",choices=("cpu","mps"),default="cpu")
    a=ap.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="eval_preflight":eval_preflight()
    elif a.phase=="smoke":smoke(torch.device(a.device))
    elif a.phase=="run":
        if a.seed is None:ap.error("run requires --seed")
        run(a.seed,torch.device(a.device))
    else:summary()

if __name__=="__main__":main()
