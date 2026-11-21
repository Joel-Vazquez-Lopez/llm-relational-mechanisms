#!/usr/bin/env python3
"""Post-hoc read-only CORE-TRANSFER audit; no model training or protected data."""
import argparse,json,math,subprocess
from collections import Counter
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean,stdev
import torch
import position_counterbalance_v0 as cb

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
LAUNCHER=ROOT/"scripts/run_position_cb_core_transfer_v0.sh"
PROTOCOL=ROOT/"paper/POSITION_CB_CORE_TRANSFER_V0_PROTOCOL.md"
OUT=ROOT/"reports/position_cb_core_transfer_v0"
MANIFEST=OUT/"manifest.json"
TIERS={"original":("original",),
       "role_flip":("flip_first","flip_second","flip_both"),
       "event_swap":("swap","swap_flip_first","swap_flip_second","swap_flip_both"),
       "repartition":("cross_ac","cross_bd","cross_ad","cross_bc")}
VARIANTS=tuple(v for ts in TIERS.values() for v in ts)
SELECTION_SEED=1282026
T95=2.364624251

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return cb.digest(x)

def variants(gold,i):
    a,b,c,d=gold
    roles={"original":gold,"flip_first":(b,a,c,d),
           "flip_second":(a,b,d,c),"flip_both":(b,a,d,c),
           "swap":(c,d,a,b),"swap_flip_first":(d,c,a,b),
           "swap_flip_second":(c,d,b,a),"swap_flip_both":(d,c,b,a)}
    other=[p for p in permutations(gold)
           if p!=gold and p not in roles.values()]
    assert len(other)==16
    for label,pair in (("cross_ac",(a,c)),("cross_bd",(b,d)),
                       ("cross_ad",(a,d)),("cross_bc",(b,c))):
        pool=[p for p in other if frozenset(p[:2])==frozenset(pair)]
        assert len(pool)==4
        roles[label]=pool[i%4]
    assert set(roles)==set(VARIANTS)
    assert len(set(roles.values()))==12
    return roles

def events_for(original,order):
    a,b,c,d=order
    x,y=original
    return (cb.gen.Event(a,b,x.verb,"A",c,d,x.layout),
            cb.gen.Event(c,d,y.verb,"A",a,b,y.layout))

def generate(dose):
    train,dev,partition=cb.gen.partitions()
    assert len(train)==300 and len(dev)==100
    lookup={}
    for j,core_id,ev in cb.gen.cores(train,"train"):
        if j==2:
            key=int(core_id.split("-")[1])
            assert key not in lookup
            lookup[key]=(core_id,ev)
    assert len(lookup)==300
    keys=list(sorted(lookup))
    Random(SELECTION_SEED).shuffle(keys)
    selected=keys[:144]
    assert len(set(selected))==144
    train_text={s.text for row in dose for s in row.values()}
    unframed={}
    prefixes=tuple(p for p in cb.frames.FRAMES if p)
    for row in dose:
        s=row["AA"]
        if not any(s.text.startswith(p) for p in prefixes):
            key=(s.core_id,s.mods)
            assert key not in unframed
            unframed[key]=row
    assert len(unframed)==2700
    groups=[];new_text=set();old_text=set()
    for ix,index in enumerate(selected):
        core_id,es=lookup[index]
        mods=cb.gen.TRAIN_MODS[ix%9]
        gold=tuple(n for e in es for n in (e.agent,e.patient))
        assert len(set(gold))==4
        row={"index":index,"mods":mods,"sources":{}}
        for variant,permutation in variants(gold,ix).items():
            rendered=events_for(es,permutation)
            row["sources"][variant]={}
            for voice in cb.DIRECTIONS:
                s=cb.gen.make_source(rendered,voice,mods,"core_transfer_train",
                                     "core-audit-%d-%s"%(index,variant))
                assert s.gold==permutation and set(s.gold)==set(gold)
                assert cb.gen.parse_independent(s.text)==s.gold
                assert not any(m=="LI" for m in s.mods)
                assert len(["<bos>"]+cb.gen.tokens(s.text)+["<sep>"]
                           +cb.gen.tokens(cb.gen.Q[0])+["<answer>"])<=cb.pilot.MAX_TOKENS
                if variant=="original":
                    assert s.text==unframed[(core_id,mods)][voice].text
                    assert s.text in train_text
                    old_text.add(s.text)
                else:
                    assert s.text not in train_text and s.text not in new_text
                    new_text.add(s.text)
                row["sources"][variant][voice]=s
        for voice in cb.DIRECTIONS:
            all_lengths={len(cb.gen.tokens(row["sources"][v][voice].text))
                         for v in VARIANTS}
            assert len(all_lengths)==1
            if mods==("bare","bare"):
                bags=[Counter(cb.gen.tokens(row["sources"][v][voice].text))
                      for v in VARIANTS]
                assert all(x==bags[0] for x in bags)
        groups.append(row)
    assert len(groups)==144 and len(old_text)==144*4 and len(new_text)==144*11*4
    assert set(Counter(g["mods"] for g in groups).values())=={16}
    material=[(g["index"],g["mods"],[
        (v,[(d,g["sources"][v][d].text,g["sources"][v][d].gold)
             for d in cb.DIRECTIONS]) for v in VARIANTS]) for g in groups]
    audit={"anchors":144,"per_modifier":16,"variants":12,
           "presentations":6912,"new_presentations":6336,
           "original_presentations":576,
           "no_new_training_surface_overlap":True,
           "fixed_voice_and_modifier_lengths_matched":True,
           "bare_bare_exact_token_bags":True,
           "no_reserved_protected_or_LI":True,
           "selection_seed":SELECTION_SEED,
           "partition_digest":partition["partition_digest"],
           "source_sha256":digest(material),
           "selected_index_sha256":digest(selected)}
    return groups,audit

def generate_manifest():
    parent,(dose,_dp,_dev,vocab),_g=cb.load_and_verify()
    groups,audit=generate(dose)
    deps=("position_counterbalance_v0.py","exp105_generator_preflight.py",
          "exp105_development_pilot_v0.py","exp112_clause_state_causal_patch_v0.py",
          "exp113_position_update_factorial_v0.py")
    protocol={"study":"POSITION-CB-CORE-TRANSFER-v0",
              "status":"POSTHOC_READONLY_FROZEN_BEFORE_CHECKPOINT_INFERENCE",
              "parent_protocol_digest":parent["protocol_digest"],
              "parent_eval_digest":parent["evaluation_audit"]["evaluation_sha256"],
              "sources_sha256":audit["source_sha256"],
              "vocab_sha256":digest(sorted(vocab.items())),
              "seeds":list(cb.SEEDS),"repeats":list(cb.REPEATS),
              "tiers":{k:list(v) for k,v in TIERS.items()},
              "fixed_selection_seed":SELECTION_SEED,
              "no_checkpoint_selection":True,"no_training_or_new_dev_scoring":True,
              "no_protected_or_LI":True,
              "statistical_unit":"seed, average two MPS repeats first",
              "interpretation":"post-hoc exploration, descriptive CIs only",
              "script_sha256":sha(SCRIPT),"launcher_sha256":sha(LAUNCHER),
              "protocol_sha256":sha(PROTOCOL),
              "dependency_sha256":{d:sha(ROOT/"scripts"/d) for d in deps}}
    manifest={"protocol":protocol,"protocol_digest":digest(protocol),
              "source_audit":audit}
    return manifest,parent,vocab,groups

def prepare():
    m,_,_,_=generate_manifest()
    cb.write_once(MANIFEST,m)
    print("PREFLIGHT PASS",m["source_audit"]["source_sha256"])
    print("144 TRAIN four-name sets / 6912 parsed source presentations")
    print("NO new inference; protected/reserved/L+I CLOSED")
    print("Commit this manifest BEFORE loading trained model checkpoints")

def verified():
    if not MANIFEST.exists():raise FileNotFoundError("Run prepare and COMMIT manifest")
    m,parent,vocab,groups=generate_manifest()
    assert json.loads(MANIFEST.read_text())==m,"Frozen protocol/source drift"
    return m,parent,vocab,groups

def freeze():
    frozen=("scripts/position_cb_core_transfer_v0.py",
            "scripts/run_position_cb_core_transfer_v0.sh",
            "paper/POSITION_CB_CORE_TRANSFER_V0_PROTOCOL.md",
            "reports/position_cb_core_transfer_v0/manifest.json")
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        for rel in frozen:
            original=subprocess.check_output(["git","show",head+":"+rel],
                                             cwd=ROOT,stderr=subprocess.DEVNULL)
            assert original==(ROOT/rel).read_bytes(),("Not committed",rel)
        return head
    except (FileNotFoundError,subprocess.CalledProcessError,AssertionError) as err:
        raise RuntimeError("Commit exact four frozen files BEFORE checkpoint scoring") from err

def original_dev(seed,repeat,arm,parent):
    p=ROOT/"reports/position_cb_v0"/(
        "seed%d_repeat%d_mps_step%d.json"%(seed,repeat,cb.STEPS))
    rec=json.loads(p.read_text())
    assert (rec["seed"],rec["repeat"])==(seed,repeat)
    assert rec["protocol_digest"]==parent["protocol_digest"]
    assert rec["evaluation_sha256"]==parent["evaluation_audit"]["evaluation_sha256"]
    familiar,_=cb.label_sets(arm)
    return {k:mean(rec["outcomes"][arm][v][k] for v in familiar)
            for k in ("C_query","accuracy4","full_vocab_accuracy")}

def score(model,groups,variant,arm,vocab,device):
    familiar,_=cb.label_sets(arm)
    def calc(rows):
        enc={v:cb.pilot.encode_sources([g["sources"][variant][v] for g in rows],
                                      vocab,check=True) for v in familiar}
        scores=cb.score(model,enc,device)
        result={}
        for metric in ("C_query","accuracy4","full_vocab_accuracy"):
            result[metric]=mean(scores[v][metric] for v in familiar)
        for event in (1,2):
            for kind in ("correct","reversed","collapsed","wrong_event_name"):
                result["event%d_%s"%(event,kind)]=mean(
                    scores[v]["event%d_pair_classes"%event][kind] for v in familiar)
        return result
    bare=[g for g in groups if g["mods"]==("bare","bare")]
    assert len(bare)==16
    return {"all":calc(groups),"bare_bare":calc(bare)}

def smoke(device):
    _man,(_dose,_dp,_dev,vocab),_original=cb.load_and_verify()
    dose,_dp,_dev,vocab=(_dose,_dp,_dev,vocab)
    groups,_audit=generate(dose)
    model=cb.pilot.Model(len(vocab)).to(device).eval()
    familiar,_=cb.label_sets("same_voice")
    enc={v:cb.pilot.encode_sources([groups[0]["sources"]["flip_first"][v]],
                                   vocab,check=True) for v in familiar}
    out=cb.score(model,enc,device)
    assert all(0<=out[v]["C_query"]<=1 for v in familiar)
    print("SMOKE PASS: random weights only; no saved checkpoint or dev score")

def report_path(seed,repeat):
    return OUT/("seed%d_repeat%d_core_audit.json"%(seed,repeat))

def run(seed,repeat,device):
    if device.type!="mps":raise ValueError("Use MPS for original checkpoint audit")
    if seed not in cb.SEEDS or repeat not in cb.REPEATS:
        raise ValueError("Only original frozen seeds/repeats")
    path=report_path(seed,repeat)
    if path.exists():print("SKIP existing sealed",path);return
    m,parent,vocab,groups=verified()
    head=freeze()
    cp=ROOT/"results/position_cb_v0"/(
        "seed%d_repeat%d_mps_step%d.pt"%(seed,repeat,cb.STEPS))
    if not cp.exists():raise FileNotFoundError("Original checkpoint missing: "+str(cp))
    state=torch.load(cp,map_location=device,weights_only=True)
    assert state["study"]=="POSITION-CB-v0"
    assert (state["seed"],state["repeat"],state["step"])==(seed,repeat,cb.STEPS)
    assert state["protocol_digest"]==parent["protocol_digest"]
    assert set(state["models"])==set(cb.ARMS)
    result={}
    for arm in cb.ARMS:
        model=cb.pilot.Model(len(vocab)).to(device)
        model.load_state_dict(state["models"][arm],strict=True)
        model.eval()
        result[arm]={"variants":{
            variant:score(model,groups,variant,arm,vocab,device)
            for variant in VARIANTS},
            "archived_dev_familiar":original_dev(seed,repeat,arm,parent)}
        del model
    record={"study":"POSITION-CB-CORE-TRANSFER-v0",
            "status":"SEALED_READ_ONLY","seed":seed,"repeat":repeat,
            "commit":head,"protocol_digest":m["protocol_digest"],
            "source_sha256":m["source_audit"]["source_sha256"],
            "parent_protocol_digest":parent["protocol_digest"],
            "no_training_or_new_dev_scoring":True,"no_protected_or_LI":True,
            "results":result}
    cb.write_once(path,record)
    print("SEALED",path)

def ci(values):
    assert len(values)==8
    avg=mean(values)
    r=T95*stdev(values)/math.sqrt(len(values))
    return {"mean":avg,"ci95":[avg-r,avg+r],
            "per_seed":dict(zip(map(str,cb.SEEDS),values)),
            "positive_seeds":sum(x>0 for x in values)}

def summary():
    m,parent,_v,_g=verified()
    records={};missing=[]
    for seed in cb.SEEDS:
        for repeat in cb.REPEATS:
            path=report_path(seed,repeat)
            if not path.exists():missing.append(str(path));continue
            r=json.loads(path.read_text())
            assert (r["seed"],r["repeat"])==(seed,repeat)
            assert r["protocol_digest"]==m["protocol_digest"]
            assert r["source_sha256"]==m["source_audit"]["source_sha256"]
            assert r["parent_protocol_digest"]==parent["protocol_digest"]
            assert r["no_training_or_new_dev_scoring"] and r["no_protected_or_LI"]
            records[seed,repeat]=r
    if missing:
        print("INCOMPLETE",len(missing),"out of 16; NO summary")
        return
    def across(fn):return ci([mean(fn(records[s,r]) for r in cb.REPEATS)
                             for s in cb.SEEDS])
    def metric(rec,arm,tier,pop,name):
        if tier=="archived_dev":
            assert pop=="all"
            return rec["results"][arm]["archived_dev_familiar"][name]
        return mean(rec["results"][arm]["variants"][v][pop][name]
                    for v in TIERS[tier])
    arm_results={}
    for arm in cb.ARMS:
        arm_results[arm]={}
        for pop in ("all","bare_bare"):
            arm_results[arm][pop]={}
            for tier in TIERS:
                arm_results[arm][pop][tier]={
                    name:across(lambda r,a=arm,t=tier,p=pop,k=name:
                                metric(r,a,t,p,k))
                    for name in ("C_query","accuracy4","full_vocab_accuracy",
                                 "event1_correct","event2_correct",
                                 "event1_reversed","event2_reversed",
                                 "event1_collapsed","event2_collapsed",
                                 "event1_wrong_event_name","event2_wrong_event_name")}
            arm_results[arm][pop]["paired_contrasts"]={
                "original_minus_"+tier:{
                    name:across(lambda r,a=arm,t=tier,p=pop,k=name:
                                metric(r,a,"original",p,k)-metric(r,a,t,p,k))
                    for name in ("C_query","accuracy4")}
                for tier in ("role_flip","event_swap","repartition")}
        arm_results[arm]["dev_familiar"]={
            name:across(lambda r,a=arm,k=name:metric(r,a,"archived_dev","all",k))
            for name in ("C_query","accuracy4","full_vocab_accuracy")}
        arm_results[arm]["per_variant_C_query"]={
            v:across(lambda r,a=arm,x=v:
                     r["results"][a]["variants"][x]["all"]["C_query"])
            for v in VARIANTS}
    pooled={("original_minus_"+tier):across(
        lambda r,t=tier:mean(metric(r,a,"original","all","C_query")
                            -metric(r,a,t,"all","C_query") for a in cb.ARMS))
        for tier in ("role_flip","event_swap","repartition")}
    final={"study":"POSITION-CB-CORE-TRANSFER-v0",
           "status":"COMPLETE_POSTHOC_READ_ONLY",
           "source_reports":16,"models":32,"independent_seeds":8,
           "no_training_or_new_dev_scoring":True,"no_protected_or_LI":True,
           "source_sha256":m["source_audit"]["source_sha256"],
           "protocol_digest":m["protocol_digest"],
           "seed_unit":"average numerical repetitions within seed",
           "arms":arm_results,"pooled_contrasts":pooled,
           "limitations":"Within familiar TRAIN four-name quartets; adjunct name frequencies may shift; bare/bare token bags exactly matched. Post-hoc descriptive only."}
    cb.write_once(OUT/"analysis_summary.json",final)
    print("SUMMARY PASS 16 processes / 32 frozen models / 8 independent seeds")
    for arm in cb.ARMS:
        for tier in TIERS:
            print(arm,tier,round(arm_results[arm]["all"][tier]["C_query"]["mean"],4))
        print(arm,"existing dev",round(arm_results[arm]["dev_familiar"]["C_query"]["mean"],4))
    for key,value in pooled.items():
        print(key,"gap",round(100*value["mean"],2),"pp CI",
              [round(100*x,2) for x in value["ci95"]])
    print("EXPLORATORY: no trained seed exclusion; no unique role-circuit claim")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,
                   choices=("prepare","smoke","run","summary"))
    p.add_argument("--seed",type=int);p.add_argument("--repeat",type=int)
    p.add_argument("--device",choices=("cpu","mps"),default="cpu")
    a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="smoke":smoke(torch.device(a.device))
    elif a.phase=="run":
        if a.seed is None or a.repeat is None:
            p.error("run requires seed and repeat")
        run(a.seed,a.repeat,torch.device(a.device))
    else:summary()

if __name__=="__main__":main()
