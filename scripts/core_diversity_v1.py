#!/usr/bin/env python3
"""CORE-DIVERSITY-v1 prospective 2x2 experiment. Commit manifests before RUN.

Fresh scratch training only, 16 new independent seeds. Source material is
reconstructed EXCLUSIVELY from 300 original TRAIN name-set identifiers. Never
train or score original dev, reserved, protected, or within-event L+I.
"""
import argparse,json,math,subprocess
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean,stdev
import torch
import core_diversity_v1_preflight as pf
import position_counterbalance_v0 as cb

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
LAUNCHER=ROOT/"scripts/run_core_diversity_v1.sh"
PROTOCOL=ROOT/"paper/CORE_DIVERSITY_V1_FROZEN_PROTOCOL.md"
SOURCE=pf.MANIFEST
OUT=ROOT/"reports/core_diversity_v1"
STUDY=OUT/"study_manifest.json"
WEIGHTS=ROOT/"results/core_diversity_v1"
SEEDS=(13901,13903,13907,13909,13913,13919,13921,13927,
       13931,13933,13939,13943,13949,13951,13959,13963)
STEPS=6400
VOICES={"same_voice":("AA","PP"),"mixed_voice":("AP","PA")}
ARMS=("same_voice_fixed","same_voice_diverse","mixed_voice_fixed","mixed_voice_diverse")
POPS=("train_original","unseen_roles","unseen_quartets")
T95=2.131449545559323
MILESTONES=(800,1600,2400,3200,4000,4800,5600,6400)

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return pf.digest(obj)
def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(obj,f,sort_keys=True,indent=2)
        f.write("\n")

def fixtures():
    if not SOURCE.exists():
        raise FileNotFoundError("Run SOURCE preflight prepare first")
    sm=json.loads(SOURCE.read_text())
    if sm!=pf.build():raise RuntimeError("Source manifest drift")
    name_sets,_dev,_part=pf.gen.partitions()
    original={}
    for j,cid,events in pf.gen.cores(name_sets,"train"):
        if j==2:
            ix=int(cid.split("-")[1])
            assert ix not in original
            original[ix]=events
    assert len(original)==300
    ids=sorted(original)
    Random(pf.SOURCE_SEED).shuffle(ids)
    train_ids=tuple(ids[:240]);withheld_ids=tuple(ids[240:])
    assert len(withheld_ids)==60
    assert digest(list(train_ids))==sm["train_identifiers_sha256"]
    assert digest(list(withheld_ids))==sm["withheld_identifiers_sha256"]
    categories=("train_permutations","evaluation_train_quartets",
                "evaluation_withheld_quartets")
    tables={k:{} for k in categories}
    signatures={k:[] for k in categories}
    def render(ix,cat,orders):
        rendered=[]
        for ordinal,roles in enumerate(orders):
            voices={}
            for voice in pf.VOICES:
                src=pf.gen.make_source(pf.new_events(original[ix],roles),voice,
                    pf.BARE,cat,"core-div-v1-"+str(ix))
                assert src.gold==roles
                voices[voice]=src
                signatures[cat].append((ix,ordinal,voice,src.text,tuple(src.gold)))
            for voice in pf.VOICES:
                assert Counter(pf.gen.tokens(voices[voice].text))==Counter(
                    pf.gen.tokens(pf.gen.make_source(
                        pf.new_events(original[ix],orders[0]),voice,
                        pf.BARE,cat,"ref").text))
            aa,pp,ap,pa=(voices[v] for v in ("AA","PP","AP","PA"))
            assert Counter(pf.gen.tokens(aa.text)+pf.gen.tokens(pp.text))==Counter(
                pf.gen.tokens(ap.text)+pf.gen.tokens(pa.text))
            rendered.append(voices)
        tables[cat][ix]=tuple(rendered)
    for ix in train_ids:
        gold=tuple(n for e in original[ix] for n in (e.agent,e.patient))
        eight,sixteen=pf.all_roles(gold)
        render(ix,"train_permutations",eight)
        render(ix,"evaluation_train_quartets",sixteen)
    for ix in withheld_ids:
        gold=tuple(n for e in original[ix] for n in (e.agent,e.patient))
        render(ix,"evaluation_withheld_quartets",tuple(permutations(gold)))
    for cat in categories:
        if digest(signatures[cat])!=sm["source_hashes"][cat]:
            raise RuntimeError("Source samples disagree with frozen preflight: "+cat)
    words=set()
    for groups in tables["train_permutations"].values():
        for voices in groups:
            for src in voices.values():words.update(pf.gen.tokens(src.text))
    for q in pf.gen.Q:words.update(pf.gen.tokens(q))
    special=("<pad>","<bos>","<sep>","<answer>")
    vocab={w:i for i,w in enumerate(special+tuple(sorted(words-set(special))))}
    for cat in categories:
        for groups in tables[cat].values():
            for vs in groups:
                for s in vs.values():
                    assert set(pf.gen.tokens(s.text)).issubset(vocab)
                    assert len(["<bos>"]+pf.gen.tokens(s.text)+["<sep>"]
                               +pf.gen.tokens(pf.gen.Q[0])+["<answer>"])<=48
    return train_ids,withheld_ids,tables,vocab,sm

def schedule(seed,ids):
    assert seed in SEEDS
    rng=Random(seed+1393101)
    counters={ix:(seed+ix)%8 for ix in ids}
    for step in range(1,STEPS+1):
        chosen=tuple(rng.sample(ids,8))
        choices=tuple(counters[ix]%8 for ix in chosen)
        for ix in chosen:counters[ix]+=1
        yield step,chosen,choices

def schedule_sha(seed,ids):
    h=sha256();freq=Counter()
    for step,chosen,choices in schedule(seed,ids):
        assert len(set(chosen))==8
        for ix,k in zip(chosen,choices):freq[ix,k]+=1
        h.update((json.dumps((step,chosen,choices),separators=(",",":"))+"\n").encode())
    for ix in ids:
        n=[freq[ix,k] for k in range(8)]
        assert min(n)>0 and max(n)-min(n)<=1
    return h.hexdigest()

def expected_manifest(ids,hids,vocab,sm):
    deps=("exp105_generator_preflight.py","exp105_development_pilot_v0.py",
          "exp105_position_control_preflight_v1.py",
          "exp112_clause_state_causal_patch_v0.py",
          "exp113_position_update_factorial_v0.py",
          "position_counterbalance_v0.py",
          "core_diversity_v1_preflight.py")
    protocol={
        "study":"CORE-DIVERSITY-v1","status":"PROSPECTIVE_NOT_RUN",
        "train_quartets":240,"heldout_quartets":60,
        "training_eligible_roles":8,"roles_heldout_per_training_quartet":16,
        "heldout_roles_per_heldout_quartet":24,
        "source_manifest_sha256":sha(SOURCE),
        "source_fixture_hashes":sm["source_hashes"],
        "train_quartet_ids_sha256":digest(list(ids)),
        "withheld_quartet_ids_sha256":digest(list(hids)),
        "vocab_sha256":digest(sorted(vocab.items())),
        "seeds":list(SEEDS),"steps":STEPS,"quartets_per_step":8,
        "arms":list(ARMS),"optimizer":"AdamW lr=0.002 wd=0.01",
        "model":"three layer width96 four attention head causal transformer",
        "paired_initializations":True,
        "training_examples_per_update_per_arm":16,
        "primary":"diverse_minus_fixed C_query on 16 never-trained "
                  "role permutations per 240 training quartets, paired across voices",
        "primary_success":"two sided t 95pct CI lower positive, "
                          "effect at least 10pp and positive in at least 12/16 seeds",
        "gated_secondary":"diverse_minus_fixed C_query on all 24 roles "
                          "of 60 withheld name quartets, test CI lower>0 if primary passes",
        "analysis_unit":"independent seed, no numerical repeats",
        "no_dev_reserved_protected_LI":True,
        "no_checkpoint_or_seed_selection":True,
        "script_sha256":sha(SCRIPT),"launcher_sha256":sha(LAUNCHER),
        "protocol_sha256":sha(PROTOCOL),
        "dependency_sha256":{d:sha(ROOT/"scripts"/d) for d in deps},
        "schedule_sha256":{str(seed):schedule_sha(seed,ids) for seed in SEEDS}}
    return {"protocol":protocol,"protocol_digest":digest(protocol)}

def prepare():
    ids,hids,tables,vocab,sm=fixtures()
    _=tables
    m=expected_manifest(ids,hids,vocab,sm)
    if STUDY.exists():
        if json.loads(STUDY.read_text())!=m:raise RuntimeError("Study manifest drift")
        print("PREPARE VERIFIED existing manifest")
    else:
        write_once(STUDY,m)
        print("PREPARE PASS study manifest created")
    print("Source/control/gold/word-bag checks PASS")
    print("Commit both source and study manifests BEFORE inference")
    print("Digest:",m["protocol_digest"])
    print("No model training, no protected or old-development scoring")

def verify():
    ids,hids,tables,vocab,sm=fixtures()
    m=expected_manifest(ids,hids,vocab,sm)
    if not STUDY.exists() or json.loads(STUDY.read_text())!=m:
        raise RuntimeError("Freeze study manifest before inference")
    return ids,hids,tables,vocab,m

def frozen():
    paths=("scripts/core_diversity_v1_preflight.py",
           "scripts/core_diversity_v1.py",
           "scripts/run_core_diversity_v1.sh",
           "paper/CORE_DIVERSITY_V1_FROZEN_PROTOCOL.md",
           "reports/core_diversity_v1/source_manifest.json",
           "reports/core_diversity_v1/study_manifest.json")
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],
                                     cwd=ROOT,text=True).strip()
        for path in paths:
            committed=subprocess.check_output(
                ["git","show",head+":"+path],cwd=ROOT,stderr=subprocess.DEVNULL)
            if committed!=(ROOT/path).read_bytes():
                raise RuntimeError("Uncommitted frozen input: "+path)
    except (subprocess.CalledProcessError,FileNotFoundError) as exc:
        raise RuntimeError("Commit frozen study protocol BEFORE model training") from exc
    return head

def generate_batch(chosen,role_choices,tables):
    samples={arm:[] for arm in ARMS}
    for ix,choice in zip(chosen,role_choices):
        core=tables["train_permutations"][ix]
        for regime,voices in VOICES.items():
            fixed=[core[0][voice] for voice in voices]
            diverse=[core[choice][voice] for voice in voices]
            assert Counter(w for s in fixed for w in pf.gen.tokens(s.text))==Counter(
                w for s in diverse for w in pf.gen.tokens(s.text))
            samples[regime+"_fixed"].extend(fixed)
            samples[regime+"_diverse"].extend(diverse)
    assert all(len(s)==16 for s in samples.values())
    return samples

def smoke(device):
    ids,hids,tables,vocab,m=verify()
    _=(hids,m)
    batch=generate_batch(ids[:8],(0,1,2,3,4,5,6,7),tables)
    model=cb.pilot.Model(len(vocab)).to(device).eval()
    for arm in ARMS:
        data=cb.pilot.encode_sources(batch[arm],vocab,check=True)
        with torch.inference_mode():
            logits=model(data[0].to(device),data[1].to(device))
        assert logits.shape==(64,len(vocab))
    print("SMOKE PASS random weights + source encoding. NO study model trained.")

def evaluate_inputs(ids,hids,tables,vocab):
    populations={
        "train_original":[tables["train_permutations"][ix][0] for ix in ids],
        "unseen_roles":[roles for ix in ids
                        for roles in tables["evaluation_train_quartets"][ix]],
        "unseen_quartets":[roles for ix in hids
                           for roles in tables["evaluation_withheld_quartets"][ix]]}
    encoded={}
    for regime,voices in VOICES.items():
        encoded[regime]={}
        for pop,rows in populations.items():
            encoded[regime][pop]={
                voice:cb.pilot.encode_sources([r[voice] for r in rows],vocab,check=False)
                for voice in voices}
    return encoded

def regime(arm):return arm.rsplit("_",1)[0]

def run(seed,device):
    if seed not in SEEDS:raise ValueError("Unknown fresh independent seed")
    if device.type!="mps":raise ValueError("Study training requires MPS")
    report=OUT/("seed%d_mps_step%d.json"%(seed,STEPS))
    checkpoint=WEIGHTS/("seed%d_mps_step%d.pt"%(seed,STEPS))
    if report.exists() or checkpoint.exists():
        raise FileExistsError("Refuse to overwrite existing report or checkpoint")
    ids,hids,tables,vocab,manifest=verify()
    commit=frozen()
    encoded=evaluate_inputs(ids,hids,tables,vocab)
    torch.manual_seed(seed)
    initial=cb.pilot.Model(len(vocab)).to(device)
    nets={}
    for arm in ARMS:
        model=deepcopy(initial).to(device)
        assert all(torch.equal(a,b) for a,b in zip(initial.parameters(),
                                                   model.parameters()))
        nets[arm]=(model,torch.optim.AdamW(
            model.parameters(),lr=cb.pilot.LR,weight_decay=cb.pilot.WEIGHT_DECAY))
    trace=sha256();history=[]
    for step,chosen,choices in schedule(seed,ids):
        batches=generate_batch(chosen,choices,tables)
        losses={}
        for arm in ARMS:
            data=cb.pilot.encode_sources(batches[arm],vocab,check=False)
            losses[arm]=cb.pilot.one_update(*nets[arm],data,device)
        trace.update((json.dumps((step,chosen,choices),separators=(",",":"))+"\n").encode())
        if step in MILESTONES:
            history.append({"step":step,"losses":losses})
            print("STEP",step," ".join("%s:%.4f"%(k,v) for k,v in losses.items()),
                  flush=True)
    assert trace.hexdigest()==manifest["protocol"]["schedule_sha256"][str(seed)]
    outputs={}
    for arm in ARMS:
        net=nets[arm][0].eval()
        outputs[arm]={pop:cb.score(net,encoded[regime(arm)][pop],device)
                      for pop in POPS}
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({"study":"CORE-DIVERSITY-v1","seed":seed,
                "step":STEPS,"protocol_digest":manifest["protocol_digest"],
                "models":{arm:nets[arm][0].state_dict() for arm in ARMS}},
               checkpoint)
    write_once(report,{"study":"CORE-DIVERSITY-v1","status":"SEALED_NEW_SEED",
                       "seed":seed,"step":STEPS,"device":"mps",
                       "freeze_commit":commit,
                       "protocol_digest":manifest["protocol_digest"],
                       "schedule_sha256":trace.hexdigest(),
                       "no_old_dev_reserved_protected_LI":True,
                       "loss_history":history,"outcomes":outputs})
    print("SEALED",report,"Do not analyze until all 16 seeds complete")

def ci(values):
    assert len(values)==16
    m=mean(values)
    r=T95*stdev(values)/math.sqrt(len(values))
    return {"mean":m,"ci95":[m-r,m+r],
            "positive_seeds":sum(x>0 for x in values),
            "per_seed":dict(zip(map(str,SEEDS),values))}

def score(record,arm,pop,key):
    return mean(record["outcomes"][arm][pop][v][key] for v in VOICES[regime(arm)])

def summary():
    ids,hids,tables,vocab,m=verify()
    _=(ids,hids,tables,vocab)
    reports={};missing=[]
    for seed in SEEDS:
        path=OUT/("seed%d_mps_step%d.json"%(seed,STEPS))
        if not path.exists():missing.append(seed);continue
        r=json.loads(path.read_text())
        assert (r["study"],r["seed"],r["step"])==(
            "CORE-DIVERSITY-v1",seed,STEPS)
        assert r["protocol_digest"]==m["protocol_digest"]
        assert r["schedule_sha256"]==m["protocol"]["schedule_sha256"][str(seed)]
        assert r["no_old_dev_reserved_protected_LI"]
        assert set(r["outcomes"])==set(ARMS)
        reports[seed]=r
    if missing:
        print("INCOMPLETE",len(missing),"/16 missing",missing)
        print("No partial confirmatory analysis")
        return
    def contrast(rec,pop):
        return mean(score(rec,reg+"_diverse",pop,"C_query")
                    -score(rec,reg+"_fixed",pop,"C_query")
                    for reg in VOICES)
    primary=ci([contrast(reports[s],"unseen_roles") for s in SEEDS])
    passed=primary["ci95"][0]>0 and primary["mean"]>=.10 and primary["positive_seeds"]>=12
    secondary=ci([contrast(reports[s],"unseen_quartets") for s in SEEDS])
    secondary_pass=(secondary["ci95"][0]>0) if passed else None
    arm_summary={}
    for arm in ARMS:
        arm_summary[arm]={
            pop:{metric:ci([score(reports[s],arm,pop,metric) for s in SEEDS])
                 for metric in ("C_query","accuracy4","full_vocab_accuracy")}
            for pop in POPS}
    result={"study":"CORE-DIVERSITY-v1","status":"COMPLETE_FROZEN_PROSPECTIVE",
            "seeds":16,"models":64,"protocol_digest":m["protocol_digest"],
            "no_old_dev_reserved_protected_LI":True,
            "primary":{"estimate":primary,"passes":passed,
                       "criteria":"95% CI lower>0; effect>=+10pp; >=12/16 seeds positive"},
            "secondary":{"estimate":secondary,"passes":secondary_pass,
                         "confirmatory_gate_open":passed},
            "arms":arm_summary,
            "analysis_unit":"16 fresh independent paired seeds; no seed excluded"}
    write_once(OUT/"analysis_summary.json",result)
    print("SUMMARY PASS: 16 independent seeds, 64 newly trained models")
    print("PRIMARY",primary,"passes",passed)
    print("GATED SECONDARY",("NOT CONFIRMATORY" if not passed else secondary_pass))
    print("No competence filtering or protected data accessed")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("prepare","smoke","run","summary"),required=True)
    p.add_argument("--seed",type=int)
    p.add_argument("--device",choices=("cpu","mps"),default="cpu")
    a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="smoke":smoke(torch.device(a.device))
    elif a.phase=="run":
        if a.seed is None:p.error("run needs --seed")
        run(a.seed,torch.device(a.device))
    else:summary()
if __name__=="__main__":main()
