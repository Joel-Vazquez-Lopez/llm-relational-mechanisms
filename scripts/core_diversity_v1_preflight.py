#!/usr/bin/env python3
"""CORE-DIVERSITY-v1: dependency-light, TRAIN-partition-only source preflight.

NO model loading, training, inference, development evaluation, or protected data.
Only original 300 TRAIN quartet identifiers are split (240 train / 60 new holdout).
Generate exhaustive 8/16 role-permutation split per train quartet, all 24 on
wholly withheld TRAIN quartets; parse/verify all four voice renderings and
all exact token bag controls. PREPARE writes manifest; CHECK only prints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from random import Random

import exp105_generator_preflight as gen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/core_diversity_v1"
MANIFEST=OUT/"source_manifest.json"
SOURCE_SEED=139310
N_TRAIN=240
N_WITHHELD=60
VOICES=("AA","PP","AP","PA")
TRAIN_PAIRS={"same_voice":("AA","PP"),"mixed_voice":("AP","PA")}
BARE=("bare","bare")
TRAINING_CYCLIC_SHIFTS=(0,1,2,3)

def digest(data):
    return sha256(json.dumps(
        data,sort_keys=True,separators=(",",":"),ensure_ascii=True
    ).encode("utf8")).hexdigest()

def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(data,sort_keys=True,indent=2)+"\n"
    if path.exists():
        if path.read_text()!=serialized:
            raise RuntimeError("Existing source manifest differs; do NOT overwrite")
        print("Existing manifest identical")
        return
    with path.open("x",encoding="utf8") as f:
        f.write(serialized)

def all_roles(gold):
    assert len(gold)==4 and len(set(gold))==4
    all24=list(permutations(gold))
    originals=[]
    for row in (gold,tuple(reversed(gold))):
        for k in TRAINING_CYCLIC_SHIFTS:
            originals.append(tuple(row[(i+k)%4] for i in range(4)))
    assert len(set(originals))==8
    held=[x for x in all24 if x not in originals]
    assert len(held)==16
    for name in gold:
        assert [sum(p[pos]==name for p in originals) for pos in range(4)]==[2,2,2,2]
    return tuple(originals),tuple(held)

def new_events(old,roles):
    a,b,c,d=roles
    x,y=old
    return (gen.Event(a,b,x.verb,"A",c,d,x.layout),
            gen.Event(c,d,y.verb,"A",a,b,y.layout))

def build():
    train_sets, dev_sets, info=gen.partitions()
    assert len(train_sets)==300 and len(dev_sets)==100
    cores={}
    for j,cid,events in gen.cores(train_sets,"train"):
        if j==2:
            index=int(cid.split("-")[1])
            assert index not in cores
            cores[index]=(cid,events)
    assert len(cores)==300
    ids=list(sorted(cores))
    Random(SOURCE_SEED).shuffle(ids)
    train_ids=ids[:N_TRAIN]
    heldout_ids=ids[N_TRAIN:]
    assert len(heldout_ids)==N_WITHHELD and not set(train_ids)&set(heldout_ids)
    train_surfaces=set()
    eval_surfaces=set()
    records={"train_permutations":[],"evaluation_train_quartets":[],
             "evaluation_withheld_quartets":[]}
    counters=Counter()
    role_balanced=True

    def emit(index,condition,orders):
        _cid,old=cores[index]
        for ordinal,roles in enumerate(orders):
            sources={}
            for voice in VOICES:
                s=gen.make_source(new_events(old,roles),voice,BARE,
                                  condition,"core-diversity-%s-%d-%d"%(
                                      condition,index,ordinal))
                assert s.gold==roles and gen.parse_independent(s.text)==roles
                assert len(set(s.gold))==4 and not any(m=="LI" for m in s.mods)
                tokenbag=Counter(gen.tokens(s.text))
                assert sum(tokenbag[name] for name in roles)==4
                assert tokenbag[roles[0]]==tokenbag[roles[1]]==1
                assert tokenbag[roles[2]]==tokenbag[roles[3]]==1
                sources[voice]=s
                counters[condition+"_source_voices"]+=1
            # The four voice constructions differ only in voice. The exact
            # aggregate token bag of AA+PP equals AP+PA for each assignment.
            aa,pp,ap,pa=(sources[v] for v in ("AA","PP","AP","PA"))
            assert aa.gold==pp.gold==ap.gold==pa.gold
            assert Counter(gen.tokens(aa.text)+gen.tokens(pp.text))==Counter(
                gen.tokens(ap.text)+gen.tokens(pa.text))
            # Each name occurs exactly once per source for bare/bare and
            # each voice has a fixed length for a given source quartet.
            for voice,s in sources.items():
                expected=gen.make_source(new_events(old,orders[0]),voice,BARE,
                                         condition,"reference")
                assert len(gen.tokens(s.text))==len(gen.tokens(expected.text))
                assert Counter(gen.tokens(s.text))==Counter(
                    gen.tokens(expected.text))
                key=(index,ordinal,voice,s.text,tuple(s.gold))
                if condition=="train_permutations":
                    assert s.text not in train_surfaces
                    train_surfaces.add(s.text)
                else:
                    assert s.text not in train_surfaces and s.text not in eval_surfaces
                    eval_surfaces.add(s.text)
                records[condition].append(key)

    for index in train_ids:
        gold=tuple(name for e in cores[index][1] for name in (e.agent,e.patient))
        train8,eval16=all_roles(gold)
        emit(index,"train_permutations",train8)
        emit(index,"evaluation_train_quartets",eval16)
    for index in heldout_ids:
        gold=tuple(name for e in cores[index][1] for name in (e.agent,e.patient))
        emit(index,"evaluation_withheld_quartets",tuple(permutations(gold)))

    assert len(train_surfaces)==N_TRAIN*8*4
    assert len(eval_surfaces)==N_TRAIN*16*4+N_WITHHELD*24*4
    assert not train_surfaces & eval_surfaces
    assert len(records["train_permutations"])==N_TRAIN*8*4
    assert len(records["evaluation_train_quartets"])==N_TRAIN*16*4
    assert len(records["evaluation_withheld_quartets"])==N_WITHHELD*24*4
    anchor={
        "study":"CORE-DIVERSITY-v1-source-preflight",
        "status":"SOURCE_ONLY_DESIGN_NOT_TRAINING_FROZEN",
        "partition":"300 old TRAIN identifiers only, repartitioned",
        "no_old_dev_or_reserved_source_rendering":True,
        "no_model_training_or_checkpoint_inference":True,
        "old_partition_digest":info["partition_digest"],
        "selection_seed":SOURCE_SEED,
        "selected_train_quartet_count":len(train_ids),
        "selected_withheld_quartet_count":len(heldout_ids),
        "train_identifiers_sha256":digest(train_ids),
        "withheld_identifiers_sha256":digest(heldout_ids),
        "training_eligible_permutations_per_quartet":8,
        "training_excluded_permutations_per_quartet":16,
        "heldout_quartet_evaluation_permutations":24,
        "voices":list(VOICES),
        "voice_pairs":TRAIN_PAIRS,
        "same_voice_and_mixed_voice_token_bags_match":True,
        "each_participant_each_of_four_role_slots_twice_in_diverse_arm":role_balanced,
        "bare_bare_exact_token_multiset_and_length_matched":True,
        "query_gold_labels_per_source":"four distinct names, one each",
        "train_eval_text_disjoint":True,
        "source_counts":dict(counters),
        "source_hashes":{name:digest(data) for name,data in records.items()},
        "source_generator_sha256":sha256(Path(gen.__file__).read_bytes()).hexdigest(),
        "preflight_script_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes":"Source-only preflight for a draft. Freeze full training code, optimizer schedule, seed grid and analysis before any model run."
    }
    return anchor

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("check","prepare"),required=True)
    a=p.parse_args()
    m=build()
    print("SOURCE PREFLIGHT PASS",m["source_hashes"])
    print("Train quartets: 240 | withheld quartets: 60")
    print("Train eligible: 8 assignments/quartet | never-train: 16 assignments/quartet")
    print("NO model inference, NO old dev/reserved/protected/L+I examples")
    if a.phase=="prepare":
        write_once(MANIFEST,m)
        print("PREPARED",MANIFEST)
    else:
        print("CHECK ONLY: no manifest written")

if __name__=="__main__":
    main()
