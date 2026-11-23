#!/usr/bin/env python3
"""TRAIN-SURFACE-FACTORIAL-v0 source-only audit. NO models or inference.

The study reuses the CORE-DIVERSITY-v1 240/60 split OF OLD TRAIN quartet IDs,
but studies four TRAINING surface distributions with a fixed event-role core.
It never renders original development, reserved, protected, or L+I sources.
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
import exp105_position_control_preflight_v1 as frame
import core_diversity_v1_preflight as previous

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/train_surface_factorial_v0"
SOURCE=OUT/"source_manifest.json"
VOICE_PAIR=("AP","PA")
FRAMES=frame.FRAMES
MODS=gen.TRAIN_MODS
ARMS=("bare","frame_only","adjunct_only","frame_adjunct")
assert len(FRAMES)==5 and len(MODS)==9 and MODS[0]==("bare","bare")
MAX_TOKENS=48

def digest(x):
    return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode("utf8")).hexdigest()
def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def partition_cores():
    """Return exact train/heldout 240/60 CORE-DIVERSITY split; no dev render."""
    train,_dev,info=gen.partitions()
    assert len(train)==300
    old={}
    for j,cid,events in gen.cores(train,"train"):
        if j==2:
            ix=int(cid.split("-")[1])
            assert ix not in old
            old[ix]=events
    assert len(old)==300
    ids=sorted(old)
    Random(previous.SOURCE_SEED).shuffle(ids)
    tr=tuple(ids[:240]);ho=tuple(ids[240:])
    old_manifest=json.loads(previous.MANIFEST.read_text())
    assert digest(tr)==old_manifest["train_identifiers_sha256"]
    assert digest(ho)==old_manifest["withheld_identifiers_sha256"]
    assert info["partition_digest"]==old_manifest["old_partition_digest"]
    return tr,ho,old,info

def source(old,roles,voice,fi,mi,group="surface"):
    assert voice in VOICE_PAIR and 0<=fi<len(FRAMES) and 0<=mi<len(MODS)
    base=gen.make_source(previous.new_events(old,roles),voice,MODS[mi],group,"surface-v0")
    framed=frame.expand(base,FRAMES[fi])
    assert framed.gold==roles
    assert gen.parse_independent(frame.unframe(framed.text))==roles
    assert tuple(dict.fromkeys(roles))==roles
    assert not any(mod=="LI" for mod in framed.mods)
    for q in gen.Q:
        tokens=["<bos>"]+gen.tokens(framed.text)+["<sep>"]+gen.tokens(q)+["<answer>"]
        if len(tokens)>MAX_TOKENS:
            raise RuntimeError("Input length exceeds the ORIGINAL 48-token limit: "
                               +str((voice,fi,mi,len(tokens))))
    return framed

def input_profile(s,by_arm,arm):
    bag=gen.tokens(s.text)
    by_arm[arm]["n_sources"]+=1
    by_arm[arm]["token_length_sum"]+=len(bag)
    by_arm[arm]["min_tokens"]=min(by_arm[arm]["min_tokens"],len(bag))
    by_arm[arm]["max_tokens"]=max(by_arm[arm]["max_tokens"],len(bag))
    by_arm[arm]["name_mentions_sum"]+=sum(t in gen.NAME_SET for t in bag)
    for i,tok in enumerate(bag):
        if tok in gen.NAME_SET:by_arm[arm]["name_position_histogram"][str(i+1)]+=1

def build():
    tr,ho,old,info=partition_cores()
    assert len(FRAMES)==5 and tuple(len(gen.tokens(p)) if p else 0
                                   for p in FRAMES)==frame.EXPECTED_SHIFTS
    for prefix in FRAMES:
        if prefix:
            assert not (set(gen.tokens(prefix))&(gen.NAME_SET|set(gen.VERBS)))
    records={"train_variants":[],"test_unseen_roles":[],"test_unseen_quartets":[]}
    training_text=set();testing_text=set()
    per_arm={arm:{"n_sources":0,"token_length_sum":0,"name_mentions_sum":0,
                  "min_tokens":100000,"max_tokens":0,
                  "name_position_histogram":Counter()} for arm in ARMS}
    labels_train=Counter()
    role_orders={}
    for ix in tr:
        events=old[ix]
        gold=tuple(x for e in events for x in (e.agent,e.patient))
        orders,evaluation=previous.all_roles(gold)
        assert orders[0]==gold and len(evaluation)==16
        role_orders[ix]=evaluation
        for fi in range(len(FRAMES)):
            for mi in range(len(MODS)):
                by_voice={}
                for voice in VOICE_PAIR:
                    s=source(events,gold,voice,fi,mi)
                    assert s.text not in training_text
                    training_text.add(s.text)
                    by_voice[voice]=s
                    records["train_variants"].append(
                        (ix,fi,mi,voice,s.text,s.gold))
                    for answer in s.gold:labels_train[answer]+=1
                # Equal-length within arm is not asserted; frame/adjunct presence
                # intentionally changes length, positions and name-mention counts.
                cases={"bare":by_voice if fi==0 and mi==0 else None,
                       "frame_only":by_voice if mi==0 else None,
                       "adjunct_only":by_voice if fi==0 else None,
                       "frame_adjunct":by_voice}
                for arm,voice_samples in cases.items():
                    if voice_samples is not None:
                        for s in voice_samples.values():
                            input_profile(s,per_arm,arm)
    for cat,ids in (("test_unseen_roles",tr),("test_unseen_quartets",ho)):
        for ix in ids:
            events=old[ix]
            gold=tuple(x for e in events for x in (e.agent,e.patient))
            orders=role_orders[ix] if cat=="test_unseen_roles" else tuple(permutations(gold))
            for idx,roles in enumerate(orders):
                for voice in VOICE_PAIR:
                    s=source(events,roles,voice,0,0,group=cat)
                    assert s.text not in testing_text
                    assert s.text not in training_text
                    testing_text.add(s.text)
                    records[cat].append((ix,idx,voice,s.text,s.gold))
    assert len(records["train_variants"])==240*5*9*2
    assert len(records["test_unseen_roles"])==240*16*2
    assert len(records["test_unseen_quartets"])==60*24*2
    assert len(testing_text)==len(records["test_unseen_roles"])+len(records["test_unseen_quartets"])
    assert not training_text&testing_text
    assert sum(labels_train.values())==len(records["train_variants"])*4
    summaries={}
    for arm,datum in per_arm.items():
        n=datum["n_sources"]
        assert n>0
        summaries[arm]={"possible_source_count":n,
                        "mean_source_tokens":datum["token_length_sum"]/n,
                        "min_source_tokens":datum["min_tokens"],
                        "max_source_tokens":datum["max_tokens"],
                        "mean_name_mentions":datum["name_mentions_sum"]/n,
                        "name_position_histogram":dict(datum["name_position_histogram"])}
    assert summaries["bare"]["mean_name_mentions"]==4
    assert summaries["frame_only"]["mean_name_mentions"]==4
    assert summaries["adjunct_only"]["mean_name_mentions"]>4
    assert summaries["frame_adjunct"]["mean_name_mentions"]>4
    training_words={word for x in records["train_variants"] for word in gen.tokens(x[4])}
    for q in gen.Q:training_words.update(gen.tokens(q))
    for x in records["test_unseen_roles"]+records["test_unseen_quartets"]:
        assert set(gen.tokens(x[3]))<=training_words
    return {
      "study":"TRAIN-SURFACE-FACTORIAL-v0-source-only",
      "status":"DESIGN_PREFLIGHT_NO_MODEL_INFERENCE",
      "reuses_core_diversity_train_quartet_identifiers":True,
      "old_train_partition_digest":info["partition_digest"],
      "train_ids_sha256":digest(tr),"withheld_ids_sha256":digest(ho),
      "train_quartets":len(tr),"withheld_quartets":len(ho),
      "voices":list(VOICE_PAIR),
      "frames":list(FRAMES),"mods":[list(m) for m in MODS],
      "arms":list(ARMS),
      "source_counts":{k:len(v) for k,v in records.items()},
      "source_hashes":{k:digest(v) for k,v in records.items()},
      "arm_distribution_audit":summaries,
      "train_and_test_text_disjoint":True,
      "role_assignments_train_fixed":1,
      "role_assignments_test_train_quartets":16,
      "role_assignments_test_withheld_quartets":24,
      "no_original_dev_reserved_protected_LI_source_rendering":True,
      "max_model_tokens":MAX_TOKENS,
      "training_source_vocabulary_complete_for_bare_tests":True,
      "script_sha256":file_sha(__file__),
      "dependency_sha256":{
          "exp105_generator_preflight.py":file_sha(gen.__file__),
          "exp105_position_control_preflight_v1.py":file_sha(frame.__file__),
          "core_diversity_v1_preflight.py":file_sha(previous.__file__)},
      "explicitly_unmatched_factors":["source token length","participant name mention count",
          "absolute participant-name positions","prefix tokens",
          "adjunct name-bearing words","syntactic modifier complexity"],
      "no_training_or_checkpoint_inference":True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("check","prepare"))
    a=p.parse_args()
    report=build()
    print("SOURCE PREFLIGHT PASS:",report["source_counts"])
    for arm,datum in report["arm_distribution_audit"].items():
        print("DISTRIBUTION",arm,
              "mean source tokens %.2f"%datum["mean_source_tokens"],
              "mean name mentions %.2f"%datum["mean_name_mentions"])
    if a.phase=="prepare":
        OUT.mkdir(parents=True,exist_ok=True)
        serial=json.dumps(report,sort_keys=True,indent=2)+"\n"
        if SOURCE.exists():
            if SOURCE.read_text()!=serial:raise RuntimeError("Immutable source manifest differs")
        else:SOURCE.write_text(serial)
        print("SOURCE MANIFEST PREPARED",SOURCE)
    else:print("CHECK ONLY: no model or manifest created")

if __name__=="__main__":main()
