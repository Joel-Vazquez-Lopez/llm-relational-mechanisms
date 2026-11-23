#!/usr/bin/env python3
"""ROLE-DIVERSITY-COMPLEX-v0: source-only feasibility and invariance checks.

DESIGN ONLY. No model training, no checkpoints, no protected/old DEV data.
Test nested 1/2/4/8 semantic-role assignments per original TRAIN quartet
under the SAME complex prefixes+named-adjunct source grammar.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random

import train_surface_factorial_v0_preflight as prev

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/role_diversity_complex_v0"
MANIFEST=OUT/"design_source_audit.json"
DIVERSITIES=(1,2,4,8)
ARMS=tuple("roles_%d"%x for x in DIVERSITIES)
SELECTION_SEED=160021
FULL_CELL_AUDIT_QUARTETS=24
STEPS_PROPOSED=6400
QUARTETS_PER_STEP_PROPOSED=8
MAX_TOKENS=48
VOICES=prev.VOICE_PAIR
assert VOICES==("AP","PA")

def digest(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def role_orders(events):
    gold=tuple(n for e in events for n in (e.agent,e.patient))
    eight,sixteen=prev.previous.all_roles(gold)
    assert eight[0]==gold
    assert len(eight)==8 and len(sixteen)==16
    assert not set(eight)&set(sixteen)
    subsets={n:eight[:n] for n in DIVERSITIES}
    assert subsets[1][0]==gold
    assert subsets[2]==(eight[0],eight[1])
    assert subsets[4]==eight[:4] and subsets[8]==eight
    for n,orders in subsets.items():
        assert len(orders)==len(set(orders))==n
        # Every person occupies n different source slots when n in {1,2,4}.
        for name in gold:
            occupancies=Counter(pos for roles in orders for pos in range(4)
                                if roles[pos]==name)
            if n==2:assert len(occupancies)==2
            if n==4:assert set(occupancies)=={0,1,2,3}
            if n==8:assert set(occupancies.values())=={2}
    return gold,eight,sixteen,subsets

def nonname_bag(surface):
    return Counter(tok for tok in prev.gen.tokens(surface) if tok not in prev.gen.NAME_SET)

def audit():
    train_ids,held_ids,original,info=prev.partition_cores()
    assert len(train_ids)==240 and len(held_ids)==60
    ids=list(train_ids);Random(SELECTION_SEED).shuffle(ids)
    sample=tuple(sorted(ids[:FULL_CELL_AUDIT_QUARTETS]))
    signatures={"train_all_base_slots":[],"train_full_cells_sample":[],
                "eval_unseen_bare":[],"eval_unseen_rich":[],
                "eval_heldout_bare":[],"eval_heldout_rich":[]}
    expected_train_slot_counts={n:240*n for n in DIVERSITIES}
    unknown_role_overlaps=0
    for ix in train_ids:
        events=original[ix]
        gold,eight,sixteen,subsets=role_orders(events)
        assert gold==eight[0]
        for n in DIVERSITIES:
            assert len(set(subsets[n])&set(sixteen))==0
        for ordinal,roles in enumerate(eight):
            for voice in VOICES:
                src=prev.source(events,roles,voice,0,0,group="design_train")
                signatures["train_all_base_slots"].append(
                    (ix,ordinal,voice,src.text,src.gold))
        if ix in sample:
            for fi in range(5):
                for mi in range(9):
                    for voice in VOICES:
                        base=None
                        for ordinal,roles in enumerate(eight):
                            src=prev.source(events,roles,voice,fi,mi,
                                            group="design_full_cell")
                            bag=nonname_bag(src.text)
                            if base is None:base=(len(prev.gen.tokens(src.text)),bag)
                            else:
                                assert len(prev.gen.tokens(src.text))==base[0]
                                assert bag==base[1],(ix,fi,mi,voice,ordinal)
                            signatures["train_full_cells_sample"].append(
                                (ix,ordinal,fi,mi,voice,src.text,src.gold))
        for ordinal,roles in enumerate(sixteen):
            k=ix*16+ordinal
            fi=1+(k%4);mi=1+((k//4)%8)
            for voice in VOICES:
                b=prev.source(events,roles,voice,0,0,group="eval_roles")
                d=prev.source(events,roles,voice,fi,mi,group="eval_roles")
                assert b.gold==d.gold==roles
                signatures["eval_unseen_bare"].append((ix,ordinal,voice,b.text,b.gold))
                signatures["eval_unseen_rich"].append((ix,ordinal,fi,mi,voice,d.text,d.gold))
    for ix in held_ids:
        events=original[ix]
        gold=tuple(n for e in events for n in (e.agent,e.patient))
        for ordinal,roles in enumerate(permutations(gold)):
            k=ix*24+ordinal
            fi=1+(k%4);mi=1+((k//4)%8)
            for voice in VOICES:
                b=prev.source(events,roles,voice,0,0,group="eval_new_quartets")
                d=prev.source(events,roles,voice,fi,mi,group="eval_new_quartets")
                signatures["eval_heldout_bare"].append((ix,ordinal,voice,b.text,b.gold))
                signatures["eval_heldout_rich"].append((ix,ordinal,fi,mi,voice,d.text,d.gold))

    counts={k:len(v) for k,v in signatures.items()}
    assert counts=={
      "train_all_base_slots":240*8*2,
      "train_full_cells_sample":FULL_CELL_AUDIT_QUARTETS*8*5*9*2,
      "eval_unseen_bare":240*16*2,
      "eval_unseen_rich":240*16*2,
      "eval_heldout_bare":60*24*2,
      "eval_heldout_rich":60*24*2,
    }
    for key in signatures:
        assert len({x[-2] for x in signatures[key]})==counts[key],key
    orig_train_text={x[-2] for x in signatures["train_all_base_slots"]}
    assert not orig_train_text&{x[-2] for x in signatures["eval_unseen_bare"]}
    assert not orig_train_text&{x[-2] for x in signatures["eval_heldout_bare"]}
    assert len({x[-2] for x in signatures["eval_unseen_rich"]})==counts["eval_unseen_rich"]
    assert len({x[-2] for x in signatures["eval_heldout_rich"]})==counts["eval_heldout_rich"]
    assert not {x[-2] for x in signatures["eval_unseen_rich"]}&{x[-2] for x in signatures["eval_heldout_rich"]}
    parent_manifest=json.loads(prev.ROOT.joinpath(
          "reports/train_surface_factorial_v0/study_manifest.json").read_text())
    return {
        "study":"ROLE-DIVERSITY-COMPLEX-v0",
        "status":"SOURCE_ONLY_DESIGN_NOT_TRAINING_FROZEN",
        "old_train_partition_digest":info["partition_digest"],
        "reuse_old_factorial_train_identifiers":True,
        "parent_factorial_digest":parent_manifest["protocol_digest"],
        "train_quartet_count":240,
        "heldout_quartet_count":60,
        "role_diversities":list(DIVERSITIES),
        "nested_training_indices":{
           "roles_1":[0],"roles_2":[0,1],
           "roles_4":[0,1,2,3],"roles_8":list(range(8))},
        "never_train_role_permutations_per_quartet":16,
        "heldout_quartet_role_permutations":24,
        "all_arms_use_both_familiar_voices":list(VOICES),
        "surface_cells":45,
        "train_surface":"all arms use the same 5 neutral frames x 9 bare/L/I modifier pairs",
        "train_source_slot_counts":expected_train_slot_counts,
        "potential_train_source_count_by_arm":{
           "roles_%d"%n:240*n*45*len(VOICES) for n in DIVERSITIES},
        "model_train_examples_per_update_if_run":2*QUARTETS_PER_STEP_PROPOSED,
        "model_answer_labels_per_update_if_run":2*4*QUARTETS_PER_STEP_PROPOSED,
        "planned_train_steps":STEPS_PROPOSED,
        "new_seed_list_and_training_schedule":"UNFROZEN_NOT_YET_IMPLEMENTED",
        "role_assignment_schedule_and_cell_orthogonality":"TO_AUDIT_BEFORE_FREEZE",
        "source_only_full_cell_audited_train_quartets":list(sample),
        "source_only_all_train_bare_role_permutations_checked":True,
        "source_counts":counts,
        "source_hashes":{k:digest(v) for k,v in signatures.items()},
        "different_roles_same_cell_have_identical_length_and_nonname_token_bags":True,
        "original_train_dev_reserved_protected_LI_not_rendered":True,
        "no_training_or_checkpoint_inference":True,
        "confounds_to_control_and_report":[
            "diversity arms differ in number of distinct training surfaces",
            "participant-name position, role and distractor marginals differ by design",
            "all arms must have identical quartet, prefix, modifier, voice and step schedules",
            "source cell and role-selection interactions must be audited",
            "one extra role repeatedly sampled is NOT one-shot exposure"],
        "source_generator_sha256":sha(__file__),
        "dependency_sha256":{
            "train_surface_factorial_v0_preflight.py":sha(prev.__file__),
            "core_diversity_v1_preflight.py":sha(prev.previous.__file__),
            "exp105_generator_preflight.py":sha(prev.gen.__file__),
            "exp105_position_control_preflight_v1.py":sha(prev.frame.__file__)}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",required=True,choices=("check","prepare"))
    args=ap.parse_args()
    r=audit()
    print("ROLE-DIVERSITY DESIGN SOURCE AUDIT PASS:",r["source_counts"])
    print("nested training role counts: 1,2,4,8; untouched eval permutations 16")
    print("NO training or model inference, original DEV/reserved/protected/L+I excluded")
    if args.phase=="prepare":
        OUT.mkdir(parents=True,exist_ok=True)
        data=json.dumps(r,indent=2,sort_keys=True)+"\n"
        if MANIFEST.exists():
            if MANIFEST.read_text()!=data:raise RuntimeError("Design audit drift")
        else:MANIFEST.write_text(data)
        print("DESIGN AUDIT WRITTEN",MANIFEST)

if __name__=="__main__":main()
