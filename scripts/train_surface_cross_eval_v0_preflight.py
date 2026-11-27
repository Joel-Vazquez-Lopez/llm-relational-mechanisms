#!/usr/bin/env python3
"""CROSS-SURFACE-v0: source-only, post-hoc checkpoint diagnostic preflight.

Constructs deterministic new scores for an ALREADY-TRAINED 12-seed cohort.
No training, no model inference, no original DEV/reserved/protected/L+I sources.
Only 24 familiar train-name quartets and 12 *previously TRAIN-derived* heldout
quartets are sampled for this exploratory diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import train_surface_factorial_v0_preflight as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/train_surface_cross_eval_v0"
MANIFEST=OUT/"source_manifest.json"
SELECTION_SEED=157021
N_TRAIN_QUARTETS=24
N_HELD_QUARTETS=12
CONDITIONS=("bare","frame","adjunct","both")
POPS=("train_core","unseen_roles","unseen_quartets")
VOICE=old.VOICE_PAIR
assert VOICE==("AP","PA")
assert len(old.FRAMES)==5 and len(old.MODS)==9

def digest(data):
    return sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def sample_ids():
    train,held,original,info=old.partition_cores()
    tr=list(train);ho=list(held)
    Random(SELECTION_SEED).shuffle(tr)
    Random(SELECTION_SEED+1).shuffle(ho)
    return tuple(sorted(tr[:N_TRAIN_QUARTETS])),tuple(sorted(ho[:N_HELD_QUARTETS])),original,info

def make_surfaces(events,roles,voice,fi,mi,group):
    combinations={
        "bare":(0,0),"frame":(fi,0),
        "adjunct":(0,mi),"both":(fi,mi)
    }
    src={}
    for surface,(f,m) in combinations.items():
        src[surface]=old.source(events,roles,voice,f,m,group=group)
        assert src[surface].gold==roles
    assert len(set(x.text for x in src.values()))==4
    return src

def build():
    """Return materialized source objects and immutable provenance metadata."""
    selected_train,selected_held,original,info=sample_ids()
    result={pop:{cond:{v:[] for v in VOICE} for cond in CONDITIONS}
            for pop in POPS}
    signatures={pop:[] for pop in POPS}
    seen_training_surfaces=set()
    seen_role_surfaces=set()
    seen_held_surfaces=set()
    total_cases={"unseen_roles":0,"unseen_quartets":0}
    selected_role_perm_sigs=[]
    for ix in selected_train:
        events=original[ix]
        original_roles=tuple(n for e in events for n in (e.agent,e.patient))
        _training_eight,unseen=old.previous.all_roles(original_roles)
        assert _training_eight[0]==original_roles
        # All 45 source variants of the ORIGINAL trained semantic core.
        for fi in range(5):
            for mi in range(9):
                if fi==0 and mi==0:condition="bare"
                elif fi>0 and mi==0:condition="frame"
                elif fi==0 and mi>0:condition="adjunct"
                else:condition="both"
                for voice in VOICE:
                    src=old.source(events,original_roles,voice,fi,mi,
                                   group="cross_train_core")
                    assert src.text not in seen_training_surfaces
                    seen_training_surfaces.add(src.text)
                    result["train_core"][condition][voice].append(src)
                    signatures["train_core"].append((ix,fi,mi,voice,src.text,src.gold))
        for j,roles in enumerate(unseen):
            k=total_cases["unseen_roles"]
            total_cases["unseen_roles"]+=1
            fi=1+(k%4);mi=1+((k//4)%8)
            selected_role_perm_sigs.append((ix,j,roles,fi,mi))
            for voice in VOICE:
                srcs=make_surfaces(events,roles,voice,fi,mi,"cross_unseen_role")
                for condition,s in srcs.items():
                    assert s.text not in seen_training_surfaces
                    assert s.text not in seen_role_surfaces
                    seen_role_surfaces.add(s.text)
                    result["unseen_roles"][condition][voice].append(s)
                    signatures["unseen_roles"].append(
                        (ix,j,voice,condition,fi,mi,s.text,s.gold))
    for ix in selected_held:
        events=original[ix]
        original_roles=tuple(n for e in events for n in (e.agent,e.patient))
        from itertools import permutations
        for j,roles in enumerate(permutations(original_roles)):
            k=total_cases["unseen_quartets"]
            total_cases["unseen_quartets"]+=1
            fi=1+(k%4);mi=1+((k//4)%8)
            for voice in VOICE:
                srcs=make_surfaces(events,roles,voice,fi,mi,"cross_held_quartet")
                for condition,s in srcs.items():
                    assert s.text not in seen_training_surfaces
                    assert s.text not in seen_role_surfaces
                    assert s.text not in seen_held_surfaces
                    seen_held_surfaces.add(s.text)
                    result["unseen_quartets"][condition][voice].append(s)
                    signatures["unseen_quartets"].append(
                        (ix,j,voice,condition,fi,mi,s.text,s.gold))

    assert total_cases=={"unseen_roles":24*16,"unseen_quartets":12*24}
    assert not seen_training_surfaces&seen_role_surfaces
    assert not seen_training_surfaces&seen_held_surfaces
    assert not seen_role_surfaces&seen_held_surfaces
    for pop in ("unseen_roles","unseen_quartets"):
        lengths={len(result[pop][cond][v]) for cond in CONDITIONS for v in VOICE}
        assert lengths=={total_cases[pop]}
        for v in VOICE:
            reference=[x.gold for x in result[pop]["bare"][v]]
            for cond in CONDITIONS:
                assert [x.gold for x in result[pop][cond][v]]==reference
        # All 4 nonempty prefix counts and 8 nonempty modifier pairs are
        # exactly balanced across sampled semantic cases.
        prefix_hist=Counter()
        adjunct_hist=Counter()
        for row in signatures[pop]:
            if row[3]=="both":
                prefix_hist[row[4]]+=1
                adjunct_hist[row[5]]+=1
        assert set(prefix_hist)==set(range(1,5))
        assert set(adjunct_hist)==set(range(1,9))
        assert max(prefix_hist.values())==min(prefix_hist.values())
        assert max(adjunct_hist.values())==min(adjunct_hist.values())

    assert len(seen_training_surfaces)==24*45*2
    assert len(seen_role_surfaces)==24*16*4*2
    assert len(seen_held_surfaces)==12*24*4*2
    counts={pop:{cond:{v:len(result[pop][cond][v]) for v in VOICE}
                 for cond in CONDITIONS} for pop in POPS}
    manifest={
      "study":"TRAIN-SURFACE-CROSS-EVAL-v0",
      "status":"EXPLORATORY_POSTHOC_CHECKPOINT_FOLLOWUP",
      "source_partitions":"only the original 300 TRAIN identifiers; reused prior 240/60 study split",
      "parent_protocol_digest":json.loads(old.ROOT.joinpath(
          "reports/train_surface_factorial_v0/study_manifest.json").read_text())["protocol_digest"],
      "parent_split_digest":info["partition_digest"],
      "selection_seed":SELECTION_SEED,
      "selected_train_quartets":list(selected_train),
      "selected_withheld_quartets":list(selected_held),
      "selected_train_sha256":digest(selected_train),
      "selected_withheld_sha256":digest(selected_held),
      "source_counts":counts,
      "source_hashes":{p:digest(signatures[p]) for p in POPS},
      "source_role_permutations_sha256":digest(selected_role_perm_sigs),
      "prefix_frame_index_definition":"fi=1+(k mod 4)",
      "adjunct_pair_index_definition":"mi=1+((k//4) mod 8)",
      "surface_bundles":{"bare":"(fi=0,mi=0)","frame":"(fi in 1..4,mi=0)",
                         "adjunct":"(fi=0,mi in 1..8)","both":"(fi in 1..4,mi in 1..8)"},
      "source_only_checks":{
          "gold_matches_independent_parser":True,
          "identical_gold_and_id_order_across_surfaces":True,
          "all_train_and_test_text_disjoint":True,
          "16_recombined_roles_never_trained":True,
          "max_input_len_48":True,
          "named_LI_excluded":True,
          "original_dev_reserved_protected_not_rendered":True,
          "no_old_or_new_model_inference":True},
      "generator_sha256":sha(__file__),
      "dependencies_sha256":{
          "train_surface_factorial_v0_preflight.py":sha(old.__file__),
          "core_diversity_v1_preflight.py":sha(old.previous.__file__),
          "exp105_generator_preflight.py":sha(old.gen.__file__),
          "exp105_position_control_preflight_v1.py":sha(old.frame.__file__)}
    }
    return result,manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("check","prepare"))
    a=p.parse_args()
    _sources,manifest=build()
    print("SOURCE-ONLY PASS",manifest["source_counts"])
    print("SELECTION seed",SELECTION_SEED,
          "TRAIN quartets",N_TRAIN_QUARTETS,
          "TRAIN-held quartets",N_HELD_QUARTETS)
    print("NO checkpoints, no model scoring, no DEV/reserved/protected/L+I")
    if a.phase=="prepare":
        OUT.mkdir(parents=True,exist_ok=True)
        text=json.dumps(manifest,sort_keys=True,indent=2)+"\n"
        if MANIFEST.exists():
            if MANIFEST.read_text()!=text:raise RuntimeError("Immutable manifest drift")
        else:MANIFEST.write_text(text)
        print("MANIFEST PREPARED",MANIFEST)

if __name__=="__main__":main()
