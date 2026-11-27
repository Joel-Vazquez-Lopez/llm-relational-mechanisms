#!/usr/bin/env python3
"""Read-only positional-shortcut baselines for the frozen EXP-109 development set.

Dependency-free: standard-library Python plus the repository's original pure
EXP105 generator and frozen EXP109 evaluation *function*. No torch, models,
training, reserved/protected cases, or within-event L+I data are touched.

Reconstruct the exact EXP109 dev set through extracting only the three frozen
generator/evaluation functions from the original scripts. The manifest SHA256
must match before any evaluation. Report aggregate results only.

Run from checkout root:
    python papers/learning-relations/construction-shift/baselines/shortcut_baselines.py
    python papers/learning-relations/construction-shift/baselines/shortcut_baselines.py --emit-json
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from random import Random
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = ROOT / "scripts"
REPORT = ROOT / "reports/exp109/replication_v0/manifest.json"
MODEL_AUDIT = ROOT / "reports/exp110/read_only_error_audit_v0/analysis.json"
CONDITIONS = ("AP", "PA_literal", "AA", "PP", "AP_rename", "AP_reverse_second")
ARMS = ("pa_only", "indirect_pp_aa", "direct_pa_ap")
MODELS = ("raw_mention_index", "main_participant_slot")
EVAL_SEED = 1092026
EVAL_PER_MOD = 16
sys.path.insert(0, str(SCRIPT_DIR))
import exp105_generator_preflight as gen  # noqa:E402; stdlib-only
import exp105_position_control_preflight_v1 as frames  # noqa:E402; stdlib-only

def digest(x):
    return sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def load_original_function(filename, name, env):
    """Execute just one pinned function definition, not the model-training module."""
    path = SCRIPT_DIR / filename
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(nodes) != 1:
        raise AssertionError(f"Cannot isolate {name} from {filename}")
    namespace = dict(env)
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   str(path), "exec")
    exec(code, namespace)
    return namespace[name]

def frozen_evaluation():
    manifest = json.loads(REPORT.read_text(encoding="utf8"))
    expected = manifest["protocol"]["dependency_hashes"]
    for filename in (
        "exp105_generator_preflight.py", "exp105_position_control_preflight_v1.py",
        "exp105_development_pilot_v0.py", "exp106_voice_recombination_pilot_v0.py",
        "exp109_replication_v0.py"):
        actual = sha(SCRIPT_DIR / filename)
        assert expected[filename] == actual, f"EXP109 source changed: {filename}"
    familiar = load_original_function(
        "exp105_development_pilot_v0.py", "familiar_dev_sources",
        dict(v0=gen, defaultdict=defaultdict, Random=Random,
             DEV_SAMPLE_SEED=105450, DEV_PER_CELL=8))
    pilot = SimpleNamespace(familiar_dev_sources=familiar, DEV_PER_CELL=8, MAX_TOKENS=48)
    dev_fn = load_original_function(
        "exp106_voice_recombination_pilot_v0.py", "dev_sources",
        dict(pilot=pilot, gen=gen, defaultdict=defaultdict, Random=Random))
    dev = dev_fn()
    assert len(dev) == 288
    exp109_eval = load_original_function(
        "exp109_replication_v0.py", "evaluation",
        dict(gen=gen, frames=frames, pilot=pilot,
             Random=Random, permutations=permutations, replace=replace, Counter=Counter,
             EVAL_SEED=EVAL_SEED, EVAL_PER_MOD=EVAL_PER_MOD,
             CONDITIONS=CONDITIONS, digest=digest))
    groups, audit = exp109_eval(dev)
    assert audit["evaluation_sha256"] == manifest["evaluation_audit"]["evaluation_sha256"]
    assert audit["anchors"] == 144 and audit["presentations"] == 864
    assert audit["no_reserved_or_protected"] and audit["no_LI"]
    return groups, audit["evaluation_sha256"], manifest

def clauses(src):
    unframed = frames.unframe(src.text)
    assert unframed.endswith(".") and unframed.count(" ; ") == 1
    return unframed[:-1].split(" ; ")

def all_mentions(clause):
    """Raw name mentions. A clause can also mention distractors in modifiers."""
    return tuple(w for w in gen.tokens(clause) if w in gen.NAME_SET)

def core_mentions(clause):
    """Generous clause-slot oracle: strip known adjuncts, never read voice.

    This extractor knows which name mentions are *core participants* and the
    event separator; it is thus stronger than an autonomous slot-only learner.
    It must not consult gold labels or 'was', 'by', voice or question answers.
    """
    s = re.sub(r"\b(?:By|beside) [A-Z][a-z]+'s gate\b", " ", clause)
    s = re.sub(r"\bnear [A-Z][a-z]+\b", " ", s)
    names = all_mentions(s)
    assert len(names) == 2, f"Core-slot extraction returned {len(names)} names"
    assert len(set(names)) == 2
    return names

def source_mentions(source, mode):
    return tuple((core_mentions(c) if mode=="main_participant_slot" else all_mentions(c))
                 for c in clauses(source))

def expected_exposure_pools():
    """Recover the unframed source-family distribution (7 AA + 7 PA + 2 dose).

    Every frame and modifier cell is sampled evenly by the original training
    schedule; these *pool expectation weights* are not a replay of any seed's
    realized stochastic training sequence.
    """
    train, _dev, _part = gen.partitions()
    pools = {"AA_common":[], "PA_common":[], "AA_dose":[],
             "PA_dose":[], "AP_dose":[], "PP_dose":[]}
    for j,cid,events in gen.cores(train,"train"):
        for mods in gen.TRAIN_MODS:
            if j in (0,1):
                for voice in ("AA","PA"):
                    pools[voice+"_common"].append(gen.make_source(events,voice,mods,"base",cid))
            else:
                for voice in ("AA","PA","AP","PP"):
                    pools[voice+"_dose"].append(gen.make_source(events,voice,mods,"dose",cid))
    assert len(pools["AA_common"]) == len(pools["PA_common"]) == 5400
    assert all(len(pools[v+"_dose"])==2700 for v in ("AA","PA","AP","PP"))
    arms = {
        "pa_only":[("AA_common",7),("PA_common",7),("PA_dose",2)],
        "indirect_pp_aa":[("AA_common",7),("PA_common",7),("PP_dose",1),("AA_dose",1)],
        "direct_pa_ap":[("AA_common",7),("PA_common",7),("PA_dose",1),("AP_dose",1)]
    }
    return pools, arms

def fit(pools, specification, mode):
    # Key is (event index, role index, number of surface name mentions).
    # Response is a mention index. Core oracle baseline has two candidates;
    # raw baseline can be led astray by modifier distractors.
    prior = defaultdict(Counter)
    fallback = defaultdict(Counter)
    for name,frequency_per_update in specification:
        rows = pools[name]
        weight = frequency_per_update / len(rows)
        for src in rows:
            ns = source_mentions(src,mode)
            for event_idx, pair in enumerate(ns):
                for role_idx in range(2):
                    target = src.gold[2*event_idx+role_idx]
                    matches = [i for i,v in enumerate(pair) if v==target]
                    assert matches and len(matches)==1, (mode,src.core_id)
                    key = (event_idx,role_idx,len(pair))
                    prior[key][matches[0]] += weight
                    fallback[(event_idx,role_idx)][matches[0]] += weight
    return prior, fallback

def predict(src, mode, fitted):
    prior,fallback=fitted
    all_event_names = source_mentions(src,mode)
    ans=[]
    for event_idx,names in enumerate(all_event_names):
        for role_idx in range(2):
            table = prior.get((event_idx,role_idx,len(names))) or fallback[(event_idx,role_idx)]
            # Deterministic, disclosed tie-breaking: lowest mention index.
            idx = min(table, key=lambda z:(-table[z],z))
            assert 0<=idx<len(names)
            ans.append(names[idx])
    return tuple(ans)

def summarize(groups,mode,fitted):
    results={}
    for cond in CONDITIONS:
        questions=0;correct=0;wr=0;we=0;joint=0
        pairs=Counter()
        for group in groups:
            src=group[cond]; pred=predict(src,mode,fitted);gold=src.gold
            assert len(pred)==len(gold)==4
            joint+=int(pred==gold)
            for i,(a,g) in enumerate(zip(pred,gold)):
                questions+=1
                if a==g:correct+=1
                elif a==gold[(i//2)*2+1-(i%2)]:wr+=1
                else:we+=1
            second_p=pred[2:4];second_g=gold[2:4]
            if any(x not in second_g for x in second_p):
                pairs["wrong_event_name"]+=1
            elif second_p==second_g:
                pairs["correct"]+=1
            elif second_p==second_g[::-1]:
                pairs["reversed"]+=1
            elif second_p[0]==second_p[1]:
                pairs["collapsed"]+=1
            else:
                raise AssertionError("Nonexhaustive paired taxonomy")
        assert correct+wr+we == questions
        assert sum(pairs.values()) == len(groups)
        results[cond] = {
            "n":len(groups),"C_query":joint/len(groups),
            "accuracy4":correct/questions,
            "wrong_role_rate":wr/questions,
            "wrong_event_rate":we/questions,
            "second_event_pair_classes":{k:pairs[k]/len(groups)
                for k in ("correct","reversed","collapsed","wrong_event_name")}}
    results["contrast"]={
        "PA_literal_minus_AP_C_query":results["PA_literal"]["C_query"]-results["AP"]["C_query"],
        "AP_second_event_complete_reversal":results["AP"]["second_event_pair_classes"]["reversed"]}
    return results

def self_test():
    names = gen.NAMES[:4]
    a,b,c,d=names
    for voice in ("AA","PA","AP","PP"):
        for mods in gen.TRAIN_MODS:
            for layout in (0,1):
                ev=(gen.Event(a,b,gen.VERBS[0],"A",c,d,layout),
                    gen.Event(c,d,gen.VERBS[1],"A",a,b,1-layout))
                source=gen.make_source(ev,voice,mods,"fixture","fixture")
                cores=source_mentions(source,"main_participant_slot")
                raws=source_mentions(source,"raw_mention_index")
                assert all(set(core)==set(source.gold[i*2:i*2+2]) for i,core in enumerate(cores))
                assert all(set(core).issubset(set(raw)) for core,raw in zip(cores,raws))
    return 4*len(gen.TRAIN_MODS)*2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--emit-json",action="store_true",help="Print aggregate result JSON for CI archival")
    args=parser.parse_args()
    count=self_test()
    groups,dig,manifest=frozen_evaluation()
    pools,arms=expected_exposure_pools()
    out={
       "study":"PAPER_SHORTCUT_BASELINES_V0",
       "status":"POST_HOC_DESCRIPTIVE_BASELINE_NOT_NEW_MODEL_EVALUATION",
       "date":"2026-11-21",
       "evaluation_sha256":dig,
       "exp109_manifest_sha256":sha(REPORT),
       "n_dev_anchors":len(groups),
       "fixture_count":count,
       "uses_protected_or_LI":False,
       "model_checkpoints_loaded":False,
       "methods":{
        "raw_mention_index":"Exposure-weighted (event index, role, count of name mentions) -> mention index; no voice words; includes distractor mentions.",
        "main_participant_slot":"Same mapping, but clause-core names are supplied by explicit adjunct-stripping segmentation; optimistic oracle segmentation, not an autonomous parser.",
        "training_counts":"Expected training source-pool weights: 7 AA + 7 PA shared, 2 dose sources, with per-dose variants by arm; not individual 6400-step stochastic schedule trajectories.",
        "tie_rule":"Choose the lowest name-mention index in an exact prior tie.",
        "metrics":"Deterministic 144-anchor results for each construction. Not n=8 model seed estimates and no baseline confidence intervals."
       },
       "baselines":{}
    }
    for arm,spec in arms.items():
        out["baselines"][arm]={}
        for mode in MODELS:
            f=fit(pools,spec,mode)
            out["baselines"][arm][mode]=summarize(groups,mode,f)
    neural=json.loads(MODEL_AUDIT.read_text(encoding="utf8"))
    assert neural["evaluation_sha256"]==dig
    pa=neural["descriptive_arm_conditions"]["pa_only"]
    out["archived_model_reference"]={
        "n_independent_seeds":8,"n_mps_repetitions_per_seed":2,
        "pa_only_AP_C_query_mean":pa["AP"]["C_query"]["mean"],
        "pa_only_PA_literal_C_query_mean":pa["PA_literal"]["C_query"]["mean"],
        "pa_only_AP_accuracy4_mean":pa["AP"]["accuracy4"]["mean"],
        "pa_only_AP_wrong_event_rate_mean":pa["AP"]["wrong_event_rate"]["mean"],
        "pa_only_AP_wrong_role_rate_mean":pa["AP"]["wrong_role_rate"]["mean"],
        "pa_only_AP_second_event_pair_classes":{
            key:neural["descriptive_AP_second_event_pair_classes"]["pa_only"][key]["mean"]
            for key in ("correct","reversed","collapsed","wrong_event_name")}
       }
    if args.emit_json:
        print("BEGIN_SHORTCUT_BASELINE_JSON")
        print(json.dumps(out,sort_keys=True,indent=2))
        print("END_SHORTCUT_BASELINE_JSON")
    else:
        print("PASS: frozen EXP-109 evaluation digest",dig)
        print("Fixture checks:",count,"; 144 anchors, 6 exact conditions; no model inference.")
        print("Neural PA-only C_query AP/PA:",round(out["archived_model_reference"]["pa_only_AP_C_query_mean"],4),
              round(out["archived_model_reference"]["pa_only_PA_literal_C_query_mean"],4))
        for arm,b in out["baselines"].items():
            for mode,metrics in b.items():
                ap=metrics["AP"];pa_=metrics["PA_literal"]
                print(arm,mode,
                      "C_query AP/PA",round(ap["C_query"],4),round(pa_["C_query"],4),
                      "AP reversal/collapse/wrong-event",
                      *(round(ap["second_event_pair_classes"][k],4)
                        for k in ("reversed","collapsed","wrong_event_name")))
        print("DESCRIPTIVE; generous core-slot baseline has oracle clause-participant extraction.")
if __name__=="__main__": main()
