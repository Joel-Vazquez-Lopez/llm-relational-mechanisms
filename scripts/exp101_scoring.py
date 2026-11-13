#!/usr/bin/env python3
"""EXP-101 predeclared scoring/uncertainty library. No model inference at import.

Predictions are *person identities*, not A/B labels.  The primary statistic
requires both agent and patient correct in *both* active/passive renderings.
"""
from __future__ import annotations
from collections import defaultdict
import random
import math

CONDITIONS = ("bare", "L", "I", "LI")
ARMS = ("familiar", "target")
SEEDS = (907,911,919,929,937,941,947,953)
BOOTSTRAP_SEED = 101101
BOOTSTRAP_REPS = 20_000


def score_two_role(gold_agent: str, gold_patient: str,
                   predicted_agent: str, predicted_patient: str) -> dict:
    """Truth and response consistency are never conflated."""
    gold = (gold_agent, gold_patient)
    pred = (predicted_agent, predicted_patient)
    return {"correct_both": int(pred==gold),
            "individual_correct": int(pred[0]==gold[0])+int(pred[1]==gold[1]),
            "same_person": int(pred[0]==pred[1]),
            "complete_reversal": int(pred==(gold[1],gold[0])),
            "predicted": pred, "gold": gold}


def score_voice_pair(active: dict, passive: dict) -> dict:
    """Same true event, same semantic roles, two surface voices."""
    a = score_two_role(active['gold_agent'],active['gold_patient'],
                       active['pred_agent'],active['pred_patient'])
    b = score_two_role(passive['gold_agent'],passive['gold_patient'],
                       passive['pred_agent'],passive['pred_patient'])
    if a['gold'] != b['gold']:
        raise ValueError('Voice pair changed semantic roles')
    return {"C_voice": int(a['correct_both'] and b['correct_both']),
            "S_voice": int(a['predicted']==b['predicted']),
            "active_both": a['correct_both'], "passive_both": b['correct_both'],
            "active_same_person":a['same_person'], "passive_same_person":b['same_person']}


def score_role_reverse(original: dict, reversed_: dict) -> dict:
    a=score_two_role(original['gold_agent'],original['gold_patient'],
                     original['pred_agent'],original['pred_patient'])
    b=score_two_role(reversed_['gold_agent'],reversed_['gold_patient'],
                     reversed_['pred_agent'],reversed_['pred_patient'])
    if a['gold'] != b['gold'][::-1]:
        raise ValueError('Reversal did not swap gold roles')
    return {"C_reverse":int(a['correct_both'] and b['correct_both']),
            "S_reverse":int(a['predicted']==b['predicted'][::-1]),
            "collapsed_initial":a['same_person']}


def score_event_reorder(original:dict, moved:dict)->dict:
    a=score_two_role(original['gold_agent'],original['gold_patient'],
                     original['pred_agent'],original['pred_patient'])
    b=score_two_role(moved['gold_agent'],moved['gold_patient'],
                     moved['pred_agent'],moved['pred_patient'])
    if a['gold'] != b['gold'] or original['source']==moved['source']:
        raise ValueError('Event reorder is not a changed source with same truth')
    return {"C_event":int(a['correct_both'] and b['correct_both']),
            "S_event":int(a['predicted']==b['predicted'])}


def score_four_query(rows: list[dict]) -> dict:
    """Four role/event queries on exactly the same source string."""
    if len(rows)!=4 or len({r['source'] for r in rows})!=1:
        raise ValueError('Expected four queries of one identical source')
    q={(r['event_position'],r['role']):r for r in rows}
    if set(q)!={(0,'agent'),(0,'patient'),(1,'agent'),(1,'patient')}:
        raise ValueError('Expected 2 event queries x 2 semantic roles')
    gold = [q[e,r]['gold'] for e in (0,1) for r in ('agent','patient')]
    if len(set(gold))!=4:
        raise ValueError('Four gold participant identities must be distinct')
    answers = [q[e,r]['prediction'] for e in (0,1) for r in ('agent','patient')]
    return {'C_query':int(gold==answers),'individual_accuracy':sum(a==g for a,g in zip(answers,gold))/4}


def development_cell_scores(packet_predictions: list[dict]) -> dict:
    """Input: one two-role record per distinct *familiar* DEV packet.

    Each item has condition, target_first, voice, source, correct_agent,
    correct_patient, pred_agent, pred_patient.  Caller must verify real model
    inference provenance separately. Missing cells stay missing, never zero.
    """
    groups=defaultdict(list)
    for row in packet_predictions:
        if row['condition'] not in ('bare','L','I'):
            raise ValueError('Development gate cannot inspect LI')
        if (row['target_first'],row['voice']) not in ((True,'active'),(True,'passive'),(False,'active')):
            raise ValueError('Development gate cannot use target-second passive')
        cell=f"{row['condition']}/{'first' if row['target_first'] else 'second'}/{row['voice']}"
        groups[cell].append(row)
    output={}
    for cell,rows in groups.items():
        if len({(x['source'],x['correct_agent'],x['correct_patient']) for x in rows})!=len(rows):
            raise ValueError('Repeated development event')
        checks=[score_two_role(x['correct_agent'],x['correct_patient'],x['pred_agent'],x['pred_patient']) for x in rows]
        output[cell]={"events":len(rows),
          "individual_accuracy":sum(z['individual_correct'] for z in checks)/(2*len(rows)),
          "both_correct":sum(z['correct_both'] for z in checks)/len(rows),
          "same_person":sum(z['same_person'] for z in checks)/len(rows)}
    return output


def _percentile(sorted_values, q):
    idx=(len(sorted_values)-1)*q
    lo=math.floor(idx);hi=math.ceil(idx)
    return sorted_values[lo]*(hi-idx)+sorted_values[hi]*(idx-lo) if lo!=hi else sorted_values[lo]


def paired_bootstrap(diffs:list[float],reps:int=BOOTSTRAP_REPS,seed:int=BOOTSTRAP_SEED)->dict:
    if len(diffs)!=8 or not all(math.isfinite(z) for z in diffs):
        raise ValueError('Eight finite paired-seed effects required')
    rng=random.Random(seed)
    n=8
    means=[sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(reps)]
    means.sort()
    return {'mean':sum(diffs)/n,'ci95':[_percentile(means,.025),_percentile(means,.975)],
            'n_positive':sum(x>0 for x in diffs),'seed_differences':list(diffs),
            'resamples':reps,'bootstrap_seed':seed}


def synchronized_bootstrap(views: dict[str,list[float]],reps:int=BOOTSTRAP_REPS,seed:int=BOOTSTRAP_SEED)->dict:
    if not views or any(len(v)!=8 for v in views.values()):
        raise ValueError('Same eight seeds required across outcomes')
    rng=random.Random(seed)
    samples={name:[] for name in views}
    for _ in range(reps):
        idx=[rng.randrange(8) for _ in range(8)]
        for name,v in views.items():samples[name].append(sum(v[i] for i in idx)/8)
    return {name:{'mean':sum(v)/8,'ci95':[_percentile(sorted(samples[name]),.025),
                _percentile(sorted(samples[name]),.975)]} for name,v in views.items()}


def decision_rule(primary_by_seed:dict, reference_by_seed:dict, cquery_by_seed:dict|None=None,
                  strongest_eventblind_Cquery:float=0.0)->dict:
    """Pure-data R3 criteria; all eight seed pairs, no selection. Values in [0,1].

    Inputs: {seed: {'target': float, 'familiar': float}} for each condition.
    The query-binding interpretation gate is separate from primary success.
    """
    if set(primary_by_seed)!=set(SEEDS) or set(reference_by_seed)!=set(CONDITIONS):
        raise ValueError('Incomplete seed or reference data')
    if any(set(reference_by_seed[c])!=set(SEEDS) for c in CONDITIONS):
        raise ValueError('All eight seeds required for every condition')
    effects={c:[reference_by_seed[c][s]['target']-reference_by_seed[c][s]['familiar'] for s in SEEDS]
             for c in CONDITIONS}
    if effects['LI'] != [primary_by_seed[s]['target']-primary_by_seed[s]['familiar'] for s in SEEDS]:
        raise ValueError('Primary and LI reference must use identical effects')
    main=paired_bootstrap(effects['LI'])
    abs_t=[reference_by_seed['LI'][s]['target'] for s in SEEDS]
    abs_f=[reference_by_seed['LI'][s]['familiar'] for s in SEEDS]
    other={c:paired_bootstrap(effects[c]) for c in CONDITIONS}
    target_within=[reference_by_seed['LI'][s]['target']-
      (reference_by_seed['L'][s]['target']+reference_by_seed['I'][s]['target'])/2 for s in SEEDS]
    interaction=[effects['LI'][i]-(effects['L'][i]+effects['I'][i])/2 for i in range(8)]
    iw=paired_bootstrap(interaction);tw=paired_bootstrap(target_within)
    strong=main['mean']>=.15 and main['ci95'][0]>0 and main['n_positive']>=6 and sum(abs_t)/8>=.65
    ref_pass=all(other[c]['mean']>=.15 and other[c]['ci95'][0]>0 for c in ('bare','L','I'))
    attenuation=ref_pass and main['ci95'][1]<.10 and iw['ci95'][1]<0
    poor=attenuation and sum(abs_t)/8<.75 and tw['ci95'][1]<-.10
    label=('strong_second_event_transfer' if strong else
           'attenuated_incremental_advantage_and_poor_LI_performance' if poor else
           'attenuated_incremental_advantage' if attenuation else
           'partial_or_inconclusive')
    saturation=sum(abs_t)/8>=.90 and sum(abs_f)/8>=.90
    query_gate=None
    if cquery_by_seed is not None:
        if set(cquery_by_seed)!=set(SEEDS): raise ValueError('All eight C_query seeds required')
        qa=[cquery_by_seed[s]['target'] for s in SEEDS]
        qd=[cquery_by_seed[s]['target']-cquery_by_seed[s]['familiar'] for s in SEEDS]
        qb=paired_bootstrap(qd)
        absolute=sum(qa)/8
        query_gate={'absolute_C_query':absolute,'paired_exposure_effect':qb,
                    'query_competence':absolute>=.65 and absolute-strongest_eventblind_Cquery>=.15,
                    'exposure_improves_query_binding':absolute>=.65 and absolute-strongest_eventblind_Cquery>=.15 and qb['ci95'][0]>0,
                    'best_event_blind_baseline_C_query':strongest_eventblind_Cquery}
    return {'decision':label,'strong_second_event':strong,'reference_replication':ref_pass,
        'attenuation_of_incremental_benefit':attenuation,'poor_compositional_transfer':poor,
        'high_high_saturation':saturation,'primary':main,'reference_effects':other,
        'within_target_LI_minus_LI_mean':tw,'interaction':iw,
        'absolute_LI_target_mean':sum(abs_t)/8,'absolute_LI_familiar_mean':sum(abs_f)/8,
        'cquery_interpretation_gate':query_gate}
