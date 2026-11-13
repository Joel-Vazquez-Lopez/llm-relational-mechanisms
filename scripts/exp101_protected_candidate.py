#!/usr/bin/env python3
"""EXP-101 future one-time protected evaluation implementation (LOCKED).

No CLI phase can render protected prompts. Pure metadata counts can be audited
pre-freeze. The full renderer and inference are callable ONLY after approved
release, a recomputed eligible dev gate, a separate one-time unlock, and a
persistent pre-inference lock. Do not invoke those functions in review.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import json
import hashlib

from exp101_production_candidate import all_matchings,VERBS,STEPS,SEEDS,ARMS,stable_hash
from exp101_scoring import (score_voice_pair,score_role_reverse,
    score_event_reorder,score_four_query,score_two_role)


def future_anchor_metadata():
    """Only frozen *pair identifiers*, no sentence, question or gold labels."""
    _,_,sealed=all_matchings()
    out=[]
    for matching in sealed:
        for i,pair in enumerate(matching):
            for vi,verb in enumerate(VERBS):
                for choice in (0,1):
                    shift=1+(vi+choice+i)%7
                    other=matching[(i+shift)%8]
                    assert pair!=other
                    out.append({'pair':pair,'other_pair':other,
                      'verb':verb,'other_verb':VERBS[(vi+choice+1)%6],
                      'other_voice':('active','passive')[(i+vi+choice)%2],
                      'target_layout':(i+choice)%2,'other_layout':(i+vi+choice)%2,
                      'target_loc_swap':choice,'other_loc_swap':(i+choice)%2,
                      'choice':choice})
    assert len(out)==288
    return out


def future_manifest_plan():
    """No protected examples generated; exact predicted inference counts only."""
    ids=future_anchor_metadata()
    return {'anchors':len(ids),'anchor_metadata_sha256':stable_hash(ids),
            'conditions':['bare','L','I','LI'],
            'variants_per_anchor_condition':8,
            'target_queries_per_variant':2,
            'extra_LI_other_event_queries_per_variant':2,
            'models':16,'target_role_questions_per_model':288*4*8*2,
            'extra_LI_questions_per_model':288*8*2,
            'total_questions_per_model':23040,
            'total_questions_across_models':368640,
            'no_protected_text_generated':True}


def _require_unlock(root:Path,release:dict):
    """Separate protected authorization; even a valid train approval is insufficient."""
    from exp101_release import require_approved_release
    root=Path(root)
    approved=json.loads((root/'reports/exp101/frozen_release.json').read_text())
    if not approved.get('protected_evaluation_authorized'):
        raise PermissionError('Protected test not independently authorized')
    if approved.get('fingerprint')!=release['fingerprint']:
        raise PermissionError('Protected authorization refers to wrong release')
    gatefile=root/'reports/exp101/dev_gate_report.json'
    if not gatefile.exists():raise PermissionError('No scored nine-cell dev competence report')
    gate=json.loads(gatefile.read_text())
    if gate.get('release_fingerprint')!=release['fingerprint'] or not gate.get('eligible'):
        raise PermissionError('Wrong or failing dev gate')
    # Recompute gate from individual cell results, not a manually set flag.
    from exp101_production_candidate import competence_gate
    scores={(int(row['seed']),row['arm']):row['cells'] for row in gate['per_model']}
    if not competence_gate(scores)['eligible']:
        raise PermissionError('Recomputed joint competence failed')
    approved_unlock=root/'reports/exp101/protected_unlock_approved.json'
    if not approved_unlock.exists():raise PermissionError('No separately approved one-time unlock')
    obj=json.loads(approved_unlock.read_text())
    if obj.get('fingerprint')!=release['fingerprint'] or obj.get('status')!='ONE_TIME_APPROVED':
        raise PermissionError('Invalid separate protected unlock')
    lock=root/'reports/exp101/protected_opened_once.json'
    if lock.exists():raise PermissionError('Protected evaluation already opened once')
    # All expected final checkpoints must exist before consuming the single unlock.
    # Contents are not opened here; their hashes are verified on model load.
    for seed in SEEDS:
        for arm in ARMS:
            if not (root/f'results/exp101/seed_{seed}/{arm}/state.pt').is_file():
                raise PermissionError(f'Missing final checkpoint {seed}/{arm}')
    # Persistent lock is written atomically BEFORE any protected item generation.
    fd=lock.open('x')
    try:
        json.dump({'release_fingerprint':release['fingerprint'],
                   'status':'OPENED_BEFORE_GENERATION'},fd)
    finally:fd.close()
    return lock


def _build_packets_AFTER_unlock():
    """Restricted renderer. Never call except after _require_unlock."""
    from exp101_generator_preflight import Event,render_packet,other_event_query,validate_case
    out=[]
    meta=future_anchor_metadata()
    for anchor,info in enumerate(meta):
        pair=info['pair'];other_pair=info['other_pair']
        names=tuple(sorted(set(pair+other_pair)))
        assert len(names)==4
        for cond in ('bare','L','I','LI'):
            for reverse in (0,1):
                a,p=pair if reverse==0 else pair[::-1]
                c,d=other_pair
                c,d=(c,d) if (anchor%2)==0 else (d,c)
                for first in (True,False):
                    for voice in ('active','passive'):
                        trefs=(c,d) if info['target_loc_swap']==0 else (d,c)
                        orefs=pair if info['other_loc_swap']==0 else pair[::-1]
                        target=Event(a,p,info['verb'],*trefs,voice,info['target_layout'])
                        other=Event(c,d,info['other_verb'],*orefs,info['other_voice'],info['other_layout'])
                        pkt=render_packet(target,other,names,cond,first)
                        validate_case(pkt)
                        key=(anchor,cond,reverse,first,voice)
                        out.append((key,'target',pkt))
                        if cond=='LI':
                            other_pkt=other_event_query(pkt)
                            validate_case(other_pkt)
                            out.append((key,'other',other_pkt))
    # Each target packet is two questions. Additional LI other query also 2.
    assert len(out)==(288*4*8)+(288*8)
    return out


def protected_inference_after_authorization(*,root:Path,device:str='cpu'):
    """Future-only helper: one-time protected packets and inference, no CLI.

    Models/checkpoints must already exist and have passed DEV; this function
    never trains. Calling before explicit approval raises PermissionError.
    """
    from exp101_release import require_approved_release
    from exp101_production_candidate import make_model,encode,VOCAB_WORDS,VOCAB
    import exp101_runtime as rt
    import torch
    root=Path(root)
    release=require_approved_release(root/'reports/exp101/frozen_release.json')
    _require_unlock(root,release)
    packets=_build_packets_AFTER_unlock()
    results={}
    for seed in SEEDS:
        for arm in ARMS:
            model,optim,gb,gd=rt.init_state(seed,make_model,device=device)
            chk=root/f'results/exp101/seed_{seed}/{arm}/state.pt'
            datahash=release['dataset_manifest_sha256']
            step=rt.load_state(chk,model=model,optimizer=optim,gbase=gb,gdose=gd,
                seed=seed,arm=arm,device=device,
                release_fingerprint=release['fingerprint'],data_hash=datahash)
            if step!=STEPS:raise PermissionError('Non-final checkpoint')
            model.eval();preds=[]
            with torch.inference_mode():
                for start in range(0,len(packets),32):
                    batch=packets[start:start+32]
                    items=[(p,r) for _,_,p in batch for r in ('agent','patient')]
                    ids,rds,gold,choices=encode(items)
                    logits=model(ids.to(device),rds.to(device)).float()
                    c=choices.to(device)
                    arg=c.gather(1,logits.gather(1,c).argmax(1,keepdim=True)).flatten().cpu().tolist()
                    full=logits.argmax(1).cpu().tolist()
                    for j,(key,q,p) in enumerate(batch):
                        preds.append({'key':key,'query':q,'source':p.text,
                            'gold_agent':p.correct_agent,'gold_patient':p.correct_patient,
                            'pred_agent':VOCAB_WORDS[arg[2*j]],'pred_patient':VOCAB_WORDS[arg[2*j+1]],
                            'full_agent':VOCAB_WORDS[full[2*j]],'full_patient':VOCAB_WORDS[full[2*j+1]],
                            'target_first':p.target_first,'voice':p.event.voice,
                            'other_event_names':(p.other_event.agent,p.other_event.patient),
                            'candidate_names':p.names})
            results[f'{seed}/{arm}']=analyze_prediction_rows(preds)
    from exp101_scoring import decision_rule
    ref={cond:{} for cond in ('bare','L','I','LI')}
    primary={};query={}
    for seed in SEEDS:
        for arm in ARMS:
            item=results[f'{seed}/{arm}']
            for cond in ref:ref[cond].setdefault(seed,{})[arm]=item['C_voice_by_condition'][cond]
            query.setdefault(seed,{})[arm]=item['C_query_LI']
    primary=ref['LI']
    inference=decision_rule(primary,ref,query,strongest_eventblind_Cquery=0.0)
    out=root/'reports/exp101/protected_one_time_summary.json'
    if out.exists():raise PermissionError('Never overwrite protected summary')
    out.write_text(json.dumps({'release_fingerprint':release['fingerprint'],
        'results':results,'cohort_inference':inference},sort_keys=True,indent=2)+'\n')
    return out


def analyze_prediction_rows(rows:list[dict],anchor_count:int=288):
    """Pure-data scoring; can be unit-tested with synthetic all-correct rows."""
    ix={(tuple(r['key']),r['query']):r for r in rows}
    if len(ix)!=len(rows):raise ValueError('Repeated evaluation key')
    sums=defaultdict(list)
    voice_consistency=defaultdict(list)
    reversal=defaultdict(list)
    event_order=defaultdict(list)
    query_scores=[]
    error_counts=defaultdict(Counter)
    for row in rows:
        if row['query']!='target':continue
        cond=row['key'][1]
        own={row['gold_agent'],row['gold_patient']}
        other=set(row.get('other_event_names',()))
        for role in ('agent','patient'):
            pred=row['pred_'+role];gold=row['gold_'+role]
            category=('correct' if pred==gold else
                      'wrong_role' if pred in own else
                      'wrong_event' if pred in other else 'outside_pool')
            error_counts[cond][category]+=1
            if 'full_'+role in row:
                error_counts[cond]['full_vocab_correct']+=int(row['full_'+role]==gold)
        error_counts[cond]['pairs']+=1
        error_counts[cond]['same_person_pairs']+=int(row['pred_agent']==row['pred_patient'])
        error_counts[cond]['complete_role_reversal_pairs']+=int(
            (row['pred_agent'],row['pred_patient'])==(row['gold_patient'],row['gold_agent']))
    for anchor in range(anchor_count):
      for cond in ('bare','L','I','LI'):
        for rev in (0,1):
          for first in (True,False):
            ka=(anchor,cond,rev,first,'active')
            kp=(anchor,cond,rev,first,'passive')
            a=ix[(ka,'target')];p=ix[(kp,'target')]
            x=score_voice_pair(a,p)
            if first is False:
                sums[cond].append(x['C_voice'])
                voice_consistency[cond].append(x['S_voice'])
            if cond=='LI':
              for k in (ka,kp):
                t=ix[(k,'target')];o=ix[(k,'other')]
                # Two queries per event, four gold-distinct people on same source.
                f=0 if t['target_first'] else 1
                queries=[]
                for e,row in ((f,t),(1-f,o)):
                  for role in ('agent','patient'):
                    queries.append({'source':row['source'],'event_position':e,
                       'role':role,'gold':row['gold_'+role],
                       'prediction':row['pred_'+role]})
                query_scores.append(score_four_query(queries)['C_query'])
    for anchor in range(anchor_count):
      for cond in ('bare','L','I','LI'):
        for first in (True,False):
          for voice in ('active','passive'):
            r0=ix[((anchor,cond,0,first,voice),'target')]
            r1=ix[((anchor,cond,1,first,voice),'target')]
            reversal[cond].append(score_role_reverse(r0,r1))
        for reverse in (0,1):
          for voice in ('active','passive'):
            r0=ix[((anchor,cond,reverse,True,voice),'target')]
            r1=ix[((anchor,cond,reverse,False,voice),'target')]
            event_order[cond].append(score_event_reorder(r0,r1))
    if (set(sums)!=set(('bare','L','I','LI')) or
        any(len(v)!=2*anchor_count for v in sums.values()) or
        len(query_scores)!=8*anchor_count):
        raise ValueError('Missing reference/LI primary or cross-event packets')
    return {'C_voice_by_condition':{c:sum(v)/len(v) for c,v in sums.items()},
      'S_voice_by_condition':{c:sum(v)/len(v) for c,v in voice_consistency.items()},
      'C_reverse_by_condition':{c:sum(z['C_reverse'] for z in v)/len(v) for c,v in reversal.items()},
      'S_reverse_by_condition':{c:sum(z['S_reverse'] for z in v)/len(v) for c,v in reversal.items()},
      'C_event_by_condition':{c:sum(z['C_event'] for z in v)/len(v) for c,v in event_order.items()},
      'S_event_by_condition':{c:sum(z['S_event'] for z in v)/len(v) for c,v in event_order.items()},
      'C_query_LI':sum(query_scores)/len(query_scores),
      'voice_pair_counts':{c:len(v) for c,v in sums.items()},
      'same_source_packet_count':len(query_scores),
      'error_taxonomy_counts':{c:dict(v) for c,v in error_counts.items()},
      'fully_scored':True}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('metadata-only',))
    p.parse_args()
    print(json.dumps(future_manifest_plan(),sort_keys=True,indent=2))
    print('NO PROTECTED TEXT GENERATED; NO CHECKPOINTS LOADED')
