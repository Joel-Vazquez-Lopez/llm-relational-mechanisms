#!/usr/bin/env python3
"""PHASE2-01: matched direct-construction exposure dose-response.

New scratch-trained models; new non-reserved development permutations; same
original grammar. No protected/reserved/L+I generation or scoring. This is a
separate Phase 2 study, NOT an extra EXP-109--116 exploratory intervention.

  python3 scripts/phase2_01_coverage_curve_v0.py --phase prepare
  python3 scripts/phase2_01_coverage_curve_v0.py --phase smoke --device mps
  # Freeze this script, launcher, protocol, and manifest in ONE Git commit.
  bash scripts/run_phase2_01_all_v0.sh
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean, stdev
import json
import math
import subprocess

import torch

import exp105_generator_preflight as gen
import exp105_position_control_preflight_v1 as frames
import exp105_training_schedule_preflight_v3 as sched
import exp105_development_pilot_v0 as pilot
import exp106_voice_recombination_pilot_v0 as exp106
import exp109_replication_v0 as exp109
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/phase2_01/coverage_curve_v0'
CKPT=ROOT/'results/phase2_01/coverage_curve_v0'
MANIFEST=OUT/'manifest.json'
PROTOCOL=ROOT/'paper/PHASE2_01_coverage_curve_v0_protocol.md'
LAUNCHER=ROOT/'scripts/run_phase2_01_all_v0.sh'
SEEDS=(11701,11703,11707,11709,11713,11719,11721,11727)
REPEATS=(1,2)
STEPS=6400
# p is the fraction of 2-source, token-balanced PP+AA dose pairs REPLACED
# by PA+AP. It is NOT the overall fraction of AP in the 16-source minibatch.
DOSES=(('p000',0),('p001',64),('p005',320),('p020',1280),('p100',6400))
ARMS=tuple(a for a,_ in DOSES)
DIRECTIONS=('AP','PA_literal','AA','PP')
EVAL_SEED=1172026
EVAL_PER_MOD=16
TC95=2.364624251
assert not set(SEEDS)&(set(exp109.SEEDS)|set(exp114.SEEDS)|set(sched.SEEDS))
assert STEPS==exp114.STEPS==2*sched.STEPS


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:
            raise FileExistsError('Existing file differs; will not overwrite: '+str(path))
        print('VERIFIED',path,flush=True)
    else:
        with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
        print('SAVED',path,flush=True)


def previous_dev_exclusions(dev):
    """Exclude old dev, EXP109 and EXP114 lexical surfaces/core orders."""
    old_surfaces,old_perms,exp109_digest=exp114.previous_sources(dev)
    old114,audit114=exp114.novel_evaluation(dev)
    legacy114=json.loads(exp114.MANIFEST.read_text())
    assert audit114['evaluation_sha256']==legacy114['evaluation_audit']['evaluation_sha256']
    _train,dev_sets,_=gen.partitions()
    set2ix={frozenset(x):i for i,x in enumerate(dev_sets)}
    for g in old114:
        for src in g.values():old_surfaces.add(src.text)
        ix=set2ix[frozenset(g['AP'].gold)]
        old_perms[ix].add(tuple(g['AP'].gold))
        old_perms[ix].add(tuple(g['PA_literal'].gold))
    return old_surfaces,old_perms,exp109_digest,audit114['evaluation_sha256']


def evaluation(dev):
    """Create new held-out source orders without opening any reserved partition."""
    _train,dev_sets,partition=gen.partitions()
    old_surfaces,old_orders,d109,d114=previous_dev_exclusions(dev)
    rng=Random(EVAL_SEED)
    cells={m:[] for m in gen.TRAIN_MODS}
    checked_cores=0
    for i,nameset in enumerate(dev_sets):
        options=[p for p in permutations(nameset,4) if p not in old_orders[i]]
        rng.shuffle(options)
        accepted=False
        for p in options:
            a,b,c,d=p
            if (c,d,a,b) in old_orders[i] or (b,c,d,a) in old_orders[i]:continue
            v0=gen.VERBS[(i+3)%len(gen.VERBS)]
            v1=gen.VERBS[(i+1)%len(gen.VERBS)]
            assert v0!=v1
            events=(gen.Event(a,b,v0,'A',c,d,i%2),
                    gen.Event(c,d,v1,'A',a,b,(i+1)%2))
            staged=[]
            for mod in gen.TRAIN_MODS:
                core=f'newdev117-{i}'
                ap=gen.make_source(events,'AP',mod,'phase2_newdev',core)
                left,right=ap.text[:-1].split(' ; ')
                pa=replace(ap,text=right+' ; '+left+'.',gold=ap.gold[2:]+ap.gold[:2],
                           voice='PA',mods=ap.mods[::-1])
                aa=gen.make_source(events,'AA',mod,'phase2_newdev',core)
                pp=gen.make_source(events,'PP',mod,'phase2_newdev',core)
                group={'AP':ap,'PA_literal':pa,'AA':aa,'PP':pp}
                if any(s.text in old_surfaces for s in group.values()):break
                assert all(gen.parse_independent(s.text)==s.gold for s in group.values())
                assert all('LI' not in s.mods for s in group.values())
                assert len(set(ap.gold))==4
                assert Counter(gen.tokens(ap.text))==Counter(gen.tokens(pa.text))
                for src in group.values():
                    for q in gen.Q:
                        assert len(['<bos>']+gen.tokens(src.text)+['<sep>']+
                                   gen.tokens(q)+['<answer>']) <= pilot.MAX_TOKENS
                staged.append((mod,group))
            else:
                for mod,g in staged:cells[mod].append(g)
                accepted=True
                checked_cores+=1
                break
        assert accepted,(i,len(options))
    groups=[]
    for mod in sorted(cells):
        assert len(cells[mod])>=EVAL_PER_MOD
        groups.extend(rng.sample(cells[mod],EVAL_PER_MOD))
    assert len(groups)==144
    all_surfaces=[g[d].text for g in groups for d in DIRECTIONS]
    assert len(all_surfaces)==len(set(all_surfaces))
    assert all(s not in old_surfaces for s in all_surfaces)
    audit={
      'anchor_count':144,'presentations':len(all_surfaces),
      'per_modifier':{str(x):EVAL_PER_MOD for x in sorted(gen.TRAIN_MODS)},
      'evaluated_directions':list(DIRECTIONS),
      'dev_name_sets_considered':len(dev_sets),'novel_core_orderings_found':checked_cores,
      'prior_exp109_digest':d109,'prior_exp114_digest':d114,
      'partition_digest':partition['partition_digest'],
      'new_surface_overlap_with_prior_dev_exp109_exp114':0,
      'new_core_ordering_overlap_with_prior_dev_exp109_exp114':0,
      'same_grammar_and_dev_name_sets':True,
      'no_protected_reserved_or_LI':True,
      'evaluation_sha256':digest([[g[d].text,list(g[d].gold),list(g[d].mods)]
                                      for g in groups for d in DIRECTIONS])}
    return groups,audit


def schedule(seed,common,dose,bp,dp,limit=STEPS):
    """EXP109/114 identical sampling algorithm with newly assigned seed set."""
    assert seed in SEEDS and 0<limit<=STEPS
    paired=[(d['PA'],d['AP'],0) for d in dose]
    count=0
    for half,s in enumerate((seed,seed+100000)):
        rng=Random(s+1060000)
        cells=[];block=-1
        for inner,base,_old,_bs,_ds in sched.iter_schedule(s,common,paired,bp,dp):
            if (inner-1)//sched.BLOCK!=block:
                block=(inner-1)//sched.BLOCK
                cells=list(sched.STRATA);rng.shuffle(cells)
            mod,fi=cells[(inner-1)%sched.BLOCK]
            a=rng.choice(dp[(mod,fi,0)]);b=rng.choice(dp[(mod,fi,1)])
            assert dose[a]['PA'].core_id!=dose[b]['PA'].core_id
            if rng.randrange(2):a,b=b,a
            step=half*sched.STEPS+inner
            count+=1
            yield step,base,(a,b),(mod,fi)
            if count==limit:return
    assert count==limit


def allocate_coverage(seed,strata_by_step):
    """Frozen nested, approximately balanced allocations across all 45 cells."""
    assert len(strata_by_step)==STEPS
    strata=defaultdict(list)
    for step,cell in enumerate(strata_by_step,1):strata[cell].append(step)
    assert set(strata)==set(sched.STRATA) and len(strata)==45
    rng=Random(seed+1170227)
    ordered_cells=list(sorted(strata));rng.shuffle(ordered_cells)
    for cell in ordered_cells:rng.shuffle(strata[cell])
    allocations={}
    for arm,n in DOSES:
        if n==STEPS:
            chosen=set(range(1,STEPS+1))
        else:
            q,r=divmod(n,len(ordered_cells))
            chosen=set()
            for ix,cell in enumerate(ordered_cells):
                allocation=q+int(ix<r)
                assert len(strata[cell])>=allocation
                chosen.update(strata[cell][:allocation])
        assert len(chosen)==n
        allocations[arm]=chosen
    for left,right in zip(ARMS,ARMS[1:]):
        assert allocations[left].issubset(allocations[right])
    return allocations


def training_schedule(seed,common,dose,bp,dp,limit=STEPS):
    """Yield step indices and preallocated coverage labels, never tuned on scores."""
    all_rows=list(schedule(seed,common,dose,bp,dp))
    allocations=allocate_coverage(seed,[r[3] for r in all_rows])
    for step,base,pair,cell in all_rows[:limit]:
        direct={arm:step in allocations[arm] for arm in ARMS}
        yield step,base,pair,cell,direct


def sources_for(common,dose,base,pair):
    """Matched pairs: equal gold, event-voice marginals, and token multiset."""
    arms=exp106.arm_sources(common,dose,base,pair)
    exp106.verify_step(arms)
    return arms['indirect_pp_aa'],arms['direct_pa_ap']


def schedule_audit(seed,common,dose,bp,dp):
    h=sha256();cells=Counter();counts=Counter();hist=defaultdict(Counter)
    for step,base,pair,cell,direct in training_schedule(seed,common,dose,bp,dp):
        # The 14 shared base sources are exactly identical across arms.
        # Hence checking only the TWO changed dose sources proves full-batch
        # token multiset equality without re-tokenizing 14 shared sources.
        a,b=pair;da,db=dose[a],dose[b]
        assert da['PP'].gold==da['PA'].gold and db['AA'].gold==db['AP'].gold
        assert Counter(gen.tokens(da['PP'].text)+gen.tokens(db['AA'].text))==Counter(
                       gen.tokens(da['PA'].text)+gen.tokens(db['AP'].text))
        # Both variants have 1 passive first and 1 passive second event.
        assert da['PP'].voice=='PP' and db['AA'].voice=='AA'
        assert da['PA'].voice=='PA' and db['AP'].voice=='AP'
        assert da['PA'].core_id != db['AP'].core_id
        cells[cell]+=1
        for arm,flag in direct.items():
            if flag:counts[arm]+=1;hist[arm][cell]+=1
        hh={'step':step,'base':base,'dose':list(pair),
            'stratum':[list(cell[0]),cell[1]],'direct':direct}
        h.update((json.dumps(hh,separators=(',',':'))+'\n').encode())
    assert {a:counts[a] for a in ARMS}==dict(DOSES)
    assert set(cells)==set(sched.STRATA)
    assert all(max(hist[a].values(),default=0)-min((hist[a][c] for c in cells),default=0)<=1 for a in ('p001','p005','p020'))
    return {'seed':seed,'steps':STEPS,'schedule_sha256':h.hexdigest(),
            'direct_pair_counts':dict(DOSES),'dose_pair_total':STEPS,
            'n_train_sources_per_step':16,'n_answers_per_step':64,
            'matched_token_bags_every_step':True,'matched_event_position_voice_marginals':True,
            'all_rates_nested':True,'modifier_frame_use_counts':{str(k):v for k,v in cells.items()},
            'no_LI_or_protected_examples':True}


def dependency_hashes():
    return {p.name:sha(p) for p in
            (SCRIPT,Path(gen.__file__),Path(frames.__file__),Path(sched.__file__),
             Path(pilot.__file__),Path(exp106.__file__),Path(exp109.__file__),
             Path(exp112.__file__),Path(exp113.__file__),Path(exp114.__file__))}


def protocol(vocab,dev,audit,partition):
    return {'study':'PHASE2-01','version':'MATCHED_COVERAGE_CURVE_V0',
     'stage':'PROSPECTIVE_STUDY_NOT_EXTERNAL_PREREGISTRATION',
     'seeds':list(SEEDS),'repeats':list(REPEATS),
     'arms':list(ARMS),'dose_counts_per_6400':dict(DOSES),
     'dose_definition':'percentage of PP+AA pairs converted to matched PA+AP pairs',
     'train_batch_sources':16,'train_batch_questions':64,
     'steps':STEPS,'device':'mps','training_schedule':'EXP109/114 new seeds with nested stratified allocation',
     'vocab_digest':digest(sorted(vocab.items())),
     'legacy_dev_digest':digest([(s.text,s.gold) for s in dev]),
     'partition_digest':partition,
     'evaluation_seed':EVAL_SEED,'evaluation_digest':audit['evaluation_sha256'],
     'evaluation_count_per_mod':EVAL_PER_MOD,'evaluation_conditions':list(DIRECTIONS),
     'primary':'p020 minus p000 AP second-agent/patient mean candidate-restricted accuracy',
     'secondary':'p100 minus p000 AP second-agent/patient mean candidate-restricted accuracy; gated on primary',
     'criterion':'paired n=8 seed-level two-sided 95% Student t interval lower>0; ordered gate',
     'outcomes':['AP per-query accuracy and strict C_query','same-event wrong-role',
                 'wrong-event accuracy','PA first-event retention','AA and PP controls'],
     'zero_optional_stopping':True,'no_competence_filter':True,'no_checkpoint_selection':True,
     'no_experimental_patch':True,'no_protected_or_LI':True,
     'same_grammar_not_structural_OOD':True,
     'dependency_hashes':dependency_hashes()}


def load_and_verify():
    if not MANIFEST.exists():raise FileNotFoundError('First --phase prepare and commit frozen manifest')
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    p=protocol(vocab,dev,audit,partition)
    m=json.loads(MANIFEST.read_text())
    assert m['protocol']==p and m['evaluation_audit']==audit
    assert m['protocol_digest']==digest(p)
    return m,(common,dose,bp,dp,dev,vocab),groups


def prepare():
    if MANIFEST.exists():
        load_and_verify();print('PREPARE PASS: existing frozen PHASE2-01 manifest verified');return
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    assert len(groups)==144
    p=protocol(vocab,dev,audit,partition)
    print('PHASE2-01: auditing eight independent training schedules; NO model scoring',flush=True)
    schedules={str(s):schedule_audit(s,common,dose,bp,dp) for s in SEEDS}
    assert len({v['schedule_sha256'] for v in schedules.values()})==8
    write_once(MANIFEST,{'protocol':p,'protocol_digest':digest(p),
                         'evaluation_audit':audit,'schedule_audit':schedules})
    print('PREPARE PASS: matched 0/1/5/20/100% dose curve; 144 new 4-condition dev anchors; no reserved/LI',flush=True)
    print('COMMIT runner + launcher + protocol + manifest in SAME commit before RUN')


def freeze_verified():
    frozen=('scripts/phase2_01_coverage_curve_v0.py',
            'scripts/run_phase2_01_all_v0.sh',
            'paper/PHASE2_01_coverage_curve_v0_protocol.md',
            'reports/phase2_01/coverage_curve_v0/manifest.json')
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        c=subprocess.check_output(['git','log','-n','1','--format=%H','--',frozen[0]],cwd=ROOT,text=True).strip()
        if not c:raise RuntimeError('Runner never committed')
        for rel in frozen:
            committed=subprocess.check_output(['git','show',c+':'+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
            if committed!=(ROOT/rel).read_bytes():raise RuntimeError('Frozen file differs: '+rel)
        subprocess.run(['git','merge-base','--is-ancestor',c,head],cwd=ROOT,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return c
    except (FileNotFoundError,subprocess.CalledProcessError,RuntimeError) as e:
        raise RuntimeError('Before training commit the exact 4 PHASE2-01 frozen files together') from e


def models(seed,device,n_vocab):
    torch.manual_seed(seed)
    shared=pilot.Model(n_vocab).to(device)
    result={}
    for arm in ARMS:
        model=deepcopy(shared).to(device)
        assert all(torch.equal(a,b) for a,b in zip(model.parameters(),shared.parameters()))
        optimizer=torch.optim.AdamW(model.parameters(),lr=pilot.LR,weight_decay=pilot.WEIGHT_DECAY)
        result[arm]=(model,optimizer)
    return result


def encoded_data(groups,vocab):
    return {d:pilot.encode_sources([g[d] for g in groups],vocab) for d in DIRECTIONS}


def score_model(model,encoded,device):
    return {d:exp112.metrics(exp113.no_touch_identity(model,encoded[d],device),encoded[d])
            for d in DIRECTIONS}


def smoke(device):
    common,dose,bp,dp,_partition,dev,vocab=exp109.resources()
    # Smoke uses OLD, already explored sentences; must never score PHASE2 new set.
    rng=Random(SEEDS[0])
    train=list(training_schedule(SEEDS[0],common,dose,bp,dp,limit=2))
    assert train[0][0]==1 and train[1][0]==2
    net=models(SEEDS[0],device,len(vocab))
    for _,base,pair,cell,selected in train:
        indirect,direct=sources_for(common,dose,base,pair)
        ie=pilot.encode_sources(indirect,vocab,check=False)
        de=pilot.encode_sources(direct,vocab,check=False)
        for arm in ARMS:
            enc=de if selected[arm] else ie
            loss=pilot.one_update(*net[arm],enc,device)
            assert math.isfinite(loss)
    legacy=exp109.evaluation(dev)[0][:2]
    data={d:pilot.encode_sources([g[d] for g in legacy],vocab) for d in ('AP','PA_literal')}
    model=net[ARMS[0]][0].eval()
    for d in data:
        out=exp113.no_touch_identity(model,data[d],device)
        met=exp112.metrics(out,data[d])
        assert 0<=met['C_query']<=1
    print('PHASE2-01 SMOKE PASS: matched training modes, two updates, legacy-dev scoring; NEW EVAL NOT SCORED')


def run(seed,repeat,device):
    if device.type!='mps':raise ValueError('Frozen PHASE2 training requires MPS; CPU smoke only')
    assert seed in SEEDS and repeat in REPEATS
    man,res,groups=load_and_verify()
    freeze_commit=freeze_verified()
    common,dose,bp,dp,dev,vocab=res
    record_path=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
    cp_path=CKPT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
    if record_path.exists() or cp_path.exists():
        raise FileExistsError('Refusing overwrite: '+str(record_path))
    encoded=encoded_data(groups,vocab)
    nets=models(seed,device,len(vocab))
    h=sha256();hist=[]
    for step,base,pair,cell,selected in training_schedule(seed,common,dose,bp,dp):
        indirect,direct=sources_for(common,dose,base,pair)
        ie=pilot.encode_sources(indirect,vocab,check=False)
        de=pilot.encode_sources(direct,vocab,check=False)
        losses={}
        for arm in ARMS:
            losses[arm]=pilot.one_update(*nets[arm],de if selected[arm] else ie,device)
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
              'stratum':[list(cell[0]),cell[1]],'direct':selected},separators=(',',':'))+'\n').encode())
        if step in exp109.MILESTONES:
            hist.append({'step':step,'losses':losses})
            print('STEP',step,'losses',' '.join(f'{a}:{losses[a]:.4f}' for a in ARMS),flush=True)
    assert h.hexdigest()==man['schedule_audit'][str(seed)]['schedule_sha256']
    outcomes={}
    for arm in ARMS:
        outcomes[arm]=score_model(nets[arm][0].eval(),encoded,device)
    record={'study':'PHASE2-01','status':'SEALED_PROSPECTIVE_COVERAGE_CURVE',
            'seed':seed,'repeat':repeat,'steps':STEPS,'device':'mps',
            'freeze_commit':freeze_commit,'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'schedule_sha256':h.hexdigest(),'script_sha256':sha(SCRIPT),
            'loss_history':hist,'outcomes':outcomes,
            'no_protected_or_LI':True,'not_out_of_grammar':True}
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
    torch.save({'study':'PHASE2-01','seed':seed,'repeat':repeat,'step':STEPS,
                'protocol_digest':man['protocol_digest'],
                'models':{a:nets[a][0].state_dict() for a in ARMS}},cp_path)
    with record_path.open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print('SAVED',record_path,flush=True);print('SAVED',cp_path,flush=True)
    print('REPORT SEALED; inspect no individual scores until all 16 training runs complete',flush=True)


def ci(vals):
    assert len(vals)==8
    m=mean(vals);radius=TC95*stdev(vals)/math.sqrt(8)
    return {'mean':m,'ci95':[m-radius,m+radius],
            'per_seed':dict(zip(map(str,SEEDS),vals)),
            'positive_seeds':sum(x>0 for x in vals),'negative_seeds':sum(x<0 for x in vals)}


def focus(rec,arm,direction,metric):
    qs=('second_agent','second_patient') if direction=='AP' else ('first_agent','first_patient')
    r=rec['outcomes'][arm][direction]
    return mean(r['by_query'][q][metric] for q in qs)


def summary():
    man,_,_=load_and_verify()
    records={};missing=[]
    for seed in SEEDS:
        for repeat in REPEATS:
            p=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
            if not p.exists():missing.append(p.name);continue
            r=json.loads(p.read_text())
            assert (r['study'],r['seed'],r['repeat'],r['steps'])==('PHASE2-01',seed,repeat,STEPS)
            assert r['script_sha256']==sha(SCRIPT)
            assert r['protocol_digest']==man['protocol_digest']
            assert r['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
            assert r['schedule_sha256']==man['schedule_audit'][str(seed)]['schedule_sha256']
            assert r['no_protected_or_LI'] and r['not_out_of_grammar']
            assert set(r['outcomes'])==set(ARMS)
            records[seed,repeat]=r
    if missing:
        print('PHASE2-01 INCOMPLETE: '+str(len(missing))+' of 16 runs missing; no aggregate written')
        print('Missing:',', '.join(missing[:8]));return
    def aggregate(fn):return ci([mean(fn(records[s,rep]) for rep in REPEATS) for s in SEEDS])
    arms={}
    for arm in ARMS:
        arms[arm]={}
        for d in DIRECTIONS:
            arms[arm][d]={
                'C_query':aggregate(lambda r,a=arm,x=d:r['outcomes'][a][x]['C_query']),
                'focus_accuracy':aggregate(lambda r,a=arm,x=d:focus(r,a,x,'native_correct')),
                'focus_wrong_event':aggregate(lambda r,a=arm,x=d:focus(r,a,x,'wrong_event')),
                'focus_wrong_role':aggregate(lambda r,a=arm,x=d:focus(r,a,x,'wrong_role')),
                'role_margin':aggregate(lambda r,a=arm,x=d:focus(r,a,x,'role_margin')),
            }
    # Only the sequential, fixed predictions are inferentially designated.
    primary=aggregate(lambda r:focus(r,'p020','AP','native_correct')-
                               focus(r,'p000','AP','native_correct'))
    secondary=aggregate(lambda r:focus(r,'p100','AP','native_correct')-
                                 focus(r,'p000','AP','native_correct'))
    primary_pass=primary['ci95'][0]>0
    secondary_pass=primary_pass and secondary['ci95'][0]>0
    tests={'primary_p020_minus_p000_AP_accuracy':{'estimate':primary,'passes':primary_pass},
           'secondary_p100_minus_p000_AP_accuracy':{'estimate':secondary,
                          'passes':secondary_pass if primary_pass else None},
           'gated_tests_complete':secondary_pass}
    result={'study':'PHASE2-01','status':'NEW_SEED_MATCHED_COVERAGE_CURVE',
            'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'source_records':16,'model_states':len(ARMS)*len(SEEDS)*len(REPEATS),
            'arms':arms,'tests':tests,
            'dose_fraction_of_pairs':{a:n/STEPS for a,n in DOSES},
            'same_grammar_not_new_structural_family':True,'no_protected_or_LI':True,
            'unit':'8 independent training seeds with 2 numerical MPS repeats averaged within each seed'}
    write_once(OUT/'analysis_summary.json',result)
    print('PHASE2-01 SUMMARY PASS: 16/16 training runs, 80/80 model states, no protected/LI')
    for arm,n in DOSES:
        a=arms[arm]['AP'];m=a['focus_accuracy']
        print(f' {arm} ({100*n/STEPS:g}% direct dose pairs): AP second accuracy {100*m["mean"]:.2f}% '
              f'CI [{100*m["ci95"][0]:.2f}, {100*m["ci95"][1]:.2f}]')
    for k in ('primary_p020_minus_p000_AP_accuracy','secondary_p100_minus_p000_AP_accuracy'):
        o=tests[k];v=o['estimate']
        print(f'{k}: {100*v["mean"]:+.2f} pp 95% CI [{100*v["ci95"][0]:+.2f}, {100*v["ci95"][1]:+.2f}] pass={o["passes"]}')
    print('One controlled same-grammar exposure study; NOT yet an OOD grammar result')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--repeat',type=int,choices=REPEATS)
    parser.add_argument('--device',choices=('mps','cpu'),default='mps')
    a=parser.parse_args();device=torch.device(a.device)
    if a.phase in ('smoke','run') and a.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use the Mac main-repo virtualenv')
    if a.phase=='prepare':prepare()
    elif a.phase=='smoke':smoke(device)
    elif a.phase=='summary':summary()
    else:
        if a.seed is None or a.repeat is None:raise ValueError('run requires --seed and --repeat')
        run(a.seed,a.repeat,device)

if __name__=='__main__':main()
