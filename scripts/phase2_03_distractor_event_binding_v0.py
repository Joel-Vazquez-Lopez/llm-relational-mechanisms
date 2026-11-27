#!/usr/bin/env python3
"""PHASE2-03 — frozen prospective distractor-placement/event-binding test.

Fresh p000/p100 scratch models and fresh dev core orders, fixed vocabulary.
Compare familiar, passive-subject-infix, passive-tail placements of the SAME
'near OTHER-EVENT PERSON' token bigram. Balance which other-event role is named.
No reserved or within-event L+I evaluation.

    python3 scripts/phase2_03_distractor_event_binding_v0.py --phase prepare
    python3 scripts/phase2_03_distractor_event_binding_v0.py --phase smoke --device mps
    # Commit the exact four files (runner, launcher, protocol, manifest) together.
    bash scripts/run_phase2_03_all_v0.sh
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean, stdev

import torch

import exp105_generator_preflight as gen
import exp105_development_pilot_v0 as pilot
import exp105_training_schedule_preflight_v3 as sched
import exp109_replication_v0 as exp109
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114
import phase2_01_coverage_curve_v0 as phase201
import phase2_02_structural_transfer_v0 as phase202

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/phase2_03/distractor_event_binding_v0'
CKPT=ROOT/'results/phase2_03/distractor_event_binding_v0'
MANIFEST=OUT/'manifest.json'
PROTOCOL=ROOT/'paper/PHASE2_03_distractor_event_binding_v0_protocol.md'
LAUNCHER=ROOT/'scripts/run_phase2_03_all_v0.sh'
SEEDS=(11901,11903,11907,11909,11913,11919,11921,11927)
REPEATS=(1,2)
ARMS=('p000','p100')
DIRECTIONS=('AP','PA_literal','AA','PP')
REFERENTS=('first_agent','first_patient')
POSITIONS=('canonical','subject_infix','tail')
MODS=(('bare','I'),('L','I'))
PER_MOD=72
STEPS=6400
EVAL_SEED=1192026
TC95=2.364624251
assert not set(SEEDS)&(set(phase201.SEEDS)|set(phase202.SEEDS)|set(exp114.SEEDS)|set(exp109.SEEDS))
assert STEPS==phase202.STEPS==phase201.STEPS


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:
            raise FileExistsError('Existing immutable report differs: '+str(path))
        print('VERIFIED',path,flush=True)
        return
    with path.open('x',encoding='utf8') as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
    print('SAVED',path,flush=True)


def variants(events,mods,direction):
    """Four voice presentations; reposition only original second event when passive.

    For PA_literal that original event is physically first. No punctuation or
    lexical edits in the manipulated clause; all token bags and lengths equal.
    """
    assert direction in DIRECTIONS and mods[1]=='I' and mods[0] in ('bare','L')
    if direction=='PA_literal':
        ee=gen.voiced(events,'AP')[::-1];mm=mods[::-1];target=0
    else:
        ee=gen.voiced(events,direction);mm=mods;target=1
    base=[gen.render_event(e,m) for e,m in zip(ee,mm)]
    result={}
    for pos in POSITIONS:
        parts=list(base)
        e=ee[target]
        if pos!='canonical' and e.voice=='P':
            assert mm[target]=='I'
            if pos=='subject_infix':
                parts[target]=f'{e.patient} near {e.near_person} was {e.verb} by {e.agent}'
            else:
                parts[target]=f'{e.patient} was {e.verb} by {e.agent} near {e.near_person}'
        result[pos]=' ; '.join(parts)+'.'
    for pos in POSITIONS:
        assert Counter(gen.tokens(result[pos]))==Counter(gen.tokens(result['canonical']))
        assert len(gen.tokens(result[pos]))==len(gen.tokens(result['canonical']))
    if direction=='AA': assert len(set(result.values()))==1
    else:assert len(set(result.values()))==3
    return result


def evaluation(dev):
    """New non-reserved dev name permutations; no prior source/core reuse.

    Reconstruct previous *development* audit/exclusion lists only, not model
    outputs. The previously sealed Phase2-02 manifest is verified.
    """
    old_surfaces,old_orders,d109,d114=phase201.previous_dev_exclusions(dev)
    old114,_=exp114.novel_evaluation(dev)
    old117,a117=phase201.evaluation(dev)
    m117=json.loads(phase201.MANIFEST.read_text())
    assert a117['evaluation_sha256']==m117['evaluation_audit']['evaluation_sha256']
    old118,a118=phase202.evaluation(dev)
    m118=json.loads(phase202.MANIFEST.read_text())
    assert a118['evaluation_sha256']==m118['evaluation_audit']['evaluation_sha256']
    _train,dev_sets,partition=gen.partitions()
    idx={frozenset(names):i for i,names in enumerate(dev_sets)}
    for prior in (old114,old117):
        for g in prior:
            for src in g.values():old_surfaces.add(src.text)
            ix=idx[frozenset(g['AP'].gold)]
            old_orders[ix].add(tuple(g['AP'].gold))
            old_orders[ix].add(tuple(g['PA_literal'].gold))
    for g in old118:
        for pos in phase202.VERSIONS:
            for src in g['source'][pos].values():old_surfaces.add(src.text)
        ix=idx[frozenset(g['source']['canonical']['AP'].gold)]
        old_orders[ix].add(tuple(g['source']['canonical']['AP'].gold))
        old_orders[ix].add(tuple(g['source']['canonical']['PA_literal'].gold))
    rng=Random(EVAL_SEED)
    used_orders=defaultdict(set)
    seen=set(old_surfaces)
    groups=[]
    for mods in MODS:
        candidates=list(range(len(dev_sets)));rng.shuffle(candidates)
        selected=0
        for i in candidates:
            ps=[p for p in permutations(dev_sets[i],4)
                if p not in old_orders[i] and p not in used_orders[i]
                and (p[2],p[3],p[0],p[1]) not in old_orders[i]
                and (p[2],p[3],p[0],p[1]) not in used_orders[i]]
            rng.shuffle(ps)
            for p in ps:
                a,b,c,d=p
                v0=gen.VERBS[(i+7)%len(gen.VERBS)]
                v1=gen.VERBS[(i+4)%len(gen.VERBS)]
                assert v0!=v1
                staged={};surfaces=set()
                for ref in REFERENTS:
                    near=a if ref=='first_agent' else b
                    events=(gen.Event(a,b,v0,'A',c,d,i%2),
                            gen.Event(c,d,v1,'A',a,near,(i+1)%2))
                    core=f'newdev119-{i}-{selected}'
                    ap=gen.make_source(events,'AP',mods,'phase2_03_dev',core)
                    left,right=ap.text[:-1].split(' ; ')
                    base={
                      'AP':ap,
                      'PA_literal':replace(ap,text=right+' ; '+left+'.',
                            gold=ap.gold[2:]+ap.gold[:2],voice='PA',mods=mods[::-1]),
                      'AA':gen.make_source(events,'AA',mods,'phase2_03_dev',core),
                      'PP':gen.make_source(events,'PP',mods,'phase2_03_dev',core),
                    }
                    staged[ref]={}
                    for d0 in DIRECTIONS:
                        strings=variants(events,mods,d0)
                        assert strings['canonical']==base[d0].text
                        assert gen.parse_independent(strings['canonical'])==base[d0].gold
                        staged[ref][d0]={}
                        for pos,text in strings.items():
                            src=replace(base[d0],text=text)
                            assert 'LI' not in src.mods
                            assert len(set(src.gold))==4
                            if any(len(['<bos>']+gen.tokens(text)+['<sep>']+
                                      gen.tokens(q)+['<answer>'])>pilot.MAX_TOKENS
                                   for q in gen.Q):
                                raise AssertionError('Long evaluation sequence')
                            staged[ref][d0][pos]=src
                            surfaces.add(text)
                    # AA identical across locations, and 3 other directions vary.
                    assert len({s.text for f in staged[ref].values() for s in f.values()})==10
                if surfaces&seen:continue
                if len(surfaces)!=20:continue
                assert all(gen.parse_independent(staged[ref][d0]['canonical'].text)==
                           staged[ref][d0]['canonical'].gold for ref in REFERENTS for d0 in DIRECTIONS)
                # Each referent has one other-event answer name, selected from E1.
                assert all(staged[ref]['AP']['canonical'].gold[:2]==(a,b) for ref in REFERENTS)
                seen.update(surfaces)
                used_orders[i].add(p);used_orders[i].add((c,d,a,b))
                groups.append({'mods':mods,'near':{'first_agent':a,'first_patient':b},
                               'sources':staged})
                selected+=1
                break
            if selected==PER_MOD:break
        assert selected==PER_MOD,('insufficient unexamined dev name permutations',mods,selected)
    assert len(groups)==2*PER_MOD==144
    assert len(seen-old_surfaces)==144*20
    material=[(g['mods'],[(ref,d0,pos,g['sources'][ref][d0][pos].text,
                            g['sources'][ref][d0][pos].gold)
                      for ref in REFERENTS for d0 in DIRECTIONS for pos in POSITIONS])
              for g in groups]
    audit={'evaluation_sha256':digest(material),'anchor_count':144,
           'two_cross_event_referents_per_anchor':True,'positions':list(POSITIONS),
           'directions':list(DIRECTIONS),'referents':list(REFERENTS),
           'presentations':144*2*3*4,'distinct_texts':144*20,
           'all_within_anchor_token_bags_and_lengths_identical':True,
           'no_old_dev_exp109_exp114_phase2_01_phase2_02_surface_or_core_overlap':True,
           'no_protected_or_LI':True,'non_reserved_dev_partition':True,
           'prior_exp109_digest':d109,'prior_exp114_digest':d114,
           'prior_phase2_01_digest':a117['evaluation_sha256'],
           'prior_phase2_02_digest':a118['evaluation_sha256'],
           'partition_digest':partition['partition_digest']}
    return groups,audit


def encoded(groups,vocab):
    result={}
    for ref in REFERENTS:
        result[ref]={}
        for pos in POSITIONS:
            result[ref][pos]={}
            for d in DIRECTIONS:
                src=[g['sources'][ref][d][pos] for g in groups]
                result[ref][pos][d]=pilot.encode_sources(src,vocab,check=False)
    return result


def near_metric(logits,enc,groups,ref,d,vocab):
    """Classify normal candidate-restricted predictions on selected-event queries.

    No gold-conditioned intervention or model evaluation filtering. Scores include
    all 144 sources; the near name and opposite-event roles are fixed by source.
    """
    _,_,gold,candidates=enc
    pred=candidates.gather(1,logits.gather(1,candidates).argmax(1,keepdim=True)).flatten().view(-1,4)
    g=gold.view(-1,4)
    qi=(0,1) if d=='PA_literal' else (2,3)
    near=torch.tensor([vocab[x['near'][ref]] for x in groups],dtype=torch.long)
    other=torch.tensor([vocab[x['near']['first_patient' if ref=='first_agent' else 'first_agent']]
                       for x in groups],dtype=torch.long)
    assert len(near)==pred.shape[0]==len(groups)
    other_event = g[:,[2,3]] if d=='PA_literal' else g[:,[0,1]]
    for q in qi:
        assert (near!=g[:,q]).all() and (other!=g[:,q]).all()
    assert (near[:,None]==other_event).any(1).all()
    assert (other[:,None]==other_event).any(1).all()
    target=pred[:,qi]
    rate=lambda x:float(x.float().mean())
    near_hit=(target==near[:,None]);other_hit=(target==other[:,None])
    assert torch.all(~(near_hit&other_hit))
    m=exp112.metrics(logits,enc)
    we=mean(m['by_query'][q]['wrong_event'] for q in
            (('first_agent','first_patient') if d=='PA_literal' else ('second_agent','second_patient')))
    assert abs(rate(near_hit|other_hit)-we)<1e-5
    return {'near_hit':rate(near_hit),'other_event_other_hit':rate(other_hit),
            'wrong_event':we,
            'correct':mean(m['by_query'][q]['native_correct'] for q in
                (('first_agent','first_patient') if d=='PA_literal' else ('second_agent','second_patient'))),
            'wrong_role':mean(m['by_query'][q]['wrong_role'] for q in
                (('first_agent','first_patient') if d=='PA_literal' else ('second_agent','second_patient'))),
            'C_query':m['C_query'],'near_hit_minus_other_hit':rate(near_hit)-rate(other_hit)} ,pred[:,qi]


def score(model,data,groups,vocab,device):
    result={'by_referent':{},'referent_switch':{}}
    cache={}
    for ref in REFERENTS:
        result['by_referent'][ref]={};cache[ref]={}
        for pos in POSITIONS:
            result['by_referent'][ref][pos]={};cache[ref][pos]={}
            for d in DIRECTIONS:
                if d=='AA' and pos!='canonical':
                    result['by_referent'][ref][pos][d]=result['by_referent'][ref]['canonical'][d]
                    cache[ref][pos][d]=cache[ref]['canonical'][d]
                    assert torch.equal(data[ref][pos][d][0],data[ref]['canonical'][d][0])
                else:
                    logits=exp113.no_touch_identity(model,data[ref][pos][d],device)
                    v,pr=near_metric(logits,data[ref][pos][d],groups,ref,d,vocab)
                    result['by_referent'][ref][pos][d]=v
                    cache[ref][pos][d]=pr
    for pos in POSITIONS:
        result['referent_switch'][pos]={}
        for d in DIRECTIONS:
            a=cache['first_agent'][pos][d];b=cache['first_patient'][pos][d]
            aa=torch.tensor([vocab[x['near']['first_agent']] for x in groups])[:,None]
            bb=torch.tensor([vocab[x['near']['first_patient']] for x in groups])[:,None]
            # Questions are unchanged between the two paired near-referent variants.
            rate=lambda x:float(x.float().mean())
            result['referent_switch'][pos][d]={
                'near_follows_swap':rate((a==aa)&(b==bb)),
                'reverse_follows_swap':rate((a==bb)&(b==aa)),
                'any_prediction_change':rate(a!=b),
            }
    return result


def schedule(seed,common,dose,bp,dp,limit=STEPS):
    assert seed in SEEDS and 0<limit<=STEPS
    paired=[(d['PA'],d['AP'],0) for d in dose]
    count=0
    for half,initial in enumerate((seed,seed+100000)):
        rng=Random(initial+1060000);cells=[];block=-1
        for inner,base,_old,_bs,_ds in sched.iter_schedule(initial,common,paired,bp,dp):
            if (inner-1)//sched.BLOCK!=block:
                block=(inner-1)//sched.BLOCK;cells=list(sched.STRATA);rng.shuffle(cells)
            cell=cells[(inner-1)%sched.BLOCK]
            a=rng.choice(dp[(cell[0],cell[1],0)])
            b=rng.choice(dp[(cell[0],cell[1],1)])
            assert dose[a]['PA'].core_id!=dose[b]['PA'].core_id
            if rng.randrange(2):a,b=b,a
            count+=1
            yield count,base,(a,b),cell
            if count==limit:return
    assert count==limit


def schedule_audit(seed,common,dose,bp,dp):
    h=sha256();n=0
    for step,base,pair,cell in schedule(seed,common,dose,bp,dp):
        i,j=pair
        assert dose[i]['PP'].gold==dose[i]['PA'].gold
        assert dose[j]['AA'].gold==dose[j]['AP'].gold
        assert Counter(gen.tokens(dose[i]['PP'].text)+gen.tokens(dose[j]['AA'].text))==Counter(
               gen.tokens(dose[i]['PA'].text)+gen.tokens(dose[j]['AP'].text))
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
                              'stratum':[list(cell[0]),cell[1]]},separators=(',',':'))+'\n').encode())
        n+=1
    assert n==STEPS
    return {'seed':seed,'steps':STEPS,'schedule_sha256':h.hexdigest(),
            'dose_counts':{'p000':0,'p100':STEPS},
            'matched_per_step_token_bags':True,'same_gold_labels':True,
            'no_protected_LI':True}


def dependency_hashes():
    paths=(SCRIPT,Path(gen.__file__),Path(pilot.__file__),Path(sched.__file__),
           Path(exp109.__file__),Path(exp112.__file__),Path(exp113.__file__),
           Path(exp114.__file__),Path(phase201.__file__),Path(phase202.__file__))
    return {p.name:sha(p) for p in paths}


def study_protocol(vocab,dev,audit,partition):
    return {'study':'PHASE2-03','version':'DISTRACTOR_EVENT_BINDING_V0',
       'stage':'FROZEN_PROSPECTIVE_NOT_EXTERNAL_PREREGISTRATION',
       'seeds':list(SEEDS),'repeats':list(REPEATS),'arms':list(ARMS),
       'steps':STEPS,'device':'mps','evaluation_seed':EVAL_SEED,
       'eval_anchor_count':144,'cross_event_referents':list(REFERENTS),
       'modifier_cells':[list(x) for x in MODS],
       'positions':list(POSITIONS),'directions':list(DIRECTIONS),
       'focus':'two original second-event role questions per paired anchor',
       'primary':'p000 AP specificity: (tail-canonical near_hit) - (tail-canonical other_event_other_hit)',
       'secondary':'p000 AP specificity: (tail-subject_infix near_hit) - (tail-subject_infix other_event_other_hit); gated',
       'tertiary':'p100 AP specificity: (tail-canonical near_hit) - (tail-canonical other_event_other_hit); gated',
       'decision':'lower bound of n=8 paired two-sided Student-t 95% CI above zero; fixed order',
       'vocab_digest':digest(sorted(vocab.items())),
       'legacy_dev_digest':digest([(s.text,s.gold) for s in dev]),
       'partition_digest':partition,'evaluation_digest':audit['evaluation_sha256'],
       'training':'same matched PP+AA vs PA+AP source pairs, identical per-step token bags and labels',
       'no_checkpoint_selection':True,'no_exclusion_based_on_competence':True,
       'no_optional_stopping':True,'no_protected_LI':True,
       'dependency_hashes':dependency_hashes()}


def load_and_verify():
    if not MANIFEST.exists():raise FileNotFoundError('Run --phase prepare and commit first')
    common,dose,bp,dp,part,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    prot=study_protocol(vocab,dev,audit,part)
    m=json.loads(MANIFEST.read_text())
    assert m['protocol']==prot and m['protocol_digest']==digest(prot)
    assert m['evaluation_audit']==audit
    return m,(common,dose,bp,dp,dev,vocab),groups


def prepare():
    common,dose,bp,dp,part,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    for g in groups:
        for ref in REFERENTS:
            for d in DIRECTIONS:
                for pos in POSITIONS:
                    src=g['sources'][ref][d][pos]
                    assert set(gen.tokens(src.text)) <= set(vocab)
    p=study_protocol(vocab,dev,audit,part)
    print('PHASE2-03 PREPARE: checking 8 fresh schedules; NO training/scoring',flush=True)
    schedules={str(seed):schedule_audit(seed,common,dose,bp,dp) for seed in SEEDS}
    assert len({v['schedule_sha256'] for v in schedules.values()})==8
    write_once(MANIFEST,{'protocol':p,'protocol_digest':digest(p),
                         'evaluation_audit':audit,'schedule_audit':schedules})
    print('PHASE2-03 PREPARE PASS: 144 anchors x 2 cross-event referents x 3 near placements; 32 new models; protected/LI closed')
    print('Commit runner+launcher+protocol+manifest in one commit BEFORE training')


def freeze_verified():
    files=('scripts/phase2_03_distractor_event_binding_v0.py',
           'scripts/run_phase2_03_all_v0.sh',
           'paper/PHASE2_03_distractor_event_binding_v0_protocol.md',
           'reports/phase2_03/distractor_event_binding_v0/manifest.json')
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        c=subprocess.check_output(['git','log','-n','1','--format=%H','--',files[0]],cwd=ROOT,text=True).strip()
        if not c:raise RuntimeError('Runner never committed')
        for rel in files:
            known=subprocess.check_output(['git','show',c+':'+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
            if known!=(ROOT/rel).read_bytes():raise RuntimeError('Frozen content differs: '+rel)
        subprocess.run(['git','merge-base','--is-ancestor',c,head],cwd=ROOT,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return c
    except (FileNotFoundError,subprocess.CalledProcessError,RuntimeError) as e:
        raise RuntimeError('Commit PHASE2-03 runner+launcher+protocol+manifest together BEFORE first run') from e


def models(seed,device,n_vocab):
    torch.manual_seed(seed)
    initial=pilot.Model(n_vocab).to(device)
    result={}
    for arm in ARMS:
        m=deepcopy(initial).to(device)
        assert all(torch.equal(a,b) for a,b in zip(m.parameters(),initial.parameters()))
        opt=torch.optim.AdamW(m.parameters(),lr=pilot.LR,weight_decay=pilot.WEIGHT_DECAY)
        result[arm]=(m,opt)
    return result


def smoke(device):
    # Random weights + already explored legacy dev only. No new eval scored.
    common,dose,bp,dp,_part,dev,vocab=exp109.resources()
    nets=models(SEEDS[0],device,len(vocab))
    rows=list(schedule(SEEDS[0],common,dose,bp,dp,limit=2))
    assert len(rows)==2
    for _,base,pair,cell in rows:
        indirect,direct=phase201.sources_for(common,dose,base,pair)
        src={'p000':indirect,'p100':direct}
        for arm in ARMS:
            loss=pilot.one_update(*nets[arm],pilot.encode_sources(src[arm],vocab,check=False),device)
            assert math.isfinite(loss)
    legacy=exp109.evaluation(dev)[0][:2]
    enc=pilot.encode_sources([g['AP'] for g in legacy],vocab)
    output=exp113.no_touch_identity(nets['p000'][0].eval(),enc,device)
    m=exp112.metrics(output,enc)
    assert 0<=m['C_query']<=1
    # Check all new diagnostic transformations against old, already explored
    # development cores only. This is random-weight code validation, not scoring
    # the 144 fresh PHASE2-03 evaluation anchors.
    _tr,old_dev_sets,_p=gen.partitions()
    old_core=next(gen.cores(old_dev_sets,'dev'))[2]
    mock_groups=[]
    for gix in range(2):
        old_a,old_b=old_core
        near_a=old_a.agent;near_b=old_a.patient
        bundle={}
        for ref in REFERENTS:
            target_near=near_a if ref=='first_agent' else near_b
            events=(old_a,replace(old_b,near_person=target_near))
            bundle[ref]={}
            for dr in DIRECTIONS:
                texts=variants(events,('bare','I'),dr)
                sample=gen.make_source(events,'AP',('bare','I'),'legacy_smoke',f'old-{gix}')
                left,right=sample.text[:-1].split(' ; ')
                baseline={
                    'AP':sample,
                    'PA_literal':replace(sample,text=right+' ; '+left+'.',
                            gold=sample.gold[2:]+sample.gold[:2],voice='PA',mods=('I','bare')),
                    'AA':gen.make_source(events,'AA',('bare','I'),'legacy_smoke',f'old-{gix}'),
                    'PP':gen.make_source(events,'PP',('bare','I'),'legacy_smoke',f'old-{gix}')
                }[dr]
                bundle[ref][dr]={pos:replace(baseline,text=t) for pos,t in texts.items()}
        mock_groups.append({'mods':('bare','I'),
                            'near':{'first_agent':near_a,'first_patient':near_b},
                            'sources':bundle})
    fake_data=encoded(mock_groups,vocab)
    diagnostics=score(nets['p000'][0].eval(),fake_data,mock_groups,vocab,device)
    assert set(diagnostics['by_referent'])==set(REFERENTS)
    assert set(diagnostics['referent_switch'])==set(POSITIONS)
    print('PHASE2-03 SMOKE PASS: 2 arms / 2 updates, legacy + old-core diagnostic controls; NEW EVAL UNSEEN')


def run(seed,repeat,device):
    if device.type!='mps':raise ValueError('Real training requires Apple MPS; CPU smoke only')
    assert seed in SEEDS and repeat in REPEATS
    man,res,groups=load_and_verify()
    freeze_commit=freeze_verified()
    common,dose,bp,dp,dev,vocab=res
    report=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
    checkpoint=CKPT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
    if report.exists() or checkpoint.exists():
        raise FileExistsError('Refusing overwrite: '+str(report))
    data=encoded(groups,vocab)
    nets=models(seed,device,len(vocab))
    h=sha256();hist=[]
    for step,base,pair,cell in schedule(seed,common,dose,bp,dp):
        indirect,direct=phase201.sources_for(common,dose,base,pair)
        sources={'p000':indirect,'p100':direct}
        losses={arm:pilot.one_update(*nets[arm],
                    pilot.encode_sources(sources[arm],vocab,check=False),device)
                for arm in ARMS}
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
                 'stratum':[list(cell[0]),cell[1]]},separators=(',',':'))+'\n').encode())
        if step in exp109.MILESTONES:
            hist.append({'step':step,'losses':losses})
            print('STEP',step,'losses',
                  ' '.join(f'{a}:{losses[a]:.4f}' for a in ARMS),flush=True)
    assert h.hexdigest()==man['schedule_audit'][str(seed)]['schedule_sha256']
    outcomes={arm:score(nets[arm][0].eval(),data,groups,vocab,device) for arm in ARMS}
    rec={'study':'PHASE2-03','status':'SEALED_FRESH_SEED_DISTRACTOR_POSITION',
         'seed':seed,'repeat':repeat,'steps':STEPS,'device':'mps',
         'freeze_commit':freeze_commit,'script_sha256':sha(SCRIPT),
         'protocol_digest':man['protocol_digest'],
         'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
         'schedule_sha256':h.hexdigest(),'loss_history':hist,
         'outcomes':outcomes,'no_protected_or_LI':True}
    CKPT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({'study':'PHASE2-03','seed':seed,'repeat':repeat,'step':STEPS,
                'protocol_digest':man['protocol_digest'],
                'models':{a:nets[a][0].state_dict() for a in ARMS}},checkpoint)
    write_once(report,rec)
    print('SEALED; do not inspect per-run results until all 16 runs finished',flush=True)


def ci(vals):
    assert len(vals)==8
    m=mean(vals);radius=TC95*stdev(vals)/math.sqrt(8)
    return {'mean':m,'ci95':[m-radius,m+radius],
            'positive_seeds':sum(v>0 for v in vals),
            'negative_seeds':sum(v<0 for v in vals),
            'per_seed':dict(zip(map(str,SEEDS),vals))}


def summary():
    man,_res,_groups=load_and_verify()
    records={};missing=[]
    for seed in SEEDS:
        for repeat in REPEATS:
            p=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
            if not p.exists():missing.append(p.name);continue
            r=json.loads(p.read_text())
            assert (r['study'],r['seed'],r['repeat'],r['steps'])==('PHASE2-03',seed,repeat,STEPS)
            assert r['script_sha256']==sha(SCRIPT)
            assert r['protocol_digest']==man['protocol_digest']
            assert r['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
            assert r['schedule_sha256']==man['schedule_audit'][str(seed)]['schedule_sha256']
            assert r['no_protected_or_LI'] and set(r['outcomes'])==set(ARMS)
            for arm in ARMS:
                item=r['outcomes'][arm]
                assert set(item)=={'by_referent','referent_switch'}
                assert set(item['by_referent'])==set(REFERENTS)
                assert set(item['referent_switch'])==set(POSITIONS)
                for ref in REFERENTS:
                    assert set(item['by_referent'][ref])==set(POSITIONS)
                    for pos in POSITIONS:
                        assert set(item['by_referent'][ref][pos])==set(DIRECTIONS)
                        for d in DIRECTIONS:
                            m=item['by_referent'][ref][pos][d]
                            assert set(m)=={'near_hit','other_event_other_hit',
                              'wrong_event','correct','wrong_role','C_query',
                              'near_hit_minus_other_hit'}
                            assert abs(m['near_hit']+m['other_event_other_hit']-
                                       m['wrong_event'])<1e-5
                            assert abs(m['correct']+m['wrong_role']+
                                       m['wrong_event']-1)<1e-5
                            assert all(math.isfinite(v) and -1e-6<=v<=1.000001
                                       for k,v in m.items() if k!='near_hit_minus_other_hit')
                            assert -1.000001<=m['near_hit_minus_other_hit']<=1.000001
                            if d=='AA':
                                assert m==item['by_referent'][ref]['canonical'][d]
                for pos in POSITIONS:
                    assert set(item['referent_switch'][pos])==set(DIRECTIONS)
                    for d in DIRECTIONS:
                        assert set(item['referent_switch'][pos][d])=={
                           'near_follows_swap','reverse_follows_swap','any_prediction_change'}
            records[seed,repeat]=r
    if missing:
        print(f'PHASE2-03 INCOMPLETE: {len(missing)}/16 runs missing; no summary written')
        print('Missing:',', '.join(missing[:10]));return
    def across(fn):return ci([mean(fn(records[s,rep]) for rep in REPEATS) for s in SEEDS])
    def metric(rec,arm,pos,dir,key,ref):
        return rec['outcomes'][arm]['by_referent'][ref][pos][dir][key]
    def avg(rec,arm,pos,d,key):
        return mean(metric(rec,arm,pos,d,key,ref) for ref in REFERENTS)
    def selectivity(rec,arm,against):
        near=(avg(rec,arm,'tail','AP','near_hit')-avg(rec,arm,against,'AP','near_hit'))
        other=(avg(rec,arm,'tail','AP','other_event_other_hit')-
               avg(rec,arm,against,'AP','other_event_other_hit'))
        return near-other
    primary=across(lambda r:selectivity(r,'p000','canonical'))
    secondary=across(lambda r:selectivity(r,'p000','subject_infix'))
    tertiary=across(lambda r:selectivity(r,'p100','canonical'))
    passes_primary=primary['ci95'][0]>0
    passes_secondary=(secondary['ci95'][0]>0) if passes_primary else None
    passes_tertiary=(tertiary['ci95'][0]>0) if passes_secondary else None
    arms={}
    for arm in ARMS:
        arms[arm]={'by_referent':{},'reference_switch':{}}
        for ref in REFERENTS:
            arms[arm]['by_referent'][ref]={}
            for pos in POSITIONS:
                arms[arm]['by_referent'][ref][pos]={}
                for d in DIRECTIONS:
                    arms[arm]['by_referent'][ref][pos][d]={k:across(lambda r,a=arm,b=ref,p=pos,dr=d,key=k:
                         metric(r,a,p,dr,key,b)) for k in
                         ('near_hit','other_event_other_hit','wrong_event','correct',
                          'wrong_role','C_query','near_hit_minus_other_hit')}
        for pos in POSITIONS:
            arms[arm]['reference_switch'][pos]={}
            for d in DIRECTIONS:
                arms[arm]['reference_switch'][pos][d]={k:across(lambda r,a=arm,p=pos,dr=d,key=k:
                     r['outcomes'][a]['referent_switch'][p][dr][key]) for k in
                    ('near_follows_swap','reverse_follows_swap','any_prediction_change')}
    contrasts={}
    for arm in ARMS:
        for pos in ('canonical','subject_infix'):
            for k in ('near_hit','other_event_other_hit','wrong_event','correct','wrong_role'):
                contrasts[f'{arm}_tail_minus_{pos}_AP_{k}']=across(lambda r,a=arm,p=pos,key=k:
                         avg(r,a,'tail','AP',key)-avg(r,a,p,'AP',key))
        contrasts[f'{arm}_tail_minus_canonical_AP_near_selectivity']=across(lambda r,a=arm:
                   selectivity(r,a,'canonical'))
    report={'study':'PHASE2-03','status':'NEW_SEEDS_NEAR_DISTRACTOR_POSITION',
            'source_records':16,'model_states':32,'independent_seeds':8,
            'no_protected_or_LI':True,'same_grammar':True,
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'protocol_digest':man['protocol_digest'],
            'tests':{'primary_p000_tail_minus_canonical_AP_near_specificity':{'estimate':primary,'passes':passes_primary},
                     'secondary_p000_tail_minus_subject_infix_AP_near_specificity':{'estimate':secondary,'passes':passes_secondary},
                     'tertiary_p100_tail_minus_canonical_AP_near_specificity':{'estimate':tertiary,'passes':passes_tertiary}},
            'arms':arms,'descriptive_contrasts':contrasts,
            'limits':'same vocab and grammar; text and logits not saved; unadjusted descriptive intervals'}
    write_once(OUT/'analysis_summary.json',report)
    print('PHASE2-03 SUMMARY PASS: 16/16 runs, 32/32 models, 144 x 2 referent anchors, protected/LI closed')
    for arm in ARMS:
        for p in POSITIONS:
            near=mean(arms[arm]['by_referent'][ref][p]['AP']['near_hit']['mean'] for ref in REFERENTS)
            other=mean(arms[arm]['by_referent'][ref][p]['AP']['other_event_other_hit']['mean'] for ref in REFERENTS)
            print(f'{arm} {p}: AP other-event named distractor={100*near:.2f}% other participant={100*other:.2f}%')
    for name,v in report['tests'].items():
        x=v['estimate'];print(name,f'{100*x["mean"]:+.2f} pp 95% CI '
              f'[{100*x["ci95"][0]:+.2f},{100*x["ci95"][1]:+.2f}] pass={v["passes"]}')
    print('Targeted same-grammar positional/distractor attribution, not unique semantic event-binding circuit')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    p.add_argument('--device',default='mps',choices=('cpu','mps'))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--repeat',type=int,choices=REPEATS)
    args=p.parse_args();device=torch.device(args.device)
    if args.phase in ('smoke','run') and args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS is unavailable; use Mac training environment')
    if args.phase=='prepare':prepare()
    elif args.phase=='smoke':smoke(device)
    elif args.phase=='summary':summary()
    else:
        if args.seed is None or args.repeat is None:raise ValueError('--phase run requires --seed and --repeat')
        run(args.seed,args.repeat,device)

if __name__=='__main__':main()
