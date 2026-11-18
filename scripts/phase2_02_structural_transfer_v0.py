#!/usr/bin/env python3
"""PHASE2-02: new-seed structural-transfer test of matched direct AP exposure.

This is a new prospectively frozen study. It reuses the *training grammar* but
holds out two passive-adjunct word-order templates never used in training.
The novel and familiar versions have exactly identical token multisets/lengths
within each matched sentence, vocabulary and gold answers. No protected/LI.

    python3 scripts/phase2_02_structural_transfer_v0.py --phase prepare
    python3 scripts/phase2_02_structural_transfer_v0.py --phase smoke --device mps
    # Freeze script, protocol, launcher, manifest in one commit.
    bash scripts/run_phase2_02_all_v0.sh
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
import exp109_replication_v0 as exp109
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114
import phase2_01_coverage_curve_v0 as old

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/phase2_02/structural_transfer_v0'
CKPT=ROOT/'results/phase2_02/structural_transfer_v0'
MANIFEST=OUT/'manifest.json'
PROTOCOL=ROOT/'paper/PHASE2_02_structural_transfer_v0_protocol.md'
LAUNCHER=ROOT/'scripts/run_phase2_02_all_v0.sh'
SEEDS=(11801,11803,11807,11809,11813,11819,11821,11827)
REPEATS=(1,2)
STEPS=6400
# Each direct step has ONE AP source among 16 training sources. The percent
# refers to dose-pair steps, not percentage of all supervised examples.
DOSES=(('p000',0),('p020',1280),('p100',6400))
ARMS=tuple(x for x,_ in DOSES)
DIRECTIONS=('AP','PA_literal','AA','PP')
FAMILIES=('near_tail','gate_infix')
VERSIONS=('canonical','shifted')
EVAL_SEED=1182026
PER_CELL=24
TC95=2.364624251
assert not set(SEEDS)&(set(old.SEEDS)|set(exp114.SEEDS)|set(exp109.SEEDS))
assert STEPS==old.STEPS


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:
            raise FileExistsError('Existing file differs: '+str(path))
        print('VERIFIED',path,flush=True)
        return
    with path.open('x',encoding='utf8') as f:
        json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
    print('SAVED',path,flush=True)


def target_clause(e,mod,family):
    """Novel adjunct position on passive voice with same lexical tokens."""
    assert e.voice=='P'
    if family=='near_tail':
        assert mod=='I'
        # Familiar: P was VERBed near NAME by A
        return f'{e.patient} was {e.verb} by {e.agent} near {e.near_person}'
    if family=='gate_infix':
        assert mod=='L' and e.layout==1
        # Familiar: P was VERBed by A beside NAME's gate
        return f"{e.patient} beside {e.gate_owner}'s gate was {e.verb} by {e.agent}"
    raise ValueError(family)


def construct(events,mods,direction,family):
    """Build paired canonical and novel variants; only original event #2 may change."""
    assert direction in DIRECTIONS and family in FAMILIES
    if direction=='PA_literal':
        es=gen.voiced(events,'AP')[::-1]
        mm=mods[::-1]
        target=0
    else:
        es=gen.voiced(events,direction)
        mm=mods
        target=1
    familiar_parts=[gen.render_event(e,m) for e,m in zip(es,mm)]
    shifted_parts=list(familiar_parts)
    if es[target].voice=='P':
        shifted_parts[target]=target_clause(es[target],mm[target],family)
    familiar=' ; '.join(familiar_parts)+'.'
    shifted=' ; '.join(shifted_parts)+'.'
    # No added/deleted words, commas, separators, or new token IDs.
    assert Counter(gen.tokens(familiar))==Counter(gen.tokens(shifted))
    assert len(gen.tokens(familiar))==len(gen.tokens(shifted))
    if direction=='AA':assert shifted==familiar  # identity lexical control
    else: assert shifted!=familiar
    return familiar,shifted


def evaluation(dev):
    """New development core orderings, two previously untrained word orders.

    This function constructs examples only. It never scores models and never
    accesses the protected/reserved or within-event L+I partition.
    """
    # Reconstruct exact prior public development sets; no previous outcomes read.
    prior,old_orders,d109,d114=old.previous_dev_exclusions(dev)
    old114_groups,_=exp114.novel_evaluation(dev)
    old117_groups,prior117audit=old.evaluation(dev)
    legacy117=json.loads(old.MANIFEST.read_text())
    assert legacy117['evaluation_audit']['evaluation_sha256']==prior117audit['evaluation_sha256']
    _train,dev_sets,part=gen.partitions()
    index={frozenset(x):i for i,x in enumerate(dev_sets)}
    for gs in (old114_groups,old117_groups):
        for g in gs:
            for src in g.values():prior.add(src.text)
            i=index[frozenset(g['AP'].gold)]
            old_orders[i].add(tuple(g['AP'].gold))
            old_orders[i].add(tuple(g['PA_literal'].gold))
    rng=Random(EVAL_SEED)
    used_orders=defaultdict(set)
    seen=set(prior)
    groups=[]
    chosen_mods={
      'near_tail':(('bare','I'),('L','I'),('I','I')),
      'gate_infix':(('bare','L'),('L','L'),('I','L')),
    }
    counts=Counter()
    for family in FAMILIES:
        for mods in chosen_mods[family]:
            candidate_ids=list(range(len(dev_sets)))
            rng.shuffle(candidate_ids)
            selected=0
            for i in candidate_ids:
                ps=[p for p in permutations(dev_sets[i],4)
                    if p not in old_orders[i] and p not in used_orders[i]
                    and (p[2],p[3],p[0],p[1]) not in old_orders[i]
                    and (p[2],p[3],p[0],p[1]) not in used_orders[i]]
                rng.shuffle(ps)
                for p in ps:
                    a,b,c,d=p
                    v0=gen.VERBS[(i+5)%len(gen.VERBS)]
                    v1=gen.VERBS[(i+2)%len(gen.VERBS)]
                    assert v0!=v1
                    events=(gen.Event(a,b,v0,'A',c,d,i%2),
                            gen.Event(c,d,v1,'A',a,b,1 if family=='gate_infix' else (i+1)%2))
                    core=f'newdev118-{family}-{i}'
                    ap=gen.make_source(events,'AP',mods,'phase2_02_dev',core)
                    left,right=ap.text[:-1].split(' ; ')
                    base={
                      'AP':ap,
                      'PA_literal':replace(ap,text=right+' ; '+left+'.',
                            gold=ap.gold[2:]+ap.gold[:2], voice='PA',mods=mods[::-1]),
                      'AA':gen.make_source(events,'AA',mods,'phase2_02_dev',core),
                      'PP':gen.make_source(events,'PP',mods,'phase2_02_dev',core),
                    }
                    candidates={version:{} for version in VERSIONS}
                    for direction in DIRECTIONS:
                        familiar,shifted=construct(events,mods,direction,family)
                        assert familiar==base[direction].text
                        assert gen.parse_independent(familiar)==base[direction].gold
                        for version,text in (('canonical',familiar),('shifted',shifted)):
                            candidates[version][direction]=replace(base[direction],text=text)
                    # Distinguish in-grammar controls from OOD targets. AA is
                    # identical and should not be counted as a new surface.
                    surfaces={s.text for g in candidates.values() for s in g.values()}
                    if surfaces&seen:continue
                    if any('LI' in m for x in candidates.values() for s in x.values() for m in s.mods):
                        raise AssertionError('Protected L+I construction found')
                    if any(len(['<bos>']+gen.tokens(s.text)+['<sep>']+gen.tokens(q)+['<answer>'])>pilot.MAX_TOKENS
                           for x in candidates.values() for s in x.values() for q in gen.Q):
                        continue
                    # Independently verify the OOD strings reverse exactly to
                    # canonical, which is checked by the existing source parser.
                    assert len({s.text for s in candidates['canonical'].values()})==4
                    assert len({s.text for s in candidates['shifted'].values()})==4
                    seen.update(surfaces)
                    used_orders[i].add(p)
                    used_orders[i].add((c,d,a,b))
                    groups.append({'family':family,'mods':mods,'source':candidates})
                    counts[(family,mods)]+=1
                    selected+=1
                    break
                if selected==PER_CELL:break
            assert selected==PER_CELL,(family,mods,selected)
    assert len(groups)==2*3*PER_CELL==144
    assert all(sum(1 for g in groups if g['family']==f)==72 for f in FAMILIES)
    assert len(seen-prior)==144*7  # 8 presentations minus identical AA control
    # Deterministic digest includes all texts, gold labels, classes and mods.
    values=[(g['family'],list(g['mods']),[(v,d,g['source'][v][d].text,
                                list(g['source'][v][d].gold))
                 for v in VERSIONS for d in DIRECTIONS]) for g in groups]
    audit={
        'evaluation_sha256':digest(values),'anchor_count':144,
        'canonical_and_novel_presentations':144*8,
        'distinct_surface_presentations':144*7,
        'per_family':{f:72 for f in FAMILIES},
        'per_family_mod_cell':{str(k):v for k,v in counts.items()},
        'all_shifted_pairs_same_token_multiset':True,
        'all_shifted_pairs_same_token_count':True,
        'all_AA_shifted_identity':True,
        'no_old_dev_exp109_exp114_phase2_01_surface_overlap':True,
        'no_old_dev_exp109_exp114_phase2_01_core_order_overlap':True,
        'no_within_event_LI_or_protected':True,
        'prior_exp109_digest':d109,'prior_exp114_digest':d114,
        'prior_phase2_01_digest':prior117audit['evaluation_sha256'],
        'partition_digest':part['partition_digest'],
        'novelty_scope':'two novel passive-adjunct placements within existing grammar/vocab',
    }
    return groups,audit


def score(model,encoded,device):
    result={}
    for family in FAMILIES:
        result[family]={}
        for version in VERSIONS:
            result[family][version]={}
            for direction in DIRECTIONS:
                if version=='shifted' and direction=='AA':
                    # The AA source strings and questions are bit-identical.
                    # Reuse the native metric to avoid a redundant MPS forward.
                    a=encoded[family]['canonical']['AA']
                    b=encoded[family]['shifted']['AA']
                    assert len(a)==len(b)
                    assert all(torch.equal(x,y) if torch.is_tensor(x) else x==y
                               for x,y in zip(a,b))
                    result[family][version][direction]=result[family]['canonical']['AA']
                    continue
                enc=encoded[family][version][direction]
                pred=exp113.no_touch_identity(model,enc,device)
                result[family][version][direction]=exp112.metrics(pred,enc)
    return result


def encode(groups,vocab):
    data={}
    for family in FAMILIES:
        sub=[g for g in groups if g['family']==family]
        data[family]={v:{d:pilot.encode_sources([g['source'][v][d] for g in sub],vocab,check=False)
                               for d in DIRECTIONS} for v in VERSIONS}
    return data


def dependency_hashes():
    # Study is anchored to previous frozen training & generator code, but
    # does not alter earlier experiments or use their result estimates.
    res=old.dependency_hashes()
    res.pop('phase2_02_structural_transfer_v0.py',None)
    res['phase2_01_coverage_curve_v0.py']=sha(Path(old.__file__))
    res['phase2_02_structural_transfer_v0.py']=sha(SCRIPT)
    return res


def study_protocol(vocab,dev,audit,partition):
    return {'study':'PHASE2-02','version':'STRUCTURAL_TRANSFER_V0',
        'stage':'PROSPECTIVELY_FROZEN_NOT_EXTERNAL_PREREGISTRATION',
        'seeds':list(SEEDS),'repeats':list(REPEATS),'arms':list(ARMS),
        'dose_counts':dict(DOSES),'steps':STEPS,'device':'mps',
        'architecture':'exact phase2_01 / exp109 model',
        'vocab_digest':digest(sorted(vocab.items())),
        'legacy_dev_digest':digest([(s.text,s.gold) for s in dev]),
        'partition_digest':partition,'evaluation_seed':EVAL_SEED,
        'evaluation_digest':audit['evaluation_sha256'],
        'evaluation_families':list(FAMILIES),'evaluation_versions':list(VERSIONS),
        'eval_per_family':72,'directions':list(DIRECTIONS),
        'primary':'p100 minus p000 shifted AP second-query average correctness (two families 50/50)',
        'secondary':'p100 minus p000 canonical AP second-query average correctness on same anchors; gated',
        'criterion':'paired n=8 seed-level two-sided 95% Student t CI lower >0; ordered gate',
        'descriptive':['p020 and family-specific outcomes','OOD-canonical within-model gap',
                       'wrong_role and wrong_event','C_query', 'PA_literal/AA/PP controls'],
        'not_claiming_novel_semantic_grammar':True,
        'token_identical_canonical_shifted_pairs':True,
        'no_protected_reserved_LI':True,
        'no_posterior_competence_filter':True,'no_optional_stopping':True,
        'no_checkpoint_selection':True,
        'dependency_hashes':dependency_hashes()}


def load_and_verify():
    if not MANIFEST.exists():raise FileNotFoundError('Run --phase prepare first')
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    p=study_protocol(vocab,dev,audit,partition)
    m=json.loads(MANIFEST.read_text())
    assert m['protocol']==p and m['protocol_digest']==digest(p)
    assert m['evaluation_audit']==audit
    return m,(common,dose,bp,dp,dev,vocab),groups


def prepare():
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=evaluation(dev)
    assert len(groups)==144
    for g in groups:
        for version in VERSIONS:
            for src in g['source'][version].values():
                assert set(gen.tokens(src.text)).issubset(vocab)
    protocol=study_protocol(vocab,dev,audit,partition)
    print('PHASE2-02: auditing eight matched three-arm schedules; NO model scoring',flush=True)
    schedules={}
    for seed in SEEDS:
        schedules[str(seed)]=schedule_audit(seed,common,dose,bp,dp)
    assert len(set(x['schedule_sha256'] for x in schedules.values()))==len(SEEDS)
    write_once(MANIFEST,{'protocol':protocol,'protocol_digest':digest(protocol),
                         'evaluation_audit':audit,'schedule_audit':schedules})
    print('PHASE2-02 PREPARE PASS: 144 paired new-word-order dev anchors; 48 new model states; no protected/LI',flush=True)
    print('COMMIT runner + launcher + protocol + manifest together BEFORE run',flush=True)


def training_schedule(seed,common,dose,bp,dp,limit=STEPS):
    assert seed in SEEDS and 0<limit<=STEPS
    # Previous algorithm's seed assertion is intentionally *not* reused;
    # same fixed construction, fresh independent training seeds.
    # Re-implement exact sampling from parent with new seeds.
    paired=[(d['PA'],d['AP'],0) for d in dose]
    rows=[]
    import exp105_training_schedule_preflight_v3 as sched
    for half,s in enumerate((seed,seed+100000)):
        rng=Random(s+1060000);cells=[];block=-1
        for inner,base,_old,_bs,_ds in sched.iter_schedule(s,common,paired,bp,dp):
            if (inner-1)//sched.BLOCK!=block:
                block=(inner-1)//sched.BLOCK
                cells=list(sched.STRATA);rng.shuffle(cells)
            mod,fi=cells[(inner-1)%sched.BLOCK]
            a=rng.choice(dp[(mod,fi,0)]);b=rng.choice(dp[(mod,fi,1)])
            assert dose[a]['PA'].core_id!=dose[b]['PA'].core_id
            if rng.randrange(2):a,b=b,a
            rows.append((half*sched.STEPS+inner,base,(a,b),(mod,fi)))
    assert len(rows)==STEPS
    # Exactly 20% of 6400 dose pairs in each seed, stratified over 45 cells.
    strata=defaultdict(list)
    for step,_base,_pair,cell in rows:strata[cell].append(step)
    assert len(strata)==45
    randomizer=Random(seed+1180227)
    cells=sorted(strata);randomizer.shuffle(cells)
    for cell in cells:randomizer.shuffle(strata[cell])
    q,r=divmod(1280,len(cells))
    chosen={st for k,cell in enumerate(cells) for st in strata[cell][:q+int(k<r)]}
    assert len(chosen)==1280
    for step,base,pair,cell in rows[:limit]:
        direct={'p000':False,'p020':step in chosen,'p100':True}
        yield step,base,pair,cell,direct


def schedule_audit(seed,common,dose,bp,dp):
    h=sha256();counts=Counter();hist=Counter();strata=Counter()
    for step,base,pair,cell,direct in training_schedule(seed,common,dose,bp,dp):
        i,j=pair
        assert dose[i]['PP'].gold==dose[i]['PA'].gold and dose[j]['AA'].gold==dose[j]['AP'].gold
        assert Counter(gen.tokens(dose[i]['PP'].text)+gen.tokens(dose[j]['AA'].text))==Counter(
                gen.tokens(dose[i]['PA'].text)+gen.tokens(dose[j]['AP'].text))
        assert dose[i]['PA'].core_id!=dose[j]['AP'].core_id
        strata[cell]+=1
        for arm,flag in direct.items():
            if flag:counts[arm]+=1
        if direct['p020']:hist[cell]+=1
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
               'stratum':[list(cell[0]),cell[1]],'direct':direct},separators=(',',':'))+'\n').encode())
    assert counts['p000']==0 and counts['p020']==1280 and counts['p100']==STEPS
    assert set(strata)==set(old.sched.STRATA)
    assert max(hist.values())-min(hist.values())<=1
    return {'seed':seed,'schedule_sha256':h.hexdigest(),
            'dose_counts':dict(DOSES),'steps':STEPS,'matched_token_bags_every_step':True,
            'matched_passive_position_marginals':True,'no_protected_LI':True}


def freeze_verified():
    frozen=('scripts/phase2_02_structural_transfer_v0.py',
            'scripts/run_phase2_02_all_v0.sh',
            'paper/PHASE2_02_structural_transfer_v0_protocol.md',
            'reports/phase2_02/structural_transfer_v0/manifest.json')
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        c=subprocess.check_output(['git','log','-n','1','--format=%H','--',frozen[0]],cwd=ROOT,text=True).strip()
        if not c:raise RuntimeError('Runner never committed')
        for rel in frozen:
            committed=subprocess.check_output(['git','show',c+':'+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
            if committed!=(ROOT/rel).read_bytes():raise RuntimeError('Frozen content differs: '+rel)
        subprocess.run(['git','merge-base','--is-ancestor',c,head],cwd=ROOT,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return c
    except (FileNotFoundError,subprocess.CalledProcessError,RuntimeError) as e:
        raise RuntimeError('Commit all four PHASE2-02 frozen files together BEFORE training') from e


def models(seed,device,nvocab):
    torch.manual_seed(seed)
    shared=pilot.Model(nvocab).to(device)
    result={}
    for arm in ARMS:
        m=deepcopy(shared).to(device)
        assert all(torch.equal(a,b) for a,b in zip(m.parameters(),shared.parameters()))
        opt=torch.optim.AdamW(m.parameters(),lr=pilot.LR,weight_decay=pilot.WEIGHT_DECAY)
        result[arm]=(m,opt)
    return result


def smoke(device):
    # This uses only OLD dev scoring (never looks at new PHASE2-02 outcomes).
    common,dose,bp,dp,_part,dev,vocab=exp109.resources()
    net=models(SEEDS[0],device,len(vocab))
    rows=list(training_schedule(SEEDS[0],common,dose,bp,dp,limit=2))
    assert [r[0] for r in rows]==[1,2]
    for step,base,pair,cell,direct in rows:
        indirect,direct_sources=old.sources_for(common,dose,base,pair)
        ie=pilot.encode_sources(indirect,vocab,check=False)
        de=pilot.encode_sources(direct_sources,vocab,check=False)
        for arm in ARMS:
            loss=pilot.one_update(*net[arm],de if direct[arm] else ie,device)
            assert math.isfinite(loss)
    legacy=exp109.evaluation(dev)[0][:2]
    enc=pilot.encode_sources([g['AP'] for g in legacy],vocab)
    result=exp112.metrics(exp113.no_touch_identity(net['p000'][0].eval(),enc,device),enc)
    assert 0<=result['C_query']<=1
    print('PHASE2-02 SMOKE PASS: 3 arms, 2 updates, old dev scoring; NEW OOD EVAL NOT SCORED')


def run(seed,repeat,device):
    if device.type!='mps':raise ValueError('MPS required for frozen training; CPU for smoke only')
    assert seed in SEEDS and repeat in REPEATS
    manifest,res,groups=load_and_verify()
    commit=freeze_verified()
    common,dose,bp,dp,_dev,vocab=res
    report_path=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
    cp_path=CKPT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
    if report_path.exists() or cp_path.exists():raise FileExistsError('Refuse overwrite '+str(report_path))
    encoded=encode(groups,vocab)
    nets=models(seed,device,len(vocab))
    h=sha256();hist=[]
    for step,base,pair,cell,direct in training_schedule(seed,common,dose,bp,dp):
        indirect,direct_sources=old.sources_for(common,dose,base,pair)
        ie=pilot.encode_sources(indirect,vocab,check=False)
        de=pilot.encode_sources(direct_sources,vocab,check=False)
        losses={arm:pilot.one_update(*nets[arm],de if direct[arm] else ie,device) for arm in ARMS}
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
                  'stratum':[list(cell[0]),cell[1]],'direct':direct},separators=(',',':'))+'\n').encode())
        if step in exp109.MILESTONES:
            hist.append({'step':step,'losses':losses})
            print('STEP',step,'losses',' '.join(f'{arm}:{losses[arm]:.4f}' for arm in ARMS),flush=True)
    assert h.hexdigest()==manifest['schedule_audit'][str(seed)]['schedule_sha256']
    outcomes={arm:score(nets[arm][0].eval(),encoded,device) for arm in ARMS}
    record={'study':'PHASE2-02','status':'SEALED_NEW_SEED_STRUCTURAL_TRANSFER',
            'seed':seed,'repeat':repeat,'steps':STEPS,'device':'mps',
            'freeze_commit':commit,'script_sha256':sha(SCRIPT),
            'protocol_digest':manifest['protocol_digest'],
            'evaluation_sha256':manifest['evaluation_audit']['evaluation_sha256'],
            'schedule_sha256':h.hexdigest(),
            'loss_history':hist,'outcomes':outcomes,
            'no_protected_or_LI':True,
            'structural_ood_limited_to_passive_adjunctor_word_order':True}
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
    torch.save({'study':'PHASE2-02','seed':seed,'repeat':repeat,'step':STEPS,
                'protocol_digest':manifest['protocol_digest'],
                'models':{a:nets[a][0].state_dict() for a in ARMS}},cp_path)
    with report_path.open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print('SAVED',report_path,flush=True);print('SAVED',cp_path,flush=True)
    print('SEALED; do not inspect individual scores until 16/16 complete',flush=True)


def ci(v):
    assert len(v)==len(SEEDS)==8
    m=mean(v);r=TC95*stdev(v)/math.sqrt(8)
    return {'mean':m,'ci95':[m-r,m+r],
            'positive_seeds':sum(x>0 for x in v),
            'negative_seeds':sum(x<0 for x in v),
            'per_seed':dict(zip(map(str,SEEDS),v))}


def focus(record,arm,family,version,dir,metric):
    target=('first_agent','first_patient') if dir=='PA_literal' else ('second_agent','second_patient')
    m=record['outcomes'][arm][family][version][dir]
    return mean(m['by_query'][q][metric] for q in target)


def summary():
    manifest,res,groups=load_and_verify()
    records={};missing=[]
    for seed in SEEDS:
        for rep in REPEATS:
            f=OUT/f'seed{seed}_repeat{rep}_mps_step{STEPS}.json'
            if not f.is_file():missing.append(f.name);continue
            r=json.loads(f.read_text())
            assert (r['study'],r['seed'],r['repeat'],r['steps'])==('PHASE2-02',seed,rep,STEPS)
            assert r['script_sha256']==sha(SCRIPT)
            assert r['protocol_digest']==manifest['protocol_digest']
            assert r['evaluation_sha256']==manifest['evaluation_audit']['evaluation_sha256']
            assert r['schedule_sha256']==manifest['schedule_audit'][str(seed)]['schedule_sha256']
            assert r['no_protected_or_LI']
            assert set(r['outcomes'])==set(ARMS)
            for arm in ARMS:
                assert set(r['outcomes'][arm])==set(FAMILIES)
                for fam in FAMILIES:
                    assert set(r['outcomes'][arm][fam])==set(VERSIONS)
                    for version in VERSIONS:
                        assert set(r['outcomes'][arm][fam][version])==set(DIRECTIONS)
                        # AA control receives no source modification in either version.
                    # Equal logits should produce exactly equal metrics for AA.
                    assert r['outcomes'][arm][fam]['canonical']['AA']==r['outcomes'][arm][fam]['shifted']['AA']
            records[seed,rep]=r
    if missing:
        print('INCOMPLETE PHASE2-02: '+str(len(missing))+' of 16 reports missing')
        print('Missing: '+', '.join(missing[:12]));return
    def across(fn):return ci([mean(fn(records[seed,rep]) for rep in REPEATS) for seed in SEEDS])
    def avg_focus(record,arm,version,metric,dir='AP'):
        return mean(focus(record,arm,fam,version,dir,metric) for fam in FAMILIES)
    arms={}
    for arm in ARMS:
        arms[arm]={}
        for fam in FAMILIES:
            arms[arm][fam]={}
            for version in VERSIONS:
                arms[arm][fam][version]={}
                for d in DIRECTIONS:
                    arms[arm][fam][version][d]={
                        'C_query':across(lambda r,a=arm,f=fam,v=version,q=d:
                                          r['outcomes'][a][f][v][q]['C_query']),
                        'focus_correct':across(lambda r,a=arm,f=fam,v=version,q=d:
                                          focus(r,a,f,v,q,'native_correct')),
                        'focus_wrong_role':across(lambda r,a=arm,f=fam,v=version,q=d:
                                          focus(r,a,f,v,q,'wrong_role')),
                        'focus_wrong_event':across(lambda r,a=arm,f=fam,v=version,q=d:
                                          focus(r,a,f,v,q,'wrong_event')),
                    }
    def diff(x,y,version,metric,dir='AP'):
        return across(lambda rec:avg_focus(rec,x,version,metric,dir)-avg_focus(rec,y,version,metric,dir))
    primary=diff('p100','p000','shifted','native_correct')
    secondary=diff('p100','p000','canonical','native_correct')
    yes=primary['ci95'][0]>0
    tests={'primary_p100_minus_p000_SHIFTED_AP_accuracy':{'estimate':primary,'passes':yes},
           'secondary_p100_minus_p000_CANONICAL_AP_accuracy':{'estimate':secondary,
                              'passes':secondary['ci95'][0]>0 if yes else None},
           'fixed_order_gate_completed':yes and secondary['ci95'][0]>0}
    contrast={
        'p100_minus_p000_shifted_wrong_role':diff('p100','p000','shifted','wrong_role'),
        'p100_minus_p000_shifted_wrong_event':diff('p100','p000','shifted','wrong_event'),
        'p020_minus_p000_shifted_accuracy':diff('p020','p000','shifted','native_correct'),
    }
    for arm in ARMS:
        contrast[f'{arm}_shifted_minus_canonical_AP_accuracy']=across(lambda rec,a=arm:
             avg_focus(rec,a,'shifted','native_correct')-avg_focus(rec,a,'canonical','native_correct'))
    output={'study':'PHASE2-02','status':'INDEPENDENT_SEEDS_NEW_PASSIVE_WORD_ORDER',
            'source_records':16,'model_states':len(SEEDS)*len(REPEATS)*len(ARMS),
            'independent_seeds':len(SEEDS),
            'no_protected_or_LI':True,
            'evaluation_sha256':manifest['evaluation_audit']['evaluation_sha256'],
            'protocol_digest':manifest['protocol_digest'],
            'arms':arms,'tests':tests,'descriptive_contrasts':contrast,
            'scope':'new passive adjunct positions only, fixed vocabulary and two-event grammar'}
    write_once(OUT/'analysis_summary.json',output)
    print('PHASE2-02 SUMMARY PASS: 16/16 training runs, 48/48 models, 144 paired canonical/shifted anchors, no protected/LI')
    for arm in ARMS:
        vals=[mean(arms[arm][fam][version]['AP']['focus_correct']['mean'] for fam in FAMILIES)
                   for version in VERSIONS]
        print(f'{arm}: canonical AP accuracy {100*vals[0]:.2f}%, shifted AP accuracy {100*vals[1]:.2f}%')
    for name in ('primary_p100_minus_p000_SHIFTED_AP_accuracy','secondary_p100_minus_p000_CANONICAL_AP_accuracy'):
        item=tests[name];v=item['estimate']
        print(f'{name}: {100*v["mean"]:+.2f} pp 95% CI [{100*v["ci95"][0]:+.2f}, {100*v["ci95"][1]:+.2f}], pass={item["passes"]}')
    print('Novel passive adjunct word order only; NOT new vocabulary or fully unseen grammatical relations.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    ap.add_argument('--device',choices=('cpu','mps'),default='mps')
    ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--repeat',type=int,choices=REPEATS)
    args=ap.parse_args()
    device=torch.device(args.device)
    if args.phase in ('run','smoke') and args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this environment; use the Mac .venv')
    if args.phase=='prepare':prepare()
    elif args.phase=='smoke':smoke(device)
    elif args.phase=='summary':summary()
    else:
        if args.seed is None or args.repeat is None:raise ValueError('RUN requires --seed --repeat')
        run(args.seed,args.repeat,device)


if __name__=='__main__':main()
