#!/usr/bin/env python3
"""EXP-109 prospective within-grammar replication; protocol lock before any test scoring.

IMPORTANT: the protected / reserved L+I evaluation is never generated or opened.
The independent evaluation samples new cores from the existing NON-RESERVED dev
name-set partition; it is independent of earlier development source *sentences*,
not of the grammar family, individual names, or historical scientific decisions.

Run from exp102-clean-design:
  python3 scripts/exp109_replication_v0.py --phase prepare
  python3 scripts/exp109_replication_v0.py --phase smoke --device mps
  # REVIEW and COMMIT scripts/protocol/manifest before executing --phase run.
  python3 -u scripts/exp109_replication_v0.py --phase run --seed 10901 --repeat 1 --device mps
  # ... complete all 16 prespecified runs; in separate processes
  python3 scripts/exp109_replication_v0.py --phase summary
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
import json
import os
import subprocess

import torch

import exp105_generator_preflight as gen
import exp105_position_control_preflight_v1 as frames
import exp105_training_schedule_preflight_v3 as sched
import exp105_development_pilot_v0 as pilot
import exp106_voice_recombination_pilot_v0 as exp106
import exp107_counterfactual_diagnostic_v0 as cf

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp109/replication_v0'
CKPT=ROOT/'results/exp109/replication_v0'
SEEDS=(10901,10903,10907,10909,10913,10919,10921,10927)
REPEATS=(1,2)
STEPS=6400
MILESTONES=(800,1600,2400,3200,4000,4800,5600,6400)
EVAL_SEED=1092026
EVAL_PER_MOD=16
CONDITIONS=('AP','PA_literal','AA','PP','AP_rename','AP_reverse_second')
ARMS=exp106.ARMS
assert STEPS==2*sched.STEPS


def digest(x):
    return sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_once(p,x):
    if p.exists():
        if json.loads(p.read_text())!=x:raise RuntimeError('Frozen record differs: '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
    print('SAVED',p)


def sha_file(p):return sha256(Path(p).read_bytes()).hexdigest()


def hashes():
    inputs=(SCRIPT,Path(gen.__file__),Path(frames.__file__),Path(sched.__file__),
            Path(pilot.__file__),Path(exp106.__file__),Path(cf.__file__))
    return {p.name:sha_file(p) for p in inputs}


def resources():
    # Unlike exp106.setup(), does not require previous EXP105 report files. It
    # still reuses the exact training pool, labels, model and vocabulary builder.
    common,dose,bp,dp,partition=exp106.pools()
    dev=exp106.dev_sources()  # reference ONLY for vocabulary and overlap audit
    vocab=pilot.vocab_for(common,[(d['PA'],d['AP'],0) for d in dose],dev)
    return common,dose,bp,dp,partition,dev,vocab


def evaluation(dev):
    """New dev-partition permutations; no reserved four-name identifiers used.

    No model is scored here, and no example surfaces are printed or persisted.
    Reconstructs identical instances after protocol freeze.
    """
    _train,dev_sets,_part=gen.partitions()
    old_perms={}
    for _j,cid,events in gen.cores(dev_sets,'dev'):
        i=int(cid.split('-')[1]);old_perms.setdefault(i,set()).add(
            (events[0].agent,events[0].patient,events[1].agent,events[1].patient))
    old_surfaces={s.text for s in dev}
    rng=Random(EVAL_SEED)
    cells={m:[] for m in gen.TRAIN_MODS}
    for i,nameset in enumerate(dev_sets):
        options=[p for p in permutations(nameset,4) if p not in old_perms[i]]
        rng.shuffle(options)
        accepted=0
        for p in options:
            a,b,c,d=p
            # A cyclic rename must also avoid old core permutations.
            if (b,c,d,a) in old_perms[i]:continue
            v0=gen.VERBS[(i+accepted+1)%len(gen.VERBS)]
            v1=gen.VERBS[(i+accepted+2)%len(gen.VERBS)]
            assert v0!=v1
            events=(gen.Event(a,b,v0,'A',c,d,(i+accepted)%2),
                    gen.Event(c,d,v1,'A',a,b,(i+accepted+1)%2))
            for mod in gen.TRAIN_MODS:
                cid=f'newdev-{i}-{accepted}'
                base=gen.make_source(events,'AP',mod,'exp109_newdev',cid)
                if base.text in old_surfaces:break
                cells[mod].append((base,events))
            else:
                accepted+=1
            if accepted==2:break
        assert accepted==2,(i,accepted)
    anchors=[]
    for mod in sorted(cells):
        sample=rng.sample(cells[mod],EVAL_PER_MOD)
        anchors.extend(sample)
    assert len(anchors)==len(gen.TRAIN_MODS)*EVAL_PER_MOD==144
    groups=[]
    all_surfaces={}
    for base,events in anchors:
        left,right=base.text[:-1].split(' ; ')
        literal=replace(base,text=right+' ; '+left+'.',
                        gold=base.gold[2:]+base.gold[:2],voice='PA',mods=base.mods[::-1])
        aa=gen.make_source(events,'AA',base.mods,base.group,base.core_id)
        pp=gen.make_source(events,'PP',base.mods,base.group,base.core_id)
        a,b,c,d=base.gold
        remap=dict(zip((a,b,c,d),(b,c,d,a)))
        renamed_events=tuple(replace(e,agent=remap[e.agent],patient=remap[e.patient],
                    gate_owner=remap[e.gate_owner],near_person=remap[e.near_person]) for e in events)
        rename=gen.make_source(renamed_events,'AP',base.mods,base.group,base.core_id)
        e0,e1=events
        rev=gen.make_source((e0,replace(e1,agent=e1.patient,patient=e1.agent)),
                            'AP',base.mods,base.group,base.core_id)
        group=dict(zip(CONDITIONS,(base,literal,aa,pp,rename,rev)))
        assert len(group)==len(CONDITIONS)
        for key,s in group.items():
            assert all(m != 'LI' for m in s.mods)
            assert gen.parse_independent(frames.unframe(s.text))==s.gold
            assert len(set(s.gold))==4
            assert len(['<bos>']+gen.tokens(s.text)+['<sep>']+gen.tokens(gen.Q[0])+['<answer>'])<=pilot.MAX_TOKENS
            assert set(s.gold)==set(base.gold)
            assert s.text not in old_surfaces,(key,'overlaps explored dev set')
            if s.text in all_surfaces:assert all_surfaces[s.text]==s.gold
            all_surfaces[s.text]=s.gold
        assert Counter(gen.tokens(base.text))==Counter(gen.tokens(literal.text))
        assert literal.gold==base.gold[2:]+base.gold[:2]
        assert (aa.gold,pp.gold)==(base.gold,base.gold)
        assert rename.gold==(b,c,d,a)
        assert rev.gold==(a,b,d,c)
        groups.append(group)
    assert len({g['AP'].text for g in groups})==len(groups)
    counts=Counter(g['AP'].mods for g in groups)
    assert set(counts.values())=={EVAL_PER_MOD}
    return groups,{'anchors':len(groups),'presentations':len(groups)*len(CONDITIONS),
                   'per_modifier':{str(k):v for k,v in sorted(counts.items())},
                   'distinct_presentations':len(all_surfaces),
                   'new_source_core_permutations':True,'no_LI':True,
                   'no_reserved_or_protected':True,
                   'no_overlap_with_previous_288_dev_surfaces':True,
                   'evaluation_sha256':digest([[(key,g[key].text,g[key].gold)
                             for key in CONDITIONS] for g in groups])}


def schedule(seed,common,dose,bp,dp,limit=STEPS):
    """Replicate EXP106 schedule logic on NEW independent training seeds.

    No confirmatory 107xx, EXP106 108xx, or held-out evaluation seed is used.
    """
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
            a=rng.choice(dp[(mod,fi,0)])
            b=rng.choice(dp[(mod,fi,1)])
            assert dose[a]['PA'].core_id!=dose[b]['PA'].core_id
            if rng.randrange(2):a,b=b,a
            step=half*sched.STEPS+inner
            count+=1
            yield step,base,(a,b),(mod,fi)
            if count==limit:return
    assert count==limit


def schedule_token_hash(seed,common,dose,bp,dp,limit=STEPS):
    hh=sha256(); match=0;length_diff=0
    for step,base,pair,stratum in schedule(seed,common,dose,bp,dp,limit):
        arms=exp106.arm_sources(common,dose,base,pair)
        length_diff+=not exp106.verify_step(arms,*stratum)
        match+=1
        hh.update((json.dumps({'step':step,'base':base,'dose':list(pair),
            'stratum':[list(stratum[0]),stratum[1]]},separators=(',',':'))+'\n').encode())
    return {'steps':match,'schedule_sha256':hh.hexdigest(),
            'matched_token_bag_updates':match,
            'unmatched_indirect_direct_length_hist_updates':length_diff}


def protocol(partition,vocab,dev,eval_audit):
    return {'study':'EXP-109','stage':'PROSPECTIVE_WITHIN_GRAMMAR_REPLICATION_V0',
            'selection':'FIXED_FULL_6400_NO_CHECKPOINT_SELECTION',
            'seeds':list(SEEDS),'repeats':list(REPEATS),'steps':STEPS,
            'arms':list(ARMS),'train_policy':'exact EXP106 7AA+7PA shared + 2-dose arms',
            'primary_outcome':'Per-run pa_only [PA_literal C_query minus AP C_query] on 144 paired anchors',
            'primary_aggregation':'Mean of 2 numerical repeats WITHIN each seed; then mean over 8 independent seeds',
            'secondary_outcome':'Per-run indirect_pp_aa AP C_query minus pa_only AP C_query, NOT controlled for second-position passive exposure',
            'other_secondary':['direct_pa_ap AP','per-query and second-event pair error classes',
                               'unrestricted-vocabulary accuracy','AP rename joint accuracy',
                               'AP reverse-second joint accuracy','AA and PA_literal descriptive competence'],
            'competence_gate':'AA and PA_literal C_query >= 0.75 reported per arm/run, NOT exclusion',
            'all_runs_included':True,'no_optional_stopping':True,
            'fixed_endpoint_updates':STEPS,'numeric_repetition_device':'mps',
            'eval_source':'new participant orderings in existing NON-RESERVED development name sets; no new grammar or new names',
            'eval_anchor_count':144,'eval_conditions':list(CONDITIONS),
            'evaluation_seed':EVAL_SEED,'evaluation_digest':eval_audit['evaluation_sha256'],
            'eval_overlap_legacy':0,'split_partition_digest':partition,
            'legacy_dev_digest':digest([(s.text,s.gold) for s in dev]),
            'vocab_digest':digest(sorted(vocab.items())),'vocab_size':len(vocab),
            'dependency_hashes':hashes(),
            'analysis':'report paired per-seed differences, mean, SD and 95% t interval across 8 independent seeds; descriptive only; inspect full repeats',
            'caveats':['No claim of independent grammatical-family validation: same generator, names and modifier family.',
                       'EXP106 and EXP108 informed these hypotheses; this prospective evaluation does not erase earlier exploratory selection.',
                       'Indirect vs baseline differs in second-slot passive exposure and sentence-length contexts; not isolated abstract abstraction.',
                       'MPS numerical nondeterminism demonstrated in EXP105; two repeats quantify but do not remove it.',
                       'Do NOT use existing reserved 95 name sets, 107xx candidate seeds or protected L+I evaluation.',
                       'The protocol is not externally preregistered until committed or archived before any EXP109 scoring.']}


def manifest_path():return OUT/'manifest.json'


def check_manifest(p,eval_audit):
    path=manifest_path()
    if not path.exists():raise FileNotFoundError('First --phase prepare, then commit the manifest before any run')
    man=json.loads(path.read_text())
    if man['protocol']!=p or man['evaluation_audit']!=eval_audit:
        raise RuntimeError('Frozen protocol/evaluation differs; STOP, do not modify after scoring')
    return man


def prepare(res):
    common,dose,bp,dp,part,dev,vocab=res
    groups,eval_audit=evaluation(dev)
    p=protocol(part,vocab,dev,eval_audit)
    path=manifest_path()
    if path.exists():
        check_manifest(p,eval_audit)
        man=json.loads(path.read_text())
        print('VERIFIED existing manifest',path)
    else:
        print('AUDITING 8 seeds x 6400 updates, no training or evaluation scoring')
        audits={str(s):schedule_token_hash(s,common,dose,bp,dp) for s in SEEDS}
        assert all(a['matched_token_bag_updates']==STEPS for a in audits.values())
        man={'protocol':p,'evaluation_audit':eval_audit,'schedule_audit':audits,
             'protocol_digest':digest(p)}
        write_once(path,man)
    print('PREPARE PASS: 144 independent sentence cores, 6 conditions, no old-dev overlap; NO held-out scoring')
    print('PROTOCOL DIGEST',man['protocol_digest'])
    print('EVALUATION DIGEST',eval_audit['evaluation_sha256'])
    return groups,vocab,man


def freeze_verified(man):
    """Require user to commit experiment code and manifest before any test eval.

    We verify that the installed code and manifest bytes appear unchanged in a
    Git ancestor of HEAD. The manifest is commonly gitignored; user must add -f.
    The initial freeze commit is recorded in the run report.
    """
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        # Historical commit containing the exact manifest and experiment script.
        # No commit auto-created by the evaluator.
        probe=subprocess.run(['git','log','-n','1','--format=%H','--',
            'scripts/exp109_replication_v0.py','reports/exp109/replication_v0/manifest.json'],
            cwd=ROOT,text=True,capture_output=True,check=True)
        commit=probe.stdout.strip()
        if not commit:raise RuntimeError('No commit with EXP109 source and manifest')
        for rel in ('scripts/exp109_replication_v0.py','reports/exp109/replication_v0/manifest.json'):
            data=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
            if data!=((ROOT/rel).read_bytes()):raise RuntimeError(f'Committed freeze differs: {rel}')
        # Verify run is descendant of frozen commit (git log gives ancestor).
        subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=ROOT,check=True)
        return commit
    except (subprocess.CalledProcessError,RuntimeError) as ex:
        raise RuntimeError('EXP109 freeze not committed/unchanged. Commit the exact script and '+
                           'manifest (git add -f ...; git commit) BEFORE scoring.') from ex


def paired_metrics(pred,full,gold):
    n=len(pred)//len(CONDITIONS)
    assert len(pred)==len(full)==len(gold)==n*len(CONDITIONS)
    by={k:[] for k in CONDITIONS}
    for i in range(n):
        for j,key in enumerate(CONDITIONS):
            ix=i*len(CONDITIONS)+j
            by[key].append((pred[ix],full[ix],gold[ix]))
    metrics={k:cf.metric_group([r[0] for r in v],[r[1] for r in v],[r[2] for r in v])
             for k,v in by.items()}
    paired=[];rename_joint=0;reverse_joint=0;eq_rename=0
    second=Counter()
    for i in range(n):
        ap=by['AP'][i];pa=by['PA_literal'][i]
        ac=all(x==y for x,y in zip(ap[0],ap[2]))
        pc=all(x==y for x,y in zip(pa[0],pa[2]))
        paired.append((ac,pc))
        for key,which in (('AP_rename','rename'),('AP_reverse_second','reverse')):
            target=by[key][i]
            both=ac and all(x==y for x,y in zip(target[0],target[2]))
            if which=='rename':rename_joint+=both
            else:reverse_joint+=both
        remap=dict(zip(ap[2],ap[2][1:]+ap[2][:1]))
        # Candidate ids for names are unknown here; build expected rename as
        # gold index permutation: equivariance only when AP was correct.
        if ac and by['AP_rename'][i][0]==by['AP_rename'][i][2]:eq_rename+=1
        p=ap[0][2:];g=ap[2][2:]
        if p==g: second['correct']+=1
        elif any(z not in g for z in p):second['wrong_event_name']+=1
        elif p==(g[1],g[0]):second['reversed']+=1
        elif p[0]==p[1]:second['collapsed']+=1
        else:raise AssertionError('unclassified event pair')
    outcomes={'both_correct':sum(a and b for a,b in paired),
              'AP_wrong_PA_correct':sum((not a) and b for a,b in paired),
              'AP_correct_PA_wrong':sum(a and (not b) for a,b in paired),
              'both_wrong':sum((not a) and (not b) for a,b in paired)}
    assert sum(outcomes.values())==n
    return {'conditions':metrics,'AP_PA_literal_paired_n':outcomes,
            'AP_second_event_pair_classes_n':{k:second[k] for k in
                ('correct','reversed','collapsed','wrong_event_name')},
            'AP_rename_joint_both_correct':rename_joint/n,
            'AP_reverse_second_joint_both_correct':reverse_joint/n}


def run(seed,rep,device,res):
    common,dose,bp,dp,part,dev,vocab=res
    groups,eval_audit=evaluation(dev)
    p=protocol(part,vocab,dev,eval_audit)
    man=check_manifest(p,eval_audit)
    commit=freeze_verified(man)
    if device.type!='mps':raise ValueError('Frozen protocol requires MPS; CPU is smoke only')
    out=OUT/f'seed{seed}_repeat{rep}_{device.type}_step{STEPS}.json'
    weights=CKPT/f'seed{seed}_repeat{rep}_{device.type}_step{STEPS}.pt'
    if out.exists() or weights.exists():raise FileExistsError('Refusing to overwrite existing run outputs: '+str(out))
    nets=exp106.models(seed,device,len(vocab))
    # exp106.models currently has no arbitrary seed restriction.
    all_dev=[g[k] for g in groups for k in CONDITIONS]
    history=[];h=sha256()
    for step,base,pair,stratum in schedule(seed,common,dose,bp,dp):
        sources=exp106.arm_sources(common,dose,base,pair)
        assert exp106.verify_step(sources,*stratum) in (True,False)
        for arm in ARMS:
            enc=pilot.encode_sources(sources[arm],vocab,check=False)
            loss=pilot.one_update(nets[arm][0],nets[arm][1],enc,device)
            if arm=='pa_only':l0=loss
            if arm=='indirect_pp_aa':l1=loss
            if arm=='direct_pa_ap':l2=loss
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
            'stratum':[list(stratum[0]),stratum[1]]},separators=(',',':'))+'\n').encode())
        if step in MILESTONES:
            history.append({'step':step,'losses':{'pa_only':l0,'indirect_pp_aa':l1,'direct_pa_ap':l2}})
            print(f'SEED {seed} REPEAT {rep} STEP {step} losses F={l0:.4f} I={l1:.4f} D={l2:.4f}',flush=True)
    assert h.hexdigest()==man['schedule_audit'][str(seed)]['schedule_sha256']
    results={}
    for arm in ARMS:
        predicted,full,gold=cf.decode_model(nets[arm][0],all_dev,vocab,device)
        results[arm]=paired_metrics(predicted,full,gold)
        assert results[arm]['conditions']['AP']['n']==EVAL_PER_MOD*len(gen.TRAIN_MODS)
    summary={
        'primary_pa_only_gap':results['pa_only']['conditions']['PA_literal']['C_query']-
                            results['pa_only']['conditions']['AP']['C_query'],
        'secondary_indirect_AP_lift':results['indirect_pp_aa']['conditions']['AP']['C_query']-
                            results['pa_only']['conditions']['AP']['C_query'],
        'arm_competence_descriptive':{arm:bool(results[arm]['conditions']['AA']['C_query']>=.75
              and results[arm]['conditions']['PA_literal']['C_query']>=.75) for arm in ARMS}}
    report={'study':'EXP-109','seed':seed,'repeat':rep,'device':device.type,
            'steps':STEPS,'freeze_commit':commit,'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':eval_audit['evaluation_sha256'],
            'schedule_sha256':h.hexdigest(),'results':results,'predeclared_outcomes':summary,
            'loss_history':history,'no_protected_or_LI':True}
    OUT.mkdir(parents=True,exist_ok=True)
    CKPT.mkdir(parents=True,exist_ok=True)
    # Save weights-only; no continuation from selected checkpoints is planned.
    torch.save({'study':'EXP-109','seed':seed,'repeat':rep,'step':STEPS,
       'protocol_digest':man['protocol_digest'],
       'models':{arm:nets[arm][0].state_dict() for arm in ARMS}},weights)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print('SAVED',out)
    print('SAVED',weights)
    print('REPORT SEALED: prespecified results saved; full analysis after all 16 runs')


def summary():
    from statistics import mean,stdev
    from math import sqrt
    if not manifest_path().exists():raise FileNotFoundError('Need --phase prepare')
    manifest=json.loads(manifest_path().read_text())
    records={}
    missing=[]
    for seed in SEEDS:
        for rep in REPEATS:
            path=OUT/f'seed{seed}_repeat{rep}_mps_step{STEPS}.json'
            if not path.exists():missing.append((seed,rep));continue
            r=json.loads(path.read_text())
            assert r['protocol_digest']==manifest['protocol_digest']
            assert r['schedule_sha256']==manifest['schedule_audit'][str(seed)]['schedule_sha256']
            assert r['evaluation_sha256']==manifest['evaluation_audit']['evaluation_sha256']
            records[(seed,rep)]=r
    if missing:
        print('INCOMPLETE: missing prespecified runs:',missing)
        print('No partial statistical estimate is reported. Do not select only completed/successful runs.')
        return
    def across(metric):
        per_seed=[mean(records[(seed,r)]['predeclared_outcomes'][metric] for r in REPEATS)
                   for seed in SEEDS]
        m=mean(per_seed);sd=stdev(per_seed);se=sd/sqrt(len(per_seed))
        # t_0.975, df=7; fixed at registration
        ci=(m-2.364624251*se,m+2.364624251*se)
        print(metric,'per_seed=',[round(x,4) for x in per_seed])
        print('mean=',round(m,4),'sd=',round(sd,4),'95% t interval=',tuple(round(x,4) for x in ci))
    print('EXP109 COMPLETE: 8 independent seeds x 2 MPS numerical repetitions x 3 arms')
    across('primary_pa_only_gap')
    across('secondary_indirect_AP_lift')
    for arm in ARMS:
        ok=sum(records[(s,r)]['predeclared_outcomes']['arm_competence_descriptive'][arm]
                for s in SEEDS for r in REPEATS)
        print('COMPETENCE (DESCRIPTIVE, NOT FILTERED):',arm,ok,'/16')
    print('All runs included; within-grammar replication only; original protected/LI evaluations remain CLOSED')


def smoke(device,res):
    common,dose,bp,dp,part,dev,vocab=res
    # Smoke uses ONLY the OLD explored development corpus (and random model).
    # The independent 144-anchor evaluation is NOT scored in smoke.
    network=exp106.models(SEEDS[0],device,len(vocab))
    for _step,base,pair,stratum in schedule(SEEDS[0],common,dose,bp,dp,limit=2):
        src=exp106.arm_sources(common,dose,base,pair)
        exp106.verify_step(src,*stratum)
        for arm in ARMS:
            pilot.one_update(*network[arm],pilot.encode_sources(src[arm],vocab,check=False),device)
    sample=dev[:3]
    r=pilot.score(network[ARMS[0]][0],pilot.encode_sources(sample,vocab),sample,device,len(vocab))
    assert r['sentences']==3 and 0<=r['C_query']<=1
    print('SMOKE PASS: 2 training updates, existing-dev scorer only; no new evaluation scored')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    parser.add_argument('--device',choices=('mps','cpu'),default='mps')
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--repeat',type=int,choices=REPEATS,default=1)
    args=parser.parse_args()
    if args.phase=='summary':summary();return
    res=resources()
    if args.phase=='prepare':prepare(res);return
    if args.phase=='smoke':smoke(torch.device(args.device),res);return
    if args.phase=='run':run(args.seed,args.repeat,pilot.get_device(args.device),res)

if __name__=='__main__':main()
