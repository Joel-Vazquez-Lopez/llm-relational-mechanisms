#!/usr/bin/env python3
"""EXP-114: locked independent-seed/new-development-sentence validation.

Do not use the protected/reserved/L+I evaluation. No analysis-driven choice of
model, checkpoint, seed, run, source or intervention is permitted.

  python3 scripts/exp114_independent_replication_v0.py --phase prepare
  python3 scripts/exp114_independent_replication_v0.py --phase smoke --device mps
  # COMMIT THIS SCRIPT, PROTOCOL AND THE PREPARED MANIFEST IN ONE COMMIT BEFORE RUN.
  python3 -u scripts/exp114_independent_replication_v0.py --phase run --seed 11401 --repeat 1 --device mps
  python3 scripts/exp114_independent_replication_v0.py --phase summary
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from random import Random
from statistics import mean,stdev
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

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp114/independent_replication_v0'
CKPT=ROOT/'results/exp114/independent_replication_v0'
MANIFEST=OUT/'manifest.json'
SEEDS=(11401,11403,11407,11409,11413,11419,11421,11427)
REPEATS=(1,2)
ARMS=exp106.ARMS
STEPS=exp109.STEPS
EVAL_SEED=1142026
EVAL_PER_MOD=16
EVAL_MODS=gen.TRAIN_MODS
DIRECTIONS=('AP','PA_literal')
# These eight modes and three families of fixed tests are frozen in the protocol.
MODES=('input_passive_all','input_active_all','input_passive_core_names',
       'input_passive_voice_cues','input_passive_reverse',
       'late_position_passive','late_update_passive','late_full_passive')
TC95=2.364624251 # t(7), descriptive 95% intervals
MILESTONES=exp109.MILESTONES
assert STEPS==2*sched.STEPS
assert set(MODES).issubset(set(exp113.MODES))
assert not set(SEEDS)&set(exp109.SEEDS)
assert not set(SEEDS)&set(sched.SEEDS)


def digest(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise FileExistsError('Existing frozen file differs; refusing overwrite: '+str(path))
        print('VERIFIED',path,flush=True)
    else:
        with path.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
        print('SAVED',path,flush=True)


def dependency_hashes():
    return {p.name:sha(p) for p in (SCRIPT,Path(gen.__file__),Path(frames.__file__),
             Path(sched.__file__),Path(pilot.__file__),Path(exp106.__file__),
             Path(exp109.__file__),Path(exp112.__file__),Path(exp113.__file__))}


def previous_sources(dev):
    """Audit only previously explored non-reserved development sources."""
    old_groups,old_audit=exp109.evaluation(dev)
    old_man=json.loads(exp109.manifest_path().read_text())
    assert old_audit['evaluation_sha256']==old_man['evaluation_audit']['evaluation_sha256']
    old_surfaces={s.text for s in dev}
    old_perms={}
    _,dev_sets,_=gen.partitions()
    set2ix={frozenset(nameset):i for i,nameset in enumerate(dev_sets)}
    for _j,cid,events in gen.cores(dev_sets,'dev'):
        i=int(cid.split('-')[1]);old_perms.setdefault(i,set()).add(
            (events[0].agent,events[0].patient,events[1].agent,events[1].patient))
    for g in old_groups:
        for s in g.values():old_surfaces.add(s.text)
        i=set2ix[frozenset(g['AP'].gold)]
        old_perms[i].add(tuple(g['AP'].gold))
        old_perms[i].add(tuple(g['PA_literal'].gold))
    return old_surfaces,old_perms,old_audit['evaluation_sha256']


def novel_evaluation(dev):
    """Fresh deterministic 144 AP/PA paired sources, disjoint from EXP109 and old dev.

    Neither the 95 reserved four-name sets nor any LI examples are rendered.
    Only development name sets and the same existing grammar are used.
    """
    _train,dev_sets,part=gen.partitions()
    old_surfaces,old_perms,old_digest=previous_sources(dev)
    rng=Random(EVAL_SEED)
    cells={m:[] for m in EVAL_MODS}
    for i,nameset in enumerate(dev_sets):
        options=[p for p in permutations(nameset,4) if p not in old_perms[i]]
        rng.shuffle(options)
        accepted=0
        for p in options:
            a,b,c,d=p
            # The cyclic relabel counterpart must also be disjoint from prior cores.
            if (b,c,d,a) in old_perms[i]:continue
            v0=gen.VERBS[(i+accepted+1)%len(gen.VERBS)]
            v1=gen.VERBS[(i+accepted+2)%len(gen.VERBS)]
            assert v0!=v1
            events=(gen.Event(a,b,v0,'A',c,d,(i+accepted)%2),
                    gen.Event(c,d,v1,'A',a,b,(i+accepted+1)%2))
            staged=[]
            for mod in EVAL_MODS:
                ap=gen.make_source(events,'AP',mod,'exp114_newdev',f'newdev114-{i}-{accepted}')
                left,right=ap.text[:-1].split(' ; ')
                pa=replace(ap,text=right+' ; '+left+'.',gold=ap.gold[2:]+ap.gold[:2],
                           voice='PA',mods=ap.mods[::-1])
                if ap.text in old_surfaces or pa.text in old_surfaces:break
                assert gen.parse_independent(ap.text)==ap.gold
                assert gen.parse_independent(pa.text)==pa.gold
                assert all(m!='LI' for m in ap.mods+pa.mods)
                assert len(set(ap.gold))==4
                assert len(['<bos>']+gen.tokens(ap.text)+['<sep>']+gen.tokens(gen.Q[0])+['<answer>'])<=pilot.MAX_TOKENS
                assert len(['<bos>']+gen.tokens(pa.text)+['<sep>']+gen.tokens(gen.Q[0])+['<answer>'])<=pilot.MAX_TOKENS
                staged.append((mod,{'AP':ap,'PA_literal':pa}))
            else:
                for mod,g in staged:cells[mod].append(g)
                accepted+=1
            if accepted==2:break
        assert accepted==2,(i,accepted,len(options))
    groups=[]
    for mod in sorted(cells):
        assert len(cells[mod])>=EVAL_PER_MOD
        groups.extend(rng.sample(cells[mod],EVAL_PER_MOD))
    assert len(groups)==144
    all_surfaces=[]
    for g in groups:
        assert g['AP'].text not in old_surfaces
        assert g['PA_literal'].text not in old_surfaces
        assert Counter(gen.tokens(g['AP'].text))==Counter(gen.tokens(g['PA_literal'].text))
        all_surfaces.extend([g['AP'].text,g['PA_literal'].text])
    assert len(all_surfaces)==len(set(all_surfaces))
    assert {x:sum(g['AP'].mods==x for g in groups) for x in EVAL_MODS}=={x:EVAL_PER_MOD for x in EVAL_MODS}
    audit={'anchor_count':len(groups),'presentations':len(all_surfaces),
           'per_modifier':{str(m):EVAL_PER_MOD for m in sorted(EVAL_MODS)},
           'partition_digest':part['partition_digest'],
           'previous_exp109_evaluation_digest':old_digest,
           'no_old_dev_or_exp109_surface_overlap':True,
           'no_old_dev_or_exp109_core_ordering_reuse':True,
           'no_reserved_or_LI':True,
           'evaluation_sha256':digest([[g[d].text,list(g[d].gold),list(g[d].mods)]
                                        for g in groups for d in DIRECTIONS])}
    return groups,audit


def schedule(seed,common,dose,bp,dp,limit=STEPS):
    """Exact EXP109 scheduler, with only the NEW seed membership changed."""
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


def schedule_hash(seed,common,dose,bp,dp,limit=STEPS):
    hh=sha256();n=0;matched=0
    for step,base,pair,stratum in schedule(seed,common,dose,bp,dp,limit):
        sources=exp106.arm_sources(common,dose,base,pair)
        exp106.verify_step(sources,*stratum)
        hh.update((json.dumps({'step':step,'base':base,'dose':list(pair),
             'stratum':[list(stratum[0]),stratum[1]]},separators=(',',':'))+'\n').encode())
        n+=1;matched+=1
    return {'steps':n,'schedule_sha256':hh.hexdigest(),'audited_updates':matched}


def make_protocol(vocab,dev,audit,partition):
    return {
        'study':'EXP-114','version':'INDEPENDENT_SEED_AND_DEV_EXAMPLES_V0',
        'stage':'PROSPECTIVE_FROZEN_IN_BRANCH_BEFORE_TRAINING',
        'seeds':list(SEEDS),'repeats':list(REPEATS),'arms':list(ARMS),
        'steps':STEPS,'device':'mps','training_algorithm':'EXP109 6400-step 7AA+7PA+2 dose unchanged',
        'vocab_digest':digest(sorted(vocab.items())),
        'legacy_dev_digest':digest([(s.text,s.gold) for s in dev]),
        'partition_digest':partition,'evaluation_digest':audit['evaluation_sha256'],
        'evaluation_seed':EVAL_SEED,'evaluation_anchors':audit['anchor_count'],
        'evaluation_conditions':list(DIRECTIONS),'per_modifier_anchors':EVAL_PER_MOD,
        'interventions':list(MODES),
        'primary':'pa_only AP second-event mean query accuracy: input_passive_all minus unpatched',
        'secondary_hierarchy':['pa_only AP second-event mean query accuracy: late_update_passive minus late_position_passive',
                               'pa_only PA first-event mean query accuracy: late_update_passive minus unpatched (negative)'],
        'criterion':'two-sided 95% seed-level t interval excludes zero in prespecified direction; fixed-order secondary gate',
        'controls':['input_active_all','input_passive_reverse','input_passive_core_names','input_passive_voice_cues',
                    'late_full_passive','native wrong-event rates and nonfocus effects'],
        'unit':'8 independent training seeds, averaging 2 MPS numerical repetitions within seed; no filtering',
        'no_optional_stopping':True,'no_checkpoint_selection':True,
        'no_protected_or_reserved_or_LI':True,
        'evaluation_scope':'new development permutations, same synthetic grammar and same dev name sets',
        'dependency_hashes':dependency_hashes(),
        'limitations':['Not independent grammatical family or novel vocabulary.',
                       'Input positional edits are interventions, not proof of position-only heuristics.',
                       'Block1 updates depend on original absolute position and context.',
                       'Large source-clause patches can transfer event/context and create off-manifold states.',
                       'No claim of individual head, neuron or semantic-role circuit.']}


def load_and_verify(man=None):
    if not MANIFEST.is_file():raise FileNotFoundError('Run --phase prepare, commit manifest, then run')
    m=json.loads(MANIFEST.read_text()) if man is None else man
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=novel_evaluation(dev)
    p=make_protocol(vocab,dev,audit,partition)
    assert m['protocol']==p and m['evaluation_audit']==audit
    assert m['protocol_digest']==digest(p)
    return m,(common,dose,bp,dp,partition,dev,vocab),groups


def prepare():
    if MANIFEST.exists():
        m,_,_=load_and_verify()
        print('PREPARE PASS: existing frozen manifest and new examples verified')
        print('MANIFEST DIGEST',m['protocol_digest'])
        return
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,audit=novel_evaluation(dev)
    p=make_protocol(vocab,dev,audit,partition)
    print('AUDIT: 8 independent training seeds x 6400 updates; no training or model scoring',flush=True)
    audits={str(seed):schedule_hash(seed,common,dose,bp,dp) for seed in SEEDS}
    assert len({x['schedule_sha256'] for x in audits.values()})==len(SEEDS)
    manifest={'protocol':p,'protocol_digest':digest(p),
              'evaluation_audit':audit,'schedule_audit':audits}
    write_once(MANIFEST,manifest)
    print('PREPARE PASS: 144 novel AP/PA pairs, no prior-dev/eval overlap, NO protected/LI')
    print('FREEZE COMMIT REQUIRED BEFORE TRAINING',flush=True)


def freeze_verified():
    """Require immutable code + manifest in SAME ancestor commit before any model scoring."""
    source='scripts/exp114_independent_replication_v0.py'
    manifest='reports/exp114/independent_replication_v0/manifest.json'
    protocol='paper/EXP114_independent_replication_v0_protocol.md'
    launcher='scripts/run_exp114_all_v0.sh'
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        c=subprocess.check_output(['git','log','-n','1','--format=%H','--',source],cwd=ROOT,text=True).strip()
        if not c:raise RuntimeError('No committed code')
        for rel in (source,manifest,protocol,launcher):
            stored=subprocess.check_output(['git','show',c+':'+rel],cwd=ROOT,stderr=subprocess.DEVNULL)
            if stored!=(ROOT/rel).read_bytes():raise RuntimeError('Freeze mismatch: '+rel)
        subprocess.run(['git','merge-base','--is-ancestor',c,head],cwd=ROOT,check=True)
        return c
    except (FileNotFoundError,subprocess.CalledProcessError,RuntimeError) as e:
        raise RuntimeError('Before RUN: commit the exact EXP114 code, launcher, protocol and manifest together; do not edit after freeze') from e


def encoded_data(groups,vocab):
    encoded={d:pilot.encode_sources([g[d] for g in groups],vocab) for d in DIRECTIONS}
    maps=exp112.aligned_maps(groups,encoded['AP'],encoded['PA_literal'])
    split,counts=exp113.maps_for(groups,maps,encoded)
    assert len(maps)==len(groups)==144
    return encoded,split,counts


def score_model(model,encoded,split,device):
    donor={d:exp112.source_traces(model,encoded[d],device) for d in DIRECTIONS}
    native={d:exp113.no_touch_identity(model,encoded[d],device) for d in DIRECTIONS}
    out={d:{'native':exp112.metrics(native[d],encoded[d]),'patches':{}} for d in DIRECTIONS}
    for recipient in DIRECTIONS:
        other='PA_literal' if recipient=='AP' else 'AP'
        for mode in MODES:
            logits=exp113.intervene(model,encoded[recipient],donor[other],split[recipient],mode,device)
            assert torch.isfinite(logits).all()
            out[recipient]['patches'][mode]=exp112.outcome(native[recipient],logits,encoded[recipient])
    return out


def smoke(device):
    """Random model and old explored sources; does NOT score new EXP114 examples."""
    # Run prior algebraic guards on random model + old dev, no new evaluation.
    exp113.smoke(device)
    common,dose,bp,dp,_part,dev,vocab=exp109.resources()
    seed=SEEDS[0]
    model_set=exp106.models(seed,device,len(vocab))
    samples=list(schedule(seed,common,dose,bp,dp,limit=2))
    assert len(samples)==2
    for _step,base,pair,stratum in samples:
        sources=exp106.arm_sources(common,dose,base,pair)
        assert exp106.verify_step(sources,*stratum) in (True,False)
        for arm in ARMS:
            pilot.one_update(*model_set[arm],pilot.encode_sources(sources[arm],vocab,check=False),device)
    assert all(torch.isfinite(torch.stack([p.float().norm() for p in model_set[a][0].parameters()])).all() for a in ARMS)
    print('EXP114 SMOKE PASS: 2 training updates, random-model intervention controls; no new evaluation scored')


def run(seed,repeat,device):
    assert seed in SEEDS and repeat in REPEATS
    if device.type!='mps':raise ValueError('Frozen EXP114 training requires MPS; CPU for --phase smoke only')
    man,res,groups=load_and_verify()
    freeze_commit=freeze_verified()
    common,dose,bp,dp,partition,dev,vocab=res
    encoded,split,counts=encoded_data(groups,vocab)
    record_path=OUT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.json'
    checkpoint_path=CKPT/f'seed{seed}_repeat{repeat}_mps_step{STEPS}.pt'
    if record_path.exists() or checkpoint_path.exists():
        raise FileExistsError('Refusing to overwrite training/report checkpoint: '+str(record_path))
    models=exp106.models(seed,device,len(vocab))
    h=sha256();hist=[]
    for step,base,pair,stratum in schedule(seed,common,dose,bp,dp):
        sources=exp106.arm_sources(common,dose,base,pair)
        assert exp106.verify_step(sources,*stratum) in (True,False)
        losses={}
        for arm in ARMS:
            enc=pilot.encode_sources(sources[arm],vocab,check=False)
            losses[arm]=pilot.one_update(*models[arm],enc,device)
        h.update((json.dumps({'step':step,'base':base,'dose':list(pair),
             'stratum':[list(stratum[0]),stratum[1]]},separators=(',',':'))+'\n').encode())
        if step in MILESTONES:
            hist.append({'step':step,'losses':losses})
            print('STEP',step,'losses',' '.join(f'{a}:{losses[a]:.4f}' for a in ARMS),flush=True)
    assert h.hexdigest()==man['schedule_audit'][str(seed)]['schedule_sha256']
    outcomes={}
    for arm in ARMS:
        model=models[arm][0].eval()
        outcomes[arm]=score_model(model,encoded,split,device)
    record={'study':'EXP-114','status':'LOCKED_INDEPENDENT_SEEDS_NEW_DEV_EXAMPLES',
            'seed':seed,'repeat':repeat,'device':'mps','steps':STEPS,
            'freeze_commit':freeze_commit,'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'schedule_sha256':h.hexdigest(),
            'script_sha256':sha(SCRIPT),'loss_history':hist,
            'source_token_partitions':counts,
            'outcomes':outcomes,'no_protected_or_LI':True,
            'independent_from_exp109_seed_and_source_text':True,
            'same_generator_and_dev_name_partition':True}
    CKPT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({'study':'EXP-114','seed':seed,'repeat':repeat,'step':STEPS,
         'protocol_digest':man['protocol_digest'],
         'models':{a:models[a][0].state_dict() for a in ARMS}},checkpoint_path)
    with record_path.open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    print('SAVED',record_path,flush=True)
    print('SAVED',checkpoint_path,flush=True)
    print('REPORT SEALED. Do not inspect/select interim results; aggregate only after all 16 runs.',flush=True)


def get_focus(record,arm,recipient,mode,measure):
    patch=record['outcomes'][arm][recipient]['patches'][mode]
    qs=('second_agent','second_patient') if recipient=='AP' else ('first_agent','first_patient')
    return mean(patch['change']['by_query'][q][measure+'_delta'] for q in qs)


def summarize(vals):
    assert len(vals)==8
    m=mean(vals);sd=stdev(vals);radius=TC95*sd/math.sqrt(8)
    return {'mean':m,'sd':sd,'ci95':[m-radius,m+radius],
            'per_seed':dict(zip(map(str,SEEDS),vals)),
            'positive_seeds':sum(v>0 for v in vals),'negative_seeds':sum(v<0 for v in vals)}


def summary():
    man,_,_=load_and_verify()
    records={};missing=[]
    for seed in SEEDS:
        for rep in REPEATS:
            path=OUT/f'seed{seed}_repeat{rep}_mps_step{STEPS}.json'
            if not path.is_file():missing.append(path.name);continue
            r=json.loads(path.read_text())
            assert (r['study'],r['seed'],r['repeat'],r['steps'])==('EXP-114',seed,rep,STEPS)
            assert r['script_sha256']==sha(SCRIPT)
            assert r['protocol_digest']==man['protocol_digest']
            assert r['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
            assert r['schedule_sha256']==man['schedule_audit'][str(seed)]['schedule_sha256']
            assert r['no_protected_or_LI'] is True
            assert set(r['outcomes'])==set(ARMS)
            records[seed,rep]=r
    if missing:
        print('INCOMPLETE:',len(missing),'of 16 training runs not available; no aggregate or test decision')
        print('Missing:',', '.join(missing));return
    def across(fn):
        vals=[mean(fn(records[s,rep]) for rep in REPEATS) for s in SEEDS]
        return summarize(vals)
    arms={}
    for arm in ARMS:
        arms[arm]={}
        for recipient in DIRECTIONS:
            arms[arm][recipient]={}
            for mode in MODES:
                arms[arm][recipient][mode]={m:across(lambda r,aa=arm,rr=recipient,mm=mode,measure=m:
                    get_focus(r,aa,rr,mm,measure)) for m in ('native_correct','role_margin','wrong_event','wrong_role')}
            arms[arm][recipient]['native_C_query']=across(lambda r,aa=arm,rr=recipient:
                r['outcomes'][aa][rr]['native']['C_query'])
    primary=arms['pa_only']['AP']['input_passive_all']['native_correct']
    contrast=across(lambda r:get_focus(r,'pa_only','AP','late_update_passive','native_correct')-
                             get_focus(r,'pa_only','AP','late_position_passive','native_correct'))
    reciprocal=arms['pa_only']['PA_literal']['late_update_passive']['native_correct']
    # Sequential fixed-order evaluation, not an invitation to vary predictions.
    gate1=primary['ci95'][0]>0
    gate2=gate1 and contrast['ci95'][0]>0
    gate3=gate2 and reciprocal['ci95'][1]<0
    tests={
      'primary_AP_input_position_accuracy_rescue':{'estimate':primary,'direction':'>0','passes':gate1},
      'secondary_1_AP_update_greater_than_late_position_accuracy':{'estimate':contrast,'direction':'>0','passes':gate2 if gate1 else None},
      'secondary_2_PA_reciprocal_update_disruption':{'estimate':reciprocal,'direction':'<0','passes':gate3 if gate2 else None},
      'hierarchical_test_completed':gate3,
      'warning':'Controls and arm-level estimates exploratory/descriptive; no claim of unique circuit or out-of-grammar generalization.'}
    result={'study':'EXP-114','status':'INDEPENDENT_SEEDS_AND_DISJOINT_DEV_SURFACES',
        'protocol_digest':man['protocol_digest'],'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
        'source_records':16,'model_states':48,
        'unit':'8 independent seeds x 2 MPS numerical repetitions; one seed-level estimate per arm',
        'test_decisions':tests,'arms':arms,'no_protected_or_LI':True}
    write_once(OUT/'analysis_summary.json',result)
    print('EXP114 SUMMARY PASS: 16/16 seed/repeat training runs, 48/48 model states, all three arms')
    for k in ('primary_AP_input_position_accuracy_rescue',
              'secondary_1_AP_update_greater_than_late_position_accuracy',
              'secondary_2_PA_reciprocal_update_disruption'):
        d=tests[k];v=d['estimate'];print(k,':',f'{100*v["mean"]:+.2f} pp',
                  f'95% CI [{100*v["ci95"][0]:+.2f}, {100*v["ci95"][1]:+.2f}] pp',
                  'pass='+str(d['passes']))
    print('No protected L+I scoring. This is a new-seed/new-surface confirmation WITHIN the same grammar only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--repeat',type=int,choices=REPEATS)
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    a=p.parse_args()
    device=torch.device(a.device)
    if a.phase in ('smoke','run') and a.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Activate your Mac PyTorch .venv. CPU is smoke-only.')
    if a.phase=='prepare':prepare()
    elif a.phase=='smoke':smoke(device)
    elif a.phase=='summary':summary()
    else:
        if a.seed is None or a.repeat is None:raise ValueError('RUN requires --seed and --repeat')
        run(a.seed,a.repeat,device)


if __name__=='__main__':main()
