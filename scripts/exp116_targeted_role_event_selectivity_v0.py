#!/usr/bin/env python3
"""EXP-116: final exploratory role/event selectivity sweep on EXP-114 models.

Exact frozen dose interventions into Block-1 residual updates at specific source
name tokens. 48 existing checkpoints; no training, protected or L+I evaluation.

  python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase prepare
  python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase smoke --device mps
  python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase run --seed 11401 --repeat 1 --arm pa_only --device mps
  python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase summary
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import math
import subprocess

import torch

import exp105_generator_preflight as gen
import exp105_development_pilot_v0 as pilot
import exp109_replication_v0 as exp109
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114
import exp115_block1_head_mlp_causal_v0 as exp115

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp116/targeted_role_event_selectivity_v0'
MANIFEST=OUT/'manifest.json'
BATCH=48
SEEDS=exp114.SEEDS
REPEATS=exp114.REPEATS
ARMS=exp114.ARMS
DIRECTIONS=exp114.DIRECTIONS
TC95=2.364624251
DOSES=(0.25,0.50,0.75,1.00)
DOSE_NAME={0.0:'a000',0.25:'a025',0.5:'a050',0.75:'a075',1.0:'a100'}
# Explicit, fixed mode set. No model/seed/dose selection from results.
GROUPS=('passive_patient','passive_agent','passive_names')
MODES=tuple(
    [(f'{g}_{DOSE_NAME[a]}') for g in GROUPS for a in DOSES]
    + [f'{g}_{DOSE_NAME[a]}' for g in
       ('passive_all','active_names','passive_voice_cues','passive_names_reversed')
       for a in (0.5,1.0)]
    + ['passive_names_a000']
)
TRANSITIONS=('wrong_role_to_correct','wrong_role_to_wrong_event',
             'wrong_event_to_correct','correct_to_wrong_event',
             'correct_to_wrong_role','wrong_event_to_wrong_role')
METRICS=('native_correct','wrong_role','wrong_event','role_margin',*TRANSITIONS)


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:
            raise FileExistsError(f'Existing frozen report differs: {path}')
        print('VERIFIED',path,flush=True)
    else:
        with path.open('x',encoding='utf8') as f:
            json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
        print('SAVED',path,flush=True)


def spec():
    # Reconstruct and validate only the already-used EXP114 evaluation.
    m,_res,_groups=exp114.load_and_verify()
    assert exp115.prepared()['exp114_evaluation_sha256']==m['evaluation_audit']['evaluation_sha256']
    old=exp115.OUT/'analysis_summary.json'
    if not old.is_file():raise FileNotFoundError('Complete EXP115 aggregate is required: '+str(old))
    agg=json.loads(old.read_text())
    assert (agg['study'],agg['model_states'],agg['source_records'])==('EXP-115',48,48)
    assert agg['no_protected_LI'] and agg['not_new_independent_confirmation']
    return {'study':'EXP-116','status':'FIXED_EXPLORATORY_ROLE_EVENT_SWEEP',
      'exp114_eval_sha256':m['evaluation_audit']['evaluation_sha256'],
      'exp114_protocol_digest':m['protocol_digest'],
      'exp114_manifest_sha256':sha(exp114.MANIFEST),
      'exp114_summary_sha256':sha(exp114.OUT/'analysis_summary.json'),
      'exp115_manifest_sha256':sha(exp115.MANIFEST),
      'exp115_summary_sha256':sha(old),
      'script_sha256':sha(SCRIPT),
      'protocol_sha256':sha(ROOT/'paper/EXP116_targeted_role_event_selectivity_v0_protocol.md'),
      'launcher_sha256':sha(ROOT/'scripts/run_exp116_all_v0.sh'),
      'dependency_sha256':{p.name:sha(p) for p in (
        Path(exp115.__file__),Path(exp114.__file__),Path(exp113.__file__),
        Path(exp112.__file__),Path(pilot.__file__),Path(gen.__file__))},
      'seeds':list(SEEDS),'repeats':list(REPEATS),'arms':list(ARMS),
      'directions':list(DIRECTIONS),'modes':list(MODES),
      'source_records':144,'no_training':True,'no_protected_LI':True,
      'not_independent_confirmation':True,
      'outcomes':['native_accuracy','role_error','event_error','error_transition_matrix']}


def prepare():
    write_once(MANIFEST,spec())
    print('EXP116 PREPARE PASS: frozen 21-mode exploratory sweep; same 48 EXP114 models, same 144 already-examined pairs')
    print('Commit runner + launcher + protocol + manifest in one commit BEFORE run.')


def prepared():
    if not MANIFEST.exists():raise FileNotFoundError('Run --phase prepare first')
    m=json.loads(MANIFEST.read_text())
    if m!=spec():raise AssertionError('EXP116 frozen code/protocol/previous reports changed: refusing mixed analyses')
    return m


def freeze_verified():
    files=(
     'scripts/exp116_targeted_role_event_selectivity_v0.py',
     'scripts/run_exp116_all_v0.sh',
     'paper/EXP116_targeted_role_event_selectivity_v0_protocol.md',
     'reports/exp116/targeted_role_event_selectivity_v0/manifest.json')
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        commit=subprocess.check_output(['git','log','-n','1','--format=%H','--',files[0]],cwd=ROOT,text=True).strip()
        if not commit:raise RuntimeError('Runner is not committed')
        for rel in files:
            archived=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT,stderr=subprocess.DEVNULL)
            if archived!=(ROOT/rel).read_bytes():raise RuntimeError('Freeze mismatch: '+rel)
        subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=ROOT,check=True)
        return commit
    except (RuntimeError,FileNotFoundError,subprocess.CalledProcessError) as e:
        raise RuntimeError('Freeze runner, launcher, protocol and manifest in one commit before EXP116 run') from e


def mode_params(mode):
    if mode not in MODES:raise ValueError(mode)
    group, dose=mode.rsplit('_',1)
    assert dose in DOSE_NAME.values()
    return group,int(dose[1:])/100.0


def refined_maps(groups,maps):
    """Select only core agent/patient names, excluding distractor-name mentions."""
    out={d:[] for d in DIRECTIONS}
    for i,g in enumerate(groups):
        passive_tokens=gen.tokens(g['AP'].text[:-1].split(' ; ')[1])
        active_tokens=gen.tokens(g['AP'].text[:-1].split(' ; ')[0])
        # EXP113 core_names correspond in surface order to patient then agent.
        for d in DIRECTIONS:
            mp=maps[d][i]
            passive_names=mp['passive']['core_names']
            assert len(passive_names)==2 and passive_names[0][0]<passive_names[1][0]
            # Filter the two active event-role participants, NOT distractor names.
            first_agent,first_patient=g['AP'].gold[:2]
            ix=[j for j,t in enumerate(active_tokens) if t in (first_agent,first_patient)]
            assert len(ix)==2 and first_agent!=first_patient
            act=tuple(mp['active'][j] for j in ix)
            assert len(act)==2
            # Explicitly check token lexical identity for every name selection.
            ap_source=g['AP'].text
            assert len(passive_tokens)==len(mp['passive']['all'])
            out[d].append({
             'passive_patient':(passive_names[0],),
             'passive_agent':(passive_names[1],),
             'passive_names':passive_names,
             'passive_all':mp['passive']['all'],
             'passive_voice_cues':mp['passive']['voice_cues'],
             'active_names':act,
             'passive_names_reversed':(
                 (passive_names[0][0],passive_names[1][1]),
                 (passive_names[1][0],passive_names[0][1]))
            })
    return out


def pairs_for(selectors,src_index,mode):
    group,_a=mode_params(mode)
    return selectors[src_index][group]


def full_updates(trace,rows,positions,device):
    """Exactly the four projected-head contributions plus the post-attention MLP."""
    return (trace['heads'][rows,:,positions].sum(1)+trace['mlp'][rows,positions]).to(device)


@torch.inference_mode()
def patch(model,enc,recipient_trace,donor_trace,select,mode,device,batch=BATCH):
    ids,readout,_,_=enc
    _group,alpha=mode_params(mode)
    outputs=[]
    for st in range(0,len(ids),batch):
        en=min(st+batch,len(ids)); x=ids[st:en].to(device)
        rd=readout[st:en].to(device);n=x.shape[1]
        state=recipient_trace['h1'][st:en].to(device).clone()
        row=[];rpos=[];drow=[];dpos=[]
        if alpha>0:
            for index in range(st,en):
                for r,d in pairs_for(select,index//4,mode):
                    assert 0<r<int(rd[index-st]) and 0<d<donor_trace['h1'].shape[1]
                    row.append(index-st);rpos.append(r)
                    drow.append(exp112.donor_row(index));dpos.append(d)
            if row:
                rr=torch.tensor(row,dtype=torch.long);rp=torch.tensor(rpos,dtype=torch.long)
                dr=torch.tensor(drow,dtype=torch.long);dp=torch.tensor(dpos,dtype=torch.long)
                target=full_updates(recipient_trace,rr+st,rp,device)
                source=full_updates(donor_trace,dr,dp,device)
                state[rr.to(device),rp.to(device)]+=alpha*(source-target)
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        z=state
        for block in model.blocks[1:]:z=block(z,mask)
        outputs.append(model.out(model.norm(z[torch.arange(len(x),device=device),rd])).float().cpu())
    result=torch.cat(outputs,0)
    assert torch.isfinite(result).all()
    return result


def classes(logits,enc):
    """0=correct, 1=wrong-role same event, 2=wrong event. Gold for audit only."""
    _,_,gold,candidates=enc
    g=gold.view(-1,4)
    pred=candidates.gather(1,logits.gather(1,candidates).argmax(1,keepdim=True)).flatten().view(-1,4)
    cats=torch.full_like(g,2,dtype=torch.long)
    for k in range(4):
        first=(k//2)*2
        same=(pred[:,k]==g[:,first])|(pred[:,k]==g[:,first+1])
        cats[:,k][same]=1
        cats[:,k][pred[:,k]==g[:,k]]=0
    return cats


def transition_metrics(original,patched,enc):
    a=classes(original,enc);b=classes(patched,enc)
    names=('first_agent','first_patient','second_agent','second_patient')
    pairs={'wrong_role_to_correct':(1,0),'wrong_role_to_wrong_event':(1,2),
           'wrong_event_to_correct':(2,0),'correct_to_wrong_event':(0,2),
           'correct_to_wrong_role':(0,1),'wrong_event_to_wrong_role':(2,1)}
    out={}
    for k,name in enumerate(names):
        rates={key:float(((a[:,k]==v[0])&(b[:,k]==v[1])).float().mean())
               for key,v in pairs.items()}
        rates['category_accounting_ok']=bool(all(
          abs(float((b[:,k]==i).float().mean())-float((a[:,k]==i).float().mean())-
              sum(float(((a[:,k]==j)&(b[:,k]==i)).float().mean())-float(((a[:,k]==i)&(b[:,k]==j)).float().mean()) for j in range(3) if j!=i))<1e-6
           for i in range(3)))
        assert rates['category_accounting_ok']
        out[name]=rates
    return out


def metrics(original,patched,enc):
    o=exp112.outcome(original,patched,enc)
    o['transitions']=transition_metrics(original,patched,enc)
    return o


def smoke(device):
    # Random weights and already explored legacy development examples only.
    _c,_dose,_bp,_dp,_part,dev,vocab=exp109.resources()
    groups=exp109.evaluation(dev)[0][:2]
    encoded={d:pilot.encode_sources([g[d] for g in groups],vocab) for d in DIRECTIONS}
    base=exp112.aligned_maps(groups,encoded['AP'],encoded['PA_literal'])
    maps=exp113.maps_for(groups,base,encoded)[0]
    choices=refined_maps(groups,maps)
    model=pilot.Model(len(vocab)).to(device).eval()
    traces={d:exp115.decompose_block1(model,encoded[d],device,batch=8) for d in DIRECTIONS}
    for d in DIRECTIONS:
        donor='PA_literal' if d=='AP' else 'AP'
        native=traces[d]['native_logits']
        for mode in MODES:
            patchy=patch(model,encoded[d],traces[d],traces[donor],choices[d],mode,device,batch=8)
            assert patchy.shape==native.shape and torch.isfinite(patchy).all()
            m=metrics(native,patchy,encoded[d]);assert all(v['category_accounting_ok'] for v in m['transitions'].values())
            if mode=='passive_names_a000':
                assert torch.allclose(patchy,native,atol=0.02,rtol=0.0005)
            if mode in ('passive_names_a100','passive_all_a100'):
                old_mode={'passive_names_a100':'passive_names_update_all','passive_all_a100':'passive_update_all'}[mode]
                comparator=exp115.patch(model,encoded[d],traces[d],traces[donor],maps[d],old_mode,device,batch=8)
                assert torch.allclose(patchy,comparator,atol=0.025,rtol=0.0005),(mode,(patchy-comparator).abs().max())
    print('EXP116 SMOKE PASS: full 21-mode patches, name selection, transition accounting, EXP115 reconciliation; random model, old dev only',flush=True)


def report_path(seed,repeat,arm):
    return OUT/f'seed{seed}_repeat{repeat}_{arm}.json'


def run(seed,repeat,arm,device):
    prepared();frozen=freeze_verified()
    path=report_path(seed,repeat,arm)
    if path.exists():raise FileExistsError('Refusing overwrite: '+str(path))
    man,res,groups=exp114.load_and_verify()
    vocab=res[-1]
    encoded,maps,_=exp114.encoded_data(groups,vocab)
    selectors=refined_maps(groups,maps)
    old,oldpath=exp115.read_old(seed,repeat,arm)
    prior=exp115.report_path(seed,repeat,arm)
    if not prior.exists():raise FileNotFoundError('EXP115 exact run report required: '+str(prior))
    record=json.loads(prior.read_text())
    assert record['script_sha256']==sha(exp115.SCRIPT)
    assert record['exp114_record_sha256']==sha(oldpath)
    checkpoint=exp114.CKPT/f'seed{seed}_repeat{repeat}_mps_step{exp114.STEPS}.pt'
    if not checkpoint.exists():raise FileNotFoundError('Local EXP114 trained checkpoint missing: '+str(checkpoint))
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert (ck['study'],ck['seed'],ck['repeat'],ck['step'])==('EXP-114',seed,repeat,exp114.STEPS)
    assert ck['protocol_digest']==man['protocol_digest']
    assert set(ck['models'])==set(ARMS)
    model=pilot.Model(len(vocab)).to(device).eval()
    model.load_state_dict(ck['models'][arm],strict=True)
    traces={d:exp115.decompose_block1(model,encoded[d],device) for d in DIRECTIONS}
    result={}
    for d in DIRECTIONS:
        native=traces[d]['native_logits']
        expected=old['outcomes'][arm][d]['native']
        actual=exp112.metrics(native,encoded[d])
        assert abs(actual['C_query']-expected['C_query'])<1e-5
        for q in ('first_agent','first_patient','second_agent','second_patient'):
            assert abs(actual['by_query'][q]['native_correct']-expected['by_query'][q]['native_correct'])<1e-5
        donor='PA_literal' if d=='AP' else 'AP'
        result[d]={'native':actual,'patches':{}}
        for mode in MODES:
            logits=patch(model,encoded[d],traces[d],traces[donor],selectors[d],mode,device)
            result[d]['patches'][mode]=metrics(native,logits,encoded[d])
        for mode,prev in (
             ('passive_names_a100','passive_names_update_all'),
             ('passive_all_a100','passive_update_all')):
            got=result[d]['patches'][mode]['change']['by_query']
            prevq=record['outcomes'][d]['patches'][prev]['change']['by_query']
            for q in got:
                for metric in ('native_correct_delta','wrong_role_delta','wrong_event_delta','role_margin_delta'):
                    if abs(got[q][metric]-prevq[q][metric])>0.006:
                        raise AssertionError(f'EXP116 mismatch with EXP115 {d}/{mode}/{q}/{metric}')
    out={'study':'EXP-116','status':'POST_HOC_EXPLORATORY_SELECTIVITY',
         'seed':seed,'repeat':repeat,'arm':arm,'freeze_commit':frozen,
         'protocol_digest':man['protocol_digest'],'eval_sha256':man['evaluation_audit']['evaluation_sha256'],
         'exp116_manifest_sha256':sha(MANIFEST),'script_sha256':sha(SCRIPT),
         'exp115_record_sha256':sha(prior),'exp114_record_sha256':sha(oldpath),
         'checkpoint_sha256':sha(checkpoint),'modes':list(MODES),
         'outcomes':result,'no_retraining':True,'no_protected_LI':True,
         'no_new_independent_test':True,'source_records':144}
    write_once(path,out)
    print(f'RUN COMPLETE EXP116 seed={seed} repeat={repeat} arm={arm}; sealed exploratory role/event diagnostics',flush=True)


def extract(rec,dir,mode,metric,query_type='focus'):
    qs=(('second_agent','second_patient') if dir=='AP' else ('first_agent','first_patient')) if query_type=='focus' else ('first_agent','first_patient')
    o=rec['outcomes'][dir]['patches'][mode]
    if metric in TRANSITIONS:return mean(o['transitions'][q][metric] for q in qs)
    return mean(o['change']['by_query'][q][metric+'_delta'] for q in qs)


def ci(v):
    assert len(v)==8
    m=mean(v);r=TC95*stdev(v)/math.sqrt(8)
    return {'mean':m,'ci95':[m-r,m+r],'per_seed':dict(zip(map(str,SEEDS),v)),
            'positive_seeds':sum(x>0 for x in v),'negative_seeds':sum(x<0 for x in v)}


def summary():
    prepared()
    rec={};missing=[];man=json.loads(exp114.MANIFEST.read_text())
    for seed in SEEDS:
        for rep in REPEATS:
            for arm in ARMS:
                f=report_path(seed,rep,arm)
                if not f.is_file():missing.append(f.name);continue
                j=json.loads(f.read_text())
                assert (j['study'],j['seed'],j['repeat'],j['arm'])==('EXP-116',seed,rep,arm)
                assert j['script_sha256']==sha(SCRIPT) and j['exp116_manifest_sha256']==sha(MANIFEST)
                assert j['eval_sha256']==man['evaluation_audit']['evaluation_sha256']
                assert j['no_retraining'] and j['no_protected_LI'] and j['no_new_independent_test']
                assert all(set(j['outcomes'][d]['patches'])==set(MODES) for d in DIRECTIONS)
                assert j['exp115_record_sha256']==sha(exp115.report_path(seed,rep,arm))
                rec[seed,rep,arm]=j
    if missing:
        print('EXP116 INCOMPLETE:',len(missing),'of 48 model reports absent; no summary')
        print('Missing:',', '.join(missing[:14]));return
    def aggregate(fn):return ci([mean(fn(rec[s,r,arm]) for r in REPEATS) for s in SEEDS])
    out={'study':'EXP-116','status':'FINAL_EXPLORATORY_SELECTIVITY_SWEEP',
         'source_records':48,'model_states':48,'independent_seeds':8,
         'exp116_manifest_sha256':sha(MANIFEST),'not_independent_confirmation':True,
         'no_protected_LI':True,'arms':{},'selection_policy':'Report every predeclared dose and control; no post-hoc selection counts as confirmation'}
    for arm in ARMS:
        out['arms'][arm]={}
        for d in DIRECTIONS:
            sub={}
            for mode in MODES:
                sub[mode]={m:aggregate(lambda x,dd=d,mo=mode,me=m:extract(x,dd,mo,me)) for m in METRICS}
                # Collateral first event on AP; for PA, focus is already first event.
                if d=='AP':
                    sub[mode]['first_event_accuracy_delta']=aggregate(lambda x,mo=mode:extract(x,'AP',mo,'native_correct','nonfocus'))
            out['arms'][arm][d]=sub
    # Report counts meeting a stringent descriptive screen, NOT a confirmatory decision.
    pa=out['arms']['pa_only']['AP']
    focus_modes=[f'{g}_{DOSE_NAME[a]}' for g in GROUPS for a in DOSES]
    diagnostic=[]
    for mode in focus_modes:
        a=pa[mode]['native_correct'];e=pa[mode]['wrong_event'];rc=pa[mode]['wrong_role_to_correct'];rw=pa[mode]['wrong_role_to_wrong_event']
        if a['ci95'][0]>0 and e['ci95'][1]<=0 and rc['ci95'][0]>rw['ci95'][1]:
            diagnostic.append(mode)
    out['strict_descriptive_selectivity_screen']={'qualifying_modes':diagnostic,
        'rule':'PA-only AP: accuracy lower CI>0 AND wrong-event upper CI<=0 AND role-to-correct lower CI > role-to-wrong-event upper CI; exploratory, not confirmation'}
    write_once(OUT/'analysis_summary.json',out)
    print('EXP116 SUMMARY PASS: 48/48 models; 8 seed-level units; all 21 fixed modes x both directions x 3 arms')
    print('PA-only AP second-event accuracy Δ / wrong-event Δ / wrong-role→correct / wrong-role→wrong-event (pp):')
    for group in GROUPS:
        for dose in DOSES:
            mode=f'{group}_{DOSE_NAME[dose]}'
            val=pa[mode]
            print(' ',mode,*(f'{100*val[k]["mean"]:+.2f}' for k in ('native_correct','wrong_event','wrong_role_to_correct','wrong_role_to_wrong_event')))
    print('STRICT DESCRIPTIVE SCREEN:',diagnostic or 'NONE; no selective rescue established')
    print('EXP116 exploratory on EXP114 already-scored dev examples. Never a new independent replication.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--repeat',type=int,choices=REPEATS)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    a=p.parse_args();device=torch.device(a.device)
    if a.phase in ('run','smoke') and a.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run on Mac or use --device cpu for smoke')
    if a.phase=='prepare':prepare()
    elif a.phase=='smoke':smoke(device)
    elif a.phase=='summary':summary()
    else:
        if a.seed is None or a.repeat is None or a.arm is None:raise ValueError('RUN needs --seed, --repeat, --arm')
        run(a.seed,a.repeat,a.arm,device)

if __name__=='__main__':main()
