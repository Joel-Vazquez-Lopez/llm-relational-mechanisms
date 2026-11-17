#!/usr/bin/env python3
"""EXP-115: read-only, exploratory causal patching of Block 1 residual writes.

Uses ALL 48 previously trained EXP-114 model states and its SAME 144 evaluated
within-grammar development anchors. Does NOT train, open protected / L+I,
or evaluate any new distribution. Fixed exhaustive conditions; no cherry-picking.

  python3 scripts/exp115_block1_head_mlp_causal_v0.py --phase prepare
  python3 scripts/exp115_block1_head_mlp_causal_v0.py --phase smoke --device mps
  python3 -u scripts/exp115_block1_head_mlp_causal_v0.py --phase run --seed 11401 --repeat 1 --arm pa_only --device mps
  bash scripts/run_exp115_all_v0.sh
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import math

import torch
import torch.nn.functional as F

import exp105_development_pilot_v0 as pilot
import exp106_voice_recombination_pilot_v0 as exp106
import exp112_clause_state_causal_patch_v0 as exp112
import exp113_position_update_factorial_v0 as exp113
import exp114_independent_replication_v0 as exp114

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp115/block1_head_mlp_causal_v0'
MANIFEST=OUT/'manifest.json'
BATCH=48
DIRECTIONS=exp114.DIRECTIONS
SEEDS=exp114.SEEDS
REPEATS=exp114.REPEATS
ARMS=exp114.ARMS
TC95=2.364624251
HEADS=pilot.HEADS
WIDTH=pilot.WIDTH
assert HEADS==4 and WIDTH%HEADS==0
MODES=(
    *(f'passive_head{i}' for i in range(HEADS)),
    'passive_attention_all','passive_mlp','passive_update_all',
    'passive_names_update_all','passive_reverse_update_all',
    *(f'active_head{i}' for i in range(HEADS)),
    'active_attention_all','active_mlp','active_update_all'
)


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()

def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:
            raise FileExistsError('Existing result differs; refusing overwrite: '+str(path))
        print('VERIFIED',path,flush=True)
    else:
        with path.open('x',encoding='utf8') as f:
            json.dump(obj,f,sort_keys=True,indent=2);f.write('\n')
        print('SAVED',path,flush=True)


def spec():
    old,_res,_groups=exp114.load_and_verify() # Deterministically audits 144 original EXP114 evaluation anchors
    old_summary=exp114.OUT/'analysis_summary.json'
    if not old_summary.exists():raise FileNotFoundError('EXP114 aggregate summary required')
    summary=json.loads(old_summary.read_text())
    assert summary['study']=='EXP-114' and summary['source_records']==16 and summary['model_states']==48
    assert summary['protocol_digest']==old['protocol_digest']
    return {'study':'EXP-115','status':'POST_HOC_EXPLORATORY_READ_ONLY',
            'origin_study':'EXP-114','exp114_protocol_digest':old['protocol_digest'],
            'exp114_evaluation_sha256':old['evaluation_audit']['evaluation_sha256'],
            'exp114_manifest_sha256':sha(exp114.MANIFEST),
            'exp114_aggregate_sha256':sha(old_summary),
            'script_sha256':sha(SCRIPT),
            'protocol_sha256':sha(ROOT/'paper/EXP115_block1_head_mlp_causal_v0_protocol.md'),
            'modes':list(MODES),'arms':list(ARMS),'directions':list(DIRECTIONS),
            'seeds':list(SEEDS),'repeats':list(REPEATS),
            'no_training':True,'no_protected_LI':True,
            'unit':'8 seeds; 2 MPS numerical repeats per seed and arm; exploratory intervals'}


def prepare():
    write_once(MANIFEST,spec())
    print('EXP115 PREPARE PASS: existing EXP114 144 pairs and 48 trained models only; no scoring')


def prepared():
    if not MANIFEST.exists():raise FileNotFoundError('First run --phase prepare')
    val=json.loads(MANIFEST.read_text())
    if val!=spec():raise AssertionError('EXP115 code/protocol/EXP114 baseline changed since prepare; refusing mixed analysis')
    return val


@torch.inference_mode()
def decompose_block1(model,encoded,device,batch=BATCH):
    """Exact per-head residual WRITE after W_o and the post-attention MLP write.

    h1 = input_embed + sum(head contributions) + attn output bias + mlp write.
    Compare manual qkv/softmax against native Block1 for each batch.
    """
    ids,readout,_,_=encoded
    results={'heads':[],'mlp':[],'h1':[],'native_logits':[]}
    block=model.blocks[0]
    h=HEADS;w=WIDTH;dim=w//h
    for st in range(0,len(ids),batch):
        en=min(st+batch,len(ids));x=ids[st:en].to(device);rd=readout[st:en].to(device)
        n=x.shape[1];bs=x.shape[0]
        init=model.embed(x)+model.pos(torch.arange(n,device=device))
        norm=block.n1(init)
        # torch.nn.MultiheadAttention's packed q,k,v projection; batch-first.
        qkv=F.linear(norm,block.attn.in_proj_weight,block.attn.in_proj_bias)
        q,k,v=qkv.reshape(bs,n,3,h,dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-1,-2))/math.sqrt(dim)
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        scores=scores.masked_fill(mask[None,None,:,:],float('-inf'))
        weights=torch.softmax(scores,dim=-1)
        outheads=weights@v  # [batch,heads,token,head_dim]
        contributions=torch.stack([
            F.linear(outheads[:,i],block.attn.out_proj.weight[:,i*dim:(i+1)*dim],None)
            for i in range(h)],dim=1)  # [batch,heads,tokens,width]
        attn_out=contributions.sum(1)+block.attn.out_proj.bias
        mlp=block.mlp(block.n2(init+attn_out))
        h1=init+attn_out+mlp
        native=block(init,mask)
        if not torch.allclose(h1,native,atol=0.002,rtol=0.0005):
            diff=(h1-native).abs().max().item()
            raise AssertionError(f'Head/MLP decomposition failed Block 1 consistency max_diff={diff}')
        z=h1
        for later in model.blocks[1:]:z=later(z,mask)
        log=model.out(model.norm(z[torch.arange(bs,device=device),rd])).float().cpu()
        truth=model(x,rd).float().cpu()
        if not torch.allclose(log,truth,atol=0.02,rtol=0.0005):
            raise AssertionError('Native output reconciliation failed')
        results['heads'].append(contributions.float().cpu())
        results['mlp'].append(mlp.float().cpu())
        results['h1'].append(h1.float().cpu())
        results['native_logits'].append(truth)
    return {key:torch.cat(val,dim=0) for key,val in results.items()}


def positions(maps,index,mode):
    mp=maps[index]
    if mode.startswith('active_'):
        return mp['active']
    if mode=='passive_names_update_all':return mp['passive']['core_names']
    full=mp['passive']['all']
    if mode=='passive_reverse_update_all':
        return tuple((a,full[-i-1][1]) for i,(a,_) in enumerate(full))
    return full


def select_components(trace,rows,positions,mode,device):
    if '_head' in mode:
        idx=int(mode.rsplit('head',1)[-1])
        return trace['heads'][rows,idx,positions].to(device)
    if mode.endswith('attention_all'):
        return trace['heads'][rows,:,positions].sum(1).to(device)
    if mode.endswith('_mlp'):
        return trace['mlp'][rows,positions].to(device)
    if mode.endswith('_update_all'):
        return (trace['heads'][rows,:,positions].sum(1)+trace['mlp'][rows,positions]).to(device)
    raise ValueError(mode)


@torch.inference_mode()
def patch(model,recipient_enc,recipient_trace,donor_trace,recipient_maps,mode,device,batch=BATCH):
    assert mode in MODES
    ids,readout,_,_=recipient_enc
    outputs=[]
    for st in range(0,len(ids),batch):
        en=min(st+batch,len(ids));x=ids[st:en].to(device);rd=readout[st:en].to(device)
        row=[];rpos=[];donorrow=[];donpos=[]
        for ix in range(st,en):
            for r,d in positions(recipient_maps,ix//4,mode):
                assert 0<r<rd[ix-st].item() and 0<d<donor_trace['h1'].shape[1]
                row.append(ix-st);rpos.append(r);donorrow.append(exp112.donor_row(ix));donpos.append(d)
        rr=torch.tensor(row,dtype=torch.long);rp=torch.tensor(rpos,dtype=torch.long)
        dr=torch.tensor(donorrow,dtype=torch.long);dp=torch.tensor(donpos,dtype=torch.long)
        state=recipient_trace['h1'][st:en].to(device).clone()
        before=select_components(recipient_trace,rr+st,rp,mode,device)
        after=select_components(donor_trace,dr,dp,mode,device)
        state[rr.to(device),rp.to(device)]+=after-before
        n=x.shape[1];mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        z=state
        for block in model.blocks[1:]:z=block(z,mask)
        logits=model.out(model.norm(z[torch.arange(len(x),device=device),rd])).float().cpu()
        if not torch.isfinite(logits).all():raise AssertionError('Nonfinite logits')
        outputs.append(logits)
    return torch.cat(outputs)


def smoke(device):
    """Only random weights and previously explored old development samples."""
    _common,_dose,_bp,_dp,_part,dev,vocab=exp114.exp109.resources()
    groups,_=exp114.exp109.evaluation(dev)
    groups=groups[:2]
    e={d:pilot.encode_sources([g[d] for g in groups],vocab) for d in DIRECTIONS}
    base=exp112.aligned_maps(groups,e['AP'],e['PA_literal'])
    maps,_=exp113.maps_for(groups,base,e)
    model=pilot.Model(len(vocab)).to(device).eval()
    traces={d:decompose_block1(model,e[d],device,batch=8) for d in DIRECTIONS}
    # Show that full residual-update intervention matches the independently defined EXP113 operation.
    for rec in DIRECTIONS:
        donor='PA_literal' if rec=='AP' else 'AP'
        a=patch(model,e[rec],traces[rec],traces[donor],maps[rec],
                'passive_update_all',device,batch=8)
        b=exp113.intervene(model,e[rec],{
            'embed':exp112.source_traces(model,e[donor],device)['embed'],
            'block1':exp112.source_traces(model,e[donor],device)['block1']},
            maps[rec],'late_update_passive',device,batch=8)
        if not torch.allclose(a,b,atol=0.025,rtol=0.0005):
            raise AssertionError('EXP113 update-patch cross-check failure')
        for m in MODES:
            got=patch(model,e[rec],traces[rec],traces[donor],maps[rec],m,device,batch=8)
            assert got.shape==a.shape and torch.isfinite(got).all()
    print('EXP115 SMOKE PASS: exact head decomposition, old-dev patch controls, EXP113 update agreement; NO protected/EXP114 scoring',flush=True)


def report_path(seed,repeat,arm):
    return OUT/f'seed{seed}_repeat{repeat}_{arm}.json'


def read_old(seed,repeat,arm):
    p=exp114.OUT/f'seed{seed}_repeat{repeat}_mps_step{exp114.STEPS}.json'
    r=json.loads(p.read_text())
    assert (r['seed'],r['repeat'],r['study'])==(seed,repeat,'EXP-114')
    assert r['no_protected_or_LI'] and r['protocol_digest']==json.loads(exp114.MANIFEST.read_text())['protocol_digest']
    return r,p


def run(seed,repeat,arm,device):
    prepared()
    path=report_path(seed,repeat,arm)
    if path.exists():raise FileExistsError('Existing EXP115 report; do not overwrite '+str(path))
    man,res,groups=exp114.load_and_verify()
    vocab=res[-1]
    encoded,maps,_=exp114.encoded_data(groups,vocab)
    record,oldpath=read_old(seed,repeat,arm)
    cp=exp114.CKPT/f'seed{seed}_repeat{repeat}_mps_step{exp114.STEPS}.pt'
    if not cp.exists():raise FileNotFoundError('Missing LOCAL EXP114 checkpoint (not on GitHub): '+str(cp))
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    assert (ck['study'],ck['seed'],ck['repeat'],ck['step'])==('EXP-114',seed,repeat,exp114.STEPS)
    assert ck['protocol_digest']==man['protocol_digest']
    assert set(ck['models'])==set(ARMS)
    model=pilot.Model(len(vocab)).to(device).eval()
    model.load_state_dict(ck['models'][arm],strict=True)
    traces={d:decompose_block1(model,encoded[d],device) for d in DIRECTIONS}
    result={}
    for d in DIRECTIONS:
        original=traces[d]['native_logits']
        expected=record['outcomes'][arm][d]['native']
        actual=exp112.metrics(original,encoded[d])
        assert abs(actual['C_query']-expected['C_query'])<0.00001, 'Original EXP114 native accuracy mismatch'
        for q in ('first_agent','first_patient','second_agent','second_patient'):
            assert abs(actual['by_query'][q]['native_correct']-expected['by_query'][q]['native_correct'])<1e-5
        result[d]={'native':actual,'patches':{}}
        donor='PA_literal' if d=='AP' else 'AP'
        for m in MODES:
            logits=patch(model,encoded[d],traces[d],traces[donor],maps[d],m,device)
            outcome=exp112.outcome(original,logits,encoded[d])
            result[d]['patches'][m]=outcome
        # Reproduce EXP114 old update intervention exactly as an implementation guard.
        old=record['outcomes'][arm][d]['patches']['late_update_passive']
        got=result[d]['patches']['passive_update_all']
        for q in ('first_agent','first_patient','second_agent','second_patient'):
            for k in ('native_correct_delta','role_margin_delta','wrong_event_delta','wrong_role_delta'):
                va=got['change']['by_query'][q][k];vb=old['change']['by_query'][q][k]
                if abs(va-vb)>0.006:
                    raise AssertionError(f'EXP115 vs EXP114 update patch mismatch {d}/{q}/{k}: {va} != {vb}')
    report={'study':'EXP-115','status':'READ_ONLY_EXPLORATORY',
            'seed':seed,'repeat':repeat,'arm':arm,'protocol_digest':man['protocol_digest'],
            'eval_sha256':man['evaluation_audit']['evaluation_sha256'],
            'exp114_record_sha256':sha(oldpath),'exp114_checkpoint_sha256':sha(cp),
            'exp115_manifest_sha256':sha(MANIFEST),'script_sha256':sha(SCRIPT),
            'modes':list(MODES),'outcomes':result,'no_protected_LI':True,
            'no_retraining':True,'source_records':144}
    write_once(path,report)
    print(f'RUN COMPLETE seed={seed} repeat={repeat} arm={arm} exploratory block1 causal contributions',flush=True)


def measure(report,recipient,mode,met):
    qs=('second_agent','second_patient') if recipient=='AP' else ('first_agent','first_patient')
    return mean(report['outcomes'][recipient]['patches'][mode]['change']['by_query'][q][met+'_delta'] for q in qs)


def ci(v):
    assert len(v)==8
    m=mean(v);s=stdev(v);r=TC95*s/math.sqrt(len(v))
    return {'mean':m,'ci95':[m-r,m+r],
            'positive_seeds':sum(x>0 for x in v),
            'negative_seeds':sum(x<0 for x in v),
            'per_seed':dict(zip(map(str,SEEDS),v))}


def summary():
    prepared()
    records={};missing=[];oldman=json.loads(exp114.MANIFEST.read_text())
    for seed in SEEDS:
        for repeat in REPEATS:
            for arm in ARMS:
                p=report_path(seed,repeat,arm)
                if not p.exists():missing.append(p.name);continue
                r=json.loads(p.read_text())
                assert (r['study'],r['status'],r['seed'],r['repeat'],r['arm'])==('EXP-115','READ_ONLY_EXPLORATORY',seed,repeat,arm)
                assert r['script_sha256']==sha(SCRIPT) and r['exp115_manifest_sha256']==sha(MANIFEST)
                assert r['protocol_digest']==oldman['protocol_digest'] and r['eval_sha256']==oldman['evaluation_audit']['evaluation_sha256']
                assert r['exp114_record_sha256']==sha(exp114.OUT/f'seed{seed}_repeat{repeat}_mps_step{exp114.STEPS}.json')
                assert r['no_protected_LI'] and r['no_retraining'] and set(r['outcomes'])==set(DIRECTIONS)
                assert all(set(r['outcomes'][d]['patches'])==set(MODES) for d in DIRECTIONS)
                records[seed,repeat,arm]=r
    if missing:
        print(f'EXP115 INCOMPLETE: {len(missing)} of 48 reports missing. No aggregate written.')
        print('Missing:',', '.join(missing[:12]));return
    def across(fn):return ci([mean(fn(records[seed,rep,arm]) for rep in REPEATS) for seed in SEEDS])
    result={'study':'EXP-115','status':'EXPLORATORY_CONTRIBUTION_PATCH_DIAGNOSTIC',
            'source_records':48,'model_states':48,'independent_seeds':8,
            'exp114_evaluation_sha256':oldman['evaluation_audit']['evaluation_sha256'],
            'exp115_manifest_sha256':sha(MANIFEST),'no_protected_LI':True,
            'not_new_independent_confirmation':True,'arms':{}}
    for arm in ARMS:
        result['arms'][arm]={}
        for d in DIRECTIONS:
            result['arms'][arm][d]={}
            for mode in MODES:
                result['arms'][arm][d][mode]={metric:across(lambda rec,mo=mode,me=metric,dd=d:measure(rec,dd,mo,me))
                    for metric in ('native_correct','role_margin','wrong_role','wrong_event')}
            result['arms'][arm][d]['candidate_contrasts']={
                'sum_four_heads_minus_attention_all_accuracy':across(lambda rec,dd=d:
                    sum(measure(rec,dd,f'{"passive"}_head{i}','native_correct') for i in range(HEADS))-
                        measure(rec,dd,'passive_attention_all','native_correct')),
                'mlp_minus_attention_accuracy':across(lambda rec,dd=d:
                    measure(rec,dd,'passive_mlp','native_correct')-measure(rec,dd,'passive_attention_all','native_correct')),
                'passive_update_minus_active_update_accuracy':across(lambda rec,dd=d:
                    measure(rec,dd,'passive_update_all','native_correct')-measure(rec,dd,'active_update_all','native_correct')),
                'full_clause_minus_names_only_update_accuracy':across(lambda rec,dd=d:
                    measure(rec,dd,'passive_update_all','native_correct')-measure(rec,dd,'passive_names_update_all','native_correct')),
                'aligned_minus_reversed_update_accuracy':across(lambda rec,dd=d:
                    measure(rec,dd,'passive_update_all','native_correct')-measure(rec,dd,'passive_reverse_update_all','native_correct'))
            }
    write_once(OUT/'analysis_summary.json',result)
    print('EXP115 SUMMARY PASS: 48/48 model-state contribution diagnostics, 8 seed-level independent units; EXPLORATORY')
    for arm in ARMS:
        print('ARM',arm)
        for d in DIRECTIONS:
            qs='AP second-event' if d=='AP' else 'PA first-event'
            vals=result['arms'][arm][d]
            print(' ',qs,'accuracy Δ pp: '+', '.join(
              f'{k}={100*vals[k]["native_correct"]["mean"]:+.2f}' for k in
              ('passive_attention_all','passive_mlp','passive_update_all')))
            for h in range(HEADS):
                k=f'passive_head{h}';v=vals[k]['native_correct']
                print(f'   head {h}: {100*v["mean"]:+.2f} pp CI [{100*v["ci95"][0]:+.2f}, {100*v["ci95"][1]:+.2f}]')
    print('No protected / L+I opened. Head+MLP residual-write patches, not isolated head-role circuits.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--repeat',type=int,choices=REPEATS)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    a=p.parse_args();device=torch.device(a.device)
    if a.phase=='prepare':prepare()
    elif a.phase=='smoke':smoke(device)
    elif a.phase=='summary':summary()
    else:
        if a.seed is None or a.repeat is None or a.arm is None:raise ValueError('RUN needs --seed --repeat --arm')
        if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; use Mac .venv')
        run(a.seed,a.repeat,a.arm,device)

if __name__=='__main__':main()
