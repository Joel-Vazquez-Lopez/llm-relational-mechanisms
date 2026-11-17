#!/usr/bin/env python3
"""EXP-111: read-only, POST-HOC exploratory logit and residual-layer diagnostic.

Source: all saved EXP-109 checkpoints, at fixed step 6400. Native readout
logit margins + fixed-L2 linear name probes fitted on OLD AA/PA dev only;
probe scoring is on the ALREADY-SCORED EXP-109 AP and literal PA examples.
No model updates, no new training, no activation patching, no LI/protected.

Phases: prepare | smoke | run | summary. Run once per (seed,repeat,arm).
Read scientific protocol before interpreting a layerwise or probe result.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import json
import math
from statistics import mean,stdev
from typing import Mapping

import torch

import exp105_development_pilot_v0 as pilot
import exp106_voice_recombination_pilot_v0 as exp106
import exp109_replication_v0 as exp109

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/exp111/checkpoint_role_evidence_v0'
SOURCE=ROOT/'reports/exp109/replication_v0'
CKPT=ROOT/'results/exp109/replication_v0'
DEPTHS=('embed','block1','block2','block3')
CONDITIONS=('AP','PA_literal')
NAMES_PROBE=('AA','PA')
LAMBDA=10.0  # fixed, not selected on EXP-109 data
BATCH=64
TC=2.364624251  # t_0.975, df=7


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read_manifest():
    path=SOURCE/'manifest.json'
    if not path.is_file():raise FileNotFoundError(f'Missing frozen EXP-109 manifest: {path}')
    man=json.loads(path.read_text())
    assert man['protocol_digest']==exp109.digest(man['protocol'])
    assert man['protocol']['seeds']==list(exp109.SEEDS)
    assert man['protocol']['repeats']==list(exp109.REPEATS)
    assert man['protocol']['arms']==list(exp109.ARMS)
    return man


def check_run_metadata(seed,repeat,arm,manifest):
    assert seed in exp109.SEEDS and repeat in exp109.REPEATS and arm in exp109.ARMS
    j=SOURCE/f'seed{seed}_repeat{repeat}_mps_step{exp109.STEPS}.json'
    p=CKPT/f'seed{seed}_repeat{repeat}_mps_step{exp109.STEPS}.pt'
    if not j.is_file() or not p.is_file():raise FileNotFoundError(f'Missing EXP109 report/checkpoint: {j}, {p}')
    report=json.loads(j.read_text())
    assert report['study']=='EXP-109' and report['seed']==seed and report['repeat']==repeat
    assert report['steps']==exp109.STEPS and report['device']=='mps'
    assert report['no_protected_or_LI'] is True
    assert report['protocol_digest']==manifest['protocol_digest']
    assert report['evaluation_sha256']==manifest['evaluation_audit']['evaluation_sha256']
    assert report['schedule_sha256']==manifest['schedule_audit'][str(seed)]['schedule_sha256']
    return j,p,report


def forward_layers(model,encoded,device,batch=BATCH):
    """Reproduce pilot.Model.forward; capture answer-token residual for each block.

    The shared FINAL out+norm head applied early is an uncalibrated logit lens.
    Last depth must agree with the native forward pass (checked below).
    """
    ids,readout,labels,candidates=encoded
    traces={k:[] for k in DEPTHS}
    model.eval()
    with torch.inference_mode():
        for st in range(0,len(labels),batch):
            en=min(st+batch,len(labels))
            x=ids[st:en].to(device)
            rd=readout[st:en].to(device)
            n=x.shape[1]
            pos=torch.arange(n,device=device)
            z=model.embed(x)+model.pos(pos)
            ix=torch.arange(len(x),device=device)
            traces['embed'].append(z[ix,rd].float().cpu())
            mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
            for j,block in enumerate(model.blocks,1):
                z=block(z,mask)
                traces[f'block{j}'].append(z[ix,rd].float().cpu())
            native=model(x,rd)
            reproduced=model.out(model.norm(z[ix,rd]))
            if not torch.allclose(native,reproduced,rtol=2e-4,atol=2e-4):
                raise AssertionError('Last-layer feature readout disagrees with native forward')
    return {k:torch.cat(v,0) for k,v in traces.items()}


def lens_logits(model,features,device,batch=256):
    logits={}
    with torch.inference_mode():
        for key in DEPTHS:
            out=[]
            for st in range(0,len(features[key]),batch):
                z=features[key][st:st+batch].to(device)
                out.append(model.out(model.norm(z)).float().cpu())
            logits[key]=torch.cat(out,0)
    return logits


def all4_metrics(logits,encoded):
    _,_,gold,candidates=encoded
    choice=candidates.gather(1,logits.gather(1,candidates).argmax(1,keepdim=True)).flatten()
    n=len(choice)//4
    assert n>0 and len(gold)==4*n
    correct=(choice==gold).reshape(n,4)
    return {'C_query':float(correct.all(dim=1).float().mean()),
            'accuracy4':float(correct.float().mean()),
            'full_vocab_accuracy':float((logits.argmax(1)==gold).float().mean())}


def logit_diagnostics(logits,encoded):
    _,_,gold,candidates=encoded
    n=len(gold)//4
    assert len(gold)==n*4
    gt=gold.reshape(n,4)
    # Correct-event two-name role contrast. Not an oracle EVALUATION; merely a
    # logit difference. A positive margin can coexist with wrong-event choice.
    opposite=gt[:,[1,0,3,2]].reshape(-1)
    event_other=gt[:,[2,3,2,3,0,1,0,1]].reshape(n,4,2)
    # The first two questions compete against names from event 2; last two from event 1.
    assert torch.equal(event_other[0,0],gt[0,2:])
    assert torch.equal(event_other[0,2],gt[0,:2])
    correct_logits=logits.gather(1,gold.view(-1,1)).flatten()
    opposite_logits=logits.gather(1,opposite.view(-1,1)).flatten()
    other_logits=logits.gather(1,event_other.reshape(-1,2)).amax(dim=1)
    margin_role=(correct_logits-opposite_logits).reshape(n,4)
    margin_event=(correct_logits-other_logits).reshape(n,4)
    select=candidates.gather(1,logits.gather(1,candidates).argmax(1,keepdim=True)).flatten().reshape(n,4)
    by_query={}
    for j,key in enumerate(('first_agent','first_patient','second_agent','second_patient')):
        r=margin_role[:,j]
        ev=margin_event[:,j]
        pred=select[:,j]
        gg=gt[:,j]
        other=gt[:,(j//2)*2:(j//2)*2+2]
        role_wrong=((pred!=gg)&((pred==other[:,0])|(pred==other[:,1]))).float().mean()
        event_wrong=((pred!=gg)&((pred!=other[:,0])&(pred!=other[:,1]))).float().mean()
        by_query[key]={'native_correct':float((pred==gg).float().mean()),
                       'wrong_role':float(role_wrong),
                       'wrong_event':float(event_wrong),
                       'mean_correct_vs_other_role_logit':float(r.mean()),
                       'positive_correct_role_margin':float((r>0).float().mean()),
                       'mean_correct_vs_other_event_max_logit':float(ev.mean()),
                       'positive_event_margin':float((ev>0).float().mean())}
    return {'native':all4_metrics(logits,encoded),
            'by_query':by_query,
            'overall':{
                'mean_role_margin':float(margin_role.mean()),
                'positive_role_margin':float((margin_role>0).float().mean()),
                'mean_event_margin':float(margin_event.mean()),
                'positive_event_margin':float((margin_event>0).float().mean())}}


def ridge_fit_predict(train_features,train_y,eval_features,vocab_size,shuffle_labels=False):
    """Fixed ridge name decoder; CPU float64 and no AP/EXP109 fitting labels."""
    X=train_features.double()
    Z=eval_features.double()
    mu=X.mean(0,keepdim=True)
    std=X.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
    X=(X-mu)/std
    Z=(Z-mu)/std
    idx=train_y.long()
    if shuffle_labels:
        generator=torch.Generator(device='cpu').manual_seed(111)
        idx=idx[torch.randperm(len(idx),generator=generator)]
    Y=torch.zeros((len(idx),vocab_size),dtype=torch.float64)
    Y[torch.arange(len(idx)),idx]=1.0
    # An unregularized intercept is implemented by centering target columns.
    mean_y=Y.mean(0,keepdim=True)
    Y=Y-mean_y
    lhs=X.T@X+LAMBDA*torch.eye(X.shape[1],dtype=torch.float64)
    rhs=X.T@Y
    W=torch.linalg.solve(lhs,rhs)
    result=(Z@W+mean_y).float()
    assert bool(torch.isfinite(result).all())
    return result


def measure(model,vocab,probe_sources,eval_sources,device):
    encoded_probe=pilot.encode_sources(probe_sources,vocab)
    encoded_eval={k:pilot.encode_sources(v,vocab) for k,v in eval_sources.items()}
    feat_train=forward_layers(model,encoded_probe,device)
    feat_test={k:forward_layers(model,x,device) for k,x in encoded_eval.items()}
    data={}
    for cond in CONDITIONS:
        enc=encoded_eval[cond]
        lens=lens_logits(model,feat_test[cond],device)
        data[cond]={'lens':{},'probe':{}}
        for depth in DEPTHS:
            data[cond]['lens'][depth]=logit_diagnostics(lens[depth],enc)
            probe_logits=ridge_fit_predict(feat_train[depth],encoded_probe[2],feat_test[cond][depth],len(vocab))
            shuffle_logits=ridge_fit_predict(feat_train[depth],encoded_probe[2],feat_test[cond][depth],len(vocab),True)
            data[cond]['probe'][depth]={**all4_metrics(probe_logits,enc),
                                       'shuffled_label_C_query':all4_metrics(shuffle_logits,enc)['C_query']}
    return data


def audit_against_report(metrics,report,arm):
    for condition in CONDITIONS:
        got=metrics[condition]['lens']['block3']['native']
        reference=report['results'][arm]['conditions'][condition]
        for key in ('C_query','accuracy4','full_vocab_accuracy'):
            if abs(got[key]-reference[key])>1/576+1e-5:
                raise AssertionError(f'EXP111 baseline mismatch {condition} {key}: {got[key]} != {reference[key]}')


def run(seed,repeat,arm,device):
    manifest=read_manifest()
    report_path,checkpoint_path,report=check_run_metadata(seed,repeat,arm,manifest)
    out=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
    if out.exists():raise FileExistsError(f'EXP111 output exists; refusing overwrite: {out}')
    # Reconstruct the EXACT frozen EXP109 vocabulary and already-scored examples.
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    groups,ev_audit=exp109.evaluation(dev)
    if ev_audit['evaluation_sha256']!=manifest['evaluation_audit']['evaluation_sha256']:
        raise RuntimeError('EXP109 evaluation digest does not match frozen manifest')
    assert exp109.digest(sorted(vocab.items()))==manifest['protocol']['vocab_digest']
    probes=[s for s in dev if s.voice in NAMES_PROBE]
    assert len(probes)==144 and {s.voice for s in probes}==set(NAMES_PROBE)
    assert all(all(m!='LI' for m in s.mods) for s in probes)
    eval_sources={k:[g[k] for g in groups] for k in CONDITIONS}
    assert all(len(v)==144 for v in eval_sources.values())
    try: ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    except TypeError: ck=torch.load(checkpoint_path,map_location='cpu')
    assert ck['study']=='EXP-109' and ck['seed']==seed and ck['repeat']==repeat
    assert ck['step']==exp109.STEPS and ck['protocol_digest']==manifest['protocol_digest']
    assert set(ck['models'])==set(exp109.ARMS)
    model=pilot.Model(len(vocab)).to(device)
    model.load_state_dict(ck['models'][arm],strict=True)
    model.eval()
    metrics=measure(model,vocab,probes,eval_sources,device)
    audit_against_report(metrics,report,arm)
    artifact={'study':'EXP-111','status':'POST_HOC_EXPLORATORY_READ_ONLY',
              'model_source':'EXP-109','seed':seed,'repeat':repeat,'arm':arm,
              'depths':DEPTHS,'probe_fit_sources':'earlier 144 AA/PA development sources only, no AP or EXP109 targets',
              'probe_lambda':LAMBDA,'evaluation':'already scored EXP109 AP and literal PA',
              'manifest_digest':manifest['protocol_digest'],
              'evaluation_sha256':ev_audit['evaluation_sha256'],
              'source_report_sha256':sha(report_path),
              'checkpoint_sha256':sha(checkpoint_path),
              'results':metrics,'no_model_training':True,
              'no_protected_or_LI':True}
    OUT.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as fp:json.dump(artifact,fp,indent=2,sort_keys=True);fp.write('\n')
    b=metrics['AP']['lens']['block3']['by_query']
    print(f'SAVED {out}',flush=True)
    print(f'RUN seed={seed} rep={repeat} arm={arm} '+
          f'AP_native_Cq={metrics["AP"]["lens"]["block3"]["native"]["C_query"]:.3f} '+
          f'AP_second_agent_role_margin={b["second_agent"]["mean_correct_vs_other_role_logit"]:.3f} '+
          f'AP_second_patient_role_margin={b["second_patient"]["mean_correct_vs_other_role_logit"]:.3f}',flush=True)
    print('EXPLORATORY: no checkpoint modification, no model training, no LI/protected scoring',flush=True)


def avg_ci(values):
    assert len(values)==8
    m=mean(values);s=stdev(values)
    return {'mean':m,'ci95':[m-TC*s/math.sqrt(8),m+TC*s/math.sqrt(8)],
            'per_seed':values}


def summary():
    records={}
    missing=[]
    for seed in exp109.SEEDS:
        for repeat in exp109.REPEATS:
            for arm in exp109.ARMS:
                path=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
                if not path.is_file():missing.append(path.name);continue
                r=json.loads(path.read_text())
                assert (r['seed'],r['repeat'],r['arm'])==(seed,repeat,arm)
                assert r['status']=='POST_HOC_EXPLORATORY_READ_ONLY'
                records[seed,repeat,arm]=r
    if missing:
        print(f'INCOMPLETE: {len(missing)} of 48 EXP111 reports missing; no partial aggregate analysis')
        print('First missing:',', '.join(missing[:10]));return
    manifest=read_manifest()
    assert {r['manifest_digest'] for r in records.values()}=={manifest['protocol_digest']}
    assert {r['evaluation_sha256'] for r in records.values()}=={manifest['evaluation_audit']['evaluation_sha256']}
    def metric_seed(arm,cond,kind,depth,*path):
        vals=[]
        for seed in exp109.SEEDS:
            repv=[]
            for rep in exp109.REPEATS:
                cur=records[seed,rep,arm]['results'][cond][kind][depth]
                for p in path:cur=cur[p]
                repv.append(cur)
            vals.append(mean(repv))
        return avg_ci(vals)
    result={'study':'EXP-111','status':'POST_HOC_EXPLORATORY_READ_ONLY',
            'unit':'8 independent seeds, averaging 2 MPS repetitions within each seed',
            'source_records':48, 'no_new_confirmatory_evaluation':True,'arms':{}}
    for arm in exp109.ARMS:
        result['arms'][arm]={}
        for condition in CONDITIONS:
            bylayer={}
            for depth in DEPTHS:
                bylayer[depth]={
                    'native_lens_C_query':metric_seed(arm,condition,'lens',depth,'native','C_query'),
                    'native_lens_positive_role_margin':metric_seed(arm,condition,'lens',depth,'overall','positive_role_margin'),
                    'native_lens_mean_role_margin':metric_seed(arm,condition,'lens',depth,'overall','mean_role_margin'),
                    'linear_probe_C_query':metric_seed(arm,condition,'probe',depth,'C_query'),
                    'shuffled_label_C_query':metric_seed(arm,condition,'probe',depth,'shuffled_label_C_query')}
            last={}
            for q in ('first_agent','first_patient','second_agent','second_patient'):
                last[q]={k:metric_seed(arm,condition,'lens','block3','by_query',q,k) for k in
                         ('native_correct','wrong_role','wrong_event',
                          'mean_correct_vs_other_role_logit','positive_correct_role_margin')}
            result['arms'][arm][condition]={'layerwise':bylayer,'native_query_detail':last}
        print(f'{arm}: AP_Cq={100*result["arms"][arm]["AP"]["layerwise"]["block3"]["native_lens_C_query"]["mean"]:.1f}% '
              f'AP_probe_block3_Cq={100*result["arms"][arm]["AP"]["layerwise"]["block3"]["linear_probe_C_query"]["mean"]:.1f}%')
    out=OUT/'analysis_summary.json'
    if out.exists():
        if json.loads(out.read_text())!=result:raise FileExistsError('Existing EXP111 aggregate differs; refusing overwrite')
    else:
        with out.open('x',encoding='utf8') as fp:json.dump(result,fp,indent=2,sort_keys=True);fp.write('\n')
    print('SUMMARY PASS: 48/48 runs, 8 seeds x 2 numerical repetitions x 3 arms; POST-HOC ONLY')
    print('SAVED/VERIFIED',out)


def prepare():
    man=read_manifest()
    reports=0;ckpts=0
    for seed in exp109.SEEDS:
        for rep in exp109.REPEATS:
            _,ckp,_=check_run_metadata(seed,rep,'pa_only',man)
            reports+=1;ckpts+=int(ckp.is_file())
    print(f'PREPARE PASS: {reports}/16 source reports, {ckpts}/16 checkpoint files, '+
          '48 model states, EXP109 frozen provenance verified')
    print('No examples reconstructed or models scored in prepare')


def smoke(device):
    # Smoke tests only earlier AA/PA development examples and a RANDOM model.
    common,dose,bp,dp,partition,dev,vocab=exp109.resources()
    probes=[s for s in dev if s.voice in NAMES_PROBE]
    assert len(probes)==144
    model=pilot.Model(len(vocab)).to(device).eval()
    enc=pilot.encode_sources(probes[:6],vocab)
    feat=forward_layers(model,enc,device,batch=8)
    lg=lens_logits(model,feat,device)
    baseline=pilot.score(model,enc,probes[:6],device,len(vocab))
    assert abs(all4_metrics(lg['block3'],enc)['C_query']-baseline['C_query'])<1e-6
    assert len(feat)==len(DEPTHS)
    train_enc=pilot.encode_sources(probes[:12],vocab)
    train_features=forward_layers(model,train_enc,device,batch=12)
    x=ridge_fit_predict(train_features['block3'],train_enc[2],feat['block3'],len(vocab))
    assert x.shape==(24,len(vocab))
    assert all4_metrics(x,enc)['C_query']>=0
    di=logit_diagnostics(lg['block3'],enc)
    assert 'second_patient' in di['by_query']
    print('SMOKE PASS: residual forward == native, old-dev ridge fit and logit metrics; NO EXP109 scoring')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    ap.add_argument('--seed',type=int,choices=exp109.SEEDS,default=exp109.SEEDS[0])
    ap.add_argument('--repeat',type=int,choices=exp109.REPEATS,default=1)
    ap.add_argument('--arm',choices=exp109.ARMS,default='pa_only')
    ap.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    args=ap.parse_args()
    if args.phase=='summary':summary();return
    if args.phase=='prepare':prepare();return
    device=pilot.get_device(args.device)
    if args.phase=='smoke':smoke(device);return
    run(args.seed,args.repeat,args.arm,device)

if __name__=='__main__':main()
