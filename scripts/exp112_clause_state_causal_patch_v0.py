#!/usr/bin/env python3
"""EXP-112 exploratory within-model causal clause-state patching (no retraining).

Run from exp102-clean-design using the EXISTING main repo virtualenv:
  python3 scripts/exp112_clause_state_causal_patch_v0.py --phase prepare
  python3 scripts/exp112_clause_state_causal_patch_v0.py --phase smoke --device mps
  python3 scripts/exp112_clause_state_causal_patch_v0.py --phase run --seed 10901 --repeat 1 --arm pa_only --device mps
  python3 scripts/exp112_clause_state_causal_patch_v0.py --phase summary

Post-hoc exploratory analysis of already-scored EXP109 development examples.
NO protected / reserved / within-event LI sources are generated or evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
from statistics import mean, stdev

import torch

import exp105_generator_preflight as gen
import exp105_development_pilot_v0 as pilot
import exp109_replication_v0 as exp109
import exp111_checkpoint_role_evidence_v0 as exp111

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp112/clause_state_patch_v0'
DEPTHS=('embed','block1','block2','block3')
PATCHES=('identity','passive','active','both','passive_misaligned')
DIRECTIONS=(('AP','PA_literal'),('PA_literal','AP'))
BATCH=48
TC=2.364624251


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_once(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=record:raise FileExistsError('Existing artifact differs: '+str(path))
        print('VERIFIED',path)
    else:
        with path.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
        print('SAVED',path)


def clause_map(ap,pa):
    """Map semantically identical clause TOKENS, not answers, after AP<->PA swap.

    Source prefix starts after <bos>; semicolon and period are NEVER patched.
    """
    a,b=ap.text[:-1].split(' ; ')
    p,q=pa.text[:-1].split(' ; ')
    assert (a,b)==(q,p)
    act, pas=gen.tokens(a),gen.tokens(b)
    assert gen.tokens(p)==pas and gen.tokens(q)==act
    # Left AP active begins at 1, right AP passive begins after left + semicolon.
    # Left PA passive begins at 1, right PA active begins after left + semicolon.
    active=tuple((1+i, 2+len(pas)+i) for i in range(len(act)))
    passive=tuple((2+len(act)+i, 1+i) for i in range(len(pas)))
    assert len(active)>0 and len(passive)>0
    return {'active':active,'passive':passive}


def aligned_maps(groups, encoded_ap, encoded_pa):
    """For each anchor check that every mapped pair is the same lexical token."""
    xa=encoded_ap[0];xp=encoded_pa[0]
    assert len(groups)*4==len(xa)==len(xp)
    maps=[]
    for i,g in enumerate(groups):
        mp=clause_map(g['AP'],g['PA_literal'])
        for recipient,donor in mp['active']+mp['passive']:
            assert recipient<xa.shape[1] and donor<xp.shape[1]
            assert all(xa[4*i+j,recipient]==xp[4*i+j,donor] for j in range(4))
        maps.append(mp)
    return maps


def donor_row(index):
    """Same semantic event+role donor: clause states are query-independent by causal mask."""
    return (index//4)*4+((index%4+2)%4)


@torch.inference_mode()
def source_traces(model,encoded,device):
    """Capture ALL token residual streams, with readout->native consistency."""
    ids,readout,_,_=encoded
    traces={layer:[] for layer in DEPTHS}
    for st in range(0,len(ids),BATCH):
        x=ids[st:st+BATCH].to(device)
        rd=readout[st:st+BATCH].to(device)
        n=x.shape[1]
        pos=torch.arange(n,device=device)
        z=model.embed(x)+model.pos(pos)
        traces['embed'].append(z.float().cpu())
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        for j,block in enumerate(model.blocks,1):
            z=block(z,mask)
            traces[f'block{j}'].append(z.float().cpu())
        xlog=model.out(model.norm(z[torch.arange(len(x),device=device),rd]))
        native=model(x,rd)
        if not torch.allclose(native,xlog,rtol=2e-4,atol=2e-4):
            raise AssertionError('EXP112 trace does not match native logits')
    return {k:torch.cat(v,dim=0) for k,v in traces.items()}


def positions_for(maps,idx,direction,patch):
    """Return (recipient token index, donor token index), no answer-token states."""
    mp=maps[idx]
    if direction=='AP':
        raw=mp
    else:
        raw={name:tuple((d,r) for r,d in pairs) for name,pairs in mp.items()}
    if patch=='identity':return ()
    if patch=='passive':return raw['passive']
    if patch=='active':return raw['active']
    if patch=='both':return raw['passive']+raw['active']
    if patch=='passive_misaligned':
        # Same donor token-state multiset as passive but reversed lexical alignment.
        src=raw['passive'];return tuple((r,src[-i-1][1]) for i,(r,_) in enumerate(src))
    raise ValueError(patch)


@torch.inference_mode()
def patch_logits(model, encoded, donor_states, maps, recipient_kind,depth,patch,device):
    """Patch ONLY source-clause hidden states after depth; query/answer unchanged.

    For each corresponding PA/AP source pair the donor query index is swapped
    (first<->second), but donor SOURCE token states are invariant to question
    due to causal masking. The donor answer-token state is never read or copied.
    """
    ids,readout,_,_=encoded
    out=[]
    for st in range(0,len(ids),BATCH):
        en=min(st+BATCH,len(ids));x=ids[st:en].to(device)
        rd=readout[st:en].to(device)
        n=x.shape[1]
        z=model.embed(x)+model.pos(torch.arange(n,device=device))
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        def modify(states):
            if patch=='identity':return states
            rows=[];rpos=[];drows=[];dpos=[]
            for off,global_ix in enumerate(range(st,en)):
                target_positions=positions_for(maps,global_ix//4,recipient_kind,patch)
                source_ix=donor_row(global_ix)
                for a,b in target_positions:
                    # Only clause body, never question/end-of-sentence/readout.
                    assert 0<a<rd[off].item() and 0<b<donor_states.shape[1]
                    rows.append(off);rpos.append(a);drows.append(source_ix);dpos.append(b)
            if rows:
                patch_v=donor_states[drows,dpos].to(device)
                states=states.clone()
                states[rows,rpos]=patch_v
            return states
        if depth=='embed':z=modify(z)
        for j,block in enumerate(model.blocks,1):
            z=block(z,mask)
            if depth==f'block{j}':z=modify(z)
        out.append(model.out(model.norm(z[torch.arange(len(x),device=device),rd])).float().cpu())
    return torch.cat(out,dim=0)


def metrics(logits,encoded):
    """No oracle predictions: all accuracy uses normal four-name competitors."""
    _,_,gold,candidates=encoded
    g=gold.view(-1,4)
    pred=candidates.gather(1,logits.gather(1,candidates).argmax(1,keepdim=True)).flatten().view(-1,4)
    correct=(pred==g)
    r=(logits.gather(1,gold.view(-1,1)).flatten()-
        logits.gather(1,g[:,[1,0,3,2]].reshape(-1,1)).flatten()).view(-1,4)
    out={'C_query':float(correct.all(1).float().mean()),
         'accuracy4':float(correct.float().mean()),
         'full_vocab_accuracy':float((logits.argmax(1)==gold).float().mean()),
         'by_query':{}}
    for k,name in enumerate(('first_agent','first_patient','second_agent','second_patient')):
        first=(k//2)*2
        membership=((pred[:,k]==g[:,first])|(pred[:,k]==g[:,first+1]))
        iscorrect=correct[:,k]
        out['by_query'][name]={
            'native_correct':float(iscorrect.float().mean()),
            'wrong_role':float((membership&~iscorrect).float().mean()),
            'wrong_event':float((~membership).float().mean()),
            'role_margin':float(r[:,k].mean()),
            'positive_role_margin':float((r[:,k]>0).float().mean()),
        }
    return out


def outcome(clean,patched,encoded):
    result=metrics(patched,encoded)
    orig=metrics(clean,encoded)
    result['change']={'C_query_pp':100*(result['C_query']-orig['C_query']),
                      'accuracy4_pp':100*(result['accuracy4']-orig['accuracy4']),
                      'by_query':{q:{k+'_delta':result['by_query'][q][k]-orig['by_query'][q][k]
                          for k in ('native_correct','role_margin','wrong_role','wrong_event')}
                          for q in orig['by_query']}}
    # Distinguish rescue from an arbitrary disruption, using every source/query.
    _,_,gold,candidates=encoded
    def pred(z):return candidates.gather(1,z.gather(1,candidates).argmax(1,keepdim=True)).flatten()
    a=(pred(clean)==gold).view(-1,4)
    b=(pred(patched)==gold).view(-1,4)
    result['changed_query_outcomes']={q:{'incorrect_to_correct':int((~a[:,i]&b[:,i]).sum()),
                                        'correct_to_incorrect':int((a[:,i]&~b[:,i]).sum())}
          for i,q in enumerate(('first_agent','first_patient','second_agent','second_patient'))}
    return result


def get_data():
    man=exp111.read_manifest()
    _,_,_,_,_,dev,vocab=exp109.resources()
    groups,ev_audit=exp109.evaluation(dev)
    assert ev_audit['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
    assert digest(sorted(vocab.items()))==man['protocol']['vocab_digest']
    e={name:pilot.encode_sources([g[name] for g in groups],vocab) for name in ('AP','PA_literal')}
    maps=aligned_maps(groups,e['AP'],e['PA_literal'])
    assert len(maps)==144
    # Every question's source-token span is invariant to suffix wording.
    for cond in e:
        x=e[cond][0]
        for anchor in range(len(groups)):
            s=groups[anchor][cond]
            limit=1+len(gen.tokens(s.text))
            assert all(torch.equal(x[4*anchor,:limit],x[4*anchor+j,:limit]) for j in range(1,4))
    return man,groups,e,maps,vocab


def prepare():
    man=exp111.read_manifest()
    n=0
    for seed in exp109.SEEDS:
        for repeat in exp109.REPEATS:
            exp111.check_run_metadata(seed,repeat,'pa_only',man)
            n+=1
    print(f'PREPARE PASS: {n}/16 source reports and checkpoint bundles, all 48 model states accessible')
    print('No EXP109 evaluation scored; no training, no reserved, protected or LI access')


def smoke(device):
    # Exercise injection on EARLIER development AA/PA source sentences, not EXP109.
    _,_,_,_,_,dev,vocab=exp109.resources()
    p=[s for s in dev if s.voice=='PA'][:4]
    assert len(p)==4
    x=pilot.encode_sources(p,vocab)
    torch.manual_seed(112)
    m=pilot.Model(len(vocab)).to(device).eval()
    traces=source_traces(m,x,device)
    # Fake pair: source tokens identical; only query token changes. Identity must match.
    dummy=[{'active':((1,1),),'passive':((2,2),)} for _ in p]
    baseline=patch_logits(m,x,traces['embed'],dummy,'AP','embed','identity',device)
    with torch.inference_mode():
        true_logits=m(x[0].to(device),x[1].to(device)).float().cpu()
    assert torch.allclose(baseline,true_logits,atol=1e-4,rtol=1e-4)
    for depth in DEPTHS:
        got=patch_logits(m,x,traces[depth],dummy,'AP',depth,'passive',device)
        assert torch.isfinite(got).all()
        if depth=='block3':assert torch.allclose(got,true_logits,atol=1e-4,rtol=1e-4)
    assert len(traces)==len(DEPTHS)
    print('SMOKE PASS: source-only patching works; identity and block3 no-op controls pass')
    print('Random model, familiar dev sentences only; no EXP109 scoring')


def run(seed,repeat,arm,device):
    man,groups,encoded,maps,vocab=get_data()
    report_path,ckpt_path,record=exp111.check_run_metadata(seed,repeat,arm,man)
    path=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
    if path.exists():raise FileExistsError('Existing EXP112 run artifact, do not overwrite: '+str(path))
    try:checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=True)
    except TypeError:checkpoint=torch.load(ckpt_path,map_location='cpu')
    assert checkpoint['seed']==seed and checkpoint['repeat']==repeat
    assert checkpoint['study']=='EXP-109' and checkpoint['step']==exp109.STEPS
    assert checkpoint['protocol_digest']==man['protocol_digest']
    m=pilot.Model(len(vocab)).to(device).eval()
    m.load_state_dict(checkpoint['models'][arm],strict=True)
    original={kind:source_traces(m,encoded[kind],device) for kind in encoded}
    # Causal masks ensure source-token activations do not encode suffix queries.
    for kind in encoded:
        for depth in DEPTHS:
            t=original[kind][depth]
            for anchor,g in enumerate(groups):
                end=1+len(gen.tokens(g[kind].text))
                base=t[4*anchor,:end]
                for q in range(1,4):
                    assert torch.allclose(base,t[4*anchor+q,:end],atol=3e-5,rtol=3e-5),(
                        'SOURCE token hidden state unexpectedly depends on future question',kind,depth,anchor,q)
    clean={kind:patch_logits(m,encoded[kind],original[kind]['embed'],maps,kind,
                             'embed','identity',device) for kind in encoded}
    for condition in encoded:
        native=metrics(clean[condition],encoded[condition])
        ref=record['results'][arm]['conditions'][condition]
        for metric in ('C_query','accuracy4','full_vocab_accuracy'):
            if abs(native[metric]-ref[metric])>1/576+1e-5:
                raise AssertionError(f'EXP109 checkpoint native mismatch: {arm} {condition} {metric}: {native[metric]} vs {ref[metric]}')
    result={kind:{'native':metrics(clean[kind],encoded[kind]),'patches':{}} for kind in encoded}
    for recipient,donor in DIRECTIONS:
        for depth in DEPTHS:
            for patch in PATCHES:
                got=patch_logits(m,encoded[recipient],original[donor][depth],
                                  maps,recipient,depth,patch,device)
                if patch=='identity' or depth=='block3':
                    assert torch.allclose(got,clean[recipient],rtol=2e-4,atol=2e-4),(
                        'Identity or last-block clause patch unexpectedly changed readout',recipient,depth,patch)
                result[recipient]['patches'][f'{depth}__{patch}']=outcome(clean[recipient],got,encoded[recipient])
        q='second_agent' if recipient=='AP' else 'first_agent'
        p=result[recipient]['patches']['block1__passive']['change']['by_query'][q]['role_margin_delta']
        print(f'  {recipient} <- {donor} block1/passive {q} logit-margin delta={p:+.3f}',flush=True)
    rec={'study':'EXP-112','status':'POST_HOC_EXPLORATORY_CAUSAL_INTERVENTION',
         'source':'frozen EXP109 already-evaluated development examples',
         'seed':seed,'repeat':repeat,'arm':arm,'device':device.type,
         'protocol_digest':man['protocol_digest'],
         'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
         'checkpoint_sha256':exp111.sha(ckpt_path),
         'script_sha256':exp111.sha(SCRIPT),
         'source_report_sha256':exp111.sha(report_path),
         'depths':DEPTHS,'patches':PATCHES,'outcomes':result,
         'no_model_training':True,'no_protected_or_LI':True,
         'caveat':'Moves entire SOURCE-CLAUSE residual states across positions: cannot isolate abstract binding vs position/context, and may induce off-manifold interventions.'}
    write_once(path,rec)
    print(f'RUN COMPLETE seed={seed} repeat={repeat} arm={arm}; exploratory source-only patching',flush=True)


def ci(values):
    assert len(values)==8
    m=mean(values);err=TC*stdev(values)/math.sqrt(8)
    return {'mean':m,'ci95':[m-err,m+err],'per_seed':values}


def summary():
    rows={};missing=[]
    for seed in exp109.SEEDS:
        for repeat in exp109.REPEATS:
            for arm in exp109.ARMS:
                path=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
                if not path.exists():missing.append(path.name);continue
                r=json.loads(path.read_text());assert (r['seed'],r['repeat'],r['arm'])==(seed,repeat,arm)
                assert r['no_protected_or_LI'] and r['no_model_training']
                assert r['script_sha256']==exp111.sha(SCRIPT), 'EXP112 script changed after run; do not aggregate mixed code'
                rows[seed,repeat,arm]=r
    if missing:
        print(f'INCOMPLETE: {len(missing)}/48 models unprocessed. First missing: '+', '.join(missing[:6]))
        return
    man=exp111.read_manifest()
    assert all(r['protocol_digest']==man['protocol_digest'] for r in rows.values())
    def across(arm,recipient,depth,patch,query,metric):
        vals=[]
        for seed in exp109.SEEDS:
            values=[]
            for rep in exp109.REPEATS:
                z=rows[seed,rep,arm]['outcomes'][recipient]['patches'][f'{depth}__{patch}']['change']['by_query'][query][metric+'_delta']
                values.append(z)
            vals.append(mean(values))
        return ci(vals)
    res={'study':'EXP-112','status':'POST_HOC_EXPLORATORY_CAUSAL_INTERVENTION',
         'unit':'8 seeds with two within-seed numerical repetitions, no filtering',
         'source_records':48,'no_new_confirmatory_evaluation':True,'arms':{}}
    for arm in exp109.ARMS:
        res['arms'][arm]={}
        for recipient,_donor in DIRECTIONS:
            res['arms'][arm][recipient]={}
            for depth in DEPTHS:
                res['arms'][arm][recipient][depth]={}
                for patch in PATCHES:
                    res['arms'][arm][recipient][depth][patch]={q:{metric:across(arm,recipient,depth,patch,q,metric)
                            for metric in ('native_correct','role_margin','wrong_role','wrong_event')}
                              for q in ('first_agent','first_patient','second_agent','second_patient')}
        q='second_agent'
        for depth in ('block1','block2'):
            for p in ('passive','active','passive_misaligned'):
                v=res['arms'][arm]['AP'][depth][p][q]['role_margin']
                print(f'{arm} AP<-PA {depth}/{p}: second-agent role-margin Δ={v["mean"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]')
    # Helpful post-hoc diagnostic contrasts, *not* confirmatory selection tests.
    def contrast(arm,recipient,depth,patch,qs,metric):
        vals=[]
        for seed in exp109.SEEDS:
            repvals=[]
            for repeat in exp109.REPEATS:
                r=rows[seed,repeat,arm]['outcomes'][recipient]['patches'][f'{depth}__{patch}']['change']['by_query']
                repvals.append(mean(r[q][metric+'_delta'] for q in qs))
            vals.append(mean(repvals))
        return vals
    res['candidate_contrasts']={}
    for arm in exp109.ARMS:
        res['candidate_contrasts'][arm]={}
        for depth in ('embed','block1','block2'):
            ap_q=('second_agent','second_patient')
            pa_q=('first_agent','first_patient')
            ap_first=('first_agent','first_patient')
            p=contrast(arm,'AP',depth,'passive',ap_q,'role_margin')
            a=contrast(arm,'AP',depth,'active',ap_q,'role_margin')
            x=contrast(arm,'AP',depth,'passive_misaligned',ap_q,'role_margin')
            rev=contrast(arm,'PA_literal',depth,'passive',pa_q,'role_margin')
            preserve=contrast(arm,'AP',depth,'passive',ap_first,'role_margin')
            acc=contrast(arm,'AP',depth,'passive',ap_q,'native_correct')
            res['candidate_contrasts'][arm][depth]={
                'AP_second_passive_role_margin_delta':ci(p),
                'AP_second_passive_minus_active_delta':ci([u-v for u,v in zip(p,a)]),
                'AP_second_aligned_minus_misaligned_delta':ci([u-v for u,v in zip(p,x)]),
                'AP_first_when_passive_patched_role_margin_delta':ci(preserve),
                'PA_first_passive_reverse_patch_role_margin_delta':ci(rev),
                'AP_second_passive_native_accuracy_delta':ci(acc),
            }
    write_once(OUT/'analysis_summary.json',res)
    print('SUMMARY PASS: 48/48; exploratory causal tests on previously seen development examples only')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    ap.add_argument('--seed',type=int,choices=exp109.SEEDS,default=10901)
    ap.add_argument('--repeat',type=int,choices=exp109.REPEATS,default=1)
    ap.add_argument('--arm',choices=exp109.ARMS,default='pa_only')
    ap.add_argument('--device',choices=('cpu','mps','cuda','auto'),default='auto')
    args=ap.parse_args()
    if args.phase=='prepare':prepare();return
    if args.phase=='summary':summary();return
    device=pilot.get_device(args.device)
    if args.phase=='smoke':smoke(device);return
    run(args.seed,args.repeat,args.arm,device)


if __name__=='__main__':main()
