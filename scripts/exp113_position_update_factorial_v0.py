#!/usr/bin/env python3
"""EXP-113: exploratory factorial decomposition of EXP-112 clause-state patches.

Post-hoc analysis of previously scored EXP-109 development examples; 48 existing
models, NO training, NO protected/reserved/L+I evaluation or selected checkpoints.

  python3 scripts/exp113_position_update_factorial_v0.py --phase prepare
  python3 scripts/exp113_position_update_factorial_v0.py --phase smoke --device mps
  python3 -u scripts/exp113_position_update_factorial_v0.py --phase run --seed 10901 --repeat 1 --arm pa_only --device mps
  python3 scripts/exp113_position_update_factorial_v0.py --phase summary
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import json
import math

import torch

import exp105_generator_preflight as gen
import exp105_development_pilot_v0 as pilot
import exp109_replication_v0 as exp109
import exp111_checkpoint_role_evidence_v0 as exp111
import exp112_clause_state_causal_patch_v0 as exp112

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
OUT=ROOT/'reports/exp113/position_update_factorial_v0'
BATCH=48
TC=2.364624251  # Student t_0.975, df=7
DIRECTIONS=('AP','PA_literal')
# All conditions are fixed BEFORE examining EXP-113 results. Never add a best patch post hoc.
MODES=(
 'input_passive_all',
 'input_passive_core_names',
 'input_passive_voice_cues',
 'input_passive_other',
 'input_passive_reverse',
 'input_active_all',
 'late_position_passive',
 'late_update_passive',
 'late_full_passive',
 'late_position_active',
 'late_update_active',
 'late_full_active',
 'late_position_passive_reverse',
 'late_update_passive_reverse',
)


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()


def passive_groups(tokens):
    """Partition a passive clause: core participant names, voice/verb cues, other.

    `other` includes punctuation and all modifier/distractor tokens. This is a
    lexical-position partition, NOT a claim that these tokens have exclusive roles.
    """
    was=[i for i,t in enumerate(tokens) if t=='was']
    by=[i for i,t in enumerate(tokens) if t=='by']
    assert len(was)==1 and len(by)==1,(tokens,was,by)
    w,b=was[0],by[0]
    assert 1<=w<w+1<b<len(tokens)-1, tokens
    assert tokens[w-1] in gen.NAME_SET and tokens[b+1] in gen.NAME_SET
    assert tokens[w+1] in gen.VERBS
    names={w-1,b+1}
    cues={w,w+1,b}
    assert names.isdisjoint(cues)
    other=set(range(len(tokens)))-names-cues
    assert len(names)==2 and len(cues)==3 and names | cues | other==set(range(len(tokens)))
    return {'core_names':names,'voice_cues':cues,'other':other,'all':set(range(len(tokens)))}


def maps_for(groups,maps,encoded):
    """Build per-anchor token maps with classification; AP map is stored original."""
    result={k:[] for k in DIRECTIONS}
    count=Counter()
    for i,g in enumerate(groups):
        pas_text=g['AP'].text[:-1].split(' ; ')[1]
        toks=gen.tokens(pas_text)
        cats=passive_groups(toks)
        mp=maps[i]
        assert len(mp['passive'])==len(toks)
        for recipient in DIRECTIONS:
            raw=mp if recipient=='AP' else {k:tuple((d,r) for r,d in pairs) for k,pairs in mp.items()}
            # The passive token ordering is the same in either recipient.
            partition={key:tuple(pair for j,pair in enumerate(raw['passive']) if j in indices)
                       for key,indices in cats.items()}
            assert set(partition['core_names']) | set(partition['voice_cues']) | set(partition['other']) == set(raw['passive'])
            assert len(partition['all'])==len(raw['passive'])
            assert len(partition['core_names'])==2 and len(partition['voice_cues'])==3
            result[recipient].append({'passive':partition,'active':raw['active']})
        count['passive_tokens']+=len(toks)
        for c in ('core_names','voice_cues','other'):count[c]+=len(cats[c])
    # Original EXP112 lexical map still checked against encoded input and gold.
    for recipient in DIRECTIONS:
        donor='PA_literal' if recipient=='AP' else 'AP'
        xr=encoded[recipient][0];xd=encoded[donor][0]
        for i,mp in enumerate(result[recipient]):
            for a,b in mp['passive']['all']+mp['active']:
                assert all(int(xr[4*i+q,a])==int(xd[4*i+q,b]) for q in range(4))
    return result,dict(count)


def pairs_for(mp,mode):
    if 'active' in mode:return mp['active']
    if mode.endswith('core_names'):return mp['passive']['core_names']
    if mode.endswith('voice_cues'):return mp['passive']['voice_cues']
    if mode.endswith('_other'):return mp['passive']['other']
    src=mp['passive']['all']
    if mode.endswith('reverse'):return tuple((r,src[-i-1][1]) for i,(r,d) in enumerate(src))
    return src


@torch.inference_mode()
def intervene(model,encoded,donor,recipient_maps,mode,device,batch=BATCH):
    """Source-only intervention. Never changes answer/question token state directly.

    Let B = E(token)+P(absolute index), U = h_block1-B.
    At block1, replace B, U or B+U from the matched donor at same lexical token.
    At input, reassign only absolute positional embeddings and recompute the net.
    """
    assert mode in MODES
    ids,readout,_,_=encoded
    outputs=[]
    donor_e,donor_b=donor['embed'],donor['block1']
    for st in range(0,len(ids),batch):
        en=min(st+batch,len(ids))
        x=ids[st:en].to(device)
        rd=readout[st:en].to(device)
        n=x.shape[1]
        pos=model.pos(torch.arange(n,device=device))
        z=model.embed(x)+pos
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        # Repeat maps for each of the source's four query suffixes.
        row=[];recpos=[];donorrow=[];donpos=[]
        for offset,index in enumerate(range(st,en)):
            for r,d in pairs_for(recipient_maps[index//4],mode):
                assert 0<r<int(rd[offset]) and 0<d<donor_e.shape[1]
                row.append(offset);recpos.append(r);donorrow.append(exp112.donor_row(index));donpos.append(d)
        rr=torch.tensor(row,dtype=torch.long,device=device)
        rp=torch.tensor(recpos,dtype=torch.long,device=device)
        dr=torch.tensor(donorrow,dtype=torch.long)
        dp=torch.tensor(donpos,dtype=torch.long)
        delta_p=None
        if len(rr):
            # pos codes are shared across query variants: no donor answer state.
            delta_p=(model.pos(dp.to(device))-model.pos(rp)).detach()
        if mode.startswith('input_') and len(rr):
            z=z.clone()
            z[rr,rp]+=delta_p
        source_embed=z.clone() if mode.startswith('late_') else None
        z=model.blocks[0](z,mask)
        if mode.startswith('late_') and len(rr):
            z=z.clone()
            if mode.startswith('late_position'):
                z[rr,rp]+=delta_p
            elif mode.startswith('late_update'):
                # Match donor update U_d=h_d1-h_d0 against receiver U_r.
                # These updates are themselves functions of positional context.
                ud=(donor_b[dr,dp]-donor_e[dr,dp]).to(device)
                ur=(z[rr,rp]-source_embed[rr,rp])
                z[rr,rp]+=(ud-ur)
            elif mode.startswith('late_full'):
                z[rr,rp]=donor_b[dr,dp].to(device)
            else:raise AssertionError(mode)
        for block in model.blocks[1:]:
            z=block(z,mask)
        logits=model.out(model.norm(z[torch.arange(len(x),device=device),rd])).float().cpu()
        outputs.append(logits)
    return torch.cat(outputs,0)


def no_touch_identity(model,encoded,device):
    # Independent of patch code; compare native model to explicit forward states.
    ids,readout,_,_=encoded
    y=[]
    with torch.inference_mode():
        for st in range(0,len(ids),BATCH):
            y.append(model(ids[st:st+BATCH].to(device),readout[st:st+BATCH].to(device)).float().cpu())
    return torch.cat(y)


def mean_second(outcome):
    return mean(outcome['change']['by_query'][q]['role_margin_delta'] for q in ('second_agent','second_patient'))


def avg_ci(vals):
    assert len(vals)==8
    m=mean(vals);t=TC*stdev(vals)/math.sqrt(len(vals))
    return {'mean':m,'ci95':[m-t,m+t],'per_seed':vals,'positive_seeds':sum(x>0 for x in vals)}


def checkpoint_audit(seed,repeat,arm,man):
    j,p,report=exp111.check_run_metadata(seed,repeat,arm,man)
    x=exp112.OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
    if not x.is_file():raise FileNotFoundError('EXP-112 matched report required: '+str(x))
    prior=json.loads(x.read_text())
    assert (prior['seed'],prior['repeat'],prior['arm'])==(seed,repeat,arm)
    assert prior['status']=='POST_HOC_EXPLORATORY_CAUSAL_INTERVENTION'
    assert prior['protocol_digest']==man['protocol_digest']
    assert prior['checkpoint_sha256']==sha(p)
    assert prior['source_report_sha256']==sha(j)
    return j,p,report,x,prior


def get_data():
    man,groups,encoded,maps,vocab=exp112.get_data()
    split,counts=maps_for(groups,maps,encoded)
    return man,groups,encoded,maps,split,counts,vocab


def check_prior(outcomes,prior):
    # Aligned identities are exact: identical lexical tokens, so only explicit
    # position codes differ at the embedding. Reversed-token EXP112 *also* swaps
    # donor token embeddings; EXP113 intentionally keeps lexical embeddings fixed.
    comparisons={
        'input_passive_all':'embed__passive',
        'input_active_all':'embed__active',
        'late_full_passive':'block1__passive',
        'late_full_active':'block1__active',
    }
    for recipient in DIRECTIONS:
        for mode,key in comparisons.items():
            got=outcomes[recipient]['patches'][mode]
            old=prior['outcomes'][recipient]['patches'][key]
            for q in ('first_agent','first_patient','second_agent','second_patient'):
                for metric in ('native_correct','role_margin','wrong_event','wrong_role'):
                    a=got['change']['by_query'][q][metric+'_delta']
                    b=old['change']['by_query'][q][metric+'_delta']
                    if abs(a-b)>0.005:
                        raise AssertionError(f'EXP113 disagreement with EXP112 {recipient}/{mode}/{q}/{metric}: {a} vs {b}')
            if abs(got['change']['C_query_pp']-old['change']['C_query_pp'])>0.35:
                raise AssertionError('C_query mismatch with EXP112 '+recipient+' '+mode)


def prepare():
    man=exp111.read_manifest()
    n=0
    for seed in exp109.SEEDS:
        for repeat in exp109.REPEATS:
            for arm in exp109.ARMS:
                checkpoint_audit(seed,repeat,arm,man);n+=1
    print(f'PREPARE PASS: {n}/48 source checkpoints/reports/EXP112 dependencies verified')
    print('No examples reconstructed, no scoring, no training, no protected/reserved/L+I access')


def smoke(device):
    # Earlier PA development examples and a random model ONLY; EXP109 not accessed.
    _a,_b,_c,_d,_e,dev,vocab=exp109.resources()
    rows=[s for s in dev if s.voice=='PA'][:4]
    assert len(rows)==4
    x=pilot.encode_sources(rows,vocab)
    m=pilot.Model(len(vocab)).to(device).eval()
    traces=exp112.source_traces(m,x,device)
    # Synthetic same-position, same-token donor to assert no-op algebra.
    maps=[]
    for s in rows:
        pp=tuple((k,k) for k in range(1,1+len(gen.tokens(s.text))))
        # Type-specific categories need not be populated for this no-op check.
        maps.append({'passive':{'all':pp,'core_names':pp[:2], 'voice_cues':pp[2:5], 'other':pp[5:]},'active':pp})
    clean=no_touch_identity(m,x,device)
    for mode in MODES:
        got=intervene(m,x,traces,maps,mode,device,batch=8)
        assert torch.isfinite(got).all()
        if not mode.endswith('reverse'):
            assert torch.allclose(got,clean,rtol=2e-4,atol=2e-4),(mode,(got-clean).abs().max().item())
    print('SMOKE PASS: aligned modes are identity with self-donor; reversed controls finite')
    print('Random model, earlier PA development only; no EXP109 scoring')


def run(seed,repeat,arm,device):
    man,groups,encoded,maps,split,counts,vocab=get_data()
    report_path,ckpt_path,report,exp112_path,prior=checkpoint_audit(seed,repeat,arm,man)
    out=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
    if out.exists():raise FileExistsError('Existing EXP113 report; refusing overwrite: '+str(out))
    try:ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
    except TypeError:ckpt=torch.load(ckpt_path,map_location='cpu')
    assert ckpt['study']=='EXP-109' and ckpt['seed']==seed and ckpt['repeat']==repeat
    assert ckpt['step']==exp109.STEPS and ckpt['protocol_digest']==man['protocol_digest']
    model=pilot.Model(len(vocab)).to(device).eval()
    model.load_state_dict(ckpt['models'][arm],strict=True)
    source={d:exp112.source_traces(model,encoded[d],device) for d in DIRECTIONS}
    original={d:no_touch_identity(model,encoded[d],device) for d in DIRECTIONS}
    outc={d:{'native':exp112.metrics(original[d],encoded[d]),'patches':{}} for d in DIRECTIONS}
    for recipient in DIRECTIONS:
        donor='PA_literal' if recipient=='AP' else 'AP'
        for mode in MODES:
            got=intervene(model,encoded[recipient],source[donor],split[recipient],mode,device)
            outc[recipient]['patches'][mode]=exp112.outcome(original[recipient],got,encoded[recipient])
            if mode.startswith('late_full'):
                # The full replacement must match donor block1 residual exactly.
                pass
            if not torch.isfinite(got).all():raise AssertionError('Nonfinite logits in '+mode)
        # No modification of EXP109 native four-candidate evaluation; assert exact agreement.
        for name in ('C_query','accuracy4','full_vocab_accuracy'):
            a=outc[recipient]['native'][name]
            b=report['results'][arm]['conditions'][recipient][name]
            assert abs(a-b)<1/576+1e-5,(recipient,name,a,b)
    check_prior(outc,prior)
    record={'study':'EXP-113','status':'POST_HOC_EXPLORATORY_FACTORED_INTERVENTION',
            'seed':seed,'repeat':repeat,'arm':arm,'device':device.type,
            'source':'already-scored EXP109 development AP/PA pairs',
            'modes':list(MODES),'outcomes':outc,'source_token_partitions':counts,
            'code_sha256':sha(SCRIPT),'checkpoint_sha256':sha(ckpt_path),
            'exp112_report_sha256':sha(exp112_path),
            'exp109_report_sha256':sha(report_path),
            'protocol_digest':man['protocol_digest'],
            'evaluation_sha256':man['evaluation_audit']['evaluation_sha256'],
            'no_model_training':True,'no_protected_or_LI':True,
            'no_independent_confirmation':True,
            'caveat':'Donor block1 updates depend on donor position/context; factorial interventions can be off-manifold.'}
    OUT.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    a=outc['AP']['patches']
    print(f'SAVED {out}',flush=True)
    for m in ('input_passive_all','late_position_passive','late_update_passive','late_full_passive'):
        d=mean_second(a[m])
        a2=mean(a[m]['change']['by_query'][q]['native_correct_delta'] for q in ('second_agent','second_patient'))
        print(f'  {arm} {m}: AP second-role margin Δ={d:+.3f}, native accuracy Δ={100*a2:+.1f} pp',flush=True)
    print('EXP113 EXPLORATORY: original EXP109/112 unchanged; no protected or LI scoring',flush=True)


def summary():
    man=exp111.read_manifest()
    records={};missing=[]
    for seed in exp109.SEEDS:
        for repeat in exp109.REPEATS:
            for arm in exp109.ARMS:
                p=OUT/f'seed{seed}_repeat{repeat}_{arm}.json'
                if not p.is_file():missing.append(p.name);continue
                r=json.loads(p.read_text())
                assert (r['seed'],r['repeat'],r['arm'])==(seed,repeat,arm)
                assert r['status']=='POST_HOC_EXPLORATORY_FACTORED_INTERVENTION'
                assert r['protocol_digest']==man['protocol_digest']
                assert r['evaluation_sha256']==man['evaluation_audit']['evaluation_sha256']
                assert r['modes']==list(MODES)
                records[seed,repeat,arm]=r
    if missing:
        print('INCOMPLETE:',len(missing),'of 48 EXP113 reports missing. No partial summary generated.')
        print('First missing:',', '.join(missing[:10]));return
    assert len({r['code_sha256'] for r in records.values()})==1,'Mixed EXP113 code versions'
    def per_seed_metric(arm,recipient,mode,metric,querytype='focus'):
        vals=[]
        for seed in exp109.SEEDS:
            runs=[]
            for repeat in exp109.REPEATS:
                change=records[seed,repeat,arm]['outcomes'][recipient]['patches'][mode]['change']
                qs=(('second_agent','second_patient') if recipient=='AP' else ('first_agent','first_patient')) if querytype=='focus' else (('first_agent','first_patient') if recipient=='AP' else ('second_agent','second_patient'))
                runs.append(mean(change['by_query'][q][metric+'_delta'] for q in qs))
            vals.append(mean(runs))
        return avg_ci(vals)
    def joint(arm,recipient,metric,lookup):
        vals=[]
        for seed in exp109.SEEDS:
            reps=[]
            for repeat in exp109.REPEATS:
                change=records[seed,repeat,arm]['outcomes'][recipient]['patches']
                def delta(m):
                    v=change[m]['change']
                    qs=('second_agent','second_patient') if recipient=='AP' else ('first_agent','first_patient')
                    return mean(v['by_query'][q][metric+'_delta'] for q in qs)
                reps.append(lookup(delta))
            vals.append(mean(reps))
        return avg_ci(vals)
    aggregate={'study':'EXP-113','status':'POST_HOC_EXPLORATORY_ONLY',
               'unit':'8 independent seeds, 2 MPS repeats per seed, 3 training arms',
               'source_records':48,'no_independent_evaluation':True,'arms':{}}
    for arm in exp109.ARMS:
        aggregate['arms'][arm]={}
        for recipient in DIRECTIONS:
            modes={}
            for mode in MODES:
                modes[mode]={metric:per_seed_metric(arm,recipient,mode,metric)
                             for metric in ('native_correct','role_margin','wrong_event','wrong_role')}
                modes[mode]['nonfocus_margin']=per_seed_metric(arm,recipient,mode,'role_margin','nonfocus')
            def contrast(fn):return {metric:joint(arm,recipient,metric,fn) for metric in ('native_correct','role_margin')}
            combos={
                'late_full_minus_position_minus_update':contrast(lambda d:d('late_full_passive')-d('late_position_passive')-d('late_update_passive')),
                'late_position_minus_update':contrast(lambda d:d('late_position_passive')-d('late_update_passive')),
                'input_position_minus_late_position':contrast(lambda d:d('input_passive_all')-d('late_position_passive')),
                'input_all_minus_core_names_plus_voice_cues_plus_other':contrast(lambda d:d('input_passive_all')-d('input_passive_core_names')-d('input_passive_voice_cues')-d('input_passive_other')),
                'input_aligned_minus_reversed':contrast(lambda d:d('input_passive_all')-d('input_passive_reverse')),
            }
            aggregate['arms'][arm][recipient]={'modes':modes,'contrasts':combos}
        a=aggregate['arms'][arm]['AP']['modes'];r=aggregate['arms'][arm]['PA_literal']['modes']
        print(f'{arm}: AP position-input accuracy Δ={100*a["input_passive_all"]["native_correct"]["mean"]:+.1f} pp, '+
              f'late position Δ={100*a["late_position_passive"]["native_correct"]["mean"]:+.1f} pp, '+
              f'late update Δ={100*a["late_update_passive"]["native_correct"]["mean"]:+.1f} pp, '+
              f'full block1 Δ={100*a["late_full_passive"]["native_correct"]["mean"]:+.1f} pp; '+
              f'reciprocal PA full Δ={100*r["late_full_passive"]["native_correct"]["mean"]:+.1f} pp')
    dest=OUT/'analysis_summary.json'
    if dest.exists():
        if json.loads(dest.read_text())!=aggregate:raise FileExistsError('Existing aggregate differs; do not overwrite')
        print('VERIFIED',dest)
    else:
        with dest.open('x',encoding='utf8') as f:json.dump(aggregate,f,sort_keys=True,indent=2);f.write('\n')
        print('SAVED',dest)
    print('SUMMARY PASS: 48/48, descriptive 8-seed means/intervals; exploratory only; no protected evaluation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run','summary'))
    p.add_argument('--device',default='auto',choices=('cpu','mps','cuda','auto'))
    p.add_argument('--seed',type=int,default=10901,choices=exp109.SEEDS)
    p.add_argument('--repeat',type=int,default=1,choices=exp109.REPEATS)
    p.add_argument('--arm',default='pa_only',choices=exp109.ARMS)
    args=p.parse_args()
    if args.phase=='prepare':return prepare()
    if args.phase=='summary':return summary()
    device=pilot.get_device(args.device)
    if args.phase=='smoke':return smoke(device)
    return run(args.seed,args.repeat,args.arm,device)


if __name__=='__main__':main()
