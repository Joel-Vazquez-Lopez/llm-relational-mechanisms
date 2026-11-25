#!/usr/bin/env python3
"""Discovery-only output diagnostics for frozen recovery checkpoints."""
import argparse, csv, json, math, hashlib, subprocess
from pathlib import Path
import torch
import role_diversity_recovery_v0 as recovery

parent = recovery.parent
ROOT = parent.ROOT
OUT = ROOT / 'reports/role_circuit_diagnostics_v0'
MANIFEST = OUT / 'study_manifest.json'
PROTOCOL = ROOT / 'paper/ROLE_CIRCUIT_DIAGNOSTICS_V0_PROTOCOL.md'
SCRIPT = Path(__file__).resolve()
POPS = tuple(parent.POPS)
METRICS = ('C_query','accuracy4','full_vocab_accuracy')
TOL = .001


def read(path):
    return json.loads(path.read_text())


def manifest():
    train, held, training, withheld, vocab, m = recovery.inputs()
    rm = recovery.build_manifest(train, m)
    assert read(recovery.MANIFEST) == rm
    inputs = {}
    for seed in recovery.SEEDS:
        for step in recovery.MILESTONES:
            ck, rp = recovery.checkpoint_path(seed, step), recovery.report_path(seed, step)
            r = read(rp)
            assert (r['seed'], r['step'], r['protocol_digest']) == (seed, step, rm['protocol_digest'])
            inputs[f'{seed}_{step}'] = dict(weights=parent.sha(ck), report=parent.sha(rp))
    p = dict(study='ROLE-CIRCUIT-DIAGNOSTICS-v0', classification='exploratory_discovery',
             parent_digest=m['protocol_digest'], recovery_digest=rm['protocol_digest'],
             script_sha256=parent.sha(SCRIPT), protocol_sha256=parent.sha(PROTOCOL),
             seeds=list(recovery.SEEDS), steps=list(recovery.MILESTONES), arms=list(parent.ARMS),
             populations=list(POPS), voices=list(parent.VOICES), tolerance=TOL, inputs=inputs)
    return dict(protocol=p, protocol_digest=parent.digest(p)), (train,held,training,withheld,vocab)


def prepare():
    m, _ = manifest()
    if MANIFEST.exists(): assert read(MANIFEST) == m
    else: parent.save_once(MANIFEST,m)
    print('PREPARE PASS: 16 checkpoint files / 64 model states; no inference',m['protocol_digest'])


def js(logp, logq):
    mid = torch.logaddexp(logp,logq)-math.log(2)
    return .5*((logp.exp()*(logp-mid)).sum(-1)+(logq.exp()*(logq-mid)).sum(-1))


@torch.inference_mode()
def diagnose(model, encoded, device):
    ids, readout, labels, candidates = encoded
    assert len(ids)%4==0
    gold=labels.reshape(-1,4)
    assert torch.all(gold.sort(1).values[:,None,:]==candidates.reshape(-1,4,4).sort(-1).values)
    assert torch.all(gold.sort(1).values.diff(dim=1)>0)
    logits=torch.cat([model(ids[i:i+128].to(device),readout[i:i+128].to(device)).float().cpu()
                      for i in range(0,len(ids),128)])
    assert torch.isfinite(logits).all()
    v=logits.shape[-1]
    logits=logits.reshape(-1,4,v).double()
    lp=logits.log_softmax(-1)
    # Gather in semantic order, not sorted candidate order.
    gl=logits.gather(-1,gold[:,None,:].expand(-1,4,-1))
    gp=lp.gather(-1,gold[:,None,:].expand(-1,4,-1))
    q=torch.arange(4); mate=q^1
    correct=gp[:,q,q]; other=gp[:,q,mate]
    event_log=torch.logaddexp(correct,other)
    role_margin=gl[:,q,q]-gl[:,q,mate]
    wrong1=q^2; wrong2=q^3
    # Correct event vs the other event, independent of vocabulary leakage.
    wrong_log=torch.logsumexp(gp[:,q[:,None],torch.stack((wrong1,wrong2),1)],-1)
    cand_log=torch.logsumexp(gp,-1)
    pred=candidates.gather(1,logits.reshape(-1,v).gather(1,candidates).argmax(1,keepdim=True)).reshape(-1,4)
    ok=pred==gold
    native=dict(C_query=float(ok.all(1).double().mean()),accuracy4=float(ok.double().mean()),full_vocab_accuracy=float((logits.argmax(-1)==gold).double().mean()))
    for event in range(2):
        j=2*event; p0,p1=pred[:,j],pred[:,j+1]; g0,g1=gold[:,j],gold[:,j+1]
        wrong=~(((p0==g0)|(p0==g1))&((p1==g0)|(p1==g1)))
        native[f'event{event+1}_pair_classes']={k:float(x.double().mean()) for k,x in dict(correct=(p0==g0)&(p1==g1)&~wrong,reversed=(p0==g1)&(p1==g0)&~wrong,collapsed=(p0==p1)&~wrong,wrong_event_name=wrong).items()}
    avg=lambda x:float(x.double().mean())
    metrics=dict(n_sources=len(gold),n_queries=len(labels),event_mass=avg(event_log.exp()),candidate_mass=avg(cand_log.exp()),outside_candidate_mass=avg(-torch.expm1(cand_log)),conditional_role_probability=avg(role_margin.sigmoid()),conditional_role_accuracy=avg((role_margin>0).double()+.5*(role_margin==0).double()),candidate_normalized_event_mass=avg((event_log-cand_log).exp()),event_log_odds=avg(event_log-wrong_log),role_logit_margin=avg(role_margin),abs_role_logit_margin=avg(role_margin.abs()),cross_entropy=avg(-correct),event_nll=avg(-event_log),conditional_role_nll=avg(event_log-correct))
    assert abs(metrics['cross_entropy']-metrics['event_nll']-metrics['conditional_role_nll'])<1e-9
    role_pairs=((0,1),(2,3)); event_pairs=((0,2),(1,3))
    for name,pairs in [('role',role_pairs),('event',event_pairs)]:
        metrics[name+'_js_full']=avg(torch.stack([js(lp[:,a],lp[:,b]) for a,b in pairs]))
        cl=gl.log_softmax(-1)
        metrics[name+'_js_candidates']=avg(torch.stack([js(cl[:,a],cl[:,b]) for a,b in pairs]))
        metrics[name+'_prediction_agreement']=avg(torch.stack([pred[:,a]==pred[:,b] for a,b in pairs]))
    # Direction-sensitive query switching, holding the sentence fixed.
    metrics['role_switch_margin']=avg(torch.stack([(gl[:,a,a]-gl[:,a,b])-(gl[:,b,a]-gl[:,b,b]) for a,b in role_pairs]))
    return native, metrics


def verify(actual, expected):
    errors={}
    for k,v in actual.items():
        if isinstance(v,dict):
            errors.update({k+'.'+kk:abs(vv-expected[k][kk]) for kk,vv in v.items()})
        else: errors[k]=abs(v-expected[k])
    assert max(errors.values())<=TOL, errors
    return max(errors.values())


def run(seed, step, device):
    m, (train,held,training,withheld,vocab)=manifest()
    assert read(MANIFEST)==m
    for path in (SCRIPT,PROTOCOL,MANIFEST): recovery.committed(path)
    parent.require_frozen()
    assert seed in recovery.SEEDS and step in recovery.MILESTONES
    out=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if out.exists():
        assert read(out)['protocol_digest']==m['protocol_digest']; print('ALREADY SEALED',seed,step); return
    ck=torch.load(recovery.checkpoint_path(seed,step),map_location='cpu',weights_only=True)
    ref=read(recovery.report_path(seed,step))
    assert (ck['study'],ck['seed'],ck['step'],ck['protocol_digest'])==(recovery.STUDY,seed,step,m['protocol']['recovery_digest'])
    assert ck['schedule_prefix_sha256']==ref['schedule_prefix_sha256']
    assert set(ck['models'])==set(parent.ARMS)
    tests=parent.evaluations(train,held,training,withheld,vocab)
    rows=[]
    for arm in parent.ARMS:
        model=parent.old.Model(len(vocab)).to(device).eval(); model.load_state_dict(ck['models'][arm],strict=True)
        for pop in POPS:
            data=tests[pop][arm] if pop=='train_seen_rich' else tests[pop]
            for voice in parent.VOICES:
                native,metrics=diagnose(model,data[voice],device)
                delta=verify(native,ref['outcomes'][arm][pop][voice])
                rows.append(dict(seed=seed,step=step,arm=arm,population=pop,voice=voice,native=native,metrics=metrics,max_native_delta=delta))
        print('SCORED',seed,step,arm,flush=True)
    parent.save_once(out,dict(protocol_digest=m['protocol_digest'],device=str(device),rows=rows))
    print('SEALED',out,flush=True)


def summary():
    m=read(MANIFEST); rows=[]
    for seed in recovery.SEEDS:
        for step in recovery.MILESTONES:
            p=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
            assert p.exists(), f'Missing {p}; no partial aggregation'
            r=read(p); assert r['protocol_digest']==m['protocol_digest']
            assert len(r['rows'])==len(parent.ARMS)*len(POPS)*len(parent.VOICES)
            rows.extend(r['rows'])
    keys=[(r['seed'],r['step'],r['arm'],r['population'],r['voice']) for r in rows]
    assert len(set(keys))==len(keys)==896
    flat=[{**{k:r[k] for k in ('seed','step','arm','population','voice','max_native_delta')},**r['metrics'],**{k:r['native'][k] for k in METRICS}} for r in rows]
    path=OUT/'diagnostics.csv'; assert not path.exists()
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Preserve individual seeds: equal voice average, not pooled trial inference.
    selected=[]
    for seed,step,arm,pop in sorted(set(k[:4] for k in keys)):
        group=[r for r in flat if (r['seed'],r['step'],r['arm'],r['population'])==(seed,step,arm,pop)]
        assert len(group)==2
        selected.append(dict(seed=seed,step=step,arm=arm,population=pop,**{k:sum(r[k] for r in group)/2 for k in ('C_query','accuracy4','event_mass','conditional_role_probability','conditional_role_accuracy','event_nll','conditional_role_nll','role_js_full','event_js_full','role_switch_margin','candidate_mass')}))
    parent.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',n_rows=len(rows),max_native_delta=max(r['max_native_delta'] for r in rows),voice_averaged=selected))
    print('SUMMARY PASS',len(rows),'rows')


def smoke():
    # Exact synthetic distributions: same-event ambiguity and correct role.
    gold=torch.arange(4).reshape(1,4)
    class Dummy:
        def __call__(self,ids,readout):
            return torch.tensor([[0.,0.,-30.,-30.],[0.,0.,-30.,-30.],[-30.,-30.,0.,0.],[-30.,-30.,0.,0.]])
    enc=(torch.zeros(4,1,dtype=torch.long),torch.zeros(4,dtype=torch.long),gold.flatten(),gold.expand(4,4))
    n,d=diagnose(Dummy(),enc,torch.device('cpu'))
    assert abs(d['event_mass']-1)<1e-9 and abs(d['conditional_role_probability']-.5)<1e-9
    assert abs(d['cross_entropy']-math.log(2))<1e-9 and d['role_js_full']==0
    assert d['event_js_full']>.69 and n['C_query']==0
    print('SMOKE PASS: analytic ambiguity case')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True);ap.add_argument('--seed',type=int);ap.add_argument('--step',type=int);ap.add_argument('--device',choices=('mps','cpu'),default='mps');a=ap.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='smoke':smoke()
    elif a.phase=='summary':summary()
    else:run(a.seed,a.step,torch.device(a.device))
if __name__=='__main__':main()
