#!/usr/bin/env python3
"""Discovery-only readout localisation, grouped transfer and candidate binding."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
import torch
import role_circuit_diagnostics_v0 as diag
r=diag.recovery
p=r.parent
ROOT=p.ROOT
OUT=ROOT/'reports/role_representation_localisation_v0'
MANIFEST=OUT/'study_manifest.json'
PROTOCOL=ROOT/'paper/ROLE_REPRESENTATION_LOCALISATION_V0_PROTOCOL.md'
SCRIPT=Path(__file__).resolve()
SITES=['embedding']+[f'L{i}.{s}' for i in range(3) for s in ('attention','mlp','residual')]+['norm']
POPS=('unseen_roles_bare','unseen_roles_rich','unseen_quartets_bare','unseen_quartets_rich')
REG=1.0

def read(path): return json.loads(path.read_text())

def manifest():
    dm, fixtures=diag.manifest()
    assert read(diag.MANIFEST)==dm
    train,held,_,_,vocab=fixtures
    protocol=dict(study='ROLE-REPRESENTATION-LOCALISATION-v0',classification='exploratory',diagnostic_digest=dm['protocol_digest'],diagnostic_summary_sha=p.sha(diag.OUT/'analysis_summary.json'),script_sha=p.sha(SCRIPT),protocol_sha=p.sha(PROTOCOL),sites=SITES,populations=list(POPS),seeds=list(r.SEEDS),steps=list(r.MILESTONES),arms=list(p.ARMS),ridge=REG,split='fit unseen-role quartets; test disjoint withheld quartets; both AP/PA transfer directions and same voice; same bare/rich plus cross surface',train_ids=p.digest(train),held_ids=p.digest(held),inputs=dm['protocol']['inputs'])
    assert not set(train)&set(held)
    return dict(protocol=protocol,protocol_digest=p.digest(protocol)),fixtures

def prepare():
    m,_=manifest()
    if MANIFEST.exists(): assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

@torch.inference_mode()
def extract(model,encoded,device):
    ids,readout,labels,choices=encoded
    pieces={s:[] for s in SITES}; logits=[]; baseline=[]
    model.eval()
    for i in range(0,len(ids),128):
        a=ids[i:i+128].to(device); b=readout[i:i+128].to(device); batch,n=a.shape
        ix=torch.arange(batch,device=device); pos=torch.arange(n,device=device)
        def keep(name,x):pieces[name].append(x[ix,b].float().cpu())
        x=model.embed(a)+model.pos(pos);keep('embedding',x)
        mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
        for j,block in enumerate(model.blocks):
            z=block.n1(x); att=block.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
            keep(f'L{j}.attention',att); x=x+att
            mlp=block.mlp(block.n2(x));keep(f'L{j}.mlp',mlp);x=x+mlp
            keep(f'L{j}.residual',x)
        h=model.norm(x[ix,b]);pieces['norm'].append(h.float().cpu()); logits.append(model.out(h).float().cpu())
        baseline.append(model(a,b).float().cpu())
    l=torch.cat(logits);native=torch.cat(baseline)
    assert torch.allclose(l,native,atol=1e-4,rtol=1e-5),float((l-native).abs().max())
    # Native argmax uses sorted choices, including exact tie convention.
    pred=choices.gather(1,l.gather(1,choices).argmax(1,keepdim=True)).flatten()
    ok=(pred==labels).reshape(-1,4)
    scores=dict(C_query=float(ok.all(1).double().mean()),accuracy4=float(ok.double().mean()),full_vocab_accuracy=float((l.argmax(1)==labels).double().mean()))
    gold=labels.reshape(-1,4).numpy(); assert len(ids)%4==0
    assert np.all(np.sort(gold,axis=1)==choices.reshape(-1,4,4)[:,0,:].numpy())
    return {k:torch.cat(v).numpy().reshape(-1,4,p.old.WIDTH) for k,v in pieces.items()},gold,scores,float((l-native).abs().max())

def targets(gold):
    # Four queries per source: event and role labels; lexical rank of correct entity.
    n=len(gold); ranks=np.argsort(np.argsort(gold,axis=1),axis=1)
    return dict(event=np.tile([0,0,1,1],n),role=np.tile([0,1,0,1],n),binding_rank=ranks.reshape(-1))

def features(h,gold,task):
    if task!='binding_rank':return h.reshape(-1,h.shape[-1])
    # Candidate rank is a source-specific label, not a global name classifier.
    return h.reshape(-1,h.shape[-1])

def ridge_fit(x,y,k):
    x=x.astype(np.float64);mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-8,1.,sd)
    z=(x-mu)/sd; z=np.column_stack((z,np.ones(len(z))))
    reg=np.eye(z.shape[1])*REG;reg[-1,-1]=0
    w=np.linalg.solve(z.T@z+reg,z.T@np.eye(k)[y]);return mu,sd,w

def ridge_predict(f,x):
    mu,sd,w=f; z=(x.astype(np.float64)-mu)/sd
    return np.column_stack((z,np.ones(len(z))))@w

def score(f,x,y):return float((ridge_predict(f,x).argmax(1)==y).mean())

def sensitivity(h):
    def delta(pairs):
        return float(np.mean([np.mean(np.sum((h[:,a]-h[:,b])**2,axis=-1)) for a,b in pairs]))
    return dict(role_rms=float(np.sqrt(delta(((0,1),(2,3)))/h.shape[-1])),event_rms=float(np.sqrt(delta(((0,2),(1,3)))/h.shape[-1])),state_rms=float(np.sqrt(np.mean(h**2))))

def run(seed,step,device):
    m,(train,held,training,withheld,vocab)=manifest();assert read(MANIFEST)==m
    for path in (SCRIPT,PROTOCOL,MANIFEST):r.committed(path)
    p.require_frozen(); assert seed in r.SEEDS and step in r.MILESTONES
    out=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if out.exists():assert read(out)['protocol_digest']==m['protocol_digest'];print('ALREADY SEALED',seed,step);return
    ck=torch.load(r.checkpoint_path(seed,step),map_location='cpu',weights_only=True);ref=read(r.report_path(seed,step))
    assert (ck['seed'],ck['step'],ck['protocol_digest'],ck['schedule_prefix_sha256'])==(seed,step,ref['protocol_digest'],ref['schedule_prefix_sha256'])
    tests=p.evaluations(train,held,training,withheld,vocab); rows=[];sens=[];checks=[]
    for arm in p.ARMS:
        model=p.old.Model(len(vocab)).to(device);model.load_state_dict(ck['models'][arm],strict=True);model.eval()
        extracted={}
        for pop in POPS:
            for voice in p.VOICES:
                enc=tests[pop][voice]; h,g,sc,ld=extract(model,enc,device)
                nd=max(abs(sc[k]-ref['outcomes'][arm][pop][voice][k]) for k in sc);assert nd<=diag.TOL,(arm,pop,voice,nd)
                checks.append(dict(arm=arm,population=pop,voice=voice,max_logit_delta=ld,max_native_delta=nd))
                extracted[pop,voice]=(h,g)
                for site in SITES:sens.append(dict(arm=arm,population=pop,voice=voice,site=site,**sensitivity(h[site])))
        for sv in p.VOICES:
            for ss in ('bare','rich'):
                h,g=extracted['unseen_roles_'+ss,sv]; ys=targets(g)
                # Within-source shuffled labels preserve quartet balance and candidate set.
                rng=np.random.default_rng(20261125); perm=np.argsort(rng.random((len(g),4)),axis=1)
                for site in SITES:
                    for task,y in ys.items():
                        k=4 if task=='binding_rank' else 2
                        x=features(h[site],g,task); f=ridge_fit(x,y,k)
                        shuffled=np.take_along_axis(y.reshape(-1,4),perm,axis=1).reshape(-1)
                        control=ridge_fit(x,shuffled,k)
                        for tv in p.VOICES:
                            for ts in ('bare','rich'):
                                hh,gg=extracted['unseen_quartets_'+ts,tv];yy=targets(gg)[task];xx=features(hh[site],gg,task)
                                rows.append(dict(arm=arm,site=site,task=task,fit_voice=sv,test_voice=tv,fit_surface=ss,test_surface=ts,accuracy=score(f,xx,yy),shuffled_accuracy=score(control,xx,yy),chance=1/k))
        print('LOCALISED',seed,step,arm,flush=True)
    p.save_once(out,dict(seed=seed,step=step,protocol_digest=m['protocol_digest'],device=str(device),rows=rows,sensitivity=sens,checks=checks));print('SEALED',seed,step,flush=True)

def summary():
    m=read(MANIFEST);allrows=[];allsens=[];checks=[]
    for seed in r.SEEDS:
        for step in r.MILESTONES:
            obj=read(OUT/'per_checkpoint'/f'seed{seed}_step{step}.json');assert obj['protocol_digest']==m['protocol_digest'];assert (obj['seed'],obj['step'])==(seed,step)
            assert len(obj['rows'])==4*11*3*16 and len(obj['sensitivity'])==4*4*2*11
            for src,dst in ((obj['rows'],allrows),(obj['sensitivity'],allsens),(obj['checks'],checks)):dst.extend(dict(seed=seed,step=step,**x) for x in src)
    for name,rows in [('probes.csv',allrows),('sensitivity.csv',allsens)]:
        with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',n_probe_rows=len(allrows),n_sensitivity_rows=len(allsens),max_native_delta=max(x['max_native_delta'] for x in checks),max_logit_delta=max(x['max_logit_delta'] for x in checks)))
    print('SUMMARY PASS',len(allrows),len(allsens),flush=True)

def smoke():
    g=np.array([[5,2,9,4],[9,4,2,5]]);y=targets(g);assert list(y['binding_rank'])==[2,0,3,1,3,1,0,2]
    x=np.eye(4)[np.tile(np.arange(4),20)];labels=np.tile(np.arange(4),20);assert score(ridge_fit(x,labels,4),x,labels)==1
    h=np.zeros((2,4,96));h[:,2:]=1;assert sensitivity(h)['role_rms']==0 and sensitivity(h)['event_rms']==1
    model=p.old.Model(12).eval();ids=torch.randint(0,12,(8,9));gold=torch.tensor([1,2,3,4,5,6,7,8]);choices=gold.reshape(-1,4).repeat_interleave(4,0);hs,_,_,_=extract(model,(ids,torch.full((8,),8),gold,choices),torch.device('cpu'));assert list(hs)==SITES
    print('SMOKE PASS: labels, ridge, sensitivity, native forward',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));a.add_argument('--seed',type=int);a.add_argument('--step',type=int);a.add_argument('--device',choices=('cpu','mps'),default='mps');v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.seed,v.step,torch.device(v.device))
if __name__=='__main__':main()
