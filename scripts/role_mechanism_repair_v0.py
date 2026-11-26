#!/usr/bin/env python3
"""Frozen selected-cohort repair pilot; no changes to replication or discovery."""
import argparse, collections, hashlib, itertools, json, math
from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/role_mechanism_repair_v0'; WEIGHTS=ROOT/'results/role_mechanism_repair_v0'
SCRIPT=Path(__file__).resolve(); PROTOCOL=ROOT/'paper/ROLE_MECHANISM_REPAIR_V0_PROTOCOL.md'; LAUNCH=ROOT/'scripts/run_role_mechanism_repair_v0.sh'; MANIFEST=OUT/'study_manifest.json'
STUDY='ROLE-MECHANISM-REPAIR-v0'; SEEDS=(27103,27107,27109,27114); START=51200
STEPS=(0,400,800,1600,3200,6400); LAMBDA=4.; LR=.002; WD=.01
ARMS={'frozen':('none',1.),'full_ce':('full',1.),'random_ce':('random',1.),'target_ce':('target',1.),'full_weighted':('full',LAMBDA),'random_weighted':('random',LAMBDA),'target_weighted':('target',LAMBDA)}
HEADS=tuple(f'L{l}.head{h}' for l in range(3) for h in range(4)); MLPS=tuple(f'L{l}.mlp' for l in range(3))

def parents():
    import role_mechanism_discovery_v1 as d
    return d,d.rep,d.p,d.r

def read(path):return json.loads(path.read_text())
def cp(seed,arm,step):return WEIGHTS/f'seed{seed}_{arm}_step{step}.pt'
def rp(seed,arm,step):return OUT/'per_arm'/f'seed{seed}_{arm}_step{step}.json'

def schedule(seed,train,cells,end=START+STEPS[-1]):
    rng=Random(seed+16131013); perms={};counts=collections.Counter()
    for ix in train:
        cc=list(cells);rng.shuffle(cc);perms[ix]=cc
    for step in range(1,end+1):
        chosen=tuple(rng.sample(train,8));events=[]
        for ix in chosen:
            k=counts[ix];fi,mi=perms[ix][k%45];counts[ix]+=1;events.append((ix,fi,mi,k))
        yield step,chosen,tuple(events)

def random_match(target,seed):
    target=set(target);nh=len(target&set(HEADS));nm=len(target&set(MLPS));assert nh+nm==len(target) and nh+nm>0
    options=[tuple(sorted(h+m)) for h in itertools.combinations(HEADS,nh) for m in itertools.combinations(MLPS,nm)]
    overlap=min(len(set(o)&target) for o in options);eligible=[o for o in options if len(set(o)&target)==overlap]
    return Random(seed+272901).choice(eligible)

def masks(model,sites):
    params=dict(model.named_parameters());out={}
    def mark(name,sl=None):
        a=out.setdefault(name,torch.zeros_like(params[name],dtype=torch.bool))
        if sl is None:a.fill_(True)
        else:a[sl]=True
    for s in sites:
        l=int(s[1]);pre=f'blocks.{l}.'
        if s.endswith('.mlp'):
            for k in params:
                if k.startswith(pre+'mlp.'):mark(k)
        else:
            assert s in HEADS;h=int(s[-1]);att=model.blocks[l].attn;w=att.embed_dim;dim=w//att.num_heads;lo=h*dim;hi=(h+1)*dim
            for offset in (0,w,2*w):
                mark(pre+'attn.in_proj_weight',slice(offset+lo,offset+hi));mark(pre+'attn.in_proj_bias',slice(offset+lo,offset+hi))
            mark(pre+'attn.out_proj.weight',(slice(None),slice(lo,hi)))
    assert out and all(m.any() for m in out.values());return out

def count(m):return sum(int(x.sum().item()) for x in m.values())

def loss_parts(logits,y):
    assert len(y)%4==0
    mate=y.reshape(-1,4)[:,[1,0,3,2]].reshape(-1);ix=torch.arange(len(y),device=y.device)
    pair=torch.stack([logits[ix,y],logits[ix,mate]],1);pair_lse=pair.logsumexp(1)
    event=logits.logsumexp(1)-pair_lse;role=pair_lse-logits[ix,y]
    return event.mean(),role.mean()

class Trainer:
    def __init__(self,model,mode,sites,lam):
        self.model=model;self.lam=lam;self.partial={};self.selected=0
        m=masks(model,sites) if mode in ('target','random') else None
        self.trainable=[]
        for name,param in model.named_parameters():
            active=mode=='full' or (m is not None and name in m)
            param.requires_grad_(active)
            if active:
                self.trainable.append(param);self.selected+=param.numel() if m is None else int(m[name].sum().item())
                if m is not None and not bool(m[name].all()):self.partial[name]=(param,m[name],param.detach().clone())
        self.opt=torch.optim.AdamW(self.trainable,lr=LR,weight_decay=WD) if self.trainable else None
    def update(self,enc,device):
        self.model.train();a,ro,y,_=enc;y=y.to(device);z=self.model(a.to(device),ro.to(device));ev,role=loss_parts(z,y)
        loss=F.cross_entropy(z,y) if self.lam==1. else ev+self.lam*role
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        self.opt.zero_grad(set_to_none=True);loss.backward()
        for param,mask,_ in self.partial.values():param.grad.masked_fill_(~mask,0.)
        assert all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in self.trainable)
        self.opt.step()
        # AdamW decays even zero-gradient entries; restore exact original frozen values.
        with torch.no_grad():
            for param,mask,initial in self.partial.values():param.copy_(torch.where(mask,param,initial))
        return [float(x.detach()) for x in (loss,ev,role)]
    def audit(self,initial,mode,sites):
        m=masks(self.model,sites) if mode in ('target','random') else {};delta=0.;frozen_max=0.
        for name,param in self.model.named_parameters():
            old=initial[name].cpu();diff=(param.detach().cpu()-old).abs();delta+=float(diff.double().square().sum())
            if mode!='full':
                frozen=diff[~m[name].cpu()] if name in m else diff
                if frozen.numel():frozen_max=max(frozen_max,float(frozen.max()))
        assert frozen_max==0.;return dict(frozen_max=frozen_max,l2_displacement=math.sqrt(delta),trainable_scalars=self.selected)

@torch.inference_mode()
def metrics(model,e,device):
    model.eval();a,ro,y,ch=e;parts=[]
    for i in range(0,len(a),128):parts.append(model(a[i:i+128].to(device),ro[i:i+128].to(device)).double().cpu())
    z=torch.cat(parts);ix=torch.arange(len(y));mate=y.reshape(-1,4)[:,[1,0,3,2]].reshape(-1);lp=z.log_softmax(1);event=(lp[ix,y].exp()+lp[ix,mate].exp());role=torch.sigmoid(z[ix,y]-z[ix,mate]);pr=z.argmax(1);rr=ch.gather(1,z.gather(1,ch).argmax(1,keepdim=True)).flatten();g=y.reshape(-1,4);q=rr.reshape(-1,4)
    collapse=(q[:,0]==q[:,1])&(q[:,2]==q[:,3]); avg=lambda x:float(x.double().mean())
    ev_nll,role_nll=loss_parts(z,y)
    return dict(C_query=avg((q==g).all(1)),joint_full=avg((pr==y).reshape(-1,4).all(1)),accuracy4=avg(rr==y),full_vocab_accuracy=avg(pr==y),event_mass=avg(event),event_accuracy=avg((rr==y)|(rr==mate)),role_probability=avg(role),role_accuracy=avg(z[ix,y]>z[ix,mate]),event_nll=float(ev_nll),role_nll=float(role_nll),both_pairs_collapsed=avg(collapse))

def setup():
    d,rep,p,r=parents();rm,(train,training,vocab,tests,enc)=rep.setup();assert read(rep.MANIFEST)==rm
    ds=read(d.OUT/'analysis_summary.json');dm=read(d.MANIFEST);assert ds['protocol_digest']==dm['protocol_digest'] and ds['n_checkpoints']==66 and ds['original_replication_primary_replicated'] is False
    rs=read(rep.OUT/'analysis_summary.json');assert rs['protocol_digest']==rm['protocol_digest'] and rs['primary_replicated'] is False
    inputs={}
    for seed in SEEDS:
        ck=rep.ckpath(seed,START);prior=rep.repath(seed,START);disc=d.result_path(seed,START);a=read(prior);b=read(disc)
        assert a['protocol_digest']==rm['protocol_digest'] and b['protocol_digest']==dm['protocol_digest'] and (a['seed'],a['step'])==(seed,START) and (b['seed'],b['step'])==(seed,START)
        assert all(a['native'][pop][v]['C_query']==0. for pop in rep.POPS for v in rep.c.VOICES)
        target=tuple(b['sets']['expanded'][0]);random=random_match(target,seed);net=p.old.Model(len(vocab));n=count(masks(net,target));assert n==count(masks(net,random))
        trace=hashlib.sha256();prefix=None
        for row in schedule(seed,train,p.CELLS):
            trace.update(r.line(row))
            if row[0]==START:prefix=trace.hexdigest()
        assert prefix==a['schedule_digest']
        inputs[str(seed)]=dict(weights=p.sha(ck),prior=p.sha(prior),discovery=p.sha(disc),target=list(target),random=list(random),matched_scalars=n,overlap=sorted(set(target)&set(random)),prefix=prefix,extended=trace.hexdigest())
    dependencies=[SCRIPT,PROTOCOL,LAUNCH,Path(d.__file__),Path(rep.__file__),Path(p.__file__),Path(r.__file__),Path(p.old.__file__)]
    protocol=dict(study=STUDY,classification='prospective_selected_cohort_pilot',seeds=list(SEEDS),steps=list(STEPS),arms={k:list(v) for k,v in ARMS.items()},start=START,lr=LR,wd=WD,replication_digest=rm['protocol_digest'],replication_summary_sha=p.sha(rep.OUT/'analysis_summary.json'),discovery_digest=dm['protocol_digest'],discovery_summary_sha=p.sha(d.OUT/'analysis_summary.json'),inputs=inputs,files={str(x.relative_to(ROOT)):p.sha(x) for x in dependencies},primary='paired target_ce minus random_ce on mean native C_query at 6400 additional updates; each seed weighted equally',secondary=['full_weighted - full_ce','target_weighted - target_ce','target_weighted - random_weighted','target_weighted - full_weighted','full_ce - frozen'],target_rule='expanded role set from pre-existing discovery, candidate not verified causal in collapsed models; head slices QKV rows+bias and output columns; MLP full; random minimum overlap matched type counts',schedule='same continued roles8 stream; fresh optimizer for every active arm; no early stopping',eval='two original rich held-out populations and both voices; full-vocabulary and restricted, event and role; reused benchmark not independent dataset')
    return dict(protocol=protocol,protocol_digest=p.digest(protocol)),(train,training,vocab,tests,enc)

def prepare():
    d,rep,p,r=parents();m,_=setup()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def run_one(seed,arm,m,data,device):
    d,rep,p,r=parents();train,training,vocab,tests,enc=data;cfg=m['protocol']['inputs'][str(seed)];source=rep.ckpath(seed,START);assert p.sha(source)==cfg['weights'];base=torch.load(source,map_location='cpu',weights_only=True);assert (base['seed'],base['step'],base['protocol_digest'],base['schedule_digest'])==(seed,START,m['protocol']['replication_digest'],cfg['prefix'])
    mode,lam=ARMS[arm];sites=cfg[mode] if mode in ('random','target') else [];torch.manual_seed(seed);torch.mps.manual_seed(seed);model=p.old.Model(len(vocab)).to(device);model.load_state_dict(base['model']);trainer=Trainer(model,mode,sites,lam)
    current=-1;state=None
    for t in STEPS:
        a,b=cp(seed,arm,t),rp(seed,arm,t)
        if a.exists()!=b.exists():raise RuntimeError(f'Partial checkpoint {seed}/{arm}/{t}; inspect before restart')
        if a.exists():
            assert current==STEPS[STEPS.index(t)-1] if t else current==-1
            report=read(b);assert (report['seed'],report['arm'],report['step'],report['protocol_digest'])==(seed,arm,t,m['protocol_digest']);current=t
    if current==STEPS[-1]:print('ALREADY SEALED',seed,arm,flush=True);return
    if current>=0:
        state=torch.load(cp(seed,arm,current),map_location='cpu',weights_only=True);assert (state['seed'],state['arm'],state['step'],state['protocol_digest'])==(seed,arm,current,m['protocol_digest']);model.load_state_dict(state['model']);
        if trainer.opt is not None:trainer.opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng']);trainer.audit(base['model'],mode,sites)
    trace=hashlib.sha256();losses=[]
    def seal(t):
        native={pop:{v:metrics(model,tests[pop][v],device) for v in rep.c.VOICES} for pop in rep.POPS};audit=trainer.audit(base['model'],mode,sites)
        if mode in ('random','target'):assert audit['trainable_scalars']==cfg['matched_scalars']
        if t==0:
            prior=read(rep.repath(seed,START))['native']
            for pop in rep.POPS:
                for v in rep.c.VOICES:
                    for k in ('C_query','accuracy4','full_vocab_accuracy'):assert abs(native[pop][v][k]-prior[pop][v][k])<1e-6,(seed,pop,v,k)
        causal=None
        if t in (0,STEPS[-1]):
            heads,gains=rep.discover(model,enc[rep.POPS[0]],device);rows=rep.evaluate(model,enc[rep.POPS[1]],heads,device);causal=dict(heads=list(heads),gains=gains,rows=rows)
        ck=dict(seed=seed,arm=arm,step=t,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),model=model.state_dict(),optimizer=trainer.opt.state_dict() if trainer.opt else None,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state())
        report=dict(study=STUDY,seed=seed,arm=arm,step=t,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),native=native,audit=audit,causal=causal,mean_training_loss=[sum(x[i] for x in losses)/len(losses) for i in range(3)] if losses else None)
        r.save_state(cp(seed,arm,t),ck);p.save_once(rp(seed,arm,t),report);losses.clear();print('SEALED',seed,arm,t,flush=True)
    for row in schedule(seed,train,p.CELLS):
        step,_,events=row;trace.update(r.line(row));t=step-START
        if t<0:continue
        if t==0:assert trace.hexdigest()==cfg['prefix']
        if t<=current:
            if t==current:assert trace.hexdigest()==state['schedule_digest']
            continue
        if t>0 and trainer.opt is not None:losses.append(trainer.update(rep.batches(events,training,vocab),device))
        if t in STEPS:seal(t)
    assert trace.hexdigest()==cfg['extended']

def run(seed,arm,all_tasks):
    d,rep,p,r=parents();assert torch.backends.mps.is_available();m,data=setup();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,LAUNCH,MANIFEST):r.committed(f)
    p.require_frozen();device=torch.device('mps')
    for s,a in itertools.product(SEEDS,ARMS) if all_tasks else [(seed,arm)]:run_one(s,a,m,data,device)

def summary():
    d,rep,p,r=parents();m=read(MANIFEST);rows=[];reports={}
    for s,a,t in itertools.product(SEEDS,ARMS,STEPS):
        q=read(rp(s,a,t));assert (q['seed'],q['arm'],q['step'],q['protocol_digest'])==(s,a,t,m['protocol_digest']);assert q['audit']['frozen_max']==0.;reports[s,a,t]=q
        for pop,voices in q['native'].items():
            for v,me in voices.items():rows.append(dict(seed=s,arm=a,step=t,population=pop,voice=v,**me,**q['audit']))
    mean=lambda q,k:sum(v[k] for pop in q['native'].values() for v in pop.values())/4
    compact=[]
    for (s,a,t),q in reports.items():
        item=dict(seed=s,arm=a,step=t,**{k:mean(q,k) for k in next(iter(next(iter(q['native'].values())).values()))},**q['audit'])
        if q['causal']:
            g=q['causal']['rows'];item.update(role_transfer=sum(x['recipient_full'] for x in g if x['condition']=='donor/1')/2,cross_voice_role=sum(x['recipient_full'] for x in g if x['condition']=='cross_voice/1')/2,within_role_joint=sum(x['recipient_joint_full'] for x in g if x['condition']=='within/1')/2)
        else:item.update(role_transfer=None,cross_voice_role=None,within_role_joint=None)
        compact.append(item)
    # All arms must start identically; frozen control must stay unchanged, schedules matched at each step.
    for s in SEEDS:
        for a in ARMS:
            assert reports[s,a,0]['native']==reports[s,'frozen',0]['native'];assert reports[s,'frozen',STEPS[-1]]['native']==reports[s,'frozen',0]['native']
        for t in STEPS:assert len({reports[s,a,t]['schedule_digest'] for a in ARMS})==1
    contrasts={}
    for a,b in [('target_ce','random_ce'),('full_weighted','full_ce'),('target_weighted','target_ce'),('target_weighted','random_weighted'),('target_weighted','full_weighted'),('full_ce','frozen'),('random_weighted','random_ce')]:
        vals={str(s):mean(reports[s,a,STEPS[-1]],'C_query')-mean(reports[s,b,STEPS[-1]],'C_query') for s in SEEDS};contrasts[a+' minus '+b]=dict(per_seed=vals,mean=sum(vals.values())/len(vals),positive=sum(v>0 for v in vals.values()))
    result=dict(study=STUDY,classification='selected_cohort_pilot',protocol_digest=m['protocol_digest'],n_seeds=len(SEEDS),n_arms=len(ARMS),n_reports=len(reports),primary=contrasts['target_ce minus random_ce'],contrasts=contrasts,final=[q for q in compact if q['step']==STEPS[-1]])
    d.w.write_csv(OUT/'rows.csv',rows);d.w.write_csv(OUT/'compact.csv',compact);f=OUT/'analysis_summary.json'
    if f.exists():assert read(f)==result
    else:p.save_once(f,result)
    print('SUMMARY',json.dumps(result),flush=True)

def smoke():
    import copy
    import role_causal_routing_v0 as c
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel();initial=copy.deepcopy(model.state_dict());y=torch.tensor([1,2,3,4]*2);a=torch.tensor([[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,8,7],[0,1,2,3,9,5,6,7],[0,1,2,3,9,5,8,7]]).repeat(2,1);ro=torch.full((8,),7);enc=(a,ro,y,torch.tensor([[1,2,3,4]]*8));dev=torch.device('cpu');z=model(a,ro);e,r=loss_parts(z,y);assert torch.allclose(e+r,F.cross_entropy(z,y),atol=1e-6)
    target=('L0.head2','L1.mlp','L2.head1');random=random_match(target,27103);assert count(masks(model,target))==count(masks(model,random));assert not set(target)&set(random)
    for arm,(mode,lam) in ARMS.items():
        net=copy.deepcopy(model);sites=target if mode=='target' else random if mode=='random' else [];tr=Trainer(net,mode,sites,lam)
        if tr.opt:
            for _ in range(3):tr.update(enc,dev)
        au=tr.audit(initial,mode,sites);assert au['frozen_max']==0.;assert (au['l2_displacement']>0)==(mode!='none');assert math.isfinite(metrics(net,enc,dev)['role_nll'])
    # CPU restart equivalence for a partially masked AdamW arm.
    a1=copy.deepcopy(model);t1=Trainer(a1,'target',target,LAMBDA);t1.update(enc,dev);saved=copy.deepcopy(a1.state_dict());op=copy.deepcopy(t1.opt.state_dict());a2=copy.deepcopy(model);t2=Trainer(a2,'target',target,LAMBDA);a2.load_state_dict(saved);t2.opt.load_state_dict(op);t1.update(enc,dev);t2.update(enc,dev);assert all(torch.equal(x,y) for x,y in zip(a1.state_dict().values(),a2.state_dict().values()))
    print('SMOKE PASS: CE decomposition, seven arms, exact scalar masks, AdamW frozen restoration, matched controls, metrics',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('smoke','prepare','run','run-all','summary'));ap.add_argument('--seed',type=int);ap.add_argument('--arm',choices=tuple(ARMS));a=ap.parse_args()
    if a.phase=='smoke':smoke()
    elif a.phase=='prepare':prepare()
    elif a.phase=='summary':summary()
    else:
        if a.phase=='run':assert a.seed in SEEDS and a.arm in ARMS
        run(a.seed,a.arm,a.phase=='run-all')
if __name__=='__main__':main()
