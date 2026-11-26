#!/usr/bin/env python3
"""Prospective continuation-schedule study conditional on one selected checkpoint."""
import argparse, hashlib, itertools, json, subprocess, math
from pathlib import Path
from random import Random
import torch
import role_mechanism_repair_v0 as repair
ROOT=repair.ROOT; OUT=ROOT/'reports/role_repair_specificity_v1'; WEIGHTS=ROOT/'results/role_repair_specificity_v1'; MANIFEST=OUT/'study_manifest.json'
SCRIPT=Path(__file__).resolve(); PROTOCOL=ROOT/'paper/ROLE_REPAIR_SPECIFICITY_V1_PROTOCOL.md'; LAUNCH=ROOT/'scripts/run_role_repair_specificity_v1.sh'
STUDY='ROLE-REPAIR-SPECIFICITY-v1'; SEED=27114; RUNS=tuple(range(28101,28109)); STEPS=(0,400,800,1600,3200,6400); BRANCH='experiments/role-repair-specificity-v1'
ARMS={f'{mode}_{loss}':(mode,lam) for mode in ('full','target','random','layer') for loss,lam in [('ce',1.),('weighted',4.)]}
TARGET=('L0.head0','L0.mlp','L2.mlp')
def read(p):return json.loads(p.read_text())
def cp(run,arm,t):return WEIGHTS/f'run{run}_{arm}_step{t}.pt'
def rp(run,arm,t):return OUT/'per_arm'/f'run{run}_{arm}_step{t}.json'
def stream(run,train,cells):
    # Independent resampling of the same training population; no held-out examples.
    import collections
    rng=Random(run+16131013);counts=collections.Counter();perms={}
    for ix in train:
        cc=list(cells);rng.shuffle(cc);perms[ix]=cc
    for t in range(1,STEPS[-1]+1):
        chosen=tuple(rng.sample(train,8));events=[]
        for ix in chosen:
            k=counts[ix];fi,mi=perms[ix][k%45];counts[ix]+=1;events.append((ix,fi,mi,k))
        yield t,chosen,tuple(events)
def controls():
    # Type-matched, minimum overlap; unique across eight runs; layer controls balanced 3/3/2.
    options=sorted(tuple(sorted((h,*ms))) for h in repair.HEADS for ms in itertools.combinations(repair.MLPS,2) if h!='L0.head0' and set(ms)!=set(TARGET[1:]))
    rng=Random(281900);rng.shuffle(options);layers=[1,2,3,1,2,3,1,2];rng.shuffle(layers)
    return {str(run):{'target':list(TARGET),'random':list(options[i]),'layer':[f'L0.head{layers[i]}',*TARGET[1:]]} for i,run in enumerate(RUNS)}
def setup():
    d,rep,p,r=repair.parents(); rm,(train,training,vocab,tests,enc)=rep.setup();assert read(rep.MANIFEST)==rm
    pm=read(repair.MANIFEST);ps=read(repair.OUT/'analysis_summary.json');assert ps['protocol_digest']==pm['protocol_digest'] and ps['n_reports']==168
    source=rep.ckpath(SEED,repair.START);prior=rep.repath(SEED,repair.START);cfg=pm['protocol']['inputs'][str(SEED)]
    assert p.sha(source)==cfg['weights'] and p.sha(prior)==cfg['prior'] and tuple(cfg['target'])==TARGET
    assert all(read(prior)['native'][pop][v]['C_query']==0 for pop in rep.POPS for v in rep.c.VOICES)
    cs=controls();net=p.old.Model(len(vocab));n=repair.count(repair.masks(net,TARGET))
    for run in RUNS:
        item=cs[str(run)];assert all(repair.count(repair.masks(net,item[k]))==n for k in ('target','random','layer'));trace=hashlib.sha256()
        for row in stream(run,train,p.CELLS):trace.update(r.line(row))
        item['schedule']=trace.hexdigest()
    deps=[SCRIPT,PROTOCOL,LAUNCH,Path(repair.__file__),Path(d.__file__),Path(rep.__file__),Path(p.__file__),Path(r.__file__),Path(p.old.__file__)]
    protocol=dict(study=STUDY,classification='prospective_followup_conditional_on_selected_checkpoint',source_seed=SEED,source_step=repair.START,source_sha=p.sha(source),prior_sha=p.sha(prior),repair_digest=pm['protocol_digest'],repair_summary_sha=p.sha(repair.OUT/'analysis_summary.json'),replication_digest=rm['protocol_digest'],runs=list(RUNS),steps=list(STEPS),arms={k:list(v) for k,v in ARMS.items()},controls=cs,matched_scalars=n,files={str(x.relative_to(ROOT)):p.sha(x) for x in deps},primary='target_weighted minus layer_weighted mean restricted C_query at 1600 updates; paired by schedule',secondary=['target_weighted minus random_weighted','target_weighted minus target_ce','target_weighted minus full_weighted','target_weighted minus full_ce','target_ce minus layer_ce','target_ce minus random_ce'],lr=repair.LR,wd=repair.WD)
    return dict(protocol=protocol,protocol_digest=p.digest(protocol)),(train,training,vocab,tests,enc)
def prepare():
    m,_=setup();_,_,p,_=repair.parents()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)
def run_one(run,arm,m,data,device):
    d,rep,p,r=repair.parents();train,training,vocab,tests,enc=data;cfg=m['protocol'];source=rep.ckpath(SEED,repair.START);assert p.sha(source)==cfg['source_sha'];base=torch.load(source,map_location='cpu',weights_only=True);assert (base['seed'],base['step'],base['protocol_digest'])==(SEED,repair.START,cfg['replication_digest'])
    mode,lam=ARMS[arm];sites=[] if mode=='full' else cfg['controls'][str(run)][mode];tm='full' if mode=='full' else 'target';torch.manual_seed(run);torch.mps.manual_seed(run);model=p.old.Model(len(vocab)).to(device);model.load_state_dict(base['model']);trainer=repair.Trainer(model,tm,sites,lam);current=-1;state=None
    for t in STEPS:
        a,b=cp(run,arm,t),rp(run,arm,t)
        if a.exists()!=b.exists():raise RuntimeError(f'Partial checkpoint {run}/{arm}/{t}; inspect before restart')
        if a.exists():
            assert current==(STEPS[STEPS.index(t)-1] if t else -1);q=read(b);assert (q['run'],q['arm'],q['step'],q['protocol_digest'])==(run,arm,t,m['protocol_digest']);current=t
    if current==STEPS[-1]:print('ALREADY SEALED',run,arm,flush=True);return
    if current>=0:
        state=torch.load(cp(run,arm,current),map_location='cpu',weights_only=True);assert (state['run'],state['arm'],state['step'],state['protocol_digest'])==(run,arm,current,m['protocol_digest']);model.load_state_dict(state['model']);trainer.opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng']);trainer.audit(base['model'],tm,sites)
    trace=hashlib.sha256();losses=[]
    def seal(t):
        native={pop:{v:repair.metrics(model,tests[pop][v],device) for v in rep.c.VOICES} for pop in rep.POPS};audit=trainer.audit(base['model'],tm,sites)
        if mode!='full':assert audit['trainable_scalars']==cfg['matched_scalars']
        if t==0:
            prior=read(rep.repath(SEED,repair.START))['native']
            for pop in rep.POPS:
                for v in rep.c.VOICES:
                    for k in ('C_query','accuracy4','full_vocab_accuracy'):assert abs(native[pop][v][k]-prior[pop][v][k])<1e-6
        causal=None
        if t==STEPS[-1]:
            heads,gains=rep.discover(model,enc[rep.POPS[0]],device);rows=rep.evaluate(model,enc[rep.POPS[1]],heads,device);causal=dict(heads=list(heads),gains=gains,rows=rows)
        common=dict(run=run,arm=arm,step=t,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest())
        ck=dict(**common,model=model.state_dict(),optimizer=trainer.opt.state_dict(),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state());q=dict(**common,native=native,audit=audit,causal=causal,mean_training_loss=[sum(x[i] for x in losses)/len(losses) for i in range(3)] if losses else None)
        r.save_state(cp(run,arm,t),ck);p.save_once(rp(run,arm,t),q);losses.clear();print('SEALED',run,arm,t,flush=True)
    if current==-1:seal(0)
    elif current==0:assert state['schedule_digest']==trace.hexdigest()
    for row in stream(run,train,p.CELLS):
        t,_,events=row;trace.update(r.line(row))
        if t<=current:
            if t==current:assert trace.hexdigest()==state['schedule_digest']
            continue
        losses.append(trainer.update(rep.batches(events,training,vocab),device))
        if t in STEPS:seal(t)
    assert trace.hexdigest()==cfg['controls'][str(run)]['schedule']
def run(run_id,arm,all_tasks):
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH
    d,rep,p,r=repair.parents();assert torch.backends.mps.is_available();m,data=setup();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,LAUNCH,MANIFEST):r.committed(f)
    p.require_frozen()
    for s,a in itertools.product(RUNS,ARMS) if all_tasks else [(run_id,arm)]:run_one(s,a,m,data,torch.device('mps'))
def summary():
    d,rep,p,r=repair.parents();m=read(MANIFEST);reports={};compact=[];rows=[]
    for run,arm,t in itertools.product(RUNS,ARMS,STEPS):
        q=read(rp(run,arm,t));assert (q['run'],q['arm'],q['step'],q['protocol_digest'])==(run,arm,t,m['protocol_digest']);assert q['audit']['frozen_max']==0;reports[run,arm,t]=q
        vals=[v for pop in q['native'].values() for v in pop.values()];item=dict(run=run,arm=arm,step=t,**{k:sum(v[k] for v in vals)/4 for k in vals[0]},**q['audit']);compact.append(item)
        for pop,vs in q['native'].items():
            for voice,v in vs.items():rows.append(dict(run=run,arm=arm,step=t,population=pop,voice=voice,**v))
    for run in RUNS:
        for t in STEPS:assert len({reports[run,a,t]['schedule_digest'] for a in ARMS})==1
        for a in ARMS:assert reports[run,a,0]['native']==reports[run,'full_ce',0]['native']
    idx={(q['run'],q['arm'],q['step']):q for q in compact};pairs=[('target_weighted','layer_weighted'),('target_weighted','random_weighted'),('target_weighted','target_ce'),('target_weighted','full_weighted'),('target_weighted','full_ce'),('target_ce','layer_ce'),('target_ce','random_ce')];contrasts={}
    for t in (1600,6400):
        for a,b in pairs:
            v=[idx[s,a,t]['C_query']-idx[s,b,t]['C_query'] for s in RUNS];obs=sum(v)/len(v);perm=[sum(x*y for x,y in zip(v,sign))/len(v) for sign in itertools.product((-1,1),repeat=len(v))];contrasts[f'{t}:{a}-{b}']=dict(per_run=dict(zip(map(str,RUNS),v)),mean=obs,positive=sum(x>0 for x in v),two_sided_sign_flip=sum(abs(x)>=abs(obs)-1e-12 for x in perm)/len(perm))
    recovery={a:{str(s):next((t for t in STEPS if idx[s,a,t]['C_query']>=.95),None) for s in RUNS} for a in ARMS};result=dict(study=STUDY,protocol_digest=m['protocol_digest'],n_runs=len(RUNS),n_arms=len(ARMS),n_reports=len(reports),primary=contrasts['1600:target_weighted-layer_weighted'],contrasts=contrasts,first_observed_recovery=recovery)
    d.w.write_csv(OUT/'rows.csv',rows);d.w.write_csv(OUT/'compact.csv',compact);f=OUT/'analysis_summary.json'
    if f.exists():assert read(f)==result
    else:p.save_once(f,result)
    print('SUMMARY',json.dumps(result),flush=True)
def smoke():
    import copy
    from torch import nn
    torch.set_num_threads(1);torch.manual_seed(1)
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.attn=nn.MultiheadAttention(16,4,batch_first=True);self.mlp=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        def forward(self,x):return x+self.attn(x,x,x,need_weights=False)[0]+self.mlp(x)
    class Model(nn.Module):
        def __init__(self):super().__init__();self.embed=nn.Embedding(20,16);self.blocks=nn.ModuleList([Block() for _ in range(3)]);self.out=nn.Linear(16,20)
        def forward(self,a,ro):
            x=self.embed(a)
            for b in self.blocks:x=b(x)
            return self.out(x[torch.arange(len(a)),ro])
    model=Model();initial=copy.deepcopy(model.state_dict());cs=controls();assert len({tuple(v['random']) for v in cs.values()})==8;assert len({tuple(v['layer']) for v in cs.values()})==3
    a=torch.randint(0,20,(8,6));y=torch.tensor([1,2,3,4]*2);enc=(a,torch.full((8,),5),y,torch.tensor([[1,2,3,4]]*8));dev=torch.device('cpu');n=repair.count(repair.masks(model,TARGET))
    for mode,lam in ARMS.values():
        net=copy.deepcopy(model);sites=[] if mode=='full' else cs[str(RUNS[0])][mode];tm='full' if mode=='full' else 'target';tr=repair.Trainer(net,tm,sites,lam);tr.update(enc,dev);audit=tr.audit(initial,tm,sites);assert audit['frozen_max']==0 and (mode=='full' or audit['trainable_scalars']==n);assert math.isfinite(repair.metrics(net,enc,dev)['role_nll'])
    cells=list(itertools.product(range(5),range(9)));train=list(range(20));a=list(stream(RUNS[0],train,cells));assert a==list(stream(RUNS[0],train,cells)) and a!=list(stream(RUNS[1],train,cells));assert len(a)==6400
    print('SMOKE PASS: eight arms, matched masks, frozen audit, CPU metrics, deterministic distinct schedules',flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('smoke','prepare','run','run-all','summary'));ap.add_argument('--run',type=int);ap.add_argument('--arm',choices=tuple(ARMS));a=ap.parse_args()
    if a.phase=='smoke':smoke()
    elif a.phase=='prepare':prepare()
    elif a.phase=='summary':summary()
    else:
        if a.phase=='run':assert a.run in RUNS and a.arm in ARMS
        run(a.run,a.arm,a.phase=='run-all')
if __name__=='__main__':main()
