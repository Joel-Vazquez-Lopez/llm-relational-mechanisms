#!/usr/bin/env python3
"""Crossed initialisation and training-stream seeds; no parent files modified."""
import argparse, collections, csv, hashlib, itertools, json, math, os, subprocess, time
from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve(); PROTOCOL=ROOT/'paper/ROLE_LEARNING_ORIGINS_V0_PROTOCOL.md'; LAUNCH=ROOT/'scripts/run_role_learning_origins_v0.sh'
OUT=ROOT/'reports/role_learning_origins_v0'; WEIGHTS=ROOT/'results/role_learning_origins_v0'; MANIFEST=OUT/'study_manifest.json'
STUDY='ROLE-LEARNING-ORIGINS-v0'; BRANCH='experiments/role-learning-origins-v0'
INIT=tuple(range(38101,38109)); ORDER=tuple(range(39101,39109)); PILOT_INIT=(38001,38002); PILOT_ORDER=(39001,39002)
STEPS=tuple([0]+list(range(200,6401,200))+list(range(6800,12801,400))+list(range(13600,51201,800)))
ARCHIVE=(0,800,1600,3200,6400,12800,25600,51200); GRAD=ARCHIVE
POPS=('train_seen_rich','unseen_roles_rich','unseen_quartets_rich'); N=128

def parents():
    import role_mechanism_replication_v0 as rep
    import role_mechanism_repair_v0 as repair
    return rep,repair,rep.p,rep.r

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def line(x):return (json.dumps(x,separators=(',',':'))+'\n').encode()
def read(path):return json.loads(path.read_text())
def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def schedule(order,train,cells,end):
    rng=Random(order+16131013);perms={};counts=collections.Counter()
    for ix in train:
        cc=list(cells);rng.shuffle(cc);perms[ix]=cc
    for t in range(1,end+1):
        chosen=tuple(rng.sample(train,8));events=[]
        for ix in chosen:
            k=counts[ix];fi,mi=perms[ix][k%len(cells)];counts[ix]+=1;events.append((ix,fi,mi,k))
        yield t,chosen,tuple(events)
def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        a=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def make_model(p,vocab,init,device):
    torch.manual_seed(init)
    model=p.old.Model(len(vocab)).to(device)
    # No order seed enters torch's RNG. Reset stochastic training RNG to fixed seed.
    torch.manual_seed(40001)
    if device.type=='mps':torch.mps.manual_seed(40001)
    return model

def subset(e,idx):
    ii=torch.tensor([4*j+k for j in idx for k in range(4)],dtype=torch.long)
    return tuple(x[ii] for x in e)
def setup():
    rep,repair,p,r=parents();train,held,training,withheld,vocab,parent=p.verified();tests=p.evaluations(train,held,training,withheld,vocab)
    full={pop:(tests[pop]['roles_8'] if pop=='train_seen_rich' else tests[pop]) for pop in POPS};samples={};small={}
    for j,pop in enumerate(POPS):
        n=len(full[pop][p.VOICES[0]][0])//4;idx=sorted(Random(41201+j).sample(range(n),N));small[pop]={v:subset(full[pop][v],idx) for v in p.VOICES}
        for v in p.VOICES:rep.c.audit_pairs(*small[pop][v])
        samples[pop]={'indices':idx,'sha':{v:digest([x.tolist() for x in small[pop][v]]) for v in p.VOICES},'full_sha':{v:digest([x.tolist() for x in full[pop][v]]) for v in p.VOICES}}
    schedules={}
    for o in ORDER:
        h=hashlib.sha256()
        for row in schedule(o,train,p.CELLS,STEPS[-1]):h.update(line(row))
        schedules[str(o)]=h.hexdigest()
    init_hash={str(i):state_hash(make_model(p,vocab,i,torch.device('cpu'))) for i in INIT}
    assert len(set(init_hash.values()))==len(INIT) and len(set(schedules.values()))==len(ORDER)
    deps=[SCRIPT,PROTOCOL,LAUNCH,Path(rep.__file__),Path(repair.__file__),Path(p.__file__),Path(r.__file__),Path(p.old.__file__)]
    protocol=dict(study=STUDY,init=list(INIT),order=list(ORDER),steps=list(STEPS),archive=list(ARCHIVE),gradient_steps=list(GRAD),parent_digest=parent['protocol_digest'],samples=samples,schedule_hashes=schedules,initial_hashes=init_hash,lr=p.old.LR,wd=p.old.WD,files={str(x.relative_to(ROOT)):sha(x) for x in deps},primary='balanced two-way descriptive sums of squares of final mean full heldout C_query; init, stream, residual interaction (not separately identified from run noise)',success='C_query >= .95 in each heldout population and voice on full final evaluation',evaluation='fixed 128 source quartets per population at every checkpoint, full populations at final; previously used benchmark, not independent linguistic replication',gradient='fixed train-seen batch first 16 sampled quartets per voice, eval mode, separate event and role gradient norms and cosine by parameter group; diagnostic only')
    return dict(protocol=protocol,protocol_digest=digest(protocol)),(train,training,vocab,full,small)

def require_branch():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!=BRANCH:raise RuntimeError(f'Expected {BRANCH}, got {branch}')
def prepare():
    require_branch();m,_=setup()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:atomic_json(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def grad_stats(model,e,device,loss_parts):
    model.eval();a,ro,y,_=e;z=model(a.to(device),ro.to(device));ev,role=loss_parts(z,y.to(device));params=list(model.named_parameters());pp=[v for _,v in params]
    g1=torch.autograd.grad(ev,pp,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(role,pp,allow_unused=True)
    groups=collections.defaultdict(lambda:[0.,0.,0.])
    for (name,param),x,y in zip(params,g1,g2):
        x=torch.zeros_like(param) if x is None else x;y=torch.zeros_like(param) if y is None else y
        x=x.detach().cpu().double();y=y.detach().cpu().double();vals=[float(x.square().sum()),float(y.square().sum()),float((x*y).sum())]
        group='.'.join(name.split('.')[:3]) if name.startswith('blocks.') else name.split('.')[0]
        for key in ('all',group):
            for k,v in enumerate(vals):groups[key][k]+=v
    return {k:dict(event_norm=math.sqrt(a),role_norm=math.sqrt(b),cosine=c/math.sqrt(a*b) if a*b>0 else None) for k,(a,b,c) in groups.items()}

def run_one(i,o,m,data,device,pilot=False,stop=None):
    rep,repair,p,r=parents();train,training,vocab,full,small=data;end=400 if pilot else STEPS[-1];steps=(0,200,400) if pilot else STEPS
    out=OUT/'engineering_pilot' if pilot else OUT;weights=WEIGHTS/'engineering_pilot' if pilot else WEIGHTS
    run=f'i{i}_d{o}';path=weights/run/'latest.pt';folder=out/'per_run'/run;model=make_model(p,vocab,i,device);initial=state_hash(model)
    if not pilot:assert initial==m['protocol']['initial_hashes'][str(i)]
    opt=torch.optim.AdamW(model.parameters(),lr=p.old.LR,weight_decay=p.old.WD);trace=hashlib.sha256();current=-1;state=None;losses=[];started=time.monotonic()
    def report_path(t):return folder/f'step{t}.json'
    def restore_report(st):
        t=st['step'];f=report_path(t)
        if f.exists():assert read(f)==st['report']
        else:atomic_json(f,st['report'])
        if t in ARCHIVE and not pilot:
            ar=weights/run/f'step{t}.pt'
            if not ar.exists():atomic_torch(ar,st)
            else:
                old=torch.load(ar,map_location='cpu',weights_only=True);assert old['report']==st['report']
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True);assert (state['init'],state['order'],state['protocol_digest'],state['pilot'])==(i,o,m['protocol_digest'],pilot);current=state['step'];assert current in steps
        restore_report(state);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng'])
        if device.type=='mps':torch.mps.set_rng_state(state['mps_rng'])
        for t in steps:
            assert report_path(t).exists()==(t<=current),('report continuity',run,t,current)
        if current>=end or (stop is not None and current>=stop):print('ALREADY SEALED',run,current,flush=True);return
    elif folder.exists() and list(folder.glob('step*.json')):raise RuntimeError('Reports without latest state: '+run)
    def seal(t):
        native={pop:{v:repair.metrics(model,small[pop][v],device) for v in p.VOICES} for pop in POPS}
        final={pop:{v:repair.metrics(model,full[pop][v],device) for v in p.VOICES} for pop in POPS} if t==end and not pilot else None
        grad=None
        if t in GRAD or (pilot and t==end):
            grad={v:grad_stats(model,tuple(x[:64] for x in small['train_seen_rich'][v]),device,repair.loss_parts) for v in p.VOICES}
        q=dict(study=STUDY,init=i,order=o,step=t,pilot=pilot,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),initial_hash=initial,native=native,full_final=final,gradient=grad,mean_training_loss=sum(losses)/len(losses) if losses else None)
        st=dict(init=i,order=o,step=t,pilot=pilot,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,report=q)
        atomic_torch(path,st);restore_report(st);losses.clear();print('SEALED',run,t,'session_seconds',round(time.monotonic()-started,1),flush=True)
    if current<0:seal(0)
    elif current==0:assert trace.hexdigest()==state['schedule_digest']
    for row in schedule(o,train,p.CELLS,end):
        t,_,events=row;trace.update(line(row))
        if t<=current:
            if t==current:assert trace.hexdigest()==state['schedule_digest']
            continue
        loss=p.old.one_update(model,opt,rep.batches(events,training,vocab),device);assert math.isfinite(loss);losses.append(loss)
        if t in steps:seal(t)
        if stop is not None and t==stop:return
    if not pilot:assert trace.hexdigest()==m['protocol']['schedule_hashes'][str(o)]

def run(pilot=False,init=None,order=None,stop=None):
    require_branch();rep,repair,p,r=parents();m,data=setup();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,LAUNCH,MANIFEST):r.committed(f)
    p.require_frozen();assert torch.backends.mps.is_available(),'MPS required on Mac';dev=torch.device('mps')
    ii,oo=(PILOT_INIT,PILOT_ORDER) if pilot else (INIT,ORDER)
    assert (init is None)==(order is None)
    tasks=list(itertools.product(ii,oo)) if init is None else [(init,order)]
    assert all(i in ii and o in oo for i,o in tasks)
    assert stop is None or (pilot and stop==200)
    for i,o in tasks:run_one(i,o,m,data,dev,pilot,stop)

def decompose(grid):
    n=len(grid);k=len(grid[0]);grand=sum(map(sum,grid))/(n*k);rows=[sum(x)/k for x in grid];cols=[sum(grid[i][j] for i in range(n))/n for j in range(k)]
    ss_i=k*sum((v-grand)**2 for v in rows);ss_o=n*sum((v-grand)**2 for v in cols);ss_r=sum((grid[i][j]-rows[i]-cols[j]+grand)**2 for i in range(n) for j in range(k));total=sum((v-grand)**2 for row in grid for v in row)
    assert math.isclose(ss_i+ss_o+ss_r,total,abs_tol=1e-10)
    return dict(grand_mean=grand,init_means=rows,order_means=cols,ss=dict(init=ss_i,order=ss_o,interaction_residual=ss_r,total=total),fraction=dict(init=ss_i/total,order=ss_o/total,interaction_residual=ss_r/total) if total else None)
def summary():
    m=read(MANIFEST);rows=[];final={};schedules={};initials={}
    for i,o in itertools.product(INIT,ORDER):
        run=f'i{i}_d{o}'
        for t in STEPS:
            q=read(OUT/'per_run'/run/f'step{t}.json');assert (q['init'],q['order'],q['step'],q['protocol_digest'],q['pilot'])==(i,o,t,m['protocol_digest'],False)
            assert q['initial_hash']==m['protocol']['initial_hashes'][str(i)]
            key=(o,t);assert schedules.setdefault(key,q['schedule_digest'])==q['schedule_digest']
            for pop,voices in q['native'].items():
                for v,me in voices.items():rows.append(dict(init=i,order=o,step=t,population=pop,voice=v,scope='sample',**me))
            if t==STEPS[-1]:
                assert q['schedule_digest']==m['protocol']['schedule_hashes'][str(o)];f=q['full_final'];vals=[me['C_query'] for pop in POPS[1:] for me in f[pop].values()];final[i,o]=(sum(vals)/len(vals),min(vals)>=.95)
                for pop,voices in f.items():
                    for v,me in voices.items():rows.append(dict(init=i,order=o,step=t,population=pop,voice=v,scope='full',**me))
    grid=[[final[i,o][0] for o in ORDER] for i in INIT];success=[[final[i,o][1] for o in ORDER] for i in INIT]
    result=dict(study=STUDY,protocol_digest=m['protocol_digest'],n_models=len(final),n_reports=len(final)*len(STEPS),init=list(INIT),order=list(ORDER),final_joint=grid,success=success,n_success=sum(map(sum,success)),primary=decompose(grid))
    with (OUT/'trajectories.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(OUT/'analysis_summary.json',result);print(json.dumps(result,indent=2),flush=True)
def smoke():
    torch.set_num_threads(1);train=list(range(20));cells=list(itertools.product(range(5),range(9)));a=list(schedule(123,train,cells,500));b=list(schedule(124,train,cells,500));assert a==list(schedule(123,train,cells,500)) and a!=b
    counts=collections.Counter();seen=collections.defaultdict(set)
    for t,chosen,events in a:
        assert len(set(chosen))==8
        for ix,fi,mi,k in events:
            assert k==counts[ix];counts[ix]+=1
            if k<45:seen[ix].add((fi,mi))
    assert all(len(seen[ix])==45 for ix in train)
    assert decompose([[0.,0.],[1.,1.]])['fraction']['init']==1.
    assert decompose([[0.,1.],[0.,1.]])['fraction']['order']==1.
    assert decompose([[0.,1.],[1.,0.]])['fraction']['interaction_residual']==1.
    e=tuple(torch.arange(16) for _ in range(4));assert subset(e,[1,3])[0].tolist()==[4,5,6,7,12,13,14,15]
    model=torch.nn.Linear(4,4);opt=torch.optim.AdamW(model.parameters());x=torch.eye(4);y=torch.arange(4);F.cross_entropy(model(x),y).backward();opt.step()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'state.pt';atomic_torch(p,{'model':model.state_dict(),'optimizer':opt.state_dict()});s=torch.load(p,weights_only=True);assert all(torch.equal(v,s['model'][k]) for k,v in model.state_dict().items());q=Path(td)/'report.json';atomic_json(q,{'ok':True});assert read(q)=={'ok':True}
    print('SMOKE PASS: schedule independence, cell coverage, balanced SS, grouped subset, atomic state/report',flush=True)
def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','pilot','run','summary'));a.add_argument('--init',type=int);a.add_argument('--order',type=int);a.add_argument('--stop',type=int);v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.phase=='pilot',v.init,v.order,v.stop)
if __name__=='__main__':main()
