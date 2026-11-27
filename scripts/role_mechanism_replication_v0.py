#!/usr/bin/env python3
"""Prospective independent-seed replication, one roles_8 model per seed."""
import argparse, collections, hashlib, json, math, os
from pathlib import Path
from random import Random
import torch
import role_binding_localisation_v0 as loc
w=loc.w;c=loc.c
import role_diversity_complex_v0 as p
import role_diversity_recovery_v0 as r
ROOT=p.ROOT; OUT=ROOT/'reports/role_mechanism_replication_v0'; WEIGHTS=ROOT/'results/role_mechanism_replication_v0'
SCRIPT=Path(__file__).resolve(); PROTOCOL=ROOT/'paper/ROLE_MECHANISM_REPLICATION_V0_PROTOCOL.md'; LAUNCH=ROOT/'scripts/run_role_mechanism_replication_v0.sh'; MANIFEST=OUT/'study_manifest.json'
SEEDS=tuple(range(27101,27117)); STEPS=(6400,12800,25600,51200); N=128; CHUNK=128; STUDY='ROLE-MECHANISM-REPLICATION-v0'
SITES=tuple(f'L{l}.head{h}' for l in range(3) for h in range(4)); POPS=('unseen_roles_rich','unseen_quartets_rich')

def read(path):return json.loads(path.read_text())
def inputs():return p.verified()
def schedule(seed,train):
    assert seed in SEEDS
    rng=Random(seed+16131013); perms={};counts=collections.Counter()
    for ix in train:
        cells=list(p.CELLS);rng.shuffle(cells);perms[ix]=cells
    for step in range(1,STEPS[-1]+1):
        chosen=tuple(rng.sample(train,8)); events=[]
        for ix in chosen:
            k=counts[ix];fi,mi=perms[ix][k%45];counts[ix]+=1;events.append((ix,fi,mi,k))
        yield step,chosen,tuple(events)

def batches(events,training,vocab):
    sources=[]
    for ix,fi,mi,k in events:
        ev,eligible,_=training[ix]
        sources.extend(p.source(ev,eligible[k%8],v,fi,mi,'replication_train') for v in p.VOICES)
    return p.old.encode_sources(sources,vocab,check=False)

def setup():
    train,held,training,withheld,vocab,parent=inputs(); tests=p.evaluations(train,held,training,withheld,vocab)
    # Discovery uses training-quartet identities but never-trained assignments; test uses withheld quartet identities.
    previous=loc.read(loc.MANIFEST); assert loc.read(loc.OUT/'analysis_summary.json')['protocol_digest']==previous['protocol_digest']
    excludes={pop:set() for pop in POPS}
    for pop in POPS:
        for v in c.VOICES:
            key=f'{pop}/{v}'
            excludes[pop].update(loc.read(c.MANIFEST)['protocol']['samples'][key]['source_indices'])
            excludes[pop].update(loc.read(w.MANIFEST)['protocol']['samples'][key]['indices'])
    excludes[POPS[1]].update(loc.read(loc.b.MANIFEST)['protocol']['indices']);excludes[POPS[1]].update(previous['protocol']['indices'])
    samples={}; enc={}; sep=vocab['<sep>']
    for j,pop in enumerate(POPS):
        full=tests[pop]; n=len(full[c.VOICES[0]][0])//4;idx=sorted(Random(271800+j).sample(sorted(set(range(n))-excludes[pop]),N)); samples[pop]={'indices':idx,'voices':{}};enc[pop]={}
        for v in c.VOICES:
            e=c.subset(full[v],idx);c.audit_pairs(*e); entry={'original':e}
            if j==1:
                names=loc.disjoint_names(e,full[v]); recipient,_=loc.replace(e,names,sep);entry['recipient']=recipient
            enc[pop][v]=entry;samples[pop]['voices'][v]={k:p.digest([t.tolist() for t in a]) for k,a in entry.items()}
        assert torch.equal(enc[pop][c.VOICES[0]]['original'][2],enc[pop][c.VOICES[1]]['original'][2])
        if j==1:assert torch.equal(enc[pop][c.VOICES[0]]['recipient'][2],enc[pop][c.VOICES[1]]['recipient'][2])
    dep=[SCRIPT,PROTOCOL,LAUNCH,Path(loc.__file__),Path(w.__file__),Path(c.__file__),Path(p.__file__),Path(p.old.__file__)]
    protocol=dict(study=STUDY,parent_digest=parent['protocol_digest'],exploration_digest=previous['protocol_digest'],seeds=list(SEEDS),steps=list(STEPS),arm='roles_8',n_sources=N,chunk=CHUNK,samples=samples,lr=p.old.LR,wd=p.old.WD,files={str(x.relative_to(ROOT)):p.sha(x) for x in dep},selection='distinct head pair maximizing summed mean within-source full-vocab target probability gain across discovery voices; lexicographic ties',thresholds=dict(native_joint=.95,within_joint=.95,transfer=.95,donor=.8,recipient_max=.1))
    m=dict(protocol=protocol,protocol_digest=p.digest(protocol));return m,(train,training,vocab,tests,enc)

def prepare():
    m,_=setup()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def score_logits(z,enc,bit):
    y=enc[2].reshape(-1,4)[:,torch.arange(4)^bit].flatten();z=z.double().cpu();pred=z.argmax(1);ok=pred==y
    return dict(full=float(ok.double().mean()),joint=float(ok.reshape(-1,4).all(1).double().mean()),prob=float(z.softmax(1)[torch.arange(len(y)),y].mean()))

def cached(model,e,device):
    a=e[0].to(device);ro=e[1].to(device);z,cache,err=w.forward(model,a,ro,capture=True);native=model(a,ro);assert torch.allclose(z,native,atol=.0001,rtol=.00001);return a,ro,z,cache,err

@torch.inference_mode()
def discover(model,entries,device):
    gains={s:{1:[],2:[]} for s in SITES}
    for v in c.VOICES:
        e=entries[v]['original'];out={s:{1:[],2:[]} for s in SITES};base=[]
        for i in range(0,len(e[0]),CHUNK):
            sub=tuple(x[i:i+CHUNK] for x in e);a,ro,z,cache,_=cached(model,sub,device);base.append(z.cpu())
            for s in SITES:
                for bit in (1,2):
                    h=cache[s]['h'];patch={s:('fixed',h[c.mate_indices(len(h),bit,device)])};q,_,_=w.forward(model,a,ro,patch);out[s][bit].append(q.cpu())
        for s in SITES:
            for bit in (1,2):gains[s][bit].append(score_logits(torch.cat(out[s][bit]),e,bit)['prob']-score_logits(torch.cat(base),e,bit)['prob'])
    scores={s:{str(bit):sum(gains[s][bit])/len(c.VOICES) for bit in (1,2)} for s in SITES}
    rh,eh=min(((a,b) for a in SITES for b in SITES if a!=b),key=lambda pair:(-(scores[pair[0]]['1']+scores[pair[1]]['2']),pair))
    return (rh,eh),scores

@torch.inference_mode()
def evaluate(model,entries,heads,device):
    rows=[];rh,eh=heads
    conditions=[(kind,bit) for kind in ('within','donor','cross_voice') for bit in (0,1,2,3)]
    for v in c.VOICES:
        ori=entries[v]['original'];rec=entries[v]['recipient'];other=entries[next(q for q in c.VOICES if q!=v)]['original'];outputs={f'{kind}/{bit}':[] for kind,bit in conditions};outputs['baseline']=[]
        for i in range(0,len(ori[0]),CHUNK):
            sub=lambda x:tuple(t[i:i+CHUNK] for t in x)
            a,ro,z,cr,_=cached(model,sub(rec),device);_,_,_,cd,_=cached(model,sub(ori),device);_,_,_,cv,_=cached(model,sub(other),device);outputs['baseline'].append(z.cpu())
            q,_,_=w.forward(model,a,ro,{s:('fixed',cr[s]['h']) for s in heads});assert torch.allclose(q,z,atol=.0001,rtol=.00001)
            for kind,bit in conditions:
                cache={'within':cr,'donor':cd,'cross_voice':cv}[kind];patch={}
                for s,axis in ((rh,1),(eh,2)):
                    if bit==0 or bit&axis:
                        h=cache[s]['h'];patch[s]=('fixed',h[c.mate_indices(len(h),axis if bit&axis else 0,device)])
                q,_,_=w.forward(model,a,ro,patch);assert torch.isfinite(q).all();outputs[f'{kind}/{bit}'].append(q.cpu())
        for condition,parts in outputs.items():
            bit=0 if condition=='baseline' else int(condition[-1]);z=torch.cat(parts);rows.append(dict(voice=v,condition=condition,**loc.target_metrics(z,rec,ori,bit),recipient_joint_full=score_logits(z,rec,bit)['joint'],donor_joint_full=score_logits(z,ori,bit)['joint']))
    return rows

def classify(native,rows):
    g={(x['voice'],x['condition']):x for x in rows}
    def allpass(conds,metric,threshold,lower=True):return all((g[v,k][metric]>=threshold if lower else g[v,k][metric]<=threshold) for v in c.VOICES for k in conds)
    if not all(native[pop][v]['C_query']>=.95 for pop in POPS for v in c.VOICES):return 'not_behaviourally_successful'
    if not allpass(['baseline','within/1','within/2','within/3'],'recipient_joint_full',.95):return 'successful_unresolved_control'
    if allpass(['donor/1','cross_voice/1','donor/2','cross_voice/2','donor/3','cross_voice/3'],'recipient_full',.95):return 'transferable_control'
    if allpass(['donor/1','cross_voice/1'],'recipient_full',.95) and allpass(['donor/2','cross_voice/2','donor/3','cross_voice/3'],'donor_full',.8) and allpass(['donor/2','cross_voice/2','donor/3','cross_voice/3'],'recipient_full',.1,False):return 'source_dependent_event'
    return 'successful_other_mechanism'

def ckpath(seed,step):return WEIGHTS/f'seed{seed}_step{step}.pt'
def repath(seed,step):return OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'

def run(seed,device):
    assert seed in SEEDS; assert device.type=='mps' and torch.backends.mps.is_available()
    m,(train,training,vocab,tests,enc)=setup();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,LAUNCH,MANIFEST):r.committed(f)
    p.require_frozen();torch.manual_seed(seed);torch.mps.manual_seed(seed)
    model=p.old.Model(len(vocab)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=p.old.LR,weight_decay=p.old.WD)
    current=0;state=None
    for step in STEPS:
        ck,rep=ckpath(seed,step),repath(seed,step)
        if ck.exists()!=rep.exists():raise RuntimeError(f'Partial checkpoint {seed}/{step}; inspect before restart')
        if ck.exists():
            report=read(rep);assert report['protocol_digest']==m['protocol_digest'] and report['seed']==seed and report['step']==step;current=step
        elif any(ckpath(seed,t).exists() for t in STEPS if t>step):raise RuntimeError('Noncontiguous checkpoints')
    if current==STEPS[-1]:print('ALREADY SEALED',seed,flush=True);return
    if current:
        state=torch.load(ckpath(seed,current),map_location='cpu',weights_only=True);assert (state['protocol_digest'],state['seed'],state['step'])==(m['protocol_digest'],seed,current)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.mps.set_rng_state(state['mps_rng'])
    trace=hashlib.sha256();losses=[]
    for row in schedule(seed,train):
        step,_,events=row;trace.update(r.line(row))
        if step<=current:
            if step==current:assert trace.hexdigest()==state['schedule_digest']
            continue
        loss=p.old.one_update(model,opt,batches(events,training,vocab),device);assert math.isfinite(loss);losses.append(loss)
        if step%800==0:print('TRAIN',seed,step,sum(losses)/len(losses),flush=True);losses=[]
        if step in STEPS:
            model.eval();native={pop:p.old.score_native(model,tests[pop],device) for pop in POPS};heads,scores=discover(model,enc[POPS[0]],device);rows=evaluate(model,enc[POPS[1]],heads,device);category=classify(native,rows)
            report=dict(study=STUDY,seed=seed,step=step,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),native=native,heads=list(heads),discovery=scores,rows=rows,category=category)
            ck=dict(study=STUDY,seed=seed,step=step,protocol_digest=m['protocol_digest'],schedule_digest=trace.hexdigest(),model=model.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state())
            r.save_state(ckpath(seed,step),ck);p.save_once(repath(seed,step),report);print('SEALED',seed,step,category,heads,flush=True)

def wilson(k,n):
    if not n:return None
    z=1.95996398454;a=k/n;d=1+z*z/n;b=(a+z*z/(2*n))/d;h=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/d;return [max(0,b-h),min(1,b+h)]

def summary():
    m=read(MANIFEST); reports=[];rows=[]
    for seed in SEEDS:
        for step in STEPS:
            f=repath(seed,step)
            if not f.exists():raise RuntimeError(f'Incomplete: {seed}/{step}')
            a=read(f);assert (a['seed'],a['step'],a['protocol_digest'])==(seed,step,m['protocol_digest']);reports.append(a);rows.extend(dict(seed=seed,step=step,**x) for x in a['rows'])
    final=[x for x in reports if x['step']==STEPS[-1]];counts=collections.Counter(x['category'] for x in final);n=len(final);success=n-counts['not_behaviourally_successful'];cats=('not_behaviourally_successful','successful_unresolved_control','transferable_control','source_dependent_event','successful_other_mechanism')
    result=dict(study=STUDY,protocol_digest=m['protocol_digest'],n_models=n,n_successful=success,counts={k:counts[k] for k in cats},all_seed_wilson={k:wilson(counts[k],n) for k in cats},successful_seed_wilson={k:wilson(counts[k],success) for k in cats[1:]},primary_replicated=counts['transferable_control']>=2 and counts['source_dependent_event']>=2,per_seed=[dict(seed=x['seed'],step=x['step'],heads=x['heads'],category=x['category'],native_joint={pop:{v:x['native'][pop][v]['C_query'] for v in c.VOICES} for pop in POPS}) for x in reports]);w.write_csv(OUT/'rows.csv',rows)
    f=OUT/'analysis_summary.json'
    if f.exists():assert read(f)==result
    else:p.save_once(f,result)
    print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='per_seed'}),flush=True)

def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel().eval();a=torch.tensor([[0,1,2,3,4,10,5,6,7],[0,1,2,3,4,10,5,8,7],[0,1,2,3,4,10,9,6,7],[0,1,2,3,4,10,9,8,7]]).repeat(2,1);ro=torch.full((8,),8);y=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8);orig=(a,ro,y,ch);rec,_=loc.replace(orig,torch.tensor([[11,12,13,14]]*2),10);entries={v:dict(original=orig,recipient=rec) for v in c.VOICES};heads,scores=discover(model,entries,torch.device('cpu'));assert len(scores)==12 and heads[0]!=heads[1];rows=evaluate(model,entries,heads,torch.device('cpu'));assert len(rows)==26;assert classify({pop:{v:{'C_query':0} for v in c.VOICES} for pop in POPS},rows)=='not_behaviourally_successful';print('SMOKE PASS: discovery, disjoint transfer, classification',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));a.add_argument('--seed',type=int);v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.seed,torch.device('mps'))
if __name__=='__main__':main()
