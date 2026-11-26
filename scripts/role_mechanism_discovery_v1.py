#!/usr/bin/env python3
"""Exploratory multi-component causal search; never changes replication v0."""
import argparse, itertools, json, collections, hashlib
from pathlib import Path
from random import Random
import torch
import role_mechanism_replication_v0 as rep
loc,w,c,p,r=rep.loc,rep.w,rep.c,rep.p,rep.r
ROOT=rep.ROOT; OUT=ROOT/'reports/role_mechanism_discovery_v1'; MANIFEST=OUT/'study_manifest.json'
SCRIPT=Path(__file__).resolve(); PROTOCOL=ROOT/'paper/ROLE_MECHANISM_DISCOVERY_V1_PROTOCOL.md'; LAUNCH=ROOT/'scripts/run_role_mechanism_discovery_v1.sh'
STUDY='ROLE-MECHANISM-DISCOVERY-v1'; HEADS=rep.SITES; MLPS=tuple(f'L{i}.mlp' for i in range(3)); SITES=HEADS+MLPS
RESIDUALS=tuple(f'L{i}.residual' for i in range(3)); POPS=rep.POPS; STEPS=rep.STEPS; SEEDS=rep.SEEDS
CONTROLS=(16101,16127); TASKS=tuple((s,51200) for s in CONTROLS)+tuple((s,t) for t in reversed(STEPS) for s in SEEDS)
DISC_N=64; TEST_N=128; CHUNK=128; SHORT=5; EXPAND=2; SAMPLE_SEED=272001
BASE_SETS=tuple((s,) for s in SITES)+tuple(tuple(sorted(x)) for x in itertools.combinations(SITES,2))

def read(path):return json.loads(path.read_text())
def source_paths(seed,step):return (r.checkpoint_path(seed,step),r.report_path(seed,step)) if seed in CONTROLS else (rep.ckpath(seed,step),rep.repath(seed,step))
def result_path(seed,step):return OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'

def setup():
    rm,(_,_,vocab,tests,_)=rep.setup();assert read(rep.MANIFEST)==rm
    rs=read(rep.OUT/'analysis_summary.json');assert rs['protocol_digest']==rm['protocol_digest'] and rs['n_models']==16 and rs['primary_replicated'] is False
    previous=read(loc.MANIFEST); assert read(loc.OUT/'analysis_summary.json')['protocol_digest']==previous['protocol_digest']
    cm=read(c.MANIFEST);wm=read(w.MANIFEST);bm=read(loc.b.MANIFEST); sep=vocab['<sep>']; samples={}; entries={}
    for j,pop in enumerate(POPS):
        full=tests[pop];n=len(full[c.VOICES[0]][0])//4;exclude=set(rm['protocol']['samples'][pop]['indices'])
        for v in c.VOICES:
            key=f'{pop}/{v}';exclude.update(cm['protocol']['samples'][key]['source_indices']);exclude.update(wm['protocol']['samples'][key]['indices'])
        if j==1:exclude.update(bm['protocol']['indices']);exclude.update(previous['protocol']['indices'])
        idx=sorted(Random(SAMPLE_SEED+j).sample(sorted(set(range(n))-exclude),DISC_N if j==0 else TEST_N));entries[pop]={};samples[pop]={'indices':idx,'voices':{}}
        for v in c.VOICES:
            e=c.subset(full[v],idx);c.audit_pairs(*e);item={'original':e}
            if j==1:
                names=loc.disjoint_names(e,full[v])
                assert not (names[:,:,None]==e[2].reshape(-1,4)[:,None,:]).any()
                item['recipient']=loc.replace(e,names,sep)[0]
            entries[pop][v]=item;samples[pop]['voices'][v]={k:p.digest([t.tolist() for t in z]) for k,z in item.items()}
        for k in entries[pop][c.VOICES[0]]:assert torch.equal(entries[pop][c.VOICES[0]][k][2],entries[pop][c.VOICES[1]][k][2])
    inputs={}
    for seed,step in TASKS:
        ck,rp=source_paths(seed,step);report=read(rp)
        assert (report['seed'],report['step'])==(seed,step)
        if seed not in CONTROLS:assert report['protocol_digest']==rm['protocol_digest']
        inputs[f'{seed}/{step}']={'weights':p.sha(ck),'report':p.sha(rp)}
    deps=[SCRIPT,PROTOCOL,LAUNCH,Path(rep.__file__),Path(loc.__file__),Path(w.__file__),Path(c.__file__),Path(p.__file__),Path(r.__file__),Path(p.old.__file__)]
    protocol=dict(study=STUDY,classification='exploratory_followup_after_failed_replication',replication_digest=rm['protocol_digest'],replication_summary_sha=p.sha(rep.OUT/'analysis_summary.json'),localisation_digest=previous['protocol_digest'],inputs=inputs,files={str(x.relative_to(ROOT)):p.sha(x) for x in deps},tasks=[list(x) for x in TASKS],samples=samples,base_sets=[list(x) for x in BASE_SETS],residuals=list(RESIDUALS),shortlist=SHORT,expand=EXPAND,chunk=CHUNK,selection='rank axis sets by worst-voice strict target joint, then mean target probability, then size and lexicographic; expand top two pairs to triples; shortlist top five plus best singleton; disjoint role/event sets selected by worst voice-and-axis strict joint over bits 1,2,3, then mean probability, then total size and lexicographic',thresholds=rm['protocol']['thresholds'])
    return dict(protocol=protocol,protocol_digest=p.digest(protocol)),(vocab,entries)

def prepare():
    m,_=setup()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

@torch.inference_mode()
def forward(model,ids,ro,patch=None,capture=False):
    """Fixed clean donor outputs at readout; head deltas relative to current forward."""
    patch=patch or {};b,n=ids.shape;dev=ids.device;ix=torch.arange(b,device=dev)
    x=model.embed(ids)+model.pos(torch.arange(n,device=dev));mask=torch.ones(n,n,dtype=torch.bool,device=dev).triu(1);cache={};maxerr=0.
    for j,block in enumerate(model.blocks):
        pre=f'L{j}.';z=block.n1(x);a=block.attn(z,z,z,attn_mask=mask,need_weights=False)[0];hp=[s for s in patch if s.startswith(pre+'head')]
        if capture or hp:
            hh=w.parts(block.attn,z,ro);recon=sum(t['h'] for t in hh.values())+block.attn.out_proj.bias;err=float((recon-a[ix,ro]).abs().max());maxerr=max(maxerr,err);assert torch.allclose(recon,a[ix,ro],atol=.003,rtol=.0002)
            if capture:cache.update({pre+f'head{k}':t['h'] for k,t in hh.items()})
            if hp:
                delta=sum((patch[s]-hh[int(s[-1])]['h'] for s in hp),torch.zeros_like(a[ix,ro]));a=a.clone();a[ix,ro]+=delta
        x=x+a;mlp=block.mlp(block.n2(x));s=pre+'mlp'
        if capture:cache[s]=mlp[ix,ro].clone()
        if s in patch:mlp=mlp.clone();mlp[ix,ro]=patch[s]
        x=x+mlp;s=pre+'residual'
        if capture:cache[s]=x[ix,ro].clone()
        if s in patch:x=x.clone();x[ix,ro]=patch[s]
    return model.out(model.norm(x[ix,ro])),cache,maxerr

def chunks(e):
    for i in range(0,len(e[0]),CHUNK):yield tuple(t[i:i+CHUNK] for t in e)

def make_cache(model,enc,device,checks):
    a,ro=enc[0].to(device),enc[1].to(device);z,cache,err=forward(model,a,ro,capture=True);native=model(a,ro)
    checks['native_logit']=max(checks['native_logit'],float((native-z).abs().max()));checks['head_reconstruction']=max(checks['head_reconstruction'],err);assert torch.allclose(z,native,atol=.0001,rtol=.00001)
    return a,ro,z,cache

def apply(model,a,ro,cache,assign,checks):
    patch={s:cache[s][c.mate_indices(len(a),bit,a.device)] for s,bit in assign.items()};z,_,err=forward(model,a,ro,patch);checks['head_reconstruction']=max(checks['head_reconstruction'],err);assert torch.isfinite(z).all();return z

def assignment(role,event,bit):
    assert not set(role)&set(event)
    return {s:axis if bit&axis else 0 for sites,axis in [(role,1),(event,2)] for s in sites if bit==0 or bit&axis}

def empty_checks():return dict(native_logit=0.,head_reconstruction=0.,self_patch=0.,residual_positive_control=0.)

def run_specs(model,entries,specs,device,checks):
    """Discovery specs name -> (target bit, component->donor bit)."""
    rows=[]
    for v in c.VOICES:
        e=entries[v]['original'];out={k:[] for k in specs}
        for sub in chunks(e):
            a,ro,z,cache=make_cache(model,sub,device,checks)
            for name,(bit,assign) in specs.items():out[name].append(apply(model,a,ro,cache,assign,checks).cpu())
        for name,(bit,assign) in specs.items():rows.append(dict(voice=v,condition=name,bit=bit,**rep.score_logits(torch.cat(out[name]),e,bit)))
    return rows

def key(sites):return '+'.join(sites)
def axis_rank(rows,sets,bit):
    by=collections.defaultdict(list)
    for x in rows:
        if x['bit']==bit:by[x['condition']].append(x)
    return sorted(sets,key=lambda s:(-min(x['joint'] for x in by[key(s)]),-sum(x['prob'] for x in by[key(s)])/2,len(s),s))

@torch.inference_mode()
def discover(model,entries,device,checks):
    # Separate axis passes keep condition names unique and memory bounded.
    axes={};ranks={};shortlists={}
    for bit in (1,2):
        sets=list(BASE_SETS);specs={key(s):(bit,{k:bit for k in s}) for s in sets};rows=run_specs(model,entries,specs,device,checks)
        ranked=axis_rank(rows,sets,bit);top_pairs=[s for s in ranked if len(s)==2][:EXPAND]
        triples=sorted({tuple(sorted((*s,k))) for s in top_pairs for k in SITES if k not in s});rows+=run_specs(model,entries,{key(s):(bit,{k:bit for k in s}) for s in triples},device,checks);sets+=triples
        ranked=axis_rank(rows,sets,bit);ranks[bit]=ranked;axes[bit]=rows;best_single=next(s for s in ranked if len(s)==1);shortlists[bit]=list(dict.fromkeys(ranked[:SHORT]+[best_single]))
    # Ensure a disjoint singleton fallback even when both best singletons coincide.
    pairs=[(a,b) for a in shortlists[1] for b in shortlists[2] if not set(a)&set(b)]
    fallback=next((a,b) for a in ranks[1] if len(a)==1 for b in ranks[2] if len(b)==1 and a!=b)
    pairs=list(dict.fromkeys(pairs+[fallback]));specs={}
    for i,(a,b) in enumerate(pairs):
        for bit in (1,2,3):specs[f'{i}/{bit}']=(bit,assignment(a,b,bit))
    joint=run_specs(model,entries,specs,device,checks);by=collections.defaultdict(list)
    for row in joint:by[int(row['condition'].split('/')[0])].append(row)
    ranked=sorted(range(len(pairs)),key=lambda i:(-min(x['joint'] for x in by[i]),-sum(x['prob'] for x in by[i])/6,sum(map(len,pairs[i])),pairs[i]));best=pairs[ranked[0]]
    # Residual swaps are diagnostic ceilings, not eligible components (contain upstream signal).
    residual=[]
    for bit in (1,2):residual+=run_specs(model,entries,{s:(bit,{s:bit}) for s in RESIDUALS},device,checks)
    return best,dict(axis_rows=axes,shortlists=shortlists,pairs=[[list(a),list(b)] for a,b in pairs],joint_rows=joint,selected_index=ranked[0],residual_rows=residual)

@torch.inference_mode()
def evaluate(model,entries,sets_by_name,device,checks):
    rows=[]
    for v in c.VOICES:
        ori=entries[v]['original'];rec=entries[v]['recipient'];other=entries[next(q for q in c.VOICES if q!=v)]['original']
        outs={(name,kind,bit):[] for name in sets_by_name for kind in ('within','donor','cross_voice') for bit in (0,1,2,3)};base=[]
        for a0,b0,o0 in zip(chunks(rec),chunks(ori),chunks(other)):
            a,ro,z,cr=make_cache(model,a0,device,checks);_,_,_,cd=make_cache(model,b0,device,checks);_,_,_,cv=make_cache(model,o0,device,checks);base.append(z.cpu())
            selfz=apply(model,a,ro,cr,{s:0 for s in SITES},checks);err=float((selfz-z).abs().max());checks['self_patch']=max(checks['self_patch'],err);assert torch.allclose(selfz,z,atol=.0001,rtol=.00001)
            for bit in (1,2):
                rz=apply(model,a,ro,cr,{'L2.residual':bit},checks);target=z[c.mate_indices(len(a),bit,device)];err=float((rz-target).abs().max());checks['residual_positive_control']=max(checks['residual_positive_control'],err);assert torch.allclose(rz,target,atol=.0001,rtol=.00001)
            for (name,kind,bit),parts in outs.items():
                role,event=sets_by_name[name];cache={'within':cr,'donor':cd,'cross_voice':cv}[kind];parts.append(apply(model,a,ro,cache,assignment(role,event,bit),checks).cpu())
        def row(name,condition,bit,z):return dict(method=name,voice=v,condition=condition,**loc.target_metrics(z,rec,ori,bit),recipient_joint_full=rep.score_logits(z,rec,bit)['joint'],donor_joint_full=rep.score_logits(z,ori,bit)['joint'])
        for name in sets_by_name:rows.append(row(name,'baseline',0,torch.cat(base)))
        for (name,kind,bit),parts in outs.items():rows.append(row(name,f'{kind}/{bit}',bit,torch.cat(parts)))
    return rows

def run_one(seed,step,m,data,device):
    assert (seed,step) in TASKS;path=result_path(seed,step)
    if path.exists():
        report=read(path);assert (report['seed'],report['step'],report['protocol_digest'])==(seed,step,m['protocol_digest']);print('ALREADY SEALED',seed,step,flush=True);return
    vocab,entries=data;ck,rp=source_paths(seed,step);audit=m['protocol']['inputs'][f'{seed}/{step}'];assert p.sha(ck)==audit['weights'] and p.sha(rp)==audit['report'];state=torch.load(ck,map_location='cpu',weights_only=True);prior=read(rp)
    assert (state['seed'],state['step'],state['protocol_digest'])==(seed,step,prior['protocol_digest'])
    if seed in CONTROLS:
        assert state['schedule_prefix_sha256']==prior['schedule_prefix_sha256'];weights=state['models']['roles_8'];native=prior['outcomes']['roles_8'];old=tuple((s,) for s in w.HEADS[seed][:2])
    else:
        assert state['schedule_digest']==prior['schedule_digest'];weights=state['model'];native=prior['native'];old=tuple((s,) for s in prior['heads'])
    model=p.old.Model(len(vocab)).to(device).eval();model.load_state_dict(weights,strict=True);checks=empty_checks();print('DISCOVER',seed,step,flush=True)
    best,discovery=discover(model,entries[POPS[0]],device,checks);sets={'expanded':best,'original_pair':old};rows=evaluate(model,entries[POPS[1]],sets,device,checks);categories={k:rep.classify(native,[x for x in rows if x['method']==k]) for k in sets}
    p.save_once(path,dict(study=STUDY,seed=seed,step=step,protocol_digest=m['protocol_digest'],sets={k:[list(x) for x in pair] for k,pair in sets.items()},discovery=discovery,rows=rows,categories=categories,original_category=prior.get('category'),native_joint={pop:{v:native[pop][v]['C_query'] for v in c.VOICES} for pop in POPS},checks=checks));print('SEALED',seed,step,categories,sets,flush=True)

def run(seed,step,device,all_tasks=False):
    m,data=setup();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,LAUNCH,MANIFEST):r.committed(f)
    p.require_frozen()
    for s,t in TASKS if all_tasks else [(seed,step)]:run_one(s,t,m,data,device)

def summary():
    m=read(MANIFEST);reports=[];rows=[];compact=[]
    for s,t in TASKS:
        a=read(result_path(s,t));assert (a['seed'],a['step'],a['protocol_digest'])==(s,t,m['protocol_digest']);reports.append(a);rows.extend(dict(seed=s,step=t,**x) for x in a['rows'])
        for method,pair in a['sets'].items():
            g=[x for x in a['rows'] if x['method']==method];avg=lambda cond,k:sum(x[k] for x in g if x['condition']==cond)/2
            compact.append(dict(seed=s,step=t,method=method,role=key(pair[0]),event=key(pair[1]),category=a['categories'][method],original_category=a['original_category'] or 'positive_control',native_joint_min=min(x for d in a['native_joint'].values() for x in d.values()),within_role_joint=avg('within/1','recipient_joint_full'),within_event_joint=avg('within/2','recipient_joint_full'),within_both_joint=avg('within/3','recipient_joint_full'),role_transfer=avg('donor/1','recipient_full'),event_transfer=avg('donor/2','recipient_full'),event_donor=avg('donor/2','donor_full'),joint_transfer=avg('donor/3','recipient_full'),joint_donor=avg('donor/3','donor_full'),cross_voice_role=avg('cross_voice/1','recipient_full'),cross_voice_event=avg('cross_voice/2','recipient_full'),cross_voice_donor=avg('cross_voice/2','donor_full'),cross_voice_joint=avg('cross_voice/3','recipient_full')))
    final=[a for a in reports if a['seed'] in SEEDS and a['step']==51200];counts={k:dict(collections.Counter(a['categories'][k] for a in final)) for k in ('expanded','original_pair')};controls=[dict(seed=a['seed'],sets=a['sets'],categories=a['categories']) for a in reports if a['seed'] in CONTROLS]
    result=dict(study=STUDY,classification='exploratory',protocol_digest=m['protocol_digest'],n_checkpoints=len(reports),n_models=len(final),counts=counts,positive_controls=controls,max_checks={k:max(a['checks'][k] for a in reports) for k in empty_checks()},original_replication_result_preserved=True,original_replication_primary_replicated=False)
    w.write_csv(OUT/'rows.csv',rows);w.write_csv(OUT/'compact.csv',compact);f=OUT/'analysis_summary.json'
    if f.exists():assert read(f)==result
    else:p.save_once(f,result)
    print('SUMMARY',json.dumps(result),flush=True)

def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel().eval();a=torch.tensor([[0,1,2,3,4,10,5,6,7],[0,1,2,3,4,10,5,8,7],[0,1,2,3,4,10,9,6,7],[0,1,2,3,4,10,9,8,7]]).repeat(2,1);ro=torch.full((8,),8);y=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8);ori=(a,ro,y,ch);rec,_=loc.replace(ori,torch.tensor([[11,12,13,14]]*2),10);entries={v:dict(original=ori,recipient=rec) for v in c.VOICES};checks=empty_checks();dev=torch.device('cpu');assert len(BASE_SETS)==120 and len([s for s in BASE_SETS if len(s)==2 and all(k in HEADS for k in s)])==66
    best,d=discover(model,entries,dev,checks);assert not set(best[0])&set(best[1]);rows=evaluate(model,entries,{'expanded':best,'original_pair':(('L0.head0',),('L2.head2',))},dev,checks);assert len(rows)==52 and checks['native_logit']==0;assert checks['self_patch']<1e-5;assert checks['residual_positive_control']<1e-5
    # Verify component intervention agrees with earlier head patch implementation.
    z,cache,_=forward(model,a,ro,capture=True);z1=apply(model,a,ro,cache,{'L0.head0':1,'L2.head2':2},checks);z2=w.forward(model,a,ro,{s:('fixed',cache[s][c.mate_indices(len(a),bit,dev)]) for s,bit in [('L0.head0',1),('L2.head2',2)]})[0];assert torch.allclose(z1,z2,atol=1e-5,rtol=1e-5)
    print('SMOKE PASS: 120 base sets, adaptive triples, disjoint selection, full-vocab scoring, joint head/MLP patches, native/self/residual checks',checks,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('smoke','prepare','run','run-all','summary'),required=True);ap.add_argument('--seed',type=int);ap.add_argument('--step',type=int);ap.add_argument('--device',choices=('mps','cpu'),default='mps');a=ap.parse_args()
    if a.phase=='smoke':smoke()
    elif a.phase=='prepare':prepare()
    elif a.phase=='summary':summary()
    else:run(a.seed,a.step,torch.device(a.device),a.phase=='run-all')
if __name__=='__main__':main()
