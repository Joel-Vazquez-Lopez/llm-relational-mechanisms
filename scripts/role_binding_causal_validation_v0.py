#!/usr/bin/env python3
"""Exploratory source-side binding and cross-source control transfer."""
import argparse,json
from pathlib import Path
from random import Random
import torch
import role_causal_pathways_v0 as w
c=w.c; ROOT=c.ROOT; OUT=ROOT/'reports/role_binding_causal_validation_v0'
MANIFEST=OUT/'study_manifest.json'; SCRIPT=Path(__file__).resolve()
PROTOCOL=ROOT/'paper/ROLE_BINDING_CAUSAL_VALIDATION_V0_PROTOCOL.md'
SEEDS=w.SEEDS; STEPS=w.STEPS; N=128; SAMPLE_SEED=20261128; POP='unseen_quartets_rich'; CHUNK=128
PERMS={'role':(1,0,3,2),'event':(2,3,0,1),'mixed':(1,2,3,0)}

def rename(enc,perm):
    ids,ro,y,ch=enc; c.audit_pairs(*enc); g=y.reshape(-1,4); a=ids.reshape(-1,4,ids.shape[-1]); new=g[:,list(perm)]; z=a.clone(); mask=torch.zeros_like(a,dtype=torch.bool)
    for j in range(4):
        m=a==g[:,j,None,None]; assert m.any(-1).all(); z=torch.where(m,new[:,j,None,None],z); mask|=m
    assert torch.equal(z!=a,mask); assert torch.all(mask.nonzero()[:,-1] < ro.reshape(-1,4)[mask.nonzero()[:,0],mask.nonzero()[:,1]])
    result=(z.reshape_as(ids),ro,new.flatten(),ch); c.audit_pairs(*result)
    return result,mask.reshape_as(ids)

def manifest():
    loc,d,r,p=c.parents(); wm,fixtures,tests=w.manifest(); assert w.read(w.MANIFEST)==wm
    cm=w.read(c.MANIFEST); summary=w.read(w.OUT/'analysis_summary.json'); assert summary['protocol_digest']==wm['protocol_digest'] and summary['n_rows']==6944
    enc=tests[POP]; n=len(enc[c.VOICES[0]][0])//4; excluded=set()
    for v in c.VOICES:
        key=f'{POP}/{v}'; excluded.update(cm['protocol']['samples'][key]['source_indices']); excluded.update(wm['protocol']['samples'][key]['indices'])
    indices=sorted(Random(SAMPLE_SEED).sample(sorted(set(range(n))-excluded),N)); samples={}
    for v in c.VOICES:
        e=c.subset(enc[v],indices); samples[v]={'original':p.digest([t.tolist() for t in e])}
        for label,perm in PERMS.items():
            z,mask=rename(e,perm); samples[v][label]=p.digest([t.tolist() for t in z])
        assert torch.equal(e[2],c.subset(enc[c.VOICES[0]],indices)[2])
    proto=dict(study='ROLE-BINDING-CAUSAL-VALIDATION-v0',classification='exploratory',parent_digest=wm['protocol_digest'],parent_summary_sha=p.sha(w.OUT/'analysis_summary.json'),script_sha=p.sha(SCRIPT),protocol_sha=p.sha(PROTOCOL),indices=indices,samples=samples,permutations={k:list(v) for k,v in PERMS.items()},heads={str(k):list(v) for k,v in w.HEADS.items()},seeds=list(SEEDS),steps=list(STEPS),population=POP,n_sources=N,chunk=CHUNK,sample_seed=SAMPLE_SEED)
    return dict(protocol=proto,protocol_digest=p.digest(proto)),fixtures,tests

def prepare():
    *_,p=c.parents(); m,_,_=manifest()
    if MANIFEST.exists():assert w.read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def donor_patches(cache,heads,bit):
    out={}
    for s,b in zip(heads,(1,2)):
        if bit&b:
            h=cache[s]['h'];out[s]=('fixed',h[c.mate_indices(len(h),b,h.device)])
    return out

@torch.inference_mode()
def entity_patch(model,ids,ro,donor,changed,layer,kind,head=None):
    # One layer, readout only. Original source donor has same token positions.
    b,n=ids.shape; dev=ids.device; ix=torch.arange(b,device=dev); x=model.embed(ids)+model.pos(torch.arange(n,device=dev)); mask=torch.ones(n,n,dtype=torch.bool,device=dev).triu(1)
    for j,block in enumerate(model.blocks):
        z=block.n1(x);a=block.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        if j==layer:
            hh=w.parts(block.attn,z,ro); recon=sum(t['h'] for t in hh.values())+block.attn.out_proj.bias; assert torch.allclose(recon,a[ix,ro],atol=.003,rtol=.0002)
            delta=torch.zeros_like(a[ix,ro]); select=changed[:,:,None]
            if kind=='other':select=~select
            for h,t in hh.items():
                if head is not None and h!=head:continue
                d=donor[f'L{j}.head{h}']; k,v=t['k'],t['v']
                if kind in ('k','kv','other'):k=torch.where(select,d['k'],k)
                if kind in ('v','kv','other'):v=torch.where(select,d['v'],v)
                target=w.projected(w.weights(t['q'],k,ro),v,t['wo']);delta+=target-t['h']
            a=a.clone();a[ix,ro]+=delta
        x=x+a;x=x+block.mlp(block.n2(x))
    return model.out(model.norm(x[ix,ro]))

def evaluate(logits,enc,bit):return w.outcome(logits,enc[2],enc[3],bit)

@torch.inference_mode()
def analyse(model,original,other_voice,device,seed):
    rh,eh,rc,ec=w.HEADS[seed]; rows=[]; checks={'native_logit':0.,'head_reconstruction':0.,'self_patch':0.}
    for name,perm in PERMS.items():
        recipient,changed=rename(original,perm); outputs={}; specs={}
        def add(key,z,bit,target):
            outputs.setdefault(key,[]).append(z.cpu());specs[key]=(bit,target)
        for i in range(0,len(original[0]),CHUNK):
            a,ro=original[0][i:i+CHUNK].to(device),original[1][i:i+CHUNK].to(device); b=recipient[0][i:i+CHUNK].to(device); m=changed[i:i+CHUNK].to(device)
            ov,orr=other_voice[0][i:i+CHUNK].to(device),other_voice[1][i:i+CHUNK].to(device)
            lo,co,e=w.forward(model,a,ro,capture=True);lr,cr,f=w.forward(model,b,ro,capture=True);lv,cv,g=w.forward(model,ov,orr,capture=True)
            checks['head_reconstruction']=max(checks['head_reconstruction'],e,f,g)
            for ids,pos,l in ((a,ro,lo),(b,ro,lr),(ov,orr,lv)):
                native=model(ids,pos); checks['native_logit']=max(checks['native_logit'],float((l-native).abs().max()));assert torch.allclose(l,native,atol=.0001,rtol=.00001)
            add('baseline/original',lo,0,'original');add('baseline/renamed',lr,0,'renamed')
            for label,cache,heads in [('within',cr,(rh,eh)),('cross_source',co,(rh,eh)),('cross_voice_source',cv,(rh,eh)),('control',co,(rc,ec))]:
                for bit in (1,2,3):
                    z,_,err=w.forward(model,b,ro,donor_patches(cache,heads,bit)); checks['head_reconstruction']=max(checks['head_reconstruction'],err);add(f'query/{label}/{bit}',z,bit,'renamed')
            # Original same-query control signal on the renamed source; should preserve recipient binding.
            for label,cache in [('cross_source',co),('cross_voice_source',cv)]:
                patch={s:('fixed',cache[s]['h']) for s in (rh,eh)};z,_,err=w.forward(model,b,ro,patch);checks['head_reconstruction']=max(checks['head_reconstruction'],err);add(f'query/{label}/0',z,0,'renamed')
            self_l,_,err=w.forward(model,b,ro,{s:('fixed',cr[s]['h']) for s in (rh,eh)}); checks['self_patch']=max(checks['self_patch'],float((self_l-lr).abs().max()));assert torch.allclose(self_l,lr,atol=.0001,rtol=.00001)
            for layer in range(3):
                for kind in ('k','v','kv','other'):
                    z=entity_patch(model,b,ro,co,m,layer,kind);add(f'entity/L{layer}/{kind}',z,0,'original')
                for head in range(4):
                    z=entity_patch(model,b,ro,co,m,layer,'kv',head);add(f'entity/L{layer}/head{head}/kv',z,0,'original')
                selfz=entity_patch(model,b,ro,cr,m,layer,'kv');err=float((selfz-lr).abs().max());checks['self_patch']=max(checks['self_patch'],err);assert torch.allclose(selfz,lr,atol=.0001,rtol=.00001)
        for key,parts in outputs.items():
            bit,target=specs[key];z=torch.cat(parts);assert torch.isfinite(z).all();ref=original if target=='original' else recipient; alt=recipient if target=='original' else original
            rows.append(dict(permutation=name,condition=key,target_source=target,target_bit=bit,**evaluate(z,ref,bit),alternate_target_accuracy=evaluate(z,alt,bit)['target_accuracy']))
    return rows,checks

def run(seed,step,device):
    loc,d,r,p=c.parents();m,fixtures,tests=manifest();assert w.read(MANIFEST)==m
    for path in (SCRIPT,PROTOCOL,MANIFEST):r.committed(path)
    p.require_frozen();assert seed in SEEDS and step in STEPS
    path=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if path.exists():assert w.read(path)['protocol_digest']==m['protocol_digest'];print('ALREADY SEALED',seed,step);return
    ck=torch.load(r.checkpoint_path(seed,step),map_location='cpu',weights_only=True);ref=w.read(r.report_path(seed,step));assert (ck['study'],ck['seed'],ck['step'],ck['protocol_digest'],ck['schedule_prefix_sha256'])==(r.STUDY,seed,step,ref['protocol_digest'],ref['schedule_prefix_sha256'])
    model=p.old.Model(len(fixtures[-1])).to(device).eval();model.load_state_dict(ck['models'][w.ARM],strict=True);rows=[];checks=[]
    for v in c.VOICES:
        full=tests[POP][v];nd=d.verify(p.old.score_native(model,{v:full},device)[v],ref['outcomes'][w.ARM][POP][v]);other=next(x for x in c.VOICES if x!=v);idx=m['protocol']['indices'];rr,cc=analyse(model,c.subset(full,idx),c.subset(tests[POP][other],idx),device,seed);rows.extend(dict(voice=v,**x) for x in rr);checks.append(dict(voice=v,native_score_delta=nd,**cc));print('BINDING',seed,step,v,flush=True)
    p.save_once(path,dict(seed=seed,step=step,protocol_digest=m['protocol_digest'],rows=rows,checks=checks));print('SEALED',seed,step,flush=True)

def summary():
    *_,p=c.parents();m=w.read(MANIFEST);allrows=[];checks=[];compact=OUT/'compact';compact.mkdir(exist_ok=True)
    for seed in SEEDS:
        for step in STEPS:
            a=w.read(OUT/'per_checkpoint'/f'seed{seed}_step{step}.json');assert (a['seed'],a['step'],a['protocol_digest'])==(seed,step,m['protocol_digest']);assert len(a['rows'])==240;checks.extend(a['checks']);allrows.extend(dict(seed=seed,step=step,**x) for x in a['rows']);w.write_csv(compact/f'seed{seed}_step{step}.csv',a['rows'])
    w.write_csv(OUT/'rows.csv',allrows);p.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',n_rows=len(allrows),max_checks={k:max(x[k] for x in checks) for k in checks[0] if k!='voice'}));print('SUMMARY PASS',len(allrows))

def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel().eval();ids=torch.tensor([[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,8,7],[0,1,2,3,4,9,6,7],[0,1,2,3,4,9,8,7]]).repeat(2,1);ro=torch.full((8,),7);y=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8);enc=(ids,ro,y,ch)
    for seed in (16101,16127):
        rows,checks=analyse(model,enc,enc,torch.device('cpu'),seed);assert len(rows)==120,len(rows);assert checks['native_logit']==0
        for name,perm in PERMS.items():
            e,mask=rename(enc,perm);assert e[2].tolist()==[x+1 for x in perm]*2;assert mask.sum()==32
    print('SMOKE PASS: source permutation, cross-source/voice head patches, entity K/V, native and self checks',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));a.add_argument('--seed',type=int);a.add_argument('--step',type=int);a.add_argument('--device',choices=('cpu','mps'),default='mps');v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.seed,v.step,torch.device(v.device))
if __name__=='__main__':main()
