#!/usr/bin/env python3
"""Exploratory joint head intervention and Q/K/V hybrid decomposition."""
import argparse,csv,json,math
from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F
import role_causal_routing_v0 as c
ROOT=c.ROOT; OUT=ROOT/'reports/role_causal_pathways_v0'; MANIFEST=OUT/'study_manifest.json'
PROTOCOL=ROOT/'paper/ROLE_CAUSAL_PATHWAYS_V0_PROTOCOL.md'; SCRIPT=Path(__file__).resolve()
SEEDS=c.SEEDS; STEPS=c.STEPS; ARM='roles_8'; N=128; CHUNK=128; SAMPLE_SEED=20261127
# Fixed on discovery evidence before new inference; collapsed heads are matched hypotheses, not discoveries.
HEADS={16101:('L0.head0','L0.head3','L0.head1','L0.head2'),16127:('L0.head2','L2.head2','L0.head1','L2.head1'),16143:('L0.head2','L2.head2','L0.head1','L2.head1'),16167:('L0.head2','L2.head2','L0.head1','L2.head1')}
HYBRIDS=('q','k','v','qk','qv','kv','qkv','pattern','pattern_v','k_token','v_token','kv_token','kv_other')
def read(path):return json.loads(path.read_text())
def manifest():
    loc,d,r,p=c.parents(); cm,fixtures,tests=c.manifest(); assert read(c.MANIFEST)==cm
    cs=read(c.OUT/'analysis_summary.json'); assert cs['protocol_digest']==cm['protocol_digest'] and cs['counts']['rows']==24576
    samples={}
    for pop in c.POPS:
        for voice in c.VOICES:
            enc=tests[pop][voice]; key=f'{pop}/{voice}'; old=cm['protocol']['samples'][key]['source_indices']; n=len(enc[0])//4
            available=sorted(set(range(n))-set(old)); idx=sorted(Random(SAMPLE_SEED+n).sample(available,N)); assert not set(idx)&set(old)
            e=c.subset(enc,idx);c.audit_pairs(*e);samples[key]=dict(indices=idx,encoded_sha=p.digest([x.tolist() for x in e]))
    proto=dict(study='ROLE-CAUSAL-PATHWAYS-v0',classification='exploratory',parent_digest=cm['protocol_digest'],parent_summary_sha=p.sha(c.OUT/'analysis_summary.json'),selection_sha={m:p.sha(c.OUT/'compact'/f'final_{m}.csv') for m in ('role_swap','event_swap')},script_sha=p.sha(SCRIPT),protocol_sha=p.sha(PROTOCOL),heads={str(k):list(v) for k,v in HEADS.items()},seeds=list(SEEDS),steps=list(STEPS),arm=ARM,hybrids=list(HYBRIDS),sample_seed=SAMPLE_SEED,n_sources=N,chunk=CHUNK,samples=samples)
    return dict(protocol=proto,protocol_digest=p.digest(proto)),fixtures,tests

def prepare():
    *_,p=c.parents();m,_,_=manifest()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def weights(q,k,ro):
    score=(q[:,None,:]@k.transpose(-1,-2)).squeeze(1)/math.sqrt(q.shape[-1])
    return score.masked_fill(torch.arange(k.shape[1],device=k.device)[None,:]>ro[:,None],-torch.inf).softmax(-1)

def projected(w,v,wo):return (w[:,None,:]@v).squeeze(1)@wo.T

def parts(attn,z,ro):
    b,n,w=z.shape; h=attn.num_heads; d=w//h; ix=torch.arange(b,device=z.device)
    q,k,v=F.linear(z,attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    q=q[ix,ro].reshape(b,h,d); k=k.reshape(b,n,h,d);v=v.reshape(b,n,h,d)
    out={}
    for j in range(h):
        qq=q[:,j];kk=k[:,:,j];vv=v[:,:,j];wo=attn.out_proj.weight[:,j*d:(j+1)*d];ww=weights(qq,kk,ro)
        out[j]=dict(q=qq,k=kk,v=vv,w=ww,wo=wo,h=projected(ww,vv,wo))
    return out

def hybrid(t,mode,bit,ids,ro):
    ix=c.mate_indices(len(ids),bit,ids.device); mask=(ids!=ids[ix])[:,:,None]
    assert torch.all(mask.sum(1)==1)
    q,k,v=t['q'],t['k'],t['v']
    if mode in ('q','qk','qv','qkv'):q=q[ix]
    if mode in ('k','qk','kv','qkv'):k=k[ix]
    if mode in ('v','qv','kv','qkv','pattern_v'):v=v[ix]
    if mode in ('k_token','kv_token'):k=torch.where(mask,k[ix],k)
    if mode in ('v_token','kv_token'):v=torch.where(mask,v[ix],v)
    if mode=='kv_other':k=torch.where(mask,k,k[ix]);v=torch.where(mask,v,v[ix])
    w=t['w'][ix] if mode in ('pattern','pattern_v') else weights(q,k,ro)
    return projected(w,v,t['wo'])

@torch.inference_mode()
def forward(model,ids,ro,patches=None,capture=False):
    # patches: site -> ('fixed'|'dynamic'|'restore', tensor or donor bit).
    patches=patches or {}; b,n=ids.shape; device=ids.device; ix=torch.arange(b,device=device)
    x=model.embed(ids)+model.pos(torch.arange(n,device=device));mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
    cache={}; maxerr=0.
    for j,block in enumerate(model.blocks):
        pre=f'L{j}.';z=block.n1(x);a=block.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        hp={s:patches[s] for s in patches if s.startswith(pre+'head')}
        if capture or hp:
            hh=parts(block.attn,z,ro);recon=sum(t['h'] for t in hh.values())+block.attn.out_proj.bias
            err=float((recon-a[ix,ro]).abs().max());maxerr=max(maxerr,err);assert torch.allclose(recon,a[ix,ro],atol=.003,rtol=.0002),(pre,err)
            if capture:cache.update({pre+f'head{k}':t for k,t in hh.items()})
            if hp:
                delta=torch.zeros_like(a[ix,ro])
                for site,(kind,val) in hp.items():
                    h=hh[int(site[-1])]['h'];target=h[c.mate_indices(b,val,device)] if kind=='dynamic' else val
                    delta+=target-h
                a=a.clone();a[ix,ro]+=delta
        x=x+a; mlp=block.mlp(block.n2(x));site=pre+'mlp'
        if capture:cache[site]=mlp[ix,ro].clone()
        if site in patches:
            kind,val=patches[site];assert kind=='restore';mlp=mlp.clone();mlp[ix,ro]=val
        x=x+mlp
    return model.out(model.norm(x[ix,ro])),cache,maxerr

def layer(site):return int(site[1])
def conditions(seed):
    rh,eh,rc,ec=HEADS[seed]; out=[]
    # Names, expected target query XOR bit, intervention family.
    for label,a,b in [('candidate',rh,eh),('control',rc,ec)]:
        for kind in ('fixed','dynamic'):
            for bit in (0,1,2,3):out.append((f'{label}/{kind}/{bit}',bit,('joint',a,b,kind,bit)))
    for label,site,bit in [('role',rh,1),('event',eh,2),('role_cross',rh,2),('event_cross',eh,1),('role_control',rc,1),('event_control',ec,2)]:
        for mode in HYBRIDS:out.append((f'hybrid/{label}/{mode}',bit,('hybrid',site,mode,bit)))
    # Controlled downstream restoration, NOT direct edge/path attribution.
    for label,src,bit in [('role',rh,1),('event',eh,2)]:
        for dst in [f'L{j}.{s}' for j in range(3) for s in ('mlp',*(f'head{k}' for k in range(4)))]:
            if layer(dst)>layer(src) or (layer(dst)==layer(src) and dst.endswith('mlp')):
                out.append((f'restore/{label}/{dst}',bit,('restore',src,dst,bit)))
    return out

def build_patch(spec,cache,ids,ro):
    def donor(s,bit):return cache[s]['h'][c.mate_indices(len(ids),bit,ids.device)]
    if spec[0]=='joint':
        _,a,b,kind,bit=spec;patch={}
        for site,axis in [(a,1),(b,2)]:
            if bit&axis:patch[site]=(kind,axis if kind=='dynamic' else donor(site,axis))
        return patch
    if spec[0]=='hybrid':
        _,s,mode,bit=spec;return {s:('fixed',hybrid(cache[s],mode,bit,ids,ro))}
    _,src,dst,bit=spec;return {src:('fixed',donor(src,bit)),dst:('restore',cache[dst] if dst.endswith('mlp') else cache[dst]['h'])}

def outcome(logits,labels,choices,bit):
    l=logits.double().cpu().reshape(-1,4,logits.shape[-1]);g=labels.cpu().reshape(-1,4);q=torch.arange(4);target=q^bit
    z=l.gather(-1,g[:,None,:].expand(-1,4,-1)); lp=l.log_softmax(-1).gather(-1,g[:,None,:].expand(-1,4,-1)); pred=choices.cpu().gather(1,l.reshape(-1,l.shape[-1]).gather(1,choices.cpu()).argmax(1,keepdim=True)).reshape(-1,4)
    ok=pred==g[:,target]; mass=torch.logaddexp(lp[:,q,target],lp[:,q,target^1]).exp(); role=(z[:,q,target]-z[:,q,target^1]).sigmoid()
    avg=lambda t:float(t.double().mean())
    return dict(target_accuracy=avg(ok),target_joint=avg(ok.all(1)),target_probability=avg(lp[:,q,target].exp()),target_event_mass=avg(mass),target_conditional_role=avg(role),original_accuracy=avg(pred==g),original_event_mass=avg(torch.logaddexp(lp[:,q,q],lp[:,q,q^1]).exp()),original_conditional_role=avg((z[:,q,q]-z[:,q,q^1]).sigmoid()))

@torch.inference_mode()
def analyse(model,enc,device,seed):
    c.audit_pairs(*enc);ids,ro,y,ch=enc;cond=conditions(seed);outs={name:[] for name,_,_ in cond};base=[];maxerr=0.;nativeerr=0.;hybriderr=0.;selferr=0.
    for i in range(0,len(ids),CHUNK):
        a=ids[i:i+CHUNK].to(device);b=ro[i:i+CHUNK].to(device);l,cache,err=forward(model,a,b,capture=True);native=model(a,b);nativeerr=max(nativeerr,float((l-native).abs().max()));assert torch.allclose(l,native,atol=.0001,rtol=.00001)
        maxerr=max(maxerr,err);base.append(l.cpu())
        for s in HEADS[seed]:
            for bit in (1,2):
                full=hybrid(cache[s],'qkv',bit,a,b); pattern=hybrid(cache[s],'pattern_v',bit,a,b);target=cache[s]['h'][c.mate_indices(len(a),bit,device)]
                for val in (full,pattern):
                    hybriderr=max(hybriderr,float((val-target).abs().max()));assert torch.allclose(val,target,atol=.003,rtol=.0002),(s,bit)
            selfl,_,e=forward(model,a,b,{s:('fixed',cache[s]['h'])});selferr=max(selferr,float((selfl-l).abs().max()));assert torch.allclose(selfl,l,atol=.0001,rtol=.00001)
        for name,bit,spec in cond:
            p=build_patch(spec,cache,a,b);z,_,e=forward(model,a,b,p);maxerr=max(maxerr,e);assert torch.isfinite(z).all();outs[name].append(z.cpu())
    base=torch.cat(base);rows=[]
    for name,bit,spec in cond:
        z=torch.cat(outs[name]);post=outcome(z,y,ch,bit);pre=outcome(base,y,ch,bit)
        rows.append(dict(condition=name,target_bit=bit,**post,**{k+'_delta':post[k]-pre[k] for k in post}))
    return rows,dict(native_logit=nativeerr,head_reconstruction=maxerr,hybrid_reconstruction=hybriderr,self_patch=selferr)

def run(seed,step,device):
    loc,d,r,p=c.parents();m,fixtures,tests=manifest();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,MANIFEST):r.committed(f)
    p.require_frozen();assert seed in SEEDS and step in STEPS
    path=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if path.exists():assert read(path)['protocol_digest']==m['protocol_digest'];print('ALREADY SEALED',seed,step);return
    ck=torch.load(r.checkpoint_path(seed,step),map_location='cpu',weights_only=True);ref=read(r.report_path(seed,step));assert (ck['study'],ck['seed'],ck['step'],ck['protocol_digest'],ck['schedule_prefix_sha256'])==(r.STUDY,seed,step,ref['protocol_digest'],ref['schedule_prefix_sha256'])
    model=p.old.Model(len(fixtures[-1])).to(device).eval();model.load_state_dict(ck['models'][ARM],strict=True);rows=[];checks=[]
    for pop in c.POPS:
        for voice in c.VOICES:
            full=tests[pop][voice];sc=p.old.score_native(model,{voice:full},device)[voice];nd=d.verify(sc,ref['outcomes'][ARM][pop][voice]);enc=c.subset(full,m['protocol']['samples'][f'{pop}/{voice}']['indices']);rr,cc=analyse(model,enc,device,seed)
            rows.extend(dict(population=pop,voice=voice,**x) for x in rr);checks.append(dict(population=pop,voice=voice,native_score_delta=nd,**cc));print('PATHWAYS',seed,step,pop,voice,flush=True)
    p.save_once(path,dict(seed=seed,step=step,protocol_digest=m['protocol_digest'],rows=rows,checks=checks));print('SEALED',seed,step,flush=True)

def summary():
    *_,p=c.parents();m=read(MANIFEST);allrows=[];checks=[];compact=OUT/'compact';compact.mkdir(exist_ok=True)
    for seed in SEEDS:
        for step in STEPS:
            o=read(OUT/'per_checkpoint'/f'seed{seed}_step{step}.json');assert (o['seed'],o['step'],o['protocol_digest'])==(seed,step,m['protocol_digest']);assert len(o['rows'])==4*len(conditions(seed));assert len(o['checks'])==4
            rows=o['rows'];allrows.extend(dict(seed=seed,step=step,**x) for x in rows);checks.extend(o['checks']);agg=[]
            for pop in c.POPS:
                for name,bit,_ in conditions(seed):
                    g=[x for x in rows if x['population']==pop and x['condition']==name];assert set(x['voice'] for x in g)==set(c.VOICES) and len(g)==2
                    agg.append(dict(population=pop,condition=name,target_bit=bit,**{k:round(sum(x[k] for x in g)/2,7) for k in g[0] if k not in ('population','condition','voice','target_bit')}))
            write_csv(compact/f'seed{seed}_step{step}.csv',agg)
    write_csv(OUT/'rows.csv',allrows)
    p.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',n_rows=len(allrows),max_checks={k:max(x[k] for x in checks) for k in checks[0] if k not in ('population','voice')}));print('SUMMARY PASS',len(allrows),flush=True)

def write_csv(path,rows):
    # Idempotent derived output; refuses drift.
    import io
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);data=f.getvalue()
    if path.exists():assert path.read_bytes()==data.encode(),path
    else:path.write_bytes(data.encode())

def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel().eval();a=torch.tensor([[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,8,7],[0,1,2,3,9,5,6,7],[0,1,2,3,9,5,8,7]]).repeat(2,1);b=torch.full((8,),7);g=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8)
    for seed in (16101,16127):
        rows,checks=analyse(model,(a,b,g,ch),torch.device('cpu'),seed);assert len(rows)==len(conditions(seed));assert checks['native_logit']==0;assert checks['self_patch']<1e-5
        lookup={x['condition']:x for x in rows};assert lookup['candidate/fixed/0']=={**lookup['candidate/dynamic/0'],'condition':'candidate/fixed/0'}
        for bit in (1,2):
            f=lookup[f'candidate/fixed/{bit}'];h=lookup[f'candidate/dynamic/{bit}'];assert abs(f['target_probability']-h['target_probability'])<1e-6
    print('SMOKE PASS: joint fixed/dynamic, QKV reconstruction, self-patch, restoration, both layer layouts',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));a.add_argument('--seed',type=int);a.add_argument('--step',type=int);a.add_argument('--device',choices=('cpu','mps'),default='mps');v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.seed,v.step,torch.device(v.device))
if __name__=='__main__':main()
