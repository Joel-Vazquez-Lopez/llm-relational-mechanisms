#!/usr/bin/env python3
"""Source-conditioned control localisation; frozen exploratory follow-up."""
import argparse,itertools,json
from pathlib import Path
from random import Random
import torch
import role_binding_causal_validation_v0 as b
w=b.w;c=b.c;ROOT=c.ROOT;OUT=ROOT/'reports/role_binding_localisation_v0';MANIFEST=OUT/'study_manifest.json'
SCRIPT=Path(__file__).resolve();PROTOCOL=ROOT/'paper/ROLE_BINDING_LOCALISATION_V0_PROTOCOL.md'
SEEDS=b.SEEDS;STEPS=b.STEPS;N=64;CHUNK=128;SAMPLE_SEED=20261129
GROUPS=('names','source_other','separator','event_token','role_token','query_other','readout','non_names','visible')
MODES=('k','v','kv')

def read(p):return json.loads(p.read_text())

def replace(enc,new,sep):
    ids,ro,y,ch=enc;g=y.reshape(-1,4);a=ids.reshape(-1,4,ids.shape[-1]);z=a.clone();mask=torch.zeros_like(a,dtype=torch.bool)
    assert new.shape==g.shape and torch.all(new.sort(1).values.diff(dim=1)>0)
    for j in range(4):
        m=a==g[:,j,None,None];assert m.any(-1).all();z=torch.where(m,new[:,j,None,None],z);mask|=m
    assert torch.equal(z!=a,mask)
    s=(ids==sep).long().argmax(1);assert (ids==sep).sum(1).eq(1).all();assert not (mask.reshape_as(ids)&(torch.arange(ids.shape[1])[None,:]>=s[:,None])).any()
    choices=new.sort(1).values[:,None,:].expand(-1,4,-1).reshape(-1,4).clone();out=(z.reshape_as(ids),ro,new.flatten(),choices);c.audit_pairs(*out)
    return out,mask.reshape_as(ids)

def disjoint_names(enc,pool):
    gold=enc[2].reshape(-1,4);pg=pool[2].reshape(-1,4);out=[]
    for i,g in enumerate(gold):
        # deterministic search over held-out source tuples, no model outputs
        for j in range(len(pg)):
            q=pg[(i*137+j)%len(pg)]
            if not torch.isin(q,g).any():out.append(q);break
        else:raise ValueError('no disjoint known-name quartet')
    return torch.stack(out)

def manifest():
    loc,d,r,p=c.parents();bm,fixtures,tests=b.manifest();assert read(b.MANIFEST)==bm
    bs=read(b.OUT/'analysis_summary.json');assert bs['protocol_digest']==bm['protocol_digest'] and bs['n_rows']==3840
    cm=read(c.MANIFEST);wm=read(w.MANIFEST);full=tests[b.POP];n=len(full[c.VOICES[0]][0])//4;excluded=set(bm['protocol']['indices'])
    for v in c.VOICES:
        key=f'{b.POP}/{v}';excluded.update(cm['protocol']['samples'][key]['source_indices']);excluded.update(wm['protocol']['samples'][key]['indices'])
    idx=sorted(Random(SAMPLE_SEED).sample(sorted(set(range(n))-excluded),N));sep=fixtures[-1]['<sep>'];samples={};replacement=None
    for v in c.VOICES:
        e=c.subset(full[v],idx);g=e[2].reshape(-1,4);new=disjoint_names(e,full[v]);assert not (new[:,:,None]==g[:,None,:]).any()
        if replacement is None:replacement=new
        else:assert torch.equal(replacement,new)
        samples[v]={}
        for name,ng in [('mixed',g[:,[1,2,3,0]]),('disjoint',new)]:
            rr,mask=replace(e,ng,sep);samples[v][name]=p.digest([x.tolist() for x in rr]);masks(e[0],e[1],mask,sep)
        samples[v]['original']=p.digest([x.tolist() for x in e])
    proto=dict(study='ROLE-BINDING-LOCALISATION-v0',classification='exploratory',parent_digest=bm['protocol_digest'],parent_summary_sha=p.sha(b.OUT/'analysis_summary.json'),script_sha=p.sha(SCRIPT),protocol_sha=p.sha(PROTOCOL),indices=idx,disjoint_names=replacement.tolist(),samples=samples,heads={str(k):list(v) for k,v in w.HEADS.items()},seeds=list(SEEDS),steps=list(STEPS),population=b.POP,n_sources=N,chunk=CHUNK,sample_seed=SAMPLE_SEED,separator=sep,groups=list(GROUPS),modes=list(MODES),offsets=list(range(-12,13)))
    return dict(protocol=proto,protocol_digest=p.digest(proto)),fixtures,tests

def masks(ids,ro,changed,sep):
    pos=torch.arange(ids.shape[1],device=ids.device)[None,:];s=(ids==sep).long().argmax(1)[:,None];assert (ids==sep).sum(1).eq(1).all();visible=pos<=ro[:,None]
    ev=ids!=ids[c.mate_indices(len(ids),2,ids.device)];role=ids!=ids[c.mate_indices(len(ids),1,ids.device)]
    assert ev.sum(1).eq(1).all() and role.sum(1).eq(1).all() and not (ev&role).any()
    assert not ((ev|role)&((pos<=s)|(pos>=ro[:,None]))).any()
    out=dict(names=changed,source_other=(pos<s)&~changed,separator=pos==s,event_token=ev,role_token=role,query_other=(pos>s)&(pos<ro[:,None])&~ev&~role,readout=pos==ro[:,None],non_names=visible&~changed,visible=visible)
    assert torch.equal(sum(out[k].int() for k in GROUPS[:7]),visible.int())
    for off in range(-12,13):out[f'sep{off:+d}']=(pos==s+off)&visible
    return out

def select(t,ix):return {k:(v if k=='wo' else v[ix]) for k,v in t.items()}

def hybrid(rec,don,mask,mode,ro):
    q=don['q'] if 'q' in mode else rec['q'];k=rec['k'];v=rec['v'];m=mask[:,:,None]
    if 'k' in mode:k=torch.where(m,don['k'],k)
    if 'v' in mode:v=torch.where(m,don['v'],v)
    return w.projected(w.weights(q,k,ro),v,rec['wo'])

def target_metrics(logits,recipient,donor,bit):
    # Full vocabulary and union candidate accuracy; recipient-only accuracy is not enough for disjoint donors.
    z=logits.double().cpu();r=recipient[2].reshape(-1,4);d=donor[2].reshape(-1,4);q=torch.arange(4)^bit;rt=r[:,q].flatten();dt=d[:,q].flatten();ix=torch.arange(len(z));lp=z.log_softmax(-1);pred=z.argmax(-1)
    union=torch.cat((recipient[3],donor[3]),1);up=union.gather(1,z.gather(1,union).argmax(1,keepdim=True)).flatten();rp=recipient[3].gather(1,z.gather(1,recipient[3]).argmax(1,keepdim=True)).flatten()
    assert (rt!=dt).all()
    avg=lambda a:float(a.double().mean());return dict(recipient_full=avg(pred==rt),donor_full=avg(pred==dt),recipient_union=avg(up==rt),donor_union=avg(up==dt),recipient_restricted=avg(rp==rt),recipient_joint_union=avg((up==rt).reshape(-1,4).all(1)),donor_joint_union=avg((up==dt).reshape(-1,4).all(1)),recipient_probability=avg(lp[ix,rt].exp()),donor_probability=avg(lp[ix,dt].exp()))

def conditions(seed):
    rh,eh,rc,ec=w.HEADS[seed];out=[]
    for label,site,bit in [('role',rh,1),('event_same',eh,0),('event_flip',eh,2),('event_control',ec,2)]:
        out += [(f'{label}/recipient',bit,('full',site,'recipient')),(f'{label}/donor',bit,('full',site,'donor')),(f'{label}/cross_voice',bit,('full',site,'voice')),(f'{label}/q',bit,('hybrid',site,'visible','q')),(f'{label}/qkv',bit,('hybrid',site,'visible','qkv'))]
        for group in GROUPS:
            for mode in MODES:out.append((f'{label}/{group}/{mode}',bit,('hybrid',site,group,mode)))
        if label in ('event_same','event_flip'):
            for off in range(-12,13):out.append((f'{label}/sep{off:+d}/kv',bit,('hybrid',site,f'sep{off:+d}','kv')))
    # All L1 head subsets, source-name V/KV. Same query, original source donor.
    for k in range(1,5):
        for hs in itertools.combinations(range(4),k):
            for mode in ('v','kv'):out.append((f'entity/L1/{"".join(map(str,hs))}/{mode}',0,('subset',hs,mode)))
    return out

@torch.inference_mode()
def analyse(model,original,other,recipient,changed,sep,device,seed):
    cond=conditions(seed);outputs={n:[] for n,_,_ in cond};coverage={n:[] for n,_,_ in cond};checks=dict(native_logit=0.,head_reconstruction=0.,self_patch=0.,hybrid_reconstruction=0.);base=[]
    for i in range(0,len(original[0]),CHUNK):
        a,ro=original[0][i:i+CHUNK].to(device),original[1][i:i+CHUNK].to(device);x=recipient[0][i:i+CHUNK].to(device);ov,orr=other[0][i:i+CHUNK].to(device),other[1][i:i+CHUNK].to(device);m=masks(x,ro,changed[i:i+CHUNK].to(device),sep)
        l,cr,e=w.forward(model,x,ro,capture=True);lo,cd,f=w.forward(model,a,ro,capture=True);lv,cv,g=w.forward(model,ov,orr,capture=True);base.append(l.cpu());checks['head_reconstruction']=max(checks['head_reconstruction'],e,f,g)
        for ids,pos,z in [(x,ro,l),(a,ro,lo),(ov,orr,lv)]:
            err=float((model(ids,pos)-z).abs().max());checks['native_logit']=max(checks['native_logit'],err);assert torch.allclose(model(ids,pos),z,atol=.0001,rtol=.00001)
        for site in w.HEADS[seed]:
            z,_,e=w.forward(model,x,ro,{site:('fixed',cr[site]['h'])});checks['self_patch']=max(checks['self_patch'],float((z-l).abs().max()));assert torch.allclose(z,l,atol=.0001,rtol=.00001)
        for name,bit,spec in cond:
            ix=c.mate_indices(len(x),bit,device);kind=spec[0];mask=m['visible'];patch={}
            if kind=='full':
                _,site,which=spec;cache={'recipient':cr,'donor':cd,'voice':cv}[which];patch[site]=('fixed',cache[site]['h'][ix])
            elif kind=='hybrid':
                _,site,group,mode=spec;mask=m[group];rec=select(cr[site],ix);don=select(cd[site],ix);h=hybrid(rec,don,mask,mode,ro);patch[site]=('fixed',h)
                if group=='visible' and mode=='qkv':
                    err=float((h-don['h']).abs().max());checks['hybrid_reconstruction']=max(checks['hybrid_reconstruction'],err);assert torch.allclose(h,don['h'],atol=.003,rtol=.0002)
            else:
                _,heads,mode=spec;mask=m['names']
                for h in heads:
                    site=f'L1.head{h}';patch[site]=('fixed',hybrid(cr[site],cd[site],mask,mode,ro))
            z,_,e=w.forward(model,x,ro,patch);checks['head_reconstruction']=max(checks['head_reconstruction'],e);assert torch.isfinite(z).all();outputs[name].append(z.cpu());coverage[name].append(mask.any(1).cpu())
    rows=[dict(condition='baseline/recipient',target_bit=0,coverage=1.,**target_metrics(torch.cat(base),recipient,original,0))]
    for name,bit,spec in cond:
        rows.append(dict(condition=name,target_bit=bit,coverage=float(torch.cat(coverage[name]).float().mean()),**target_metrics(torch.cat(outputs[name]),recipient,original,bit)))
    return rows,checks

def prepare():
    *_,p=c.parents();m,_,_=manifest()
    if MANIFEST.exists():assert read(MANIFEST)==m
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'],flush=True)

def run(seed,step,device):
    loc,d,r,p=c.parents();m,fixtures,tests=manifest();assert read(MANIFEST)==m
    for f in (SCRIPT,PROTOCOL,MANIFEST):r.committed(f)
    p.require_frozen();assert seed in SEEDS and step in STEPS
    path=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if path.exists():assert read(path)['protocol_digest']==m['protocol_digest'];print('ALREADY SEALED',seed,step);return
    ck=torch.load(r.checkpoint_path(seed,step),map_location='cpu',weights_only=True);ref=read(r.report_path(seed,step));assert (ck['study'],ck['seed'],ck['step'],ck['protocol_digest'],ck['schedule_prefix_sha256'])==(r.STUDY,seed,step,ref['protocol_digest'],ref['schedule_prefix_sha256'])
    model=p.old.Model(len(fixtures[-1])).to(device).eval();model.load_state_dict(ck['models'][w.ARM],strict=True);rows=[];checks=[];proto=m['protocol'];idx=proto['indices'];sep=proto['separator']
    for v in c.VOICES:
        full=tests[b.POP][v];nd=d.verify(p.old.score_native(model,{v:full},device)[v],ref['outcomes'][w.ARM][b.POP][v]);other=next(x for x in c.VOICES if x!=v);orig=c.subset(full,idx);ov=c.subset(tests[b.POP][other],idx);g=orig[2].reshape(-1,4)
        for name,new in [('mixed',g[:,[1,2,3,0]]),('disjoint',torch.tensor(proto['disjoint_names']))]:
            rec,mask=replace(orig,new,sep);rr,cc=analyse(model,orig,ov,rec,mask,sep,device,seed);rows.extend(dict(voice=v,source_type=name,**x) for x in rr);checks.append(dict(voice=v,source_type=name,native_score_delta=nd,**cc));print('LOCALISATION',seed,step,v,name,flush=True)
    p.save_once(path,dict(seed=seed,step=step,protocol_digest=m['protocol_digest'],rows=rows,checks=checks));print('SEALED',seed,step,flush=True)

def summary():
    *_,p=c.parents();m=read(MANIFEST);rows=[];checks=[];compact=OUT/'compact';compact.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        for step in STEPS:
            a=read(OUT/'per_checkpoint'/f'seed{seed}_step{step}.json');assert (a['seed'],a['step'],a['protocol_digest'])==(seed,step,m['protocol_digest']);assert len(a['rows'])==4*(1+len(conditions(seed)));checks.extend(a['checks']);rows.extend(dict(seed=seed,step=step,**x) for x in a['rows']);agg=[]
            for source in ('mixed','disjoint'):
                for name in ['baseline/recipient']+[x[0] for x in conditions(seed)]:
                    g=[x for x in a['rows'] if x['source_type']==source and x['condition']==name];assert len(g)==2 and set(x['voice'] for x in g)==set(c.VOICES);agg.append(dict(source_type=source,condition=name,target_bit=g[0]['target_bit'],**{k:round(sum(x[k] for x in g)/2,8) for k in g[0] if k not in ('source_type','condition','target_bit','voice')}))
            w.write_csv(compact/f'seed{seed}_step{step}.csv',agg)
    w.write_csv(OUT/'rows.csv',rows);p.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',n_rows=len(rows),max_checks={k:max(x[k] for x in checks) for k in checks[0] if k not in ('voice','source_type')}));print('SUMMARY PASS',len(rows),flush=True)

def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=c.ToyModel().eval();a=torch.tensor([[0,1,2,3,4,10,5,6,7],[0,1,2,3,4,10,5,8,7],[0,1,2,3,4,10,9,6,7],[0,1,2,3,4,10,9,8,7]]).repeat(2,1);ro=torch.full((8,),8);y=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8);orig=(a,ro,y,ch)
    for new in [torch.tensor([[2,3,4,1]]*2),torch.tensor([[11,12,13,14]]*2)]:
        rec,mask=replace(orig,new,10)
        for seed in (16101,16127):
            rows,checks=analyse(model,orig,orig,rec,mask,10,torch.device('cpu'),seed);assert len(rows)==1+len(conditions(seed));assert checks['native_logit']==0;assert checks['self_patch']<1e-5
            lookup={x['condition']:x for x in rows}
            for label in ('role','event_same','event_flip','event_control'):
                for key in ('recipient_probability','donor_probability'):
                    assert abs(lookup[label+'/qkv'][key]-lookup[label+'/donor'][key])<1e-6
        # Independent synthetic logits prove donor vs recipient target indexing (including disjoint names).
        for bit in (0,1,2,3):
            logits=torch.full((8,20),-10.);target=y.reshape(-1,4)[:,torch.arange(4)^bit].flatten();logits[torch.arange(8),target]=10.;m=target_metrics(logits,rec,orig,bit);assert m['donor_full']==1 and m['recipient_full']==0 and m['donor_union']==1
    print('SMOKE PASS: disjoint names, position masks, full/union scoring, QKV reconstruction, both layouts',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));a.add_argument('--seed',type=int);a.add_argument('--step',type=int);a.add_argument('--device',choices=('cpu','mps'),default='mps');v=a.parse_args()
    if v.phase=='smoke':smoke()
    elif v.phase=='prepare':prepare()
    elif v.phase=='summary':summary()
    else:run(v.seed,v.step,torch.device(v.device))
if __name__=='__main__':main()
