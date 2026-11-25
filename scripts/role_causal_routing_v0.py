#!/usr/bin/env python3
"""Exploratory, within-model readout-site interventions on frozen checkpoints."""
import argparse, csv, json, math
from pathlib import Path
from random import Random
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/role_causal_routing_v0'
MANIFEST=OUT/'study_manifest.json'
PROTOCOL=ROOT/'paper/ROLE_CAUSAL_ROUTING_V0_PROTOCOL.md'
SCRIPT=Path(__file__).resolve()
STUDY='ROLE-CAUSAL-ROUTING-v0'
SEEDS=(16101,16127,16143,16167)
STEPS=(6400,12800,25600,51200)
ARMS=('roles_1','roles_2','roles_4','roles_8')
POPS=('unseen_roles_rich','unseen_quartets_rich')
VOICES=('AP','PA')
N=128; CHUNK=128; SAMPLE_SEED=20261126
SITES=[f'L{i}.{s}' for i in range(3) for s in ('attention','mlp','residual',*(f'head{h}' for h in range(4)))]
MODES=('role_swap','event_swap','role_mean','event_mean','zero')
# Only heads receive zero ablations. Pair means preserve the common part of a pair.
CONDITIONS=[(s,m) for s in SITES for m in MODES if m!='zero' or '.head' in s]

def read(path): return json.loads(path.read_text())
def parents():
    import role_representation_localisation_v0 as loc
    return loc,loc.diag,loc.r,loc.p

def sample_indices(n):
    assert n>=N
    return sorted(Random(SAMPLE_SEED+n).sample(range(n),N))

def manifest():
    loc,d,r,p=parents(); lm,fixtures=loc.manifest()
    assert read(loc.MANIFEST)==lm,'Localisation manifest drift'
    summary=read(loc.OUT/'analysis_summary.json')
    assert summary['protocol_digest']==lm['protocol_digest'] and summary['n_probe_rows']==33792 and summary['n_sensitivity_rows']==5632
    train,held,training,withheld,vocab=fixtures
    tests=p.evaluations(train,held,training,withheld,vocab)
    samples={}
    for pop in POPS:
        for voice in VOICES:
            enc=tests[pop][voice]; n=len(enc[0])//4; idx=sample_indices(n)
            a,b,y,c=subset(enc,idx); audit_pairs(a,b,y,c)
            samples[f'{pop}/{voice}']=dict(source_indices=idx,full_sources=n,encoded_sha=p.digest([x.tolist() for x in (a,b,y,c)]))
    proto=dict(study=STUDY,classification='exploratory_discovery',localisation_digest=lm['protocol_digest'],localisation_summary_sha=p.sha(loc.OUT/'analysis_summary.json'),script_sha=p.sha(SCRIPT),protocol_sha=p.sha(PROTOCOL),seeds=list(SEEDS),steps=list(STEPS),arms=list(ARMS),populations=list(POPS),voices=list(VOICES),conditions=[list(x) for x in CONDITIONS],sample_seed=SAMPLE_SEED,n_sources=N,chunk=CHUNK,samples=samples,inputs=lm['protocol']['inputs'],head_reconstruction_atol=.003,head_reconstruction_rtol=.0002,native_atol=.0001,native_rtol=.00001)
    return dict(protocol=proto,protocol_digest=p.digest(proto)),fixtures,tests

def prepare():
    _,_,_,p=parents(); m,_,_=manifest()
    if MANIFEST.exists(): assert read(MANIFEST)==m,'Manifest drift'
    else:p.save_once(MANIFEST,m)
    print('PREPARE PASS',m['protocol_digest'], 'conditions',len(CONDITIONS),flush=True)

def subset(enc,idx):
    ix=(torch.tensor(idx)[:,None]*4+torch.arange(4)).flatten()
    return tuple(x[ix] for x in enc)

def audit_pairs(ids,ro,labels,choices):
    a=ids.reshape(-1,4,ids.shape[-1]); b=ro.reshape(-1,4); gold=labels.reshape(-1,4)
    assert torch.all(b==b[:,:1]); assert torch.all(gold.sort(1).values.diff(dim=1)>0)
    assert torch.all(choices.reshape(-1,4,4).sort(-1).values==gold.sort(1).values[:,None,:])
    for bit in (1,2):
        diff=a!=a[:,torch.arange(4)^bit]; assert torch.all(diff.sum(-1)==1)
        assert torch.all((diff.nonzero()[:,-1]) < b.flatten())

def mate_indices(n,bit,device):
    assert n%4==0
    return torch.arange(n,device=device)^bit

@torch.inference_mode()
def head_parts(attn,z,ro):
    """Readout-only per-head contributions after the output projection, no bias."""
    b,n,w=z.shape; h=attn.num_heads; d=w//h; ix=torch.arange(b,device=z.device)
    q,k,v=F.linear(z,attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    q=q[ix,ro].reshape(b,h,d)
    k=k.reshape(b,n,h,d).transpose(1,2); v=v.reshape(b,n,h,d).transpose(1,2)
    scores=(q.unsqueeze(-2)@k.transpose(-1,-2)).squeeze(-2)/math.sqrt(d)
    scores=scores.masked_fill(torch.arange(n,device=z.device)[None,None,:]>ro[:,None,None],-torch.inf)
    weights=scores.softmax(-1)
    raw=(weights.unsqueeze(-2)@v).squeeze(-2)
    # Each head contributes its own columns of W_O; output bias is untouched.
    wo=attn.out_proj.weight.reshape(w,h,d).permute(1,0,2)
    parts=torch.einsum('bhd,hwd->bhw',raw,wo)
    return parts,weights

@torch.inference_mode()
def forward(model,ids,ro, capture=False, patch=None):
    b,n=ids.shape; device=ids.device; ix=torch.arange(b,device=device)
    x=model.embed(ids)+model.pos(torch.arange(n,device=device)); mask=torch.ones(n,n,dtype=torch.bool,device=device).triu(1)
    cache={}; attention={}; maxerr=0.
    def replace(t,site):
        if patch is not None and patch[0]==site:
            t=t.clone(); t[ix,ro]=patch[1]
        return t
    for j,block in enumerate(model.blocks):
        prefix=f'L{j}.'; z=block.n1(x); a=block.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        if capture:
            parts,weights=head_parts(block.attn,z,ro)
            reconstructed=parts.sum(1)+block.attn.out_proj.bias
            err=float((reconstructed-a[ix,ro]).abs().max()); maxerr=max(maxerr,err)
            assert torch.allclose(reconstructed,a[ix,ro],atol=.003,rtol=.0002),('head reconstruction',j,err)
            cache[prefix+'attention']=a[ix,ro].clone()
            for h in range(parts.shape[1]):
                cache[prefix+f'head{h}']=parts[:,h].clone(); attention[prefix+f'head{h}']=weights[:,h].clone()
        if patch is not None and patch[0].startswith(prefix+'head'):
            # Earlier blocks unchanged: replace only the chosen projected head contribution.
            a=a.clone(); a[ix,ro]+=patch[1]
        a=replace(a,prefix+'attention'); x=x+a
        mlp=block.mlp(block.n2(x))
        if capture:cache[prefix+'mlp']=mlp[ix,ro].clone()
        mlp=replace(mlp,prefix+'mlp'); x=x+mlp
        if capture:cache[prefix+'residual']=x[ix,ro].clone()
        x=replace(x,prefix+'residual')
    logits=model.out(model.norm(x[ix,ro]))
    return logits,cache,attention,maxerr

def patch_value(cache,site,mode):
    h=cache[site]; n=len(h); bit=1 if mode.startswith('role') else 2
    if mode=='zero': target=torch.zeros_like(h)
    else:
        donor=h[mate_indices(n,bit,h.device)]
        target=(h+donor)*.5 if mode.endswith('mean') else donor
    return target-h if '.head' in site else target

def metrics(logits,labels,choices):
    # Four queries in semantic order. Use full vocabulary mass, not restricted softmax.
    l=logits.double().cpu().reshape(-1,4,logits.shape[-1]); g=labels.cpu().reshape(-1,4); q=torch.arange(4)
    lp=l.log_softmax(-1); z=l.gather(-1,g[:,None,:].expand(-1,4,-1)); gp=lp.gather(-1,g[:,None,:].expand(-1,4,-1))
    same=torch.logaddexp(gp[:,q,q],gp[:,q,q^1]); other=torch.logaddexp(gp[:,q,q^2],gp[:,q,q^3])
    rm=z[:,q,q]-z[:,q,q^1]; donor_event_role=z[:,q,q^2]-z[:,q,q^3]
    pred=choices.cpu().gather(1,l.reshape(-1,l.shape[-1]).gather(1,choices.cpu()).argmax(1,keepdim=True)).reshape(-1,4)
    ok=pred==g
    avg=lambda t:float(t.double().mean())
    return dict(event_mass=avg(same.exp()),other_event_mass=avg(other.exp()),conditional_role_probability=avg(rm.sigmoid()),donor_event_role_probability=avg(donor_event_role.sigmoid()),event_log_odds=avg(same-other),role_margin=avg(rm),donor_event_role_margin=avg(donor_event_role),event_nll=avg(-same),role_nll=avg(same-gp[:,q,q]),cross_entropy=avg(-gp[:,q,q]),C_query=avg(ok.all(1)),accuracy4=avg(ok),full_vocab_accuracy=avg(l.argmax(-1)==g),role_donor_accuracy=avg(pred==g[:,q^1]),event_donor_accuracy=avg(pred==g[:,q^2]))

def deltas(post,base):return {k+'_delta':post[k]-base[k] for k in post}

@torch.inference_mode()
def analyse(model,enc,device):
    ids,ro,labels,choices=enc; audit_pairs(*enc)
    base=[]; native=[]; outputs={key:[] for key in CONDITIONS}; attrows={s:[] for s in SITES if '.head' in s}; checks={'head_reconstruction':0.,'native_logit':0.,'self_patch':0.}
    for start in range(0,len(ids),CHUNK):
        a=ids[start:start+CHUNK].to(device); b=ro[start:start+CHUNK].to(device)
        l,c,att,e=forward(model,a,b,capture=True); ref=model(a,b)
        checks['head_reconstruction']=max(checks['head_reconstruction'],e)
        checks['native_logit']=max(checks['native_logit'],float((l-ref).abs().max()))
        assert torch.allclose(l,ref,atol=.0001,rtol=.00001)
        base.append(l.float().cpu()); native.append(ref.float().cpu())
        # Self-patch (no-op) at every site, including head delta path.
        for s in SITES:
            val=torch.zeros_like(c[s]) if '.head' in s else c[s]
            self_l=forward(model,a,b,patch=(s,val))[0]
            checks['self_patch']=max(checks['self_patch'],float((self_l-l).abs().max()))
            assert torch.allclose(self_l,l,atol=.0001,rtol=.00001),s
        for site,mode in CONDITIONS:
            v=patch_value(c,site,mode); pred=forward(model,a,b,patch=(site,v))[0]
            assert torch.isfinite(pred).all(); outputs[site,mode].append(pred.float().cpu())
        for s,w in att.items():
            rd=(a!=a[mate_indices(len(a),1,device)]); ed=(a!=a[mate_indices(len(a),2,device)])
            attrows[s].append(torch.stack(((w*rd).sum(-1),(w*ed).sum(-1)),1).cpu())
    baseline=metrics(torch.cat(base),labels,choices); native_m=metrics(torch.cat(native),labels,choices)
    for k in ('C_query','accuracy4','full_vocab_accuracy'): assert abs(baseline[k]-native_m[k])<=.001
    rows=[]
    for (site,mode),parts in outputs.items():
        post=metrics(torch.cat(parts),labels,choices)
        assert abs(post['cross_entropy']-post['event_nll']-post['role_nll'])<1e-9
        rows.append(dict(site=site,mode=mode,**post,**deltas(post,baseline)))
    attention=[dict(site=s,role_token_attention=float(torch.cat(parts)[:,0].mean()),event_token_attention=float(torch.cat(parts)[:,1].mean())) for s,parts in attrows.items()]
    return baseline,rows,attention,checks

def run(seed,step,device):
    loc,d,r,p=parents(); m,fixtures,tests=manifest(); assert read(MANIFEST)==m,'Manifest drift'
    for path in (SCRIPT,PROTOCOL,MANIFEST):r.committed(path)
    p.require_frozen(); assert seed in SEEDS and step in STEPS
    path=OUT/'per_checkpoint'/f'seed{seed}_step{step}.json'
    if path.exists():assert read(path)['protocol_digest']==m['protocol_digest'];print('ALREADY SEALED',seed,step);return
    ck=torch.load(r.checkpoint_path(seed,step),map_location='cpu',weights_only=True); ref=read(r.report_path(seed,step))
    assert (ck['seed'],ck['step'],ck['protocol_digest'],ck['schedule_prefix_sha256'])==(seed,step,ref['protocol_digest'],ref['schedule_prefix_sha256'])
    assert ck['study']==r.STUDY and set(ck['models'])==set(ARMS)
    vocab=fixtures[-1]; rows=[]; baselines=[]; attention=[]; checks=[]
    for arm in ARMS:
        model=p.old.Model(len(vocab)).to(device).eval();model.load_state_dict(ck['models'][arm],strict=True)
        for pop in POPS:
            for voice in VOICES:
                full=tests[pop][voice]; idx=m['protocol']['samples'][f'{pop}/{voice}']['source_indices']; enc=subset(full,idx)
                # Verify full-pop native scoring separately against archived metrics.
                sc=p.old.score_native(model,{voice:full},device)[voice]
                nd=d.verify(sc,ref['outcomes'][arm][pop][voice])
                baseline,rr,aa,cc=analyse(model,enc,device); meta=dict(arm=arm,population=pop,voice=voice)
                rows.extend(dict(**meta,**x) for x in rr); attention.extend(dict(**meta,**x) for x in aa)
                baselines.append(dict(**meta,**baseline));checks.append(dict(**meta,**cc,native_score_delta=nd))
                print('INTERVENED',seed,step,arm,pop,voice,flush=True)
    p.save_once(path,dict(seed=seed,step=step,device=str(device),protocol_digest=m['protocol_digest'],rows=rows,baselines=baselines,attention=attention,checks=checks));print('SEALED',seed,step,flush=True)

def summary():
    _,_,_,p=parents();m=read(MANIFEST); tables={k:[] for k in ('rows','baselines','attention','checks')}
    for seed in SEEDS:
        for step in STEPS:
            a=read(OUT/'per_checkpoint'/f'seed{seed}_step{step}.json');assert a['protocol_digest']==m['protocol_digest'] and (a['seed'],a['step'])==(seed,step)
            assert len(a['rows'])==len(ARMS)*len(POPS)*len(VOICES)*len(CONDITIONS)
            for k in tables:tables[k].extend(dict(seed=seed,step=step,**x) for x in a[k])
    for k,rows in tables.items():
        with (OUT/(k+'.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p.save_once(OUT/'analysis_summary.json',dict(protocol_digest=m['protocol_digest'],classification='exploratory',counts={k:len(v) for k,v in tables.items()},max_checks={k:max(x[k] for x in tables['checks']) for k in ('head_reconstruction','native_logit','self_patch','native_score_delta')}));print('SUMMARY PASS',flush=True)

# Independent toy architecture for an executable CPU smoke test without repository imports.
class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.n1=torch.nn.LayerNorm(96);self.attn=torch.nn.MultiheadAttention(96,4,batch_first=True,dropout=0.);self.n2=torch.nn.LayerNorm(96);self.mlp=torch.nn.Sequential(torch.nn.Linear(96,384),torch.nn.GELU(),torch.nn.Linear(384,96))
    def forward(self,x,mask):
        z=self.n1(x);x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0];return x+self.mlp(self.n2(x))
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(20,96);self.pos=torch.nn.Embedding(48,96);self.blocks=torch.nn.ModuleList(ToyBlock() for _ in range(3));self.norm=torch.nn.LayerNorm(96);self.out=torch.nn.Linear(96,20)
    def forward(self,a,b):
        n=a.shape[1];x=self.embed(a)+self.pos(torch.arange(n,device=a.device));mask=torch.ones(n,n,dtype=torch.bool,device=a.device).triu(1)
        for block in self.blocks:x=block(x,mask)
        return self.out(self.norm(x[torch.arange(len(a),device=a.device),b]))
def smoke():
    torch.set_num_threads(1);torch.manual_seed(11);model=ToyModel().eval();a=torch.tensor([[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,8,7],[0,1,2,3,9,5,6,7],[0,1,2,3,9,5,8,7]]).repeat(2,1);b=torch.full((8,),7);g=torch.tensor([1,2,3,4]*2);ch=torch.tensor([[1,2,3,4]]*8)
    base,rows,att,checks=analyse(model,(a,b,g,ch),torch.device('cpu'));assert len(rows)==len(CONDITIONS) and len(att)==12
    # Final residual swap must reproduce the donor logits; head swaps do not necessarily do so.
    l,c,_,_=forward(model,a,b,capture=True)
    for bit,mode in [(1,'role_swap'),(2,'event_swap')]:
        v=patch_value(c,'L2.residual',mode);pred=forward(model,a,b,patch=('L2.residual',v))[0];assert torch.allclose(pred,l[mate_indices(len(a),bit,a.device)],atol=1e-5,rtol=1e-5)
    print('SMOKE PASS: paired tokens, native forward, head reconstruction, self patches, all interventions, donor readout',checks,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=('smoke','prepare','run','summary'));ap.add_argument('--seed',type=int);ap.add_argument('--step',type=int);ap.add_argument('--device',choices=('cpu','mps'),default='mps');a=ap.parse_args()
    if a.phase=='smoke':smoke()
    elif a.phase=='prepare':prepare()
    elif a.phase=='summary':summary()
    else:run(a.seed,a.step,torch.device(a.device))
if __name__=='__main__':main()
