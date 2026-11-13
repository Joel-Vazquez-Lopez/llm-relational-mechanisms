#!/usr/bin/env python3
"""EXP-101 shared optimizer/checkpoint functions; training *not* CLI exposed.

This code can be exercised with artificial tiny tensors to validate optimizer
updates and exact CPU resume. It cannot launch study training by itself.
"""
from __future__ import annotations
from pathlib import Path
import os
import hashlib
import io


def init_state(seed: int, model_factory, *, device='cpu', learning_rate=.002, weight_decay=.01):
    import torch
    torch.manual_seed(seed)
    model=model_factory().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    gbase=torch.Generator(device='cpu').manual_seed(seed+800000)
    gdose=torch.Generator(device='cpu').manual_seed(seed+810000)
    return model,optimizer,gbase,gdose


def one_optimizer_step(model,optimizer,base_data,dose_data,gbase,gdose,
                       *,base_batch:int=56,dose_batch:int=8,pad_id:int=0,device='cpu'):
    import torch
    from torch.nn import functional as F
    base_x,base_readout,base_y,_=base_data
    dose_x,dose_readout,dose_y,_=dose_data
    idxb=torch.randint(len(base_x),(base_batch,),generator=gbase)
    idxd=torch.randint(len(dose_x),(dose_batch,),generator=gdose)
    a,b=base_x[idxb],dose_x[idxd]
    m=max(a.shape[1],b.shape[1])
    ids=torch.cat((F.pad(a,(0,m-a.shape[1]),value=pad_id),
                   F.pad(b,(0,m-b.shape[1]),value=pad_id)),dim=0).to(device)
    readout=torch.cat((base_readout[idxb],dose_readout[idxd])).to(device)
    labels=torch.cat((base_y[idxb],dose_y[idxd])).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model(ids,readout),labels)
    loss.backward()
    optimizer.step()
    # Expose sampled indices in tests only, not data-dependent tuning.
    return float(loss.detach().cpu()),(tuple(idxb.tolist()),tuple(idxd.tolist()))


def save_state(path:Path, *, model,optimizer,gbase,gdose,seed:int,arm:str,
               step:int,release_fingerprint:str, data_hash:str, device:str):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    obj={'seed':seed,'arm':arm,'step':step,
         'release_fingerprint':release_fingerprint,'data_hash':data_hash,
         'device':str(device),'model':model.state_dict(),
         'optimizer':optimizer.state_dict(),
         'base_rng':gbase.get_state(), 'dose_rng':gdose.get_state(),
         'torch_cpu_rng':torch.get_rng_state()}
    if str(device).startswith('cuda'):
        obj['torch_cuda_rng']=torch.cuda.get_rng_state_all()
    if str(device).startswith('mps'):
        obj['torch_mps_rng']=torch.mps.get_rng_state()
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def load_state(path:Path, *, model,optimizer,gbase,gdose,seed:int,arm:str,
               release_fingerprint:str,data_hash:str,device:str):
    import torch
    v=torch.load(Path(path),map_location='cpu',weights_only=True)
    for key,expected in (('seed',seed),('arm',arm),
        ('release_fingerprint',release_fingerprint),('data_hash',data_hash),
        ('device',str(device))):
        if v.get(key)!=expected:raise PermissionError(f'Checkpoint {key} differs from release')
    step=v['step']
    if not (isinstance(step,int) and 0<=step<=12800):raise ValueError('Invalid checkpoint step')
    model.load_state_dict(v['model'],strict=True)
    optimizer.load_state_dict(v['optimizer'])
    gbase.set_state(v['base_rng']);gdose.set_state(v['dose_rng'])
    torch.set_rng_state(v['torch_cpu_rng'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state_all(v['torch_cuda_rng'])
    if str(device).startswith('mps'):
        torch.mps.set_rng_state(v['torch_mps_rng'])
    return step


def tensors_digest(model,optimizer,gbase,gdose):
    import torch
    buf=io.BytesIO()
    torch.save({'weights':model.state_dict(),'opt':optimizer.state_dict(),
                'base_rng':gbase.get_state(),'dose_rng':gdose.get_state()},buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()
