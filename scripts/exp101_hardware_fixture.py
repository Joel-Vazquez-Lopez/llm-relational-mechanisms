#!/usr/bin/env python3
"""EXP-101 disposable engineering smoke only. NEVER invokes study train_one/test.

Runs on caller's requested CPU/MPS device, using a non-study seed, short
artificial training arrays and familiar DEVELOPMENT-only sentences. The output
is a diagnostic report, NOT a human sign-off or study release authorization.
"""
from __future__ import annotations
import argparse, json, platform, tempfile, warnings
from pathlib import Path
import torch
from torch.nn import functional as F

import exp101_production_candidate as prod
import exp101_runtime as rt

FIXTURE_SEED = 101999
PARAM_ATOL, PARAM_RTOL = 1e-5, 1e-4
GRAD_EPS = 1e-12


def _compare_state(m1,m2):
    names=[];max_abs=0.; failures=[]
    for name,p in m1.state_dict().items():
        q=m2.state_dict()[name]
        delta=(p.detach().float().cpu()-q.detach().float().cpu()).abs()
        ma=float(delta.max())
        max_abs=max(max_abs,ma)
        if not torch.allclose(p.detach().cpu(),q.detach().cpu(),rtol=PARAM_RTOL,atol=PARAM_ATOL):
            failures.append(name)
        names.append(name)
    return len(names),max_abs,failures


def _optimizer_equal(a,b):
    asd=a.state_dict();bsd=b.state_dict()
    if asd['param_groups']!=bsd['param_groups'] or asd['state'].keys()!=bsd['state'].keys():return False
    for key in asd['state']:
      for name,v in asd['state'][key].items():
        w=bsd['state'][key][name]
        if isinstance(v,torch.Tensor):
          if not torch.allclose(v.detach().cpu(),w.detach().cpu(),rtol=PARAM_RTOL,atol=PARAM_ATOL):return False
        elif v!=w:return False
    return True



def _optimizer_state_exact(a,b):
    asd=a.state_dict();bsd=b.state_dict()
    if asd['param_groups']!=bsd['param_groups'] or asd['state'].keys()!=bsd['state'].keys():return False
    for key in asd['state']:
        if asd['state'][key].keys()!=bsd['state'][key].keys():return False
        for name,v in asd['state'][key].items():
            w=bsd['state'][key][name]
            if isinstance(v,torch.Tensor):
                if not isinstance(w,torch.Tensor) or not torch.equal(v.detach().cpu(),w.detach().cpu()):return False
            elif v!=w:return False
    return True

def _run_fixture_impl(device: str):
    if device not in ('cpu','mps'):raise ValueError('Only local CPU and Mac MPS engineering fixtures permitted')
    if device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable on this machine; no MPS validation claimed')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(606060)
    names=prod.NAMES
    v=len(prod.VOCAB_WORDS)
    x=torch.randint(0,v,(12,16),dtype=torch.long)
    r=torch.full((12,),15,dtype=torch.long)
    y=torch.tensor([prod.VOCAB[names[i%len(names)]] for i in range(12)])
    choices=torch.tensor([[prod.VOCAB[n] for n in names[:4]]]*12)
    base=(x,r,y,choices)
    dose=(x.flip(0),r,y.flip(0),choices)
    def init():return rt.init_state(FIXTURE_SEED,prod.make_model,device=device)
    # Paired initial weights, base/dose indices AND prescribed offset identical.
    a,ao,ab,ad=init();b,bo,bb,bd=init()
    n,initial_difference,initial_mismatch=_compare_state(a,b)
    if initial_mismatch:raise AssertionError('Paired initial model states differ')
    a_loss,a_idxs=rt.one_optimizer_step(a,ao,base,dose,ab,ad,base_batch=3,
                dose_batch=2,pad_id=prod.VOCAB['<pad>'],device=device,position_offset=0)
    b_loss,b_idxs=rt.one_optimizer_step(b,bo,base,dose,bb,bd,base_batch=3,
                dose_batch=2,pad_id=prod.VOCAB['<pad>'],device=device,position_offset=0)
    if a_idxs!=b_idxs:raise AssertionError('Paired batches differ on same seed')
    if abs(a_loss-b_loss)>1e-5:raise AssertionError('Paired first-step losses differ')
    # Independent run parity is a separate check from same-origin resume.
    n_first, independent_first_delta, independent_first_bad = _compare_state(a,b)
    independent_optimizer_matches = _optimizer_equal(ao,bo)
    if independent_first_bad or not independent_optimizer_matches:
        raise AssertionError('Independent first step outside frozen tolerances')
    # The uninterrupted trajectory and its resumed continuation MUST originate
    # from the *same* exact checkpoint. Saving B here confounds two independent
    # MPS executions (as discovered in the RC3 hardware audit).

    with tempfile.TemporaryDirectory(prefix='exp101_engineering_') as folder:
        ckpt=Path(folder)/'state.pt'
        rt.save_state(ckpt,model=a,optimizer=ao,gbase=ab,gdose=ad,
            seed=FIXTURE_SEED,arm='fixture',step=1,
            release_fingerprint='NONSTUDY_FIXTURE_ONLY',data_hash='ARTIFICIAL',device=device)
        c,co,cb,cd=init()
        step=rt.load_state(ckpt,model=c,optimizer=co,gbase=cb,gdose=cd,
            seed=FIXTURE_SEED,arm='fixture',
            release_fingerprint='NONSTUDY_FIXTURE_ONLY',data_hash='ARTIFICIAL',device=device)
        if step!=1:raise AssertionError('Failed fixture resume step')
        _, roundtrip_delta, roundtrip_bad = _compare_state(a,c)
        if roundtrip_bad or roundtrip_delta!=0.0:
            raise AssertionError('Same-origin save/reload changed model weights')
        if not _optimizer_state_exact(ao,co):
            raise AssertionError('Same-origin save/reload changed optimizer')
        if not torch.equal(ab.get_state(),cb.get_state()) or not torch.equal(ad.get_state(),cd.get_state()):
            raise AssertionError('Same-origin save/reload changed sampler RNG')
        for ix in (1,2):
            off=prod.POSITION_OFFSETS[ix % len(prod.POSITION_OFFSETS)]
            al,ai=rt.one_optimizer_step(a,ao,base,dose,ab,ad,base_batch=3,
                  dose_batch=2,pad_id=prod.VOCAB['<pad>'],device=device,position_offset=off)
            cl,ci=rt.one_optimizer_step(c,co,base,dose,cb,cd,base_batch=3,
                  dose_batch=2,pad_id=prod.VOCAB['<pad>'],device=device,position_offset=off)
            if ai!=ci:raise AssertionError('Interrupted batch sampler diverged')
            if abs(al-cl)>1e-5:raise AssertionError('Interrupted loss diverged')
        n,delta,mismatched=_compare_state(a,c)
        optim_matches=_optimizer_equal(ao,co)
        if mismatched or not optim_matches:raise AssertionError('Resume mismatch after 3 fixture steps')
        # Full frozen-architecture checkpoint -> actual dev scoring on TWO
        # familiar development packets (no LI, no protected examples).
        _,dev,_=prod.all_matchings()
        packets=[]
        for pkt in prod.examples(dev):
            if len(prod.token_sequence(pkt,'agent'))==34:
                packets.append(pkt)
            if len(packets)==2:break
        if len(packets)!=2:raise AssertionError('No length-34 familiar dev packets')
        score=prod.score_development_model(c,packets,device=device,batch_packets=2)
        if sum(vv['events'] for vv in score.values())!=2:raise AssertionError('Dev score integration failure')
        # Question-region gradient at previously unused evaluation indices.
        # Distinct fresh fixture weights avoid study-trained representations.
        gradmodel=prod.make_model().to(device)
        ids,ro,ys,_=prod.encode([(p,role) for p in packets for role in ('agent','patient')])
        gradmodel.zero_grad(set_to_none=True)
        F.cross_entropy(gradmodel(ids.to(device),ro.to(device),position_offset=8),ys.to(device)).backward()
        grad=gradmodel.pos.weight.grad.detach().cpu()
        grad_norms={str(i):float(grad[i].norm().item()) for i in range(34,39)}
        if not all(v>GRAD_EPS for v in grad_norms.values()):
            raise AssertionError(f'Gradient coverage failed for index 34-38: {grad_norms}')
        return {'study':'EXP-101','status':'ENGINEERING_FIXTURE_PASS_NOT_APPROVAL',
                'device':device,'platform':platform.platform(),
                'python':platform.python_version(),'torch':str(torch.__version__),
                'mps_available':torch.backends.mps.is_available(),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                'fixture_seed':FIXTURE_SEED,'study_seed_used':False,
                'initial_parameter_difference':initial_difference,
                'independent_first_step_max_abs_difference':independent_first_delta,
                'independent_first_step_optimizer_within_tolerance':independent_optimizer_matches,
                'same_origin_checkpoint_roundtrip_max_abs_difference':roundtrip_delta,
                'same_origin_checkpoint_optimizer_exact':True,
                'resume_origin':'same_model_a_checkpoint_after_step_1',
                'paired_first_step_batch_indices_equal':a_idxs==b_idxs,
                'resume_parameter_max_abs_difference':delta,
                'resume_param_abs_tol':PARAM_ATOL,'resume_param_rel_tol':PARAM_RTOL,
                'resume_optimizer_within_tolerance':optim_matches,
                'n_parameters_checked':n,'fixture_steps':3,
                'offsets_at_fixture_steps':[0,1,2],
                'dev_only_smoke_events':2,'dev_cell_keys':sorted(score),
                'position_gradient_norms_34_through_38':grad_norms,
                'gradient_minimum':GRAD_EPS,'protected_accessed':False,
                'experiment_training_started':False}

def run_fixture(device: str):
    """RC4: same-origin resume parity with backend limitations visible.

    MPS warn_only allows kernels lacking deterministic implementations to run;
    any reproducibility mismatch still fails the frozen numerical comparisons.
    """
    policy=rt.configure_determinism(device)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter('always')
        report=_run_fixture_impl(device)
    backend_warnings=[str(item.message) for item in raised
                      if 'deterministic' in str(item.message).lower() or
                         'nondeterministic' in str(item.message).lower()]
    report['determinism_policy']=policy
    report['known_backend_nondeterminism_warnings']=len(backend_warnings)
    report['nondeterminism_warning_examples']=sorted(set(backend_warnings))[:5]
    report['exact_mps_reproducibility_claimed']=False
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','mps'),required=True)
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    result=run_fixture(args.device)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True,indent=2))
