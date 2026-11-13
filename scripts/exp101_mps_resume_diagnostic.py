#!/usr/bin/env python3
"""EXP-101 RC4 engineering-only MPS/CPU trajectory diagnostic.

No study training, no protected evaluation, no study seed. Uses the locked RC3
modules WITHOUT changing them or their release manifest. Captures the separate
sources of discrepancy that RC3 conflated: independently executed first steps,
checkpoint serialization, and two continuations from an identical checkpoint.

Example:
  python scripts/exp101_mps_resume_diagnostic.py --device mps \
    --output reports/exp101/mps_resume_diagnostic_local.json

A completed diagnostic is NEVER approval to train, even if all comparisons pass.
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import warnings
from pathlib import Path

import torch

import exp101_production_candidate as prod
import exp101_runtime as rt
from exp101_hardware_fixture import FIXTURE_SEED, PARAM_ATOL, PARAM_RTOL


# Follow the predeclared RC3 tolerances; never choose them after results.

def _stats(model_a, model_b) -> dict:
    names_a = model_a.state_dict()
    names_b = model_b.state_dict()
    if names_a.keys() != names_b.keys():
        raise AssertionError("Model state keys differ")
    max_abs, bad, unequal = 0.0, [], []
    per_tensor = []
    for key in names_a:
        x = names_a[key].detach().to('cpu')
        y = names_b[key].detach().to('cpu')
        if x.shape != y.shape:
            raise AssertionError(f"Model state shape mismatch: {key}")
        diff = (x.float() - y.float()).abs()
        ma = float(diff.max()) if diff.numel() else 0.0
        max_abs = max(max_abs, ma)
        exact = bool(torch.equal(x, y))
        close = bool(torch.allclose(x, y, atol=PARAM_ATOL, rtol=PARAM_RTOL))
        if not exact:
            unequal.append(key)
        if not close:
            bad.append(key)
        if not exact or not close:
            per_tensor.append({"name": key, "max_abs": ma,
                               "exact": exact, "within_tolerance": close})
    return {"all_exact": not unequal, "all_within_tolerance": not bad,
            "max_abs": max_abs, "n_tensors": len(names_a),
            "n_not_exact": len(unequal), "n_outside_tolerance": len(bad),
            "worst_or_nonidentical_tensors": sorted(per_tensor, key=lambda z: -z['max_abs'])[:12]}


def _optimizer_stats(opt_a, opt_b) -> dict:
    sa = opt_a.state_dict()
    sb = opt_b.state_dict()
    config_equal = sa['param_groups'] == sb['param_groups']
    keys_equal = sa['state'].keys() == sb['state'].keys()
    failures, nonexact = [], []
    max_abs = 0.0
    if keys_equal:
        for key in sa['state']:
            aa, bb = sa['state'][key], sb['state'][key]
            if aa.keys() != bb.keys():
                failures.append(f"state[{key}] key mismatch")
                continue
            for field in aa:
                x, y = aa[field], bb[field]
                if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                    x, y = x.detach().to('cpu'), y.detach().to('cpu')
                    if x.shape != y.shape:
                        failures.append(f"state[{key}].{field} shape mismatch")
                        continue
                    diff = (x.float()-y.float()).abs()
                    ma = float(diff.max()) if diff.numel() else 0.0
                    max_abs = max(max_abs, ma)
                    if not torch.equal(x,y):
                        nonexact.append(f"state[{key}].{field}")
                    if not torch.allclose(x,y,atol=PARAM_ATOL,rtol=PARAM_RTOL):
                        failures.append(f"state[{key}].{field}")
                elif x != y:
                    failures.append(f"state[{key}].{field}")
                    nonexact.append(f"state[{key}].{field}")
    return {"config_equal": config_equal, "state_keys_equal": keys_equal,
            "all_exact": config_equal and keys_equal and not nonexact,
            "all_within_tolerance": config_equal and keys_equal and not failures,
            "max_abs": max_abs, "n_not_exact": len(nonexact),
            "n_outside_tolerance": len(failures), "failure_examples": failures[:10]}


def _compare(model_a, opt_a, gb_a, gd_a, model_b, opt_b, gb_b, gd_b) -> dict:
    return {
        "model": _stats(model_a, model_b),
        "optimizer": _optimizer_stats(opt_a, opt_b),
        "base_rng_equal": bool(torch.equal(gb_a.get_state(),gb_b.get_state())),
        "dose_rng_equal": bool(torch.equal(gd_a.get_state(),gd_b.get_state()))
    }


def run(device: str) -> dict:
    if device not in ('cpu', 'mps'):
        raise ValueError('Only CPU/MPS fixture backends are supported')
    policy = rt.configure_determinism(device)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(606060)
    names = prod.NAMES
    v = len(prod.VOCAB_WORDS)
    x = torch.randint(0, v, (12, 16), dtype=torch.long)
    ro = torch.full((12,),15,dtype=torch.long)
    y = torch.tensor([prod.VOCAB[names[i % len(names)]] for i in range(12)])
    choices = torch.tensor([[prod.VOCAB[n] for n in names[:4]]]*12)
    base = (x,ro,y,choices)
    dose = (x.flip(0),ro,y.flip(0),choices)
    def init(): return rt.init_state(FIXTURE_SEED, prod.make_model, device=device)
    def step(state, offset):
        m,o,gbase,gdose = state
        return rt.one_optimizer_step(m,o,base,dose,gbase,gdose,
            base_batch=3,dose_batch=2,pad_id=prod.VOCAB['<pad>'],
            device=device,position_offset=offset)
    def compare(sa,sb):return _compare(*sa,*sb)
    report = {
        'study':'EXP-101', 'diagnostic':'RC4_MPS_RESUME_ROOT_CAUSE',
        'device':device,'platform':platform.platform(),
        'torch':str(torch.__version__), 'python':platform.python_version(),
        'fixture_seed':FIXTURE_SEED,'study_seeds_used':False,
        'experiment_training_started':False,'protected_evaluation_accessed':False,
        'training_authorized':False,'release_approved':False,
        'comparison_atol':PARAM_ATOL,'comparison_rtol':PARAM_RTOL,
        'backend_policy':policy,
        'fixture_only_steps':3,'position_offsets':[0,1,2]
    }
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        a=init();b=init()
        report['initial_state_comparison']=compare(a,b)
        al,ai=step(a,0)
        bl,bi=step(b,0)
        report['independent_first_step']={
          'loss_a':al,'loss_b':bl,'abs_loss_delta':abs(al-bl),
          'sampled_indices_equal':ai==bi,'state':compare(a,b)
        }
        with tempfile.TemporaryDirectory(prefix='exp101_resume_diagnostic_') as d:
            ckpt=Path(d)/'step_1.pt'
            rt.save_state(ckpt,model=b[0],optimizer=b[1],gbase=b[2],gdose=b[3],
                seed=FIXTURE_SEED,arm='fixture',step=1,
                release_fingerprint='NONSTUDY_DIAGNOSTIC',data_hash='SYNTHETIC',device=device)
            c=init()
            ckstep=rt.load_state(ckpt,model=c[0],optimizer=c[1],gbase=c[2],gdose=c[3],
                seed=FIXTURE_SEED,arm='fixture',
                release_fingerprint='NONSTUDY_DIAGNOSTIC',data_hash='SYNTHETIC',device=device)
            report['save_load_at_step_1']={'step':ckstep,'comparison':compare(b,c)}
            # Independent continuation from a saved initial state (rather than
            # comparing only two already potentially divergent MPS first steps).
            for idx in (1,2):
                la,ixa=step(a,idx)
                lc,ixc=step(c,idx)
                report[f'uninterrupted_a_vs_resumed_c_after_step_{idx+1}']={
                    'loss_a':la,'loss_c':lc,'abs_loss_delta':abs(la-lc),
                    'sampled_indices_equal':ixa==ixc,'state':compare(a,c)}
            # Replay continuation from the same checkpoint.  Even this is not
            # guaranteed bitwise on MPS, but isolates checkpoint serialization.
            d2=init()
            rt.load_state(ckpt,model=d2[0],optimizer=d2[1],gbase=d2[2],gdose=d2[3],
                seed=FIXTURE_SEED,arm='fixture',
                release_fingerprint='NONSTUDY_DIAGNOSTIC',data_hash='SYNTHETIC',device=device)
            report['reloaded_at_step_1_again']={'comparison':compare(b,d2)}
            for idx in (1,2):
                ld,ixd=step(d2,idx)
                report[f'reloaded_d_second_branch_after_step_{idx+1}']={
                    'loss_d':ld,'samples_n':len(ixd[0])+len(ixd[1])}
            report['two_independent_resumes_from_exact_same_checkpoint'] = compare(c,d2)
        w=[str(item.message) for item in captured]
    report['nondeterminism_warning_count']=sum('determinist' in w1.lower() for w1 in w)
    report['warnings_unique']=sorted(set(w))[:6]
    pre=report['independent_first_step']['state']
    loaded=report['save_load_at_step_1']['comparison']
    final=report['uninterrupted_a_vs_resumed_c_after_step_3']['state']
    two=report['two_independent_resumes_from_exact_same_checkpoint']
    report['interpretation_flags']={
        'independent_first_steps_within_tolerance':pre['model']['all_within_tolerance'] and pre['optimizer']['all_within_tolerance'],
        'saved_and_reloaded_step1_exact':loaded['model']['all_exact'] and loaded['optimizer']['all_exact'] and loaded['base_rng_equal'] and loaded['dose_rng_equal'],
        'a_vs_c_step3_within_tolerance':final['model']['all_within_tolerance'] and final['optimizer']['all_within_tolerance'],
        'same_checkpoint_two_resumes_within_tolerance':two['model']['all_within_tolerance'] and two['optimizer']['all_within_tolerance']
    }
    report['status']='DIAGNOSTIC_COMPLETE_NOT_RELEASE_APPROVAL'
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','mps'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    try:
        report=run(args.device)
    except Exception as e:
        report={'status':'DIAGNOSTIC_ERROR','type':type(e).__name__,
                'message':str(e),'training_authorized':False,
                'protected_evaluation_accessed':False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],
          'interpretation_flags':report['interpretation_flags'],
          'model_max_abs':{
            'after_independent_first_step':report['independent_first_step']['state']['model']['max_abs'],
            'checkpoint_round_trip':report['save_load_at_step_1']['comparison']['model']['max_abs'],
            'uninterrupted_vs_resumed':report['uninterrupted_a_vs_resumed_c_after_step_3']['state']['model']['max_abs'],
            'two_resumes_same_checkpoint':report['two_independent_resumes_from_exact_same_checkpoint']['model']['max_abs']},
          'warnings':report['nondeterminism_warning_count'],
          'full_json':str(args.output)},indent=2))


if __name__=='__main__':main()