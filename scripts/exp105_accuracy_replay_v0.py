#!/usr/bin/env python3
"""EXP-105 exploratory accuracy-level repeatability check (not confirmation).

Run the same existing EXP-105 long-pilot training code in SEPARATE
processes, with the same seed and scheduled training examples.
Each repeat has a unique output path. This script does not edit any
existing source, protocol, checkpoint, or previous result.

Requires the completed EXP-105 v0 and v1 `prepare` phases.
It scores only 216 familiar-development AA/PA/AP sources; never L+I,
reserved identifiers/examples or protected test data.

Example:
  python3 scripts/exp105_accuracy_replay_v0.py --phase smoke --device mps
  python3 -u scripts/exp105_accuracy_replay_v0.py --phase run --device mps --repeat 1 --steps 3200
  python3 -u scripts/exp105_accuracy_replay_v0.py --phase run --device mps --repeat 2 --steps 3200
  python3 scripts/exp105_accuracy_replay_v0.py --phase compare --device mps --steps 3200
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

import torch

import exp105_development_pilot_v0 as pilot
import exp105_development_long_pilot_v1 as long

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/exp105/accuracy_replay_v0"
SEEDS = (10801, 10803)  # development pilot seeds only
MILESTONES = tuple(long.MILESTONES)


def weights_hash(model):
    h = sha256()
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def name(seed, device, rep, steps):
    return f"pilot_seed{seed}_{device}_repeat{rep:02d}_step{steps}.json"


def run(args):
    original, parent, common, paired, bpool, dpool, dev, vocab = long.setup()
    proto = long.make_protocol(original, parent, dev, vocab)
    manifest = long.read_frozen(proto)
    device = pilot.get_device(args.device)
    out = OUT / name(args.seed, args.device, args.repeat, args.steps)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing repeat: {out}")

    dev_enc = pilot.encode_sources(dev, vocab)
    (f, of), (t, ot) = pilot.make_models(args.seed, device, len(vocab))
    initial_f = weights_hash(f)
    initial_t = weights_hash(t)
    assert initial_f == initial_t, "Unmatched arm initialization"
    h = sha256()
    history = []
    first_step = None
    for step, base, dose, _, _ in long.schedule(args.seed, common, paired, bpool, dpool):
        fa = [common[i] for i in base] + [paired[i][0] for i in dose]
        ta = [common[i] for i in base] + [paired[i][1] for i in dose]
        assert len(fa) == len(ta) == 16
        assert [s.gold for s in fa] == [s.gold for s in ta]
        ef = pilot.encode_sources(fa, vocab, check=False)
        et = pilot.encode_sources(ta, vocab, check=False)
        assert torch.equal(ef[2],et[2]) and torch.equal(ef[3],et[3])
        h.update((json.dumps({'step':step,'base':base,'dose':dose},separators=(',', ':'))+'\n').encode())
        lf = pilot.one_update(f, of, ef, device)
        lt = pilot.one_update(t, ot, et, device)
        if step == 1:
            first_step = {'familiar': weights_hash(f), 'target': weights_hash(t),
                          'loss_familiar': lf, 'loss_target': lt}
        if step in MILESTONES:
            fm = pilot.score(f,dev_enc,dev,device,len(vocab))
            tm = pilot.score(t,dev_enc,dev,device,len(vocab))
            history.append({'step':step,'familiar':fm,'target':tm,'train_loss_familiar':lf,'train_loss_target':lt})
            print(f"REPEAT {args.repeat} STEP {step}: "
                  f"F { {k:round(v,3) for k,v in fm['per_voice_C_query'].items()} } "
                  f"T { {k:round(v,3) for k,v in tm['per_voice_C_query'].items()} }",flush=True)
        if step == args.steps:
            break
    assert history[-1]['step'] == args.steps
    if args.steps == 6400:
        assert h.hexdigest() == manifest['seeds'][str(args.seed)]['schedule_sha256']
    else:
        old_path = ROOT / 'reports/exp105/development_pilot_v0/pilot_seed10801_step3200.json'
        if args.seed == 10801 and old_path.is_file():
            old = json.loads(old_path.read_text())
            assert h.hexdigest() == old['candidate_schedule_sha256_prefix']
    record = {
        'study':'EXP105_DEVELOPMENT_ACCURACY_REPLAY_V0',
        'not_confirmatory':True,'development_only':True,'no_LI_scoring':True,'protected_evaluated':False,
        'seed':args.seed,'device':args.device,'repeat':args.repeat,'steps':args.steps,
        'parent_protocol_digest':long.digest(proto),'schedule_sha256':h.hexdigest(),
        'initial_model_hash':initial_f, 'first_step':first_step,
        'final_model_hashes':{'familiar':weights_hash(f),'target':weights_hash(t)},
        'library_versions':{'torch':torch.__version__, 'python':platform.python_version(), 'platform':platform.platform()},
        'history':history,
    }
    # Create a new isolated report; never overwrite previous artifacts.
    OUT.mkdir(parents=True,exist_ok=True)
    with out.open('x') as fh:
        json.dump(record,fh,indent=2,sort_keys=True)
        fh.write('\n')
    print('SAVED',out,flush=True)
    print('Only familiar-development AA/PA/AP scored; original experiments unchanged.',flush=True)


def compare(args):
    a,b = [OUT/name(args.seed,args.device,r,args.steps) for r in (1,2)]
    if not a.is_file() or not b.is_file():
        raise FileNotFoundError(f'Need BOTH separate-process replay files:\n{a}\n{b}')
    x,y = (json.loads(p.read_text()) for p in (a,b))
    for key in ('seed','steps','device','parent_protocol_digest','schedule_sha256','initial_model_hash'):
        assert x[key] == y[key], f'Different replay inputs: {key}'
    print(f"MATCHED INPUTS: seed={args.seed}, steps={args.steps}, device={args.device}")
    print('Same initial weights:',x['initial_model_hash'][:20]); print('Same schedule:',x['schedule_sha256'][:20])
    print('Different first-step weights:', x['first_step']['familiar']!=y['first_step']['familiar']
          or x['first_step']['target']!=y['first_step']['target'])
    print('Different final weights:',x['final_model_hashes']!=y['final_model_hashes'])
    endx,endy=x['history'][-1],y['history'][-1]
    for arm in ('familiar','target'):
        print('\n'+arm.upper())
        for voice in ('AA','PA','AP'):
            z0=endx[arm]['per_voice_C_query'][voice]
            z1=endy[arm]['per_voice_C_query'][voice]
            print(f"  {voice}: repeat1={z0:.4f} repeat2={z1:.4f} absolute_gap={abs(z1-z0):.4f}")
        for metric in ('C_query','accuracy4','wrong_event_rate','wrong_role_rate','same_answer_rate'):
            v0=endx[arm][metric];v1=endy[arm][metric]
            print(f"  {metric}: repeat1={v0:.4f} repeat2={v1:.4f} delta={v1-v0:+.4f}")
    for label, r in [('repeat1',endx),('repeat2',endy)]:
        gap=r['target']['per_voice_C_query']['AP']-r['familiar']['per_voice_C_query']['AP']
        print(f'AP exposure effect {label}: {gap:+.4f}')
    print('\nEXPLORATORY: two runs measure numerical repeatability for ONE initialization, not seed-level inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('smoke','run','compare'))
    p.add_argument('--device',default='mps',choices=('mps','cpu','cuda'))
    p.add_argument('--seed',type=int,choices=SEEDS,default=10801)
    p.add_argument('--repeat',type=int,choices=(1,2),default=1)
    p.add_argument('--steps',type=int,choices=(3200,6400),default=3200)
    args=p.parse_args()
    if args.phase=='compare':
        compare(args);return
    if args.phase=='smoke':
        _,parent,common,paired,bpool,dpool,dev,vocab=long.setup()
        device=pilot.get_device(args.device)
        (f,of),(t,ot)=pilot.make_models(args.seed,device,len(vocab))
        updates=long.schedule(args.seed,common,paired,bpool,dpool)
        for step,base,dose,_,_ in updates:
            ef=pilot.encode_sources([common[i] for i in base]+[paired[i][0] for i in dose],vocab,check=False)
            et=pilot.encode_sources([common[i] for i in base]+[paired[i][1] for i in dose],vocab,check=False)
            assert torch.equal(ef[2],et[2]) and torch.equal(ef[3],et[3])
            a=pilot.one_update(f,of,ef,device)
            b=pilot.one_update(t,ot,et,device)
            print(f'SMOKE step={step} losses={a:.6f},{b:.6f}',flush=True)
            if step==2:break
        dev_enc=pilot.encode_sources(dev,vocab)
        assert 0<=pilot.score(f,dev_enc,dev,device,len(vocab))['C_query']<=1
        assert 0<=pilot.score(t,dev_enc,dev,device,len(vocab))['C_query']<=1
        print('SMOKE PASS: no files created, no LI/protected evaluation.')
        return
    run(args)

if __name__=='__main__':main()
