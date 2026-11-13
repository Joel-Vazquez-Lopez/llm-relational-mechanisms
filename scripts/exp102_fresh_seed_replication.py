#!/usr/bin/env python3
"""EXP-102 fixed exploratory replication on previously untested initialization seeds.

Uses the same already-explored synthetic train/ID/OOD distribution; this is NOT
an independent held-out confirmatory evaluation. Never constructs protected data.
Prepare once, then run any subset of the eight prespecified seeds. No overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from exp102_learning_pilot import Model
from exp102_pair_structure_control import data
from exp102_role_trajectory import diagnose, show

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve()
OUT = ROOT / 'reports/exp102/fresh_seed_replication'
CKPT = ROOT / 'results/exp102/fresh_seed_replication'
SEEDS = (10301, 10303, 10309, 10313, 10321, 10331, 10333, 10337)
METHODS = ('quartet', 'balanced_unpaired', 'role_pair', 'event_pair')
STEPS = 1600
MILESTONES = (400, 800, 1200, 1600)
DEPENDENCIES = ('exp102_preflight.py', 'exp102_learning_pilot.py',
                'exp102_role_trajectory.py', 'exp102_pair_structure_control.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    return {p: sha(HERE.parent / p) for p in DEPENDENCIES}


def git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def indices_for(rng, method, n):
    if method == 'quartet':
        # 16 drawn sources; occasional source collision is possible, as in the pilot.
        src = torch.randint(n, (16,), generator=rng)
        queries = torch.arange(4).repeat(16, 1)
    elif method == 'balanced_unpaired':
        src = torch.randint(n, (16, 4), generator=rng)
        queries = torch.arange(4).repeat(16, 1)
    elif method in ('role_pair', 'event_pair'):
        src = torch.randperm(n, generator=rng)[:32]
        if method == 'role_pair':
            queries = torch.tensor([[0, 1]] * 16 + [[2, 3]] * 16)
        else:
            queries = torch.tensor([[0, 2]] * 16 + [[1, 3]] * 16)
        src = src[:, None]
    else:
        raise ValueError(method)
    if method == 'quartet':
        src = src[:, None]
    ix = (4 * src + queries).reshape(-1)
    assert len(ix) == 64
    assert torch.bincount(ix % 4, minlength=4).tolist() == [16] * 4
    if method in ('role_pair', 'event_pair'):
        assert len(set((ix // 4).tolist())) == 32
        assert len(set(ix.tolist())) == 64
    return ix


def protocol(dataset_digest):
    return {
        'study': 'EXP-102', 'phase': 'FRESH_SEED_EXPLORATORY_REPLICATION',
        'not_confirmatory': True,
        'reason': 'Evaluation distribution and task family were explored previously',
        'primary_contrast': 'OOD C_query at step 1600: quartet minus balanced_unpaired',
        'secondary_contrasts': ['quartet minus role_pair', 'quartet minus event_pair'],
        'metric': 'fraction of 256 development OOD sentences with all 4 answers correct',
        'fixed_seeds': list(SEEDS), 'methods': list(METHODS),
        'fixed_steps': STEPS, 'milestones': list(MILESTONES),
        'model': 'EXP102 Model 3 layers width96 heads4',
        'optimizer': 'AdamW(lr=.002,weight_decay=.01)',
        'train_sentences': 2048, 'development_id_sentences': 256,
        'development_ood_sentences': 256, 'labels_per_update': 64,
        'query_counts': [16, 16, 16, 16],
        'source_count_is_not_matched_between_methods': True,
        'no_protected_evaluation': True,
        'data_digest': dataset_digest,
        'source_files_sha256': source_manifest(),
        'runner_sha256': sha(HERE), 'git_sha': git_sha(),
    }


def read_frozen(current):
    path = OUT / 'protocol.json'
    if not path.exists():
        raise FileNotFoundError('Run --phase prepare first')
    saved = json.loads(path.read_text())
    if saved != current:
        raise RuntimeError('Protocol, runner, sources, data or Git SHA changed: refusing run')
    return saved


def run_one(seed, method, train, dev_id, dev_ood, fingerprint, device):
    path = OUT / f'seed{seed}_{method}.json'
    checkpoint = CKPT / f'seed{seed}_{method}.pt'
    if path.exists() and checkpoint.exists():
        print(f'ALREADY PRESENT: {seed}/{method}; skipping, never overwriting', flush=True)
        return
    if path.exists() or checkpoint.exists():
        raise FileExistsError(f'Incomplete artifact pair for {seed}/{method}; inspect before resuming')
    torch.manual_seed(seed)
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    sampler = torch.Generator(device='cpu').manual_seed(seed + 100)
    x, readout, labels, _ = train
    results = []
    print(f'\nRUN seed={seed} method={method}', flush=True)
    for step in range(1, STEPS + 1):
        ix = indices_for(sampler, method, len(x) // 4)
        model.train()
        logits = model(x[ix].to(device), readout[ix].to(device))
        loss = F.cross_entropy(logits, labels[ix].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in MILESTONES:
            ids = diagnose(model, dev_id, device)
            ood = diagnose(model, dev_ood, device)
            print(f'step={step} loss={loss.item():.4f}', flush=True)
            show(step, 'ID', ids)
            show(step, 'OOD', ood)
            results.append({'step': step, 'id': ids, 'ood': ood})
    obj = {'study': 'EXP-102', 'phase': 'FRESH_SEED_EXPLORATORY_REPLICATION',
           'seed': seed, 'method': method, 'steps': STEPS,
           'data_digest': fingerprint, 'runner_sha256': sha(HERE),
           'source_files_sha256': source_manifest(),
           'device': device, 'records': results,
           'primary_ood_C_query': results[-1]['ood']['C_query']}
    CKPT.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix('.pt.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'sampler_state': sampler.get_state(), 'metadata': obj}, tmp)
    os.replace(tmp, checkpoint)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n')
    print('SAVED', path, flush=True)
    print('SAVED', checkpoint, flush=True)


def summarize():
    print('\nCOMPLETE SEED PAIRS:')
    complete = []
    for seed in SEEDS:
        paths = [OUT / f'seed{seed}_{m}.json' for m in METHODS]
        if all(p.exists() for p in paths):
            vals = [json.loads(p.read_text())['primary_ood_C_query'] for p in paths]
            complete.append((seed, vals))
            print(f'{seed}: ' + ' '.join(f'{m}={v:.3f}' for m, v in zip(METHODS, vals)))
    if len(complete) == len(SEEDS):
        diffs = [vs[0] - vs[1] for _, vs in complete]
        print('PRIMARY: mean paired quartet-minus-balanced_unpaired C_query = '
              f'{sum(diffs)/len(diffs):+.4f}; positive seeds = '
              f'{sum(v > 0 for v in diffs)}/{len(diffs)}')
    else:
        print(f'{len(complete)}/{len(SEEDS)} seed quartets complete; do not make a final inference yet')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('prepare', 'run', 'summary'), required=True)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    args = ap.parse_args()
    if args.device == 'mps' and args.phase == 'run' and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    if args.seeds is not None and (not args.seeds or len(set(args.seeds)) != len(args.seeds)
                                   or any(x not in SEEDS for x in args.seeds)):
        ap.error('Use distinct seeds from the fixed protocol')
    train, dev_id, dev_ood, fingerprint = data()
    expected = protocol(fingerprint)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'protocol.json'
    if args.phase == 'prepare':
        if manifest_path.exists():
            read_frozen(expected)
            print('Protocol already frozen and unchanged:', manifest_path)
        else:
            manifest_path.write_text(json.dumps(expected, sort_keys=True, indent=2) + '\n')
            print('FROZEN EXPLORATORY REPLICATION PROTOCOL:', manifest_path)
        print('Primary:', expected['primary_contrast'])
        print('Methods:', METHODS, 'Seeds:', SEEDS)
        print('No training; no protected evaluation')
        return
    read_frozen(expected)
    if args.phase == 'summary':
        summarize()
        return
    for seed in args.seeds if args.seeds is not None else SEEDS:
        for method in METHODS:
            run_one(seed, method, train, dev_id, dev_ood, fingerprint, args.device)
    summarize()
    print('EXPLORATORY replication only. No protected evaluation.')


if __name__ == '__main__':
    main()
