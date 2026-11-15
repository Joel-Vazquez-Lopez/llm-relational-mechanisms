#!/usr/bin/env python3
"""EXP-105 development-only deterministic replay diagnostic.

Read-only with respect to original models/reports. Does not evaluate L+I,
reserved, or protected examples. Performs disposable short optimizer updates.

From the repository root:
    python3 scripts/exp105_replay_repro_check.py --device mps --steps 90
    python3 scripts/exp105_replay_repro_check.py --device cpu --steps 20

Do not confuse this numerical diagnostic with a confirmatory training run.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

import torch

import exp105_development_pilot_v0 as pilot
import exp105_development_long_pilot_v1 as long
import exp105_training_schedule_preflight_v3 as v3

SEED = 10801  # already used pilot seed; NEVER use 107xx confirmatory seeds
ROOT = Path(__file__).resolve().parents[1]


def hash_weights(model):
    h = sha256()
    for name, param in model.state_dict().items():
        val = param.detach().to('cpu').contiguous()
        h.update(name.encode())
        h.update(str(tuple(val.shape)).encode())
        h.update(val.numpy().tobytes())
    return h.hexdigest()


def get_first_updates(common, paired, bpool, dpool, n):
    # Check both generators yield identical *ordered* source indices, not only
    # matching metadata/protocol. Note that the extended runner's second half
    # is intentionally different and is not compared here.
    a = v3.iter_schedule(SEED, common, paired, bpool, dpool)
    b = long.schedule(SEED, common, paired, bpool, dpool)
    result = []
    for k in range(1, n + 1):
        s0 = next(a)
        s1 = next(b)
        assert s0[0] == s1[0] == k
        assert s0[1:] == s1[1:], f'INDEX MISMATCH AT UPDATE {k}'
        result.append((k, s0[1], s0[2]))
    return result


def replay_one(rep, device, steps, common, paired, dev, vocab, updates):
    # Mirror the original runner: encode dev BEFORE model creation.
    dev_enc = pilot.encode_sources(dev, vocab)
    (f, optf), (t, optt) = pilot.make_models(SEED, device, len(vocab))
    init_f, init_t = hash_weights(f), hash_weights(t)
    assert init_f == init_t, 'ARM WEIGHTS DO NOT MATCH AT INITIALIZATION'
    trajectory = []
    check_steps = {1, 2, 5, 10, 20, 45, 90, 180, 320}
    check_steps.add(steps)
    for step, base, dose in updates:
        familiar = [common[i] for i in base] + [paired[i][0] for i in dose]
        target = [common[i] for i in base] + [paired[i][1] for i in dose]
        assert [s.gold for s in familiar] == [s.gold for s in target]
        a = pilot.encode_sources(familiar, vocab, check=False)
        b = pilot.encode_sources(target, vocab, check=False)
        assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
        loss_f = pilot.one_update(f, optf, a, device)
        loss_t = pilot.one_update(t, optt, b, device)
        if step in check_steps:
            trajectory.append({'step': step, 'familiar_loss': loss_f,
                               'target_loss': loss_t,
                               'familiar_sha256': hash_weights(f),
                               'target_sha256': hash_weights(t)})
    assert trajectory[-1]['step'] == steps
    return {'replicate': rep, 'initial_sha256': init_f, 'trace': trajectory}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    ap.add_argument('--steps', type=int, default=90)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--schedule-only', action='store_true',
                    help='Compare the full 3200-step source-index prefix without doing any optimization')
    ap.add_argument('--json-out', type=Path, default=None,
                    help='Optional NEW diagnostic report; existing paths are never overwritten')
    args = ap.parse_args()
    if not (1 <= args.steps <= 3200 and 2 <= args.repeats <= 4):
        ap.error('Require 1 <= --steps <= 3200 and 2 <= --repeats <= 4')

    # Reuse the same verified original setup and source/vocabulary builders.
    _, _, common, paired, bpool, dpool, dev, vocab = long.setup()
    updates = get_first_updates(common, paired, bpool, dpool, args.steps)
    h = sha256()
    for step, base, dose in updates:
        h.update((json.dumps({'step': step, 'base': base, 'dose': dose},
                             separators=(',', ':')) + '\n').encode())
    sched_prefix = h.hexdigest()
    print(f'PASS: original and extended samplers emit identical ordered source indices for first {args.steps} updates', flush=True)
    print(f'SCHEDULE_PREFIX_SHA256={sched_prefix}', flush=True)
    if args.steps == 3200:
        old = ROOT / 'reports/exp105/development_pilot_v0/pilot_seed10801_step3200.json'
        if old.exists():
            data = json.loads(old.read_text())
            expected = data['candidate_schedule_sha256_prefix']
            assert sched_prefix == expected, 'Original archived 3200-step schedule digest differs!'
            print('PASS: source index digest matches original archived 3200-step report', flush=True)

    if args.schedule_only:
        print('SCHEDULE-ONLY PASS: no weights or optimizer updates created.')
        return

    device = pilot.get_device(args.device)
    result = {'study': 'EXP105_READ_ONLY_REPLAY_DIAGNOSTIC',
              'device': str(device), 'seed': SEED, 'steps': args.steps,
              'schedule_prefix_sha256': sched_prefix,
              'no_LI_or_protected_scoring': True, 'replicates': []}
    for rep in range(args.repeats):
        r = replay_one(rep + 1, device, args.steps, common, paired, dev, vocab, updates)
        result['replicates'].append(r)
        print(f'REPLAY {rep+1}/{args.repeats} INIT={r["initial_sha256"][:16]}', flush=True)
        for step in r['trace']:
            print(f'  {step["step"]:4d}: F loss={step["familiar_loss"]:.8f} hash={step["familiar_sha256"][:12]} '
                  f'T loss={step["target_loss"]:.8f} hash={step["target_sha256"][:12]}', flush=True)

    assert len({r['initial_sha256'] for r in result['replicates']}) == 1, 'DIFFERENT INITIAL WEIGHTS'
    traces = [r['trace'] for r in result['replicates']]
    for i in range(len(traces[0])):
        for label in ('familiar_sha256', 'target_sha256'):
            if len({t[i][label] for t in traces}) > 1:
                step = traces[0][i]['step']
                print(f'DIVERGED by update {step}: {label}. Initial weights and ordered minibatch indices were identical.',flush=True)
                print('Possible numerical/backend nondeterminism; verify with a CPU replay and across separate processes.',flush=True)
                break
        else:
            continue
        break
    else:
        print(f'PASS: {args.repeats} within-process numerical replays matched exactly at all checked steps.', flush=True)
        print('This does not prove separate-process MPS replays are identical.',flush=True)

    if args.json_out is not None:
        dest = args.json_out
        if dest.exists():
            raise FileExistsError(f'Will not overwrite existing diagnostic: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(f'SAVED NEW DIAGNOSTIC: {dest}')
    print('READ-ONLY TO ORIGINAL EXPERIMENTS; no reserved, protected or L+I scoring.',flush=True)


if __name__ == '__main__':
    main()
