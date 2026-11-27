#!/usr/bin/env python3
"""Exploratory checkpoint recovery; not uninterrupted AdamW continuation."""
import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from random import Random

import torch
import role_diversity_complex_v0 as parent

ROOT = parent.ROOT
OUT = ROOT / 'reports/role_diversity_recovery_v0'
WEIGHTS = ROOT / 'results/role_diversity_recovery_v0'
SEEDS = (16101, 16127, 16143, 16167)
START = 6400
END = 51200
MILESTONES = (6400, 12800, 25600, 51200)
STUDY = 'ROLE-DIVERSITY-RECOVERY-v0'
PROTOCOL = ROOT / 'paper/ROLE_DIVERSITY_RECOVERY_V0_PROTOCOL.md'
MANIFEST = OUT / 'study_manifest.json'


def schedule(seed, train):
    rng = Random(seed + 16131013)
    perm = {}
    for ix in train:
        cells = list(parent.CELLS)
        rng.shuffle(cells)
        perm[ix] = tuple(cells)
    draw = Counter()
    for step in range(1, END + 1):
        chosen = tuple(rng.sample(train, 8))
        events = []
        for ix in chosen:
            k = draw[ix]
            fi, mi = perm[ix][k % 45]
            events.append((ix, fi, mi, k))
            draw[ix] += 1
        yield step, chosen, tuple(events)


def line(row):
    return (json.dumps(row, separators=(',', ':')) + '\n').encode()


def source_path(seed):
    return parent.WEIGHTS / f'seed{seed}_mps_step6400.pt'


def source_report(seed):
    return parent.OUT / f'seed{seed}_mps_step6400.json'


def checkpoint_path(seed, step):
    return WEIGHTS / f'seed{seed}_mps_step{step}.pt'


def report_path(seed, step):
    return OUT / f'seed{seed}_mps_step{step}.json'


def build_manifest(train, m):
    hashes = {}
    for seed in SEEDS:
        prefix = hashlib.sha256()
        full = hashlib.sha256()
        original = iter(parent.schedule(seed, train))
        for row in schedule(seed, train):
            if row[0] <= START:
                assert row == next(original), (seed, row[0])
                prefix.update(line(row))
            full.update(line(row))
        assert next(original, None) is None
        assert prefix.hexdigest() == m['protocol']['schedule_sha256'][str(seed)]
        hashes[str(seed)] = dict(prefix=prefix.hexdigest(), full=full.hexdigest(),
                                 weights=parent.sha(source_path(seed)),
                                 report=parent.sha(source_report(seed)))
    p = dict(study=STUDY, classification='exploratory_fresh_optimizer_recovery',
             parent_protocol_digest=m['protocol_digest'], seeds=list(SEEDS),
             arms=list(parent.ARMS), milestones=list(MILESTONES),
             lr=parent.old.LR, weight_decay=parent.old.WD,
             script_sha256=parent.sha(Path(__file__)),
             protocol_sha256=parent.sha(PROTOCOL), inputs=hashes)
    return dict(protocol=p, protocol_digest=parent.digest(p))


def inputs():
    train, held, training, withheld, vocab, m = parent.verified()
    return train, held, training, withheld, vocab, m


def prepare():
    train, _, _, _, _, m = inputs()
    manifest = build_manifest(train, m)
    if MANIFEST.exists():
        assert json.loads(MANIFEST.read_text()) == manifest, 'Manifest drift'
    else:
        parent.save_once(MANIFEST, manifest)
    print('PREFLIGHT PASS: all original schedule prefixes and source hashes verified')
    print(manifest['protocol_digest'])


def committed(path):
    relative = str(path.relative_to(ROOT))
    assert subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT) == path.read_bytes(), relative


def load_source(seed, m):
    ck = torch.load(source_path(seed), map_location='cpu', weights_only=True)
    assert ck['study'] == 'ROLE-DIVERSITY-COMPLEX-v0'
    assert ck['seed'] == seed and ck['steps'] == START
    assert ck['protocol_digest'] == m['protocol_digest']
    assert set(ck['models']) == set(parent.ARMS)
    return ck['models']


def score(nets, tests, device):
    outcomes = {}
    for arm, (model, _) in nets.items():
        model.eval()
        outcomes[arm] = {}
        for pop in parent.POPS:
            data = tests[pop][arm] if pop == 'train_seen_rich' else tests[pop]
            outcomes[arm][pop] = parent.old.score_native(model, data, device)
        model.train()
    return outcomes


def displacement(model, initial):
    squared, baseline = 0.0, 0.0
    for name, param in model.named_parameters():
        p = param.detach().cpu().double()
        q = initial[name].double()
        squared += float((p-q).square().sum())
        baseline += float(q.square().sum())
    return math.sqrt(squared), math.sqrt(squared / baseline)


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    torch.save(state, tmp)
    tmp.rename(path)


def run(seed, resume=None):
    assert seed in SEEDS
    assert torch.backends.mps.is_available(), 'MPS required'
    device = torch.device('mps')
    train, held, training, withheld, vocab, m = inputs()
    manifest = build_manifest(train, m)
    assert json.loads(MANIFEST.read_text()) == manifest
    for path in (Path(__file__).resolve(), PROTOCOL, MANIFEST):
        committed(path)
    parent.require_frozen()
    original = load_source(seed, m)
    tests = parent.evaluations(train, held, training, withheld, vocab)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    nets = {}
    for arm in parent.ARMS:
        model = parent.old.Model(len(vocab)).to(device)
        model.load_state_dict(original[arm], strict=True)
        nets[arm] = (model, torch.optim.AdamW(model.parameters(), lr=parent.old.LR,
                                              weight_decay=parent.old.WD))
    current = START
    if resume is not None:
        assert resume in MILESTONES and START <= resume < END
        state = torch.load(checkpoint_path(seed, resume), map_location='cpu', weights_only=True)
        assert state['protocol_digest'] == manifest['protocol_digest']
        assert state['seed'] == seed and state['step'] == resume
        for arm, (model, opt) in nets.items():
            model.load_state_dict(state['models'][arm])
            opt.load_state_dict(state['optimizers'][arm])
        torch.set_rng_state(state['cpu_rng'])
        torch.mps.set_rng_state(state['mps_rng'])
        current = resume
        assert report_path(seed, resume).exists(), 'Resume requires sealed report'
    for step in MILESTONES:
        if step > current or (step == START and resume is None):
            assert not checkpoint_path(seed, step).exists()
            assert not report_path(seed, step).exists()
    trace = hashlib.sha256()
    sums = dict.fromkeys(parent.ARMS, 0.0)
    count = 0
    windows = []

    def seal(step):
        outcomes = score(nets, tests, device)
        if step == START:
            reference = json.loads(source_report(seed).read_text())['outcomes']
            for a in parent.ARMS:
                for p in parent.POPS:
                    for v in parent.VOICES:
                        for k in ('C_query', 'accuracy4', 'full_vocab_accuracy'):
                            assert abs(outcomes[a][p][v][k] - reference[a][p][v][k]) <= .001, (a,p,v,k)
        changes = {a: displacement(nets[a][0], original[a]) for a in parent.ARMS}
        state = dict(study=STUDY, seed=seed, step=step,
                     protocol_digest=manifest['protocol_digest'],
                     schedule_prefix_sha256=trace.hexdigest(),
                     models={a: n.state_dict() for a,(n,o) in nets.items()},
                     optimizers={a: o.state_dict() for a,(n,o) in nets.items()},
                     cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state())
        save_state(checkpoint_path(seed, step), state)
        parent.save_once(report_path(seed, step), dict(study=STUDY, seed=seed, step=step,
            protocol_digest=manifest['protocol_digest'], schedule_prefix_sha256=trace.hexdigest(),
            outcomes=outcomes, loss_windows=windows, displacement=changes,
            classification='exploratory_fresh_optimizer_recovery'))
        print('SEALED', seed, step, flush=True)

    for row in schedule(seed, train):
        step, _, events = row
        trace.update(line(row))
        if step <= current:
            if step == current:
                if resume is None:
                    seal(step)
                else:
                    assert trace.hexdigest() == state['schedule_prefix_sha256']
            continue
        batches = parent.make_batch(events, training)
        for arm in parent.ARMS:
            encoded = parent.old.encode_sources(batches[arm], vocab, check=False)
            loss = parent.old.one_update(*nets[arm], encoded, device)
            assert math.isfinite(loss)
            sums[arm] += loss
        count += 1
        if step % 800 == 0:
            windows.append(dict(step=step, updates=count, mean_loss={a:x/count for a,x in sums.items()}))
            print('STEP', seed, windows[-1], flush=True)
            sums = dict.fromkeys(parent.ARMS, 0.0)
            count = 0
        if step in MILESTONES:
            seal(step)
            windows.clear()
    assert trace.hexdigest() == manifest['protocol']['inputs'][str(seed)]['full']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', type=int)
    a = parser.parse_args()
    if a.phase == 'prepare':
        prepare()
    else:
        run(a.seed, a.resume)


if __name__ == '__main__':
    main()
