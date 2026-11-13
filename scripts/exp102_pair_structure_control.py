#!/usr/bin/env python3
"""EXP-102 exploratory question-pair structure comparison.

Tests two ways of using exactly 32 source sentences and 64 distinct
question/sentence pairs per update (16 examples of each query type).

ROLE_PAIR: 16 sentences with first-event agent+patient, 16 with
           second-event agent+patient.
EVENT_PAIR: 16 sentences with first+second event agent, 16 with
            first+second event patient.

Both methods hold the dataset, model, optimizer, seed, initial weights,
batch size, unique source count and query-type counts fixed. No protected cases.
The previously run quartet condition has only 16 sources/update, so this is
an exploratory *complementarity* comparison, not an exact quartet replication.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from random import Random

import torch
from torch.nn import functional as F

from exp102_learning_pilot import NAMES, Model, make_packets, encode
from exp102_role_trajectory import diagnose, show


def data():
    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    ood, known = groups[:14], groups[14:]
    train = make_packets(known, 2048, 10201)
    seen = {s for s, _ in train}
    dev_id = make_packets(known, 256, 10202, excluded=seen)
    dev_ood = make_packets(ood, 256, 10203)
    assert len(seen) == 2048
    assert not seen.intersection(s for s, _ in dev_id + dev_ood)
    assert not {s for s, _ in dev_id}.intersection(s for s, _ in dev_ood)
    fingerprint = hashlib.sha256(
        '\n'.join(s for s, _ in train).encode('utf-8')
    ).hexdigest()
    return encode(train), encode(dev_id), encode(dev_ood), fingerprint


def sample_indices(rng, method, n_sources):
    # Draw 32 *different* sources, to ensure 64 unique question-source labels.
    sources = torch.randperm(n_sources, generator=rng)[:32]
    if method == 'role_pair':
        # One event, both semantic roles, for each source.
        q = torch.tensor([0, 1], dtype=torch.long).repeat(16, 1)
        q = torch.cat([q, torch.tensor([2, 3]).repeat(16, 1)], dim=0)
    elif method == 'event_pair':
        # One semantic role, both events, for each source.
        q = torch.tensor([0, 2], dtype=torch.long).repeat(16, 1)
        q = torch.cat([q, torch.tensor([1, 3]).repeat(16, 1)], dim=0)
    else:
        raise ValueError(method)
    indices = (4 * sources[:, None] + q).reshape(-1)
    assert indices.shape == (64,)
    assert len(set(indices.tolist())) == 64
    assert torch.bincount(q.flatten(), minlength=4).tolist() == [16, 16, 16, 16]
    assert len(set(sources.tolist())) == 32
    return indices


def run(seed, method, steps, device, train, dev_id, dev_ood, digest, outdir):
    artifact = outdir / f'exploratory_seed{seed}_{method}_step{steps}.json'
    checkpoint = Path('results/exp102') / f'exploratory_seed{seed}_{method}_step{steps}.pt'
    if artifact.exists() or checkpoint.exists():
        raise FileExistsError(f'Refusing to overwrite {artifact} or {checkpoint}')
    torch.manual_seed(seed)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    # Same random source index stream in both conditions for each seed.
    generator = torch.Generator(device='cpu').manual_seed(seed + 100)
    x, pos, y, _ = train
    checkpoints = {1, 400, 800, 1200, steps}
    records = []
    print(f'\nRUN seed={seed} method={method} steps={steps}', flush=True)
    for step in range(1, steps + 1):
        indices = sample_indices(generator, method, len(x) // 4)
        model.train()
        logits = model(x[indices].to(device), pos[indices].to(device))
        loss = F.cross_entropy(logits, y[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            id_scores = diagnose(model, dev_id, device)
            ood_scores = diagnose(model, dev_ood, device)
            print(f'train step={step} loss={loss.item():.4f}', flush=True)
            show(step, 'ID', id_scores)
            show(step, 'OOD', ood_scores)
            records.append({'step': step, 'id': id_scores, 'ood': ood_scores})
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'study': 'EXP-102', 'status': 'EXPLORATORY_NOT_CONFIRMATORY',
        'seed': seed, 'method': method, 'steps': steps, 'device': device,
        'train_texts_sha256': digest, 'batch_sources': 32,
        'unique_training_question_source_pairs_per_batch': 64,
        'labels_per_query_type': 16, 'records': records,
    }
    artifact.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'sampling_rng': generator.get_state(), 'metadata': result}, checkpoint)
    print('SAVED', artifact, flush=True)
    print('SAVED', checkpoint, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[10204, 10214])
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--outdir', default='reports/exp102/pair_structure_control')
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Steps must be positive and seeds unique')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available')
    print('EXP-102 PAIR-STRUCTURE CONTROL (exploratory)', flush=True)
    print('32 distinct sources x 2 distinct questions; 64 labels per update.', flush=True)
    print('16 examples per query type in both methods.', flush=True)
    print('Compare role-complement pairs against event-complement pairs.', flush=True)
    print('No protected cases, no changes to prior checkpoints.', flush=True)
    train, dev_id, dev_ood, digest = data()
    outcomes = []
    for seed in args.seeds:
        for method in ('role_pair', 'event_pair'):
            outcomes.append(run(seed, method, args.steps, args.device,
                                train, dev_id, dev_ood, digest, Path(args.outdir)))
    print('\nFINAL OOD SUMMARY', flush=True)
    for result in outcomes:
        r = result['records'][-1]['ood']
        print(f"seed={result['seed']} {result['method']:10s} "
              f"accuracy4={r['accuracy4']:.3f} C_query={r['C_query']:.3f} "
              f"copy_first={r['first']['role_copy']:.3f} "
              f"copy_second={r['second']['role_copy']:.3f}", flush=True)
    print('EXPLORATORY ONLY. No protected evaluation.', flush=True)


if __name__ == '__main__':
    main()
