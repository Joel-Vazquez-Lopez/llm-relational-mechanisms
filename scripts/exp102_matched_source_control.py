#!/usr/bin/env python3
"""EXP-102 exploratory matched-source question-completeness control.

Both conditions use exactly 16 sampled source sentences per update, four
question slots per source (64 supervised labels), and exactly 16 slots for
*each* of the four question types. Only the within-source query coverage differs.

quartet: all four distinct queries for every source.
incomplete: three distinct queries for every source; one duplicated and one
omitted. This deliberately changes the number of *distinct* supervised labels
(64 vs 48), an important remaining confound.

No protected cases or changes to prior experiments; exploratory only.
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


# Each pattern omits one question and doubles another. Four repetitions of
# these rows produce exactly 16 appearances of every question type in a batch.
PATTERNS = torch.tensor([
    [1, 1, 2, 3],  # missing 0; repeat 1
    [0, 2, 2, 3],  # missing 1; repeat 2
    [0, 1, 3, 3],  # missing 2; repeat 3
    [0, 0, 1, 2],  # missing 3; repeat 0
], dtype=torch.long).repeat_interleave(4, dim=0)
ALL_FOUR = torch.arange(4, dtype=torch.long).repeat(16, 1)


def check_batch(source_ids, query_ids, method):
    assert source_ids.shape == (16,)
    assert query_ids.shape == (16, 4)
    assert sorted(torch.bincount(query_ids.flatten(), minlength=4).tolist()) == [16] * 4
    assert torch.all(query_ids.ge(0) & query_ids.lt(4))
    unique_counts = [len(set(row.tolist())) for row in query_ids]
    assert unique_counts == [4 if method == 'quartet' else 3] * 16
    indices = (4 * source_ids[:, None] + query_ids).reshape(-1)
    assert indices.numel() == 64
    return indices


def prepare_data():
    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    heldout_groups, training_groups = groups[:14], groups[14:]
    train = make_packets(training_groups, 2048, 10201)
    used = {text for text, _ in train}
    dev_id = make_packets(training_groups, 256, 10202, excluded=used)
    dev_ood = make_packets(heldout_groups, 256, 10203)
    assert len(train) == 2048 and len(dev_id) == len(dev_ood) == 256
    assert not used.intersection(text for text, _ in dev_id + dev_ood)
    assert not {s for s, _ in dev_id}.intersection(s for s, _ in dev_ood)
    digest = hashlib.sha256('\n'.join(s for s, _ in train).encode()).hexdigest()
    return encode(train), encode(dev_id), encode(dev_ood), digest


def run_one(seed, method, steps, device, output_dir, train_data, id_data, ood_data, digest):
    torch.manual_seed(seed)
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    # Source index stream is *identical* between conditions for a given seed.
    source_rng = torch.Generator(device='cpu').manual_seed(seed + 1)
    pattern_rng = torch.Generator(device='cpu').manual_seed(seed + 2)
    ids, readout, labels, _ = train_data
    milestones = {1, 400, 800, 1200, steps}
    records = []
    output = output_dir / f'exploratory_seed{seed}_{method}_step{steps}.json'
    checkpoint = Path('results/exp102') / f'exploratory_seed{seed}_{method}_matched_step{steps}.pt'
    for path in (output, checkpoint):
        if path.exists():
            raise FileExistsError(f'Will not overwrite existing artifact: {path}')

    print(f'\nRUN seed={seed} method={method} steps={steps}', flush=True)
    for step in range(1, steps + 1):
        source_ids = torch.randint(2048, (16,), generator=source_rng)
        if method == 'quartet':
            query_ids = ALL_FOUR
        else:
            perm = torch.randperm(16, generator=pattern_rng)
            query_ids = PATTERNS[perm]
        indices = check_batch(source_ids, query_ids, method)

        model.train()
        logits = model(ids[indices].to(device), readout[indices].to(device))
        loss = F.cross_entropy(logits, labels[indices].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in milestones:
            id_result = diagnose(model, id_data, device)
            ood_result = diagnose(model, ood_data, device)
            print(f'train step={step} loss={loss.item():.4f}', flush=True)
            show(step, 'ID', id_result)
            show(step, 'OOD', ood_result)
            records.append({'step': step, 'id': id_result, 'ood': ood_result})

    result = {
        'study': 'EXP-102', 'status': 'EXPLORATORY_NOT_CONFIRMATORY',
        'seed': seed, 'method': method, 'steps': steps,
        'train_source_text_sha256': digest,
        'batch_slots': 64, 'batch_sources_drawn': 16,
        'questions_per_type': 16,
        'unique_query_types_per_source': 4 if method == 'quartet' else 3,
        'distinct_training_labels_per_batch_before_repeated_sources': 64 if method == 'quartet' else 48,
        'records': records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'seed': seed, 'method': method, 'step': steps,
        'train_source_text_sha256': digest,
        'source_rng': source_rng.get_state(),
        'pattern_rng': pattern_rng.get_state(),
    }, checkpoint)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print('SAVED', output, flush=True)
    print('SAVED', checkpoint, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[10204, 10214])
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--out', default='reports/exp102/matched_source_control')
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error('Steps must be positive and seeds unique')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS unavailable')
    assert sorted(torch.bincount(PATTERNS.flatten(), minlength=4).tolist()) == [16] * 4
    assert len({tuple(row.tolist()) for row in PATTERNS}) == 4

    print('EXP-102 MATCHED-SOURCE EXPLORATORY CONTROL', flush=True)
    print('Same 16 sampled sources, same 64 label slots and 16 labels/query type.', flush=True)
    print('Incomplete arm repeats one query and omits another per source.', flush=True)
    print('NOTE: distinct supervised labels differ (64 vs 48); NOT a perfect isolation.', flush=True)
    print('No protected evaluation, no EXP-101 modification.', flush=True)
    train_data, id_data, ood_data, digest = prepare_data()
    results = []
    for seed in args.seeds:
        for method in ['quartet', 'incomplete']:
            results.append(run_one(seed, method, args.steps, args.device,
                                   Path(args.out), train_data, id_data, ood_data, digest))
    print('\nSUMMARY OOD at final step:')
    for result in results:
        final = result['records'][-1]['ood']
        print(f"seed={result['seed']} {result['method']:10s} "
              f"acc={final['accuracy4']:.3f} C_query={final['C_query']:.3f} "
              f"copy_first={final['first']['role_copy']:.3f} "
              f"copy_second={final['second']['role_copy']:.3f}", flush=True)
    print('EXPLORATORY FINISHED. No protected data accessed.', flush=True)


if __name__ == '__main__':
    main()
