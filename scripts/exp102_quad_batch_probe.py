#!/usr/bin/env python3
"""EXP-102 exploratory batch-structure intervention.

Compare to the saved independent-question trajectory at seed 10204.
This run changes ONLY how the 64 training questions per update are sampled:
16 source sentences x all 4 event/role questions instead of 64 IID questions.
No protected examples, early stopping, or EXP-101 modifications.
"""
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
from random import Random

import torch
from torch.nn import functional as F

from exp102_learning_pilot import NAMES, Model, encode, make_packets
from exp102_role_trajectory import diagnose, show


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be >= 1')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable')

    # Reproduce the original trajectory's train/dev/OOD split exactly.
    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    heldout_groups, training_groups = groups[:14], groups[14:]
    train = make_packets(training_groups, 2048, 10201)
    used = {text for text, _ in train}
    dev_id = make_packets(training_groups, 256, 10202, excluded=used)
    dev_ood = make_packets(heldout_groups, 256, 10203)
    assert not used.intersection(text for text, _ in dev_id + dev_ood)
    assert not {s for s, _ in dev_id}.intersection(s for s, _ in dev_ood)
    assert len(train) == 2048 and len(dev_id) == len(dev_ood) == 256

    train_data = encode(train)
    id_data, ood_data = encode(dev_id), encode(dev_ood)
    ids, readout, labels, _ = train_data
    assert len(ids) == 4 * len(train)

    # Match seed, architecture, optimizer, learning rate, and weight decay.
    torch.manual_seed(10204)
    model = Model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    sample_rng = torch.Generator(device='cpu').manual_seed(10205)
    role_offsets = torch.arange(4)
    milestones = {1, 400, 800, 1200, 1600, 2400, 3200, args.steps}

    print('EXP-102 EXPLORATORY FOUR-QUERY BATCHING INTERVENTION', flush=True)
    print('16 sentences x 4 queries = 64 labels/update; compared with 64 IID labels/update.', flush=True)
    print('Same data, initialization, architecture and optimizer as role trajectory.', flush=True)
    print('No protected data or EXP-101 checkpoint access.', flush=True)

    for step in range(1, args.steps + 1):
        # Every selected sentence contributes all four complementary questions.
        sentence_ids = torch.randint(len(train), (16,), generator=sample_rng)
        ix = (4 * sentence_ids[:, None] + role_offsets[None, :]).reshape(-1)
        assert len(ix) == 64
        assert torch.equal(ix.reshape(-1, 4)[:, 1:] - ix.reshape(-1, 4)[:, :-1],
                           torch.ones(16, 3, dtype=torch.long))

        model.train()
        logits = model(ids[ix].to(args.device), readout[ix].to(args.device))
        loss = F.cross_entropy(logits, labels[ix].to(args.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in milestones:
            print(f'train step={step} loss={loss.item():.4f}', flush=True)
            show(step, 'ID', diagnose(model, id_data, args.device))
            show(step, 'OOD', diagnose(model, ood_data, args.device))

    path = Path(f'results/exp102/exploratory_seed10204_quad_batch_step{args.steps}.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    torch.save({
        'step': args.steps, 'seed': 10204, 'batch_mode': '16_sources_x_all_four_queries',
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'sample_rng': sample_rng.get_state(), 'torch_cpu_rng': torch.get_rng_state(),
        'torch_mps_rng': torch.mps.get_rng_state() if args.device == 'mps' else None,
    }, path)
    print('Saved:', path, flush=True)
    print('EXPLORATORY ONLY: not a confirmatory result. No protected cases accessed.', flush=True)


if __name__ == '__main__':
    main()
