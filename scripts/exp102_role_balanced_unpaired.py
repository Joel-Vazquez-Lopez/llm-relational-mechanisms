#!/usr/bin/env python3
"""EXP-102 exploratory control: balanced role questions WITHOUT same-source quartets.

Compare against saved IID and quartet runs. No protected evaluation.
"""
import argparse
from itertools import combinations
from pathlib import Path
from random import Random

import torch
from torch.nn import functional as F

from exp102_learning_pilot import NAMES, Model, make_packets, encode
from exp102_role_trajectory import diagnose, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = ap.parse_args()
    if args.steps < 1:
        ap.error('--steps must be positive')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')

    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    heldout, training = groups[:14], groups[14:]
    train = make_packets(training, 2048, 10201)
    used = {text for text, _ in train}
    dev_id = make_packets(training, 256, 10202, excluded=used)
    dev_ood = make_packets(heldout, 256, 10203)
    assert not used.intersection(text for text, _ in dev_id + dev_ood)
    assert len(train) == 2048

    ids, readout, labels, _ = encode(train)
    id_data, ood_data = encode(dev_id), encode(dev_ood)

    torch.manual_seed(10204)
    model = Model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    rng = torch.Generator(device='cpu').manual_seed(10205)
    offsets = torch.arange(4)[None, :]
    milestones = {1, 400, 800, 1200, 1600, args.steps}

    print('EXP-102 BALANCED-UNPAIRED CONTROL', flush=True)
    print('16 independent sentences per query type; 64 labels/update.', flush=True)
    print('All four query types exactly balanced; no forced same-source quartets.', flush=True)
    print('Same dataset, architecture, seed and optimizer. Exploratory, no protected data.', flush=True)

    for step in range(1, args.steps + 1):
        # Each column uses independently sampled source sentences.
        # Every batch contains 16 of each query type (0, 1, 2, 3).
        sampled = torch.randint(len(train), (16, 4), generator=rng)
        indices = (4 * sampled + offsets).reshape(-1)
        assert indices.numel() == 64
        assert torch.equal((indices % 4).reshape(-1, 4)[0], torch.arange(4))

        model.train()
        logits = model(ids[indices].to(args.device), readout[indices].to(args.device))
        loss = F.cross_entropy(logits, labels[indices].to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step in milestones:
            print(f'train step={step} loss={loss.item():.4f}', flush=True)
            show(step, 'ID', diagnose(model, id_data, args.device))
            show(step, 'OOD', diagnose(model, ood_data, args.device))

    path = Path(f'results/exp102/exploratory_seed10204_balanced_unpaired_step{args.steps}.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'step': args.steps, 'seed': 10204,
                'batch_mode': 'balanced_unpaired'}, path)
    print('Saved:', path, flush=True)
    print('No protected evaluation performed.', flush=True)


if __name__ == '__main__':
    main()
