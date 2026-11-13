#!/usr/bin/env python3
"""EXP-102 exploratory diagnostic: event-selection vs role-selection learning trajectory.

Uses the identical data/initialization/optimizer/sampling configuration of
exp102_learning_pilot.py. No protected examples or exposure-arm comparisons.
"""
import argparse
from itertools import combinations
from pathlib import Path
from random import Random

import torch
from torch.nn import functional as F

from exp102_learning_pilot import NAMES, Model, make_packets, encode


@torch.inference_mode()
def diagnose(model, encoded, device):
    model.eval()
    ids, readout, gold, choices = encoded
    output = []
    for a in range(0, len(ids), 128):
        b = a + 128
        logits = model(ids[a:b].to(device), readout[a:b].to(device)).float().cpu()
        output.append(logits)
    logits = torch.cat(output).reshape(-1, 4, model.out.out_features)
    gold = gold.reshape(-1, 4)
    c = choices.reshape(-1, 4, 4)
    candidates = logits.gather(2, c)
    pred = c.gather(2, candidates.argmax(2, keepdim=True)).squeeze(2)
    hit = pred.eq(gold)

    def event_stats(e):
        j = 2 * e
        local = (pred[:, j:j+2] == gold[:, j:j+1]) | (
            pred[:, j:j+2] == gold[:, j+1:j+2])
        both = hit[:, j:j+2].all(1)
        copied = pred[:, j] == pred[:, j+1]
        # Difference between the two role-question logit margins over the
        # exact same pair of event-local gold names. A positive number means
        # the agent question *relatively* favors the agent more than the
        # patient question does; it is not proof of a complete role circuit.
        a = gold[:, j:j+1]
        p = gold[:, j+1:j+2]
        agent_query = logits[:, j]
        patient_query = logits[:, j+1]
        delta = (
            agent_query.gather(1, a) - agent_query.gather(1, p)
            - patient_query.gather(1, a) + patient_query.gather(1, p)
        ).squeeze(1)
        return {
            "event_member": local.float().mean().item(),
            "role_copy": copied.float().mean().item(),
            "both_correct": both.float().mean().item(),
            "role_signal_positive": (delta > 0).float().mean().item(),
            "role_delta_mean": delta.mean().item(),
        }

    return {
        "accuracy4": hit.float().mean().item(),
        "C_query": hit.all(1).float().mean().item(),
        "first": event_stats(0),
        "second": event_stats(1),
    }


def show(step, label, report):
    a, b = report['first'], report['second']
    print(
        f"STEP={step:5d} {label:3s} "
        f"acc={report['accuracy4']:.3f} Cquery={report['C_query']:.3f} "
        f"event_hit=({a['event_member']:.3f},{b['event_member']:.3f}) "
        f"role_copy=({a['role_copy']:.3f},{b['role_copy']:.3f}) "
        f"both_correct=({a['both_correct']:.3f},{b['both_correct']:.3f}) "
        f"role_signal=({a['role_signal_positive']:.3f},{b['role_signal_positive']:.3f}) "
        f"mean_delta=({a['role_delta_mean']:+.3f},{b['role_delta_mean']:+.3f})",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=3200)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = p.parse_args()
    if args.steps < 1:
        p.error('--steps must be positive')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS is not available')

    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    heldout_groups, training_groups = groups[:14], groups[14:]
    assert not set(heldout_groups) & set(training_groups)
    train = make_packets(training_groups, 2048, 10201)
    used = {text for text, _ in train}
    dev_id = make_packets(training_groups, 256, 10202, excluded=used)
    dev_ood = make_packets(heldout_groups, 256, 10203)
    assert not used.intersection(s for s, _ in dev_id + dev_ood)
    encoded_train = encode(train)
    id_data = encode(dev_id)
    ood_data = encode(dev_ood)

    torch.manual_seed(10204)
    model = Model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    sample_rng = torch.Generator(device='cpu').manual_seed(10205)
    inputs, readout, labels, _ = encoded_train
    milestones = {1, 400, 800, 1200, 1600, 2400, 3200, args.steps}
    print('EXP-102 exploratory role trajectory. One seed, no protected evaluation.', flush=True)
    print('First and second metrics refer to event position; role signal is logit-contrast sign.', flush=True)

    for step in range(1, args.steps + 1):
        ix = torch.randint(len(inputs), (64,), generator=sample_rng)
        model.train()
        logits = model(inputs[ix].to(args.device), readout[ix].to(args.device))
        loss = F.cross_entropy(logits, labels[ix].to(args.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in milestones:
            print(f"train step={step} loss={loss.item():.4f}", flush=True)
            show(step, 'ID', diagnose(model, id_data, args.device))
            show(step, 'OOD', diagnose(model, ood_data, args.device))

    checkpoint = Path(f'results/exp102/exploratory_seed10204_step{args.steps}.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {checkpoint}')
    torch.save({
        'step': args.steps, 'seed': 10204, 'device': args.device,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'sample_rng': sample_rng.get_state(), 'torch_cpu_rng': torch.get_rng_state(),
        'torch_mps_rng': torch.mps.get_rng_state() if args.device == 'mps' else None,
    }, checkpoint)
    print('Saved exploratory checkpoint:', checkpoint, flush=True)
    print('No protected data accessed. No EXP-101 artifacts modified.', flush=True)


if __name__ == '__main__':
    main()
