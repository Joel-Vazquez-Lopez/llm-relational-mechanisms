#!/usr/bin/env python3
"""EXP-102 exploratory learning-feasibility pilot. No protected evaluation.

Tests whether an ML-080-sized Transformer can learn the basic two-event,
four-query task before any harder grammar or exposure comparison is introduced.
"""
import argparse
from itertools import combinations
from pathlib import Path
from random import Random
import re

import torch
from torch import nn
from torch.nn import functional as F

from exp102_preflight import NAMES, VERBS, Event, sentence

QUESTIONS = (
    'In the first event, who was the agent?',
    'In the first event, who was the patient?',
    'In the second event, who was the agent?',
    'In the second event, who was the patient?',
)
VOCAB_WORDS = tuple(dict.fromkeys(
    ('<pad>', '<bos>', '<sep>', '<answer>') + NAMES + VERBS +
    ('was', 'by', 'In', 'the', 'first', 'second', 'event', 'who', 'agent', 'patient', ';', ',', '.', '?')
))
VOCAB = {word: i for i, word in enumerate(VOCAB_WORDS)}
WORD_RE = re.compile(r'[A-Za-z]+|[;,.?]')
MAX_TOKENS = 48


def make_packets(groups, count, seed, excluded=()):
    """Draw unique sources from specified four-name sets."""
    rng = Random(seed)
    packets, used = [], set(excluded)
    while len(packets) < count:
        people = list(rng.choice(groups))
        rng.shuffle(people)
        a, b, c, d = people
        verb_a, verb_b = rng.sample(VERBS, 2)
        voice_a, voice_b = rng.choice(('active', 'passive')), rng.choice(('active', 'passive'))
        events = (Event(a, b, verb_a, voice_a), Event(c, d, verb_b, voice_b))
        if rng.randrange(2):
            events = events[::-1]
        text = sentence(events)
        if text in used:
            continue
        used.add(text)
        answers = (events[0].agent, events[0].patient, events[1].agent, events[1].patient)
        assert len(set(answers)) == 4
        packets.append((text, answers))
    return packets


def encode(packets):
    sequences, gold, choices = [], [], []
    for text, answers in packets:
        names = tuple(sorted(answers))
        option_ids = [VOCAB[name] for name in names]
        for q, answer in zip(QUESTIONS, answers):
            seq = ['<bos>'] + WORD_RE.findall(text) + ['<sep>'] + WORD_RE.findall(q) + ['<answer>']
            assert all(t in VOCAB for t in seq)
            assert len(seq) <= MAX_TOKENS, len(seq)
            sequences.append([VOCAB[t] for t in seq])
            gold.append(VOCAB[answer])
            choices.append(option_ids)
    ids = torch.full((len(sequences), max(map(len, sequences))), VOCAB['<pad>'], dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq)
    readout = torch.tensor([len(s) - 1 for s in sequences])
    return ids, readout, torch.tensor(gold), torch.tensor(choices)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(96)
        self.attn = nn.MultiheadAttention(96, 4, batch_first=True)
        self.n2 = nn.LayerNorm(96)
        self.mlp = nn.Sequential(nn.Linear(96, 384), nn.GELU(), nn.Linear(384, 96))

    def forward(self, x, mask):
        z = self.n1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), 96)
        self.pos = nn.Embedding(MAX_TOKENS, 96)
        self.blocks = nn.ModuleList(Block() for _ in range(3))
        self.norm = nn.LayerNorm(96)
        self.out = nn.Linear(96, len(VOCAB))

    def forward(self, ids, readout):
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        mask = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        x = x[torch.arange(batch, device=ids.device), readout]
        return self.out(self.norm(x))


@torch.inference_mode()
def evaluate(model, encoded, device):
    model.eval()
    ids, readout, labels, choices = encoded
    pred4, gold4, full_matches = [], [], []
    for begin in range(0, len(ids), 128):
        end = begin + 128
        logits = model(ids[begin:end].to(device), readout[begin:end].to(device)).float()
        c = choices[begin:end].to(device)
        choice_index = logits.gather(1, c).argmax(dim=1, keepdim=True)
        selected = c.gather(1, choice_index).flatten()
        pred4.append(selected.cpu())
        gold4.append(labels[begin:end])
        full_matches.append((logits.argmax(dim=1) == labels[begin:end].to(device)).cpu())
    predictions = torch.cat(pred4).reshape(-1, 4)
    truth = torch.cat(gold4).reshape(-1, 4)
    matched = predictions == truth
    per_query = matched.float().mean(dim=0).tolist()
    duplicate = ((predictions[:, 0] == predictions[:, 1]) |
                 (predictions[:, 2] == predictions[:, 3]))
    return {
        'accuracy4': matched.float().mean().item(),
        'C_query': matched.all(dim=1).float().mean().item(),
        'full_vocab': torch.cat(full_matches).float().mean().item(),
        'duplicate_rate': duplicate.float().mean().item(),
        'per_query': [round(x, 3) for x in per_query],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--mode', choices=('overfit', 'pilot'), default='overfit')
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; use --device cpu for a smoke test')

    groups = list(combinations(NAMES, 4))
    Random(10200).shuffle(groups)
    heldout_groups, training_groups = groups[:14], groups[14:]
    assert not set(heldout_groups) & set(training_groups)

    if args.mode == 'overfit':
        train = make_packets(training_groups, 16, 10201)
        train_data = encode(train)
        train_probe_data = train_data
        print('EXP-102 OVERFIT SMOKE TEST: 16 sentences x 4 questions = 64 labels', flush=True)
        print('Goal: establish the architecture can fit question-conditioned answers.', flush=True)
        print('This is not a generalization result. No protected data.', flush=True)
    else:
        train = make_packets(training_groups, 2048, 10201)
        train_texts = {s for s, _ in train}
        dev_id = make_packets(training_groups, 256, 10202, excluded=train_texts)
        dev_ood = make_packets(heldout_groups, 256, 10203)
        assert not train_texts.intersection(s for s, _ in dev_id + dev_ood)
        assert not {s for s, _ in dev_id}.intersection(s for s, _ in dev_ood)
        train_data, train_probe_data, id_data, ood_data = map(
            encode, (train, train[:256], dev_id, dev_ood))
        print('EXP-102 EXPLORATORY GENERALIZATION PILOT', flush=True)
        print(f'Train: {len(train)} sentences, 4 queries each; '
              f'in-domain dev: {len(dev_id)}; four-name-set-OOD dev: {len(dev_ood)}', flush=True)
        print('No exposure-arm contrast, no protected data, no positional offsets.', flush=True)

    torch.manual_seed(10204)
    model = Model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    sample_rng = torch.Generator(device='cpu').manual_seed(10205)
    inputs, readout, labels, _ = train_data

    for step in range(1, args.steps + 1):
        model.train()
        if args.mode == 'overfit':
            ix = torch.arange(len(inputs))
        else:
            ix = torch.randint(len(inputs), (64,), generator=sample_rng)
        logits = model(inputs[ix].to(args.device), readout[ix].to(args.device))
        loss = F.cross_entropy(logits, labels[ix].to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        interval = 50 if args.mode == 'overfit' else 100
        if step == 1 or step % interval == 0 or step == args.steps:
            train_score = evaluate(model, train_probe_data, args.device)
            report = (f'STEP {step:4d} loss={loss.item():.3f} '
                      f'TRAIN4={train_score["accuracy4"]:.3f} '
                      f'TRAIN_Cquery={train_score["C_query"]:.3f} '
                      f'TRAIN_dup={train_score["duplicate_rate"]:.3f}')
            if args.mode == 'pilot':
                id_score = evaluate(model, id_data, args.device)
                ood_score = evaluate(model, ood_data, args.device)
                report += (f' ID4={id_score["accuracy4"]:.3f} '
                           f'ID_Cquery={id_score["C_query"]:.3f} '
                           f'OOD4={ood_score["accuracy4"]:.3f} '
                           f'OOD_Cquery={ood_score["C_query"]:.3f} '
                           f'OOD_dup={ood_score["duplicate_rate"]:.3f}')
            print(report, flush=True)
            if step == args.steps:
                print('Final train probe:', train_score)
                if args.mode == 'pilot':
                    print('Final in-domain:', id_score)
                    print('Final heldout-groups:', ood_score)
    print('Exploratory only. No checkpoint or protected evaluation created.', flush=True)


if __name__ == '__main__':
    main()
