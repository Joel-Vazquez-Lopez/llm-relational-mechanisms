#!/usr/bin/env python3
"""EXP-102 v2 development-only learnability study.

Read the already fixed four-name-set split. NEVER enumerate reserved name sets,
NEVER render the withheld L+I syntax, and NEVER open protected evaluation.

Two batching policies, same 64 labels/update, exactly 16 per query type:
  quartet: 16 randomly sampled sentences, all four questions each;
  balanced_unpaired: 16 independent sentence draws for each query type.

This is a FEASIBILITY pilot on familiar syntax, not a final protected study.
The 16 vs up-to-64 distinct sources/update is a known, unisolated difference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations, product
import json
import os
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

from exp102_syntax_preflight import (
    Event, NAMES, VERBS, VOICE, TRAIN, TOKEN_RE,
    render, parse_text, questions, gold,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
SPLIT = ROOT / 'reports/exp102/syntax_name_set_split_v2.json'
OUT = ROOT / 'reports/exp102/v2_familiar_pilot'
CKPT = ROOT / 'results/exp102/v2_familiar_pilot'
EXPECTED_SPLIT_HASH = '2f9ba0c6f44c67094f9fa2786853ade1c8c662fd0afd74295af98384b17dd287'
SEEDS = (10401, 10403, 10409, 10413, 10421, 10427, 10429, 10433)
METHODS = ('quartet', 'balanced_unpaired')
STEPS = 1600
MILESTONES = (400, 800, 1200, 1600)
# 9 familiar modifier cells x 4 voice-pair cells. Fixed per-cell counts.
TRAIN_PER_CELL_VOICE = 64   # 9 * 4 * 64 = 2304 distinct training sentences
DEV_PER_CELL_VOICE = 8     # 9 * 4 * 8 = 288 distinct dev sentences
DATA_SEED = 102311
MAX_TOKENS = 48
WIDTH, HEADS, LAYERS = 96, 4, 3
BATCH_LABELS = 64

VOCAB_WORDS = tuple(dict.fromkeys(
    ('<pad>', '<bos>', '<sep>', '<answer>') + NAMES + VERBS +
    ('was', 'by', 'By', 'near', 'beside', 'gate', "'s", 'In', 'the',
     'first', 'second', 'event', 'who', 'agent', 'patient', ';', ',', '.', '?')
))
VOCAB = {x: i for i, x in enumerate(VOCAB_WORDS)}
Q = questions()
assert len(Q) == 4


def stable_hash(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {filename: file_hash(SCRIPT.parent / filename) for filename in (
        'exp102_syntax_preflight.py', 'exp102_split_preflight_v2.py',
        'exp102_split_integration_audit.py',
    )}


def split_manifest():
    m = json.loads(SPLIT.read_text())
    assert m['partition_sha256'] == stable_hash(m['partitions']) == EXPECTED_SPLIT_HASH
    assert m['split_unit'] == 'unordered_four_name_set'
    p = m['partitions']
    assert len(p['train_name_sets']) == 50
    assert len(p['development_name_sets']) == 10
    assert len(p['reserved_name_sets']) == 10
    # Only read reserved metadata for disjointness; never render their examples.
    for k in ('train_name_sets', 'development_name_sets', 'reserved_name_sets'):
        assert all(len(a) == 4 and len(set(a)) == 4 for a in p[k])
    sets = [set(map(tuple, p[k])) for k in
            ('train_name_sets', 'development_name_sets', 'reserved_name_sets')]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert all('LI' not in pair for pair in TRAIN)
    return m


def make_pool(name_sets):
    """Use EXACT generator definitions from successful split integration audit.

    Each record: (text, four gold names, modifier cell, two voice values).
    No protected syntax / reserved name-set generation.
    """
    grouped = defaultdict(list)
    seen = set()
    for ns in sorted(name_sets):
        names = tuple(ns)
        for a, b, c, d in permutations(names, 4):
            v0 = VERBS[(NAMES.index(a) + NAMES.index(c)) % len(VERBS)]
            v1 = VERBS[(NAMES.index(b) + NAMES.index(d) + 1) % len(VERBS)]
            if v1 == v0:
                v1 = VERBS[(VERBS.index(v0) + 1) % len(VERBS)]
            for voice0, voice1 in product(VOICE, repeat=2):
                events = (
                    Event(a, b, v0, voice0, c, d, 0),
                    Event(c, d, v1, voice1, a, b, 1),
                )
                answer = gold(events)
                assert len(set(answer)) == 4
                for mods in TRAIN:
                    text = render(events, mods)
                    assert text not in seen, 'Repeated source'
                    seen.add(text)
                    assert tuple(x for pair in parse_text(text) for x in pair) == answer
                    grouped[(mods, (voice0, voice1))].append((text, answer, mods, (voice0, voice1)))
    assert len(grouped) == len(TRAIN) * 4
    return grouped


def select_stratified(grouped, count, seed):
    rng = Random(seed)
    output = []
    for key in sorted(grouped):
        entries = grouped[key]
        assert len(entries) >= count
        output.extend(rng.sample(entries, count))
    rng.shuffle(output)
    assert len(output) == len(TRAIN) * 4 * count
    assert len({row[0] for row in output}) == len(output)
    return output


def datasets():
    m = split_manifest()
    train_sets = m['partitions']['train_name_sets']
    dev_sets = m['partitions']['development_name_sets']
    train = select_stratified(make_pool(train_sets), TRAIN_PER_CELL_VOICE, DATA_SEED)
    dev = select_stratified(make_pool(dev_sets), DEV_PER_CELL_VOICE, DATA_SEED + 1)
    assert not {row[0] for row in train} & {row[0] for row in dev}
    assert not {tuple(sorted(row[1])) for row in train} & {tuple(sorted(row[1])) for row in dev}
    def words(group):
        return {x for text,_,_,_ in group for x in TOKEN_RE.findall(text)}
    assert words(dev) <= words(train)
    for group in (train, dev):
        for text, answer, mods, voice in group:
            assert all(mod in ('bare', 'L', 'I') for mod in mods)
            assert len(set(answer)) == 4
            assert tuple(x for pair in parse_text(text) for x in pair) == answer
            assert max(1 + len(TOKEN_RE.findall(text)) + 1 + len(TOKEN_RE.findall(q)) + 1 for q in Q) <= MAX_TOKENS
    digest = stable_hash({'train': [(x[0], x[1]) for x in train],
                          'dev': [(x[0], x[1]) for x in dev]})
    return train, dev, digest


def encode(rows):
    # Each sentence contributes four contiguous queries in canonical order.
    sequences, y, choices, lengths = [], [], [], []
    for text, answer, _, _ in rows:
        four_names = tuple(sorted(answer))
        cand = [VOCAB[n] for n in four_names]
        for query, gold_name in zip(Q, answer):
            tokens = ['<bos>'] + TOKEN_RE.findall(text) + ['<sep>'] + TOKEN_RE.findall(query) + ['<answer>']
            assert all(t in VOCAB for t in tokens), set(tokens) - set(VOCAB)
            assert len(tokens) <= MAX_TOKENS
            sequences.append([VOCAB[t] for t in tokens])
            y.append(VOCAB[gold_name]); choices.append(cand); lengths.append(len(tokens))
    width = max(lengths)
    x = torch.full((len(sequences), width), VOCAB['<pad>'], dtype=torch.long)
    for i, tokens in enumerate(sequences):
        x[i, :len(tokens)] = torch.tensor(tokens)
    readout = torch.tensor([n-1 for n in lengths])
    gold_ids = torch.tensor(y)
    cand_ids = torch.tensor(choices)
    assert len(rows) * 4 == len(gold_ids)
    assert (gold_ids[:, None] == cand_ids).any(1).all()
    return x, readout, gold_ids, cand_ids


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, WIDTH*4), nn.GELU(), nn.Linear(WIDTH*4, WIDTH))
    def forward(self, x, mask):
        z = self.n1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), WIDTH)
        self.pos = nn.Embedding(MAX_TOKENS, WIDTH)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, len(VOCAB))
    def forward(self, ids, readout):
        batch, n = ids.shape
        assert n <= MAX_TOKENS
        positions = torch.arange(n, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        mask = torch.ones(n, n, device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(self.norm(x[torch.arange(batch, device=ids.device), readout]))


def batch_indices(g, method, num_sentences):
    q = torch.arange(4).repeat(16, 1)
    if method == 'quartet':
        source = torch.randint(num_sentences, (16, 1), generator=g)
    elif method == 'balanced_unpaired':
        source = torch.randint(num_sentences, (16, 4), generator=g)
    else:
        raise ValueError(method)
    ix = (4 * source + q).reshape(-1)
    assert len(ix) == BATCH_LABELS
    assert torch.bincount(ix % 4, minlength=4).tolist() == [16] * 4
    return ix


@torch.inference_mode()
def score(model, encoded, rows, device):
    model.eval()
    x, readout, y, candidates = encoded
    picks = []
    full_vocab_correct = 0
    for start in range(0, len(x), 128):
        end = start + 128
        logits = model(x[start:end].to(device), readout[start:end].to(device)).float()
        c = candidates[start:end].to(device)
        best = c.gather(1, logits.gather(1, c).argmax(1, keepdim=True)).flatten()
        picks.append(best.cpu())
        full_vocab_correct += int((logits.argmax(1) == y[start:end].to(device)).sum().item())
    predicted = torch.cat(picks).view(-1, 4)
    truth = y.view(-1, 4)
    matched = predicted == truth
    q_correct = matched.all(dim=1)
    copy0 = predicted[:,0] == predicted[:,1]
    copy1 = predicted[:,2] == predicted[:,3]
    event_hit0 = (predicted[:,:2] == truth[:,0:1]) | (predicted[:,:2] == truth[:,1:2])
    event_hit1 = (predicted[:,2:] == truth[:,2:3]) | (predicted[:,2:] == truth[:,3:4])
    cell = defaultdict(list)
    for i, row in enumerate(rows):
        cell['/'.join(row[2])].append(i)
    cell_scores = {k: float(q_correct[v].float().mean()) for k,v in sorted(cell.items())}
    n = len(rows)
    return {
        'sentences': n,
        'accuracy4': float(matched.float().mean()),
        'C_query': float(q_correct.float().mean()),
        'role_copy_first': float(copy0.float().mean()),
        'role_copy_second': float(copy1.float().mean()),
        'correct_event_identity_first': float(event_hit0.float().mean()),
        'correct_event_identity_second': float(event_hit1.float().mean()),
        'both_correct_first': float(matched[:,:2].all(1).float().mean()),
        'both_correct_second': float(matched[:,2:].all(1).float().mean()),
        'full_vocab_accuracy': full_vocab_correct/(4*n),
        'per_cell_C_query': cell_scores,
    }


def protocol(digest):
    return {
        'study': 'EXP-102-v2', 'stage': 'FAMILIAR_SYNTAX_DEVELOPMENT_FEASIBILITY',
        'not_protected_and_not_confirmatory': True,
        'primary_dev_endpoint': 'C_query on 288 familiar-syntax development sentences at step 1600',
        'policy_contrast': 'quartet versus balanced_unpaired',
        'known_source_diversity_confound': True,
        'train_sentences': len(TRAIN)*4*TRAIN_PER_CELL_VOICE,
        'dev_sentences': len(TRAIN)*4*DEV_PER_CELL_VOICE,
        'train_selection_seed': DATA_SEED,
        'train_per_modifier_voice_cell': TRAIN_PER_CELL_VOICE,
        'dev_per_modifier_voice_cell': DEV_PER_CELL_VOICE,
        'split_partition_sha256': EXPECTED_SPLIT_HASH,
        'data_digest': digest,
        'fixed_seeds': list(SEEDS), 'methods': list(METHODS),
        'steps': STEPS, 'milestones': list(MILESTONES),
        'model': {'layers':LAYERS,'width':WIDTH,'heads':HEADS,'max_tokens':MAX_TOKENS},
        'optimizer': 'AdamW(lr=0.002,weight_decay=0.01)',
        'labels_per_update': BATCH_LABELS,
        'per_query_labels': [16,16,16,16],
        'criterion': 'cross_entropy_full_vocabulary',
        'metric': 'four-name candidate-restricted answers',
        'checkpoint_selection': 'last step only, no early stopping',
        'grammar_evaluation': 'familiar-only; LI and reserved locked',
        'runner_sha256': file_hash(SCRIPT), 'dependency_hashes': source_hashes(),
    }


def locked_protocol(now):
    path = OUT / 'protocol.json'
    if not path.exists():
        raise FileNotFoundError('Run --phase prepare first')
    frozen = json.loads(path.read_text())
    if frozen != now:
        raise PermissionError('Runner, split, grammar or data changed after prepare; refusing to run')
    return frozen


def prepare(now, train, dev):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        locked_protocol(now)
        print('EXISTING PROTOCOL: verified unchanged')
    else:
        path.write_text(json.dumps(now, indent=2, sort_keys=True) + '\n')
        print('PREPARED FAMILIAR-SYNTAX DEVELOPMENT PILOT:', path)
    print(f'Train={len(train)} sentences, development={len(dev)}; LI=0, reserved=0')
    print('Nine familiar modifier cells x four voice pairs. Full question balancing.')
    print('Source hashes and fixed name-set split verified. No training performed.')


def smoke(train_data, dev_data, train, dev, device):
    print('EXP-102 v2 ONE-STEP SMOKE TEST; NO ARTIFACTS',flush=True)
    x, readout, y, _ = train_data
    for method in METHODS:
        torch.manual_seed(SEEDS[0]); model=Model().to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        sampler=torch.Generator(device='cpu').manual_seed(SEEDS[0]+100)
        ix=batch_indices(sampler,method,len(train))
        assert len(set((ix%4).tolist()))==4
        model.train()
        pred=model(x[ix].to(device),readout[ix].to(device))
        loss=F.cross_entropy(pred,y[ix].to(device))
        assert torch.isfinite(loss).item()
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        measured=score(model, dev_data, dev, device)
        print(f'  {method}: one-step loss={loss.item():.4f} '
              f'dev_accuracy4={measured["accuracy4"]:.4f} '
              f'dev_C_query={measured["C_query"]:.4f}',flush=True)
    print('PASS: no checkpoints or protected evaluation created',flush=True)


def run_one(seed, method, train_data, dev_data, probe_data, dev, probe, digest, device):
    result_path = OUT / f'seed{seed}_{method}.json'
    checkpoint = CKPT / f'seed{seed}_{method}.pt'
    if result_path.exists() and checkpoint.exists():
        print(f'ALREADY COMPLETE: seed={seed} method={method} (not overwritten)',flush=True)
        return
    if result_path.exists() or checkpoint.exists():
        raise FileExistsError(f'Incomplete artifact pair for {seed}/{method}: inspect manually')
    torch.manual_seed(seed)
    model=Model().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    g=torch.Generator(device='cpu').manual_seed(seed+100)
    x, readout, y, _ = train_data
    rows=[]
    print(f'\nRUN seed={seed} method={method}',flush=True)
    for step in range(1, STEPS+1):
        ix=batch_indices(g,method,len(x)//4)
        model.train()
        logits=model(x[ix].to(device),readout[ix].to(device))
        loss=F.cross_entropy(logits,y[ix].to(device))
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        if step in MILESTONES:
            dev_metrics=score(model,dev_data,dev,device)
            train_metrics=score(model,probe_data,probe,device)
            rows.append({'step':step,'dev':dev_metrics,'train_probe':train_metrics})
            print(f'STEP {step:4d} loss={loss.item():.4f} '
                  f'TRAIN C_query={train_metrics["C_query"]:.3f} '
                  f'DEV C_query={dev_metrics["C_query"]:.3f} '
                  f'DEV4={dev_metrics["accuracy4"]:.3f} '
                  f'copy=({dev_metrics["role_copy_first"]:.3f},'
                  f'{dev_metrics["role_copy_second"]:.3f})',flush=True)
    obj={'study':'EXP-102-v2','stage':'FAMILIAR_SYNTAX_DEVELOPMENT_FEASIBILITY',
         'seed':seed,'method':method,'steps':STEPS,'data_digest':digest,
         'protocol_sha256':stable_hash(protocol(digest)),'device':device,
         'records':rows,'protected_accessed':False}
    CKPT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    temp_checkpoint=checkpoint.with_suffix('.pt.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),
                'sampler_state':g.get_state(),'metadata':obj},temp_checkpoint)
    os.replace(temp_checkpoint,checkpoint)
    temp_json=result_path.with_suffix('.json.tmp')
    temp_json.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    os.replace(temp_json,result_path)
    print('SAVED',result_path,flush=True)
    print('SAVED',checkpoint,flush=True)


def summarize():
    complete=[]
    print('\nFAMILIAR DEV FINAL C_query (not protected)')
    for seed in SEEDS:
        paths=[OUT / f'seed{seed}_{method}.json' for method in METHODS]
        if all(x.exists() for x in paths):
            vals=[]
            for path in paths:
                obj=json.loads(path.read_text())
                assert obj['seed']==seed and obj['steps']==STEPS
                vals.append(obj['records'][-1]['dev']['C_query'])
            complete.append((seed,vals))
            print(f'{seed}: quartet={vals[0]:.3f} balanced_unpaired={vals[1]:.3f}')
    print(f'{len(complete)}/{len(SEEDS)} matched seed pairs complete')
    if len(complete)==len(SEEDS):
        diffs=[v[0]-v[1] for _,v in complete]
        print(f'Mean paired familiar-dev advantage: {sum(diffs)/len(diffs):+.4f}; '
              f'positive={sum(x>0 for x in diffs)}/8 (DESCRIPTIVE ONLY)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    ap.add_argument('--seeds',type=int,nargs='*')
    ap.add_argument('--device',choices=('mps','cpu'),default='mps')
    args=ap.parse_args()
    if args.seeds is not None and (not args.seeds or len(set(args.seeds)) != len(args.seeds)
                                   or any(s not in SEEDS for s in args.seeds)):
        ap.error('Choose distinct seeds from the fixed list')
    if args.phase in ('smoke','run') and args.device=='mps' and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    train,dev,digest=datasets()
    fixed=protocol(digest)
    if args.phase=='prepare':
        prepare(fixed,train,dev)
        return
    locked_protocol(fixed)
    if args.phase=='summary':
        summarize();return
    train_data=encode(train)
    dev_data=encode(dev)
    if args.phase=='smoke':
        smoke(train_data,dev_data,train,dev,args.device)
        return
    probe=train[:144]
    probe_data=encode(probe)
    for seed in (args.seeds if args.seeds is not None else SEEDS):
        for method in METHODS:
            run_one(seed,method,train_data,dev_data,probe_data,dev,probe,digest,args.device)
    summarize()
    print('DEVELOPMENT FEASIBILITY ONLY: no L+I, no reserved-set evaluation',flush=True)


if __name__=='__main__':
    main()
