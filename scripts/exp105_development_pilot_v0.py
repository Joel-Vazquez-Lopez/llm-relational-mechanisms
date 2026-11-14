#!/usr/bin/env python3
"""EXP-105 PA-vs-AP disposable, development-only training pilot (UNFROZEN).

Consumes the exact deterministic candidate source schedule from the v3 audit.
No reserved examples, no protected evaluation, no L+I transfer scoring, and
never uses the eight proposed prospective/confirmatory seeds. This pilot is
for establishing basic learnability and debugging the experimental pipeline.

From repo root (after v3 --write):
    python3 scripts/exp105_development_pilot_v0.py --phase prepare
    python3 scripts/exp105_development_pilot_v0.py --phase smoke --device mps
    python3 -u scripts/exp105_development_pilot_v0.py --phase run --steps 450 --device mps
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

import exp105_generator_preflight as v0
import exp105_position_control_preflight_v1 as v1
import exp105_exposure_matching_preflight_v2 as v2
import exp105_training_schedule_preflight_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'reports/exp105/development_pilot_v0'
CKPT = ROOT / 'results/exp105/development_pilot_v0'
PRE_AUDIT = ROOT / 'reports/exp105/preflight/training_schedule_preflight_v3.json'
PILOT_SEED = 10801  # NEVER use prospective/confirmatory 107xx seeds in pilot.
MAX_TOKENS = 48
WIDTH, HEADS, LAYERS = 96, 4, 3
BATCH_LABELS = 64
DEV_PER_CELL = 8  # 3 voice cells x 9 modifier cells x 8 = 216 sources.
DEV_SAMPLE_SEED = 105450
LR, WEIGHT_DECAY = 0.002, 0.01
# Fixed development summaries only; no post-hoc checkpoint selection.
MILESTONES = (45, 90, 180, 270, 360, 450, 800, 1200, 1600, 2400, 3200)


def hash_file(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def stable_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_files():
    return (SCRIPT, Path(v0.__file__), Path(v1.__file__), Path(v2.__file__), Path(v3.__file__))


def read_and_verify_audit():
    if not PRE_AUDIT.is_file():
        raise FileNotFoundError('First run: python3 scripts/exp105_training_schedule_preflight_v3.py --write')
    audit = json.loads(PRE_AUDIT.read_text())
    assert audit['phase'] == 'UNFROZEN_SCHEDULE_PREFLIGHT_V3'
    assert audit['design_only'] and audit['training_performed'] is False
    assert audit['protected_examples_generated'] is False
    assert audit['candidate_optimizer_updates'] == v3.STEPS == 3200
    assert audit['labels_per_update'] == BATCH_LABELS
    assert tuple(audit['paired_seeds']) == v3.SEEDS
    assert not (PILOT_SEED in v3.SEEDS)
    dependencies = audit['dependency_sha256']
    for path in source_files()[1:]:
        assert hash_file(path) == dependencies[path.name], f'Changed preflight dependency: {path.name}'
    return audit


def familiar_dev_sources():
    train_sets, dev_sets, _ = v0.partitions()
    # Generate *only* familiar development grammar. Never render L+I here.
    samples = defaultdict(list)
    train_participants = {tuple(x) for x in train_sets}
    assert not train_participants & set(dev_sets)
    for _, core_id, events in v0.cores(dev_sets, 'dev'):
        for voice in v0.VOICES:
            for mods in v0.TRAIN_MODS:
                s = v0.make_source(events, voice, mods, 'dev_familiar', core_id)
                samples[(voice, mods)].append(s)
    rng = Random(DEV_SAMPLE_SEED)
    chosen = []
    for key in sorted(samples):
        cell = samples[key]
        assert len(cell) >= DEV_PER_CELL
        chosen.extend(rng.sample(cell, DEV_PER_CELL))
    # Canonical order: per-cell then randomized within cell; fixed by seed.
    assert len(chosen) == len(v0.VOICES) * len(v0.TRAIN_MODS) * DEV_PER_CELL
    assert len({s.text for s in chosen}) == len(chosen)
    assert all('LI' not in s.mods for s in chosen)
    return chosen


def vocab_for(common, paired, dev):
    specials = ('<pad>', '<bos>', '<sep>', '<answer>')
    train_words = set(v0.tokens(' '.join(s.text for s in common)))
    for pair in paired:
        train_words.update(v0.tokens(pair[0].text))
        train_words.update(v0.tokens(pair[1].text))
    train_words.update(w for q in v0.Q for w in v0.tokens(q))
    assert set(v0.NAMES) <= train_words
    assert all(set(v0.tokens(s.text)).issubset(train_words) for s in dev)
    assert all(set(v0.tokens(q)).issubset(train_words) for q in v0.Q)
    words = specials + tuple(sorted(train_words - set(specials)))
    return {token: ix for ix, token in enumerate(words)}


def encode_sources(sources, vocab, check=True):
    """Rows of four labels per source, in original question order."""
    seqs, labels, choices, lens = [], [], [], []
    for s in sources:
        if check:
            assert v0.parse_independent(v1.unframe(s.text)) == s.gold
            assert len(set(s.gold)) == 4 and all('LI' not in m for m in s.mods)
        candidate_names = tuple(sorted(s.gold))
        cand = [vocab[name] for name in candidate_names]
        for query, answer in zip(v0.Q, s.gold):
            tokens = ['<bos>'] + v0.tokens(s.text) + ['<sep>'] + v0.tokens(query) + ['<answer>']
            if not set(tokens) <= set(vocab):
                raise AssertionError('Missing train-vocabulary token(s): '+str(set(tokens) - set(vocab)))
            if len(tokens) > MAX_TOKENS:
                raise AssertionError(f'Input length {len(tokens)} > {MAX_TOKENS}')
            seqs.append([vocab[t] for t in tokens])
            labels.append(vocab[answer])
            choices.append(cand)
            lens.append(len(tokens))
    assert len(seqs) == len(sources)*4
    tensor = torch.full((len(seqs), max(lens)), vocab['<pad>'], dtype=torch.long)
    for i, tokens in enumerate(seqs):
        tensor[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return tensor, torch.tensor([n-1 for n in lens], dtype=torch.long), torch.tensor(labels), torch.tensor(choices)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, dropout=0.0, batch_first=True)
        self.n2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, WIDTH*4), nn.GELU(), nn.Linear(WIDTH*4, WIDTH))

    def forward(self, x, mask):
        z = self.n1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, WIDTH)
        self.pos = nn.Embedding(MAX_TOKENS, WIDTH)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, vocab_size)

    def forward(self, ids, readout):
        bs, n = ids.shape
        assert n <= MAX_TOKENS
        positions = torch.arange(n, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        mask = torch.ones(n,n, dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(self.norm(x[torch.arange(bs,device=ids.device),readout]))


@torch.inference_mode()
def score(model, encoded, rows, device, vocab_size, chunk_size=128):
    model.eval()
    x, readout, labels, candidates = encoded
    preds=[]
    full_correct=0
    for st in range(0,len(labels),chunk_size):
        en=st+chunk_size
        logits=model(x[st:en].to(device),readout[st:en].to(device)).float()
        c=candidates[st:en].to(device)
        best=c.gather(1,logits.gather(1,c).argmax(dim=1,keepdim=True)).flatten()
        preds.append(best.cpu())
        full_correct+=int((logits.argmax(1)==labels[st:en].to(device)).sum().item())
    predicted=torch.cat(preds).view(-1,4)
    gold=labels.view(-1,4)
    correct=(predicted==gold)
    assert len(rows)==len(predicted)
    event0=((predicted[:,:2]==gold[:,0:1])|(predicted[:,:2]==gold[:,1:2]))
    event1=((predicted[:,2:]==gold[:,2:3])|(predicted[:,2:]==gold[:,3:4]))
    hit=torch.cat((event0,event1),dim=1)
    wrong_event=int((~hit).sum().item())
    wrong_role=int((hit & ~correct).sum().item())
    assert wrong_role+wrong_event+int(correct.sum().item())==len(rows)*4
    by_voice=defaultdict(list)
    for ix, row in enumerate(rows):
        by_voice[row.voice].append(ix)
    return {
        'sentences':len(rows), 'C_query':float(correct.all(dim=1).float().mean().item()),
        'accuracy4':float(correct.float().mean().item()),
        'same_answer_rate':float((
            (predicted[:,0]==predicted[:,1]).float().mean() +
            (predicted[:,2]==predicted[:,3]).float().mean()).item()/2),
        'wrong_event_rate':wrong_event/(len(rows)*4),
        'wrong_role_rate':wrong_role/(len(rows)*4),
        'per_voice_C_query':{v:float(correct[ix].all(dim=1).float().mean()) for v,ix in sorted(by_voice.items())},
        'full_vocab_accuracy':full_correct/(len(rows)*4),
    }


def make_protocol(audit, common, paired, dev, vocab):
    assert audit['partition_digest'] == v0.partitions()[2]['partition_digest']
    info={
        'study':'EXP-105','stage':'UNFROZEN_DEVELOPMENT_LEARNABILITY_PILOT_V0',
        'not_confirmatory':True,'no_reserved_or_protected_evaluation':True,
        'no_LI_evaluation':True,'no_confirmatory_seeds':True,
        'pilot_seed':PILOT_SEED,'prospective_seeds_untouched':list(v3.SEEDS),
        'candidate_schedule':'imports v3.iter_schedule unchanged',
        'candidate_schedule_sha256_reference':audit['schedule_hashes_by_seed'],
        'preflight_v3_sha256':hash_file(PRE_AUDIT),
        'code_hashes':{p.name:hash_file(p) for p in source_files()},
        'split_digest':audit['partition_digest'],
        'shared_base_count':len(common),'paired_dose_count':len(paired),
        'dev_familiar_sources':len(dev),'dev_sample_seed':DEV_SAMPLE_SEED,
        'dev_digest':stable_hash([(s.text,s.gold) for s in dev]),
        'vocabulary_size':len(vocab),'vocabulary_sha256':stable_hash(sorted(vocab.items())),
        'model':{'width':WIDTH,'heads':HEADS,'layers':LAYERS,'positions':MAX_TOKENS,
                 'dropout':0.0,'causal':True,'parameterization':'EXP102-v2 compatible'},
        'optimizer':{'name':'AdamW','lr':LR,'weight_decay':WEIGHT_DECAY},
        'loss':'full-vocabulary cross entropy','evaluation':'four-name candidate-restricted plus full-vocabulary',
        'train_policy':'7 AA shared + 7 PA shared + 2 PA/AP dose; 4 queries each (64 labels)',
        'milestones':list(MILESTONES),'checkpoint_selection':'last step only, no early stopping',
        'pilot_budget_options':[45,90,180,270,360,450,800,1200,1600,2400,3200],
        'known_confounds':['token position distribution differs by design',
                          'source-length distribution matches per full 45-step window, not each update',
                          'one grammar family; not a new independent replication',
                          'PA/AP dose changes more than just an abstract event feature (word order/context)'],
    }
    return info


def ensure_protocol(proto, create):
    path=OUT/'protocol.json'
    if create:
        OUT.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(proto,indent=2,sort_keys=True)+'\n')
            print('SAVED',path)
    if not path.exists() or json.loads(path.read_text())!=proto:
        raise RuntimeError('Pilot protocol or dependencies changed. Do not overwrite; inspect source. Run --phase prepare only after review.')


def get_device(requested):
    if requested=='cpu':return torch.device('cpu')
    if requested=='mps':
        if not torch.backends.mps.is_available():raise ValueError('MPS is not available on this machine')
        return torch.device('mps')
    if requested=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA is not available')
        return torch.device('cuda')
    if torch.cuda.is_available():return torch.device('cuda')
    if torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')


def make_models(seed, device, vocab_size):
    torch.manual_seed(seed)
    reference = Model(vocab_size)
    # Pair identical initialization; keep independent optimizers/states.
    familiar=reference.to(device)
    target=deepcopy(reference).to(device)
    assert all(torch.equal(a,b) for a,b in zip(familiar.parameters(),target.parameters()))
    optim_f=torch.optim.AdamW(familiar.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    optim_t=torch.optim.AdamW(target.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    return (familiar,optim_f),(target,optim_t)


def one_update(model, optim, data, device):
    model.train()
    x,readout,y,_=data
    logits=model(x.to(device),readout.to(device))
    loss=F.cross_entropy(logits,y.to(device))
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    optim.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optim.step()
    return float(loss.item())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=('prepare','smoke','run'))
    p.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    p.add_argument('--steps',type=int,default=450,
                   choices=(45,90,180,270,360,450,800,1200,1600,2400,3200))
    args=p.parse_args()
    audit=read_and_verify_audit()
    common,paired,bpool,dpool,part_digest=v3.pools()
    assert audit['partition_digest']==part_digest
    dev=familiar_dev_sources()
    vocab=vocab_for(common,paired,dev)
    proto=make_protocol(audit,common,paired,dev,vocab)
    ensure_protocol(proto,create=args.phase=='prepare')
    print(f'PILOT: {len(common)} shared + {len(paired)} matched dose pairs, '
          f'{len(dev)} dev familiar sources, vocab={len(vocab)}; NO L+I scoring',flush=True)
    if args.phase=='prepare':
        print('PASS: Audited v3 generator dependencies match; read-only preparation except new pilot protocol')
        print('CONFIRMATORY SEEDS untouched; no training or protected evaluation')
        return
    device=get_device(args.device)
    seed=PILOT_SEED
    if args.phase=='smoke':
        (f,of),(t,ot)=make_models(seed,device,len(vocab))
        for step, b_ix, d_ix,_,_ in v3.iter_schedule(seed,common,paired,bpool,dpool):
            fr=[common[i] for i in b_ix]+[paired[i][0] for i in d_ix]
            tr=[common[i] for i in b_ix]+[paired[i][1] for i in d_ix]
            ef=encode_sources(fr,vocab)
            et=encode_sources(tr,vocab)
            assert torch.equal(ef[2],et[2]) and torch.equal(ef[3],et[3])
            lf=one_update(f,of,ef,device)
            lt=one_update(t,ot,et,device)
            print(f'SMOKE step={step} loss_familiar={lf:.4f} loss_target={lt:.4f}',flush=True)
            if step==2:break
        # Evaluate only a few familiar-development sources for scoring-shape verification.
        d=dev[:9]
        e=encode_sources(d,vocab)
        mf=score(f,e,d,device,len(vocab))
        mt=score(t,e,d,device,len(vocab))
        assert 0<=mf['C_query']<=1 and 0<=mt['C_query']<=1
        print('SMOKE PASS: 2 paired updates, gradients, matched gold and development scoring',flush=True)
        print('NO checkpoints, no L+I transfer, no confirmatory seeds, no protected examples')
        return
    assert args.phase=='run'
    budget=args.steps
    report_path=OUT/f'pilot_seed{seed}_step{budget}.json'
    checkpoint_path=CKPT/f'pilot_seed{seed}_step{budget}.pt'
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError('Pilot run artifact exists; cannot overwrite or auto-resume')
    dev_enc=encode_sources(dev,vocab)
    (f,of),(t,ot)=make_models(seed,device,len(vocab))
    history=[]
    schedule_digest=sha256()
    print('RUN: disposable DEVELOPMENT ONLY, no L+I evaluation, seed 10801 reserved for pilot',flush=True)
    for step, b_ix, d_ix,_,_ in v3.iter_schedule(seed,common,paired,bpool,dpool):
        source_f=[common[i] for i in b_ix]+[paired[i][0] for i in d_ix]
        source_t=[common[i] for i in b_ix]+[paired[i][1] for i in d_ix]
        assert all(sf.gold==st.gold for sf,st in zip(source_f,source_t))
        e_f=encode_sources(source_f,vocab,check=False)
        e_t=encode_sources(source_t,vocab,check=False)
        assert torch.equal(e_f[2],e_t[2]) and torch.equal(e_f[3],e_t[3])
        schedule_digest.update((json.dumps({'step':step,'base':b_ix,'dose':d_ix},separators=(',',':'))+'\n').encode())
        loss_f=one_update(f,of,e_f,device)
        loss_t=one_update(t,ot,e_t,device)
        if step in MILESTONES:
            score_f=score(f,dev_enc,dev,device,len(vocab))
            score_t=score(t,dev_enc,dev,device,len(vocab))
            history.append({'step':step,'familiar':score_f,'target':score_t})
            print(f'STEP {step:4d} lossF={loss_f:.4f} lossT={loss_t:.4f} '
                  f'F C_query={score_f["C_query"]:.3f} T C_query={score_t["C_query"]:.3f} '
                  f'F AP={score_f["per_voice_C_query"]["AP"]:.3f} '
                  f'T AP={score_t["per_voice_C_query"]["AP"]:.3f}',flush=True)
        if step == budget:
            break
    assert history and history[-1]['step']==budget
    rec={'study':'EXP-105','phase':'DISPOSABLE_DEVELOPMENT_PILOT',
         'not_confirmatory':True,'protected_evaluated':False,'LI_evaluated':False,
         'seed':seed,'steps':budget,'candidate_schedule_sha256_prefix':schedule_digest.hexdigest(),
         'protocol_sha256':stable_hash(proto),'history':history,
         'device':str(device),'note':'Do not use this pilot seed for future confirmation'}
    OUT.mkdir(parents=True,exist_ok=True)
    CKPT.mkdir(parents=True,exist_ok=True)
    # Save checkpoint as an indivisible paired artifact; do not overwrite existing.
    torch.save({'familiar':f.state_dict(),'target':t.state_dict(),
                'seed':seed,'steps':budget,'protocol_sha256':rec['protocol_sha256']},checkpoint_path)
    report_path.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
    print('SAVED',report_path,flush=True)
    print('SAVED',checkpoint_path,flush=True)
    print('PILOT COMPLETE: no protected/reserved/LI evaluation. Confirmatory seeds untouched.',flush=True)

if __name__=='__main__':main()
