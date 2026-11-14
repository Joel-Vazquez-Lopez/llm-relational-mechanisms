#!/usr/bin/env python3
"""EXP-104: source-question-matched quartet vs distributed supervision.

Uses exactly EXP-102 v2 familiar train/dev sources and its model/scoring code.
Each four-update window draws 64 distinct train sentences. Both policies see
EXACTLY the same 256 (source, question) training examples, one per pair:

  quartet:      16 distinct sources/update x their four questions
  distributed:  64 distinct sources/update x one question each, Latin-rotated

Every step = 64 labels, 16 per question type. Full-vocabulary cross-entropy.
Paired seeds share initial model weights and the *complete source-question
multiset* in every 4-update window, not the same order/optimizer trajectory.

Only familiar-syntax development; LI and reserved examples NEVER generated.
All completed EXP-102/103 files are untouched. No protected-evaluation phase.

Run from repository root:
  python3 scripts/exp104_source_matched_schedule.py --phase prepare
  python3 scripts/exp104_source_matched_schedule.py --phase smoke --device mps
  python3 -u scripts/exp104_source_matched_schedule.py --phase run --seeds 10601 --device mps
  python3 scripts/exp104_source_matched_schedule.py --phase summary
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random

import torch
import torch.nn.functional as F

import exp102_v2_familiar_training as v2

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = ROOT / 'reports/exp104/source_matched_dev'
CKPT = ROOT / 'results/exp104/source_matched_dev'
PROTOCOL = OUT / 'protocol.json'
SEEDS = (10601, 10603, 10609, 10613, 10621, 10627, 10629, 10633)
POLICIES = ('quartet', 'distributed')
STEPS = 3200
MILESTONES = (400, 800, 1200, 1600, 2000, 2400, 2800, 3200)
SOURCES_PER_WINDOW = 64
UPDATES_PER_WINDOW = 4
LABELS_PER_UPDATE = 64
QUERY_TYPES = 4
RNG_OFFSET = 104000


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def schedule_window(rng, n_sources):
    """Return four source groups and a balanced query offset per source.

    Each quartet step contains 16 different sentences. Each distributed step
    contains all 64. Each sentence's query assignment rotates over four steps.
    For each quartet group, offsets contain each query type exactly four times.
    """
    assert n_sources >= SOURCES_PER_WINDOW
    sources = rng.sample(range(n_sources), SOURCES_PER_WINDOW)
    groups = [sources[k*16:(k+1)*16] for k in range(4)]
    offsets = {}
    for group in groups:
        slots = list(range(4))*4
        rng.shuffle(slots)
        for src,q in zip(group,slots):
            assert src not in offsets
            offsets[src]=q
    assert len(offsets)==64
    return groups, sources, offsets


def step_pairs(window, policy, update):
    groups,sources,offsets=window
    assert 0 <= update < 4
    if policy == 'quartet':
        pairs = [(s,q) for s in groups[update] for q in range(4)]
    elif policy == 'distributed':
        pairs = [(s,(offsets[s]+update)%4) for s in sources]
    else:
        raise ValueError(policy)
    assert len(pairs)==LABELS_PER_UPDATE
    assert Counter(q for _,q in pairs)==Counter({q:16 for q in range(4)})
    assert len(set(pairs))==64
    return pairs


def validate_schedule(seed, n_sources, windows=STEPS//4):
    rng=Random(seed+RNG_OFFSET)
    for w in range(windows):
        window=schedule_window(rng,n_sources)
        expected=Counter((src,q) for src in window[1] for q in range(4))
        actual=[]
        for policy in POLICIES:
            allpairs=[pair for update in range(4) for pair in step_pairs(window,policy,update)]
            assert Counter(allpairs)==expected, (seed,w,policy)
            assert len(allpairs)==256 and len(set(allpairs))==256
            actual.append(Counter(allpairs))
        assert actual[0]==actual[1]
    return {'windows':windows,'labels':windows*256}


def enrich(metrics):
    m=dict(metrics)
    hit=(m['correct_event_identity_first']+m['correct_event_identity_second'])/2
    m['wrong_event_rate']=1-hit
    m['wrong_role_rate']=hit-m['accuracy4']
    m['same_answer_rate']=(m['role_copy_first']+m['role_copy_second'])/2
    m['per_event_both_correct']=(m['both_correct_first']+m['both_correct_second'])/2
    assert abs(1-(m['accuracy4']+m['wrong_event_rate']+m['wrong_role_rate'])) < 1e-5
    return m


def protocol(data_digest, ntrain, ndev):
    return {
        'study':'EXP-104-source-matched-schedule',
        'status':'EXPLORATORY_FAMILIAR_DEVELOPMENT_ONLY',
        'runner_sha256':file_hash(SCRIPT),
        'base_runner_sha256':file_hash(Path(v2.__file__)),
        'base_dependency_sha256':v2.source_hashes(),
        'split_partition_sha256':v2.EXPECTED_SPLIT_HASH,
        'data_sha256':data_digest,
        'train_sentences':ntrain,'development_sentences':ndev,
        'seeds':list(SEEDS),'policies':list(POLICIES),
        'steps':STEPS,'milestones':list(MILESTONES),
        'model':'Unchanged EXP-102-v2 3-layer, 4-head, width-96 Transformer',
        'optimization':'AdamW lr=0.002 weight_decay=0.01, full-vocabulary CE',
        'batch':'64 labels/update, 16/query; 64 distinct sources per 4-step window',
        'schedule':'Quartet 16 sources each x 4 query types / update; distributed 64 sources each x 1 question rotated / update',
        'controlled':'Exact identical 256 source-question examples per four-update window for both policies, paired initializations',
        'remaining_difference':'Within-update source diversity, order, gradient composition and optimizer trajectory still differ by design',
        'primary':'Difference in 3200-step dev C_query (quartet minus distributed), paired across eight fresh seeds',
        'secondary':['C_query trajectory','accuracy4','wrong_event_rate','wrong_role_rate',
                     'same_answer_rate','per_event_both_correct','per_cell_C_query'],
        'endpoint_selection':'Final step3200 only; no early stopping; no seed exclusion',
        'prior_discovery':'EXP-102-v2 original study and extension analyzed before this protocol; this is NOT an independent confirmatory test',
        'restricted':'Only familiar bare/L/I syntax, reserved identifiers read for separation only; no L+I or reserved examples generated',
        'gate':'Descriptive familiar competence: >=6/8 per policy with C_query>=0.90 and accuracy4>=0.95 on dev at final step; gate does not unlock evaluation',
    }


def write_once(path, obj, torch_file=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Already exists: {path}')
    tmp=path.with_name(path.name+'.tmp')
    if tmp.exists():
        raise FileExistsError(f'Stale partial artifact: {tmp}')
    try:
        if torch_file:
            torch.save(obj,tmp)
        else:
            tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def lock(now):
    if not PROTOCOL.exists():
        raise FileNotFoundError(f'No frozen protocol: run --phase prepare first: {PROTOCOL}')
    frozen=json.loads(PROTOCOL.read_text())
    if frozen!=now:
        raise PermissionError('EXP-104 protocol/source/data changed since prepare; refusing run')
    return frozen


def model_and_optimizer(seed,device):
    torch.manual_seed(seed)
    model=v2.Model().to(device)
    optim=torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    return model,optim


def train_batch(model,optim,train_tensors,examples,device):
    x,readout,y,_=train_tensors
    ids=[4*src+q for src,q in examples]
    model.train()
    pred=model(x[ids].to(device),readout[ids].to(device))
    loss=F.cross_entropy(pred,y[ids].to(device))
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    assert torch.isfinite(loss).item()
    return float(loss.detach())


def report_paths(seed,policy):
    suffix=f'seed{seed}_{policy}_step{STEPS}'
    return OUT/f'{suffix}.json', CKPT/f'{suffix}.pt'


def run_one(seed,policy,train_data,dev_data,train,dev,data_digest,now,device):
    report_path,checkpoint_path=report_paths(seed,policy)
    if report_path.exists() and checkpoint_path.exists():
        obj=json.loads(report_path.read_text())
        assert (obj['seed'],obj['policy'],obj['step'])==(seed,policy,STEPS)
        assert obj['protocol_sha256']==digest(now)
        print(f'ALREADY COMPLETE seed={seed} policy={policy}',flush=True)
        return
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f'Incomplete report/checkpoint pair: {report_path}, {checkpoint_path}')
    model,optimizer=model_and_optimizer(seed,device)
    rng=Random(seed+RNG_OFFSET)
    records=[]
    step=0
    print(f'RUN seed={seed} policy={policy} (source matched across 4-update windows)',flush=True)
    for window_idx in range(STEPS//4):
        window=schedule_window(rng,len(train))
        for update in range(4):
            loss=train_batch(model,optimizer,train_data,step_pairs(window,policy,update),device)
            step+=1
            if step in MILESTONES:
                dev_score=enrich(v2.score(model,dev_data,dev,device))
                train_probe=train[:144]
                probe_score=enrich(v2.score(model,v2.encode(train_probe),train_probe,device))
                records.append({'step':step,'dev':dev_score,'train_probe':probe_score,'last_batch_loss':loss})
                print(f'STEP {step:4d} loss={loss:.4f} DEV C_query={dev_score["C_query"]:.3f} '
                      f'DEV4={dev_score["accuracy4"]:.3f} copy={dev_score["same_answer_rate"]:.3f} '
                      f'wrong_event={dev_score["wrong_event_rate"]:.3f}',flush=True)
    assert step==STEPS and len(records)==len(MILESTONES)
    report={'study':'EXP-104-source-matched-schedule','stage':'FAMILIAR_DEV_ONLY',
            'seed':seed,'policy':policy,'step':step,'records':records,
            'device':device,'data_sha256':data_digest,'protocol_sha256':digest(now),
            'protected_accessed':False,'LI_cases_generated':False,
            'reserved_set_cases_generated':False,'evidence_status':'EXPLORATORY_ONLY'}
    write_once(checkpoint_path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                               'rng_state':rng.getstate(),'metadata':report},torch_file=True)
    write_once(report_path,report)
    print('SAVED',report_path,flush=True)
    print('SAVED',checkpoint_path,flush=True)


def show_summary():
    values=[]
    print('\nEXP-104 FAMILIAR DEV C_query (not protected)')
    for seed in SEEDS:
        ps=[report_paths(seed,p)[0] for p in POLICIES]
        if not all(p.exists() for p in ps):continue
        rs=[json.loads(p.read_text()) for p in ps]
        assert all(r['seed']==seed and r['step']==STEPS and r['policy']==p for r,p in zip(rs,POLICIES))
        scores=[r['records'][-1]['dev']['C_query'] for r in rs]
        diff=scores[0]-scores[1]
        values.append(diff)
        print(f'{seed}: quartet={scores[0]:.4f} distributed={scores[1]:.4f} paired_gap={diff:+.4f}')
    print(f'{len(values)}/{len(SEEDS)} paired seeds complete')
    if values:
        print(f'DESCRIPTIVE mean gap={sum(values)/len(values):+.4f}; '
              f'wins={sum(x>0 for x in values)} ties={sum(x==0 for x in values)} reversals={sum(x<0 for x in values)}')
    print('NO protected, LI or reserved examples opened. Fixed step3200 endpoint.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('prepare','smoke','run','summary'),required=True)
    ap.add_argument('--seeds',type=int,nargs='*')
    ap.add_argument('--device',choices=('mps','cpu'),default='mps')
    args=ap.parse_args()
    if args.seeds is not None and (not args.seeds or len(set(args.seeds))!=len(args.seeds)
                                    or any(s not in SEEDS for s in args.seeds)):
        ap.error(f'Use distinct seeds from {SEEDS}')
    if args.phase=='summary':
        show_summary();return
    if args.device=='mps' and args.phase in ('smoke','run') and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    train,dev,data_digest=v2.datasets()  # No LI/reserved examples; v2 checks splits.
    v2.locked_protocol(v2.protocol(data_digest))  # Verify unchanged EXP-102 protocol.
    now=protocol(data_digest,len(train),len(dev))
    if args.phase=='prepare':
        assert len(train)==2304 and len(dev)==288
        for seed in SEEDS:
            res=validate_schedule(seed,len(train))
            assert res=={'windows':800,'labels':204800}
        if PROTOCOL.exists():lock(now);print('EXISTING PROTOCOL VERIFIED',PROTOCOL)
        else:write_once(PROTOCOL,now);print('PREPARED IMMUTABLE PROTOCOL',PROTOCOL)
        print('PASS: 8 seeds x 800 four-step windows; exactly matched 256 source-question labels per window')
        print('TRAIN 2304 / DEV 288; no LI/reserved/protected examples generated')
        return
    lock(now)
    train_data=v2.encode(train)
    dev_data=v2.encode(dev)
    if args.phase=='smoke':
        seed=(args.seeds or [SEEDS[0]])[0]
        for policy in POLICIES:
            model,opt=model_and_optimizer(seed,args.device)
            rng=Random(seed+RNG_OFFSET)
            window=schedule_window(rng,len(train))
            loss=train_batch(model,opt,train_data,step_pairs(window,policy,0),args.device)
            measured=enrich(v2.score(model,dev_data,dev,args.device))
            print(f'SMOKE {policy} loss={loss:.4f} DEV C_query={measured["C_query"]:.4f}')
        print('PASS: no EXP-104 training checkpoints or reports written')
        return
    for seed in (args.seeds if args.seeds is not None else SEEDS):
        for policy in POLICIES:
            run_one(seed,policy,train_data,dev_data,train,dev,data_digest,now,args.device)
    show_summary()


if __name__=='__main__':
    main()
