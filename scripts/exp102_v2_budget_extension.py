#!/usr/bin/env python3
"""EXP-102 v2: immutable, development-only 1600 -> 3200 budget extension.

Starts from the ORIGINAL final weights, AdamW state, and sampler RNG.
Never rewrites original reports/checkpoints, creates LI examples, or uses reserved sets.
No new sampling policy, architecture, optimizer, seed or dataset changes.

Exploratory continuation, NOT a revision of the fixed 1600-step endpoint.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

import exp102_v2_familiar_training as v2

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORTS = ROOT / 'reports/exp102/v2_familiar_pilot'
SOURCE_MODELS = ROOT / 'results/exp102/v2_familiar_pilot'
DEST_REPORTS = ROOT / 'reports/exp102/v2_budget_extension'
DEST_MODELS = ROOT / 'results/exp102/v2_budget_extension'
MANIFEST = DEST_REPORTS / 'extension_protocol.json'
FIRST_STEP = 1600
FINAL_STEP = 3200
MILESTONES = (2000, 2400, 2800, 3200)
SEEDS = tuple(v2.SEEDS)
METHODS = tuple(v2.METHODS)


def hash_file(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_files(seed, method):
    return (
        SOURCE_REPORTS / f'seed{seed}_{method}.json',
        SOURCE_MODELS / f'seed{seed}_{method}.pt',
    )


def dest_files(seed, method):
    return (
        DEST_REPORTS / f'seed{seed}_{method}_step{FINAL_STEP}.json',
        DEST_MODELS / f'seed{seed}_{method}_step{FINAL_STEP}.pt',
    )


def original_result(seed, method, expected_digest, *, with_checkpoint=True):
    path, ck_path = source_files(seed, method)
    report = json.loads(path.read_text())
    expected_protocol_hash = canonical_hash(v2.protocol(expected_digest))
    assert report['seed'] == seed and report['method'] == method
    assert report['steps'] == FIRST_STEP and report['protected_accessed'] is False
    assert report['stage'] == 'FAMILIAR_SYNTAX_DEVELOPMENT_FEASIBILITY'
    assert report['data_digest'] == expected_digest
    assert report['protocol_sha256'] == expected_protocol_hash
    assert [r['step'] for r in report['records']] == [400, 800, 1200, 1600]
    if not with_checkpoint:
        return report, None
    checkpoint = torch.load(ck_path, map_location='cpu', weights_only=True)
    assert checkpoint['metadata'] == report
    assert 'model' in checkpoint and 'optimizer' in checkpoint
    assert 'sampler_state' in checkpoint
    optim_state = checkpoint['optimizer']['state']
    assert optim_state, 'Original optimizer state is missing'
    update_steps = {int(s['step'].item()) for s in optim_state.values() if 'step' in s}
    assert update_steps == {FIRST_STEP}, f'AdamW updates do not match {FIRST_STEP}: {update_steps}'
    return report, checkpoint


def expected_manifest(digest, *, check_all_checkpoints):
    # The old runner's own source and data lock must remain valid.
    v2.locked_protocol(v2.protocol(digest))
    file_hashes = {}
    for seed in SEEDS:
        for method in METHODS:
            report, _ = original_result(seed, method, digest,
                                        with_checkpoint=check_all_checkpoints)
            a, b = source_files(seed, method)
            file_hashes[a.relative_to(ROOT).as_posix()] = hash_file(a)
            file_hashes[b.relative_to(ROOT).as_posix()] = hash_file(b)
    return {
        'study': 'EXP-102-v2-budget-extension',
        'status': 'EXPLORATORY_FAMILIAR_DEV_ONLY',
        'original_endpoint': FIRST_STEP,
        'new_fixed_endpoint': FINAL_STEP,
        'evaluation_steps': list(MILESTONES),
        'methods': list(METHODS),
        'seeds': list(SEEDS),
        'data_digest': digest,
        'original_protocol_sha256': hash_file(SOURCE_REPORTS / 'protocol.json'),
        'original_runner_sha256': hash_file(Path(v2.__file__)),
        'extension_runner_sha256': hash_file(__file__),
        'original_artifact_sha256': file_hashes,
        'restriction': 'No LI, no reserved name sets, no protected evaluation',
        'no_overwrite_originals': True,
        'no_checkpoint_selection': True,
        'threshold_not_registered': 'Exploratory learning-trajectory continuation',
        'device_rule': 'Original checkpoint device must match runtime device',
    }


def load_locked_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError('First run --phase prepare to lock the extension')
    manifest = json.loads(MANIFEST.read_text())
    if hash_file(__file__) != manifest['extension_runner_sha256']:
        raise PermissionError('Extension runner changed since preparation')
    if hash_file(Path(v2.__file__)) != manifest['original_runner_sha256']:
        raise PermissionError('Original runner changed since preparation')
    if hash_file(SOURCE_REPORTS / 'protocol.json') != manifest['original_protocol_sha256']:
        raise PermissionError('Original protocol changed since preparation')
    for relative, expected in manifest['original_artifact_sha256'].items():
        if hash_file(ROOT / relative) != expected:
            raise PermissionError(f'Original artifact changed: {relative}')
    assert manifest['new_fixed_endpoint'] == FINAL_STEP
    assert manifest['evaluation_steps'] == list(MILESTONES)
    assert manifest['seeds'] == list(SEEDS) and manifest['methods'] == list(METHODS)
    return manifest


def put_once(path, payload, *, is_torch=False):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    if temp.exists():
        raise FileExistsError(f'Leftover temporary artifact: {temp}; inspect manually')
    if is_torch:
        torch.save(payload, temp)
    else:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def restore(seed, method, digest, device):
    report, checkpoint = original_result(seed, method, digest)
    if report['device'] != device:
        raise PermissionError(f'Original was trained on {report["device"]}; requested {device}')
    torch.manual_seed(seed)
    model = v2.Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    sampler = torch.Generator(device='cpu')
    sampler.set_state(checkpoint['sampler_state'])
    return model, optimizer, sampler, report


def run_one(seed, method, digest, train_data, dev_data, probe_data, dev, probe,
            device, manifest):
    report_file, ck_file = dest_files(seed, method)
    if report_file.exists() and ck_file.exists():
        report = json.loads(report_file.read_text())
        assert report['source_protocol_sha256'] == manifest['original_protocol_sha256']
        assert report['seed'] == seed and report['method'] == method
        assert report['steps'] == FINAL_STEP
        print(f'ALREADY COMPLETE: seed={seed} method={method}', flush=True)
        return
    if report_file.exists() or ck_file.exists():
        raise FileExistsError(f'Incomplete extended artifact pair for {seed}/{method}')

    model, optimizer, sampler, original = restore(seed, method, digest, device)
    x, readout, y, _ = train_data
    records = []
    print(f'\nEXTEND seed={seed} method={method}: {FIRST_STEP} -> {FINAL_STEP}', flush=True)
    for step in range(FIRST_STEP + 1, FINAL_STEP + 1):
        ix = v2.batch_indices(sampler, method, len(x)//4)
        model.train()
        logits = model(x[ix].to(device), readout[ix].to(device))
        loss = F.cross_entropy(logits, y[ix].to(device))
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in MILESTONES:
            dm = v2.score(model, dev_data, dev, device)
            tm = v2.score(model, probe_data, probe, device)
            records.append({'step':step, 'dev':dm, 'train_probe':tm,
                            'last_batch_loss':float(loss.item())})
            print(f'STEP {step:4d} loss={loss.item():.4f} '
                  f'TRAIN C_query={tm["C_query"]:.3f} '
                  f'DEV C_query={dm["C_query"]:.3f} '
                  f'DEV4={dm["accuracy4"]:.3f} '
                  f'copy=({dm["role_copy_first"]:.3f},'
                  f'{dm["role_copy_second"]:.3f})', flush=True)

    new_report = {
        'study': 'EXP-102-v2-budget-extension',
        'stage': 'EXPLORATORY_FAMILIAR_DEV_ONLY',
        'seed':seed, 'method':method, 'start_step':FIRST_STEP,
        'steps':FINAL_STEP, 'data_digest':digest, 'device':device,
        'source_protocol_sha256':manifest['original_protocol_sha256'],
        'source_checkpoint_sha256':manifest['original_artifact_sha256'][
            source_files(seed, method)[1].relative_to(ROOT).as_posix()],
        'source_records':original['records'],
        'extension_records':records,
        'protected_accessed':False,
        'LI_cases_generated':False,
        'reserved_set_cases_generated':False,
        'evidence_status':'EXPLORATORY_ONLY',
    }
    put_once(ck_file, {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                       'sampler_state':sampler.get_state(), 'metadata':new_report}, is_torch=True)
    put_once(report_file, new_report)
    print('SAVED', report_file, flush=True)
    print('SAVED', ck_file, flush=True)


def show_summary():
    print('\nEXTENDED FAMILIAR DEV C_query (not protected)')
    complete = []
    for seed in SEEDS:
        vals = []
        for method in METHODS:
            report_file, ck_file = dest_files(seed, method)
            if not report_file.exists() or not ck_file.exists():
                break
            report=json.loads(report_file.read_text())
            assert report['steps'] == FINAL_STEP and report['seed']==seed
            assert report['method']==method and not report['protected_accessed']
            vals.append(report['extension_records'][-1]['dev']['C_query'])
        if len(vals)==len(METHODS):
            complete.append((seed, vals))
            print(f'{seed}: quartet={vals[0]:.3f} balanced_unpaired={vals[1]:.3f}')
    print(f'{len(complete)}/{len(SEEDS)} paired extensions complete')
    if len(complete) == len(SEEDS):
        gaps = [v[0]-v[1] for _,v in complete]
        print(f'Mean extended gap = {sum(gaps)/len(gaps):+.3f} '
              f'({sum(d>0 for d in gaps)} wins, '
              f'{sum(d==0 for d in gaps)} ties, '
              f'{sum(d<0 for d in gaps)} reversals)')
    print('EXPLORATORY ONLY; original step-1600 outcomes unchanged.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('prepare','smoke','run','summary'), required=True)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--device', choices=('cpu','mps'), default='mps')
    args = ap.parse_args()
    if args.seeds is not None and (not args.seeds or len(set(args.seeds))!=len(args.seeds)
                                   or any(s not in SEEDS for s in args.seeds)):
        ap.error(f'Use distinct members of {SEEDS}')
    if args.phase in ('run','smoke') and args.device=='mps' and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')

    if args.phase == 'summary':
        load_locked_manifest()
        show_summary()
        return
    train, dev, digest = v2.datasets()  # Familiar syntax only, no LI/reserved generation.
    if args.phase == 'prepare':
        expected = expected_manifest(digest, check_all_checkpoints=True)
        if MANIFEST.exists():
            assert json.loads(MANIFEST.read_text()) == expected, 'Existing extension protocol mismatch'
            print('EXISTING EXTENSION PROTOCOL VERIFIED:', MANIFEST)
        else:
            put_once(MANIFEST, expected)
            print('PREPARED IMMUTABLE EXTENSION:', MANIFEST)
        print('Original 16 checkpoints/reports verified; no training or protected access.')
        return

    manifest = load_locked_manifest()
    assert digest == manifest['data_digest'], 'Dataset digest changed'
    v2.locked_protocol(v2.protocol(digest))
    if args.phase=='smoke':
        # A disposable model+optimizer in RAM. Does not save or modify any artifact.
        seed = (args.seeds or [SEEDS[0]])[0]
        model, optim, g, _ = restore(seed, 'quartet', digest, args.device)
        x, readout, y, _ = v2.encode(train)
        ix = v2.batch_indices(g, 'quartet', len(train))
        model.train()
        logits = model(x[ix].to(args.device),readout[ix].to(args.device))
        loss = F.cross_entropy(logits,y[ix].to(args.device))
        optim.zero_grad(set_to_none=True)
        loss.backward(); optim.step()
        assert torch.isfinite(loss).item()
        print(f'SMOKE PASS: seed {seed}, step 1601 in memory, loss={loss.item():.4f}')
        print('No files written; original checkpoints unmodified.')
        return

    train_data = v2.encode(train)
    dev_data = v2.encode(dev)
    probe = train[:144]
    probe_data = v2.encode(probe)
    for seed in (args.seeds if args.seeds is not None else SEEDS):
        for method in METHODS:
            run_one(seed, method, digest, train_data, dev_data, probe_data,
                    dev, probe, args.device, manifest)
    show_summary()


if __name__=='__main__':
    main()
