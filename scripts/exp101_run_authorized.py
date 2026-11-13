#!/usr/bin/env python3
"""EXP-101 training-only runner, release locked. Never constructs protected cases.

Executed explicitly on the author's Mac after local source/hardware/grammar seal.
No hyperparameter tuning, no per-seed selection or early stopping.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import exp101_production_candidate as prod
import exp101_release as release

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'reports/exp101/frozen_release.json'
RESULTS = ROOT / 'results/exp101'


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def commit_sha() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def save_once(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, sort_keys=True, indent=2) + '\n'
    if path.exists():
        if path.read_text() != encoded:
            raise PermissionError(f'Existing immutable artifact differs: {path}')
    else:
        path.write_text(encoded)


def approved() -> tuple[dict, dict]:
    candidate = release.require_approved_release(FROZEN)
    frozen = json.loads(FROZEN.read_text())
    git_commit = commit_sha()
    if frozen.get('source_commit_sha') != git_commit:
        raise PermissionError('Git HEAD differs from approved frozen source commit')
    if candidate['config']['training']['seeds'] != list(prod.SEEDS):
        raise PermissionError('Approved seeds differ from implementation')
    if candidate['config']['training']['arms'] != list(prod.ARMS):
        raise PermissionError('Approved arms differ from implementation')
    if candidate['protected_evaluation_authorized'] is not False:
        raise PermissionError('Protected data must remain locked during training')
    return candidate, frozen


def training():
    candidate, frozen = approved()
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('Study authorized on intended Mac MPS only; no automatic CPU fallback')
    record = {
        'study': 'EXP-101', 'phase': 'TRAINING_ONLY',
        'source_commit_sha': commit_sha(), 'candidate_fingerprint': candidate['fingerprint'],
        'config_sha256': candidate['config_sha256'],
        'dataset_manifest_sha256': candidate['dataset_manifest_sha256'],
        'configuration': candidate['config'],
        'hardware_report_sha256': frozen['intended_hardware_report_sha256'],
        'grammar_review': frozen['human_grammatical_review'],
        'protected_evaluation_authorized': False,
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'device': 'mps',
    }
    save_once(RESULTS / 'authorized_training_record.json', record)
    print('EXP-101 TRAINING-ONLY START', flush=True)
    print('commit:', record['source_commit_sha'], flush=True)
    print('fingerprint:', record['candidate_fingerprint'], flush=True)
    print('seed pairs:', prod.SEEDS, 'arms:', prod.ARMS, 'updates/model:', prod.STEPS, flush=True)
    for seed in prod.SEEDS:
        for arm in prod.ARMS:
            print(f'EXP-101 START seed={seed} arm={arm} updates={prod.STEPS} {utc()}', flush=True)
            result = prod.train_one(seed=seed, arm=arm, device='mps', root=RESULTS,
                                    approved_release_path=FROZEN)
            if result.get('step') != prod.STEPS:
                raise RuntimeError(f'Incomplete training: {seed} {arm}')
            print(f'EXP-101 COMPLETE seed={seed} arm={arm} step={prod.STEPS} {utc()}', flush=True)
    print('EXP-101 ALL 16 TRAINING RUNS COMPLETE; DEVELOPMENT GATE NEXT', flush=True)
    development_gate()


def development_gate():
    candidate, _ = approved()
    scores = {}
    for seed in prod.SEEDS:
        for arm in prod.ARMS:
            score_path = RESULTS/f'seed_{seed}'/arm/'development_final.json'
            if score_path.exists():
                # On restart, reuse the immutable, already scored final checkpoint.
                # Do not re-score potentially nondeterministic MPS inference or tune.
                cached = json.loads(score_path.read_text())
                if (cached.get('source_commit_sha')!=commit_sha() or
                    cached.get('fingerprint')!=candidate['fingerprint'] or
                    cached.get('final_training_step')!=prod.STEPS or
                    cached.get('seed')!=seed or cached.get('arm')!=arm or
                    cached.get('protected_accessed') is not False):
                    raise PermissionError('Existing development artifact does not match frozen release')
                cells = cached['development_cells']
                print(f'EXP-101 DEV CACHED seed={seed} arm={arm}',flush=True)
            else:
                print(f'EXP-101 DEV SCORE seed={seed} arm={arm} {utc()}', flush=True)
                cells = prod.development_checkpoint_scores(seed=seed, arm=arm, device='mps',
                          root=RESULTS, approved_release_path=FROZEN)
            scores[(seed,arm)] = cells
            summary = {
                'study':'EXP-101', 'seed':seed, 'arm':arm,
                'source_commit_sha':commit_sha(), 'fingerprint':candidate['fingerprint'],
                'final_training_step':prod.STEPS, 'development_cells':cells,
                'protected_accessed':False,
            }
            save_once(score_path, summary)
            for key in sorted(cells):
                x = cells[key]
                print(f'  {key}: N={x["events"]} single={x["individual_accuracy"]:.5f} both={x["both_correct"]:.5f}', flush=True)
    gate = prod.competence_gate(scores)
    report = {
       'study':'EXP-101', 'status':'DEVELOPMENT_GATE_ONLY',
       'source_commit_sha': commit_sha(), 'fingerprint':candidate['fingerprint'],
       'criteria':'nine cells in each arm: >=96 events, >=0.95 individual, >=0.90 both; >=6 jointly competent pairs',
       'jointly_competent_seed_pairs':gate['jointly_passing'],
       'eligible_for_separately_authorized_protected_evaluation':gate['eligible'],
       'per_model':{f'{seed}/{arm}':gate['per_model'][(seed,arm)] for seed in prod.SEEDS for arm in prod.ARMS},
       'protected_evaluation_authorized':False,
       'protected_evaluation_accessed':False,
    }
    save_once(RESULTS/'development_gate_final.json', report)
    print('EXP-101 DEVELOPMENT GATE:',json.dumps(report,indent=2,sort_keys=True),flush=True)
    if not gate['eligible']:
        print('STOP: competence gate not met; NO protected unlock', flush=True)
    else:
        print('Gate met. Protected evaluation still LOCKED pending separate explicit authorization.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train','development-gate'),required=True)
    args = parser.parse_args()
    if args.phase == 'train':
        training()
    else:
        development_gate()
