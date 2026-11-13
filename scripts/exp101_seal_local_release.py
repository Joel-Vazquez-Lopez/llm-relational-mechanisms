#!/usr/bin/env python3
"""EXP-101 RC5 local freeze helper. No training and no protected item access.

Input: actual Mac fixture JSON and .log files from two fresh Python processes.
Records SHA-256 of the original files, checks release hash, optionally writes
explicit author-approved TRAINING-ONLY frozen_release.json. It never launches a
model and never authorizes protected evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import exp101_release as release

EXPECTED = {
    'status': 'ENGINEERING_FIXTURE_PASS_NOT_APPROVAL',
    'device': 'mps',
    'resume_origin': 'same_model_a_checkpoint_after_step_1',
    'same_origin_checkpoint_roundtrip_max_abs_difference': 0.0,
    'same_origin_checkpoint_optimizer_exact': True,
    'resume_optimizer_within_tolerance': True,
    'paired_first_step_batch_indices_equal': True,
    'experiment_training_started': False,
    'protected_accessed': False,
    'study_seed_used': False,
    'fixture_seed': 101999,
    'fixture_steps': 3,
    'dev_only_smoke_events': 2,
    'resume_param_abs_tol': 1e-5,
    'resume_param_rel_tol': 1e-4,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(json_path: Path, log_path: Path) -> dict:
    if not json_path.is_file() or not log_path.is_file():
        raise FileNotFoundError(f'Missing Mac fixture evidence: {json_path} or {log_path}')
    data = json.loads(json_path.read_text())
    for key, val in EXPECTED.items():
        if data.get(key) != val:
            raise PermissionError(f'{json_path}: {key}: expected {val!r}, got {data.get(key)!r}')
    p = data.get('determinism_policy')
    if not isinstance(p, dict) or p.get('mode') != 'mps_warn_only' or p.get('warn_only') is not True:
        raise PermissionError('Mac fixture must report the reviewed MPS warn-only policy')
    if data.get('known_backend_nondeterminism_warnings', 0) < 1:
        raise PermissionError('Expected MPS nondeterminism warnings not documented')
    if data.get('resume_parameter_max_abs_difference', float('inf')) > data['resume_param_abs_tol']:
        raise PermissionError('Same-origin checkpoint max difference exceeds strict absolute threshold')
    if data.get('independent_first_step_optimizer_within_tolerance') is not True:
        raise PermissionError('Paired independent first-step optimizer check failed')
    norms = data.get('position_gradient_norms_34_through_38')
    if not isinstance(norms, dict) or set(norms) != {str(i) for i in range(34,39)}:
        raise PermissionError('Positional gradient report incomplete')
    if any(float(norms[str(i)]) <= 1e-12 for i in range(34,39)):
        raise PermissionError('One or more evaluation positions have no measured gradient')
    if log_path.stat().st_size < 100:
        raise PermissionError('Mac fixture stdout/stderr log unexpectedly short')
    return {'raw_sha256':sha(json_path), 'log_sha256':sha(log_path),
            'resume_parameter_max_abs_difference':data['resume_parameter_max_abs_difference'],
            'known_backend_nondeterminism_warnings':data['known_backend_nondeterminism_warnings'],
            'torch':data.get('torch'), 'platform':data.get('platform')}


def local_git_commit(root: Path) -> str:
    c = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if len(c) != 40:
        raise PermissionError('Expected a pinned 40-character release Git commit SHA')
    return c


def seal(root: Path, *, authorize_training: bool, authorized_by: str):
    root = root.resolve()
    candidate_path = root / 'reports/exp101/release_manifest_candidate.json'
    candidate = release.validate_candidate(candidate_path)
    if candidate['training_authorized'] is not False:
        raise PermissionError('Candidate must remain locked (training_authorized=false)')
    if candidate.get('human_grammatical_review') != 'AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED':
        raise PermissionError('Explicit author-review waiver is missing from the source-bound manifest')
    folder=root/'reports/exp101'
    paths={i:(folder / f'mps_RC4_run{i}.json',folder / f'mps_RC4_run{i}.log') for i in (1,2)}
    runs={i:check_run(*paths[i]) for i in (1,2)}
    if runs[1]['raw_sha256'] == runs[2]['raw_sha256']:
        raise PermissionError('Two separate hardware fixture files are required')
    if runs[1]['torch'] != runs[2]['torch'] or runs[1]['platform'] != runs[2]['platform']:
        raise PermissionError('Inconsistent backend software/hardware across two fixtures')
    commit=local_git_commit(root)
    attestation={
        'study':'EXP-101','fingerprint':candidate['fingerprint'],
        'status':'PASSED_ON_INTENDED_HARDWARE','device':'mps',
        'determinism_policy':{'mode':'mps_warn_only','bitwise_determinism_claimed':False},
        'resume_origin':'same_model_a_checkpoint_after_step_1',
        'resume_optimizer_within_tolerance':True,
        'same_origin_checkpoint_optimizer_exact':True,
        'same_origin_checkpoint_roundtrip_max_abs_difference':0.0,
        'paired_first_step_batch_indices_equal':True,
        'experiment_training_started':False,'protected_accessed':False,
        'source_commit_sha':commit,'source':'USER_MAC_TWO_INDEPENDENT_PROCESSES',
        'run_1_raw_sha256':runs[1]['raw_sha256'],'run_1_log_sha256':runs[1]['log_sha256'],
        'run_2_raw_sha256':runs[2]['raw_sha256'],'run_2_log_sha256':runs[2]['log_sha256'],
        'run_1_resume_parameter_max_abs_difference':runs[1]['resume_parameter_max_abs_difference'],
        'run_2_resume_parameter_max_abs_difference':runs[2]['resume_parameter_max_abs_difference'],
        'run_1_backend_warning_count':runs[1]['known_backend_nondeterminism_warnings'],
        'run_2_backend_warning_count':runs[2]['known_backend_nondeterminism_warnings'],
        'torch_version':runs[1]['torch'],'platform':runs[1]['platform'],
        'reviewed_by':authorized_by,
        'review_scope':'three-step engineering fixture, not long-horizon MPS determinism',
    }
    hardware_path=folder/'intended_hardware_fixture_approved.json'
    frozen_path=folder/'frozen_release.json'
    if hardware_path.exists() or frozen_path.exists():
        raise FileExistsError('A release attestation already exists; never overwrite an approved record')
    if not authorize_training:
        print(json.dumps({'status':'DRY_RUN_READY_FOR_AUTHOR_SEAL','fingerprint':candidate['fingerprint'],
                          'commit':commit,'run_1':runs[1],'run_2':runs[2],
                          'grammar':'AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED',
                          'training_authorized':False},indent=2,sort_keys=True))
        return
    if not authorized_by.strip():
        raise PermissionError('An actual author identifier is required for a training approval record')
    attestation_bytes=(json.dumps(attestation,indent=2,sort_keys=True)+'\n').encode()
    frozen = {
        'study':'EXP-101', 'status':'FROZEN_APPROVED_AFTER_DISCLOSED_AUTHOR_WAIVER',
        'fingerprint':candidate['fingerprint'],
        'source_commit_sha':commit,'training_authorized':True,
        'protected_evaluation_authorized':False,
        'human_grammatical_review':'AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED',
        'independent_grammar_review_waived':True,
        'authorized_by':authorized_by,
        'intended_hardware_review':'PASSED',
        'intended_hardware_report_sha256':hashlib.sha256(attestation_bytes).hexdigest(),
        'decision_timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'TRAINING_ONLY_AFTER_AUTHOR_ACCEPTED_GRAMMAR_WAIVER',
    }
    hardware_path.write_bytes(attestation_bytes)
    frozen_path.write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
    # Validate through the SAME actual guard used by training and protected paths.
    release.require_approved_release(frozen_path)
    print(json.dumps({'status':'FROZEN_TRAINING_ONLY','fingerprint':candidate['fingerprint'],
        'source_commit_sha':commit,'hardware_report_sha256':frozen['intended_hardware_report_sha256'],
        'protected_evaluation_authorized':False,'training_authorized':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check-hardware','freeze-training'),required=True)
    p.add_argument('--authorized-by',default='author')
    p.add_argument('--acknowledge-independent-grammar-waiver',action='store_true')
    args=p.parse_args()
    if args.phase=='freeze-training' and not args.acknowledge_independent_grammar_waiver:
        p.error('Explicit --acknowledge-independent-grammar-waiver is required')
    seal(Path(__file__).resolve().parents[1],
         authorize_training=(args.phase=='freeze-training'),authorized_by=args.authorized_by)
