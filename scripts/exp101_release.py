#!/usr/bin/env python3
"""EXP-101 reproducible PRE-FREEZE release attestation (stdlib; no models).

No test text, checkpoints, training, or evaluation is loaded. The checked-in
candidate is *not* a train authorization. Approval requires separate explicit
frozen_release.json, not included in this release candidate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SOURCE_PATHS=(
    'scripts/exp101_generator_preflight.py',
    'scripts/exp101_render_validator.py',
    'scripts/exp101_production_candidate.py',
    'scripts/exp101_runtime.py',
    'scripts/exp101_scoring.py',
    'scripts/exp101_protected_candidate.py',
    'scripts/exp101_release.py',
    'scripts/test_exp101_offline.py',
    'scripts/test_exp101_release_offline.py',
    'paper/exp101_counterfactual_generalization_protocol.md',
    'paper/exp101_release_candidate_review.md',
    'paper/exp101_position_amendment_RC2.md',
    'paper/exp101_grammar_review_RC2.md',
    'scripts/exp101_hardware_fixture.py',
    'paper/exp101_mps_determinism_RC3.md',
    'paper/exp101_mps_fixture_same_origin_RC4.md',
    'paper/exp101_author_waiver_RC5.md',
    'scripts/exp101_seal_local_release.py',
)
CONFIG={
    'architecture':{'layers':3,'width':96,'heads':4,'max_tokens':64,
       'attention':'torch.nn.MultiheadAttention, causal bool upper triangular',
       'positional_embedding':'learned_absolute_with_deterministic_training_index_offset'},
    'optimizer':{'name':'AdamW','learning_rate':.002,'weight_decay':.01,
       'scheduler':'constant','steps':12800},
    'training':{'shared_batch':56,'dose_batch':8,'seeds':[907,911,919,929,937,941,947,953],
        'arms':['familiar','target'],'checkpoint_interval':800,
        'paired_initialization':True,'paired_shared_batch_rng':True,
        'position_offset_cycle':[0,1,2,3,4,5,6,7,8],
        'position_offset_schedule':'offset_for_zero_based_update_index = update_index % 9',
        'position_offset_inference':0,
        'determinism_policy':{'cpu':'strict_error','mps':'warn_only_with_measured_tolerance',
            'mps_bitwise_reproducibility_claimed':False}},
    'scoring':{'primary':'C_voice target_second L+I',
       'choice':'native four event-participant name logits',
       'bootstrap':'paired seeds 20000 reps RNG 101101',
       'gate':'9 familiar cells >=95% individual and >=90% both, >=6 joint seed pairs',
       'cquery':'4 different people, same source 2 events x 2 roles'},
    'conditions':['bare','L','I','LI'],
    'protected':{'anchors':288,'packets_per_condition':2304,
                 'target_questions_per_condition':4608,
                 'additional_LI_opposite_questions':4608,
                 'total_inferences_per_model':23040,'models':16},
}

def hash_bytes(v:bytes)->str:return hashlib.sha256(v).hexdigest()
def canonical(v)->bytes:return json.dumps(v,sort_keys=True,separators=(',',':')).encode()

def protected_metadata_plan():
    """One canonical, independently auditable future plan; no rendered text."""
    from exp101_protected_candidate import future_manifest_plan
    plan=future_manifest_plan()
    assert plan['anchors']==288
    assert plan['total_questions_per_model']==23040
    return plan

def source_hashes(root:Path):
    root=Path(root)
    return {name:hash_bytes((root/name).read_bytes()) for name in SOURCE_PATHS}


def build_candidate(root:Path):
    from exp101_production_candidate import dataset_manifest,stable_hash,MAX_TOKENS,WIDTH,HEADS,LAYERS,STEPS,SHARED_BATCH,DOSE_BATCH,SEEDS,ARMS
    from exp101_generator_preflight import case_events,render_packet,VOICES
    from exp101_production_candidate import token_sequence
    root=Path(root)
    assert (WIDTH,HEADS,LAYERS,MAX_TOKENS,STEPS,SHARED_BATCH,DOSE_BATCH)==(96,4,3,64,12800,56,8)
    assert tuple(CONFIG['training']['seeds'])==SEEDS and tuple(CONFIG['training']['arms'])==ARMS
    dm,train,dev,dose=dataset_manifest()
    train_max=max(len(token_sequence(p,role)) for p in train for role in ('agent','patient'))
    li_max=0
    # Entirely NEW generator-only development templates, not protected IDs/text.
    for tv in VOICES:
      for ov in VOICES:
       for lt in (0,1):
        for lo in (0,1):
         a,b,nn=case_events(0,0,tv,ov,0,lt,lo,0,0)
         for pos in (False,True):
          p=render_packet(a,b,nn,'LI',pos)
          li_max=max(li_max,len(token_sequence(p,'agent')))
    from exp101_production_candidate import POSITION_OFFSETS
    # Actual non-padding question-token opportunities from all training prompts.
    # This is not trained-model evidence or proof of gradient magnitude.
    question_hits={str(i):0 for i in range(train_max,li_max)}
    token_hits={str(i):0 for i in range(train_max,li_max)}
    for pkt in train:
      for role in ('agent','patient'):
        toks=token_sequence(pkt,role)
        sep=toks.index('<sep>')
        for shift in POSITION_OFFSETS:
          for j in range(train_max,li_max):
            original_pos=j-shift
            if 0<=original_pos<len(toks):
                token_hits[str(j)]+=1
                if original_pos>=sep:question_hits[str(j)]+=1
    all_covered=all(token_hits[str(i)]>0 and question_hits[str(i)]>0
                    for i in range(train_max,li_max))
    length_issue={'training_max_tokens':train_max,'LI_development_static_max_tokens':li_max,
       'LI_within_arch_embedding_capacity':li_max<=MAX_TOKENS,
       'position_offsets':list(POSITION_OFFSETS),
       'unshifted_training_uncovered_indices':list(range(train_max,li_max)),
       'nonpad_token_opportunities_per_index':token_hits,
       'question_token_opportunities_per_index':question_hits,
       'all_LI_eval_positions_covered':all_covered,
       'augmentation_changes_training_regime':True,
       'fixed_offset_zero_for_dev_and_future_test':True,
       'measured_gradient_fixture_status':'SEE_SEPARATE_OFFLINE_VALIDATION'}
    if not all_covered:raise AssertionError('Position amendment does not cover LI evaluation indices')
    meta=protected_metadata_plan()
    data_hash=stable_hash(dm)
    obj={'study':'EXP-101','status':'PREFREEZE_REVIEW_ONLY_NO_TRAIN_AUTHORIZATION',
         'sources_sha256':source_hashes(root), 'config':CONFIG,
         'config_sha256':hash_bytes(canonical(CONFIG)),
         'dataset_manifest_sha256':data_hash,
         'dataset_provenance_sha256':dm.get('provenance_sha256'),
         'static_report_sha256':hash_bytes((root/'reports/exp101/static_preflight_RC2.json').read_bytes()),
         'cpu_engineering_fixture_sha256':hash_bytes((root/'reports/exp101/engineering_fixture_cpu_RC4.json').read_bytes()),
         'protected_plan':meta,'position_coverage':length_issue,
         'human_grammatical_review':'AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED',
         'independent_grammar_review_waived':True,
         'intended_hardware_runtime_review':'USER_REPORTED_TWO_MPS_PASSES_PENDING_RAW_ARCHIVE',
         'training_authorized':False,
         'protected_evaluation_authorized':False}
    obj['fingerprint']=hash_bytes(canonical(obj))
    return obj


def validate_candidate(path:Path):
    path=Path(path);root=path.parents[2]
    found=json.loads(path.read_text());expected=build_candidate(root)
    if found!=expected:
        raise PermissionError('Candidate manifest differs from exact source/config/data/scoring hashes')
    return found


def require_approved_release(path:Path):
    """Hard guard used before training, checkpoint inference and protected test."""
    path=Path(path)
    if not path.is_file():
        raise PermissionError('No reviewed frozen release exists; training/evaluation forbidden')
    root=path.parents[2]
    approved=json.loads(path.read_text())
    candidate=validate_candidate(root/'reports/exp101/release_manifest_candidate.json')
    if (approved.get('status')!='FROZEN_APPROVED_AFTER_DISCLOSED_AUTHOR_WAIVER'
        or approved.get('fingerprint')!=candidate['fingerprint']
        or not approved.get('training_authorized')
        or approved.get('authorized_by') is None
        or approved.get('human_grammatical_review')!='AUTHOR_SELF_REVIEW_ACCEPTED_INDEPENDENT_WAIVED'
        or approved.get('independent_grammar_review_waived') is not True
        or candidate.get('independent_grammar_review_waived') is not True
        or approved.get('protected_evaluation_authorized') is not False
        or not candidate['position_coverage']['all_LI_eval_positions_covered']):
        raise PermissionError('Source, grammar or length issue unresolved; release not approved')
    # Raw Mac MPS engineering reports must be checked locally and attested
    # against the approved exact fingerprint, without any study seed training.
    report_path=root/'reports/exp101/intended_hardware_fixture_approved.json'
    if not report_path.exists():raise PermissionError('Mac MPS fixture not reviewed')
    hardware=json.loads(report_path.read_text())
    if (hardware.get('fingerprint')!=candidate['fingerprint']
        or hardware.get('status')!='PASSED_ON_INTENDED_HARDWARE'
        or hardware.get('device')!='mps'
        or hardware.get('determinism_policy',{}).get('mode')!='mps_warn_only'
        or hardware.get('resume_optimizer_within_tolerance') is not True
        or hardware.get('same_origin_checkpoint_optimizer_exact') is not True
        or hardware.get('same_origin_checkpoint_roundtrip_max_abs_difference') != 0.0
        or hardware.get('resume_origin')!='same_model_a_checkpoint_after_step_1'
        or hardware.get('paired_first_step_batch_indices_equal') is not True
        or hardware.get('experiment_training_started') is not False
        or hardware.get('protected_accessed') is not False
        or approved.get('intended_hardware_review')!='PASSED'
        or approved.get('intended_hardware_report_sha256')!=hash_bytes(report_path.read_bytes())):
        raise PermissionError('Hardware integration review is absent or mismatched')
    if not all(hardware.get('run_'+str(i)+'_raw_sha256') and
               hardware.get('run_'+str(i)+'_log_sha256') for i in (1,2)):
        raise PermissionError('Both original raw MPS JSON/log SHA-256 records required')
    return candidate

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('check-candidate',),required=True)
    a=p.parse_args()
    manifest=validate_candidate(Path(__file__).parents[1]/'reports/exp101/release_manifest_candidate.json')
    print('Release candidate hashes match: '+manifest['fingerprint'])
    print('Training authorized: '+str(manifest['training_authorized']))
    print('Position coverage: '+json.dumps(manifest['position_coverage']))
