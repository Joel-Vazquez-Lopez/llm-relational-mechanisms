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
)
CONFIG={
    'architecture':{'layers':3,'width':96,'heads':4,'max_tokens':64,
       'attention':'torch.nn.MultiheadAttention, causal bool upper triangular',
       'positional_embedding':'learned_absolute'},
    'optimizer':{'name':'AdamW','learning_rate':.002,'weight_decay':.01,
       'scheduler':'constant','steps':12800},
    'training':{'shared_batch':56,'dose_batch':8,'seeds':[907,911,919,929,937,941,947,953],
        'arms':['familiar','target'],'checkpoint_interval':800,
        'paired_initialization':True,'paired_shared_batch_rng':True},
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
    length_issue={'training_max_tokens':train_max,'LI_development_static_max_tokens':li_max,
       'LI_within_arch_embedding_capacity':li_max<=MAX_TOKENS,
       'all_LI_positions_trained':li_max<=train_max,
       'untrained_positions':list(range(train_max,li_max)) if li_max>train_max else []}
    meta=protected_metadata_plan()
    data_hash=stable_hash(dm)
    obj={'study':'EXP-101','status':'PREFREEZE_REVIEW_ONLY_NO_TRAIN_AUTHORIZATION',
         'sources_sha256':source_hashes(root), 'config':CONFIG,
         'config_sha256':hash_bytes(canonical(CONFIG)),
         'dataset_manifest_sha256':data_hash,
         'dataset_provenance_sha256':dm.get('provenance_sha256'),
         'static_report_sha256':hash_bytes((root/'reports/exp101/static_preflight_R3.json').read_bytes()),
         'protected_plan':meta,'position_coverage':length_issue,
         'human_grammatical_review':'PENDING_REPEATED_BY_ACCEPTABILITY',
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
    if (approved.get('status')!='FROZEN_APPROVED_AFTER_INDEPENDENT_REVIEW'
        or approved.get('fingerprint')!=candidate['fingerprint']
        or not approved.get('training_authorized')
        or approved.get('authorized_by') is None
        or approved.get('human_grammatical_review')!='PASSED'
        or not candidate['position_coverage']['all_LI_positions_trained']):
        raise PermissionError('Source, grammar or length issue unresolved; release not approved')
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
