"""Paper audit: paired-role consistency on existing ML-088 calibration material.

Read-only on trained checkpoints, no protected evaluation, no retraining.
Reuses exact ML-091 calibration templates (96/voice), now saving per-source
answer choices and semantic logits for agent/patient consistency analysis.

From repository root:
  cp ~/Downloads/paper_role_consistency_audit.py scripts/
  python scripts/paper_role_consistency_audit.py --phase preflight
  python scripts/paper_role_consistency_audit.py --phase plan
  git add scripts/paper_role_consistency_audit.py reports/paper_role_consistency/frozen_plan.json
  git commit -m 'Freeze paired-role consistency calibration audit'
  python scripts/paper_role_consistency_audit.py --phase run --device auto
  python scripts/paper_role_consistency_audit.py --phase summarize

All findings are exploratory on a previously studied model cohort.
Do not claim that a simple marginal-independence comparison identifies an
internal representation or that aggregate marginals control sentence difficulty.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91

BASE = Path('reports/paper_role_consistency')
PLAN = BASE / 'frozen_plan.json'
OBS = BASE / 'calibration_observations.json'
SUMMARY = BASE / 'summary.json'
SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ('familiar', 'target')
VOICES = ('active', 'passive')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_inputs():
    # m91.prepare calls m88.prepare exactly once and configures the synthetic vocabulary.
    h, groups = m91.prepare()
    assert h == json.loads(m88.PROTOCOL.read_text())['sha256']
    prior = json.loads(m91.PLAN.read_text())
    assert prior['ml088_protocol_sha256'] == h
    assert len(groups['active']) == len(groups['passive']) == 96
    assert tuple(ml.SEEDS) == SEEDS
    for voice in VOICES:
        assert prior['sentences_sha256'][voice] == m91.digest(
            [m91.sentence(g[0]) for g in groups[voice]]
        )
        assert prior['prompt_hashes'][voice] == m91.digest(
            [ml.prompt(c) for g in groups[voice] for c in g]
        )
    return h, groups, prior


def specification(h, groups, prior):
    return {
        'experiment': 'PAPER-ROLE-CONSISTENCY-CALIBRATION',
        'type': 'exploratory_posthoc_analysis',
        'script_sha256': file_hash(__file__),
        'dependencies_sha256': {name: file_hash(module.__file__) for name, module in (
            ('ml080', ml), ('ml088', m88), ('ml091', m91)
        )},
        'ml088_protocol_sha256': h,
        'ml091_original_plan_sha256': prior['sha256'],
        'seeds': SEEDS, 'arms': ARMS, 'voices': VOICES,
        'sentences_per_voice': 96,
        'questions_per_sentence': 4,
        'question_order': m91.QUESTION_NAMES,
        'sentence_hashes': prior['sentences_sha256'],
        'scope': 'Existing ML-088 models and already analyzed ML-091 calibration sentences; NEVER protected; not independent replication.',
        'record': 'Per-source 4x4 semantic logits, four-choice predictions, full-vocabulary predictions and oracle-event two-name predictions.',
        'primary_descriptive': [
            'agent_accuracy', 'patient_accuracy', 'both_accuracy',
            'exactly_one_correct', 'same_prediction', 'both_select_correct_event',
            'same_prediction_conditioned_both_select_correct_event',
        ],
        'nulls': {
            'both_if_marginally_independent': 'agent_accuracy * patient_accuracy (heuristic only; sentence difficulty not controlled)',
            'same_if_independent_given_both_event': 'pA*(1-pP)+(1-pA)*pP on the selected both-in-event subset (descriptive, conditioned selection)',
        },
        'statistics': 'All seeds reported; seed-level paired means and bootstrap intervals; no hypothesis-search thresholds.',
        'safety': 'No training, no protected case generation, no protected result reads, no overwrite of output files.',
    }


def frozen(spec, create):
    doc = json.loads(json.dumps({**spec, 'sha256': digest(spec)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(doc, indent=2) + '\n')
    if not PLAN.exists():
        raise FileNotFoundError('Freeze the analysis first with --phase plan')
    assert json.loads(PLAN.read_text()) == doc, 'Frozen analysis/script/dependencies mismatch'
    return doc['sha256']


def per_source(group, logits):
    """Four questions, columns ordered (first A,P, second A,P)."""
    assert logits.shape[0] == 4
    labels = [ml.answer(c) for c in group]
    assert len(set(labels)) == 4
    ids = [ml.VOCAB[name] for name in labels]
    score = logits[:, ids].float().cpu()
    pred4 = score.argmax(1).tolist()
    pred_full = logits.argmax(1).tolist()
    oracle = []
    for q in range(4):
        e = q // 2
        oracle.append(2*e + int(score[q, 2*e+1] > score[q, 2*e]))
    return {
        'sentence_sha256': m91.digest(list(m91.sentence(group[0]))),
        'semantic_name_order': labels,
        'pred4': pred4,
        'pred_full_vocab_ids': pred_full,
        'oracle_pred_correct_event_pair': oracle,
        'semantic_logits_4x4': score.tolist(),
        'four_gold_vocab_ids': ids,
    }


def summarize_event(records, event):
    """Event-level paired A/P answers, with exact named alternatives.

    The both-in-correct-event subset has only two possible names; on this
    subset, duplicate predictions are logically inconsistent. The oracle
    two-name scores answer a different question: role discrimination GIVEN
    the correct event is supplied by the evaluator.
    """
    n = len(records)
    assert n > 0 and event in (0, 1)
    offset = 2 * event
    correctness_a = []; correctness_p = []; same = []
    oracle_a = []; oracle_p = []; oracle_same = []
    full_a = []; full_p = []
    both_in_event = []; both_other_event = []
    chosen_event_a = []; chosen_event_p = []
    sub_a = []; sub_p = []; sub_same = []
    both_swapped = 0
    for r in records:
        pa, pp = r['pred4'][offset:offset + 2]
        qa, qp = r['oracle_pred_correct_event_pair'][offset:offset + 2]
        ca, cp = (pa == offset), (pp == offset + 1)
        oa, op = (qa == offset), (qp == offset + 1)
        correctness_a.append(ca); correctness_p.append(cp)
        same.append(pa == pp)
        oracle_a.append(oa); oracle_p.append(op)
        oracle_same.append(qa == qp)
        full_a.append(r['pred_full_vocab_ids'][offset] == r['four_gold_vocab_ids'][offset])
        full_p.append(r['pred_full_vocab_ids'][offset+1] == r['four_gold_vocab_ids'][offset+1])
        in_a = pa // 2 == event
        in_p = pp // 2 == event
        chosen_event_a.append(in_a); chosen_event_p.append(in_p)
        both_in_event.append(in_a and in_p)
        both_other_event.append(not in_a and not in_p)
        if in_a and in_p:
            sub_a.append(ca); sub_p.append(cp); sub_same.append(pa == pp)
            both_swapped += (pa == offset+1 and pp == offset)
    pa = mean(correctness_a); pp = mean(correctness_p)
    both = mean(a and b for a,b in zip(correctness_a, correctness_p))
    exactly_one = mean(a != b for a,b in zip(correctness_a, correctness_p))
    neither = mean(not a and not b for a,b in zip(correctness_a, correctness_p))
    assert abs(2*(pa+pp)/2 - (2*both+exactly_one)) < 1e-8
    assert abs(both+exactly_one+neither-1.0) < 1e-8
    assert max(0,pa+pp-1)-1e-8 <= both <= min(pa,pp)+1e-8
    qa,qp = mean(oracle_a), mean(oracle_p)
    oboth = mean(a and b for a,b in zip(oracle_a,oracle_p))
    # With two options, exactly one role correct iff agent/patient answers duplicate.
    assert abs(mean(oracle_same) - mean(a != b for a,b in zip(oracle_a,oracle_p))) < 1e-8
    out = {
        'events': n,
        'agent_accuracy': pa, 'patient_accuracy': pp,
        'individual_accuracy': 0.5*(pa+pp),
        'both_accuracy': both,
        'exactly_one_correct': exactly_one,
        'neither_correct': neither,
        'same_prediction': mean(same),
        'both_select_correct_event': mean(both_in_event),
        'both_select_wrong_event': mean(both_other_event),
        'agent_select_correct_event': mean(chosen_event_a),
        'patient_select_correct_event': mean(chosen_event_p),
        'both_swapped_rate': both_swapped/n,
        'full_vocab_agent_accuracy': mean(full_a),
        'full_vocab_patient_accuracy': mean(full_p),
        'marginal_independence_both': pa*pp,
        'both_minus_marginal_product': both-pa*pp,
        'oracle_pair_agent_accuracy': qa,
        'oracle_pair_patient_accuracy': qp,
        'oracle_pair_both_accuracy': oboth,
        'oracle_pair_same_prediction': mean(oracle_same),
        'oracle_pair_independence_same': qa*(1-qp)+(1-qa)*qp,
        'oracle_pair_same_minus_independence': mean(oracle_same) - (qa*(1-qp)+(1-qa)*qp),
        'n_both_in_event': len(sub_a),
        'same_prediction_conditioned_both_select_correct_event': mean(sub_same) if sub_same else None,
        'agent_accuracy_conditioned_both_select_correct_event': mean(sub_a) if sub_a else None,
        'patient_accuracy_conditioned_both_select_correct_event': mean(sub_p) if sub_p else None,
        'same_minus_independence_given_both_event': None,
    }
    if sub_same:
        a,p = mean(sub_a),mean(sub_p)
        out['same_minus_independence_given_both_event'] = mean(sub_same) - (a*(1-p)+(1-a)*p)
    return out


def preflight(groups):
    # This existing test validates matching sentences and query factors.
    m91.preflight(groups)
    dummy = []
    names = [ml.answer(c) for c in groups['passive'][0]]
    ids = [ml.VOCAB[x] for x in names]
    v = len(ml.VOCAB)
    for answers in ([2,3], [2,2], [3,2], [0,1]):
        logits = torch.full((4,v),-8.)
        logits[0,ids[0]] = logits[1,ids[1]] = 10.
        logits[2,ids[answers[0]]] = 10.
        logits[3,ids[answers[1]]] = 10.
        dummy.append(per_source(groups['passive'][0],logits))
    s = summarize_event(dummy,1)
    assert abs(s['individual_accuracy']-0.375) < 1e-12
    assert abs(s['both_accuracy']-0.25) < 1e-12
    assert abs(s['same_prediction']-0.25) < 1e-12
    assert abs(s['both_swapped_rate']-0.25) < 1e-12
    assert s['n_both_in_event'] == 3
    assert abs(s['marginal_independence_both'] - 0.125) < 1e-12
    print('PAPER ROLE-CONSISTENCY PREFLIGHT PASS')
    print('Synthetic perfect/duplicate/swapped/wrong-event controls verified.')
    print('Existing ML-091 sentence construction verified. No checkpoint loaded.')


def bootstrap(xs, seed):
    rng = random.Random(seed)
    n=len(xs)
    dist=sorted(mean(xs[rng.randrange(n)] for _ in range(n)) for _ in range(20000))
    return [dist[500], dist[19499]]


def run(h, planhash, groups, device):
    if OBS.exists() or SUMMARY.exists():
        raise FileExistsError('Analysis already exists; will not overwrite.')
    ok, counts = m88.model_gate(h)
    if not ok:
        raise PermissionError(f'ML-088 existing competence gate failed: {counts}')
    rows=[]
    with torch.inference_mode():
        for seed in SEEDS:
            for arm in ARMS:
                ckpt=m91.ROOT/f'seed_{seed}'/arm/'model_step_12800.pt'
                model=ml.load_checkpoint(ckpt,seed,arm,h,device)
                group_rows={}
                for voice in VOICES:
                    cases=[c for g in groups[voice] for c in g]
                    inputs,readout,_,_=ml.encode(cases)
                    outputs=[]
                    for start in range(0,len(cases),64):
                        ids=inputs[start:start+64].to(device)
                        ro=readout[start:start+64].to(device)
                        outputs.append(model(ids,ro).float().cpu())
                    logits=torch.cat(outputs,0)
                    raw=[per_source(g,logits[4*i:4*i+4]) for i,g in enumerate(groups[voice])]
                    group_rows[voice]={'source_observations':raw, 'first':summarize_event(raw,0), 'second':summarize_event(raw,1)}
                    s=group_rows[voice]['second']
                    print(f'seed={seed} arm={arm:8s} voice={voice:7s} second_acc={s["individual_accuracy"]:.3f} both={s["both_accuracy"]:.3f} same={s["same_prediction"]:.3f} oracle_same={s["oracle_pair_same_prediction"]:.3f}',flush=True)
                rows.append({'seed':seed,'arm':arm,'voices':group_rows})
    payload={'study':'paper_paired_role_consistency','scope':'ML-088 previously examined calibration cases, no protected data','protocol_sha256':planhash,'source_ml088_protocol_sha256':h,'rows':rows}
    OBS.parent.mkdir(parents=True,exist_ok=True)
    OBS.write_text(json.dumps(payload,indent=2)+'\n')
    print('Saved detailed calibration observations:',OBS)
    make_summary(payload)


def make_summary(payload):
    if SUMMARY.exists():
        raise FileExistsError(f'Existing summary: {SUMMARY}')
    look={(r['seed'],r['arm']):r['voices'] for r in payload['rows']}
    assert set(look)=={(seed,arm) for seed in SEEDS for arm in ARMS}
    keys=['individual_accuracy','agent_accuracy','patient_accuracy','both_accuracy','exactly_one_correct','neither_correct','same_prediction','both_select_correct_event','same_prediction_conditioned_both_select_correct_event','oracle_pair_agent_accuracy','oracle_pair_patient_accuracy','oracle_pair_both_accuracy','oracle_pair_same_prediction','oracle_pair_same_minus_independence','both_minus_marginal_product','same_minus_independence_given_both_event','full_vocab_agent_accuracy','full_vocab_patient_accuracy']
    cells={};paired={}
    for voice in VOICES:
        for event in ('first','second'):
            for arm in ARMS:
                r=[look[(seed,arm)][voice][event] for seed in SEEDS]
                cells[f'{arm}/{voice}/{event}']={k:mean([x[k] for x in r if x[k] is not None]) if any(x[k] is not None for x in r) else None for k in keys}
            effects={}
            for k in keys:
                vals=[]
                for seed in SEEDS:
                    a=look[(seed,'familiar')][voice][event][k]
                    b=look[(seed,'target')][voice][event][k]
                    if a is not None and b is not None: vals.append(b-a)
                effects[k]={'n':len(vals),'mean':mean(vals),'positive_seeds':sum(x>0 for x in vals),'ci95':bootstrap(vals,100000+len(k)*37+int(voice=='passive')*71+int(event=='second')*113)} if vals else None
            paired[f'{voice}/{event}']=effects
    result={'study':'paper_paired_role_consistency','protocol_sha256':payload['protocol_sha256'],'status':'exploratory_reanalysis_same_cohort','seed_count':len(SEEDS),'sentences_per_voice_per_model':96,'cells':cells,'paired_target_minus_familiar':paired,'cautions':[
        'All 16 models and calibration sentences come from prior experiments; this is descriptive reanalysis, NOT independent confirmation.',
        'A marginal-independence product does not control sentence difficulty; differences from it do not demonstrate representational binding.',
        'Oracle correct-event two-choice accuracy supplies the correct event and therefore does not measure native event selection.',
        'Agreement or disagreement of predicted names cannot by itself identify the internal neural mechanism.',
    ]}
    SUMMARY.parent.mkdir(parents=True,exist_ok=True)
    SUMMARY.write_text(json.dumps(result,indent=2)+'\n')
    for voice in VOICES:
        print(f'\n=== SECOND EVENT / {voice.upper()} ===')
        for arm in ARMS:
            d=cells[f'{arm}/{voice}/second']
            print(f'{arm:8s} ind={d["individual_accuracy"]:.3f} both={d["both_accuracy"]:.3f} same={d["same_prediction"]:.3f} both-event={d["both_select_correct_event"]:.3f} oracle-same={d["oracle_pair_same_prediction"]:.3f} obs-minus-marginal-product={d["both_minus_marginal_product"]:+.3f}')
    print('Saved summary:',SUMMARY)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=('preflight','plan','run','summarize'))
    ap.add_argument('--device',default='auto',choices=('auto','cpu','cuda','mps'))
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.phase=='summarize':
        if not OBS.is_file(): raise FileNotFoundError(OBS)
        payload=json.loads(OBS.read_text())
        # No dependency loading needed for a read-only summary; no overwrite.
        make_summary(payload)
        return
    h,groups,prior=load_inputs()
    if args.phase=='preflight':
        preflight(groups)
        return
    planhash=frozen(specification(h,groups,prior), args.phase=='plan')
    if args.phase=='plan':
        print('Frozen paper audit SHA256:',planhash)
        print('Commit script and plan BEFORE running trained-model inference.')
        return
    run(h,planhash,groups,ml.select_device(args.device))

if __name__=='__main__':
    main()
