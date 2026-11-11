"""ML-094: source-complete anchor transfer; exploratory position-history audit.

Purpose: test whether the ML-093 donor-direction effect is restricted to
transplanting participant tokens from incompatible active/passive source positions.
Patch one source-end <sep> residual AFTER L1. By that location, the causal
prefix contains both complete events; no future-to-earlier-participant transfer.
Aligned/reversed active donors have the SAME length and source-end index.

This does not isolate pure semantic role: swapping agents in active donors
also changes lexical word order. It is NOT independent confirmation, not a
native-circuit necessity test, and not an estimate of how real LLMs operate.

Usage from repository root:
  python3 scripts/ml094_source_end_role_control.py --phase preflight
  python3 scripts/ml094_source_end_role_control.py --phase plan
  # REVIEW + COMMIT exact script and reports/ml094/frozen_discovery_plan.json
  python3 scripts/ml094_source_end_role_control.py --phase run

Inputs: existing ML-088 frozen checkpoints and preserved ML-092/093 reports.
NO protected examples are generated or re-evaluated; prior reports untouched.
"""
import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92
import ml093_equal_norm_source_audit as m93

ROOT = Path('results/ml088_fresh_models')
PLAN = Path('reports/ml094/frozen_discovery_plan.json')
REPORT = Path('reports/ml094/discovery_evaluation.json')
PREVIOUS = m93.REPORT
SEEDS = (701, 709, 719, 727, 733, 739, 743, 751)
ARMS = ('familiar', 'target')
BATCH_TEMPLATES = 12
RANDOM_REPS = 4
EPS = 1e-12
MODES = (
    'native', 'self_copy',
    'raw_active_aligned', 'raw_active_reversed',
    'equal_active_aligned', 'equal_active_reversed',
    'equal_passive_role_reversed', 'equal_passive_verb_changed',
) + tuple(f'equal_random_{i}' for i in range(RANDOM_REPS))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def second_accuracy(score):
    x = score['per_question_four_choice_accuracy']
    return .5 * (x['second_agent'] + x['second_patient'])


def bs_ci(xs, n=20000, seed=94094):
    rng = random.Random(seed)
    d = sorted(mean(xs[rng.randrange(len(xs))] for _ in xs) for _ in range(n))
    return [d[int(.025*n)], d[int(.975*n)-1]]


def source_sep(c):
    toks = ml.prompt(c)
    assert toks.count('<sep>') == 1
    i = toks.index('<sep>')
    assert all(tok not in ('agent', 'patient') for tok in toks[:i])
    return i


def pool_for(passive_groups):
    recv = [g[2] for g in passive_groups]
    old = m92.donors(passive_groups)
    def other_verb(c):
        v = next(v for v in ml.VERBS if v not in (c.target_verb, c.distractor_verb))
        return replace(c, target_verb=v)
    result = {
        'active_aligned': old['aligned'],
        'active_reversed': old['reversed'],
        'passive_role_reversed': [replace(c, target_dir=1-c.target_dir) for c in recv],
        'passive_verb_changed': [other_verb(c) for c in recv],
        'self': recv,
    }
    for i, receiver in enumerate(recv):
        assert all(ml.prompt(c).index('<sep>') < len(ml.prompt(c)) - 1
                   for c in [receiver] + [v[i] for v in result.values()])
        assert len({source_sep(result[k][i]) for k in ('active_aligned', 'active_reversed')}) == 1
        assert all(source_sep(result[k][i]) == source_sep(receiver)
                   for k in ('passive_role_reversed', 'passive_verb_changed', 'self'))
        assert tuple(m92.participants(receiver, 'second')) == tuple(m92.participants(result['active_aligned'][i], 'second'))
        assert tuple(m92.participants(receiver, 'second')) == tuple(reversed(m92.participants(result['active_reversed'][i], 'second')))
        assert tuple(m92.participants(receiver, 'second')) == tuple(reversed(m92.participants(result['passive_role_reversed'][i], 'second')))
        assert m92.participants(receiver, 'second') == m92.participants(result['passive_verb_changed'][i], 'second')
        assert m91.sentence(result['passive_verb_changed'][i]) != m91.sentence(receiver)
        assert m91.sentence(result['active_aligned'][i]) != m91.sentence(receiver)
        assert len({m91.sentence(q) for q in passive_groups[i]}) == 1
    return result


def prepare():
    # Calls m88.prepare through m92.select_cases EXACTLY once per process.
    old_hash, groups, old_plan_hash, _ = m93.prepare()
    m93_spec = m93.specification(old_hash, groups, old_plan_hash)
    m93_plan_hash = m93.freeze(m93_spec, False)
    if not PREVIOUS.is_file():
        raise FileNotFoundError('Preserve ML-093 results before running ML-094')
    prior = json.loads(PREVIOUS.read_text())
    assert prior['experiment'] == 'ML-093'
    assert prior['plan_sha256'] == m93_plan_hash
    assert prior['protected_examples_generated'] is False
    assert len(prior['rows']) == 16
    assert tuple(ml.SEEDS) == SEEDS and tuple(ml.ARMS) == ARMS
    assert len(groups['passive']) == 96
    return old_hash, groups, m93_plan_hash


def specification(old_hash, groups, previous_plan_hash):
    return {
        'experiment': 'ML-094',
        'status': 'EXPLORATORY_SOURCE_END_ANCHOR_AUDIT',
        'script_sha256': sha(__file__),
        'source_hashes': {x: sha(module.__file__) for x, module in (
            ('ml080', ml), ('ml088', m88), ('ml091', m91),
            ('ml092', m92), ('ml093', m93))},
        'ml088_protocol_sha256': old_hash,
        'ml093_plan_sha256': previous_plan_hash,
        'ml093_report_file_sha256': sha(PREVIOUS),
        'seeds': SEEDS, 'arms': ARMS,
        'cohort': 'Existing 16 ML-088 step-12800 checkpoints; no retraining',
        'cases': 'Same 96 examined ML-092/093 calibration sentences; no fresh independent evaluation',
        'prompts_sha256': digest([ml.prompt(q) for g in groups['passive'] for q in g]),
        'n_sentences': 96, 'n_questions': 384,
        'source_position': 'The <sep> source-end token, after both events and before event/role question',
        'site': 'post-L1 residual at source <sep> only; normal L2-L3 downstream',
        'donors': {
            'active_aligned': 'Second event active, same semantic participants/roles as passive receiver',
            'active_reversed': 'Same active template and word positions; names swapped so agent/patient reversed',
            'passive_role_reversed': 'Same passive voice/length/<sep> position, second-event names swapped and roles reversed',
            'passive_verb_changed': 'Same passive voice/length/<sep> position and participant roles, change only second verb to first distinct alternative',
            'self': 'Same passive receiver, must give exact zero changes',
        },
        'controls': MODES,
        'scale': 'For active aligned/reversed, per-example common=min(|delta_aligned|,|delta_reversed|), scale both DOWN only. Four deterministic isotropic random directions have exactly this norm. Passive role-vs-verb pair similarly scaled down to shared min norm.',
        'random_seed_formula': '94094000 + 1000*model_seed + replicate_index; CPU [96,96] standard normal, per-sentence',
        'primary': 'Familiar-arm paired 8-seed mean difference in ML-091 second_event_role_gating: equal_active_aligned minus equal_active_reversed at source-end <sep>, with seed-bootstrap CI and positive count',
        'secondary': 'Native accuracy, raw aligned/reversed, equal-norm source-end effects and second-question accuracy, aligned minus random, passive fixed-position role reversal vs same-role verb control, output wrong event/wrong role. No post hoc seed exclusions.',
        'interpretation': 'Both active donors are patched from identical absolute donor <sep> token positions with full source prefix available. This removes participant-token source-position swapping as the patch SITE, but active donors still differ by name order/identity-role binding, and active/passive token lengths differ. Patches may import answer-relevant semantic content; not proof of a necessary native binding circuit.',
        'protected_safety': 'Never open/regenerate ML-088 protected items; do not rerun ML-092/093; no report overwrite',
    }


def freeze(doc, create):
    expected = json.loads(json.dumps({**doc, 'sha256': digest(doc)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(expected, indent=2) + '\n')
    if not PLAN.exists():
        raise FileNotFoundError('Run --phase plan after preflight')
    assert json.loads(PLAN.read_text()) == expected, 'ML-094 frozen script/dependency inputs changed'
    return expected['sha256']


@torch.inference_mode()
def first_layer(model, cases, device):
    ids, ro, _, _ = ml.encode(cases)
    ids, ro = ids.to(device), ro.to(device)
    b, n = ids.shape
    ix = torch.arange(b, device=device)
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, device=device, dtype=torch.bool).triu(1)
    x = model.blocks[0](x, mask)
    positions = torch.tensor([source_sep(c) for c in cases], device=device)
    assert bool((positions < ro).all())
    return x[ix, positions]


@torch.inference_mode()
def get_vectors(model, passive_groups, device):
    cases = [q for g in passive_groups for q in g]
    pool = pool_for(passive_groups)
    ids, ro, _, _ = ml.encode(cases)
    ids, ro = ids.to(device), ro.to(device)
    b, n = ids.shape
    assert b == 4*len(passive_groups)
    ix = torch.arange(b, device=device)
    owners = ix // 4
    site = torch.tensor([source_sep(q) for q in cases], device=device)
    assert bool((site < ro).all())
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, device=device, dtype=torch.bool).triu(1)
    x = model.blocks[0](x, mask)
    receiver = x[ix,site]
    states = {k: first_layer(model, v, device)[owners] for k,v in pool.items()}
    deltas = {k: st-receiver for k,st in states.items()}
    assert float(deltas['self'].abs().max()) < 2e-5
    for k,v in deltas.items():
        for q in range(1,4):
            assert torch.allclose(v[0::4], v[q::4], atol=2e-5, rtol=1e-5), k
    return dict(x=x, mask=mask, ro=ro, site=site, deltas=deltas)


def random_unit(seed, rep, n_templates, width, device):
    rng = torch.Generator(device='cpu').manual_seed(94094000 + 1000*seed + rep)
    z = torch.randn((n_templates,width), generator=rng)
    return (z/z.norm(dim=-1,keepdim=True).clamp_min(EPS)).to(device)


def delta_for(v, mode, noise=None, offset=0):
    d = v['deltas']
    da = d['active_aligned']; dr = d['active_reversed']
    na = da.norm(dim=-1,keepdim=True); nr = dr.norm(dim=-1,keepdim=True)
    ca = torch.minimum(na,nr)
    dp = d['passive_role_reversed']; dv = d['passive_verb_changed']
    cp = torch.minimum(dp.norm(dim=-1,keepdim=True), dv.norm(dim=-1,keepdim=True))
    if mode=='native':
        return torch.zeros_like(da)
    if mode=='self_copy':
        return d['self']
    if mode=='raw_active_aligned':
        return da
    if mode=='raw_active_reversed':
        return dr
    if mode=='equal_active_aligned':
        return da * (ca/na.clamp_min(EPS))
    if mode=='equal_active_reversed':
        return dr * (ca/nr.clamp_min(EPS))
    if mode=='equal_passive_role_reversed':
        return dp * (cp/dp.norm(dim=-1,keepdim=True).clamp_min(EPS))
    if mode=='equal_passive_verb_changed':
        return dv * (cp/dv.norm(dim=-1,keepdim=True).clamp_min(EPS))
    if mode.startswith('equal_random_'):
        assert noise is not None
        owners = torch.arange(da.shape[0],device=da.device)//4
        n_sentences = da.shape[0]//4
        assert offset+n_sentences <= len(noise)
        return noise[offset:offset+n_sentences][owners] * ca
    raise ValueError(mode)


@torch.inference_mode()
def patched_logits(model, v, delta):
    x = v['x'].clone()
    ix = torch.arange(x.shape[0], device=x.device)
    x[ix, v['site']] = x[ix,v['site']] + delta
    for block in model.blocks[1:]:
        x=block(x,v['mask'])
    return model.out(model.norm(x[ix,v['ro']])).float().cpu()


@torch.inference_mode()
def preflight(groups):
    torch.manual_seed(94094)
    model=ml.Model().eval()
    sub=groups['passive'][:4]
    v=get_vectors(model,sub,'cpu')
    direct=m92.native_logits(model,sub,'cpu')
    outcomes={}
    for mode in MODES:
        noise=None
        if mode.startswith('equal_random_'):
            rep=int(mode.rsplit('_',1)[1]);noise=random_unit(SEEDS[0],rep,96,ml.WIDTH,'cpu')
        d=delta_for(v,mode,noise)
        z=patched_logits(model,v,d)
        assert torch.isfinite(z).all()
        outcomes[mode]=(z,d)
    err=float((outcomes['native'][0]-direct).abs().max())
    self_err=float((outcomes['self_copy'][0]-direct).abs().max())
    assert err<1e-4 and self_err<1e-4,(err,self_err)
    assert float((outcomes['raw_active_aligned'][0]-direct).abs().max())>1e-6, 'Anchor patch did not reach readout'
    for a,b in [('equal_active_aligned','equal_active_reversed'),
                ('equal_passive_role_reversed','equal_passive_verb_changed'),
                ('equal_active_aligned','equal_random_0')]:
        x=outcomes[a][1].norm(dim=-1)
        y=outcomes[b][1].norm(dim=-1)
        assert torch.allclose(x,y,rtol=1e-5,atol=2e-5),(a,b,float((x-y).abs().max()))
    assert float((outcomes['raw_active_aligned'][0]-direct).abs().max()) > 1e-6
    assert float((outcomes['raw_active_reversed'][0]-direct).abs().max()) > 1e-6
    for name in ('active_aligned','active_reversed','passive_role_reversed'):
        assert float(v['deltas'][name].abs().max())>1e-5
    # Any patch after final layer at a non-readout source position is a structural no-op.
    ids,ro,_,_=ml.encode([q for g in sub for q in g]);ids=ids
    x=model.embed(ids)+model.pos(torch.arange(ids.shape[1]))
    mask=torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).triu(1)
    for block in model.blocks:x=block(x,mask)
    ix=torch.arange(len(ro))
    unpatched=model.out(model.norm(x[ix,ro]))
    x[ix,v['site']]=x[ix,v['site']]+outcomes['equal_active_aligned'][1]
    untouched=model.out(model.norm(x[ix,ro]))
    assert float((unpatched-untouched).abs().max())<1e-6
    print('PREFLIGHT PASS: exact same source-end donor positions per active pair, question-invariant pre-question state')
    print(f'Native parity={err:.2e}; self parity={self_err:.2e}; equal-norm active, passive and random controls pass')
    print('Post-L3 source patch is answer-no-op. No trained models or protected examples accessed.')


@torch.inference_mode()
def run(old_hash, plan_hash, groups, device):
    if REPORT.exists():
        raise FileExistsError(f'ML-094 already evaluated; refusing rerun: {REPORT}')
    gate,counts=m88.model_gate(old_hash)
    if not gate:
        raise PermissionError(f'Frozen model competence gate failed: {counts}')
    previous=json.loads(PREVIOUS.read_text())
    prior={(r['seed'],r['arm']):r['interventions']['native']
           for r in previous['rows']}
    assert len(prior)==len(SEEDS)*len(ARMS)
    rows=[]
    for seed in SEEDS:
        noises={r:random_unit(seed,r,len(groups['passive']),ml.WIDTH,device)
                for r in range(RANDOM_REPS)}
        for arm in ARMS:
            model=ml.load_checkpoint(ROOT/f'seed_{seed}'/arm/'model_step_12800.pt',
                                     seed,arm,old_hash,device)
            output={m:[] for m in MODES}
            norms={m:[] for m in MODES}
            for start in range(0,len(groups['passive']),BATCH_TEMPLATES):
                sub=groups['passive'][start:start+BATCH_TEMPLATES]
                vectors=get_vectors(model,sub,device)
                for mode in MODES:
                    noise=noises[int(mode.rsplit('_',1)[1])] if mode.startswith('equal_random_') else None
                    d=delta_for(vectors,mode,noise,offset=start)
                    norms[mode].append(float(d.norm(dim=-1).mean()))
                    output[mode].append(patched_logits(model,vectors,d))
            scores={m:m91.analyze_logits(torch.cat(z),groups['passive']) for m,z in output.items()}
            reference=prior[seed,arm]
            assert abs(scores['native']['second_event_role_gating']-reference['second_event_role_gating'])<.04, (seed,arm,'gating')
            assert abs(second_accuracy(scores['native'])-reference['second_accuracy'])<.012, (seed,arm,'accuracy')
            assert (scores['native']['second_event_role_gating']-scores['self_copy']['second_event_role_gating'])**2<1e-8
            assert scores['native']['four_choice_accuracy']==scores['self_copy']['four_choice_accuracy']
            details={mode:{
                'second_event_role_gating':score['second_event_role_gating'],
                'second_accuracy':second_accuracy(score),
                'full_vocab_accuracy':score['full_vocab_accuracy'],
                'wrong_event_rate':score['wrong_event_rate'],
                'correct_event_wrong_role_rate':score['correct_event_wrong_role_rate'],
                'four_choice_accuracy':score['four_choice_accuracy'],
                'all_four_queries_correct_rate':score['all_four_queries_correct_rate'],
                'mean_applied_patch_norm':mean(norms[mode]),
            } for mode,score in scores.items()}
            rows.append({'seed':seed,'arm':arm,'metrics':details})
            a=details['equal_active_aligned'];b=details['equal_active_reversed']
            v=details['native'];r=mean(details[f'equal_random_{i}']['second_event_role_gating'] for i in range(RANDOM_REPS))
            print(f'seed={seed} arm={arm:8} native2={v["second_accuracy"]:.3f} '
                  f'sepA_dacc={a["second_accuracy"]-v["second_accuracy"]:+.3f} '
                  f'sepR_dacc={b["second_accuracy"]-v["second_accuracy"]:+.3f} '
                  f'AminusR_gate={a["second_event_role_gating"]-b["second_event_role_gating"]:+.3f} '
                  f'AminusRand_gate={a["second_event_role_gating"]-r:+.3f}',flush=True)
    summary={}
    for arm in ARMS:
        ss=[r['metrics'] for r in rows if r['arm']==arm]
        primary=[r['equal_active_aligned']['second_event_role_gating']-r['equal_active_reversed']['second_event_role_gating'] for r in ss]
        rand=[r['equal_active_aligned']['second_event_role_gating']-mean(r[f'equal_random_{i}']['second_event_role_gating'] for i in range(RANDOM_REPS)) for r in ss]
        passive=[r['equal_passive_role_reversed']['second_event_role_gating']-r['equal_passive_verb_changed']['second_event_role_gating'] for r in ss]
        summary[arm]={
            'source_end_matched_minus_reversed_by_seed':dict(zip(map(str,SEEDS),primary)),
            'source_end_matched_minus_reversed_gating_mean':mean(primary),
            'source_end_matched_minus_reversed_bootstrap95':bs_ci(primary),
            'source_end_matched_minus_reversed_positive_seeds':sum(z>0 for z in primary),
            'source_end_matched_minus_random_gating_mean':mean(rand),
            'source_end_matched_minus_random_positive_seeds':sum(z>0 for z in rand),
            'source_end_passive_roleflip_minus_verb_gating_mean':mean(passive),
            'source_end_equalaligned_minus_native_accuracy_mean':mean(r['equal_active_aligned']['second_accuracy']-r['native']['second_accuracy'] for r in ss),
            'source_end_equalaligned_minus_reversed_accuracy_mean':mean(r['equal_active_aligned']['second_accuracy']-r['equal_active_reversed']['second_accuracy'] for r in ss),
            'source_end_rawaligned_minus_native_accuracy_mean':mean(r['raw_active_aligned']['second_accuracy']-r['native']['second_accuracy'] for r in ss),
            'mean_aligned_patch_norm':mean(r['equal_active_aligned']['mean_applied_patch_norm'] for r in ss),
        }
    doc={'experiment':'ML-094','status':'EXPLORATORY_SOURCE_END_ANCHOR','plan_sha256':plan_hash,
         'ml088_protocol_sha256':old_hash,'protected_cases_generated':False,
         'n_checkpoints':len(SEEDS)*len(ARMS),'rows':rows,'summary':summary,
         'interpretation_warning':json.loads(PLAN.read_text())['interpretation']}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(doc,indent=2)+'\n')
    print('\nML-094 EXPLORATORY SUMMARY\n'+json.dumps(summary,indent=2))
    print('Saved',REPORT)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',required=True,choices=('preflight','plan','run'))
    ap.add_argument('--device',default='auto',choices=('auto','cpu','mps','cuda'))
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    old_hash,groups,prior_hash=prepare()
    spec=specification(old_hash,groups,prior_hash)
    if args.phase=='preflight':
        if PLAN.exists():freeze(spec,False)
        preflight(groups)
    else:
        h=freeze(spec,create=args.phase=='plan')
        if args.phase=='plan':
            print('ML-094 frozen exploratory plan',PLAN,'sha256',h)
            print('No trained checkpoints evaluated; protected examples untouched.')
        else:
            run(old_hash,h,groups,ml.select_device(args.device))


if __name__=='__main__':main()
