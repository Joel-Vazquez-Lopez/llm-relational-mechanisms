"""ML-098: Exploratory factorial answer-query Q-path transplant.

Does event/role selection in the native model depend on the answer-position
query vector after particular layers? Same sentence, four natural questions.

Reuses ML-088 paired checkpoints and ML-092 exploratory CALIBRATION sentences.
Never constructs or evaluates protected cases. Not independent confirmation.

From repo root:
    python scripts/ml098_question_q_path.py --phase preflight
    python scripts/ml098_question_q_path.py --phase plan
    # Review and commit the script + frozen plan before trained inference.
    python scripts/ml098_question_q_path.py --phase run --device auto

What is patched: Only the attention Q vector AT THE ANSWER TOKEN for one layer,
taken from another question about the SAME sentence. The receiver's K, V,
residual and all other tokens remain native at the patched layer. Downstream
layers compute normally. The donor Q may already encode answer-related
information, so this is a causal sufficiency test, NOT native necessity or a
proof of separate event/role features.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92

ROOT = Path('results/ml088_fresh_models')
PLAN = Path('reports/ml098_q_path/frozen_plan.json')
REPORT = Path('reports/ml098_q_path/calibration_evaluation.json')
SEEDS = tuple(m88.SEEDS)
ARMS = ('familiar', 'target')
VOICES = ('active', 'passive')
LAYERS = (0, 1, 2)
SWAPS = {
    'self': (0, 1, 2, 3),
    'event': (2, 3, 0, 1),
    'role': (1, 0, 3, 2),
    'both': (3, 2, 1, 0),
}
BATCH_GROUPS = 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def prepare():
    previous_protocol, groups, prior_plan_hash = m92.select_cases()
    assert tuple(ml.SEEDS) == SEEDS
    assert all(len(groups[v]) == 96 for v in VOICES)
    for voice in VOICES:
        for g in groups[voice]:
            assert len(g) == 4
            assert [q.role for q in g] == ['agent', 'patient', 'agent', 'patient']
            assert [q.target_pos for q in g] == [0, 0, 1, 1]
            assert len({m91.sentence(q) for q in g}) == 1
            assert len({len(ml.prompt(q)) for q in g}) == 1
            assert len({ml.answer(q) for q in g}) == 4
            assert len({tuple(ml.prompt(q)) for q in g}) == 4
    return previous_protocol, groups, prior_plan_hash


def specification(old_hash, groups, previous_hash):
    return {
        'experiment': 'ML-098',
        'status': 'EXPLORATORY_SAME_COHORT_CALIBRATION',
        'script_sha256': sha(__file__),
        'dependencies_sha256': {
            name: sha(module.__file__) for name, module in (
                ('ml080', ml), ('ml088', m88),
                ('ml091', m91), ('ml092', m92))
        },
        'ml088_protocol_sha256': old_hash,
        'ml091_prior_plan_sha256': previous_hash,
        'seeds': SEEDS, 'arms': ARMS,
        'voices': VOICES, 'layers': LAYERS,
        'swaps': SWAPS,
        'n_sentences_per_voice': len(groups['passive']),
        'source_hashes': {
            v: digest([m91.sentence(g[0]) for g in groups[v]]) for v in VOICES
        },
        'prompt_hashes': {
            v: digest([ml.prompt(q) for g in groups[v] for q in g]) for v in VOICES
        },
        'intervention': 'Replace ONLY the answer-position attention Q vector at one layer with the Q vector from a different real question on the SAME sentence. Keep receiver K, V, residual and every other query native.',
        'question_order': ('first_agent','first_patient','second_agent','second_patient'),
        'predeclared_predictions': {
            'negative_control': 'L1 Q swap is exact no-op because all answer tokens and readout positions are identical before any attention.',
            'native_access_hypothesis': 'At some later layer, event-only Q swap preferentially shifts four-choice prediction toward the other event keeping role fixed; role-only swap preferentially shifts toward opposite role keeping event fixed.',
            'alternative': 'Q swaps may be ineffective or not factor-specific: earlier source integration, other attention sites, MLP/residual or general disruption may dominate.',
            'training': 'Contrast paired familiar/target seeds on the difficult passive second-event questions without conditioning on native correctness.',
        },
        'primary_descriptive': 'Per arm, voice and layer: event/role swap donor-gold logit margin change and donor-gold prediction gain; retain per-seed paired results. No success threshold.',
        'secondary': 'Native four-choice and full-vocab accuracy; wrong-event and wrong-role transitions; donor swap outcome by receiving first vs second query; both-switch control; patched-output L2 norm.',
        'warnings': 'Q donor can already contain answer-related context from earlier layers; sufficient to modify behavior, NOT proof of native necessity, independent event/role axes, or clean information-only transfer. Reused calibration cases and models. Four-choice is oracle-scored and full-vocab is separately reported.',
        'safety': 'No protected examples; no checkpoint modifications; exclusive report write; frozen source hashes.',
    }


def freeze(doc, create=False):
    expected = {**doc, 'sha256': digest(doc)}
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(expected, indent=2) + '\n')
    if not PLAN.exists():
        raise FileNotFoundError('Run --phase plan first')
    assert json.loads(PLAN.read_text()) == expected, 'Frozen ML-098 script/data/dependencies changed'
    return expected['sha256']


def attn_q_swap(block, x, readout, swap):
    """Exact standard MHA (dropout=0) except optional answer-position Q swap."""
    z = block.n1(x)
    b, length, width = z.shape
    h = block.attn.num_heads
    d = width // h
    assert width == h * d and b % 4 == 0
    ix = torch.arange(b, device=x.device)
    qkv = F.linear(z, block.attn.in_proj_weight, block.attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    def heads(y):
        return y.reshape(b, length, h, d).transpose(1, 2).contiguous()

    q, k, v = heads(q), heads(k), heads(v)
    if swap is not None:
        map4 = torch.tensor(SWAPS[swap], device=x.device, dtype=torch.long)
        owner = (ix // 4) * 4
        donor = owner + map4[ix % 4]
        assert bool((readout.reshape(-1, 4) == readout.reshape(-1, 4)[:, :1]).all())
        original_q = q.clone()
        q = q.clone()
        q[ix, :, readout] = original_q[donor, :, readout[donor]]

    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True)
    out = out.transpose(1, 2).contiguous().reshape(b, length, width)
    out = F.linear(out, block.attn.out_proj.weight, block.attn.out_proj.bias)
    return x + out + block.mlp(block.n2(x + out))


@torch.inference_mode()
def forward(model, groups, device, layer=None, swap=None):
    """Groups must be complete same-source four-question packets."""
    cases = [q for g in groups for q in g]
    ids, readout, _, _ = ml.encode(cases)
    ids, readout = ids.to(device), readout.to(device)
    b, length = ids.shape
    ix = torch.arange(b, device=device)
    x = model.embed(ids) + model.pos(torch.arange(length, device=device))
    mask = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
    for j, block in enumerate(model.blocks):
        if layer == j:
            x = attn_q_swap(block, x, readout, swap)
        else:
            x = block(x, mask)
    return model.out(model.norm(x[ix, readout])).float().cpu()


def outcomes(native_logits, edited_logits, groups, swap):
    """Gold index 0,1,2,3 identifies the requested relationship in each source."""
    n = len(groups)
    names = torch.tensor([m91.name_ids(g) for g in groups], dtype=torch.long)
    ids = names[:, None, :].expand(n, 4, 4).reshape(n * 4, 4)
    receiver = torch.arange(n * 4) % 4
    mapping = torch.tensor(SWAPS[swap], dtype=torch.long)
    donor = mapping[receiver]
    orig = native_logits.gather(1, ids)
    edited = edited_logits.gather(1, ids)
    native_pred = orig.argmax(1)
    pred = edited.argmax(1)
    idx = torch.arange(n * 4)
    def margin(z):
        return z[idx, donor] - z[idx, receiver]
    delta = margin(edited) - margin(orig)
    full_gold = ids[idx, receiver]
    full_donor = ids[idx, donor]
    answer = {
        'native_accuracy4': float((native_pred == receiver).float().mean()),
        'native_first_accuracy4': float((native_pred[receiver < 2] == receiver[receiver < 2]).float().mean()),
        'native_second_accuracy4': float((native_pred[receiver >= 2] == receiver[receiver >= 2]).float().mean()),
        'patched_accuracy4': float((pred == receiver).float().mean()),
        'patched_donor_gold_rate4': float((pred == donor).float().mean()),
        'native_donor_gold_rate4': float((native_pred == donor).float().mean()),
        'donor_gold_gain4': float(((pred == donor).float()-(native_pred == donor).float()).mean()),
        'new_donor_gold_rate4': float(((native_pred != donor) & (pred == donor)).float().mean()),
        'lost_donor_gold_rate4': float(((native_pred == donor) & (pred != donor)).float().mean()),
        'patched_full_vocab_accuracy': float((edited_logits.argmax(1) == full_gold).float().mean()),
        'patched_full_vocab_donor_gold': float((edited_logits.argmax(1) == full_donor).float().mean()),
        'donor_minus_receiver_logit_margin_change': float(delta.mean()),
        'prediction_changed_rate4': float((pred != native_pred).float().mean()),
        'mean_output_logit_delta_l2': float((edited_logits - native_logits).norm(dim=1).mean()),
    }
    for event, members in (('first', [0, 1]), ('second', [2, 3])):
        selector = (receiver == members[0]) | (receiver == members[1])
        answer[f'{event}_donor_gold_gain4'] = float(
            ((pred[selector] == donor[selector]).float() -
             (native_pred[selector] == donor[selector]).float()).mean()
        )
        answer[f'{event}_margin_change'] = float(delta[selector].mean())
    return answer


def bootstrap(paired, n=20000):
    rng = random.Random(98098)
    dist = sorted(mean(paired[rng.randrange(len(paired))] for _ in paired)
                  for _ in range(n))
    return [dist[int(.025*n)], dist[int(.975*n)-1]]


def preflight(groups):
    torch.manual_seed(98098)
    model = ml.Model().eval()
    sample = groups['passive'][:3]
    cases = [q for g in sample for q in g]
    ids, readout, _, _ = ml.encode(cases)
    direct = model(ids, readout).float()
    own = forward(model, sample, 'cpu')
    err = float((direct - own).abs().max())
    assert err < 3e-4, f'Native parity failed: {err}'
    for layer in LAYERS:
        same = forward(model, sample, 'cpu', layer, 'self')
        err2 = float((same - direct).abs().max())
        assert err2 < 3e-4, (layer, err2)
    for swap in ('event', 'role', 'both'):
        first = forward(model, sample, 'cpu', 0, swap)
        no_effect = float((first - direct).abs().max())
        assert no_effect < 3e-4, (swap, no_effect)
    # Later-layer swaps must be operational on synthetic weights.
    altered = forward(model, sample, 'cpu', 2, 'role')
    changed = float((altered - direct).abs().max())
    assert changed > 1e-6, ('Last-layer swap did nothing', changed)
    print('PREFLIGHT PASS: native parity, identity Q swaps, L1 structural null, L3 Q effect.')
    print('  native max error:', err, 'L3 synthetic output change:', changed)
    print('  No trained checkpoints or protected examples loaded.')


@torch.inference_mode()
def run(old_hash, groups, plan_hash, device):
    if REPORT.exists():
        raise FileExistsError(f'ML-098 report already exists: {REPORT}')
    pass_gate, counts = m88.model_gate(old_hash)
    if not pass_gate:
        raise RuntimeError(f'ML-088 model competence gate failed: {counts}')
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            chk = ROOT / f'seed_{seed}' / arm / 'model_step_12800.pt'
            if not chk.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {chk}')
            model = ml.load_checkpoint(chk, seed, arm, old_hash, device)
            metrics = {}
            for voice in VOICES:
                chunks = {label: [] for label in ['native'] +
                          [f'L{layer+1}/{swap}' for layer in LAYERS for swap in SWAPS]}
                for start in range(0, len(groups[voice]), BATCH_GROUPS):
                    subset = groups[voice][start:start + BATCH_GROUPS]
                    native = forward(model, subset, device)
                    chunks['native'].append(native)
                    for layer in LAYERS:
                        for swap in SWAPS:
                            updated = forward(model, subset, device, layer, swap)
                            chunks[f'L{layer+1}/{swap}'].append(updated)
                whole = {k: torch.cat(v) for k, v in chunks.items()}
                base = whole['native']
                metrics[voice] = {
                    'native': outcomes(base, base, groups[voice], 'self'),
                    'patched': {}
                }
                for layer in LAYERS:
                    for swap in SWAPS:
                        label = f'L{layer+1}/{swap}'
                        trial = outcomes(base, whole[label], groups[voice], swap)
                        if swap == 'self' or layer == 0:
                            assert trial['mean_output_logit_delta_l2'] < 0.01, (label, seed, arm, voice, trial['mean_output_logit_delta_l2'])
                        metrics[voice]['patched'][label] = trial
            rows.append({'seed': seed, 'arm': arm, 'metrics': metrics})
            p = metrics['passive']
            print(f'seed={seed} arm={arm:8s} native_second4={p["native"]["native_second_accuracy4"]:.3f} '
                  f'L3_event_donor_gain={p["patched"]["L3/event"]["second_donor_gold_gain4"]:+.3f} '
                  f'L3_role_donor_gain={p["patched"]["L3/role"]["second_donor_gold_gain4"]:+.3f}', flush=True)

    def mean_metric(arm, voice, layer, swap, key):
        return mean(r['metrics'][voice]['patched'][f'L{layer}/{swap}'][key]
                    for r in rows if r['arm'] == arm)

    summary = {}
    for arm in ARMS:
        for voice in VOICES:
            summary[f'{arm}/{voice}'] = {
                f'L{layer}': {
                    swap: {
                        key: mean_metric(arm, voice, layer, swap, key)
                        for key in ('donor_gold_gain4', 'new_donor_gold_rate4',
                                    'donor_minus_receiver_logit_margin_change',
                                    'second_donor_gold_gain4', 'second_margin_change',
                                    'prediction_changed_rate4', 'mean_output_logit_delta_l2')
                    } for swap in ('event', 'role', 'both')
                } for layer in (1, 2, 3)
            }

    paired = []
    for seed in SEEDS:
        a = next(r for r in rows if r['seed'] == seed and r['arm'] == 'familiar')
        b = next(r for r in rows if r['seed'] == seed and r['arm'] == 'target')
        # Primary EXPLORATORY paired difference of L3 event-selective donor margin.
        x = b['metrics']['passive']['patched']['L3/event']['second_margin_change']
        y = a['metrics']['passive']['patched']['L3/event']['second_margin_change']
        paired.append(x - y)
    summary['primary_exploratory'] = {
        'definition': 'Target minus familiar, L3 event-swap answer-Q donor-vs-receiver margin shift on passive SECOND-event questions',
        'by_seed': dict(zip(map(str, SEEDS), paired)),
        'mean': mean(paired), 'seed_bootstrap95': bootstrap(paired),
        'positive_seeds': sum(x > 0 for x in paired),
        'confirmatory': False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x') as f:
        json.dump({
            'experiment': 'ML-098', 'status': 'EXPLORATORY',
            'plan_sha256': plan_hash, 'ml088_protocol_sha256': old_hash,
            'protected_examples_generated': False,
            'n_checkpoints': len(rows),
            'summary': summary, 'rows': rows,
            'interpretation_warning': 'Answer-Q patch uses naturally obtained, question-conditioned donor state; donor can contain answer information; sufficiency not necessity; no new models/cases.'
        }, f, indent=2)
        f.write('\n')
    print('\nML-098 PRIMARY EXPLORATORY RESULT')
    print(json.dumps(summary['primary_exploratory'], indent=2))
    print('Saved:', REPORT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('preflight', 'plan', 'run'))
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'mps', 'cuda'))
    args = parser.parse_args()
    old_hash, groups, prior_hash = prepare()
    doc = specification(old_hash, groups, prior_hash)
    if args.phase == 'preflight':
        preflight(groups)
    elif args.phase == 'plan':
        print('Frozen plan SHA256:', freeze(doc, True))
        print('Review and commit the script and plan BEFORE viewing trained outputs.')
    else:
        run(old_hash, groups, freeze(doc), ml.select_device(args.device))


if __name__ == '__main__':
    main()
