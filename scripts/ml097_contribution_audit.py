"""ML-097: exploratory native attention-contribution audit after ML-092--096.

Use previously explored ML-092 calibration sentences and frozen ML-088 models.
NEVER generates protected examples, trains a decoder, or imports answer donors.

Run from repo root:
  python scripts/ml097_contribution_audit.py --phase preflight
  python scripts/ml097_contribution_audit.py --phase plan
  # Commit script + plan before reading trained checkpoints.
  python scripts/ml097_contribution_audit.py --phase run --device auto

Interventions use ORIGINAL, UNCHANGED native softmax weights.
For a selected attention receiver, subtract the contributions of source tokens
from the projected attention update. No softmax redistribution occurs.
A matched-mass condition removes the same total native attention mass from
the first/second name pairs, PER example/head, by scaling the subtractions.
This does not match V-vector geometry or guarantee a causal relationship circuit.
"""
import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

import ml080_learning_curve as ml
import ml088_probe_access as m88
import ml091_counterfactual_binding as m91
import ml092_source_state_transfer as m92

EXPERIMENT = 'ML-097'
ROOT = Path('results/ml088_fresh_models')
PLAN = Path('reports/ml097_contribution_audit/frozen_plan.json')
REPORT = Path('reports/ml097_contribution_audit/calibration_evaluation.json')
SEEDS = tuple(m88.SEEDS)
ARMS = ('familiar', 'target')
VOICES = ('active', 'passive')
VERSIONS = ('original', 'role_reversed')
SOURCES = ('first_pair', 'second_pair')
MODES = ('raw', 'mass_matched')
SITES = ((0, 'role'), (1, 'role'), (0, 'answer'),
         (1, 'answer'), (2, 'answer'))
PRIMARY_SITE = 'L2/answer'
BATCH_GROUPS = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare():
    # Important: m92.select_cases initializes the generator exactly once.
    protocol_hash, original, parent_plan = m92.select_cases()
    assert tuple(ml.SEEDS) == SEEDS and len(SEEDS) == 8
    groups = {}
    for voice in VOICES:
        assert len(original[voice]) == 96
        groups[(voice, 'original')] = original[voice]
        revised = []
        for g in original[voice]:
            assert len(g) == 4 and g[2].target_pos == 1
            swapped = replace(g[2], target_dir=1 - g[2].target_dir)
            qs = m91.four_questions(swapped)
            assert {tuple(ml.prompt(q)) for q in qs} != {tuple(ml.prompt(q)) for q in g}
            assert set(map(ml.answer, qs)) == set(map(ml.answer, g))
            # First event is held constant; second agent/patient exchange.
            assert (ml.answer(qs[0]), ml.answer(qs[1])) == (ml.answer(g[0]), ml.answer(g[1]))
            assert (ml.answer(qs[2]), ml.answer(qs[3])) == (ml.answer(g[3]), ml.answer(g[2]))
            revised.append(qs)
        groups[(voice, 'role_reversed')] = revised
    for key, blocks in groups.items():
        assert len(blocks) == 96, key
        for g in blocks:
            assert len({m91.sentence(q) for q in g}) == 1
            assert len({ml.answer(q) for q in g}) == 4
            assert tuple(q.target_pos for q in g) == (0, 0, 1, 1)
            assert tuple(q.role for q in g) == ('agent', 'patient', 'agent', 'patient')
    return protocol_hash, groups, parent_plan


def specification(protocol_hash, groups, parent_plan):
    return {
        'experiment': EXPERIMENT,
        'status': 'EXPLORATORY_EXISTING_MODELS_AND_CALIBRATION_FAMILY',
        'script_sha256': sha(__file__),
        'dependency_hashes': {name: sha(module.__file__) for name, module in
                              (('ml080', ml), ('ml088', m88), ('ml091', m91), ('ml092', m92))},
        'ml088_protocol_hash': protocol_hash,
        'ml091_parent_plan_hash': parent_plan,
        'models': 'ML-088 existing step-12800 paired seeds, 8 seeds x 2 arms',
        'sources_per_condition': 96,
        'example_hashes': {
            f'{v}/{r}': digest([ml.prompt(q) for g in groups[(v, r)] for q in g])
            for v in VOICES for r in VERSIONS
        },
        'sites': [f'L{k+1}/{dest}' for k, dest in SITES],
        'sources': SOURCES,
        'modes': MODES,
        'intervention': 'Subtract selected keys native weighted V contributions at one receiver and one layer, after out_proj, without renormalizing original attention weights.',
        'matched': 'At each example, receiver and head remove equal total attention mass from first/second pair: min(native mass of two pairs).',
        'primary': 'Per seed: target-minus-familiar effect on second-event passive role queries of mass-matched source-pair removal at L2/answer: [margin after first-pair removal minus margin after second-pair removal]. Average original and role-reversed versions before comparing arms.',
        'secondary': 'Raw removals, per-head native/removed attention mass, projected edit norm, first-event role queries, active/passive, reversed roles, all 5 sites, accuracy4/full vocabulary, seed heterogeneity, native margins.',
        'interpretation': 'Sensitivity to a source-token contribution, not proof of unique pathway, semantic representation, or mediation. Equal attention mass need not equal representation or output perturbation.',
        'selection': 'No post-hoc filtering on correctness, confidence, or signs; use identical prompts for paired seeds and arms.',
        'protected': 'Never instantiate or open ML-088 protected examples.',
        'one_run_lock': True,
    }


def freeze(document, create):
    expected = json.loads(json.dumps({**document, 'sha256': digest(document)}))
    if create and not PLAN.exists():
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(expected, indent=2) + '\n')
    if not PLAN.exists():
        raise FileNotFoundError('Run --phase plan first: ' + str(PLAN))
    if json.loads(PLAN.read_text()) != expected:
        raise RuntimeError('Frozen script, dependency, or examples differ. Stop; never silently overwrite.')
    return expected['sha256']


def source_sites(c):
    # Canonicalize to second-event query solely to locate source clauses.
    canonical = c if c.target_pos == 1 else m91.swap_queried_event(c)
    tokens = ml.prompt(c)
    sep = tokens.index('<sep>')
    def find(word):
        positions = [i for i, t in enumerate(tokens[:sep]) if t == word]
        assert len(positions) == 1, (word, positions)
        return positions[0]
    first = m92.participants(canonical, 'first')
    second = m92.participants(canonical, 'second')
    pair_a, pair_b = tuple(map(find, first)), tuple(map(find, second))
    role = [i for i in range(sep + 1, len(tokens)) if tokens[i] == c.role]
    assert len(role) == 1 and len(set(pair_a + pair_b)) == 4
    assert tokens[-1] == '<answer>'
    assert all(p < sep < role[0] < len(tokens)-1 for p in pair_a + pair_b)
    return {'first_pair': pair_a, 'second_pair': pair_b,
            'role': role[0], 'answer': len(tokens) - 1}


def remove_contribution(block, z, attention, weights, source_positions,
                        dest_positions, other_positions, mode):
    """Subtract exact native weighted V->out_proj contribution, no renorm.

    attention is output of PyTorch native MultiheadAttention with need_weights=True;
    weights are per-head ORIGINAL probabilities, before any intervention.
    Returns edited attention output and audit metrics.
    """
    b, length, width = z.shape
    heads = block.attn.num_heads
    assert width % heads == 0 and weights.shape == (b, heads, length, length)
    head_dim = width // heads
    assert block.attn.in_proj_weight is not None  # packed QKV weights
    bias = block.attn.in_proj_bias
    vb = bias[2 * width:] if bias is not None else None
    v = F.linear(z, block.attn.in_proj_weight[2 * width:], vb)
    v = v.reshape(b, length, heads, head_dim).transpose(1, 2)

    idx = torch.arange(b, device=z.device)
    dst_w = weights[idx, :, dest_positions, :]  # [batch, heads, keys]

    def extracted(keys):
        k = keys[:, None, :].expand(-1, heads, -1)
        w = dst_w.gather(2, k)
        vec = v.gather(2, k.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        return w, vec

    native_w, vectors = extracted(source_positions)
    other_w, _ = extracted(other_positions)
    native_mass = native_w.sum(-1)   # [batch, heads]
    other_mass = other_w.sum(-1)
    if mode == 'raw':
        fraction = torch.ones_like(native_mass)
    elif mode == 'mass_matched':
        shared = torch.minimum(native_mass, other_mass)
        fraction = torch.where(native_mass > 1e-12,
                               shared / native_mass.clamp_min(1e-12),
                               torch.zeros_like(native_mass))
    else:
        raise ValueError(mode)
    removed_mass = native_mass * fraction
    joint = (native_w * fraction.unsqueeze(-1)).unsqueeze(-1) * vectors
    contrib = joint.sum(dim=2).reshape(b, width)
    projected = F.linear(contrib, block.attn.out_proj.weight, bias=None)
    edited = attention.clone()
    edited[idx, dest_positions] -= projected
    return edited, {
        'native_attention_mass': float(native_mass.mean()),
        'other_attention_mass': float(other_mass.mean()),
        'removed_attention_mass': float(removed_mass.mean()),
        'projected_update_norm': float(projected.norm(dim=-1).mean()),
        'native_mass_by_head': native_mass.mean(0).detach().cpu().tolist(),
        'removed_mass_by_head': removed_mass.mean(0).detach().cpu().tolist(),
    }


@torch.inference_mode()
def forward(model, cases, device, *, site=None, source=None, mode=None):
    if site is None:
        assert source is None and mode is None
    else:
        assert (
            site in SITES or site == (2, 'role')
        ) and source in SOURCES and mode in MODES
    ids, readout, _, _ = ml.encode(cases)
    ids, readout = ids.to(device), readout.to(device)
    b, n = ids.shape
    ix = torch.arange(b, device=device)
    x = model.embed(ids) + model.pos(torch.arange(n, device=device))
    mask = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
    audit = None
    if site is not None:
        places = [source_sites(c) for c in cases]
        dst = torch.tensor([p[site[1]] for p in places], device=device)
        src = torch.tensor([p[source] for p in places], device=device)
        other = 'first_pair' if source == 'second_pair' else 'second_pair'
        alt = torch.tensor([p[other] for p in places], device=device)
        assert bool((src < dst[:, None]).all())
    for layer, block in enumerate(model.blocks):
        z = block.n1(x)
        a, weights = block.attn(z, z, z, attn_mask=mask,
                                 need_weights=True, average_attn_weights=False)
        if site is not None and layer == site[0]:
            a, audit = remove_contribution(block, z, a, weights, src, dst, alt, mode)
        x = x + a
        x = x + block.mlp(block.n2(x))
    logits = model.out(model.norm(x[ix, readout])).float().cpu()
    return logits, audit


def metrics(logits, groups):
    # Gold answers are used ONLY for scoring; not for selecting causal edges.
    results = m91.analyze_logits(logits, groups)
    z = logits.reshape(len(groups), 4, -1).float()
    names = torch.tensor([m91.name_ids(g) for g in groups], dtype=torch.long)
    choices = z.gather(2, names[:, None, :].expand(len(groups), 4, 4))
    q = torch.arange(4).view(1, 4).expand(len(groups), 4)
    gold = choices.gather(2, q[:, :, None]).squeeze(-1)
    wrong_event = choices.gather(2, (q ^ 2)[:, :, None]).squeeze(-1)
    wrong_role = choices.gather(2, (q ^ 1)[:, :, None]).squeeze(-1)
    acc = results['per_question_four_choice_accuracy']
    return {
        'first_accuracy4': mean(acc[k] for k in ('first_agent', 'first_patient')),
        'second_accuracy4': mean(acc[k] for k in ('second_agent', 'second_patient')),
        'accuracy4': results['four_choice_accuracy'],
        'full_vocab_accuracy': results['full_vocab_accuracy'],
        'first_event_margin': float((gold - wrong_event)[:, :2].mean()),
        'second_event_margin': float((gold - wrong_event)[:, 2:].mean()),
        'first_role_margin': float((gold - wrong_role)[:, :2].mean()),
        'second_role_margin': float((gold - wrong_role)[:, 2:].mean()),
        'wrong_event_rate4': results['wrong_event_rate'],
    }


def bootstrap(values, n=20000):
    rng = random.Random(97097)
    draws = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                   for _ in range(n))
    return [draws[int(.025*n)], draws[int(.975*n)-1]]


def preflight(groups):
    torch.manual_seed(97097)
    model = ml.Model().cpu().eval()
    cases = [q for g in groups[('passive', 'original')][:3] for q in g]
    ids, readout, _, _ = ml.encode(cases)
    reference = model(ids, readout).float()
    native, _ = forward(model, cases, 'cpu')
    identity_error = float((native - reference).abs().max())
    assert identity_error < 3e-4, ('native MHA parity', identity_error)
    first, stats_first = forward(model, cases, 'cpu',
                                 site=(1, 'answer'), source='first_pair', mode='mass_matched')
    second, stats_second = forward(model, cases, 'cpu',
                                   site=(1, 'answer'), source='second_pair', mode='mass_matched')
    match_error = abs(stats_first['removed_attention_mass'] - stats_second['removed_attention_mass'])
    assert match_error < 1e-6, ('mass match failure', match_error)
    changed = float(max((first-native).abs().max(), (second-native).abs().max()))
    assert changed > 1e-7, 'attention-contribution intervention unexpectedly inert'
    role_final, _ = forward(model, cases, 'cpu',
                            site=(2, 'role'), source='second_pair', mode='raw')
    negative_error = float((role_final-native).abs().max())
    assert negative_error < 3e-4, ('final-layer role control', negative_error)
    print(json.dumps({'preflight': 'PASS', 'native_parity_max_abs': identity_error,
                      'mass_matching_error': match_error,
                      'output_change': changed,
                      'final_layer_role_control_max_abs': negative_error,
                      'n_models': len(SEEDS)*len(ARMS),
                      'examples_per_voice_and_version': len(groups[('passive','original')]),
                      'protected_cases_generated': False}, indent=2))


def run(protocol_hash, groups, plan_hash, device):
    if REPORT.exists():
        raise FileExistsError('Experiment already run. Refuse overwrite: ' + str(REPORT))
    passes, counts = m88.model_gate(protocol_hash)
    if not passes:
        raise RuntimeError(f'ML-088 competence gate failed: {counts}')
    for seed in SEEDS:
        for arm in ARMS:
            path = ROOT / f'seed_{seed}' / arm / 'model_step_12800.pt'
            if not path.is_file():
                raise FileNotFoundError(str(path))
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            model = ml.load_checkpoint(ROOT / f'seed_{seed}' / arm / 'model_step_12800.pt',
                                       seed, arm, protocol_hash, device)
            entry = {'seed': seed, 'arm': arm, 'conditions': {}}
            for voice in VOICES:
                for version in VERSIONS:
                    key = f'{voice}/{version}'
                    samples = groups[(voice, version)]
                    labels = ['native'] + [f'L{k+1}/{dest}/{mode}/{source}'
                                           for k, dest in SITES for mode in MODES for source in SOURCES]
                    outputs = {label: [] for label in labels}
                    audits = {label: [] for label in labels if label != 'native'}
                    for st in range(0, len(samples), BATCH_GROUPS):
                        sub = samples[st:st+BATCH_GROUPS]
                        cases = [q for g in sub for q in g]
                        z, _ = forward(model, cases, device)
                        outputs['native'].append(z)
                        for layer, dest in SITES:
                            for mode in MODES:
                                for source in SOURCES:
                                    label = f'L{layer+1}/{dest}/{mode}/{source}'
                                    altered, audit = forward(model, cases, device,
                                                             site=(layer, dest), source=source, mode=mode)
                                    outputs[label].append(altered)
                                    audits[label].append(audit)
                    native_logits = torch.cat(outputs['native'])
                    condition = {'native': metrics(native_logits, samples)}
                    for label in labels[1:]:
                        z = torch.cat(outputs[label])
                        condition[label] = {
                            **metrics(z, samples),
                            'mean_output_logit_l2_change': float((z-native_logits).norm(dim=-1).mean()),
                            **{stat: mean(a[stat] for a in audits[label])
                               for stat in ('native_attention_mass', 'other_attention_mass',
                                            'removed_attention_mass', 'projected_update_norm')},
                            'native_mass_by_head': [mean(a['native_mass_by_head'][h] for a in audits[label])
                                                    for h in range(ml.HEADS)],
                            'removed_mass_by_head': [mean(a['removed_mass_by_head'][h] for a in audits[label])
                                                     for h in range(ml.HEADS)],
                        }
                    entry['conditions'][key] = condition
            rows.append(entry)
            p = entry['conditions']['passive/original']
            base = p['native']['second_accuracy4']
            sf = p[f'{PRIMARY_SITE}/mass_matched/first_pair']['second_event_margin']
            ss = p[f'{PRIMARY_SITE}/mass_matched/second_pair']['second_event_margin']
            print(f'seed={seed} arm={arm:8s} native_second4={base:.3f} '
                  f'matched_first_minus_second_margin={sf-ss:+.3f}', flush=True)

    index = {(r['seed'], r['arm']): r for r in rows}
    by_seed = {}
    by_condition = {}
    for seed in SEEDS:
        per_arm = {}
        for arm in ARMS:
            condition_diffs = []
            for version in VERSIONS:
                p = index[(seed, arm)]['conditions'][f'passive/{version}']
                f = p[f'{PRIMARY_SITE}/mass_matched/first_pair']['second_event_margin']
                s = p[f'{PRIMARY_SITE}/mass_matched/second_pair']['second_event_margin']
                condition_diffs.append(f-s)
            per_arm[arm] = mean(condition_diffs)
        by_seed[str(seed)] = per_arm['target'] - per_arm['familiar']
    values = list(by_seed.values())
    for arm in ARMS:
        for voice in VOICES:
            for version in VERSIONS:
                records = [index[(seed, arm)]['conditions'][f'{voice}/{version}'] for seed in SEEDS]
                def agg(path, field):
                    return mean(r[path][field] for r in records)
                label = f'{PRIMARY_SITE}/mass_matched/'
                by_condition[f'{arm}/{voice}/{version}'] = {
                    'native_first_accuracy4': agg('native', 'first_accuracy4'),
                    'native_second_accuracy4': agg('native', 'second_accuracy4'),
                    'native_full_vocab_accuracy': agg('native', 'full_vocab_accuracy'),
                    'native_second_event_margin': agg('native', 'second_event_margin'),
                    'after_first_pair_second_event_margin': agg(label+'first_pair','second_event_margin'),
                    'after_second_pair_second_event_margin': agg(label+'second_pair','second_event_margin'),
                    'first_pair_removed_mass': agg(label+'first_pair','removed_attention_mass'),
                    'second_pair_removed_mass': agg(label+'second_pair','removed_attention_mass'),
                    'first_pair_native_mass': agg(label+'first_pair','native_attention_mass'),
                    'second_pair_native_mass': agg(label+'second_pair','native_attention_mass'),
                    'first_pair_edit_norm': agg(label+'first_pair','projected_update_norm'),
                    'second_pair_edit_norm': agg(label+'second_pair','projected_update_norm'),
                }
    summary = {
        'primary_exploratory': {
            'definition': 'Across role-original and role-reversed, per seed: (margin after first pair minus margin after second pair, target arm) minus same in familiar arm, passive second-event questions, L2 answer receiver, mass-matched removal',
            'by_seed': by_seed,
            'mean': mean(values),
            'seed_bootstrap95': bootstrap(values),
            'positive_seeds': sum(x > 0 for x in values),
            'n_seed_pairs': len(SEEDS),
            'status': 'EXPLORATORY; no confirmatory success threshold',
        },
        'conditions': by_condition,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x') as file:
        json.dump({'experiment': EXPERIMENT, 'status': 'EXPLORATORY_CALIBRATION',
                   'plan_sha256': plan_hash, 'protected_generated': False,
                   'rows': rows, 'summary': summary,
                   'warning': 'No softmax redistribution but not a pure semantic lesion. Headwise mass matched, not V-vector geometry, output norm, baseline accuracy or model competence. Existing checkpoints and calibration family; causal pathway attribution remains provisional.'},
                  file, indent=2)
        file.write('\n')
    print('\nML-097 PRIMARY EXPLORATORY RESULT')
    print(json.dumps(summary['primary_exploratory'], indent=2))
    print('Saved:', REPORT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('preflight', 'plan', 'run'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    old_hash, groups, parent_plan = prepare()
    doc = specification(old_hash, groups, parent_plan)
    if args.phase == 'preflight':
        preflight(groups)
        print('Proposed plan SHA256:', digest(doc))
    elif args.phase == 'plan':
        print('Frozen plan SHA256:', freeze(doc, create=True))
        print('Review and commit script + frozen plan before inspecting trained-model output.')
    else:
        run(old_hash, groups, freeze(doc, create=False), ml.select_device(args.device))


if __name__ == '__main__':
    main()
