#!/usr/bin/env python3
"""EXP-105 DESIGN-v0 generator and exposure-matching audit (development only).

Does NOT train/evaluate models, read earlier EXP-102 splits, materialize
reserved/protected examples, or unlock any protected test.

Usage from repository root:
    python3 scripts/exp105_generator_preflight.py
    python3 scripts/exp105_generator_preflight.py --write

This is a preflight *proposal*, NOT the frozen confirmatory protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from random import Random
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/exp105/preflight'
NAMES = ('Alda', 'Berto', 'Cira', 'Dalen', 'Emina', 'Faro',
         'Gilda', 'Hilo', 'Isma', 'Jorin', 'Kavi', 'Lena')
VERBS = ('admired', 'greeted', 'followed', 'thanked')
Q = tuple(f'In the {event} event, who was the {role}?'
          for event in ('first', 'second') for role in ('agent', 'patient'))
VOICES = ('AA', 'PA', 'AP')
# Nine trained modifier combinations, never LI within an event.
TRAIN_MODS = tuple(product(('bare', 'L', 'I'), repeat=2))
LI_MODS = (('bare', 'LI'), ('L', 'LI'), ('I', 'LI'))
SPLIT_SEED = 105001
CORE_SEED = 105002
TRAIN_NAME_SETS = 300
DEV_NAME_SETS = 100
# Other 95 four-name sets are identifier-only reserved metadata; no text made.
MAX_TOKENS = 48
TOKEN_RE = re.compile(r"'s|[A-Za-z]+|[;,.?]")
NAME_SET = frozenset(NAMES)


@dataclass(frozen=True)
class Event:
    agent: str
    patient: str
    verb: str
    voice: str
    gate_owner: str
    near_person: str
    layout: int


@dataclass(frozen=True)
class Source:
    text: str
    gold: tuple[str, str, str, str]
    voice: str
    mods: tuple[str, str]
    group: str
    core_id: str


def tokens(s: str) -> list[str]:
    result = TOKEN_RE.findall(s)
    assert ''.join(result).replace("'", '') != ''
    return result


def digest(x) -> str:
    return sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_event(e: Event, mods: str) -> str:
    assert mods in {'bare', 'L', 'I', 'LI'}
    assert e.voice in {'A', 'P'}
    if e.voice == 'A':
        text = (f'{e.agent}, near {e.near_person}, {e.verb} {e.patient}'
                if 'I' in mods else f'{e.agent} {e.verb} {e.patient}')
    else:
        text = (f'{e.patient} was {e.verb} near {e.near_person} by {e.agent}'
                if 'I' in mods else f'{e.patient} was {e.verb} by {e.agent}')
    if 'L' in mods:
        text = (f"By {e.gate_owner}'s gate, {text}" if e.layout == 0
                else f"{text} beside {e.gate_owner}'s gate")
    return text


def render(events: tuple[Event, Event], mods: tuple[str, str]) -> str:
    return ' ; '.join(render_event(e, mod) for e, mod in zip(events, mods)) + '.'


def parse_independent(surface: str) -> tuple[str, str, str, str]:
    """Source-only oracle: no gold metadata passed to parser."""
    assert surface.endswith('.')
    clauses = surface[:-1].split(' ; ')
    assert len(clauses) == 2
    pairs = []
    for clause in clauses:
        s = clause
        s = re.sub(r"^By [A-Z][a-z]+'s gate, ", '', s)
        s = re.sub(r" beside [A-Z][a-z]+'s gate$", '', s)
        s = re.sub(r", near [A-Z][a-z]+, ", ' ', s)
        s = re.sub(r" near [A-Z][a-z]+ by ", ' by ', s)
        passive = re.fullmatch(r"([A-Z][a-z]+) was (\w+) by ([A-Z][a-z]+)", s)
        active = re.fullmatch(r"([A-Z][a-z]+) (\w+) ([A-Z][a-z]+)", s)
        if passive:
            patient, verb, agent = passive.groups()
        elif active:
            agent, verb, patient = active.groups()
        else:
            raise AssertionError(f'Cannot independently parse: {clause!r} as {s!r}')
        assert verb in VERBS and agent != patient
        assert agent in NAME_SET and patient in NAME_SET
        pairs.extend((agent, patient))
    return tuple(pairs)


def partitions():
    name_sets = list(combinations(NAMES, 4))
    assert len(name_sets) == 495
    Random(SPLIT_SEED).shuffle(name_sets)
    train = name_sets[:TRAIN_NAME_SETS]
    dev = name_sets[TRAIN_NAME_SETS:TRAIN_NAME_SETS + DEV_NAME_SETS]
    reserved = name_sets[TRAIN_NAME_SETS + DEV_NAME_SETS:]
    assert len(reserved) == 95
    assert len(set(train) & set(dev)) == 0
    assert len(set(train) & set(reserved)) == 0
    assert len(set(dev) & set(reserved)) == 0
    # Reserved four-name identifiers are not used to make Events/Source/text.
    return train, dev, {'all_set_count': len(name_sets), 'train_sets': len(train),
                        'dev_sets': len(dev), 'reserved_identifier_only_sets': len(reserved),
                        'partition_digest': digest({'train': train, 'dev': dev, 'reserved': reserved}),
                        'train_digest': digest(train), 'dev_digest': digest(dev),
                        'reserved_identifier_digest': digest(reserved)}


def cores(name_sets, group: str):
    """Disjoint source cores for shared base vs intervention dose."""
    for i, name_set in enumerate(name_sets):
        rng = Random(CORE_SEED + i + (100_000 if group == 'dev' else 0))
        names = list(name_set)
        perms = set()
        while len(perms) < (3 if group == 'train' else 2):
            perms.add(tuple(rng.sample(names, 4)))
        orderings = sorted(perms)
        for j, (a, b, c, d) in enumerate(orderings):
            v0 = VERBS[(i + j) % len(VERBS)]
            v1 = VERBS[(i + j + 1) % len(VERBS)]
            assert v0 != v1
            core_id = f'{group}-{i}-{j}'
            # Modifiers use the other event's participants (distractors).
            events = (Event(a, b, v0, 'A', c, d, (i + j) % 2),
                      Event(c, d, v1, 'A', a, b, (i + j + 1) % 2))
            yield j, core_id, events


def voiced(events, voice: str):
    return (replace(events[0], voice=voice[0]),
            replace(events[1], voice=voice[1]))


def make_source(events, voice, mods, group, core_id):
    rendered = render(voiced(events, voice), mods)
    gold = tuple(x for e in events for x in (e.agent, e.patient))
    assert len(set(gold)) == 4
    assert parse_independent(rendered) == gold
    return Source(rendered, gold, voice, mods, group, core_id)


def generate(train_sets, dev_sets):
    base, dose_familiar, dose_target, dev_familiar, dev_composed = [], [], [], [], []
    for j, core_id, events in cores(train_sets, 'train'):
        for mods in TRAIN_MODS:
            if j in (0, 1):
                for voice in ('AA', 'PA'):
                    base.append(make_source(events, voice, mods, 'base', core_id))
            else:
                dose_familiar.append(make_source(events, 'AA', mods, 'dose', core_id))
                dose_target.append(make_source(events, 'AP', mods, 'dose', core_id))
    for j, core_id, events in cores(dev_sets, 'dev'):
        for voice in VOICES:
            for mods in TRAIN_MODS:
                dev_familiar.append(make_source(events, voice, mods, 'dev_familiar', core_id))
            # New development-only LI examples, from dev name sets, not protected.
            for mods in LI_MODS:
                dev_composed.append(make_source(events, voice, mods, 'dev_LI', core_id))
    return base, dose_familiar, dose_target, dev_familiar, dev_composed


def validate_transformations(records):
    checked = 0
    for s in records:
        # Infer first/second events from independent parser, then reconstruct.
        a, b, c, d = s.gold
        # Even when modifiers are reordered, labels must match independent text parse.
        assert parse_independent(s.text) == (a, b, c, d)
        # Semantic invariances: swap independent event subclauses swaps answer pairs.
        left, right = s.text[:-1].split(' ; ')
        assert parse_independent(right + ' ; ' + left + '.') == (c, d, a, b)
        # Switching each event's voice does not swap its agent/patient.
        # Build a fresh source with same participants and same modifier assignments.
        # Name-only reconstruction is enough for this invariance, verbs don't matter.
        es = (Event(a, b, VERBS[0], s.voice[0], c, d, 0),
              Event(c, d, VERBS[1], s.voice[1], a, b, 1))
        both_flip = voiced(es, ''.join('P' if v == 'A' else 'A' for v in s.voice))
        assert parse_independent(render(both_flip, s.mods)) == s.gold
        role_swap = (replace(es[0], agent=b, patient=a), es[1])
        assert parse_independent(render(role_swap, s.mods)) == (b, a, c, d)
        checked += 1
    return checked


def audit_training(base, familiar, target):
    assert len(base) == TRAIN_NAME_SETS * 2 * 2 * len(TRAIN_MODS)
    assert len(familiar) == len(target) == TRAIN_NAME_SETS * len(TRAIN_MODS)
    assert {x.core_id for x in base}.isdisjoint({x.core_id for x in familiar})
    assert len({x.text for x in base}) == len(base)
    assert len({x.text for x in familiar}) == len(familiar)
    assert len({x.text for x in target}) == len(target)
    assert not {x.text for x in base}.intersection(x.text for x in familiar)
    assert not {x.text for x in base}.intersection(x.text for x in target)
    assert not ({x.text for x in familiar} & {x.text for x in target})
    for f, t in zip(familiar, target):
        assert f.core_id == t.core_id and f.mods == t.mods
        assert f.gold == t.gold and f.voice == 'AA' and t.voice == 'AP'
        # The only assigned difference is the second event's voice (surface changes).
        f_first, f_second = f.text[:-1].split(' ; ')
        t_first, t_second = t.text[:-1].split(' ; ')
        assert f_first == t_first and f_second != t_second
        assert not any('LI' in m for m in f.mods + t.mods)
    for record in base:
        assert record.voice in ('AA', 'PA') and 'LI' not in record.mods
    for record in target:
        assert record.voice == 'AP'
    for record in familiar:
        assert record.voice == 'AA'
    assert not any('LI' in m for s in base + familiar + target for m in s.mods)

    def margin(rows):
        return {
            'sources': len(rows), 'labels': 4 * len(rows),
            'query_types': dict(sorted(Counter(q for _ in rows for q in range(4)).items())),
            'label_names': dict(sorted(Counter(answer for s in rows for answer in s.gold).items())),
            'modifier_cells': dict(sorted(Counter('/'.join(s.mods) for s in rows).items())),
            'voice_cells': dict(sorted(Counter(s.voice for s in rows).items())),
            'verbs': dict(sorted(Counter(tok for s in rows for tok in tokens(s.text) if tok in VERBS).items())),
        }
    f = margin(familiar)
    t = margin(target)
    for key in ('sources', 'labels', 'query_types', 'label_names', 'modifier_cells', 'verbs'):
        assert f[key] == t[key], f'Mismatched {key}'
    return {'common_base': margin(base), 'familiar_dose': f, 'target_dose': t,
            'strictly_matched_marginals': ['sources', 'labels', 'query_types', 'label_names',
                                           'modifier_cells', 'verbs'],
            'not_matched_by_design': ['second-event surface voice', 'sequence length',
                                      'surface token positions', 'voice-frequency marginals'],
            'dose_source_pairs': len(familiar)}


def feature_report(training, dev_LI):
    def input_tokens(s, q):
        return ['<bos>'] + tokens(s.text) + ['<sep>'] + tokens(Q[q]) + ['<answer>']
    def coverage(rows):
        len_hist = Counter()
        lexical = set()
        positional = set()
        keyword_pos = set()
        answer_name_positions = set()
        token_pos = set()
        max_len = 0
        for s in rows:
            source_toks = tokens(s.text)
            for pos, t in enumerate(source_toks, 1):
                if t in NAME_SET:
                    answer_name_positions.add(pos)
            for qi in range(4):
                seq = input_tokens(s, qi)
                len_hist[len(seq)] += 1
                max_len = max(max_len, len(seq))
                for pos, tok in enumerate(seq):
                    positional.add((pos, 'name' if tok in NAME_SET else tok))
                    token_pos.add((pos, tok))
                    lexical.add(tok)
                    if tok in ('was', 'by', 'By', 'near', 'beside', 'agent', 'patient'):
                        keyword_pos.add((pos, tok))
        return {'length_histogram': dict(sorted(len_hist.items())), 'vocab': lexical,
                'positional': positional, 'keyword_pos': keyword_pos,
                'answer_name_positions': answer_name_positions, 'token_pos': token_pos,
                'max_tokens': max_len}
    tr = coverage(training)
    dv = coverage(dev_LI)
    missing_vocab = sorted(dv['vocab'] - tr['vocab'])
    missing_len = sorted(set(dv['length_histogram']) - set(tr['length_histogram']))
    missing_name_pos = sorted(dv['answer_name_positions'] - tr['answer_name_positions'])
    missing_keyword = sorted(dv['keyword_pos'] - tr['keyword_pos'])
    missing_pos_types = sorted(dv['positional'] - tr['positional'])
    missing_exact_token_pos = dv['token_pos'] - tr['token_pos']
    assert not missing_vocab, f'Development vocabulary unseen in training: {missing_vocab}'
    assert tr['max_tokens'] <= MAX_TOKENS and dv['max_tokens'] <= MAX_TOKENS, 'Increase max length before freeze'
    return {'max_input_tokens_training': tr['max_tokens'],
            'max_input_tokens_dev_LI': dv['max_tokens'],
            'train_length_histogram': tr['length_histogram'],
            'dev_LI_length_histogram': dv['length_histogram'],
            'new_dev_LI_lengths': missing_len,
            'new_dev_LI_name_positions': missing_name_pos,
            'new_dev_LI_keyword_positions': [list(p) for p in missing_keyword],
            'new_dev_LI_position_token_types': [list(p) for p in missing_pos_types],
            'number_unseen_exact_token_position_pairs': len(missing_exact_token_pos),
            'dev_LI_exact_token_position_pairs': len(dv['token_pos']),
            'vocabulary_transfer': 'PASS',
            'structural_position_coverage': ('PASS' if not (missing_len or missing_name_pos or missing_keyword) else 'WARNING')}


def shortcut_report(rows):
    scores = Counter()
    for s in rows:
        gold = s.gold
        name_mentions = [t for t in tokens(s.text) if t in NAME_SET]
        unique_in_surface = tuple(dict.fromkeys(name_mentions))
        assert len(unique_in_surface) == 4
        freq = Counter(name_mentions)
        most_common = min(NAMES, key=lambda x: (-freq[x], NAMES.index(x)))
        first_clause, second_clause = s.text[:-1].split(' ; ')
        first_two = tuple(dict.fromkeys(t for t in tokens(first_clause) if t in NAME_SET))[:2]
        second_two = tuple(dict.fromkeys(t for t in tokens(second_clause) if t in NAME_SET))[:2]
        predictions = {
            'role_blind': (gold[0], gold[0], gold[2], gold[2]),
            'event_blind': (gold[0], gold[1], gold[0], gold[1]),
            'most_frequent_name': (most_common,) * 4,
            'surface_first_four_unique': unique_in_surface,
            'first_two_names_each_clause': first_two + second_two,
        }
        for k, p in predictions.items():
            scores[k] += tuple(p) == gold
    rate = {k: round(n / len(rows), 6) for k, n in sorted(scores.items())}
    assert rate['role_blind'] == rate['event_blind'] == rate['most_frequent_name'] == 0
    assert rate['surface_first_four_unique'] < 0.5
    assert rate['first_two_names_each_clause'] < 0.5
    return {'stratum': 'development-only AP with second-event LI',
            'source_count': len(rows), 'C_query_shortcuts': rate,
            'independent_parser_oracle_C_query': 1.0}


def audit():
    assert len(Q) == 4
    train_sets, dev_sets, partition = partitions()
    base, familiar, target, dev_fam, dev_LI = generate(train_sets, dev_sets)
    train_report = audit_training(base, familiar, target)
    assert {s.core_id for s in dev_fam + dev_LI}.isdisjoint({s.core_id for s in base + familiar})
    assert not {s.text for s in dev_fam + dev_LI} & {s.text for s in base + familiar + target}
    assert all('LI' not in s.mods for s in dev_fam)
    assert all('LI' in s.mods[1] for s in dev_LI)
    assert Counter(s.voice for s in dev_LI) == {v: DEV_NAME_SETS * 2 * len(LI_MODS) for v in VOICES}
    assert Counter(s.voice for s in dev_fam) == {v: DEV_NAME_SETS * 2 * len(TRAIN_MODS) for v in VOICES}
    # Independent transformation checks across all *development* examples and
    # a deterministic sample of the training sources (no reserved content).
    transform_sample = dev_fam + dev_LI + base[::19] + target[::11]
    transforms = validate_transformations(transform_sample)
    positional = feature_report(base + target, dev_LI)
    ap_LI = [s for s in dev_LI if s.voice == 'AP']
    baselines = shortcut_report(ap_LI)
    report = {
        'study': 'EXP-105 design-v0', 'phase': 'GENERATOR_PREFLIGHT_ONLY',
        'status': 'NOT_FROZEN_NOT_CONFIRMATORY',
        'no_training_or_protected_generation': True,
        'split': partition,
        'train': train_report,
        'development': {'familiar_sources': len(dev_fam), 'LI_sources': len(dev_LI),
                        'AP_LI_sources': len(ap_LI), 'voice_cells': list(VOICES),
                        'LI_second_event_only': True},
        'transformations_checked': transforms,
        'position_coverage': positional,
        'shortcut_baselines': baselines,
        'warnings': [
            'This is a *design* diagnostic; no exposure effect has been measured.',
            'New L+I sequences were generated only from DEV name sets, not reserved sets.',
            'Dose AA vs AP changes token lengths and positions. Residual confounding must be addressed or disclosed.',
            'The task reuses the conceptual EXP-102 grammar; new name partitions alone do not constitute independent grammar replication.',
            'C_query shortcuts checked are limited baselines, not proof against every possible shortcut.',
            'Position coverage warning is NOT a pass to freeze a confirmatory study.',
        ],
    }
    return report


def markdown(r):
    p=r['position_coverage']; b=r['shortcut_baselines']
    t=r['train']
    return '\n'.join([
        '# EXP-105 generator preflight — DESIGN v0',
        '',
        '**Not frozen. No model training or protected example generation.**',
        '',
        f"- Name sets: {r['split']['train_sets']} train, {r['split']['dev_sets']} dev, {r['split']['reserved_identifier_only_sets']} reserved identifiers only.",
        f"- Common base: {t['common_base']['sources']} source sentences (AA/PA only).",
        f"- Familiar dose: {t['familiar_dose']['sources']} AA sources; target dose: {t['target_dose']['sources']} AP sources.",
        f"- Pairwise matched dose sources: {t['dose_source_pairs']} (same semantic cores, modifiers and four gold labels).",
        f"- Development familiar sources: {r['development']['familiar_sources']}; development LI sources: {r['development']['LI_sources']}, including {r['development']['AP_LI_sources']} AP+LI sources.",
        f"- Parser and counterfactual transformations checked: {r['transformations_checked']} sources.",
        '',
        '## Coverage and potential confounds',
        f"- Maximum input tokens: train={p['max_input_tokens_training']}; dev-LI={p['max_input_tokens_dev_LI']} (limit={MAX_TOKENS}).",
        f"- New dev-LI total lengths: {p['new_dev_LI_lengths'] or 'none'}.",
        f"- New dev-LI name positions: {p['new_dev_LI_name_positions'] or 'none'}.",
        f"- New dev-LI structural token positions: {p['new_dev_LI_keyword_positions'] or 'none'}.",
        f"- New dev-LI position/type combinations: {len(p['new_dev_LI_position_token_types'])}.",
        f"- Unseen exact token-position pairs: {p['number_unseen_exact_token_position_pairs']} / {p['dev_LI_exact_token_position_pairs']}.",
        f"- Vocabulary transfer: {p['vocabulary_transfer']}; structural position coverage: **{p['structural_position_coverage']}**.",
        '',
        '## Shortcut baselines on AP+LI development',
        '| Baseline | Complete-query accuracy |', '|---|---:|',
        *[f'| {k} | {v:.1%} |' for k,v in b['C_query_shortcuts'].items()],
        f"| Parser oracle | {b['independent_parser_oracle_C_query']:.1%} |",
        '',
        '## Outstanding limitations',
        *[f'- {w}' for w in r['warnings']],
        '',
        'Next: review coverage and semantics **before** freezing a split or implementing training.',
        ''
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='Write new derived report files; never overwrite')
    args = parser.parse_args()
    r = audit()
    m = markdown(r)
    print(m, flush=True)
    print('PASS: generator, split, dose matching, labels, transformations, shortcuts, and vocabulary')
    print('DESIGN-ONLY: inspect positional warnings before proceeding', flush=True)
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        paths = (OUT/'generator_preflight_v0.md', OUT/'generator_preflight_v0.json')
        if any(p.exists() for p in paths):
            raise FileExistsError('Refusing to overwrite preflight results; inspect existing files')
        paths[0].write_text(m, encoding='utf-8')
        paths[1].write_text(json.dumps(r, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        for p in paths:
            print(f'SAVED {p}')


if __name__ == '__main__':
    main()
