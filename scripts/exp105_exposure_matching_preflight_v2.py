#!/usr/bin/env python3
"""EXP-105 unfrozen design audit v2: passive-count matched PA vs AP exposure.

Read-only except --write to new reports/exp105/preflight/ outputs. Requires
exp105_generator_preflight.py (v0) and exp105_position_control_preflight_v1.py.
Only train/development name-set partitions are rendered. No model training,
reserved/protected example generation, checkpoint reads, or test evaluation.

From repository root:
  python3 scripts/exp105_exposure_matching_preflight_v2.py
  python3 scripts/exp105_exposure_matching_preflight_v2.py --write
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import exp105_generator_preflight as v0
import exp105_position_control_preflight_v1 as v1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/exp105/preflight'


def checksum(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def framed(rows):
    return [v1.expand(s, prefix) for s in rows for prefix in v1.FRAMES]


def dose_marginals(rows):
    """All four query labels are used; source surface only counted once."""
    return {
        'sources': len(rows),
        'labels': 4*len(rows),
        'answer_name_counts': dict(sorted(Counter(name for s in rows for name in s.gold).items())),
        'query_types': {str(i):len(rows) for i in range(4)},
        'modifier_cells': dict(sorted(Counter('/'.join(s.mods) for s in rows).items())),
        'verb_tokens': dict(sorted(Counter(word for s in rows for word in v0.tokens(s.text)
                                           if word in v0.VERBS).items())),
        'source_tokens': dict(sorted(Counter(word for s in rows for word in v0.tokens(s.text)).items())),
        'input_length_hist': dict(sorted(Counter(len(v0.tokens(s.text)) + len(v0.tokens(v0.Q[0])) + 3
                                                  for s in rows).items())),
    }


def positional_hist(rows):
    """Count source token/absolute-position occurrences, including frames."""
    count = Counter()
    for s in rows:
        for i, tok in enumerate(v0.tokens(s.text), start=1):
            count[(i,tok)] += 1
    return count


def report():
    assert tuple(len(v0.tokens(p)) if p else 0 for p in v1.FRAMES) == v1.EXPECTED_SHIFTS
    train_sets, dev_sets, partitions = v0.partitions()
    base, aa, ap, dev_fam, dev_li = v0.generate(train_sets, dev_sets)
    assert len(aa) == len(ap) == v0.TRAIN_NAME_SETS * len(v0.TRAIN_MODS)
    assert all(s.voice == 'AA' for s in aa)
    assert all(s.voice in ('AA','PA') for s in base)
    assert all(s.voice == 'AP' for s in ap)
    assert all('LI' not in s.mods for s in base + aa + ap)
    assert len(dev_li) == 1800
    assert all('LI' in s.mods[1] for s in dev_li)

    # Rebuild a PA exposure dose using precisely the same disjoint j==2 cores.
    # This changes the proposed unfrozen comparator, NOT the original v0/v1.
    pa = []
    for j, core_id, events in v0.cores(train_sets, 'train'):
        if j == 2:
            for mods in v0.TRAIN_MODS:
                pa.append(v0.make_source(events, 'PA', mods, 'dose', core_id))
    assert len(pa) == len(ap) == len(aa)
    for familiar, target in zip(pa, ap):
        assert (familiar.core_id, familiar.gold, familiar.mods) == (target.core_id, target.gold, target.mods)
        assert familiar.voice == 'PA' and target.voice == 'AP'
        assert familiar.text != target.text
        # Exactly one passive event in each. With I modifiers, moving passive
        # shifts punctuation and may change an individual pair's length;
        # only the *aggregate* token/length histograms should be matched.
        assert familiar.voice.count('P') == target.voice.count('P') == 1
        assert v0.parse_independent(familiar.text) == familiar.gold
        assert v0.parse_independent(target.text) == target.gold

    common = framed(base)
    dose = {k:framed(rows) for k,rows in [('AA',aa),('PA',pa),('AP',ap)]}
    assert len(common) == 54000
    assert all(len(rows)==13500 for rows in dose.values())
    profiles = {k:dose_marginals(rows) for k,rows in dose.items()}
    paired_equal_lengths = sum(len(v0.tokens(f.text)) == len(v0.tokens(t.text))
                               for f,t in zip(pa,ap))
    paired_equal_token_multisets = sum(Counter(v0.tokens(f.text)) == Counter(v0.tokens(t.text))
                                       for f,t in zip(pa,ap))
    match_keys = ['sources','labels','answer_name_counts','query_types','modifier_cells',
                  'verb_tokens','source_tokens','input_length_hist']
    for key in match_keys:
        assert profiles['PA'][key] == profiles['AP'][key], f'PA/AP mismatch: {key}'

    common_texts = {s.text for s in common}
    dev_texts = {s.text for s in dev_fam + dev_li}
    dev_features = v1.feature_pos(dev_li)
    by_arm = {}
    for k, rows in dose.items():
        combined = common + rows
        texts = {s.text for s in combined}
        assert len(texts) == len(combined) == 67500
        assert not texts & dev_texts
        assert all('LI' not in s.mods for s in combined)
        assert len(common_texts & {s.text for s in rows}) == 0
        feature = v1.feature_pos(combined)
        assert not (dev_features['exact'] - feature['exact']), f'Uncovered token/position for {k}'
        assert not (dev_features['lengths'] - feature['lengths'])
        assert not (dev_features['namepos'] - feature['namepos'])
        assert max(feature['lengths']) <= v0.MAX_TOKENS
        by_arm[k] = {
            'base_sources':len(common), 'exposure_sources':len(rows),
            'total_training_sources':len(combined),
            'passive_event_occurrences':sum(s.voice.count('P') for s in combined),
            'max_full_input_tokens':max(feature['lengths']),
            'missing_dev_LI_token_position_pairs':len(dev_features['exact'] - feature['exact']),
            'missing_dev_LI_input_lengths':len(dev_features['lengths'] - feature['lengths']),
            'unique_training_sources_sha256':v0.digest(sorted(texts)),
        }

    # PA and AP are equally informative about the *existence* of passive voice,
    # but their passive-event position differs intentionally. Position-specific
    # distributions therefore cannot (and should not) be declared matched.
    pos_pa = positional_hist(dose['PA']); pos_ap = positional_hist(dose['AP'])
    different_keys = sum(pos_pa[p] != pos_ap[p] for p in (pos_pa.keys() | pos_ap.keys()))
    positional_L1 = sum(abs(pos_pa[p]-pos_ap[p]) for p in (pos_pa.keys() | pos_ap.keys()))
    assert different_keys > 0 and positional_L1 > 0
    assert by_arm['PA']['passive_event_occurrences'] == by_arm['AP']['passive_event_occurrences']
    assert by_arm['AA']['passive_event_occurrences'] < by_arm['PA']['passive_event_occurrences']
    return {
        'study':'EXP-105', 'phase':'EXPOSURE_MATCHING_PREFLIGHT_V2',
        'status':'UNFROZEN_DESIGN_ONLY',
        'no_training_or_protected_generation':True,
        'dependency_sha256':{
            'v0':checksum(v0.__file__), 'v1':checksum(v1.__file__), 'v2':checksum(__file__),
        },
        'partition_digest':partitions['partition_digest'],
        'selected_primary_exposure_contrast':'PA vs AP',
        'passive_count_matched':True,
        'paired_source_count_unframed':len(pa),
        'paired_source_equal_length_count_unframed':paired_equal_lengths,
        'paired_source_equal_token_multiset_count_unframed':paired_equal_token_multisets,
        'paired_source_count_framed':len(dose['PA']),
        'frame_offsets':[len(v0.tokens(p)) if p else 0 for p in v1.FRAMES],
        'common_base_framed_sources':len(common),
        'dev_LI_sources':len(dev_li),
        'dev_LI_distinct_token_positions':len(dev_features['exact']),
        'per_arm':by_arm,
        'matched_marginal_keys':match_keys,
        'PA_AP_source_position_hist_differing_keys':different_keys,
        'PA_AP_source_position_hist_L1_count_difference':positional_L1,
        'AA_control_status':'optional additional baseline; AA changes passive exposure count',
        'caveats':[
            'PA vs AP matches passive counts, aggregate input-length distribution, aggregate lexical counts and gold labels. Individual I-modifier pairs can differ in punctuation and length.',
            'Token-by-position occurrence frequencies still differ; full support coverage is not contextual or causal equality.',
            'A training runner must predeclare batch source/voice/modifier/position scheduling, not infer matching from aggregate pools alone.',
            'Only dev name-set compositions were generated. Existing reserved identifiers were never rendered.',
            'The design uses an existing synthetic grammar family; independent grammar replication remains a separate goal.',
            'This is not a frozen confirmatory protocol and gives no permission to train or open protected evaluations.',
        ],
    }


def markdown(r):
    lines = [
        '# EXP-105 exposure matching — development-only preflight v2', '',
        '**Status: unfrozen design audit. No training, reserved examples, or protected evaluation.**', '',
        '**Proposed primary contrast: PA (first-event passive) vs AP (second-event passive).**',
        'The earlier AA-vs-AP contrast additionally changed the number of passive events.', '',
        f"Shared framed base sources: {r['common_base_framed_sources']}; ",
        f"per-arm framed dose sources: {r['paired_source_count_framed']}; ",
        f"development L+I sources: {r['dev_LI_sources']}.", '',
        '| Training arm | Shared base | Exposure dose | Total training sources | Passive event occurrences | Missing L+I token-positions | Max input tokens |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ('AA','PA','AP'):
        a = r['per_arm'][arm]
        lines.append(f"| {arm} | {a['base_sources']} | {a['exposure_sources']} | {a['total_training_sources']} | {a['passive_event_occurrences']} | {a['missing_dev_LI_token_position_pairs']} | {a['max_full_input_tokens']} |")
    lines += [
        '', '**PASS:** PA and AP exposure doses have identical paired cores, gold answers, modifier frequencies,',
        'question counts, verb frequencies, source token counts and full-input-length histograms.',
        'Both retain zero missing development L+I token/absolute-position pairs.', '',
        f"Individual paired sources of equal length: {r['paired_source_equal_length_count_unframed']}/{r['paired_source_count_unframed']} (modifiers change comma placement).",
        f"Individual paired sources with identical token multisets: {r['paired_source_equal_token_multiset_count_unframed']}/{r['paired_source_count_unframed']}.", '',
        f"**Still different by design:** PA vs AP source token-position histograms differ at {r['PA_AP_source_position_hist_differing_keys']} token/position keys",
        f"(L1 count difference {r['PA_AP_source_position_hist_L1_count_difference']}).", '',
        '## Limits and next action',
        *[f'- {x}' for x in r['caveats']], '',
        '**Next:** write and review an explicit source/label sampling schedule for a disposable',
        'development-only training smoke test. Do not freeze or start confirmatory training yet.', ''
    ]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', help='Write only new v2 audit files; refuse overwrite')
    a=p.parse_args()
    r=report()
    print(markdown(r), flush=True)
    if a.write:
        OUT.mkdir(parents=True,exist_ok=True)
        paths = (OUT/'exposure_matching_preflight_v2.md',OUT/'exposure_matching_preflight_v2.json')
        if any(x.exists() for x in paths):
            raise FileExistsError('v2 audit files already exist; refusing overwrite')
        paths[0].write_text(markdown(r),encoding='utf-8')
        paths[1].write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        for path in paths: print('SAVED',path)

if __name__=='__main__': main()
