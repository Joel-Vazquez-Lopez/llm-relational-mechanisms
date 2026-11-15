#!/usr/bin/env python3
"""EXP-105 DESIGN-v1: positional-coverage audit using neutral temporal framing.

Requires scripts/exp105_generator_preflight.py (v0). Uses fresh train/dev
name-set generators ONLY. Reserved partition identifiers are counted by the
v0 dependency but are never rendered into sources or inspected as sentences.

This file does NOT modify v0, train models, open protected evaluation,
freeze a protocol, or assert a successful transfer result.

Run from repository root:
    python3 scripts/exp105_position_control_preflight_v1.py
    python3 scripts/exp105_position_control_preflight_v1.py --write
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import exp105_generator_preflight as prior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/exp105/preflight'

# Neutral sentence-level time frames; comma included in listed token counts.
# These are not event modifiers, have no participant names or target verbs,
# and may occur in common base and in either exposure dose. The held-out
# in-event 'LI' modification remains absent from all training sources.
FRAMES = (
    '',  # 0 tokens
    'On that day, ',  # 4 tokens
    'Earlier on that particular day, ',  # 6 tokens
    'Earlier on that unusually peaceful summer day, ',  # 8 tokens
    'Earlier on that unusually peaceful and quiet summer day, ',  # 10 tokens
)
EXPECTED_SHIFTS = (0, 4, 6, 8, 10)


def unframe(text: str) -> str:
    """Remove only a fully recognized sentence-level frame before parsing."""
    for p in sorted((p for p in FRAMES if p), key=len, reverse=True):
        if text.startswith(p):
            return text[len(p):]
    return text


def expand(source: prior.Source, prefix: str) -> prior.Source:
    assert prefix in FRAMES
    framed = prior.Source(
        prefix + source.text, source.gold, source.voice, source.mods,
        source.group, source.core_id
    )
    assert prior.parse_independent(unframe(framed.text)) == source.gold
    assert unframe(framed.text) == source.text
    # The frame must not introduce any event participant or target verb.
    prefix_words = set(prior.tokens(prefix)) if prefix else set()
    assert not (prefix_words & (prior.NAME_SET | set(prior.VERBS)))
    return framed


def feature_pos(rows):
    """Exact absolute token/position pairs in full training-style prompts."""
    lengths = set()
    exact = set()
    structural = set()
    namepos = set()
    names = prior.NAME_SET
    important = {'was', 'by', 'By', 'near', 'beside', 'agent', 'patient'}
    for s in rows:
        source_tokens = prior.tokens(s.text)
        for i, word in enumerate(source_tokens, 1):
            if word in names:
                namepos.add(i)
        for q in prior.Q:
            seq = ['<bos>'] + source_tokens + ['<sep>'] + prior.tokens(q) + ['<answer>']
            lengths.add(len(seq))
            for i, word in enumerate(seq):
                exact.add((i, word))
                if word in important:
                    structural.add((i, word))
    return {'lengths': lengths, 'exact': exact, 'structural': structural, 'namepos': namepos}


def pct(n, denom):
    return round(100*n/denom, 3) if denom else 100.0


def report():
    assert tuple(len(prior.tokens(p)) if p else 0 for p in FRAMES) == EXPECTED_SHIFTS
    train_sets, dev_sets, partitions = prior.partitions()
    base, familiar, target, dev_fam, dev_LI = prior.generate(train_sets, dev_sets)
    # Recheck the original design's core, dose and shortcut assertions.
    train_audit = prior.audit_training(base, familiar, target)
    assert prior.shortcut_report([s for s in dev_LI if s.voice == 'AP'])['independent_parser_oracle_C_query'] == 1.0
    assert all('LI' not in s.mods for s in base + familiar + target)
    assert all('LI' in s.mods[1] for s in dev_LI)

    common_framed = [expand(source, prefix) for source in base for prefix in FRAMES]
    familiar_framed = [expand(source, prefix) for source in familiar for prefix in FRAMES]
    target_framed = [expand(source, prefix) for source in target for prefix in FRAMES]
    training_familiar = common_framed + familiar_framed
    training_target = common_framed + target_framed
    assert len(training_familiar) == len(training_target)
    assert {s.text for s in common_framed}.isdisjoint(s.text for s in familiar_framed + target_framed)
    assert len({s.text for s in training_familiar}) == len(training_familiar)
    assert len({s.text for s in training_target}) == len(training_target)
    assert all(s.voice in ('AA','PA') for s in training_familiar)
    assert set(s.voice for s in target_framed) == {'AP'}
    assert all('LI' not in s.mods for s in training_familiar + training_target)
    assert {s.text for s in training_familiar + training_target}.isdisjoint(s.text for s in dev_fam + dev_LI)
    for f,t in zip(familiar_framed,target_framed):
        assert f.core_id == t.core_id and f.mods == t.mods
        assert f.gold == t.gold and f.voice == 'AA' and t.voice == 'AP'
        assert len(f.text) > 0 and len(t.text) > 0
        assert f.text.split(' ; ')[0] == t.text.split(' ; ')[0]

    # Both conditions must separately cover dev L+I absolute positions,
    # not just the union of positions across arms.
    dev = feature_pos(dev_LI)
    before = {
        'familiar': feature_pos(base+familiar),
        'target': feature_pos(base+target),
    }
    after = {
        'familiar': feature_pos(training_familiar),
        'target': feature_pos(training_target),
    }
    contrasts = {}
    for arm in ('familiar', 'target'):
        a,b=before[arm],after[arm]
        misses_before = dev['exact'] - a['exact']
        misses_after = dev['exact'] - b['exact']
        missing_lengths = sorted(dev['lengths'] - b['lengths'])
        missing_structural = sorted(dev['structural'] - b['structural'])
        missing_namepos = sorted(dev['namepos'] - b['namepos'])
        assert max(b['lengths']) <= prior.MAX_TOKENS
        assert not misses_after, f'Uncovered token position in {arm}: {sorted(misses_after)[:12]}'
        assert not missing_lengths and not missing_structural and not missing_namepos
        contrasts[arm] = {
            'unframed_training_sources': len(base)+(len(familiar) if arm=='familiar' else len(target)),
            'framed_training_sources': len(training_familiar if arm=='familiar' else training_target),
            'max_training_input_tokens': max(b['lengths']),
            'unframed_missing_dev_exact_token_positions': len(misses_before),
            'framed_missing_dev_exact_token_positions': len(misses_after),
            'uncovered_dev_input_lengths_after': missing_lengths,
            'uncovered_dev_structural_token_positions_after': [list(x) for x in missing_structural],
            'uncovered_dev_name_positions_after': missing_namepos,
            'dev_exact_token_positions_covered_pct': pct(len(dev['exact'] - misses_after),len(dev['exact'])),
        }

    # The neutral temporal-prefix frequencies are identical between two dose arms.
    assert len(familiar_framed) == len(target_framed)
    assert Counter(f.voice for f in familiar_framed) == {'AA':len(familiar_framed)}
    assert Counter(t.voice for t in target_framed) == {'AP':len(target_framed)}
    return {
        'study':'EXP-105', 'phase':'POSITION_CONTROL_PREFLIGHT_V1',
        'status':'UNFROZEN_DESIGN_ONLY', 'no_training_or_protected_examples':True,
        'source_v0_sha256':sha256(Path(prior.__file__).read_bytes()).hexdigest(),
        'source_v1_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'partition_digest':partitions['partition_digest'],
        'frames':[{'prefix':p,'tokens':len(prior.tokens(p)) if p else 0} for p in FRAMES],
        'dev_LI_sources':len(dev_LI),
        'dev_LI_max_input_tokens':max(dev['lengths']),
        'dev_unique_exact_token_positions':len(dev['exact']),
        'common_base_sources_unframed':train_audit['common_base']['sources'],
        'dose_sources_per_arm_unframed':train_audit['dose_source_pairs'],
        'arms':contrasts,
        'remaining_caveats':[
            'Absolute position coverage is necessary, not sufficient, for systematic structural generalization.',
            'Training on longer neutral time frames changes the training distribution and should be included in both arms in any future runner.',
            'Exposure AA versus AP still changes passive syntax, surface length and event-specific voice coverage; these are the intended intervention plus inseparable possible alternatives.',
            'No conditional alignment check proves that the familiar arm has seen every structural token in the same syntactic context.',
            'This is the existing EXP-102-style grammar family and not an independent grammar replication.',
            'The dev L+I examples were generated for design audit only. Reserved name-set identifiers were not rendered.',
            'Preflight passed, but a new training protocol must still be designed, piloted and frozen before confirmatory training.'
        ]
    }


def markdown(r):
    lines = [
        '# EXP-105 positional-coverage control — design preflight v1','',
        '**Unfrozen design audit only. No training, reserved examples, or protected evaluation.**','',
        f"Time-frame token offsets: {', '.join(str(x['tokens']) for x in r['frames'])}.",
        f"Development L+I sources: {r['dev_LI_sources']}; longest full input: {r['dev_LI_max_input_tokens']} tokens.",
        f"Distinct development full-prompt (token, absolute-position) pairs: {r['dev_unique_exact_token_positions']}.",
        '',
        '| Arm | Original training sources | Framed training sources | Missing token/positions before | Missing after | Max train tokens |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for arm in ('familiar','target'):
        a=r['arms'][arm]
        lines.append(f"| {arm} | {a['unframed_training_sources']} | {a['framed_training_sources']} | {a['unframed_missing_dev_exact_token_positions']} | {a['framed_missing_dev_exact_token_positions']} | {a['max_training_input_tokens']} |")
    lines += ['', '**PASS:** All development L+I full-prompt exact token/position pairs and lengths are covered separately by each training arm; no within-event L+I examples enter training.', '',
             '## Remaining interpretation limits', *['- '+s for s in r['remaining_caveats']], '',
             '**Next:** independently review the new training distribution and exposure-matching plan before using it in a model runner.', '']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--write',action='store_true',help='Write v1 audit report; never overwrite existing outputs')
    args=ap.parse_args()
    r=report()
    s=markdown(r)
    print(s, flush=True)
    if args.write:
        OUT.mkdir(parents=True,exist_ok=True)
        files=(OUT/'position_control_preflight_v1.md', OUT/'position_control_preflight_v1.json')
        if any(x.exists() for x in files):
            raise FileExistsError('Refusing to overwrite existing v1 audit files')
        files[0].write_text(s,encoding='utf-8')
        files[1].write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        for x in files: print('SAVED',x)

if __name__=='__main__':
    main()
