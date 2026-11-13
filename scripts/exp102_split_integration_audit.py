#!/usr/bin/env python3
"""EXP-102 V2: read-only integration audit of the frozen *name-set partition*
against the existing familiar-syntax generator.

Does not generate any reserved-set examples, combined L+I examples,
train models, or access checkpoints. Does not write files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path

from exp102_syntax_preflight import (
    Event, NAMES, VERBS, VOICE, TRAIN, COMPOSITION_DEV,
    TOKEN_RE, render, parse_text, gold, questions,
)

SPLIT_PATH = Path('reports/exp102/syntax_name_set_split_v2.json')
EXPECTED_SPLIT_SHA256 = '2f9ba0c6f44c67094f9fa2786853ade1c8c662fd0afd74295af98384b17dd287'


def stable_sha(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_split():
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f'Missing split file: {SPLIT_PATH}')
    manifest = json.loads(SPLIT_PATH.read_text())
    parts = manifest['partitions']
    expected = {'train_name_sets': 50, 'development_name_sets': 10, 'reserved_name_sets': 10}
    for key, count in expected.items():
        assert len(parts[key]) == count, (key, len(parts[key]))
    assert manifest['split_unit'] == 'unordered_four_name_set'
    assert stable_sha(parts) == manifest['partition_sha256'] == EXPECTED_SPLIT_SHA256, 'Split hash mismatch'
    pools = {}
    for key in expected:
        values = [tuple(x) for x in parts[key]]
        assert all(len(x) == 4 and tuple(sorted(x)) == x and len(set(x)) == 4 for x in values)
        pools[key] = set(values)
    assert not pools['train_name_sets'] & pools['development_name_sets']
    assert not pools['train_name_sets'] & pools['reserved_name_sets']
    assert not pools['development_name_sets'] & pools['reserved_name_sets']
    assert len(set.union(*pools.values())) == 70
    assert set(n for group in pools.values() for tup in group for n in tup) == set(NAMES)
    assert all('LI' not in mods for mods in TRAIN)
    assert all('LI' in mods for mods in COMPOSITION_DEV)
    return pools, manifest


def generate_familiar_group(name_sets, label):
    """Controlled *familiar* design only; no L+I, no reserved identifiers used."""
    seen = set()
    name_sets_seen = set()
    per_cell = Counter()
    per_voice = Counter()
    per_query = Counter()
    token_positions = set()
    max_len = 0
    name_positions = defaultdict(set)
    n_sentences = 0
    for names in sorted(name_sets):
        for a, b, c, d in permutations(names, 4):
            # Deterministically rotate four distinct verbs across name-set index;
            # goal here is exhaustive syntax/role validation, not a final
            # model-training dataset or claimed combinatorial verb holdout.
            v0 = VERBS[(NAMES.index(a) + NAMES.index(c)) % len(VERBS)]
            v1 = VERBS[(NAMES.index(b) + NAMES.index(d) + 1) % len(VERBS)]
            if v1 == v0:
                v1 = VERBS[(VERBS.index(v0) + 1) % len(VERBS)]
            for voice0, voice1 in product(VOICE, repeat=2):
                e0 = Event(a,b,v0,voice0,c,d,0)
                e1 = Event(c,d,v1,voice1,a,b,1)
                events = (e0,e1)
                for mods in TRAIN:
                    source = render(events, mods)
                    assert source not in seen, f'Duplicate source in {label}: {source}'
                    seen.add(source)
                    parsed = parse_text(source)
                    answer = gold(events)
                    assert tuple(name for pair in parsed for name in pair) == answer
                    assert tuple(sorted(answer)) == names
                    assert len(set(answer)) == 4
                    name_sets_seen.add(names)
                    per_cell[mods] += 1
                    per_voice[(voice0,voice1)] += 1
                    n_sentences += 1
                    for index, question in enumerate(questions()):
                        toks = ('<bos>', *TOKEN_RE.findall(source), '<sep>',
                                *TOKEN_RE.findall(question), '<answer>')
                        max_len = max(max_len, len(toks))
                        assert answer[index] in names
                        per_query[index] += 1
                        for pos,tok in enumerate(toks):
                            token_positions.add((pos,tok))
                            if tok in NAMES:
                                name_positions[tok].add(pos)
    assert name_sets_seen == name_sets
    assert len(set(per_cell.values())) == 1
    assert len(set(per_voice.values())) == 1
    assert len(set(per_query.values())) == 1
    return dict(
        label=label, sources=seen, num_sentences=n_sentences, cells=per_cell,
        voice=per_voice, questions=per_query,
        token_pos=token_positions, name_pos=name_positions, max_tokens=max_len,
    )


def main():
    pools, manifest = check_split()
    print('EXP-102 NAME-SET × FAMILIAR-SYNTAX INTEGRATION AUDIT')
    print('Split SHA256:', manifest['partition_sha256'])
    print('Split: train=50, development=10, reserved=10 four-name sets')
    print('Reserved name sets are NOT used to generate examples.')
    train = generate_familiar_group(pools['train_name_sets'], 'train')
    dev = generate_familiar_group(pools['development_name_sets'], 'dev')
    assert not train['sources'] & dev['sources']
    train_voc = {w for _,w in train['token_pos']}
    dev_voc = {w for _,w in dev['token_pos']}
    new_vocab = dev_voc - train_voc
    novel_pos = dev['token_pos'] - train['token_pos']
    print('TRAIN familiar sentences:', train['num_sentences'])
    print('DEV familiar sentences:', dev['num_sentences'])
    print('TRAIN/DEV sentence overlap:', len(train['sources'] & dev['sources']))
    print('TRAIN/DEV name-set overlap:', len(pools['train_name_sets'] & pools['development_name_sets']))
    print('TRAIN familiar cells:', {str(k):v for k,v in train['cells'].items()})
    print('DEV familiar cells:', {str(k):v for k,v in dev['cells'].items()})
    print('Max tokenized length, familiar train/dev:', train['max_tokens'],dev['max_tokens'])
    print('New DEV vocabulary:', sorted(new_vocab))
    print('Novel familiar DEV token-position pairs:', len(novel_pos))
    if novel_pos:
        print('Examples:', sorted(novel_pos)[:20])
    assert not new_vocab, 'New development vocabulary'
    print('CHECKS: split hash, no train/dev overlap, oracle labels, balanced voices/questions: PASS')
    print('STATUS: DESIGN AUDIT ONLY. No L+I composition examples or reserved examples generated.')
    print('NOTE: This audit enumerates a DESIGN pool, not the final stochastic training sample.')
    print('NOTE: Earlier L+I design examples were already inspected; that pool is NOT a fresh protected test.')


if __name__ == '__main__':
    main()
