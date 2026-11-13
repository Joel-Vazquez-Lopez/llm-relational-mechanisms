#!/usr/bin/env python3
"""EXP-102 design-only audit: no models, checkpoints, or protected set.

Run from the repository with scripts/exp102_syntax_preflight.py beside this file.
Recreates *only* the public design examples used by that preflight.
"""
from collections import Counter, defaultdict
from itertools import product
from random import Random
from hashlib import sha256
import json

from exp102_syntax_preflight import (
    Event, NAMES, VERBS, VOICE, TRAIN, COMPOSITION_DEV, TOKEN_RE,
    render, parse_text, gold, questions,
)


def tokenize(s, q):
    return ("<bos>", *TOKEN_RE.findall(s), "<sep>", *TOKEN_RE.findall(q), "<answer>")


def make_cases():
    rng = Random(102202)
    seeds = [(tuple(rng.sample(NAMES, 4)), tuple(rng.sample(VERBS, 2)))
             for _ in range(96)]
    for i, ((a,b,c,d), (v0,v1)) in enumerate(seeds):
        for v_first, v_second in product(VOICE, repeat=2):
            events = (Event(a,b,v0,v_first,c,d,0), Event(c,d,v1,v_second,a,b,1))
            for condition in (*TRAIN, *COMPOSITION_DEV):
                text = render(events, condition)
                assert tuple(x for pair in parse_text(text) for x in pair) == gold(events)
                group = 'train_design' if condition in TRAIN else 'composition_design'
                yield i, group, condition, (v_first, v_second), text, gold(events)


def main():
    all_queries = questions()
    data = defaultdict(list)
    for idx, group, condition, voices, text, answers in make_cases():
        for qi, q in enumerate(all_queries):
            toks = tokenize(text, q)
            data[group].append((idx, condition, voices, text, qi, answers[qi], toks))

    tr = data['train_design']; te = data['composition_design']
    train_pos = Counter((j, tok) for row in tr for j, tok in enumerate(row[-1]))
    comp_pos = Counter((j, tok) for row in te for j, tok in enumerate(row[-1]))
    new_pos = sorted(set(comp_pos) - set(train_pos), key=lambda k: (k[0], k[1]))
    train_words = set(tok for _,tok in train_pos)
    comp_words = set(tok for _,tok in comp_pos)
    lengths = lambda rows: Counter(len(r[-1]) for r in rows)
    first_voices = lambda rows: Counter(r[2] for r in rows)
    per_question = lambda rows: Counter(r[4] for r in rows)

    unique_train_texts = {r[3] for r in tr}
    unique_comp_texts = {r[3] for r in te}
    train_token_mult = Counter(tok for r in tr for tok in r[-1])
    comp_token_mult = Counter(tok for r in te for tok in r[-1])

    # Detect accidental identical rendered sources appearing in both pools.
    shared_sources = unique_train_texts & unique_comp_texts
    unique_name_sets_train = {frozenset(tok for tok in TOKEN_RE.findall(r[3]) if tok in NAMES) for r in tr}
    unique_name_sets_comp = {frozenset(tok for tok in TOKEN_RE.findall(r[3]) if tok in NAMES) for r in te}

    print('=== EXP-102 SYNTAX DESIGN / ABSOLUTE POSITION AUDIT ===')
    print('Training design sentences:', len(tr) // 4)
    print('Composition design sentences:', len(te) // 4)
    print('Unique training texts:', len(unique_train_texts))
    print('Unique composition texts:', len(unique_comp_texts))
    print('Identical texts shared across pools:', len(shared_sources))
    print('Distinct four-name sets (train, composition, shared):',
          len(unique_name_sets_train), len(unique_name_sets_comp),
          len(unique_name_sets_train & unique_name_sets_comp))
    print('Question counts train:', dict(sorted(per_question(tr).items())))
    print('Question counts composition:', dict(sorted(per_question(te).items())))
    print('Voice pairs train:', dict(sorted(first_voices(tr).items())))
    print('Voice pairs composition:', dict(sorted(first_voices(te).items())))
    print('Question-inclusive length range training:', (min(lengths(tr)), max(lengths(tr))))
    print('Question-inclusive length range composition:', (min(lengths(te)), max(lengths(te))))
    print('New vocabulary in composition:', sorted(comp_words - train_words))
    print('Token-position pairs seen in training:', len(train_pos))
    print('Token-position pairs in composition:', len(comp_pos))
    print('Novel token-position pairs in composition:', len(new_pos))
    if new_pos:
        print('First 25 novel token-position pairs with design frequencies:')
        for pos, tok in new_pos[:25]:
            print('  pos=%2d token=%-12r composition_count=%d' % (pos, tok, comp_pos[(pos, tok)]))
    print('Composition questions with >=1 novel token position:', sum(
        any((j, tok) not in train_pos for j, tok in enumerate(r[-1])) for r in te), '/', len(te))
    print('Composition tokens with >=1 novel token position:', sum(
        cnt for p, cnt in comp_pos.items() if p not in train_pos), '/', sum(comp_pos.values()))
    # Validate exhaustive input and label diversity counts, not performance.
    print('Design-only audit completed. No training, no protected split, no files changed.')


if __name__ == '__main__':
    main()
