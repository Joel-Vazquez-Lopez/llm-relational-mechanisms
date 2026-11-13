#!/usr/bin/env python3
"""EXP-102: generator-only, development-only syntactic composition preflight.

Not a training experiment, not a frozen protocol, and not a protected evaluation.
No torch, checkpoints, or previous EXP-101 data are accessed.
"""
from collections import Counter
from dataclasses import dataclass, replace
from itertools import product
from random import Random
import re

NAMES = ("Nila", "Taro", "Vena", "Riko", "Sela", "Miro", "Pavo", "Kina")
VERBS = ("admired", "greeted", "followed", "thanked")
VOICE = ("active", "passive")
TRAIN = (("bare", "bare"), ("L", "bare"), ("bare", "L"),
         ("I", "bare"), ("bare", "I"), ("L", "I"), ("I", "L"),
         ("L", "L"), ("I", "I"))
COMPOSITION_DEV = (("LI", "bare"), ("bare", "LI"))
TOKEN_RE = re.compile(r"'s|[A-Za-z]+|[;,.?]")


@dataclass(frozen=True)
class Event:
    agent: str
    patient: str
    verb: str
    voice: str
    gate_owner: str
    near_person: str
    layout: int


def render_event(e, mods):
    if e.voice == "active":
        core = (f"{e.agent}, near {e.near_person}, {e.verb} {e.patient}"
                if "I" in mods else f"{e.agent} {e.verb} {e.patient}")
    else:
        core = (f"{e.patient} was {e.verb} near {e.near_person} by {e.agent}"
                if "I" in mods else f"{e.patient} was {e.verb} by {e.agent}")
    if "L" in mods:
        core = (f"By {e.gate_owner}'s gate, {core}" if e.layout == 0
                else f"{core} beside {e.gate_owner}'s gate")
    return core


def render(events, mods):
    return " ; ".join(render_event(e, m) for e, m in zip(events, mods)) + "."


def parse_text(sentence):
    """Independent surface parser; never uses gold event metadata."""
    assert sentence.endswith(".")
    clauses = sentence[:-1].split(" ; ")
    assert len(clauses) == 2
    out = []
    for clause in clauses:
        s = clause
        prefix = re.match(r"^By ([A-Z][a-z]+)'s gate, ", s)
        if prefix:
            s = s[prefix.end():]
        s = re.sub(r" beside [A-Z][a-z]+'s gate$", "", s)
        s = re.sub(r", near [A-Z][a-z]+, ", " ", s)
        s = re.sub(r" near [A-Z][a-z]+ by ", " by ", s)
        passive = re.fullmatch(r"([A-Z][a-z]+) was (\w+) by ([A-Z][a-z]+)", s)
        active = re.fullmatch(r"([A-Z][a-z]+) (\w+) ([A-Z][a-z]+)", s)
        if passive:
            patient, verb, agent = passive.groups()
        elif active:
            agent, verb, patient = active.groups()
        else:
            raise AssertionError(f"Unparseable clause: {clause!r} -> {s!r}")
        assert verb in VERBS and agent != patient
        out.append((agent, patient))
    return tuple(out)


def gold(events):
    return tuple(x for e in events for x in (e.agent, e.patient))


def questions():
    return tuple(f"In the {n} event, who was the {r}?"
                 for n in ("first", "second") for r in ("agent", "patient"))


def length_with_question(text):
    return 1 + len(TOKEN_RE.findall(text)) + 1 + len(TOKEN_RE.findall(questions()[0])) + 1


def audit_one(events, mods):
    text = render(events, mods)
    parsed = parse_text(text)
    expected = gold(events)
    assert tuple(x for p in parsed for x in p) == expected
    assert len(set(expected)) == 4
    assert tuple(x for p in parse_text(render(events[::-1], mods[::-1])) for x in p) == expected[2:] + expected[:2]
    flipped = (replace(events[0], voice="passive" if events[0].voice == "active" else "active"), events[1])
    assert tuple(x for p in parse_text(render(flipped, mods)) for x in p) == expected
    reversed_roles = (replace(events[0], agent=events[0].patient, patient=events[0].agent), events[1])
    assert tuple(x for p in parse_text(render(reversed_roles, mods)) for x in p) == (expected[1], expected[0], expected[2], expected[3])
    # An event-blind answer (always event 0) or role-blind answer cannot solve all four queries.
    assert (expected[0], expected[1], expected[0], expected[1]) != expected
    assert (expected[0], expected[0], expected[2], expected[2]) != expected
    tokens = set(TOKEN_RE.findall(text + " " + questions()[0]))
    return text, length_with_question(text), tokens


def main():
    rng = Random(102202)
    examples = [(tuple(rng.sample(NAMES, 4)), tuple(rng.sample(VERBS, 2)))
                for _ in range(96)]
    train_len, dev_len = [], []
    train_vocab, dev_vocab = set(), set()
    train_counts, dev_counts = Counter(), Counter()
    sample = None
    for (a, b, c, d), (v0, v1) in examples:
        for first_voice, second_voice in product(VOICE, repeat=2):
            # Every event's modifiers refer to the two *other* event participants.
            e0 = Event(a, b, v0, first_voice, c, d, 0)
            e1 = Event(c, d, v1, second_voice, a, b, 1)
            events = (e0, e1)
            for mods in TRAIN:
                text, length, toks = audit_one(events, mods)
                train_len.append(length)
                train_vocab.update(toks)
                train_counts[mods] += 1
            for mods in COMPOSITION_DEV:
                text, length, toks = audit_one(events, mods)
                dev_len.append(length)
                dev_vocab.update(toks)
                dev_counts[mods] += 1
                if sample is None:
                    sample = (text, gold(events))
    assert dev_vocab <= train_vocab, f"Unseen tokens: {dev_vocab - train_vocab}"
    print("EXP-102 SYNTAX PREFLIGHT: DEVELOPMENT/DESIGN ONLY")
    print("Training modifier cells:", dict(train_counts))
    print("Held-out composition *design* cells:", dict(dev_counts))
    print("Example new composition:", sample[0])
    print("Gold answers:", sample[1])
    print("Four questions:", questions())
    print(f"Maximum input tokens: train={max(train_len)}, composition={max(dev_len)}")
    print("Token vocabulary transfer: PASS")
    print("Independent parsing + role/voice/order transformations: PASS")
    print("Event-blind and role-blind full-four-query shortcuts: FAIL by construction")
    if max(dev_len) > max(train_len):
        raise SystemExit("STOP: composition exceeds training position coverage; revise design before training")
    print("Global maximum-position coverage: PASS (per-position coverage still needs audit)")
    print("NOT a frozen split. NOT a pretrained-model result. NO protected examples generated.")


if __name__ == "__main__":
    main()
