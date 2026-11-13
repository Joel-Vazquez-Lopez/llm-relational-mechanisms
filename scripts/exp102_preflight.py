#!/usr/bin/env python3
"""EXP-102 development-only generator and baseline audit.

No training, checkpoints, or protected evaluation.
"""
from dataclasses import dataclass, replace
from itertools import permutations, product
from random import Random
from collections import defaultdict
import re

NAMES = (
    "Nila", "Taro", "Vena", "Riko",
    "Sela", "Miro", "Pavo", "Kina"
)
VERBS = ("admired", "greeted", "followed", "thanked")
VOICES = ("active", "passive")


@dataclass(frozen=True)
class Event:
    agent: str
    patient: str
    verb: str
    voice: str


def render(event):
    if event.voice == "active":
        return f"{event.agent} {event.verb} {event.patient}"
    return f"{event.patient} was {event.verb} by {event.agent}"


def sentence(events):
    return " ; ".join(map(render, events)) + "."


def parse(text):
    """Parse rendered text without using gold metadata."""
    if not text.endswith("."):
        raise ValueError("Missing period")

    clauses = text[:-1].split(" ; ")
    if len(clauses) != 2:
        raise ValueError("Expected two events")

    result = []
    for clause in clauses:
        parts = clause.split()

        if len(parts) == 3:
            agent, verb, patient = parts
        elif (
            len(parts) == 5
            and parts[1] == "was"
            and parts[3] == "by"
        ):
            patient, _, verb, _, agent = parts
        else:
            raise ValueError(f"Invalid clause: {clause}")

        if (
            agent not in NAMES
            or patient not in NAMES
            or verb not in VERBS
        ):
            raise ValueError(f"Unknown lexical item: {clause}")

        result.append((agent, patient))

    return tuple(result)


def questions():
    return tuple(
        f"In the {ordinal} event, who was the {role}?"
        for ordinal in ("first", "second")
        for role in ("agent", "patient")
    )


def parse_question(question):
    match = re.fullmatch(
        r"In the (first|second) event, who was the (agent|patient)\?",
        question,
    )
    if match is None:
        raise ValueError(f"Invalid question: {question}")
    return (
        0 if match.group(1) == "first" else 1,
        0 if match.group(2) == "agent" else 1,
    )


def gold(events):
    return (
        events[0].agent, events[0].patient,
        events[1].agent, events[1].patient,
    )


def baseline_answers(text, strategy):
    parsed = parse(text)

    if strategy == "first_event_only":
        return (*parsed[0], *parsed[0])

    if strategy == "second_event_only":
        return (*parsed[1], *parsed[1])

    if strategy == "role_blind":
        return (
            parsed[0][0], parsed[0][0],
            parsed[1][0], parsed[1][0],
        )

    if strategy == "surface_order":
        answer = []
        for clause in text[:-1].split(" ; "):
            answer.extend(
                word for word in clause.split()
                if word in NAMES
            )
        return tuple(answer)

    if strategy == "text_parser_oracle":
        return (*parsed[0], *parsed[1])

    raise ValueError(strategy)


def main():
    rng = Random(102001)
    assignments = list(permutations(NAMES, 4))
    rng.shuffle(assignments)

    strategies = (
        "first_event_only",
        "second_event_only",
        "role_blind",
        "surface_order",
        "text_parser_oracle",
    )

    results = defaultdict(lambda: [0, 0])
    query_indices = tuple(parse_question(q) for q in questions())
    assert query_indices == (
        (0, 0), (0, 1), (1, 0), (1, 1)
    )

    seen = set()
    sample = None
    total = 0

    for a, b, c, d in assignments[:32]:
        for verb1, verb2 in permutations(VERBS, 2):
            for voice1, voice2 in product(VOICES, repeat=2):
                first = Event(a, b, verb1, voice1)
                second = Event(c, d, verb2, voice2)

                for events in (
                    (first, second),
                    (second, first),
                ):
                    text = sentence(events)
                    if text in seen:
                        continue
                    seen.add(text)

                    truth = gold(events)
                    parsed = parse(text)

                    assert tuple(
                        parsed[i][j] for i, j in query_indices
                    ) == truth
                    assert len(set(truth)) == 4

                    # Event-order transformation
                    assert gold(events[::-1]) == (
                        truth[2:] + truth[:2]
                    )

                    # Voice transformation preserves roles
                    new_voice = (
                        "passive" if events[0].voice == "active"
                        else "active"
                    )
                    changed = (
                        replace(events[0], voice=new_voice),
                        events[1],
                    )
                    assert tuple(
                        x for pair in parse(sentence(changed))
                        for x in pair
                    ) == truth

                    # Reversing roles must change the answers
                    reversed_first = (
                        replace(
                            events[0],
                            agent=events[0].patient,
                            patient=events[0].agent,
                        ),
                        events[1],
                    )
                    assert tuple(
                        x for pair in parse(sentence(reversed_first))
                        for x in pair
                    ) == (
                        truth[1], truth[0],
                        truth[2], truth[3],
                    )

                    for strategy in strategies:
                        predicted = baseline_answers(text, strategy)
                        results[strategy][0] += sum(
                            p == g for p, g in zip(predicted, truth)
                        )
                        results[strategy][1] += int(
                            predicted == truth
                        )

                    total += 1
                    if sample is None:
                        sample = (text, truth)

    print(f"\nEXP-102 DESIGN PREFLIGHT: {total} sentences")
    print(f"Total questions: {4 * total}")
    print("Sample:", sample[0])
    print("Questions:", questions())
    print("Gold:", sample[1])

    print("\nBaseline                 Accuracy   C_query")
    for strategy in strategies:
        correct, perfect = results[strategy]
        print(
            f"{strategy:25s} "
            f"{correct / (4 * total):.3f}      "
            f"{perfect / total:.3f}"
        )

    assert results["text_parser_oracle"] == [4 * total, total]
    assert all(results[s][1] == 0 for s in strategies[:3])

    print("\nPASS: semantic and counterfactual invariants")
    print("No training or protected evaluation performed.")


if __name__ == "__main__":
    main()
