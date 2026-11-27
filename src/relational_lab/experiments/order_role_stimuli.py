"""Order-to-role stimuli: dissociating linear order, grammatical function and thematic role.

Four main constructions realize every pattern of agreement between the
first-mentioned name, the grammatical subject and the agent:

    active          Leon praised Mira.                    first=A  subj=A  agent=A
    passive         Mira was praised by Leon.             first=P  subj=P  agent=A
    object_cleft    It was Mira that Leon praised.        first=P  subj=A  agent=A
    by_cleft        It was by Leon that Mira was praised.  first=A  subj=P  agent=A

plus a subject-cleft control ("It was Leon who praised Mira.") that changes
the frame but not the order, so cleft effects can be separated from order,
and two held-out pseudo-clefts that are never used to fit anything and exist
only to test whether a fitted role code transfers to unseen surface forms:

    relative_active   The one who praised Mira was Leon.         first=P  subj=A  agent=A
    relative_passive  The one who was praised by Leon was Mira.   first=A  subj=P  agent=A

("subject" there means the subject of the event verb's clause, i.e. the
participant the relative pronoun stands for.)

Across the four main constructions the three binary labels carried by each
name token (is_first, is_subject, is_agent) are exactly uncorrelated.

Questions never name a participant ("Who praised someone?"), so the answer
cannot be found by eliminating the name mentioned in the question.

Dependency-free by design. This module generates stimuli only; it makes no
claim about any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

GENERATOR_VERSION = "1.3.0"
# 1.3.0: the passive-voice agent question is "By whom was someone {past}?".
#   1.2.0 used "Who was someone {past} by?"; the stranded preposition was
#   read as "who was {past}" (a patient question) by Gemma 4 31B-it on dev,
#   so that wording measured parsing of the question, not the sentence.
#   Names, verbs, items per family and splits are unchanged.

# Gender-balanced, common English first names. Single-token status depends on
# the tokenizer; use ``filter_names`` with a real tokenizer before freezing.
NAMES: tuple[str, ...] = (
    "Anna", "Maria", "Emma", "Sarah", "Laura", "Kate",
    "Lisa", "Julia", "Rose", "Grace", "Alice", "Helen",
    "John", "David", "Peter", "Paul", "Mark", "James",
    "Tom", "Daniel", "Michael", "Robert", "Frank", "George",
)

# Reversible transitive verbs with a regular -ed form, so the simple past and
# the passive participle are the same string. (lemma, past/participle)
VERBS: tuple[tuple[str, str], ...] = (
    ("praise", "praised"), ("help", "helped"), ("follow", "followed"),
    ("thank", "thanked"), ("visit", "visited"), ("call", "called"),
    ("push", "pushed"), ("watch", "watched"), ("chase", "chased"),
    ("greet", "greeted"), ("warn", "warned"), ("admire", "admired"),
    ("invite", "invited"), ("blame", "blamed"), ("rescue", "rescued"),
    ("tickle", "tickled"),
)

MAIN_CONSTRUCTIONS: tuple[str, ...] = ("active", "passive", "object_cleft", "by_cleft")
CONTROL_CONSTRUCTIONS: tuple[str, ...] = ("subject_cleft",)
HELDOUT_CONSTRUCTIONS: tuple[str, ...] = ("relative_active", "relative_passive")
CONSTRUCTIONS: tuple[str, ...] = MAIN_CONSTRUCTIONS + CONTROL_CONSTRUCTIONS + HELDOUT_CONSTRUCTIONS

ROLES: tuple[str, ...] = ("agent", "patient")
QUESTION_VOICES: tuple[str, ...] = ("active", "passive")

# Which participant (A = agent, P = patient) is first-mentioned and which is the
# grammatical subject, per construction. The agent is always A.
CONSTRUCTION_LABELS: dict[str, dict[str, str]] = {
    "active": {"first": "A", "subject": "A"},
    "passive": {"first": "P", "subject": "P"},
    "object_cleft": {"first": "P", "subject": "A"},
    "by_cleft": {"first": "A", "subject": "P"},
    "subject_cleft": {"first": "A", "subject": "A"},
    "relative_active": {"first": "P", "subject": "A"},
    "relative_passive": {"first": "A", "subject": "P"},
}

# Position labels shared across constructions, so a patching site means the
# same thing everywhere. Absent labels (no "by" in an active) are simply missing.
POSITION_LABELS: tuple[str, ...] = (
    "first_name", "second_name", "verb", "aux", "by", "period", "question_end", "answer_cue",
)

# The "Sentence: " prefix puts a space before every name in every construction,
# so BPE tokenizers see the same name token (" Anna") regardless of position.
PROMPT_PREFIX = "Sentence: "
PROMPT_TEMPLATE = PROMPT_PREFIX + "{sentence}\nQuestion: {question}\nAnswer:"


def render_sentence(construction: str, agent: str, patient: str, past: str) -> str:
    if construction == "active":
        return f"{agent} {past} {patient}."
    if construction == "passive":
        return f"{patient} was {past} by {agent}."
    if construction == "object_cleft":
        return f"It was {patient} that {agent} {past}."
    if construction == "by_cleft":
        return f"It was by {agent} that {patient} was {past}."
    if construction == "subject_cleft":
        return f"It was {agent} who {past} {patient}."
    if construction == "relative_active":
        return f"The one who {past} {patient} was {agent}."
    if construction == "relative_passive":
        return f"The one who was {past} by {agent} was {patient}."
    raise ValueError(f"unknown construction: {construction}")


def render_question(role: str, question_voice: str, lemma: str, past: str) -> str:
    if role == "agent":
        if question_voice == "active":
            return f"Who {past} someone?"
        if question_voice == "passive":
            return f"By whom was someone {past}?"
    if role == "patient":
        if question_voice == "active":
            return f"Who did someone {lemma}?"
        if question_voice == "passive":
            return f"Who was {past} by someone?"
    raise ValueError(f"unknown role/voice: {role}/{question_voice}")


@dataclass(frozen=True)
class NameToken:
    """Labels and character span of one name inside the sentence."""

    name: str
    char_start: int  # offset within ``sentence``
    char_end: int
    prompt_char_start: int  # offset within ``prompt`` (for tokenizer offset mapping)
    prompt_char_end: int
    is_first: bool
    is_subject: bool
    is_agent: bool


@dataclass(frozen=True)
class Item:
    item_id: str
    family_id: str  # unordered name pair: the unit of analysis
    event_id: str  # (agent, patient, verb): all forms of one event share it
    split: str  # "dev" or "test"
    test_set: int  # 0..n_test_sets-1 for test items, -1 for dev
    construction: str
    is_control: bool
    is_heldout: bool
    agent: str
    patient: str
    lemma: str
    verb: str
    role_asked: str
    question_voice: str
    sentence: str
    question: str
    prompt: str
    answer: str
    distractor: str
    answer_is_first: bool
    names: tuple[NameToken, NameToken]  # in order of mention
    # (label, prompt_char_start, prompt_char_end) for each POSITION_LABELS entry present
    spans: tuple[tuple[str, int, int], ...] = ()

    def span(self, label: str) -> tuple[int, int] | None:
        for lab, start, end in self.spans:
            if lab == label:
                return (start, end)
        return None


def _name_tokens(construction: str, sentence: str, agent: str, patient: str) -> tuple[NameToken, NameToken]:
    labels = CONSTRUCTION_LABELS[construction]
    first = agent if labels["first"] == "A" else patient
    subject = agent if labels["subject"] == "A" else patient
    tokens = []
    cursor = 0
    for name in sorted((agent, patient), key=lambda n: sentence.index(n)):
        start = sentence.index(name, cursor)
        cursor = start + len(name)
        tokens.append(
            NameToken(
                name=name,
                char_start=start,
                char_end=cursor,
                prompt_char_start=start + len(PROMPT_PREFIX),
                prompt_char_end=cursor + len(PROMPT_PREFIX),
                is_first=(name == first),
                is_subject=(name == subject),
                is_agent=(name == agent),
            )
        )
    if tokens[0].name != first:
        raise AssertionError(f"first-mention label disagrees with sentence: {sentence!r}")
    return tokens[0], tokens[1]


_WORD = re.compile(r"\w+|[^\w\s]")


def _position_spans(
    sentence: str, question: str, prompt: str, names: tuple[NameToken, NameToken], past: str
) -> tuple[tuple[str, int, int], ...]:
    off = len(PROMPT_PREFIX)
    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(sentence)]
    out = [
        ("first_name", names[0].prompt_char_start, names[0].prompt_char_end),
        ("second_name", names[1].prompt_char_start, names[1].prompt_char_end),
    ]
    verb_idx = [i for i, (w, _, _) in enumerate(words) if w == past]
    if len(verb_idx) != 1:
        raise AssertionError(f"verb {past!r} must occur once in {sentence!r}")
    vi = verb_idx[0]
    out.append(("verb", words[vi][1] + off, words[vi][2] + off))
    if vi > 0 and words[vi - 1][0] == "was":  # passive auxiliary, not a cleft copula
        out.append(("aux", words[vi - 1][1] + off, words[vi - 1][2] + off))
    by = [w for w in words if w[0] == "by"]
    if len(by) > 1:
        raise AssertionError(f"'by' occurs more than once in {sentence!r}")
    if by:
        out.append(("by", by[0][1] + off, by[0][2] + off))
    if words[-1][0] != ".":
        raise AssertionError(f"sentence must end with a period: {sentence!r}")
    out.append(("period", words[-1][1] + off, words[-1][2] + off))
    q = prompt.index(question)
    out.append(("question_end", q + len(question) - 1, q + len(question)))
    out.append(("answer_cue", len(prompt) - 1, len(prompt)))
    return tuple(out)


def _check_names(names: Sequence[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("names must be unique")
    for a, b in combinations(names, 2):
        if a in b or b in a:
            raise ValueError(f"name {a!r} is a substring of {b!r}; spans would be ambiguous")


def split_families(
    names: Sequence[str], seed: int, dev_fraction: float, n_test_sets: int
) -> dict[str, tuple[str, int]]:
    """Assign each unordered name pair to dev or to one of n test sets."""
    families = [f"{a}|{b}" for a, b in combinations(sorted(names), 2)]
    rng = random.Random(seed)
    rng.shuffle(families)
    n_dev = round(len(families) * dev_fraction)
    out: dict[str, tuple[str, int]] = {}
    for i, fam in enumerate(families):
        if i < n_dev:
            out[fam] = ("dev", -1)
        else:
            out[fam] = ("test", (i - n_dev) % n_test_sets)
    return out


def build_item(
    agent: str,
    patient: str,
    lemma: str,
    past: str,
    construction: str,
    role: str,
    question_voice: str,
    family_id: str | None = None,
    split: str = "dev",
    test_set: int = -1,
) -> Item:
    """One prompt. ``family_id`` defaults to the sorted name pair."""
    family = family_id or "|".join(sorted((agent, patient)))
    event = f"{agent}>{patient}:{lemma}"
    sentence = render_sentence(construction, agent, patient, past)
    tokens = _name_tokens(construction, sentence, agent, patient)
    answer, distractor = (agent, patient) if role == "agent" else (patient, agent)
    question = render_question(role, question_voice, lemma, past)
    prompt = PROMPT_TEMPLATE.format(sentence=sentence, question=question)
    return Item(
        item_id=f"{event}:{construction}:{role}:{question_voice}",
        family_id=family,
        event_id=event,
        split=split,
        test_set=test_set,
        construction=construction,
        is_control=construction in CONTROL_CONSTRUCTIONS,
        is_heldout=construction in HELDOUT_CONSTRUCTIONS,
        agent=agent,
        patient=patient,
        lemma=lemma,
        verb=past,
        role_asked=role,
        question_voice=question_voice,
        sentence=sentence,
        question=question,
        prompt=prompt,
        answer=answer,
        distractor=distractor,
        answer_is_first=(tokens[0].name == answer),
        names=tokens,
        spans=_position_spans(sentence, question, prompt, tokens, past),
    )


def role_swapped(item: Item) -> Item:
    """The same verb, construction and question with agent and patient exchanged.
    Same names in the same slots' token positions, opposite correct answer."""
    return build_item(
        item.patient, item.agent, item.lemma, item.verb, item.construction,
        item.role_asked, item.question_voice,
        family_id=item.family_id, split=item.split, test_set=item.test_set,
    )


def generate_items(
    names: Sequence[str] = NAMES,
    verbs: Sequence[tuple[str, str]] = VERBS,
    verbs_per_orientation: int = 2,
    seed: int = 2026,
    dev_fraction: float = 0.15,
    n_test_sets: int = 4,
    include_controls: bool = True,
    include_heldout: bool = True,
) -> list[Item]:
    """Every family appears with both role assignments, in every construction,
    with every question. Each orientation of a family gets its own verbs."""
    _check_names(names)
    if verbs_per_orientation > len(verbs):
        raise ValueError("verbs_per_orientation exceeds number of verbs")
    splits = split_families(names, seed, dev_fraction, n_test_sets)
    # Verbs are assigned in rotation through a shuffled list, so every verb is
    # used equally often (up to one) and the verbs within one orientation differ.
    verb_order = list(verbs)
    random.Random(seed + 1).shuffle(verb_order)
    verb_cursor = 0
    constructions = MAIN_CONSTRUCTIONS
    if include_controls:
        constructions += CONTROL_CONSTRUCTIONS
    if include_heldout:
        constructions += HELDOUT_CONSTRUCTIONS
    items: list[Item] = []
    for a, b in combinations(sorted(names), 2):
        family = f"{a}|{b}"
        split, test_set = splits[family]
        for agent, patient in ((a, b), (b, a)):
            chosen = [verb_order[(verb_cursor + k) % len(verb_order)] for k in range(verbs_per_orientation)]
            verb_cursor += verbs_per_orientation
            for lemma, past in chosen:
                for construction in constructions:
                    for role in ROLES:
                        for qv in QUESTION_VOICES:
                            items.append(
                                build_item(
                                    agent, patient, lemma, past, construction, role, qv,
                                    family_id=family, split=split, test_set=test_set,
                                )
                            )
    return items


def filter_names(names: Iterable[str], is_single_token: Callable[[str], bool]) -> tuple[str, ...]:
    """Keep names the tokenizer encodes as one token (pass e.g. a function that
    checks ``len(tok.encode(" " + name, add_special_tokens=False)) == 1``)."""
    return tuple(n for n in names if is_single_token(n))


def label_correlations(items: Iterable[Item]) -> dict[str, float]:
    """Pearson correlations between the three name-token labels, one
    observation per name token per distinct sentence."""
    seen = set()
    rows = []
    for it in items:
        key = (it.event_id, it.construction)
        if key in seen:
            continue
        seen.add(key)
        for tok in it.names:
            rows.append((float(tok.is_first), float(tok.is_subject), float(tok.is_agent)))
    if not rows:
        raise ValueError("no items")

    def corr(i: int, j: int) -> float:
        xs = [r[i] for r in rows]
        ys = [r[j] for r in rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx == 0 or vy == 0:
            return float("nan")
        return cov / (vx * vy) ** 0.5

    return {"first_subject": corr(0, 1), "first_agent": corr(0, 2), "subject_agent": corr(1, 2)}


def item_to_json(item: Item) -> dict:
    d = asdict(item)
    d["names"] = [asdict(t) for t in item.names]
    d["spans"] = [list(sp) for sp in item.spans]
    return d


def write_dataset(items: Sequence[Item], out_dir: Path, config: dict) -> dict:
    """Write items.jsonl plus a manifest with its SHA-256 and the config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item_to_json(it), sort_keys=True) for it in items]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / "items.jsonl").write_bytes(payload)
    manifest = {
        "generator": "relational_lab.experiments.order_role_stimuli",
        "generator_version": GENERATOR_VERSION,
        "config": config,
        "n_items": len(items),
        "n_families": len({it.family_id for it in items}),
        "n_events": len({it.event_id for it in items}),
        "items_sha256": hashlib.sha256(payload).hexdigest(),
        "label_correlations_main": label_correlations(i for i in items if i.construction in MAIN_CONSTRUCTIONS),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--verbs-per-orientation", type=int, default=2)
    p.add_argument("--dev-fraction", type=float, default=0.15)
    p.add_argument("--n-test-sets", type=int, default=4)
    p.add_argument("--no-controls", action="store_true")
    args = p.parse_args(argv)
    config = {
        "seed": args.seed,
        "verbs_per_orientation": args.verbs_per_orientation,
        "dev_fraction": args.dev_fraction,
        "n_test_sets": args.n_test_sets,
        "include_controls": not args.no_controls,
        "names": list(NAMES),
        "verbs": [list(v) for v in VERBS],
    }
    items = generate_items(
        seed=args.seed,
        verbs_per_orientation=args.verbs_per_orientation,
        dev_fraction=args.dev_fraction,
        n_test_sets=args.n_test_sets,
        include_controls=not args.no_controls,
    )
    manifest = write_dataset(items, args.out, config)
    print(json.dumps({k: manifest[k] for k in ("n_items", "n_families", "n_events", "items_sha256")}, indent=2))


if __name__ == "__main__":
    main()
