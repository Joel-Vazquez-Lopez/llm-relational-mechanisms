"""Stranded-preposition (SP) stimuli: does the stranded-"by" role reversal generalise?

Three frames with reversible human participants (doer A, prepositional object R):

  to    (dative)       "Leon gave the book to Mira."  /  "Leon gave Mira the book."
  for   (benefactive)  "Leon baked a pie for Mira."   /  "Leon baked Mira a pie."
  from  (source)       "Leon borrowed the bike from Mira."   (no alternation)

Sentence constructions (``construction`` = "<frame>_<form>"):
  po              prepositional:   A V-ed OBJ P R.            ("P R" in the sentence: aligned)
  do              double object:   A V-ed R OBJ.              (no P: like the active for "by")
  cleft_stranded  It was R that A V-ed OBJ P.
  cleft_piped     It was P R that A V-ed OBJ.
"do" exists only for "to" and "for".

Questions (``question_voice`` = question form; never name a participant):
  stranded  (asks R)  "Who did someone give the book to?"     gap at the very end
  piped     (asks R)  "To whom did someone give the book?"    cue first
  agent     (asks A)  "Who gave the book to someone?"         control

``role_asked`` is "agent" (A) or "oblique" (R). ``agent``/``patient`` hold A/R so the
generation check and answer extractor can be shared with OTR (``patient`` = R).

Alignment account predicts stranded errors where the sentence lacks "P R" (do,
cleft_stranded); the late-cue account predicts stranded < piped everywhere.

Families and splits are exactly OTR's (``order_role_stimuli.split_families``), so the
dev split here never touches OTR test families.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from relational_lab.experiments import order_role_stimuli as ors

SP_GENERATOR_VERSION = "1.0.0"

# frame -> (preposition, alternates with double object?, ((lemma, past, object), ...))
FRAMES: dict[str, tuple[str, bool, tuple[tuple[str, str, str], ...]]] = {
    "to": ("to", True, (
        ("give", "gave", "the book"), ("send", "sent", "a letter"), ("show", "showed", "the photo"),
        ("hand", "handed", "the keys"), ("throw", "threw", "the ball"), ("sell", "sold", "the car"),
        ("lend", "lent", "the bike"), ("pass", "passed", "the salt"), ("offer", "offered", "a job"),
        ("bring", "brought", "the coffee"),
    )),
    "for": ("for", True, (
        ("buy", "bought", "a gift"), ("make", "made", "a cake"), ("bake", "baked", "a pie"),
        ("cook", "cooked", "dinner"), ("build", "built", "a shelf"), ("find", "found", "a seat"),
        ("get", "got", "a coffee"), ("pour", "poured", "a drink"), ("draw", "drew", "a picture"),
        ("knit", "knitted", "a scarf"),
    )),
    "from": ("from", False, (
        ("borrow", "borrowed", "the bike"), ("steal", "stole", "the hat"), ("buy", "bought", "the car"),
        ("take", "took", "the keys"), ("receive", "received", "a letter"), ("rent", "rented", "the flat"),
        ("learn", "learned", "the song"), ("hide", "hid", "the key"),
    )),
}
FORMS_ALTERNATING = ("po", "do", "cleft_stranded", "cleft_piped")
FORMS_PLAIN = ("po", "cleft_stranded", "cleft_piped")
QUESTION_FORMS: dict[str, str] = {"stranded": "oblique", "piped": "oblique", "agent": "agent"}


@dataclass(frozen=True)
class SPItem:
    item_id: str
    family_id: str
    event_id: str
    split: str
    test_set: int
    frame: str
    form: str
    construction: str  # "<frame>_<form>"
    agent: str  # doer A
    patient: str  # prepositional object R (recipient / beneficiary / source)
    lemma: str
    verb: str  # past tense
    obj: str
    prep: str
    role_asked: str  # "agent" | "oblique"
    question_voice: str  # question form: "stranded" | "piped" | "agent"
    sentence: str
    question: str
    prompt: str
    answer: str
    distractor: str


def render_sentence(form: str, a: str, r: str, past: str, obj: str, prep: str) -> str:
    if form == "po":
        return f"{a} {past} {obj} {prep} {r}."
    if form == "do":
        return f"{a} {past} {r} {obj}."
    if form == "cleft_stranded":
        return f"It was {r} that {a} {past} {obj} {prep}."
    if form == "cleft_piped":
        return f"It was {prep} {r} that {a} {past} {obj}."
    raise ValueError(f"unknown form: {form}")


def render_question(qform: str, lemma: str, past: str, obj: str, prep: str) -> str:
    if qform == "stranded":
        return f"Who did someone {lemma} {obj} {prep}?"
    if qform == "piped":
        return f"{prep.capitalize()} whom did someone {lemma} {obj}?"
    if qform == "agent":
        return f"Who {past} {obj} {prep} someone?"
    raise ValueError(f"unknown question form: {qform}")


def build_item(a, r, frame, lemma, past, obj, form, qform, family_id=None, split="dev", test_set=-1) -> SPItem:
    prep = FRAMES[frame][0]
    family = family_id or "|".join(sorted((a, r)))
    event = f"{a}>{r}:{lemma}:{frame}"
    role = QUESTION_FORMS[qform]
    sentence = render_sentence(form, a, r, past, obj, prep)
    question = render_question(qform, lemma, past, obj, prep)
    answer, distractor = (a, r) if role == "agent" else (r, a)
    return SPItem(
        item_id=f"{event}:{form}:{role}:{qform}", family_id=family, event_id=event, split=split,
        test_set=test_set, frame=frame, form=form, construction=f"{frame}_{form}", agent=a, patient=r,
        lemma=lemma, verb=past, obj=obj, prep=prep, role_asked=role, question_voice=qform,
        sentence=sentence, question=question,
        prompt=ors.PROMPT_TEMPLATE.format(sentence=sentence, question=question),
        answer=answer, distractor=distractor,
    )


def generate_items(names: Sequence[str] = ors.NAMES, frames: Sequence[str] = tuple(FRAMES),
                   verbs_per_orientation: int = 2, seed: int = 2026, dev_fraction: float = 0.15,
                   n_test_sets: int = 4) -> list[SPItem]:
    """Every family, both role assignments, every frame, form and question form.
    Verbs rotate through a shuffled list per frame (as in OTR)."""
    splits = ors.split_families(names, seed, dev_fraction, n_test_sets)
    items: list[SPItem] = []
    for frame in frames:
        _, alternates, verbs = FRAMES[frame]
        forms = FORMS_ALTERNATING if alternates else FORMS_PLAIN
        order = list(verbs)
        random.Random(f"{seed}:{frame}").shuffle(order)
        cursor = 0
        for x, y in combinations(sorted(names), 2):
            family = f"{x}|{y}"
            split, test_set = splits[family]
            for a, r in ((x, y), (y, x)):
                chosen = [order[(cursor + k) % len(order)] for k in range(verbs_per_orientation)]
                cursor += verbs_per_orientation
                for lemma, past, obj in chosen:
                    for form in forms:
                        for qform in QUESTION_FORMS:
                            items.append(build_item(a, r, frame, lemma, past, obj, form, qform,
                                                    family_id=family, split=split, test_set=test_set))
    return items
