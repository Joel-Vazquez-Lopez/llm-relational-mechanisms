"""OTR-05: wording and format robustness for the order-to-role behaviour.

Answers the objection "is the object-question failure just this prompt
wording?". The sentences, names, splits and families are exactly those of the
v1.3.0 generator; only the question part changes. Each variant asks for one
role, in one of three formats:

wh
    Other wordings of the same question ("Whom did someone praise?",
    "Which person did someone praise?", "Who was it that someone praised?").
cloze
    No question at all: complete a statement ("... The one whom someone
    praised was").
orig
    The four v1.3.0 questions, so every run carries its own reference.

Each variant is also tagged with its *extraction type*: subject ("Who praised
someone?", "Who was praised by someone?"), object ("Who did someone praise?"),
or oblique ("By whom was someone praised?"). If the failure is about object /
oblique extraction, it should follow the tag across wordings, and it may or may
not survive the cloze format (which has no wh-movement in the question).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from relational_lab.experiments.order_role_stimuli import (
    MAIN_CONSTRUCTIONS,
    PROMPT_PREFIX,
    Item,
    generate_items,
)

ROBUSTNESS_VERSION = "1.0.0"


@dataclass(frozen=True)
class Variant:
    name: str
    role: str  # role asked for: agent / patient
    fmt: str  # orig / wh / cloze
    extraction: str  # subject / object / oblique / none (cloze)
    template: str  # uses {past} and {lemma}


VARIANTS: tuple[Variant, ...] = (
    # agent
    Variant("A_orig_active", "agent", "orig", "subject", "Who {past} someone?"),
    Variant("A_which", "agent", "wh", "subject", "Which person {past} someone?"),
    Variant("A_cleft", "agent", "wh", "subject", "Who was it that {past} someone?"),
    Variant("A_orig_passive", "agent", "orig", "oblique", "By whom was someone {past}?"),
    Variant("A_who_by", "agent", "wh", "oblique", "Who was someone {past} by?"),  # the v1.2 stranded form
    Variant("A_cloze", "agent", "cloze", "none", "The one who {past} someone was"),
    # patient
    Variant("P_orig_active", "patient", "orig", "object", "Who did someone {lemma}?"),
    Variant("P_whom", "patient", "wh", "object", "Whom did someone {lemma}?"),
    Variant("P_which", "patient", "wh", "object", "Which person did someone {lemma}?"),
    Variant("P_cleft", "patient", "wh", "object", "Who was it that someone {past}?"),
    Variant("P_orig_passive", "patient", "orig", "subject", "Who was {past} by someone?"),
    Variant("P_cloze_active", "patient", "cloze", "none", "The one whom someone {past} was"),
    Variant("P_cloze_passive", "patient", "cloze", "none", "The one who was {past} by someone was"),
)
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}


def render(item: Item, v: Variant) -> Item:
    """``item`` (any v1.3.0 item asking for v.role) with the question part replaced."""
    if item.role_asked != v.role:
        raise ValueError("variant asks for a different role than the item")
    text = v.template.format(past=item.verb, lemma=item.lemma)
    if v.fmt == "cloze":
        prompt = f"{PROMPT_PREFIX}{item.sentence}\n{text}"
    else:
        prompt = f"{PROMPT_PREFIX}{item.sentence}\nQuestion: {text}\nAnswer:"
    base = item.item_id.rsplit(":", 1)[0]  # drop the original question voice
    return replace(item, item_id=f"{base}:{v.name}", question=text, prompt=prompt, spans={})


def sentence_subject(construction: str) -> str:
    """Role of the grammatical subject of the sentence (main constructions)."""
    return {"active": "agent", "passive": "patient", "object_cleft": "agent", "by_cleft": "patient",
            "subject_cleft": "agent"}[construction]


def make_items(split: str = "dev", constructions: Sequence[str] = MAIN_CONSTRUCTIONS,
               variants: Sequence[Variant] = VARIANTS, n_events: int | None = None, seed: int = 2026) -> list[Item]:
    """One item per (event, construction, variant), from the v1.3.0 generator.
    Uses the active-voice original item of each role as the base (sentence and
    names do not depend on the question)."""
    base = [i for i in generate_items(seed=seed)
            if i.split == split and i.construction in constructions and i.question_voice == "active"]
    if n_events is not None:
        events = []
        for it in base:
            if it.event_id not in events:
                events.append(it.event_id)
        keep = set(events[:n_events])
        base = [i for i in base if i.event_id in keep]
    by_role = defaultdict(list)
    for it in base:
        by_role[it.role_asked].append(it)
    return [render(it, v) for v in variants for it in by_role[v.role]]


def summarize(items: Sequence[Item], margins: dict[str, float]) -> dict:
    """Accuracy per variant x construction, plus how often the chosen name is the
    sentence's grammatical subject (the Qwen-Instruct fallback)."""
    cells: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for it in items:
        m = margins.get(it.item_id)
        if m is None:
            continue
        v = it.item_id.rsplit(":", 1)[1]
        correct = m > 0
        chosen_role = it.role_asked if correct else ("patient" if it.role_asked == "agent" else "agent")
        cells[(v, it.construction)].append((correct, chosen_role == sentence_subject(it.construction)))
    out: dict[str, dict] = {}
    for (v, c), vals in sorted(cells.items()):
        var = VARIANT_BY_NAME[v]
        d = out.setdefault(v, {"role": var.role, "format": var.fmt, "extraction": var.extraction,
                               "template": var.template, "by_construction": {}})
        d["by_construction"][c] = {"n": len(vals), "accuracy": sum(a for a, _ in vals) / len(vals),
                                   "chose_subject": sum(s for _, s in vals) / len(vals)}
    for v, d in out.items():
        bc = d["by_construction"]
        d["accuracy"] = sum(x["accuracy"] for x in bc.values()) / len(bc)
        d["chose_subject"] = sum(x["chose_subject"] for x in bc.values()) / len(bc)
    return out
