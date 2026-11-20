"""Adapter: run the OTR mechanism experiments (OTR-03 transplants, OTR-09 repair) on the
broad English stimuli (``broad_stimuli``, BX).

The OTR machinery speaks of "agent" and "patient" questions and of ``NameToken.is_agent``.
In BX every block has an "A slot" (agent / agent / stimulus) and a "B slot" (patient /
recipient / experiencer), and ``is_agent`` already marks the A slot. ``normalize`` maps the
asked role onto that vocabulary (A-slot role -> "agent", B-slot role -> "patient") so that
``order_role_transplant`` works unchanged; ``real_role`` recovers the block's own role name
from the item id.

Psych verbs: frighten-class ("se_") and fear-class ("es_") constructions use different verb
lists, and a transplant needs source and base to share the verb, so psych sources come from
the base's own class (3 constructions: active, passive, object cleft). Within one class order,
function and role are not fully decorrelated (the fourth pattern is missing); the predictor
correlations are reported by OTR-03.

The failing question, "Who was someone {pp} by?", asks for the role a passive by-phrase
expresses for the verb (``broad_stimuli.by_phrase_role``): the A slot, except for fear-class
verbs, where it is the experiencer (B slot).
"""

from __future__ import annotations

import dataclasses
import random

from relational_lab.experiments import broad_stimuli as bx

STIMULI = ("bx-transitive", "bx-ditransitive", "bx-psych")

# Constructions used as bases (and, per class, as transplant sources).
BASE_CONSTRUCTIONS = {
    "transitive": bx.T_MAIN,
    "ditransitive": bx.D_MAIN,
    "psych": ("se_active", "se_passive", "se_object_cleft", "es_active", "es_passive", "es_object_cleft"),
}
# Left-context validity null: sources whose first name directly follows "Sentence: ".
NULL_CONSTRUCTIONS = {
    "transitive": ("active", "passive"),
    "ditransitive": ("do_active", "recipient_passive"),
    "psych": ("se_active", "se_passive", "es_active", "es_passive"),
}
# Default failing (base) and working (source) questions for OTR-09.
DEFAULT_BASE_QUESTION = {
    "transitive": "Who was someone {pp} by?",
    "ditransitive": "Who was someone {pp} {theme} by?",
    "psych": "Who was someone {pp} by?",
}
DEFAULT_SOURCE_QUESTION = "Who was the {ing} done by?"


def block_name(stimuli: str) -> str:
    if stimuli not in STIMULI:
        raise ValueError(f"stimuli must be one of {STIMULI}")
    return stimuli.removeprefix("bx-")


def block_of(item) -> str:
    return item.event_id.split(":")[0].removeprefix("bx-")


def real_role(item) -> str:
    """The block's own name for the asked role (item ids end in ':construction:role:voice')."""
    return item.item_id.split(":")[-2]


def normalize(item):
    """role_asked -> "agent" (A slot) / "patient" (B slot); everything else unchanged."""
    a_role = bx.ROLES[block_of(item)][0]
    role = "agent" if real_role(item) == a_role else "patient"
    return dataclasses.replace(item, role_asked=role)


def source_constructions(base) -> tuple[str, ...]:
    block = block_of(base)
    if block == "psych":
        cls = base.construction.split("_")[0]
        return tuple(c for c in BASE_CONSTRUCTIONS["psych"] if c.startswith(cls + "_"))
    return BASE_CONSTRUCTIONS[block]


def builder(base, a: str, b: str, construction: str):
    """Source item for a transplant: same verb, theme, role and question voice as ``base``."""
    src = bx.build_item(block_of(base), a, b, base.lemma, construction, real_role(base), base.question_voice,
                        bx.theme_of(base), family_id=base.family_id, split=base.split, test_set=base.test_set)
    return normalize(src)


def ask(item, template: str, instruction: str = ""):
    """Replace the item's question by ``template`` ({past} {pp} {ing} {lemma} {theme}); the
    sentence, and so every name position, is untouched."""
    q = bx.fill_template(template, item.lemma, bx.theme_of(item))
    prompt = item.prompt.replace(f"\nQuestion: {item.question}\n", f"\nQuestion: {q}\n")
    assert prompt.count(q) == 1, prompt
    if instruction:
        prompt = prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:")
    return dataclasses.replace(item, question=q, prompt=prompt)


def select_bases(stimuli: str, split: str, n_events: int, voices, seed: int, roles=("agent", "patient")):
    """Normalized base items from ``n_events`` sampled events (all base constructions, both
    roles unless restricted, the given question voices)."""
    block = block_name(stimuli)
    cons = BASE_CONSTRUCTIONS[block]
    pool = [normalize(i) for i in bx.generate_items(block, split=split, seed=seed, constructions_subset=cons)
            if i.question_voice in set(voices)]
    events = sorted({i.event_id for i in pool})
    random.Random(seed).shuffle(events)
    keep = set(events[:n_events])
    return [i for i in pool if i.event_id in keep and i.role_asked in set(roles)]


def with_by_question(items, template: str):
    """Keep the items that ask for the by-phrase role and give them ``template``."""
    return [ask(i, template) for i in items if real_role(i) == bx.by_phrase_role(i)]


def repair_items(stimuli: str, seed: int, constructions, split: str = "dev"):
    """OTR-09 pool: one item per (event, construction) asking for the by-phrase role
    (active question voice; the question is replaced anyway)."""
    block = block_name(stimuli)
    return [i for i in bx.generate_items(block, split=split, seed=seed, constructions_subset=tuple(constructions))
            if i.question_voice == "active" and real_role(i) == bx.by_phrase_role(i)]
