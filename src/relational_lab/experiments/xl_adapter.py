"""Adapter: run OTR-03 (name-state transplants) and OTR-09 (answer repair) on the Spanish and
Galician stimuli (``multilingual_stimuli``, XL).

XL uses the OTR names and the same decorrelated constructions (active, passive, object cleft,
by-cleft), so the transplant design carries over unchanged; only the sentence builder differs.

Questions. The failing question is XL's default passive agent question ("¿Por quién fue
V-ado alguien?" / "Por quen foi V-ado alguén?"); the working source question for OTR-09 is the
active agent question ("¿Quién V a alguien?" / "Quen V a alguén?"), which every model answers
near ceiling on actives, passives and (Spanish) object clefts. By-clefts and the OVS-clitic
construction are left out of OTR-09 by default: the working question itself is unreliable there
(by-cleft possibly read causally; Galician OVS article), pending Joel's judgements.
"""

from __future__ import annotations

import dataclasses
import random

from relational_lab.experiments import multilingual_stimuli as xl
from relational_lab.experiments.splits import in_split

STIMULI = xl.LANGS  # ("es", "gl")
TRANSPLANT_CONSTRUCTIONS = ("active", "passive", "object_cleft", "by_cleft")
REPAIR_CONSTRUCTIONS = {"es": ("active", "passive", "object_cleft"), "gl": ("active", "passive")}
ANSWER_CUE = {"es": "\nRespuesta:", "gl": "\nResposta:"}


def lang_of(item) -> str:
    return item.event_id.split(":")[0]


def _stem(lang: str, lemma: str) -> str:
    return {lm: st for lm, _pr, st in xl.VERBS[lang]}[lemma]


def builder(base, a: str, p: str, construction: str):
    lang = lang_of(base)
    return xl.build_item(lang, a, p, base.lemma, base.verb, _stem(lang, base.lemma), construction, base.role_asked,
                         base.question_voice, family_id=base.family_id, split=base.split, test_set=base.test_set)


def select_bases(lang: str, split: str, n_events: int, voices, roles, seed: int,
                 constructions=TRANSPLANT_CONSTRUCTIONS):
    pool = [i for i in xl.generate_items(lang, seed=seed)
            if in_split(i, split) and i.construction in constructions and i.question_voice in set(voices)]
    events = sorted({i.event_id for i in pool})
    random.Random(seed).shuffle(events)
    keep = set(events[:n_events])
    return [i for i in pool if i.event_id in keep and i.role_asked in set(roles)]


def with_instruction(item, instruction: str):
    if not instruction:
        return item
    cue = ANSWER_CUE[lang_of(item)]
    assert item.prompt.endswith(cue), item.prompt
    return dataclasses.replace(item, prompt=item.prompt[: -len(cue)] + f"\n{instruction}{cue}")


def repair_pairs(lang: str, seed: int, constructions, instruction: str, split: str = "dev"):
    """(base, source) per (event, construction): same sentence; base = failing passive agent
    question, source = active agent question."""
    items = {i.item_id: i for i in xl.generate_items(lang, seed=seed)
             if in_split(i, split) and i.construction in constructions and i.role_asked == "agent"}
    out = []
    for iid, it in items.items():
        if it.question_voice != "passive":
            continue
        src = items[iid.removesuffix(":passive") + ":active"]
        out.append((with_instruction(it, instruction), with_instruction(src, instruction)))
    return out
