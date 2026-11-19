"""Swedish version of the Order-to-Role stimuli (SV 0.1.0).

STATUS: NOT YET NATIVE-CHECKED. Written 19 Nov 2026; to be checked by a native speaker
before any Swedish result is reported. Bump SV_GENERATOR_VERSION after the check.

Same 24 names as English, same family splits as ``order_role_stimuli`` (dev never touches
OTR test families), 16 reversible transitive verbs without particles or prepositional
objects. Names have no gender agreement in Swedish (participles of common-gender nouns),
so the participle is invariant here.

Constructions (Daniel = agent, Laura = patient):

  active         Daniel berömde Laura.
  passive        Laura blev berömd av Daniel.            (periphrastic "bli"-passive)
  object_cleft   Det var Laura som Daniel berömde.
  by_cleft       Det var av Daniel som Laura blev berömd.
  subject_cleft  Det var Daniel som berömde Laura.        (control)

No OVS construction: with two bare names a Swedish V2 clause "Laura berömde Daniel" is
ambiguous (topicalised object or subject), so it cannot serve as an unambiguous item.

Questions never name a participant. Plain "Vem berömde någon?" is ambiguous between subject
and object questions in V2 Swedish, so the default active-voice questions are clefted:
  agent   "Vem var det som berömde någon?"      (som + verb: subject gap)
  patient "Vem var det som någon berömde?"      (som + subject + verb: object gap)
  agent, passive voice   "Av vem blev någon berömd?"   (pied-piped)
  patient, passive voice "Vem blev berömd av någon?"
Swedish also allows preposition stranding, which gives the direct analogue of the failing
English question:  "Vem blev någon berömd av?"  (template "Vem blev någon {pp} av?").

Question templates for the generation check use {past} (preteritum), {pp} (participle),
{spass} (s-passive preteritum) and {lemma} (infinitive).
"""

from __future__ import annotations

import random
import re
from itertools import combinations

from relational_lab.experiments import order_role_stimuli as ors

SV_GENERATOR_VERSION = "0.1.0-unchecked"
NATIVE_CHECKED = False

# (infinitive, preteritum, participle (common gender), s-passive preteritum)
VERBS: tuple[tuple[str, str, str, str], ...] = (
    ("berömma", "berömde", "berömd", "berömdes"),
    ("hjälpa", "hjälpte", "hjälpt", "hjälptes"),
    ("följa", "följde", "följd", "följdes"),
    ("gratulera", "gratulerade", "gratulerad", "gratulerades"),
    ("besöka", "besökte", "besökt", "besöktes"),
    ("kontakta", "kontaktade", "kontaktad", "kontaktades"),
    ("knuffa", "knuffade", "knuffad", "knuffades"),
    ("observera", "observerade", "observerad", "observerades"),
    ("jaga", "jagade", "jagad", "jagades"),
    ("välkomna", "välkomnade", "välkomnad", "välkomnades"),
    ("varna", "varnade", "varnad", "varnades"),
    ("beundra", "beundrade", "beundrad", "beundrades"),
    ("kritisera", "kritiserade", "kritiserad", "kritiserades"),
    ("anklaga", "anklagade", "anklagad", "anklagades"),
    ("rädda", "räddade", "räddad", "räddades"),
    ("krama", "kramade", "kramad", "kramades"),
)
_BY_LEMMA = {v[0]: v for v in VERBS}
SV_MAIN_CONSTRUCTIONS = ("active", "passive", "object_cleft", "by_cleft")
SV_CONSTRUCTIONS = SV_MAIN_CONSTRUCTIONS + ("subject_cleft",)
PROMPT = "Mening: {sentence}\nFråga: {question}\nSvar:"
INSTRUCTIONS = {"name": "Svara med ett enda namn.", "sentence": "Svara med en fullständig mening."}
# (first-mentioned, subject) for each construction, as "agent"/"patient"
_FLAGS = {"active": ("agent", "agent"), "passive": ("patient", "patient"), "object_cleft": ("patient", "agent"),
          "by_cleft": ("agent", "patient"), "subject_cleft": ("agent", "agent")}


def render_sentence(construction: str, a: str, p: str, past: str, pp: str) -> str:
    if construction == "active":
        return f"{a} {past} {p}."
    if construction == "passive":
        return f"{p} blev {pp} av {a}."
    if construction == "object_cleft":
        return f"Det var {p} som {a} {past}."
    if construction == "by_cleft":
        return f"Det var av {a} som {p} blev {pp}."
    if construction == "subject_cleft":
        return f"Det var {a} som {past} {p}."
    raise ValueError(f"unknown construction: {construction}")


def render_question(role: str, voice: str, past: str, pp: str) -> str:
    return {("agent", "active"): f"Vem var det som {past} någon?",
            ("patient", "active"): f"Vem var det som någon {past}?",
            ("agent", "passive"): f"Av vem blev någon {pp}?",
            ("patient", "passive"): f"Vem blev {pp} av någon?"}[(role, voice)]


def fill_template(template: str, past: str, lemma: str) -> str:
    """{past}, {pp}, {spass}, {lemma} for one verb, e.g. "Vem blev någon {pp} av?"."""
    _lm, pret, pp, spass = _BY_LEMMA[lemma]
    assert pret == past, (pret, past)
    return template.format(past=past, pp=pp, spass=spass, lemma=lemma)


def _name_span(sentence: str, name: str) -> tuple[int, int]:
    m = list(re.finditer(rf"\b{re.escape(name)}\b", sentence))
    if len(m) != 1:
        raise AssertionError(f"{name!r} must occur once in {sentence!r}")
    return m[0].start(), m[0].end()


def build_item(a, p, lemma, construction, role, voice, family_id=None, split="dev", test_set=-1):
    _lm, past, pp, _sp = _BY_LEMMA[lemma]
    family = family_id or "|".join(sorted((a, p)))
    sentence = render_sentence(construction, a, p, past, pp)
    question = render_question(role, voice, past, pp)
    prompt = PROMPT.format(sentence=sentence, question=question)
    off = prompt.index(sentence)
    first_role, subj_role = _FLAGS[construction]
    toks = []
    for name, is_agent in ((a, True), (p, False)):
        s, e = _name_span(sentence, name)
        this_role = "agent" if is_agent else "patient"
        toks.append(ors.NameToken(name=name, char_start=s, char_end=e, prompt_char_start=off + s,
                                  prompt_char_end=off + e, is_first=(this_role == first_role),
                                  is_subject=(this_role == subj_role), is_agent=is_agent))
    toks.sort(key=lambda t: t.char_start)
    answer, distractor = (a, p) if role == "agent" else (p, a)
    q0 = prompt.index(question)
    spans = (("first_name", toks[0].prompt_char_start, toks[0].prompt_char_end),
             ("second_name", toks[1].prompt_char_start, toks[1].prompt_char_end),
             ("period", off + len(sentence) - 1, off + len(sentence)),
             ("question_end", q0 + len(question) - 1, q0 + len(question)),
             ("answer_cue", len(prompt) - 1, len(prompt)))
    event = f"sv:{a}>{p}:{lemma}"
    return ors.Item(item_id=f"{event}:{construction}:{role}:{voice}", family_id=family, event_id=event,
                    split=split, test_set=test_set, construction=construction,
                    is_control=construction == "subject_cleft", is_heldout=False, agent=a, patient=p,
                    lemma=lemma, verb=past, role_asked=role, question_voice=voice, sentence=sentence,
                    question=question, prompt=prompt, answer=answer, distractor=distractor,
                    answer_is_first=(toks[0].name == answer), names=(toks[0], toks[1]), spans=spans)


def generate_items(verbs_per_orientation: int = 2, seed: int = 2026, dev_fraction: float = 0.15,
                   n_test_sets: int = 4):
    """Every family, both role assignments, every construction x role x voice (as OTR)."""
    splits = ors.split_families(ors.NAMES, seed, dev_fraction, n_test_sets)
    order = [v[0] for v in VERBS]
    random.Random(f"{seed}:sv").shuffle(order)
    cursor = 0
    items = []
    for x, y in combinations(sorted(ors.NAMES), 2):
        family = f"{x}|{y}"
        split, test_set = splits[family]
        for a, p in ((x, y), (y, x)):
            chosen = [order[(cursor + k) % len(order)] for k in range(verbs_per_orientation)]
            cursor += verbs_per_orientation
            for lemma in chosen:
                for cons in SV_CONSTRUCTIONS:
                    for role in ("agent", "patient"):
                        for voice in ("active", "passive"):
                            items.append(build_item(a, p, lemma, cons, role, voice,
                                                    family_id=family, split=split, test_set=test_set))
    return items


def sample_for_check(n: int = 60, seed: int = 2026):
    """A small, varied sample (every construction, question type and verb) for the native check."""
    items = [i for i in generate_items(seed=seed) if i.split == "dev"]
    rng = random.Random(seed)
    rng.shuffle(items)
    seen, out = set(), []
    for it in items:
        key = (it.construction, it.role_asked, it.question_voice)
        if key in seen and len(out) < 20:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= n:
            break
    return out
