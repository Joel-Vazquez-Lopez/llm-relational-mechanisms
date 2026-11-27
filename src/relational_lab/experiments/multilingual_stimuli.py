"""Spanish and Galician versions of the Order-to-Role stimuli (XL 1.0.0).

Native check: Joel Vázquez López (Spanish, Galician), 17 Nov 2026.

Same 24 names as English (first 12 female, last 12 male), same family splits as
``order_role_stimuli`` (so dev never touches OTR test families), 16 reversible verbs
(English "thank" and "tickle" replaced by "congratulate" and "hug", which take a direct
object in both languages). Passive participles agree with the patient's gender.

Constructions (Daniel = agent, Laura = patient):

  active         es  Daniel elogió a Laura.                       gl  Daniel eloxiou a Laura.
  passive        es  Laura fue elogiada por Daniel.               gl  Laura foi eloxiada por Daniel.
  object_cleft   es  Fue a Laura a quien elogió Daniel.           gl  Foi a Laura a quen eloxiou Daniel.
  by_cleft       es  Fue por Daniel por quien Laura fue elogiada. gl  Foi por Daniel por quen Laura foi eloxiada.
  subject_cleft  es  Fue Daniel quien elogió a Laura.             gl  Foi Daniel quen eloxiou a Laura.
  ovs_clitic     es  A Laura la elogió Daniel.                    gl  A Laura eloxiouna Daniel.

``ovs_clitic`` has no English counterpart: the patient comes first, and only the object
marker "a" and the clitic mark her role (first = patient, subject = agent).

Questions never name a participant; prepositions cannot be stranded, so the passive
agent question is pied-piped ("¿Por quién fue elogiado alguien?").
Items are ``order_role_stimuli.Item`` objects with name flags and position spans, so the
generation check, the rescoring and the probes can consume them unchanged.
"""

from __future__ import annotations

import random
import re
from itertools import combinations

from relational_lab.experiments import order_role_stimuli as ors

XL_GENERATOR_VERSION = "1.0.0"
FEMALE = set(ors.NAMES[:12])
LANGS = ("es", "gl")

# (lemma, preterite 3sg, participle stem without -o/-a)
VERBS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "es": (
        ("elogiar", "elogió", "elogiad"), ("ayudar", "ayudó", "ayudad"), ("seguir", "siguió", "seguid"),
        ("felicitar", "felicitó", "felicitad"), ("visitar", "visitó", "visitad"), ("llamar", "llamó", "llamad"),
        ("empujar", "empujó", "empujad"), ("observar", "observó", "observad"), ("perseguir", "persiguió", "perseguid"),
        ("saludar", "saludó", "saludad"), ("advertir", "advirtió", "advertid"), ("admirar", "admiró", "admirad"),
        ("invitar", "invitó", "invitad"), ("culpar", "culpó", "culpad"), ("rescatar", "rescató", "rescatad"),
        ("abrazar", "abrazó", "abrazad"),
    ),
    "gl": (
        ("eloxiar", "eloxiou", "eloxiad"), ("axudar", "axudou", "axudad"), ("seguir", "seguiu", "seguid"),
        ("felicitar", "felicitou", "felicitad"), ("visitar", "visitou", "visitad"), ("chamar", "chamou", "chamad"),
        ("empurrar", "empurrou", "empurrad"), ("observar", "observou", "observad"), ("perseguir", "perseguiu", "perseguid"),
        ("saudar", "saudou", "saudad"), ("advertir", "advertiu", "advertid"), ("admirar", "admirou", "admirad"),
        ("convidar", "convidou", "convidad"), ("culpar", "culpou", "culpad"), ("rescatar", "rescatou", "rescatad"),
        ("abrazar", "abrazou", "abrazad"),
    ),
}
XL_MAIN_CONSTRUCTIONS = ("active", "passive", "object_cleft", "by_cleft", "ovs_clitic")
XL_CONSTRUCTIONS = XL_MAIN_CONSTRUCTIONS + ("subject_cleft",)
PROMPT = {"es": "Oración: {sentence}\nPregunta: {question}\nRespuesta:",
          "gl": "Oración: {sentence}\nPregunta: {question}\nResposta:"}
INSTRUCTIONS = {
    "es": {"name": "Responde con un solo nombre.", "sentence": "Responde con una frase completa."},
    "gl": {"name": "Responde cun só nome.", "sentence": "Responde cunha frase completa."},
}
# (first-mentioned, subject) for each construction, as "agent"/"patient"
_FLAGS = {"active": ("agent", "agent"), "passive": ("patient", "patient"), "object_cleft": ("patient", "agent"),
          "by_cleft": ("agent", "patient"), "subject_cleft": ("agent", "agent"), "ovs_clitic": ("patient", "agent")}


def _pp(stem: str, name: str) -> str:
    return stem + ("a" if name in FEMALE else "o")


def render_sentence(lang: str, construction: str, a: str, p: str, pret: str, stem: str) -> str:
    fue, quien = ("fue", "quien") if lang == "es" else ("foi", "quen")
    Fue = fue.capitalize()
    pp = _pp(stem, p)
    if construction == "active":
        return f"{a} {pret} a {p}."
    if construction == "passive":
        return f"{p} {fue} {pp} por {a}."
    if construction == "object_cleft":
        return f"{Fue} a {p} a {quien} {pret} {a}."
    if construction == "by_cleft":
        return f"{Fue} por {a} por {quien} {p} {fue} {pp}."
    if construction == "subject_cleft":
        return f"{Fue} {a} {quien} {pret} a {p}."
    if construction == "ovs_clitic":
        if lang == "es":
            return f"A {p} {'la' if p in FEMALE else 'lo'} {pret} {a}."
        # Galician enclisis; after a verb ending in a diphthong the clitic is -na / -no
        return f"A {p} {pret}{'na' if p in FEMALE else 'no'} {a}."
    raise ValueError(f"unknown construction: {construction}")


def render_question(lang: str, role: str, voice: str, pret: str, stem: str) -> str:
    if lang == "es":
        q = {("agent", "active"): f"¿Quién {pret} a alguien?", ("patient", "active"): f"¿A quién {pret} alguien?",
             ("agent", "passive"): f"¿Por quién fue {stem}o alguien?",
             ("patient", "passive"): f"¿Quién fue {stem}o por alguien?"}
    else:
        q = {("agent", "active"): f"Quen {pret} a alguén?", ("patient", "active"): f"A quen {pret} alguén?",
             ("agent", "passive"): f"Por quen foi {stem}o alguén?",
             ("patient", "passive"): f"Quen foi {stem}o por alguén?"}
    return q[(role, voice)]


def _name_span(sentence: str, name: str) -> tuple[int, int]:
    m = [x for x in re.finditer(rf"\b{re.escape(name)}\b", sentence)]
    if len(m) != 1:
        raise AssertionError(f"{name!r} must occur once in {sentence!r}")
    return m[0].start(), m[0].end()


def build_item(lang, a, p, lemma, pret, stem, construction, role, voice, family_id=None, split="dev", test_set=-1):
    family = family_id or "|".join(sorted((a, p)))
    sentence = render_sentence(lang, construction, a, p, pret, stem)
    question = render_question(lang, role, voice, pret, stem)
    prompt = PROMPT[lang].format(sentence=sentence, question=question)
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
    event = f"{lang}:{a}>{p}:{lemma}"
    return ors.Item(item_id=f"{event}:{construction}:{role}:{voice}", family_id=family, event_id=event,
                    split=split, test_set=test_set, construction=construction,
                    is_control=construction == "subject_cleft", is_heldout=False, agent=a, patient=p,
                    lemma=lemma, verb=pret, role_asked=role, question_voice=voice, sentence=sentence,
                    question=question, prompt=prompt, answer=answer, distractor=distractor,
                    answer_is_first=(toks[0].name == answer), names=(toks[0], toks[1]), spans=spans)


def generate_items(lang: str, verbs_per_orientation: int = 2, seed: int = 2026, dev_fraction: float = 0.15,
                   n_test_sets: int = 4):
    """Every family, both role assignments, every construction x role x voice (as OTR)."""
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}")
    splits = ors.split_families(ors.NAMES, seed, dev_fraction, n_test_sets)
    order = list(VERBS[lang])
    random.Random(f"{seed}:xl:{lang}").shuffle(order)
    cursor = 0
    items = []
    for x, y in combinations(sorted(ors.NAMES), 2):
        family = f"{x}|{y}"
        split, test_set = splits[family]
        for a, p in ((x, y), (y, x)):
            chosen = [order[(cursor + k) % len(order)] for k in range(verbs_per_orientation)]
            cursor += verbs_per_orientation
            for lemma, pret, stem in chosen:
                for cons in XL_CONSTRUCTIONS:
                    for role in ("agent", "patient"):
                        for voice in ("active", "passive"):
                            items.append(build_item(lang, a, p, lemma, pret, stem, cons, role, voice,
                                                    family_id=family, split=split, test_set=test_set))
    return items
