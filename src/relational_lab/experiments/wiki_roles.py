"""WIKI-ROLES: natural sentences from English Wikipedia in which one named person does something to
another named person ("Smith was defeated by Jones in 1998", "Obama appointed Sotomayor ...").

Pipeline (scripts/wiki_extract_roles.py): cheap regex prefilter (a verb form from the person-verb
list and capitalised words) -> spaCy parse + NER of candidate sentences -> ``extract_from_doc``:
keep a VERB with lemma in PERSON_VERBS and
  active : one nsubj and one dobj, or
  passive: one nsubjpass and one agent ("by") object,
whose argument phrases are each exactly one PERSON entity, in a sentence with exactly two PERSON
entities, neither name (nor any token of it) occurring elsewhere in the sentence.

Each record keeps the real names; ``neutral`` swaps them for two OTR names (single tokens in every
model) to control for answers recalled from memory about famous people. Records are released
with the paper (Wikipedia text: CC BY-SA 4.0, attribution = article title + revision dump).

0.2.0: names must look like person names (title-case words or initials, no digits, no
organisation words such as "FC", "Party", "University"), the two names may not share a word,
and (``confirm_people``) a second, independent NER model (BERT, CoNLL PER) must also tag both
names as persons.
"""

from __future__ import annotations

import dataclasses
import random
import re

from relational_lab.experiments import order_role_stimuli as ors

WIKI_VERSION = "0.2.0"
CONSTRUCTIONS = ("wiki_active", "wiki_passive")

PERSON_VERBS = sorted(set("""
accuse admire advise appoint arrest assassinate assist attack beat befriend betray bite blame call capture
challenge chase coach congratulate contact convince criticize crown defeat defend depose dismiss divorce
elect employ encourage endorse execute expel fire follow greet help hire honor hug imprison influence
injure inspire instruct interview invite join kidnap kill kiss knight marry meet mentor murder nominate
oppose overthrow pardon persuade praise promote protect punish recruit replace represent rescue
sack save shoot sign stab succeed sue support tackle teach thank threaten train treat tutor visit warn
""".split()))

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_WORD = re.compile(r"[A-Za-z]+")
_CAP = re.compile(r"\b[A-Z][a-z]+")
ORG_WORDS = set("""
fc afc cf sc ac club united city town rovers wanderers athletic county party army navy corps council university
college school institute academy records group bank league ministry government church team association committee
republic kingdom empire company corporation inc ltd llc co plc society union federation foundation brothers
""".split())
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-zA-Z'\-]*\.?|[A-Z]\.|(?:de|van|von|der|den|da|di|du|la|le|bin|al|ibn|ben)\b)$")


def looks_like_person(name: str) -> bool:
    toks = name.split()
    if not 1 <= len(toks) <= 4 or any(ch.isdigit() for ch in name):
        return False
    if {t.lower().strip(".,") for t in toks} & ORG_WORDS:
        return False
    return all(_NAME_TOKEN.match(t) for t in toks) and toks[0][0].isupper()


def verb_forms():
    """lemma -> (past, pp, ing); lemminflect is needed only at extraction time."""
    from lemminflect import getInflection as g
    out = {}
    for lm in PERSON_VERBS:
        past, pp, ing = g(lm, "VBD"), g(lm, "VBN"), g(lm, "VBG")
        if past and pp and ing:
            out[lm] = (past[0], pp[0], ing[0])
    return out


def surface_forms(forms) -> set[str]:
    s = set()
    for lm, (past, pp, ing) in forms.items():
        s |= {lm, past, pp, lm + "s", lm + "es", ing}
    return s


def candidate_sentences(text: str, surface: set[str], max_words: int = 50):
    """Cheap prefilter: a person-verb form and >= 2 capitalised words after the first word."""
    for para in text.split("\n"):
        if len(para) < 30:
            continue
        for s in _SENT_SPLIT.split(para):
            words = _WORD.findall(s)
            if not (6 <= len(words) <= max_words):
                continue
            if not surface.intersection(w.lower() for w in words):
                continue
            if len(_CAP.findall(s[1:])) < 2:
                continue
            yield s.strip()


def _person_span(tok):
    """The PERSON entity that exactly covers ``tok``'s subtree (minus a leading 'the'), else None."""
    if tok.ent_type_ != "PERSON":
        return None
    ent = next((e for e in tok.doc.ents if e.start <= tok.i < e.end and e.label_ == "PERSON"), None)
    if ent is None:
        return None
    sub = [t.i for t in tok.subtree]
    if sub != list(range(ent.start, ent.end)):
        return None
    return ent


def extract_from_doc(doc, forms, meta: dict | None = None):
    """Records for every qualifying verb in a parsed sentence (spaCy Doc of one sentence)."""
    persons = [e for e in doc.ents if e.label_ == "PERSON"]
    if len(persons) != 2:
        return []
    out = []
    for v in doc:
        lm = v.lemma_.lower()
        if v.pos_ != "VERB" or lm not in forms:
            continue
        kids = list(v.children)
        rel = lambda d: [c for c in kids if c.dep_ == d]  # noqa: E731
        if len(rel("nsubj")) == 1 and len(rel("dobj")) == 1 and not rel("auxpass"):
            a, p, kind = _person_span(rel("nsubj")[0]), _person_span(rel("dobj")[0]), "active"
        elif len(rel("nsubjpass")) == 1 and len(rel("agent")) == 1:
            objs = [c for c in rel("agent")[0].children if c.dep_ == "pobj"]
            if len(objs) != 1:
                continue
            a, p, kind = _person_span(objs[0]), _person_span(rel("nsubjpass")[0]), "passive"
        else:
            continue
        if a is None or p is None or a.text == p.text:
            continue
        if not (looks_like_person(a.text) and looks_like_person(p.text)):
            continue
        if {t.lower_ for t in a} & {t.lower_ for t in p}:  # "Wigan Highfield" vs "Wigan"
            continue
        text = doc.text
        name_toks = {t.lower_ for t in a} | {t.lower_ for t in p}
        others = [t.lower_ for t in doc if not (a.start <= t.i < a.end or p.start <= t.i < p.end)]
        if name_toks & set(others):  # a name (part) repeated elsewhere: ambiguous answer
            continue
        past, pp, ing = forms[lm]
        out.append({
            "sentence": text, "kind": kind, "lemma": lm, "past": past, "pp": pp, "ing": ing,
            "agent": a.text, "agent_char": [a.start_char, a.end_char],
            "patient": p.text, "patient_char": [p.start_char, p.end_char],
            "verb_char": [v.idx, v.idx + len(v.text)], **(meta or {}),
        })
    return out


def confirm_people(records, ner):
    """Keep records whose agent and patient are both tagged PER by a second NER model.
    ``ner(sentences) -> list[list[dict(entity_group, start, end)]]`` (a transformers
    token-classification pipeline with aggregation_strategy="simple")."""
    if not records:
        return []
    ents = ner([r["sentence"] for r in records])

    def covered(span, es):
        s, e = span
        return any(x["entity_group"] in ("PER", "PERSON") and x["start"] <= s + 1 and x["end"] >= e - 1 for x in es)

    return [r for r, es in zip(records, ents) if covered(r["agent_char"], es) and covered(r["patient_char"], es)]


# ------------------------------------------------------------------ items for OTR-09

def substitute(rec: dict, a: str, p: str) -> str:
    s = rec["sentence"]
    spans = sorted([(rec["agent_char"], a), (rec["patient_char"], p)], key=lambda x: -x[0][0])
    for (st, en), name in spans:
        s = s[:st] + name + s[en:]
    return s


def build_item(rec: dict, idx: int, question_template: str, names: str = "real", rng: random.Random | None = None,
               pool=ors.NAMES):
    if names == "real":
        a, p, sentence = rec["agent"], rec["patient"], rec["sentence"]
    else:
        text = rec["sentence"]
        a, p = (rng or random.Random(idx)).sample([n for n in pool if n not in text], 2)
        sentence = substitute(rec, a, p)
    question = question_template.format(past=rec["past"], pp=rec["pp"], ing=rec["ing"], lemma=rec["lemma"])
    prompt = ors.PROMPT_TEMPLATE.format(sentence=sentence, question=question)
    off = prompt.index(sentence)
    found = []
    for name, is_agent in ((a, True), (p, False)):
        if sentence.count(name) != 1:
            return None
        s = sentence.index(name)
        found.append((s, name, is_agent))
    found.sort()
    passive = rec["kind"] == "passive"
    toks = tuple(ors.NameToken(name=n, char_start=s, char_end=s + len(n), prompt_char_start=off + s,
                               prompt_char_end=off + s + len(n), is_first=(k == 0),
                               is_subject=(is_ag != passive), is_agent=is_ag)
                 for k, (s, n, is_ag) in enumerate(found))
    cons = "wiki_passive" if passive else "wiki_active"
    fam = f"wiki:{rec.get('article_id', '')}:{idx}"
    return ors.Item(item_id=f"{fam}:{cons}:{names}", family_id=fam, event_id=fam, split="dev", test_set=-1,
                    construction=cons, is_control=False, is_heldout=False, agent=a, patient=p, lemma=rec["lemma"],
                    verb=rec["past"], role_asked="agent", question_voice="active", sentence=sentence,
                    question=question, prompt=prompt, answer=a, distractor=p, answer_is_first=(toks[0].name == a),
                    names=toks, spans=(("first_name", toks[0].prompt_char_start, toks[0].prompt_char_end),
                                       ("second_name", toks[1].prompt_char_start, toks[1].prompt_char_end),
                                       ("answer_cue", len(prompt) - 1, len(prompt))))


def generate_pairs(records, base_q: str, source_q: str, instruction: str = "", names: str = "real",
                   seed: int = 2026):
    """(base, source) per record: failing vs working question on the same sentence."""
    rng = random.Random(f"{seed}:wiki")
    out = []
    for idx, rec in enumerate(records):
        state = rng.getstate()
        b = build_item(rec, idx, base_q, names, rng)
        rng.setstate(state)  # the same neutral names for base and source
        s = build_item(rec, idx, source_q, names, rng)
        if b is None or s is None:
            continue
        if instruction:
            b, s = (dataclasses.replace(x, prompt=x.prompt[: -len("\nAnswer:")] + f"\n{instruction}\nAnswer:")
                    for x in (b, s))
        out.append((b, s))
    return out


def load_records(path):
    import json
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]
