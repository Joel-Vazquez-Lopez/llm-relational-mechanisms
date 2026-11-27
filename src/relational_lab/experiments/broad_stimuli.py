"""Broad English stimuli (BX 0.1.0): new names, new verbs, many more constructions.

Purpose: test whether the role code and the passive-agent-question failure generalise
beyond the OTR materials (24 names, 16 verbs, 7 templates). Everything here is NEW
vocabulary: 48 names disjoint from ``order_role_stimuli.NAMES`` and verbs that do not
occur in OTR, with their own family splits (dev = 15% of name pairs, 4 test sets).

Three blocks; in each, the "agent slot" is the first role named below and Item.agent /
Item.patient / NameToken.is_agent follow it:

  transitive    agent / patient          45 verbs (none in OTR), 18 constructions
  ditransitive  agent / recipient        10 verbs, inanimate theme ("a letter"), 7 constructions
  psych         stimulus / experiencer   12 frighten-class + 10 fear-class verbs, 4 constructions each
                (frighten-class: subject = stimulus; fear-class: subject = experiencer, so the
                grammatical subject and the semantic role come apart even in plain actives)

In every block the four MAIN constructions decorrelate order, function and role exactly
(each of the four (first, subject) patterns occurs once). Extra constructions are variants
(tense, get-passive, reduced relatives, adjuncts, coordination, long-distance clefts).

Questions never name a participant. Per block: role x voice, e.g. transitive
  agent/active "Who V-ed someone?", patient/active "Who did someone V?",
  agent/passive "By whom was someone V-ed?", patient/passive "Who was V-ed by someone?".
Custom templates use {past} {pp} {ing} {lemma} {theme}; the failing wording is
"Who was someone {pp} by?" (ditransitive: "Who was someone given {theme} by?").
"""

from __future__ import annotations

import random
import re
from itertools import combinations

from relational_lab.experiments import order_role_stimuli as ors

BX_GENERATOR_VERSION = "0.1.0"

NAMES: tuple[str, ...] = (
    "Beth", "Clara", "Diana", "Ella", "Fiona", "Hannah", "Irene", "Jane", "Karen", "Lucy", "Megan", "Nora",
    "Olivia", "Paula", "Ruth", "Sophie", "Tina", "Vera", "Wendy", "Zoe", "Amy", "Joan", "Nina", "Carol",
    "Adam", "Brian", "Carl", "Derek", "Eric", "Felix", "Gary", "Henry", "Ian", "Jack", "Kevin", "Luke",
    "Martin", "Neil", "Oscar", "Ryan", "Simon", "Victor", "Walter", "Alan", "Bruce", "Colin", "Dennis", "Hugo",
)
assert not set(NAMES) & set(ors.NAMES) and len(set(NAMES)) == 48

# (lemma, past, past participle, gerund)
TRANSITIVE: tuple[tuple[str, str, str, str], ...] = tuple(
    (l, p, p, g) for l, p, g in (
        ("accuse", "accused", "accusing"), ("attack", "attacked", "attacking"), ("avoid", "avoided", "avoiding"),
        ("comfort", "comforted", "comforting"), ("contact", "contacted", "contacting"), ("criticize", "criticized", "criticizing"),
        ("defend", "defended", "defending"), ("distract", "distracted", "distracting"), ("embrace", "embraced", "embracing"),
        ("encourage", "encouraged", "encouraging"), ("examine", "examined", "examining"), ("guide", "guided", "guiding"),
        ("hire", "hired", "hiring"), ("hug", "hugged", "hugging"), ("ignore", "ignored", "ignoring"),
        ("interview", "interviewed", "interviewing"), ("kick", "kicked", "kicking"), ("kiss", "kissed", "kissing"),
        ("mock", "mocked", "mocking"), ("nominate", "nominated", "nominating"), ("notice", "noticed", "noticing"),
        ("obey", "obeyed", "obeying"), ("punish", "punished", "punishing"), ("recognize", "recognized", "recognizing"),
        ("recommend", "recommended", "recommending"), ("remind", "reminded", "reminding"), ("reward", "rewarded", "rewarding"),
        ("rob", "robbed", "robbing"), ("save", "saved", "saving"), ("scold", "scolded", "scolding"),
        ("serve", "served", "serving"), ("slap", "slapped", "slapping"), ("support", "supported", "supporting"),
        ("tease", "teased", "teasing"), ("thank", "thanked", "thanking"), ("trick", "tricked", "tricking"),
        ("welcome", "welcomed", "welcoming"), ("arrest", "arrested", "arresting"), ("betray", "betrayed", "betraying"),
        ("congratulate", "congratulated", "congratulating"), ("forgive", "forgave", "forgiving"),
        ("photograph", "photographed", "photographing"), ("question", "questioned", "questioning"),
        ("rescue", "rescued", "rescuing"), ("tackle", "tackled", "tackling"), ("train", "trained", "training"),
        ("visit", "visited", "visiting"), ("wake", "woke", "waking"),
    )
)
# irregular participles
_PP_FIX = {"forgive": "forgiven", "wake": "woken"}
TRANSITIVE = tuple((l, p, _PP_FIX.get(l, pp), g) for l, p, pp, g in TRANSITIVE
                   if l not in {v[0] for v in ors.VERBS})  # keep only verbs absent from OTR
DITRANSITIVE: tuple[tuple[str, str, str, str], ...] = (
    ("give", "gave", "given", "giving"), ("send", "sent", "sent", "sending"), ("show", "showed", "shown", "showing"),
    ("hand", "handed", "handed", "handing"), ("offer", "offered", "offered", "offering"), ("lend", "lent", "lent", "lending"),
    ("pass", "passed", "passed", "passing"), ("throw", "threw", "thrown", "throwing"), ("sell", "sold", "sold", "selling"),
    ("bring", "brought", "brought", "bringing"),
)
THEMES: tuple[str, ...] = ("a letter", "a book", "a ticket", "a present", "a map", "a photo", "a key", "a note")
PSYCH_SE: tuple[tuple[str, str, str, str], ...] = tuple(  # subject = stimulus
    (l, p, p, g) for l, p, g in (
        ("frighten", "frightened", "frightening"), ("surprise", "surprised", "surprising"), ("amuse", "amused", "amusing"),
        ("annoy", "annoyed", "annoying"), ("impress", "impressed", "impressing"), ("bore", "bored", "boring"),
        ("worry", "worried", "worrying"), ("scare", "scared", "scaring"), ("delight", "delighted", "delighting"),
        ("disgust", "disgusted", "disgusting"), ("confuse", "confused", "confusing"), ("startle", "startled", "startling"),
    )
)
PSYCH_ES: tuple[tuple[str, str, str, str], ...] = tuple(  # subject = experiencer
    (l, p, p, g) for l, p, g in (
        ("fear", "feared", "fearing"), ("love", "loved", "loving"), ("hate", "hated", "hating"), ("envy", "envied", "envying"),
        ("trust", "trusted", "trusting"), ("respect", "respected", "respecting"), ("pity", "pitied", "pitying"),
        ("resent", "resented", "resenting"), ("adore", "adored", "adoring"), ("miss", "missed", "missing"),
    )
)
FORMS = {v[0]: v for v in TRANSITIVE + DITRANSITIVE + PSYCH_SE + PSYCH_ES}

BLOCKS = ("transitive", "ditransitive", "psych")
ROLES = {"transitive": ("agent", "patient"), "ditransitive": ("agent", "recipient"),
         "psych": ("stimulus", "experiencer")}

# construction -> (template, first-mentioned role, subject role) with A = agent slot, B = other slot.
# Subject role None = neither name is the grammatical subject.
T_CONS = {
    # main (decorrelated)
    "active": ("{A} {past} {B}.", "A", "A"),
    "passive": ("{B} was {pp} by {A}.", "B", "B"),
    "object_cleft": ("It was {B} that {A} {past}.", "B", "A"),
    "by_cleft": ("It was by {A} that {B} was {pp}.", "A", "B"),
    # variants
    "subject_cleft": ("It was {A} who {past} {B}.", "A", "A"),
    "relative_active": ("The one who {past} {B} was {A}.", "B", "A"),
    "relative_passive": ("The one who was {pp} by {A} was {B}.", "A", "B"),
    "get_passive": ("{B} got {pp} by {A}.", "B", "B"),
    "reduced_relative": ("The one {pp} by {A} was {B}.", "A", "B"),
    "perfect_active": ("{A} had {pp} {B}.", "A", "A"),
    "perfect_passive": ("{B} had been {pp} by {A}.", "B", "B"),
    "progressive_passive": ("{B} was being {pp} by {A}.", "B", "B"),
    "adjunct_active": ("Earlier that day, {A} {past} {B} near the station.", "A", "A"),
    "adjunct_passive": ("Earlier that day, {B} was {pp} by {A} near the station.", "B", "B"),
    "coord_active": ("{A} {past} {B} and then went home.", "A", "A"),
    "coord_passive": ("{B} was {pp} by {A} and then went home.", "B", "B"),
    "long_object_cleft": ("It was {B} that everyone said {A} {past}.", "B", "A"),
    "long_by_cleft": ("It was by {A} that everyone said {B} was {pp}.", "A", "B"),
}
T_MAIN = ("active", "passive", "object_cleft", "by_cleft")
D_CONS = {
    "do_active": ("{A} {past} {B} {theme}.", "A", "A"),
    "recipient_passive": ("{B} was {pp} {theme} by {A}.", "B", "B"),
    "to_cleft": ("It was to {B} that {A} {past} {theme}.", "B", "A"),
    "by_cleft": ("It was by {A} that {B} was {pp} {theme}.", "A", "B"),
    "pd_active": ("{A} {past} {theme} to {B}.", "A", "A"),
    "theme_passive": ("{Theme} was {pp} to {B} by {A}.", "B", None),
    "stranded_cleft": ("It was {B} that {A} {past} {theme} to.", "B", "A"),
}
D_MAIN = ("do_active", "recipient_passive", "to_cleft", "by_cleft")
# psych: A = stimulus, B = experiencer
P_CONS = {
    "se_active": ("{A} {past} {B}.", "A", "A", "se"),          # Leon frightened Mira
    "se_passive": ("{B} was {pp} by {A}.", "B", "B", "se"),    # Mira was frightened by Leon
    "se_object_cleft": ("It was {B} that {A} {past}.", "B", "A", "se"),
    "es_active": ("{B} {past} {A}.", "B", "B", "es"),          # Mira feared Leon
    "es_passive": ("{A} was {pp} by {B}.", "A", "A", "es"),    # Leon was feared by Mira
    "es_object_cleft": ("It was {A} that {B} {past}.", "A", "B", "es"),
}
P_MAIN = ("se_active", "se_object_cleft", "es_active", "es_object_cleft")  # (A,A) (B,A) (B,B) (A,B)

PROMPT = "Sentence: {sentence}\nQuestion: {question}\nAnswer:"


def constructions(block: str) -> dict:
    return {"transitive": T_CONS, "ditransitive": D_CONS, "psych": P_CONS}[block]


def main_constructions(block: str) -> tuple[str, ...]:
    return {"transitive": T_MAIN, "ditransitive": D_MAIN, "psych": P_MAIN}[block]


def verbs_for(block: str, construction: str):
    if block == "transitive":
        return TRANSITIVE
    if block == "ditransitive":
        return DITRANSITIVE
    return PSYCH_SE if construction.startswith("se_") else PSYCH_ES


def render_question(block: str, role: str, voice: str, lemma: str, theme: str = "", psych_class: str = "") -> str:
    _l, past, pp, _g = FORMS[lemma]
    if block == "transitive":
        return {("agent", "active"): f"Who {past} someone?", ("patient", "active"): f"Who did someone {lemma}?",
                ("agent", "passive"): f"By whom was someone {pp}?",
                ("patient", "passive"): f"Who was {pp} by someone?"}[(role, voice)]
    if block == "ditransitive":
        return {("agent", "active"): f"Who {past} someone {theme}?",
                ("recipient", "active"): f"Who did someone {lemma} {theme} to?",
                ("agent", "passive"): f"By whom was someone {pp} {theme}?",
                ("recipient", "passive"): f"Who was {pp} {theme} by someone?"}[(role, voice)]
    # psych: which grammatical role the asked thematic role has depends on the verb class
    subj_role = "stimulus" if psych_class == "se" else "experiencer"
    if voice == "active":
        return f"Who {past} someone?" if role == subj_role else f"Who did someone {lemma}?"
    return f"By whom was someone {pp}?" if role == subj_role else f"Who was {pp} by someone?"


def fill_template(template: str, lemma: str, theme: str = "") -> str:
    _l, past, pp, ing = FORMS[lemma]
    return template.format(past=past, pp=pp, ing=ing, lemma=lemma, theme=theme)


def _name_span(sentence: str, name: str) -> tuple[int, int]:
    m = list(re.finditer(rf"\b{re.escape(name)}\b", sentence))
    if len(m) != 1:
        raise AssertionError(f"{name!r} must occur once in {sentence!r}")
    return m[0].start(), m[0].end()


def build_item(block, a, b, lemma, construction, role, voice, theme="", family_id=None, split="dev", test_set=-1):
    """a fills the agent slot (agent / agent / stimulus), b the other slot."""
    spec = constructions(block)[construction]
    tpl, first_slot, subj_slot = spec[0], spec[1], spec[2]
    psych_class = spec[3] if block == "psych" else ""
    _l, past, pp, _g = FORMS[lemma]
    sentence = tpl.format(A=a, B=b, past=past, pp=pp, theme=theme, Theme=theme[:1].upper() + theme[1:])
    question = render_question(block, role, voice, lemma, theme, psych_class)
    prompt = PROMPT.format(sentence=sentence, question=question)
    off = prompt.index(sentence)
    toks = []
    for name, slot in ((a, "A"), (b, "B")):
        s, e = _name_span(sentence, name)
        toks.append(ors.NameToken(name=name, char_start=s, char_end=e, prompt_char_start=off + s,
                                  prompt_char_end=off + e, is_first=(slot == first_slot),
                                  is_subject=(slot == subj_slot), is_agent=(slot == "A")))
    toks.sort(key=lambda t: t.char_start)
    r_a, r_b = ROLES[block]
    answer, distractor = (a, b) if role == r_a else (b, a)
    q0 = prompt.index(question)
    spans = (("first_name", toks[0].prompt_char_start, toks[0].prompt_char_end),
             ("second_name", toks[1].prompt_char_start, toks[1].prompt_char_end),
             ("period", off + len(sentence) - 1, off + len(sentence)),
             ("question_end", q0 + len(question) - 1, q0 + len(question)),
             ("answer_cue", len(prompt) - 1, len(prompt)))
    event = f"bx-{block}:{a}>{b}:{lemma}" + (f":{theme}" if theme else "")
    return ors.Item(item_id=f"{event}:{construction}:{role}:{voice}", family_id=family_id or "|".join(sorted((a, b))),
                    event_id=event, split=split, test_set=test_set, construction=construction,
                    is_control=construction not in main_constructions(block), is_heldout=False, agent=a, patient=b,
                    lemma=lemma, verb=past, role_asked=role, question_voice=voice, sentence=sentence,
                    question=question, prompt=prompt, answer=answer, distractor=distractor,
                    answer_is_first=(toks[0].name == answer), names=(toks[0], toks[1]), spans=spans)


def by_phrase_role(item) -> str:
    """The role a passive by-phrase expresses for this item's verb ("Who was someone V-ed by?" asks for it):
    the agent slot, except for fear-class psych verbs, whose by-phrase is the experiencer."""
    block = item.event_id.split(":")[0].removeprefix("bx-")
    if block == "psych" and item.construction.startswith("es_"):
        return "experiencer"
    return ROLES[block][0]


def theme_of(item) -> str:
    parts = item.event_id.split(":")
    return parts[3] if len(parts) > 3 else ""


def generate_items(block: str, split: str | None = "dev", verbs_per_orientation: int = 1, seed: int = 2026,
                   dev_fraction: float = 0.15, n_test_sets: int = 4, constructions_subset=None):
    """Every family (or only ``split``), both slot assignments, every construction x role x voice."""
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}")
    splits = ors.split_families(NAMES, seed, dev_fraction, n_test_sets)
    rng = random.Random(f"{seed}:bx:{block}")
    cons = constructions_subset or tuple(constructions(block))
    pools = {}
    items = []
    for x, y in combinations(sorted(NAMES), 2):
        family = f"{x}|{y}"
        sp, test_set = splits[family]
        if split is not None and sp != split:
            for _ in range(2 * verbs_per_orientation):  # keep the random stream identical across splits
                rng.random()
            continue
        for a, b in ((x, y), (y, x)):
            for _ in range(verbs_per_orientation):
                r = rng.random()
                theme = THEMES[int(r * 1e6) % len(THEMES)] if block == "ditransitive" else ""
                for c in cons:
                    pool = verbs_for(block, c)
                    pools.setdefault(id(pool), pool)
                    lemma = pool[int(r * len(pool))][0]
                    for role in ROLES[block]:
                        for voice in ("active", "passive"):
                            items.append(build_item(block, a, b, lemma, c, role, voice, theme,
                                                    family_id=family, split=sp, test_set=test_set))
    return items
