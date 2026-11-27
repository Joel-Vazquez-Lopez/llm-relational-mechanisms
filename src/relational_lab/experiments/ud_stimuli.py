"""Natural-sentence validation set (UD): real English sentences from UD_English-EWT and -GUM with a
transitive person-type verb, whose subject and object noun phrases are replaced by two names.

The rest of each sentence (context, clauses, punctuation, length) is natural; only the two
argument noun phrases become names, so the bare-name forced choice and the OTR-09 repair can be
measured exactly as on the templates. Records (tokens, argument spans, verb forms) are in
``ud_en_natural.json`` (next to this module) (built by ``scripts/ud_extract_natural.py``); no UD or
inflection library is needed at run time.

Constructions: "ud_active" (nsubj + obj) and "ud_passive" (nsubj:pass + obl:agent, "by" kept).
Family = the UD sentence, so folds never split one sentence.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from relational_lab.experiments import order_role_stimuli as ors

UD_VERSION = "0.1.0"
DATA = Path(__file__).with_name("ud_en_natural.json")
CONSTRUCTIONS = ("ud_active", "ud_passive")


def load_records(path: Path = DATA):
    return json.loads(path.read_text(encoding="utf8"))["records"]


def _detok(tokens, space_after) -> str:
    return "".join(t + (" " if sp else "") for t, sp in zip(tokens, space_after)).strip()


def substitute(rec: dict, a: str, p: str) -> tuple[str, dict]:
    """Sentence with the agent and patient phrases replaced by ``a`` / ``p`` (1-based spans)."""
    toks, sps = list(rec["tokens"]), list(rec["space_after"])
    spans = sorted([("a", rec["agent_span"]), ("p", rec["patient_span"])], key=lambda x: -x[1][0])
    for who, (s, e) in spans:  # right to left keeps indices valid
        name = a if who == "a" else p
        toks[s - 1:e] = [name]
        sps[s - 1:e] = [sps[e - 1]]
    sentence = _detok(toks, sps)
    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return sentence, {"a": a, "p": p}


def fill(template: str, rec: dict) -> str:
    return template.format(past=rec["past"], pp=rec["pp"], ing=rec["ing"], lemma=rec["lemma"])


def build_item(rec: dict, a: str, p: str, question_template: str, idx: int):
    sentence, _ = substitute(rec, a, p)
    question = fill(question_template, rec)
    prompt = ors.PROMPT_TEMPLATE.format(sentence=sentence, question=question)
    off = prompt.index(sentence)
    toks = []
    for name, is_agent in ((a, True), (p, False)):
        hits = [i for i in range(len(sentence)) if sentence.startswith(name, i)
                and (i == 0 or not sentence[i - 1].isalnum())
                and (i + len(name) == len(sentence) or not sentence[i + len(name)].isalnum())]
        if len(hits) != 1:
            return None
        s = hits[0]
        toks.append([name, s, is_agent])
    toks.sort(key=lambda t: t[1])
    passive = rec["kind"] == "passive"
    names = []
    for k, (name, s, is_agent) in enumerate(toks):
        names.append(ors.NameToken(name=name, char_start=s, char_end=s + len(name), prompt_char_start=off + s,
                                   prompt_char_end=off + s + len(name), is_first=(k == 0),
                                   is_subject=(is_agent != passive), is_agent=is_agent))
    cons = "ud_passive" if passive else "ud_active"
    sid = rec["sent_id"]
    event = f"ud:{sid}:{a}>{p}"
    return ors.Item(item_id=f"{event}:{cons}:agent:{idx}", family_id=f"ud:{sid}", event_id=event, split="dev",
                    test_set=-1, construction=cons, is_control=False, is_heldout=False, agent=a, patient=p,
                    lemma=rec["lemma"], verb=rec["past"], role_asked="agent", question_voice="active",
                    sentence=sentence, question=question, prompt=prompt, answer=a, distractor=p,
                    answer_is_first=(names[0].name == a), names=(names[0], names[1]),
                    spans=(("first_name", names[0].prompt_char_start, names[0].prompt_char_end),
                           ("second_name", names[1].prompt_char_start, names[1].prompt_char_end),
                           ("answer_cue", len(prompt) - 1, len(prompt))))


def generate_pairs(base_q: str, source_q: str, instruction: str = "", pairs_per_sentence: int = 2, seed: int = 2026,
                   names=ors.NAMES):
    """(base item, source item) per (sentence, name pair): same substituted sentence, failing vs
    working question. Names never already present in the sentence."""
    rng = random.Random(f"{seed}:ud")
    out = []
    for rec in load_records():
        text = " ".join(rec["tokens"])
        pool = [n for n in names if n not in text]
        for k in range(pairs_per_sentence):
            a, p = rng.sample(pool, 2)
            b = build_item(rec, a, p, base_q, k)
            s = build_item(rec, a, p, source_q, k)
            if b is None or s is None:
                continue
            if instruction:
                b = _instr(b, instruction)
                s = _instr(s, instruction)
            out.append((b, s))
    return out


def _instr(it, instruction):
    import dataclasses
    assert it.prompt.endswith("\nAnswer:")
    return dataclasses.replace(it, prompt=it.prompt[: -len("\nAnswer:")] + f"\n{instruction}\nAnswer:")
