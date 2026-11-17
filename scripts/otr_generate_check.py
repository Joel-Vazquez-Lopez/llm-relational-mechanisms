"""Generation check: what does a chat model actually answer?

Greedy-decodes a short answer for a sample of OTR-01 prompts (dev split) with the
same chat settings as the scoring runs, and reports (a) the share of answers
that start with one of the two candidate names, (b) generation-based accuracy,
(c) per construction x role accuracy, and (d) example outputs. Use it to check
that forced-choice scoring fits a model (top-1 token often not a name).

Answers are classified by ``extract_answer`` (direct / inferred / ambiguous / none);
quoted text and copies of the sentence or question are ignored, so "In the sentence
"Lisa chased Julia," someone chased Julia." counts as Julia. ``--instruction "Reply with
one name only."`` adds a line before "Answer:" to ask for a bare name (a prompt change:
report it as a separate condition).

``--question-template`` replaces the question with any wording ({past}, {lemma}, {ing} are filled
from the item); ``--question-role`` says which role it asks for (the gold answer). The
sample is then balanced over the four main constructions only, e.g. the stranded agent
question and its patient permutation (same five words):

    python scripts/otr_generate_check.py --model google/gemma-4-31B-it --chat-template \
        --instruction "Reply with one name only." \
        --question-template "Who was someone {past} by?" --question-role agent

    python scripts/otr_generate_check.py --model unsloth/Mistral-Small-3.2-24B-Instruct-2506 --chat-template

``--stimuli sp`` uses the stranded-preposition frames instead (to/for/from; see
``relational_lab.experiments.stranded_p_stimuli``); cells are then
<frame>_<form>/<role>/<question form>. ``--n 5412`` covers every dev item.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402
from relational_lab.experiments.question_templates import fill as fill_template


def first_name(text: str, names) -> str | None:
    """The first of ``names`` that appears as a word in ``text`` (None if neither)."""
    hits = [(m.start(), n) for n in names for m in [re.search(rf"\b{re.escape(n)}\b", text)] if m]
    return min(hits)[1] if hits else None


_QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d|\'[^\']{8,}\'')


def event_statement(text: str, agent: str, patient: str):
    """If the first sentence of ``text`` is a bare event statement about the two names,
    return (doer, undergoer) as the statement asserts them, else None.

    Recognised (whole first sentence, quoted spans removed first, any verb):
      "P was/were/got V-ed by A."   -> (A, P)
      "A V-ed P."                    -> (A, P)
      "It was A who/that V-ed P."    -> (A, P)
      "It was P that/whom A V-ed."   -> (A, P)
      "It was by A that P was V-ed." -> (A, P)
    The same patterns with the names swapped return the reversed reading. Statements
    embedded in longer explanations are not matched (they fall through to "inferred").
    """
    s = _QUOTED.sub(" ", text).strip().split("\n")[0].strip()
    s = re.split(r"(?<=[.!?])\s", s)[0].strip().rstrip(".!").strip().strip("*").strip()
    for x, y in ((agent, patient), (patient, agent)):  # x = doer, y = undergoer as asserted
        X, Y = re.escape(x), re.escape(y)
        D = r"(?:the |a |an )?\w+"  # a direct object such as "the book" or "dinner"
        AUX = r"(?:was|were|got|is|has been|had been|have been|was being|got)"
        IND = (r"(?:(?:a person|somebody|someone|the person|the other person|"
               r"the (?:other )?(?:person|one) who (?:was|got)) )?")  # echoes of the question / relatives
        pats = (
            rf"{Y} {AUX} {IND}(?:\w+ )?\w+ by {X}",
            # three-place frames (stranded-preposition stimuli): X = doer, Y = to/for/from object
            rf"{X} \w+ {D} (?:to|for|from) {Y}",
            rf"{X} \w+ {Y} {D}",
            rf"{Y} was \w+ {D} by {X}",
            rf"(?:(?:The|A|An) )?\w+ (?:was|were) \w+ (?:to|for|from) {Y} by {X}",
            rf"(?:(?:The|A|An) )?\w+ (?:was|were) \w+ by {X} (?:to|for|from) {Y}",
            rf"It was {Y} (?:that|whom|who) {X} \w+ {D} (?:to|for|from)",
            rf"It was (?:to|for|from) {Y} that {X} \w+ {D}",
            rf"{X} \w+ {Y}",
            rf"{X} was the (?:other )?(?:person|one) who \w+ {Y}",
            rf"The (?:other )?(?:person|one), {Y}, was \w+ by {X}",
            rf"The (?:other )?(?:person|one) \w+ by {X} was {Y}",
            rf"It was {X} (?:who|that) \w+ {Y}",
            rf"It was {Y} (?:that|whom|who) {X} \w+",
            rf"It was by {X} that {Y} was \w+",
        )
        if any(re.fullmatch(p, s) for p in pats):
            return x, y
    return None


def extract_answer(text: str, names, sentence: str = "", question: str = "",
                   agent: str | None = None, patient: str | None = None, role: str | None = None):
    """Classify a generated answer. Returns (kind, name).

    kind = "direct"    the first line is just one candidate name (punctuation allowed);
           "frame"     the question with its wh-word replaced by one name ("Daniel had someone
                       been thanked by."); the answer is that name;
           "statement" (only when agent, patient and role are given) the first sentence is an
                       event statement such as "Frank was watched by Daniel."; the answer is
                       the name the statement puts in the asked role. A verbatim copy of the
                       sentence also counts (``is_copy`` in the rows says so);
           "inferred"  after removing quoted spans and copies of the sentence and question,
                       exactly one candidate name is left;
           "ambiguous" both names are left;
           "none"      no candidate name.
    """
    first = text.strip().split("\n")[0].strip()
    bare = re.sub(r"[^A-Za-z]", "", first)
    if bare in names:
        return "direct", bare
    if question:
        # question-frame answer: the wh-word replaced by a name, e.g.
        # "Daniel had someone been thanked by." for "Who had someone been thanked by?"
        body = re.escape(re.sub(r"^\s*(?:Who|whom)\b", "", question).rstrip("?").strip())
        if body:
            m = re.fullmatch(rf"\s*(\w+) {body}[.!]?\s*", _QUOTED.sub(" ", first))
            if m and m.group(1) in names:
                return "frame", m.group(1)
    if agent and patient and role:
        st = event_statement(text, agent, patient)
        if st is not None:
            return "statement", (st[0] if role == "agent" else st[1])
    rest = text
    for chunk in (sentence, sentence.rstrip("."), question):
        if chunk:
            rest = rest.replace(chunk, " ")
    rest = _QUOTED.sub(" ", rest)
    left = [n for n in names if re.search(rf"\b{re.escape(n)}\b", rest)]
    if len(left) == 1:
        return "inferred", left[0]
    return ("ambiguous" if left else "none"), None


def prepared_prompts(stimuli="otr", seed=2026, n=256, question_template="", question_role=None, instruction=""):
    """The balanced dev sample and final prompts used by every generation run (local or API).
    Returns [(item, prompt)]; with a question template the item's question is rewritten."""
    if stimuli == "sp":
        from relational_lab.experiments import stranded_p_stimuli as sp
        items = [i for i in sp.generate_items(seed=seed) if i.split == "dev"]
    else:
        items = [i for i in ors.generate_items(seed=seed)
                 if i.split == "dev" and i.construction in ors.MAIN_CONSTRUCTIONS]
    if question_template:
        # one item per event and role: keep the active-voice question slot, then rewrite it
        items = [i for i in items if i.role_asked == question_role and i.question_voice == "active"]
    random.Random(seed).shuffle(items)
    cells = defaultdict(list)
    for it in items:
        cells[(it.construction, it.role_asked, it.question_voice)].append(it)
    per = max(1, n // len(cells))
    out = []
    for c in sorted(cells):
        for it in cells[c][:per]:
            if question_template:
                question = fill_template(question_template, it.verb, it.lemma)
                it = dataclasses.replace(it, question=question, question_voice="custom",
                                         prompt=it.prompt.replace(f"\nQuestion: {it.question}\n",
                                                                  f"\nQuestion: {question}\n"))
                assert it.prompt.count(question) == 1, it.prompt
            prompt = it.prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:") if instruction else it.prompt
            out.append((it, prompt))
    return out


def make_row(it, prompt: str, text: str) -> dict:
    said = first_name(text, [it.answer, it.distractor])
    kind, name = extract_answer(text, [it.answer, it.distractor], it.sentence, it.question,
                                agent=it.agent, patient=it.patient, role=it.role_asked)
    return {"item_id": it.item_id, "construction": it.construction, "role": it.role_asked,
            "voice": it.question_voice, "question": it.question, "prompt": prompt,
            "answer": it.answer, "output": text,
            "named": said is not None, "correct_first_name": said == it.answer,
            "kind": kind, "extracted": name,
            "is_copy": text.strip().rstrip(".") == it.sentence.rstrip("."), "correct": name == it.answer}


def make_report(model: str, rows: list, seconds: float, **meta) -> dict:
    n = len(rows)
    summ = defaultdict(list)
    for r in rows:
        summ[f"{r['construction']}/{r['role']}/{r['voice']}"].append(r["correct"])
    return {
        "model": model, **meta,
        "n": n, "named_rate": sum(r["named"] for r in rows) / n,
        "kinds": {k: sum(r["kind"] == k for r in rows) / n for k in ("direct", "frame", "statement", "inferred", "ambiguous", "none")},
        "generation_accuracy": sum(r["correct"] for r in rows) / n,
        "accuracy_direct_only": (sum(r["correct"] for r in rows if r["kind"] == "direct")
                                 / max(1, sum(r["kind"] == "direct" for r in rows))),
        "accuracy_first_name_rule": sum(r["correct_first_name"] for r in rows) / n,
        "cells": {k: sum(v) / len(v) for k, v in sorted(summ.items())},
        "seconds": round(seconds, 1), "rows": rows,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=256, help="number of prompts (balanced sample)")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--instruction", default="",
                   help='text inserted on its own line before "Answer:", e.g. "Reply with one name only."')
    p.add_argument("--stimuli", choices=("otr", "sp"), default="otr",
                   help="otr = order-to-role constructions; sp = stranded-preposition frames (to/for/from)")
    p.add_argument("--question-template", default="",
                   help='custom question, e.g. "Who was someone {past} by?" ({past}, {lemma}, {ing} filled per item)')
    p.add_argument("--question-role", choices=("agent", "patient"), default=None,
                   help="role the custom question asks for (required with --question-template)")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    if bool(args.question_template) != bool(args.question_role):
        p.error("--question-template and --question-role go together")
    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.instrumentation.hf_scoring import _prompt_ids, load_lm, load_tokenizer

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype)).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(model.parameters()).device
    if args.stimuli == "sp" and args.question_template:
        p.error("--question-template applies to the OTR stimuli only")
    t0 = time.time()
    rows = []
    for it, prompt in prepared_prompts(args.stimuli, args.seed, args.n, args.question_template,
                                       args.question_role, args.instruction):
        ids = _prompt_ids(tok, prompt, args.chat_template)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        rows.append(make_row(it, prompt, text))
    report = make_report(args.model, rows, time.time() - t0, chat_template=args.chat_template,
                         chat_template_kwargs=chat_kwargs, stimuli=args.stimuli, instruction=args.instruction,
                         question_template=args.question_template or None, question_role=args.question_role,
                         max_new_tokens=args.max_new_tokens)
    n = len(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_sp" if args.stimuli == "sp" else ""
    if args.question_template:
        tag = "_q-" + re.sub(r"[^a-z]+", "-", args.question_template.lower()).strip("-")
    out = args.out_dir / f"otrgen_{args.model.replace('/', '__')}_dev{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out}")
    print(f"answer kinds: {report['kinds']}")
    print(f"generation accuracy (ambiguous/none = wrong): {report['generation_accuracy']:.2f}   "
          f"direct answers only: {report['accuracy_direct_only']:.2f}")
    for k, v in report["cells"].items():
        print(f"  {k:32s} {v:.2f}")
    print("examples:")
    for r in rows[:: max(1, n // 12)][:12]:
        print(f"  [{r['construction']}/{r['role']}/{r['voice']}] answer={r['answer']!r} -> {r['output']!r}")


if __name__ == "__main__":
    main()
