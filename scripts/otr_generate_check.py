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

``--question-template`` replaces the question with any wording ({past}, {lemma} are filled
from the item); ``--question-role`` says which role it asks for (the gold answer). The
sample is then balanced over the four main constructions only, e.g. the stranded agent
question and its patient permutation (same five words):

    python scripts/otr_generate_check.py --model google/gemma-4-31B-it --chat-template \
        --instruction "Reply with one name only." \
        --question-template "Who was someone {past} by?" --question-role agent

    python scripts/otr_generate_check.py --model unsloth/Mistral-Small-3.2-24B-Instruct-2506 --chat-template
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

from relational_lab.experiments import order_role_stimuli as ors


def first_name(text: str, names) -> str | None:
    """The first of ``names`` that appears as a word in ``text`` (None if neither)."""
    hits = [(m.start(), n) for n in names for m in [re.search(rf"\b{re.escape(n)}\b", text)] if m]
    return min(hits)[1] if hits else None


_QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d|\'[^\']{8,}\'')


def extract_answer(text: str, names, sentence: str = "", question: str = ""):
    """Classify a generated answer. Returns (kind, name).

    kind = "direct"    the first line is just one candidate name (punctuation allowed);
           "inferred"  after removing quoted spans and copies of the sentence and question,
                       exactly one candidate name is left;
           "ambiguous" both names are left (e.g. "Daniel was warned by Kate.");
           "none"      no candidate name.
    """
    first = text.strip().split("\n")[0].strip()
    bare = re.sub(r"[^A-Za-z]", "", first)
    if bare in names:
        return "direct", bare
    rest = text
    for chunk in (sentence, sentence.rstrip("."), question):
        if chunk:
            rest = rest.replace(chunk, " ")
    rest = _QUOTED.sub(" ", rest)
    left = [n for n in names if re.search(rf"\b{re.escape(n)}\b", rest)]
    if len(left) == 1:
        return "inferred", left[0]
    return ("ambiguous" if left else "none"), None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=256, help="number of prompts (balanced sample)")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--instruction", default="",
                   help='text inserted on its own line before "Answer:", e.g. "Reply with one name only."')
    p.add_argument("--question-template", default="",
                   help='custom question, e.g. "Who was someone {past} by?" ({past}, {lemma} filled per item)')
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
    items = [i for i in ors.generate_items(seed=args.seed)
             if i.split == "dev" and i.construction in ors.MAIN_CONSTRUCTIONS]
    if args.question_template:
        # one item per event and role: keep the active-voice question slot, then rewrite it
        items = [i for i in items if i.role_asked == args.question_role and i.question_voice == "active"]
    random.Random(args.seed).shuffle(items)
    cells = defaultdict(list)
    for it in items:
        cells[(it.construction, it.role_asked, it.question_voice)].append(it)
    per = max(1, args.n // len(cells))
    sample = [it for c in sorted(cells) for it in cells[c][:per]]

    t0 = time.time()
    rows = []
    for it in sample:
        question = it.question
        if args.question_template:
            question = args.question_template.format(past=it.verb, lemma=it.lemma)
            it = dataclasses.replace(it, question=question, question_voice="custom",
                                     prompt=it.prompt.replace(f"\nQuestion: {it.question}\n",
                                                              f"\nQuestion: {question}\n"))
            assert it.prompt.count(question) == 1, it.prompt
        prompt = it.prompt.replace("\nAnswer:", f"\n{args.instruction}\nAnswer:") if args.instruction else it.prompt
        ids = _prompt_ids(tok, prompt, args.chat_template)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        said = first_name(text, [it.answer, it.distractor])
        kind, name = extract_answer(text, [it.answer, it.distractor], it.sentence, it.question)
        rows.append({"item_id": it.item_id, "construction": it.construction, "role": it.role_asked,
                     "voice": it.question_voice, "question": it.question, "prompt": prompt,
                     "answer": it.answer, "output": text,
                     "named": said is not None, "correct_first_name": said == it.answer,
                     "kind": kind, "extracted": name, "correct": name == it.answer})
    n = len(rows)
    summ = defaultdict(list)
    for r in rows:
        summ[f"{r['construction']}/{r['role']}/{r['voice']}"].append(r["correct"])
    report = {
        "model": args.model, "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "instruction": args.instruction, "question_template": args.question_template or None,
        "question_role": args.question_role, "max_new_tokens": args.max_new_tokens,
        "n": n, "named_rate": sum(r["named"] for r in rows) / n,
        "kinds": {k: sum(r["kind"] == k for r in rows) / n for k in ("direct", "inferred", "ambiguous", "none")},
        "generation_accuracy": sum(r["correct"] for r in rows) / n,
        "accuracy_direct_only": (sum(r["correct"] for r in rows if r["kind"] == "direct")
                                 / max(1, sum(r["kind"] == "direct" for r in rows))),
        "accuracy_first_name_rule": sum(r["correct_first_name"] for r in rows) / n,
        "cells": {k: sum(v) / len(v) for k, v in sorted(summ.items())},
        "seconds": round(time.time() - t0, 1), "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = ""
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
