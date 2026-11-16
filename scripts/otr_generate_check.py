"""Generation check: what does a chat model actually answer?

Greedy-decodes a short answer for a sample of OTR-01 prompts (dev split) with the
same chat settings as the scoring runs, and reports (a) the share of answers
that start with one of the two candidate names, (b) generation-based accuracy,
(c) per construction x role accuracy, and (d) example outputs. Use it to check
that forced-choice scoring fits a model (top-1 token often not a name).

    python scripts/otr_generate_check.py --model unsloth/Mistral-Small-3.2-24B-Instruct-2506 --chat-template
"""

from __future__ import annotations

import argparse
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=256, help="number of prompts (balanced sample)")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
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
    random.Random(args.seed).shuffle(items)
    cells = defaultdict(list)
    for it in items:
        cells[(it.construction, it.role_asked, it.question_voice)].append(it)
    per = max(1, args.n // len(cells))
    sample = [it for c in sorted(cells) for it in cells[c][:per]]

    t0 = time.time()
    rows = []
    for it in sample:
        ids = _prompt_ids(tok, it.prompt, args.chat_template)
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        said = first_name(text, [it.answer, it.distractor])
        rows.append({"item_id": it.item_id, "construction": it.construction, "role": it.role_asked,
                     "voice": it.question_voice, "answer": it.answer, "output": text,
                     "named": said is not None, "correct": said == it.answer})
    n = len(rows)
    summ = defaultdict(list)
    for r in rows:
        summ[f"{r['construction']}/{r['role']}/{r['voice']}"].append(r["correct"])
    report = {
        "model": args.model, "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "n": n, "named_rate": sum(r["named"] for r in rows) / n,
        "generation_accuracy": sum(r["correct"] for r in rows) / n,
        "cells": {k: sum(v) / len(v) for k, v in sorted(summ.items())},
        "seconds": round(time.time() - t0, 1), "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otrgen_{args.model.replace('/', '__')}_dev_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out}")
    print(f"named a candidate: {report['named_rate']:.2f}   generation accuracy: {report['generation_accuracy']:.2f}")
    for k, v in report["cells"].items():
        print(f"  {k:32s} {v:.2f}")
    print("examples:")
    for r in rows[:: max(1, n // 12)][:12]:
        print(f"  [{r['construction']}/{r['role']}/{r['voice']}] answer={r['answer']!r} -> {r['output']!r}")


if __name__ == "__main__":
    main()
