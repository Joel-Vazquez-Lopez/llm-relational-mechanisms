#!/usr/bin/env python
"""OTR-15: is the failure a prompt-format artefact? (behaviour, generation, one model load)

The failing question ("Who was someone {past} by?") and the working question ("Who was the {ing}
done by?") on the same balanced dev sentences (agent questions, four main constructions), under
several prompt formats:

  labels    the format of every other run: "Sentence: ...\\nQuestion: ...\\n<instr>\\nAnswer:"
  nolabel   "<sentence>\\n<question>\\n<instr>"
  natural   'In the sentence "<sentence>", <question> <instr>'
  system    labels + a system message ("You are a helpful assistant ...")
  fewshot   labels preceded by two solved examples with other names and verbs, using ordinary
            questions (never the failing or the working one)
  sampling  labels, temperature 0.7 / top-p 0.95 instead of greedy decoding

Each reply is scored with the generation-check extractor. Output: accuracy per format x question
x construction, and the failing - working drop per format.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from otr_generate_check import make_row, prepared_prompts  # noqa: E402

OTR15_VERSION = "1.0.0"
FORMATS = ("labels", "nolabel", "natural", "system", "fewshot", "sampling")
QUESTIONS = {"failing": "Who was someone {past} by?", "working": "Who was the {ing} done by?"}
SYSTEM = "You are a helpful assistant. Answer questions about sentences accurately."
# two solved examples with names and verbs outside the stimulus lists, ordinary questions only
FEWSHOT = ("Sentence: Olivia hugged Marcus.\nQuestion: Who hugged someone?\n{instr}\nAnswer: Olivia\n\n"
           "Sentence: Hannah was teased by Victor.\nQuestion: Who did someone tease?\n{instr}\nAnswer: Hannah\n\n")


def reformat(it, prompt: str, fmt: str, instruction: str) -> str:
    if fmt in ("labels", "system", "sampling"):
        return prompt
    if fmt == "nolabel":
        return f"{it.sentence}\n{it.question}\n{instruction}".strip()
    if fmt == "natural":
        q = it.question[0].lower() + it.question[1:]
        return f'In the sentence "{it.sentence}", {q} {instruction}'.strip()
    if fmt == "fewshot":
        return FEWSHOT.format(instr=instruction) + prompt
    raise ValueError(fmt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=128, help="prompts per question (balanced over constructions)")
    p.add_argument("--formats", default=",".join(FORMATS))
    p.add_argument("--instruction", default="Reply with one name only.")
    p.add_argument("--max-new-tokens", type=int, default=40)
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
    formats = [f.strip() for f in args.formats.split(",")]
    torch.manual_seed(args.seed)
    t0 = time.time()
    rows = []
    for qname, qtpl in QUESTIONS.items():
        pairs = prepared_prompts("otr", args.seed, args.n, qtpl, "agent", args.instruction)
        for fmt in formats:
            _chat.SYSTEM_PROMPT[0] = SYSTEM if fmt == "system" else ""
            for it, prompt in pairs:
                text_prompt = reformat(it, prompt, fmt, args.instruction)
                ids = _prompt_ids(tok, text_prompt, args.chat_template)
                x = torch.tensor([ids], device=dev)
                gen = dict(do_sample=True, temperature=0.7, top_p=0.95) if fmt == "sampling" else dict(do_sample=False)
                with torch.no_grad():
                    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id, **gen)
                text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
                row = make_row(it, text_prompt, text)
                row.update({"format": fmt, "question_kind": qname})
                rows.append(row)
            print(f"  {qname}/{fmt}: {len(pairs)} prompts, {time.time() - t0:.0f}s", flush=True)
    _chat.SYSTEM_PROMPT[0] = ""

    acc = defaultdict(list)
    for r in rows:
        acc[(r["format"], r["question_kind"], "all")].append(r["correct"])
        acc[(r["format"], r["question_kind"], r["construction"])].append(r["correct"])
    cells = {"|".join(k): round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}
    drop = {f: round(cells.get(f"{f}|working|all", float("nan")) - cells.get(f"{f}|failing|all", float("nan")), 3)
            for f in formats}
    report = {"experiment": "OTR-15 prompt robustness", "otr15_version": OTR15_VERSION, "model": args.model,
              "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs, "formats": formats,
              "questions": QUESTIONS, "instruction": args.instruction, "system_prompt": SYSTEM,
              "fewshot": FEWSHOT, "n_per_question": args.n, "accuracy": cells, "working_minus_failing": drop,
              "kinds": {k: sum(r["kind"] == k for r in rows) for k in ("direct", "frame", "statement", "inferred",
                                                                        "ambiguous", "none")},
              "seconds": round(time.time() - t0), "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr15_{args.model.replace('/', '__')}_dev_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    for f in formats:
        print(f"  {f:9} failing {cells.get(f'{f}|failing|all')}  working {cells.get(f'{f}|working|all')}  drop {drop[f]}")


if __name__ == "__main__":
    main()
