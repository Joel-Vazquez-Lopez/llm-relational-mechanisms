"""OTR-05: wording and format robustness of the order-to-role behaviour.

Same sentences as OTR-01 (v1.3.0 generator, dev split by default); the
question is replaced by other wordings of the same question, and by cloze
statements with no question. Reports accuracy per variant x construction and
how often the chosen name is the sentence's grammatical subject.

Example:
    python scripts/otr05_robustness.py --model Qwen/Qwen2.5-7B-Instruct --chat-template
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_robustness as orr
from relational_lab.experiments import order_role_stimuli as ors


def main(argv=None):
    p = argparse.ArgumentParser(description="OTR-05 wording/format robustness")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--n-events", type=int, default=None, help="limit events (default: all in the split)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true",
                   help="pass enable_thinking=False to the chat template (Qwen3-type / OLMo-3 Think models)")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra chat-template variable, e.g. reasoning_strength=low (repeatable)")
    p.add_argument("--assistant-prefix", default="",
                   help="text appended after the generation prompt, e.g. ' to=user<|message|>' (Muse Glimmer)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)

    import torch
    import transformers
    from transformers import AutoTokenizer

    from relational_lab.instrumentation.hf_scoring import HFScorer, load_lm, load_tokenizer

    tok = load_tokenizer(args.model, revision=args.revision)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(args.model, revision=args.revision, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    items = orr.make_items(split=args.split, n_events=args.n_events, seed=args.seed)
    print(f"Scoring {len(items)} prompts ({len(orr.VARIANTS)} variants) with {args.model} ...", flush=True)
    t0 = time.time()
    scores = HFScorer(model, tok, batch_size=args.batch_size, use_chat_template=args.chat_template)(items)
    margins = {s.item_id: s.logprob_answer - s.logprob_distractor for s in scores}
    summary = orr.summarize(items, margins)

    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr05_{slug}_{args.split}_{stamp}.json"
    out.write_text(json.dumps({
        "experiment": "OTR-05",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": device,
        "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "split": args.split, "seed": args.seed, "n_events": args.n_events,
        "generator_version": ors.GENERATOR_VERSION, "robustness_version": orr.ROBUSTNESS_VERSION,
        "variants": [v.__dict__ for v in orr.VARIANTS],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
        "summary": summary,
        "margins": margins,
    }, indent=1))
    print(f"\nWrote {out}\n")
    cons = list(ors.MAIN_CONSTRUCTIONS)
    print(f"{'variant':18s} {'extraction':10s} {'acc':>5s} {'subj':>5s}  " + " ".join(f"{c[:8]:>8s}" for c in cons))
    for v in orr.VARIANTS:
        d = summary.get(v.name)
        if not d:
            continue
        row = " ".join(f"{d['by_construction'][c]['accuracy']:8.2f}" for c in cons if c in d["by_construction"])
        print(f"{v.name:18s} {v.extraction:10s} {d['accuracy']:5.2f} {d['chose_subject']:5.2f}  {row}")


if __name__ == "__main__":
    main()
