"""OTR-01: behavioral gate for the order-to-role study.

Scores every prompt with one Hugging Face causal LM and reports accuracy by
construction x role x question voice, both-roles-correct per event,
cross-construction consistency, question-wording effects and the
pre-registered gate (candidate-restricted accuracy >= 0.90 for every
construction x role among the four main constructions).

Examples:
    python scripts/otr01_behavior_gate.py --model Qwen/Qwen2.5-1.5B --split dev
    python scripts/otr01_behavior_gate.py --model Qwen/Qwen2.5-1.5B --split dev --limit 200

Run the dev split to settle the prompt format; open a test set only once the
protocol is frozen. Exploratory until then.
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

from relational_lab.experiments import order_role_behavior as beh
from relational_lab.experiments import order_role_stimuli as ors


def select_items(items, split: str, test_set: int | None, limit: int | None):
    chosen = [i for i in items if i.split == split and (test_set is None or i.test_set == test_set)]
    if limit is not None:
        # Keep whole events so both-roles-correct and consistency stay computable.
        events = []
        for it in chosen:
            if it.event_id not in events:
                events.append(it.event_id)
        n_events = max(1, limit // (len(ors.CONSTRUCTIONS) * 4))
        keep = set(events[:n_events])
        chosen = [i for i in chosen if i.event_id in keep]
    return chosen


def items_digest(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(ors.item_to_json(it), sort_keys=True).encode())
        h.update(b"\n")
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description="OTR-01 behavioral gate")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None, help="pin a commit hash before any frozen run")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda if available)")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--test-set", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="approximate number of prompts, whole events kept")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from relational_lab.instrumentation.hf_scoring import HFScorer, single_token_checker

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    check = single_token_checker(tok, leading_space=not args.chat_template)
    multi = [n for n in ors.NAMES if not check(n)]
    if multi:
        raise SystemExit(
            f"These names are not single tokens for {args.model}: {multi}. "
            "Agree on one name list that is single-token for every model before running."
        )

    dtype = getattr(torch, args.dtype)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype)
    model.to(device).eval()

    items = select_items(ors.generate_items(seed=args.seed), args.split, args.test_set, args.limit)
    print(f"Scoring {len(items)} prompts with {args.model} ...", flush=True)
    t0 = time.time()
    scores = HFScorer(model, tok, batch_size=args.batch_size, use_chat_template=args.chat_template)(items)
    summary = beh.summarize(items, scores, seed=args.seed)

    record = {
        "experiment": "OTR-01",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "device": device,
        "chat_template": args.chat_template,
        "split": args.split,
        "test_set": args.test_set,
        "generator_version": ors.GENERATOR_VERSION,
        "seed": args.seed,
        "items_sha256": items_digest(items),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "seconds": round(time.time() - t0, 1),
        "summary": summary,
        "scores": [beh.score_to_json(s) for s in scores],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.out_dir / f"otr01_{slug}_{args.split}_{stamp}.json"
    out.write_text(json.dumps(record, indent=1))

    print(f"\nWrote {out}")
    print(f"Overall choice accuracy: {summary['overall_choice_accuracy']:.3f}")
    print("Accuracy by construction x role (candidate-restricted, 95% CI over families):")
    for key, cell in summary["by_construction_role"].items():
        lo, hi = cell["choice_accuracy_ci95"]
        print(f"  {key:24s} {cell['choice_accuracy']:.3f}  [{lo:.3f}, {hi:.3f}]  top1={cell['full_vocab_top1']:.3f}")
    print(f"Both roles correct: {summary['both_roles_correct_by_construction']}")
    print(f"Cross-construction consistency: {summary['cross_construction_consistency']:.3f}")
    print(f"Gate passed: {summary['gate']['passed']}")


if __name__ == "__main__":
    main()
