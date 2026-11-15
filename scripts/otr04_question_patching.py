"""OTR-04: question-side patching — where is the question linked to the roles?

Base and source share the sentence and differ only in the question. Mode
``flip``: the source asks for the other role (same voice); mode ``repair``: the
base is answered wrongly and the source asks for the same role in the other
voice and is answered correctly. See relational_lab/experiments/order_role_question.py.

Example:
    python scripts/otr04_question_patching.py --model google/gemma-4-31B-it --chat-template --plot

Exploratory (dev split) until the protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_question as orq
from relational_lab.experiments import order_role_stimuli as ors


def select(items, split, constructions, n_events, seed):
    import random

    pool = [i for i in items if i.split == split and i.construction in constructions]
    events = sorted({i.event_id for i in pool})
    random.Random(seed).shuffle(events)
    keep = set(events[:n_events])
    return [i for i in pool if i.event_id in keep]


def plot(agg, out_png, metric="ratio_of_means"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [g for g in agg if not g.endswith("/ALL")]
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 0.35 * 12 * len(groups) / 2 + 1),
                             squeeze=False, sharex=True, layout="constrained")
    for ax, g in zip(axes[:, 0], groups):
        labels = orq.label_order(list(agg[g]))
        grid = [agg[g][lab][metric] for lab in labels]
        im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{g} (n={agg[g][labels[0]]['n']})", fontsize=9, loc="left")
    n_layers = len(next(iter(agg[groups[0]].values()))[metric])
    step = max(1, n_layers // 14)
    axes[-1, 0].set_xticks(range(0, n_layers, step))
    axes[-1, 0].set_xlabel("layer (output of decoder block)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.5,
                 label="share of the answer moved toward the source (ratio of means)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main(argv=None):
    p = argparse.ArgumentParser(description="OTR-04 question-side patching")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--constructions", nargs="+", default=list(ors.MAIN_CONSTRUCTIONS))
    p.add_argument("--modes", nargs="+", default=list(orq.MODES), choices=list(orq.ALL_MODES),
                   help="flip repair (same sentence); repair_x control_x flip_x (cross-sentence sources)")
    p.add_argument("--n-events", type=int, default=32)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap prompts in the tokenizer's chat template (Instruct/-it models), as in OTR-01")
    p.add_argument("--no-thinking", action="store_true",
                   help="pass enable_thinking=False to the chat template (Qwen3-type / OLMo-3 Think models)")
    p.add_argument("--include-unselected", action="store_true",
                   help="diagnostic only: analyse every pair regardless of which ones the model gets right")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking)

    import torch
    import transformers
    from transformers import AutoTokenizer

    from relational_lab.instrumentation.hf_scoring import check_names_for_patching, load_lm

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if not tok.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mappings.")
    if args.chat_template and tok.chat_template is None:
        raise SystemExit(f"{args.model} has no chat template.")
    problems = check_names_for_patching(tok, ors.NAMES, args.chat_template)
    if problems:
        raise SystemExit(f"Name list unusable for {args.model}: " + "; ".join(problems))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(args.model, revision=args.revision, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    items = select(ors.generate_items(seed=args.seed), args.split, set(args.constructions), args.n_events, args.seed)
    from relational_lab.instrumentation.patching import decoder_layers
    n_layers = len(decoder_layers(model))
    layers = list(range(0, n_layers, args.layer_stride))
    if layers[-1] != n_layers - 1:
        layers.append(n_layers - 1)
    print(f"{len(items)} base prompts, {args.model}, modes {args.modes}, {len(layers)} layers", flush=True)
    t0 = time.time()
    records, skipped = [], {}
    for mode in args.modes:
        recs, sk = orq.sweep(model, tok, items, mode=mode, layers=layers, batch_size=args.batch_size,
                             require_selection=not args.include_unselected, chat_template=args.chat_template)
        records += recs
        skipped.update({f"{mode}:{k}": v for k, v in sk.items()})
        print(f"  {mode}: {len(recs)} analysed, skipped {sk}", flush=True)
    agg = orq.aggregate(records)

    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr04_{slug}_{args.split}_{stamp}.json"
    out.write_text(json.dumps({
        "experiment": "OTR-04",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": device,
        "generator_version": ors.GENERATOR_VERSION, "seed": args.seed,
        "constructions": args.constructions, "modes": args.modes, "n_events": args.n_events,
        "layers": layers, "include_unselected": args.include_unselected, "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "n_base_prompts": len(items), "n_analysed": len(records), "skipped": skipped,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "records": [r.__dict__ for r in records],
    }))
    print(f"Wrote {out}  ({len(records)} analysed, skipped {skipped})")
    if args.plot and records:
        png = out.with_suffix(".png")
        plot(agg, png)
        print(f"Wrote {png}")

    print("\nPer group: peak share moved (ratio of means) and its layer, for each position")
    for g in agg:
        print(f"  {g} (n={next(iter(agg[g].values()))['n']})")
        for lab in orq.label_order(list(agg[g])):
            eff = agg[g][lab]["ratio_of_means"]
            vals = [(e, i) for i, e in enumerate(eff) if not math.isnan(e)]
            e, i = max(vals) if vals else (float("nan"), -1)
            fr = agg[g][lab]["flip_rate"]
            j = max(range(len(fr)), key=lambda k: fr[k])
            print(f"    {lab:8s} effect peak {e:5.2f} at layer {layers[i] if i >= 0 else '-'}"
                  f"   flip peak {fr[j]:.2f} at layer {layers[j]}")


if __name__ == "__main__":
    main()
