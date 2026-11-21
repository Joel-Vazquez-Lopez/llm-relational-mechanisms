"""OTR-02 (E2): position x layer patching map — where is the answer decided?

For every base prompt the source is its role-swapped twin; the residual stream
at one (position, layer) is copied from source to base and the change in the
answer logit difference is recorded. Only pairs the model gets right in both
directions are analysed (counts of the rest are reported).

Example:
    python scripts/otr02_position_layer_patching.py --model Qwen/Qwen2.5-7B --n-events 48 --plot

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

from relational_lab.experiments import order_role_patching as orp
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

    constructions = [c for c in agg if c != "ALL"] + ["ALL"]
    fig, axes = plt.subplots(len(constructions), 1, figsize=(10, 2.0 * len(constructions)),
                             squeeze=False, sharex=True, layout="constrained")
    for ax, c in zip(axes[:, 0], constructions):
        labels = [lab for lab in ors.POSITION_LABELS if lab in agg[c]]
        grid = [agg[c][lab][metric] for lab in labels]
        im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{c} (n={agg[c][labels[0]]['n']})", fontsize=10, loc="left")
    n_layers = len(next(iter(agg["ALL"].values()))[metric])
    step = max(1, n_layers // 14)
    axes[-1, 0].set_xticks(range(0, n_layers, step))
    axes[-1, 0].set_xlabel("layer (output of decoder block)")
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.5,
                 label="share of the answer moved (ratio of means, clipped to 0-1)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main(argv=None):
    p = argparse.ArgumentParser(description="OTR-02 position x layer patching")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--constructions", nargs="+", default=list(ors.MAIN_CONSTRUCTIONS + ors.HELDOUT_CONSTRUCTIONS))
    p.add_argument("--n-events", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--include-incorrect", action="store_true",
                   help="diagnostic only: also analyse pairs the model gets wrong")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from relational_lab.instrumentation.hf_scoring import single_token_checker

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if not tok.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mappings.")
    check = single_token_checker(tok)
    multi = [n for n in ors.NAMES if not check(n)]
    if multi:
        raise SystemExit(f"Names not single-token for {args.model}: {multi}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    items = select(ors.generate_items(seed=args.seed), args.split, set(args.constructions), args.n_events, args.seed)
    print(f"{len(items)} base prompts, {args.model}", flush=True)
    t0 = time.time()
    records, skipped = orp.sweep(model, tok, items, batch_size=args.batch_size,
                                 require_correct=not args.include_incorrect)
    agg = orp.aggregate(records)

    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr02_{slug}_{args.split}_{stamp}.json"
    out.write_text(json.dumps({
        "experiment": "OTR-02",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": device,
        "generator_version": ors.GENERATOR_VERSION, "seed": args.seed,
        "constructions": args.constructions, "n_events": args.n_events,
        "include_incorrect": args.include_incorrect,
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

    # Text summary: for each position, the layer where pooled effect first exceeds 0.5.
    print("\nPooled over constructions: first layer with effect (ratio of means) > 0.5")
    for lab in ors.POSITION_LABELS:
        if lab in agg.get("ALL", {}):
            eff = agg["ALL"][lab]["ratio_of_means"]
            hit = next((i for i, e in enumerate(eff) if not math.isnan(e) and e > 0.5), None)
            print(f"  {lab:13s} {'-' if hit is None else hit:>4}   peak {max(e for e in eff if not math.isnan(e)):.2f}")


if __name__ == "__main__":
    main()
