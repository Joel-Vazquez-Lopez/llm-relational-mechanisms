"""OTR-03 (E3): code transplant profile — order vs function vs role, layer by layer.

Copies one name's residual state from a source sentence with different names
into a name slot of the base sentence, at every layer, and regresses the shift
in the answer on three uncorrelated predictors (order, function, role). The
three coefficient curves are the code profile. See
src/relational_lab/experiments/order_role_transplant.py for the full design.

Example:
    python scripts/otr03_code_transplant.py --model Qwen/Qwen2.5-7B --n-events 32 --plot

Exploratory (dev split) until the protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments import order_role_transplant as otr


def select_bases(items, split, n_events, question_voices, seed):
    pool = [i for i in items if i.split == split and i.construction in ors.MAIN_CONSTRUCTIONS
            and i.question_voice in question_voices]
    events = sorted({i.event_id for i in pool})
    random.Random(seed).shuffle(events)
    keep = set(events[:n_events])
    return [i for i in pool if i.event_id in keep]


def plot(profile, out_png, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    colors = {"order": "#4C72B0", "function": "#DD8452", "role": "#55A868"}
    n = len(profile["role"]["coef"])
    xs = list(range(n))
    for c in otr.CODES:
        ax.plot(xs, profile[c]["coef"], label=c, color=colors[c], lw=2)
        if profile[c].get("ci95"):
            lo = [v[0] for v in profile[c]["ci95"]]
            hi = [v[1] for v in profile[c]["ci95"]]
            ax.fill_between(xs, lo, hi, color=colors[c], alpha=0.2, lw=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("layer at which the name state is transplanted")
    ax.set_ylabel("shift toward the slot's name (logits)\nper unit of predictor")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description="OTR-03 code transplant profile")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--n-events", type=int, default=32)
    p.add_argument("--question-voices", nargs="+", default=["active"], choices=list(ors.QUESTION_VOICES))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--include-incorrect", action="store_true",
                   help="also analyse bases the model answers wrongly (diagnostic)")
    p.add_argument("--plot", action="store_true")
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

    from relational_lab.instrumentation.patching import decoder_layers

    n_layers = len(decoder_layers(model))
    layers = list(range(0, n_layers, args.layer_stride))
    bases = select_bases(ors.generate_items(seed=args.seed), args.split, args.n_events,
                         set(args.question_voices), args.seed)
    transplants = otr.make_transplants(bases, seed=args.seed)
    print(f"{len(bases)} bases, {len(transplants)} transplants x {len(layers)} layers, {args.model}", flush=True)
    t0 = time.time()
    records, skipped = otr.sweep(model, tok, transplants, layers=layers, batch_size=args.batch_size,
                                 require_base_correct=not args.include_incorrect)
    if not records:
        raise SystemExit(f"No transplants analysed (skipped: {skipped}). The model may fail the task; "
                         "try --include-incorrect for a diagnostic run.")
    profile = otr.fit_profile(records, n_boot=args.n_boot, seed=args.seed)

    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr03_{slug}_{args.split}_{stamp}.json"
    out.write_text(json.dumps({
        "experiment": "OTR-03",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": device,
        "generator_version": ors.GENERATOR_VERSION, "seed": args.seed, "layers": layers,
        "n_events": args.n_events, "question_voices": args.question_voices,
        "include_incorrect": args.include_incorrect,
        "n_bases": len(bases), "n_transplants": len(transplants), "n_analysed": len(records),
        "skipped": skipped,
        "predictor_correlations": otr.predictor_correlations([t for t in transplants]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
        "profile": profile,
        "records": [r.__dict__ for r in records],
    }))
    print(f"Wrote {out}  ({len(records)} analysed, skipped {skipped})")
    if args.plot:
        png = out.with_suffix(".png")
        plot(profile, png, f"{args.model}: code profile (n={len(records)} transplants)")
        print(f"Wrote {png}")

    print("\nCode profile: coefficient [95% CI] at selected layers")
    for li in sorted({0, len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4, len(layers) - 2}):
        cells = []
        for c in otr.CODES:
            coef = profile[c]["coef"][li]
            ci = profile[c]["ci95"][li] if profile[c].get("ci95") else [float("nan")] * 2
            cells.append(f"{c} {coef:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
        print(f"  layer {layers[li]:>2}: " + "   ".join(cells))


if __name__ == "__main__":
    main()
