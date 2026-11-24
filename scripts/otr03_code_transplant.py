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
import dataclasses
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import bx_adapter as bxa
from relational_lab.experiments.splits import SPLITS, in_split
from relational_lab.experiments import broad_stimuli as bx
from relational_lab.experiments import order_role_stimuli as ors
from relational_lab.experiments import multilingual_stimuli as xl
from relational_lab.experiments import xl_adapter as xla
from relational_lab.experiments import order_role_transplant as otr
from relational_lab.experiments.question_templates import fill as fill_template


def select_bases(items, split, n_events, question_voices, seed):
    pool = [i for i in items if in_split(i, split) and i.construction in ors.MAIN_CONSTRUCTIONS
            and i.question_voice in question_voices]
    events = sorted({i.event_id for i in pool})
    random.Random(seed).shuffle(events)
    keep = set(events[:n_events])
    return [i for i in pool if i.event_id in keep]


def with_agent_question(items, template):
    """Replace the question of agent-question items by ``template`` ({past}/{lemma}/{ing}).

    Only the question text changes: the sentence, and so every name position, is untouched."""
    out = []
    for it in items:
        if it.role_asked == "agent":
            q = fill_template(template, it.verb, it.lemma)
            prompt = it.prompt.replace(f"\nQuestion: {it.question}\n", f"\nQuestion: {q}\n")
            assert prompt.count(q) == 1, prompt
            it = dataclasses.replace(it, question=q, prompt=prompt)
        out.append(it)
    return out


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
    p.add_argument("--split", default="dev", choices=SPLITS, help="test0..test3 only after FREEZE.md")
    p.add_argument("--stimuli", default="otr", choices=["otr", *bxa.STIMULI, *xla.STIMULI],
                   help="otr = the original materials; bx-* = broad English blocks (new names, verbs, "
                        "constructions). For bx, --roles agent/patient mean the block's A/B slot "
                        "(agent/patient, agent/recipient, stimulus/experiencer). es/gl = Spanish/Galician (XL); "
                        "their passive agent question (--question-voices passive --roles agent) is the failing one")
    p.add_argument("--n-events", type=int, default=32)
    p.add_argument("--question-voices", nargs="+", default=["active"], choices=list(ors.QUESTION_VOICES))
    p.add_argument("--roles", nargs="+", default=["agent", "patient"], choices=["agent", "patient"],
                   help="which base questions to analyse")
    p.add_argument("--agent-question", default=None, metavar="TEMPLATE",
                   help='replace the agent question, e.g. "Who was someone {past} by?" (the failing wording). '
                        'With bx stimuli: the question for the by-phrase role, e.g. "Who was someone {pp} by?" '
                        '(only by-phrase-role bases are kept; fear-class psych verbs: the experiencer)')
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--layer-stride", type=int, default=1)
    p.add_argument("--layers", default=None, metavar="L1,L2,...",
                   help="explicit layer list (overrides --layer-stride), e.g. to include frozen layers")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--include-incorrect", action="store_true",
                   help="also analyse bases the model answers wrongly (diagnostic)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap prompts in the tokenizer's chat template (Instruct/-it models), as in OTR-01")
    p.add_argument("--no-thinking", action="store_true",
                   help="pass enable_thinking=False to the chat template (Qwen3-type / OLMo-3 Think models)")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra chat-template variable, e.g. reasoning_strength=low (repeatable)")
    p.add_argument("--assistant-prefix", default="",
                   help="text appended after the generation prompt, e.g. ' to=user<|message|>' (Muse Glimmer)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)

    import torch
    import transformers
    from transformers import AutoTokenizer

    from relational_lab.instrumentation.hf_scoring import check_names_for_patching, load_lm, load_tokenizer

    tok = load_tokenizer(args.model, revision=args.revision)
    if not tok.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mappings.")
    if args.chat_template and tok.chat_template is None:
        raise SystemExit(f"{args.model} has no chat template.")
    is_bx = args.stimuli in bxa.STIMULI
    is_xl = args.stimuli in xla.STIMULI
    names = bx.NAMES if is_bx else ors.NAMES
    problems = check_names_for_patching(tok, names, args.chat_template)
    dropped_names = []
    if problems and is_bx:
        # 48 new names: drop the ones this tokenizer cannot patch (multi-token, or a first answer
        # token shared with another name) rather than abandoning the run; they are recorded.
        bad = {n for n in names if any(repr(n) in msg for msg in problems)}
        names = tuple(n for n in names if n not in bad)
        dropped_names, problems = sorted(bad), check_names_for_patching(tok, names, args.chat_template)
        print(f"dropping {len(dropped_names)} names this tokenizer cannot patch: {dropped_names}", flush=True)
    if problems:
        raise SystemExit(f"Name list unusable for {args.model}: " + "; ".join(problems))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lm(args.model, revision=args.revision, dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    from relational_lab.instrumentation.patching import decoder_layers

    n_layers = len(decoder_layers(model))
    if args.layers:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        if not layers or layers[0] < 0 or layers[-1] >= n_layers:
            raise SystemExit(f"--layers must lie in 0..{n_layers - 1}")
    else:
        layers = list(range(0, n_layers, args.layer_stride))
    # Test splits: donor name pairs come from the same split as the bases (no dev pair enters a
    # test run). Dev keeps the original unrestricted draw, so archived dev runs reproduce exactly.
    pair_ok = None
    if args.split != "dev":
        fam_split = ors.split_families(bx.NAMES if is_bx else ors.NAMES, args.seed, 0.15, 4)
        want = ("test", int(args.split[-1]))
        pair_ok = lambda a, b: fam_split["|".join(sorted((a, b)))] == want  # noqa: E731
    if is_xl:
        if args.agent_question:
            raise SystemExit("es/gl: use --question-voices passive --roles agent for the failing question")
        bases = xla.select_bases(args.stimuli, args.split, args.n_events, args.question_voices, args.roles, args.seed)
        transplants = otr.make_transplants(bases, xla.TRANSPLANT_CONSTRUCTIONS, seed=args.seed, builder=xla.builder,
                                           pair_ok=pair_ok)
        null_cons = ("active", "passive")
    elif is_bx:
        bases = bxa.select_bases(args.stimuli, args.split, args.n_events, args.question_voices, args.seed,
                                 roles=("agent", "patient") if args.agent_question else args.roles)
        bases = [b for b in bases if b.agent in names and b.patient in names]
        if args.agent_question:
            bases = bxa.with_by_question(bases, args.agent_question)
        transplants = otr.make_transplants(bases, seed=args.seed, names=names, builder=bxa.builder,
                                           constructions_for=bxa.source_constructions, pair_ok=pair_ok)
        null_cons = bxa.NULL_CONSTRUCTIONS[bxa.block_name(args.stimuli)]
    else:
        bases = select_bases(ors.generate_items(seed=args.seed), args.split, args.n_events,
                             set(args.question_voices), args.seed)
        bases = [b for b in bases if b.role_asked in args.roles]
        if args.agent_question:
            bases = with_agent_question(bases, args.agent_question)
        transplants = otr.make_transplants(bases, seed=args.seed, pair_ok=pair_ok)
        null_cons = ("active", "passive")
    donor_pairs = {t.base.item_id: "|".join(sorted((t.source.agent, t.source.patient))) for t in transplants}
    if pair_ok is not None:
        bad = [b.item_id for b in bases if not pair_ok(b.agent, b.patient)]
        bad += [i for i, fam in donor_pairs.items() if not pair_ok(*fam.split("|"))]
        if bad:
            raise SystemExit(f"split leak: {len(bad)} base/donor pairs outside {args.split}: {bad[:5]}")
    print(f"{len(bases)} bases, {len(transplants)} transplants x {len(layers)} layers, {args.model}", flush=True)
    t0 = time.time()
    records, skipped = otr.sweep(model, tok, transplants, layers=layers, batch_size=args.batch_size,
                                 require_base_correct=not args.include_incorrect,
                                 chat_template=args.chat_template)
    if not records:
        raise SystemExit(f"No transplants analysed (skipped: {skipped}). The model may fail the task; "
                         "try --include-incorrect for a diagnostic run.")
    fit_error = None
    try:
        profile = otr.fit_profile(records, n_boot=args.n_boot, seed=args.seed)
    except ValueError as exc:  # never lose the forward passes over an analysis problem
        profile, fit_error = None, str(exc)
        print(f"!! profile fit failed ({exc}); records are still written")
    null = otr.left_context_null(records, null_cons)
    # With --include-incorrect: the same profile on bases the model answers correctly vs wrongly.
    by_base = {}
    if args.include_incorrect:
        for key, keep in (("base_correct", True), ("base_wrong", False)):
            sub = [r for r in records if r.base_correct is keep]
            if len({r.base_id for r in sub}) >= 8:
                try:
                    by_base[key] = {"n_records": len(sub),
                                    "profile": otr.fit_profile(sub, n_boot=args.n_boot, seed=args.seed)}
                except ValueError as exc:
                    by_base[key] = {"n_records": len(sub), "error": str(exc)}

    # Psych: frighten-class and fear-class separately (role = stimulus vs experiencer; in fear-class
    # actives the subject is the experiencer, so "role" and "subject" come apart within the class).
    by_class = {}
    if args.stimuli == "bx-psych":
        for cls in ("se", "es"):
            sub = [r for r in records if r.base_construction.startswith(cls + "_")]
            if len({r.base_id for r in sub}) >= 8:
                try:
                    by_class[cls] = {"n_records": len(sub),
                                     "profile": otr.fit_profile(sub, n_boot=args.n_boot, seed=args.seed)}
                except ValueError as exc:
                    by_class[cls] = {"n_records": len(sub), "error": str(exc)}

    slug = args.model.replace("/", "__")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qtag = "_aq" if args.agent_question else ""
    stag = "" if args.stimuli == "otr" else f"_{args.stimuli}"
    out = args.out_dir / f"otr03_{slug}_{args.split}{stag}{qtag}_{stamp}.json"
    out.write_text(json.dumps({
        "experiment": "OTR-03",
        "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED",
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": device,
        "generator_version": ors.GENERATOR_VERSION, "stimuli": args.stimuli,
        "bx_generator_version": bx.BX_GENERATOR_VERSION if is_bx else None,
        "xl_generator_version": xl.XL_GENERATOR_VERSION if is_xl else None,
        "seed": args.seed, "layers": layers, "donor_pairs": donor_pairs,
        "donor_pairs_restricted_to_split": pair_ok is not None,
        "base_constructions": sorted({b.construction for b in bases}),
        "dropped_names": dropped_names,
        "n_events": args.n_events, "question_voices": args.question_voices, "roles": args.roles,
        "agent_question": args.agent_question,
        "include_incorrect": args.include_incorrect, "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "n_bases": len(bases), "n_transplants": len(transplants), "n_analysed": len(records),
        "skipped": skipped,
        "predictor_correlations": otr.predictor_correlations([t for t in transplants]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "seconds": round(time.time() - t0, 1),
        "profile": profile,
        "profile_fit_error": fit_error,
        "profile_by_base_correctness": by_base,
        "profile_by_psych_class": by_class,
        "left_context_null_role_slope": null,
        "records": [r.__dict__ for r in records],
        "base_real_roles": ({b.item_id: bxa.real_role(b) for b in bases} if is_bx else None),
    }))
    print(f"Wrote {out}  ({len(records)} analysed, skipped {skipped})")
    if args.plot and profile is not None:
        png = out.with_suffix(".png")
        plot(profile, png, f"{args.model}: code profile (n={len(records)} transplants)")
        print(f"Wrote {png}")

    if profile is None:
        return
    for key, v in by_base.items():
        if "profile" not in v:
            print(f"{key}: {v['n_records']} records, fit failed: {v['error']}")
            continue
        rc = v["profile"]["role"]["coef"]
        i = max(range(len(rc)), key=lambda j: rc[j])
        print(f"{key}: {v['n_records']} records, peak role {rc[i]:+.2f} at layer {layers[i]}")
    worst = max((abs(v) for v in null[:-1]), default=float("nan"))
    print(f"\nValidity check (must be ~0): max |role slope| for identical-left-context sources = {worst:.3f}")
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
