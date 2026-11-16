"""OTR-06: logit lens at the answer position, by layer, for any set of question wordings.

For each dev event and each condition (role asked + question template), the residual
stream after every decoder block at the LAST prompt position (where the answer is
generated) is passed through the model's final norm and unembedding. The score per layer
is the logit difference answer - distractor on each name's first answer token (no leading
space in chat-template mode). The last layer reproduces the model's own logits up to
numerical error (``final_layer_check`` in the output; it must be ~0).

Typical use: the stranded agent question, its patient permutation and the pied-piped
control, with and without an instruction line:

    python scripts/otr06_logit_lens.py --model google/gemma-4-31B-it --chat-template \\
        --condition agent "Who was someone {past} by?" \\
        --condition patient "Who was {past} by someone?" \\
        --condition agent "By whom was someone {past}?"

Shapes to look for (per construction): a curve that rises towards the correct name and
then falls late ("integrated, then overridden"), one that never rises ("never
integrated"), or one that stays near zero until a late decision ("late").
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

LENS_VERSION = "1.0.0"
_NORM_PATHS = ("model.norm", "model.language_model.norm", "language_model.model.norm",
               "model.model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm")


def _get(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a)
    return obj


def final_norm(model):
    for p in _NORM_PATHS:
        try:
            return _get(model, p)
        except AttributeError:
            continue
    raise ValueError(f"cannot find the final norm on {type(model).__name__}")


def softcap(model) -> float | None:
    cfg = model.config
    for c in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(c, "final_logit_softcapping", None) if c is not None else None
        if v:
            return float(v)
    return None


def build_prompts(seed: int, conditions, instruction: str, n_per: int | None, stimuli: str = "otr"):
    """One prompt per (dev event in the four main constructions, condition).

    With ``stimuli="sp"`` each condition is (role, question form) on the stranded-preposition
    frames, e.g. ("oblique", "stranded"); every SP construction is included."""
    if stimuli == "sp":
        from relational_lab.experiments import stranded_p_stimuli as sp
        dev = [i for i in sp.generate_items(seed=seed) if i.split == "dev"]
        out = []
        for ci, (role, qform) in enumerate(conditions):
            if qform not in sp.QUESTION_FORMS or sp.QUESTION_FORMS[qform] != role:
                raise ValueError(f"SP condition must be one of {sp.QUESTION_FORMS} as ROLE FORM, got {role} {qform}")
            per = defaultdict(int)
            for it in dev:
                if it.question_voice != qform or (n_per and per[it.construction] >= n_per):
                    continue
                per[it.construction] += 1
                prompt = it.prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:") if instruction else it.prompt
                out.append((ci, dataclasses.replace(it, prompt=prompt)))
        return out
    items = [i for i in ors.generate_items(seed=seed)
             if i.split == "dev" and i.construction in ors.MAIN_CONSTRUCTIONS
             and i.question_voice == "active"]
    out = []
    for ci, (role, template) in enumerate(conditions):
        per = defaultdict(int)
        for it in items:
            if it.role_asked != role:
                continue
            if n_per and per[it.construction] >= n_per:
                continue
            per[it.construction] += 1
            q = template.format(past=it.verb, lemma=it.lemma)
            prompt = it.prompt.replace(f"\nQuestion: {it.question}\n", f"\nQuestion: {q}\n")
            assert prompt.count(q) == 1, prompt
            if instruction:
                prompt = prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:")
            out.append((ci, dataclasses.replace(it, question=q, prompt=prompt)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--condition", nargs=2, action="append", metavar=("ROLE", "TEMPLATE"), required=True,
                   help='role asked (agent|patient) and question template with {past}/{lemma}')
    p.add_argument("--instruction", default="", help='line before "Answer:", e.g. "Reply with one name only."')
    p.add_argument("--stimuli", choices=("otr", "sp"), default="otr",
                   help='sp: stranded-preposition frames; conditions are ROLE FORM, e.g. --condition oblique stranded')
    p.add_argument("--n-per-construction", type=int, default=None, help="default: all dev events")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    for role, _ in args.condition:
        if role not in (("agent", "oblique") if args.stimuli == "sp" else ("agent", "patient")):
            p.error(f"role must be agent or patient, got {role!r}")

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.instrumentation.hf_scoring import _prompt_ids, load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import ResidualPatcher, answer_token

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    patcher = ResidualPatcher(model)
    norm, head, cap = final_norm(model), model.get_output_embeddings(), softcap(model)
    lead = not args.chat_template

    prompts = build_prompts(args.seed, args.condition, args.instruction, args.n_per_construction, args.stimuli)
    t0 = time.time()
    records, worst_check = [], 0.0
    for ci, it in prompts:
        ids = torch.tensor([_prompt_ids(tok, it.prompt, args.chat_template)])
        mask = torch.ones_like(ids)
        a, d = answer_token(tok, it.answer, lead), answer_token(tok, it.distractor, lead)
        store, logits = patcher.capture(ids, mask)
        with torch.no_grad():
            h = torch.stack([s[0, -1] for s in store])            # [L, D]
            lg = head(norm(h.to(head.weight.dtype))).float()       # [L, V]
            if cap:
                lg = cap * torch.tanh(lg / cap)
            ld = (lg[:, a] - lg[:, d]).tolist()
            true_ld = float(logits[0, -1, a].float() - logits[0, -1, d].float())
        worst_check = max(worst_check, abs(ld[-1] - true_ld))
        records.append({"item_id": it.item_id, "condition": ci, "construction": it.construction,
                        "role": it.role_asked, "answer": it.answer,
                        "ld": [round(x, 3) for x in ld], "ld_model": round(true_ld, 3)})

    n_layers = patcher.n_layers
    constructions = sorted({r["construction"] for r in records}, key=lambda c: (c.split("_")[0], c))
    agg = {}
    for ci, (role, template) in enumerate(args.condition):
        for cons in constructions:
            rs = [r for r in records if r["condition"] == ci and r["construction"] == cons]
            if not rs:
                continue
            mean = [sum(r["ld"][l] for r in rs) / len(rs) for l in range(n_layers)]
            frac = [sum(r["ld"][l] > 0 for r in rs) / len(rs) for l in range(n_layers)]
            agg[f"{ci}|{role}|{template}|{cons}"] = {
                "n": len(rs), "mean_ld": [round(x, 3) for x in mean], "frac_correct": [round(x, 3) for x in frac],
                "final_accuracy": sum(r["ld_model"] > 0 for r in rs) / len(rs),
                "peak_correct_layer": max(range(n_layers), key=lambda l: mean[l]),
                "peak_correct_ld": round(max(mean), 3),
            }
    report = {
        "experiment": "OTR-06", "status": "EXPLORATORY", "lens_version": LENS_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "stimuli": args.stimuli, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "instruction": args.instruction,
        "conditions": [{"role": r, "template": t} for r, t in args.condition], "n_layers": n_layers,
        "softcap": cap, "final_layer_check_max_abs": round(worst_check, 4),
        "seconds": round(time.time() - t0, 1), "aggregate": agg, "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("_sp" if args.stimuli == "sp" else "") + ("_instr" if args.instruction else "")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.out_dir / f"otr06_{args.model.replace('/', '__')}_dev{tag}_{stamp}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    print(f"final-layer check (must be ~0): {worst_check:.4f}")
    for k, v in agg.items():
        ci, role, template, cons = k.split("|")
        print(f"  [{role:7}] {template:32} {cons:13} final acc {v['final_accuracy']:.2f}  "
              f"peak {v['peak_correct_ld']:+.2f} @L{v['peak_correct_layer']}  last {v['mean_ld'][-1]:+.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conds = list(enumerate(args.condition))
        ncol = min(4, len(constructions))
        nrow = -(-len(constructions) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharey=True, squeeze=False)
        axes = axes.ravel()
        depth = [l / max(1, n_layers - 1) for l in range(n_layers)]
        for ax, cons in zip(axes, constructions):
            for ci, (role, template) in conds:
                key = f"{ci}|{role}|{template}|{cons}"
                if key in agg:
                    ax.plot(depth, agg[key]["mean_ld"], label=f"{template} [{role}]")
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_title(cons)
            ax.set_xlabel("relative depth")
        axes[0].set_ylabel("logit diff (correct - other)")
        axes[len(constructions) - 1].legend(fontsize=7)
        for ax in axes[len(constructions):]:
            ax.axis("off")
        fig.suptitle(f"OTR-06 logit lens: {args.model}{' + instruction' if args.instruction else ''}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=130)
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:  # plotting is optional
        print(f"(no plot: {exc})")
    if not math.isfinite(worst_check) or worst_check > 0.5:
        print("WARNING: last-layer lens does not reproduce the model's logits; check final_norm / softcap.")


if __name__ == "__main__":
    main()
