"""OTR-09: can the failing bare answer be REPAIRED from the working question? Causal test.

Same sentence, two agent questions:
  base   = the failing wording   (default "Who was someone {past} by?")
  source = a working wording     (default "Who was the {ing} done by?"; Gemma 1.00)
Everything is read at the answer position (the last prompt token), where OTR-07 places
the decision and where name-mover heads fetch a name.

1. Residual patching, layer by layer: replace the base run's residual stream at the
   answer position with the source run's. repair = (ld - ld_base) / (ld_source - ld_base),
   ld = logit(agent) - logit(patient). Controls at every layer:
     donor   the same patch from the working question about a DIFFERENT sentence (other
             names, same construction): carries "answer an agent question" but not this
             sentence's answer -> should not repair;
     reverse the failing question's state patched into the working run -> how much it
             breaks a correct answer (normalised the same way, 1 = fully broken).
2. Head patching: on a ranking half of the items (families in folds 0-2), patch each
   attention head's output at the answer position (input slice of o_proj) from source to
   base, one head at a time, in layers from --head-from (relative depth). Rank heads by
   mean repair.
3. Circuit test on the held-out half (folds 3-4): patch the top-k heads jointly,
   k in --top-k, vs k random heads from the same layers; plus any --heads given
   (e.g. OTR-07's name movers "47.17,53.26,53.18,57.7").

Linear-attention layers of hybrid models (no self_attn.o_proj) are skipped for heads.

    python scripts/otr09_answer_patching.py --model google/gemma-4-31B-it --chat-template \\
        --instruction "Reply with one name only." --heads 47.17,53.26,53.18,57.7
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402
from relational_lab.experiments.question_templates import fill as fill_template  # noqa: E402

OTR09_VERSION = "1.0.0"


def make_pairs(seed: int, base_q: str, source_q: str, instruction: str, constructions, n_per: int):
    """(base item, source item, donor item) triples: same sentence for base/source; donor =
    another dev sentence of the same construction with a disjoint name pair."""
    items = [i for i in ors.generate_items(seed=seed)
             if i.split == "dev" and i.construction in constructions and i.role_asked == "agent"
             and i.question_voice == "active"]

    def ask(it, template):
        q = fill_template(template, it.verb, it.lemma)
        prompt = it.prompt.replace(f"\nQuestion: {it.question}\n", f"\nQuestion: {q}\n")
        assert prompt.count(q) == 1, prompt
        if instruction:
            prompt = prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:")
        return dataclasses.replace(it, question=q, prompt=prompt)

    rng = random.Random(seed)
    by_cons = defaultdict(list)
    for it in items:
        by_cons[it.construction].append(it)
    out = []
    for c in constructions:
        pool = by_cons[c][:]
        rng.shuffle(pool)
        for it in pool[:n_per]:
            donors = [d for d in by_cons[c] if not {d.agent, d.patient} & {it.agent, it.patient}]
            d = rng.choice(donors)
            out.append((ask(it, base_q), ask(it, source_q), ask(d, source_q)))
    return out


def repair(ld, ld_base, ld_src):
    gap = ld_src - ld_base
    return (ld - ld_base) / gap if abs(gap) > 1e-6 else float("nan")


def attn_modules(layers):
    """{layer index: (o_proj module, n_heads, head_dim)} for layers with standard attention."""
    out = {}
    for li, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        o = getattr(attn, "o_proj", None) if attn is not None else None
        if o is None:
            continue
        cfg = getattr(attn, "config", None)
        n_heads = (getattr(attn, "num_heads", None) or getattr(cfg, "num_attention_heads", None))
        if not n_heads:
            continue
        out[li] = (o, int(n_heads), o.in_features // int(n_heads))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-question", default="Who was someone {past} by?")
    p.add_argument("--source-question", default="Who was the {ing} done by?")
    p.add_argument("--instruction", default="")
    p.add_argument("--constructions", default="active,object_cleft,passive,by_cleft")
    p.add_argument("--n-per", type=int, default=32, help="sentences per construction")
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--head-from", type=float, default=0.55, help="relative depth from which heads are scanned")
    p.add_argument("--n-rank", type=int, default=24, help="ranking items for the head scan (largest gaps)")
    p.add_argument("--top-k", default="1,3,5,10,20")
    p.add_argument("--min-gap", type=float, default=1.0,
                   help="analyse items whose working-question logit gap exceeds the failing one by this much")
    p.add_argument("--heads", default="", help='extra head set to test jointly, "layer.head,..."')
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--assistant-prefix", default="", help='forced start of the reply, e.g. gpt-oss "<|channel|>final<|message|>"')
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.instrumentation.hf_scoring import load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, _replace_hidden, answer_token, decoder_layers

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    layers = decoder_layers(model)
    L = len(layers)
    res_layers = sorted(set(range(0, L, args.layer_stride)) | {L - 1})
    attn = attn_modules(layers)
    head_layers = [li for li in sorted(attn) if li >= args.head_from * (L - 1)]
    lead = not args.chat_template
    cons = [c.strip() for c in args.constructions.split(",")]
    triples = make_pairs(args.seed, args.base_question, args.source_question, args.instruction, cons, args.n_per)
    fams = sorted({b.family_id for b, _, _ in triples})

    def ids_of(it):
        return torch.tensor([_encode_one(tok, it.prompt, args.chat_template)[0]], device=device)

    # ---- capture helpers: residual and per-head outputs at the last position ----
    def run(it, res_patch=None, head_patch=None, capture=False):
        """res_patch: (layer, vector); head_patch: {layer: {head: vector}}. Returns (ld, caps)."""
        caps = {"res": {}, "head": {}}
        hs = []
        if capture:
            for li in res_layers:
                hs.append(layers[li].register_forward_hook(
                    lambda m, i, o, li=li: caps["res"].__setitem__(li, _hidden(o)[0, -1].detach().clone())))
            for li in head_layers:
                hs.append(attn[li][0].register_forward_pre_hook(
                    lambda m, a, li=li: caps["head"].__setitem__(li, a[0][0, -1].detach().clone())))
        if res_patch is not None:
            li, vec = res_patch

            def rhook(_m, _i, out, vec=vec):
                h = _hidden(out).clone()
                h[0, -1] = vec.to(h.dtype)
                return _replace_hidden(out, h)
            hs.append(layers[li].register_forward_hook(rhook))
        for li, heads in (head_patch or {}).items():
            o, nh, hd = attn[li]

            def phook(_m, a, heads=heads, hd=hd):
                x = a[0].clone()
                for h, v in heads.items():
                    x[0, -1, h * hd:(h + 1) * hd] = v.to(x.dtype)
                return (x,) + tuple(a[1:])
            hs.append(o.register_forward_pre_hook(phook))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids_of(it)).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        a_t, p_t = answer_token(tok, it.agent, lead), answer_token(tok, it.patient, lead)
        return float(lg[a_t] - lg[p_t]), caps

    t0 = time.time()
    recs = []
    for n, (b, s, d) in enumerate(triples):
        ld_b, cb = run(b, capture=True)
        ld_s, cs = run(s, capture=True)
        _, cd = run(d, capture=True)
        rec = {"item_id": b.item_id, "family_id": b.family_id, "construction": b.construction,
               "ld_base": ld_b, "ld_source": ld_s, "res": {}, "donor": {}, "reverse": {}, "heads_src": cs["head"]}
        for li in res_layers:
            rec["res"][li] = run(b, res_patch=(li, cs["res"][li]))[0]
            rec["donor"][li] = run(b, res_patch=(li, cd["res"][li]))[0]
            rec["reverse"][li] = run(s, res_patch=(li, cb["res"][li]))[0]
        recs.append(rec)
        if n and n % 20 == 0:
            print(f"  residual: {n}/{len(triples)} items, {time.time() - t0:.0f}s", flush=True)

    def mean(xs):
        xs = [x for x in xs if x == x]
        return round(sum(xs) / len(xs), 4) if xs else None

    residual = {}
    for group in ["all"] + cons + ["base_wrong"]:
        sub = [r for r in recs if group == "all" or r["construction"] == group or (group == "base_wrong" and r["ld_base"] < 0)]
        sub = [r for r in sub if r["ld_source"] - r["ld_base"] > args.min_gap]  # a gap to repair
        if not sub:
            continue
        residual[group] = {
            "n": len(sub),
            "repair": [mean([repair(r["res"][li], r["ld_base"], r["ld_source"]) for r in sub]) for li in res_layers],
            "donor_repair": [mean([repair(r["donor"][li], r["ld_base"], r["ld_source"]) for r in sub]) for li in res_layers],
            "reverse_break": [mean([repair(r["reverse"][li], r["ld_source"], r["ld_base"]) for r in sub]) for li in res_layers],
            "flip_to_correct": [mean([float(r["res"][li] > 0) for r in sub if r["ld_base"] < 0]) for li in res_layers],
        }

    # ---- head scan on ranking items (folds 0-2, largest gaps) ----
    rank_pool = [r for r in recs if fams.index(r["family_id"]) % 5 < 3 and r["ld_source"] - r["ld_base"] > args.min_gap]
    rank_pool.sort(key=lambda r: r["ld_source"] - r["ld_base"], reverse=True)
    rank = rank_pool[: args.n_rank]
    eval_set = [r for r in recs if fams.index(r["family_id"]) % 5 >= 3 and r["ld_source"] - r["ld_base"] > args.min_gap]
    item_of = {b.item_id: b for b, _, _ in triples}
    head_scores = {}
    t1 = time.time()
    for li in head_layers:
        _, nh, hd = attn[li]
        for h in range(nh):
            vals = []
            for r in rank:
                vec = r["heads_src"][li][h * hd:(h + 1) * hd]
                ld = run(item_of[r["item_id"]], head_patch={li: {h: vec}})[0]
                vals.append(repair(ld, r["ld_base"], r["ld_source"]))
            head_scores[f"{li}.{h}"] = mean(vals)
        print(f"  heads: layer {li} done, {time.time() - t1:.0f}s", flush=True)
    ranked = sorted(head_scores, key=lambda k: -(head_scores[k] if head_scores[k] is not None else -1e9))

    def joint(head_keys, items):
        out = []
        for r in items:
            patch = defaultdict(dict)
            for k in head_keys:
                li, h = (int(x) for x in k.split("."))
                hd = attn[li][2]
                patch[li][h] = r["heads_src"][li][h * hd:(h + 1) * hd]
            ld = run(item_of[r["item_id"]], head_patch=dict(patch))[0]
            out.append((repair(ld, r["ld_base"], r["ld_source"]), float(ld > 0) if r["ld_base"] < 0 else float("nan")))
        return {"repair": mean([x for x, _ in out]), "flip_to_correct": mean([f for _, f in out]), "n": len(out)}

    rng = random.Random(args.seed)
    all_heads = [f"{li}.{h}" for li in head_layers for h in range(attn[li][1])]
    circuit = {}
    for k in [int(x) for x in args.top_k.split(",")]:
        k = min(k, len(ranked))
        circuit[str(k)] = {"top": joint(ranked[:k], eval_set),
                           "random": joint(rng.sample(all_heads, k), eval_set)}
    named = None
    if args.heads:
        keys = [h.strip() for h in args.heads.split(",") if h.strip()]
        keys = [k for k in keys if int(k.split(".")[0]) in attn]
        named = {"heads": keys, **joint(keys, eval_set)} if keys else None

    report = {
        "experiment": "OTR-09", "status": "EXPLORATORY", "version": OTR09_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "instruction": args.instruction,
        "base_question": args.base_question, "source_question": args.source_question, "n_layers": L,
        "residual_layers": res_layers, "head_layers": head_layers, "n_items": len(recs),
        "baseline": {c: {"n": len(sub), "base_agent_preferred": mean([float(r["ld_base"] > 0) for r in sub]),
                         "source_agent_preferred": mean([float(r["ld_source"] > 0) for r in sub]),
                         "mean_ld_base": mean([r["ld_base"] for r in sub]),
                         "mean_ld_source": mean([r["ld_source"] for r in sub])}
                     for c in cons for sub in [[r for r in recs if r["construction"] == c]]},
        "residual": residual,
        "head_scan": {"n_rank_items": len(rank), "scores": head_scores, "top20": ranked[:20]},
        "circuit_eval": {"n_eval_items": len(eval_set), "by_k": circuit, "named_heads": named},
        "seconds": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr09_{args.model.replace('/', '__')}_dev_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    print("baseline:", {c: (v["base_agent_preferred"], v["source_agent_preferred"]) for c, v in report["baseline"].items()})
    if "all" in residual:
        r = residual["all"]
        i = next((j for j, v in enumerate(r["repair"]) if v is not None and v >= 0.5), None)
        print(f"residual repair (all, n={r['n']}): first layer >= 0.5: {res_layers[i] if i is not None else None}; "
              f"donor max {max(v for v in r['donor_repair'] if v is not None):.2f}")
    print("top heads:", [(k, head_scores[k]) for k in ranked[:8]])
    print("circuit (held-out):", {k: (v["top"]["repair"], v["random"]["repair"]) for k, v in circuit.items()})
    if named:
        print("named heads:", named)
    try:
        _plot(report, out.with_suffix(".png"))
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.3})
    L = report["n_layers"]
    xs = [li / (L - 1) for li in report["residual_layers"]]
    r = report["residual"].get("all")
    ax = axes[0]
    if r:
        for key, col, lab in (("repair", "#2a78d6", "working-question state, same sentence"),
                              ("donor_repair", "#898781", "working-question state, other sentence"),
                              ("reverse_break", "#e34948", "reverse: failing state breaks working run")):
            ax.plot(xs, [v if v is not None else float("nan") for v in r[key]], color=col, lw=2, label=lab)
    ax.axhline(0, color="#c4c3be", lw=0.8)
    ax.axhline(1, color="#c4c3be", lw=0.8, ls=":")
    ax.set_xlabel("relative depth of the patch (answer position)")
    ax.set_ylabel("fraction of the gap closed")
    ax.set_title("1. residual patching at the answer", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    top = report["head_scan"]["top20"][:12]
    ax.barh(range(len(top))[::-1], [report["head_scan"]["scores"][k] for k in top], color="#2a78d6", height=0.6)
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels([f"L{k.split('.')[0]}H{k.split('.')[1]}" for k in top], fontsize=7)
    ax.set_xlabel("repair when patched alone (ranking items)")
    ax.set_title("2. single heads", fontsize=9, loc="left")
    ax = axes[2]
    ks = sorted(report["circuit_eval"]["by_k"], key=int)
    for key, col in (("top", "#2a78d6"), ("random", "#898781")):
        ax.plot([int(k) for k in ks], [report["circuit_eval"]["by_k"][k][key]["repair"] for k in ks],
                color=col, lw=2, marker="o", ms=5, label=f"{key} heads")
    ax.set_xscale("log")
    ax.set_xlabel("number of heads patched jointly")
    ax.set_ylabel("fraction of the gap closed (held-out items)")
    ax.set_title("3. top-k heads vs random, held-out families", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"OTR-09 {report['model']}: repairing \"{report['base_question']}\" from \"{report['source_question']}\"",
                 fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
