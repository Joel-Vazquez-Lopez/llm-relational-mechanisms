#!/usr/bin/env python
"""OTR-16: does the "by" signal reach the answer through the handover heads? (mediation)

OTR-13/14 showed that the state of "by" carries the instruction (C6, C8) and OTR-09 that a small
set of heads at the answer position repairs the answer (C5). Both are causal, but separately. OTR-16
asks whether they are one chain: "by" -> handover heads -> answer.

For every item (same sentence, failing vs working question), with the model's frozen "by" layer Lb
and its frozen top-k heads H (FREEZE.md / FREEZE_70B.md):

  F   failing run                          ld_F, head outputs h_F[H] at the answer position
  W   working run                          ld_W, residual of "by" at Lb
  B   failing run, "by" at Lb from W       ld_B (the "by" repair), head outputs h_B[H]
  M   failing run, heads H set to h_B[H]   ld_M  (no "by" patch: only the heads see the change)
  K   run B with heads H held at h_F[H]    ld_K  ("by" patched, heads blocked)
  R   as M with k random heads from the same layers (several draws)

  by_repair     = (ld_B - ld_F) / (ld_W - ld_F)
  mediated      = (ld_M - ld_F) / (ld_B - ld_F)   share of the "by" effect carried by the heads
  blocked       = (ld_B - ld_K) / (ld_B - ld_F)   share lost when the heads cannot see it
  mediated_rand = as mediated, random heads

Only items whose "by" patch moves the answer (ld_B - ld_F > --min-by-effect) enter the shares.
ld = logit(agent) - logit(patient) at the answer position (first answer token), as in OTR-09/13.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import otr09_answer_patching as o9  # noqa: E402
import otr13_question_words as o13  # noqa: E402

from relational_lab.experiments.splits import SPLITS  # noqa: E402
from relational_lab.experiments.splits import status as split_status  # noqa: E402

OTR16_VERSION = "1.0.0"


def share(num, den):
    return num / den if abs(den) > 1e-6 else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--by-layer", type=int, required=True, help='frozen "by" layer of this model')
    p.add_argument("--rank-from", type=Path, required=True, help="dev OTR-09 file; its head_scan.top20 are the heads")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--base-question", default="Who was someone {past} by?")
    p.add_argument("--source-question", default="Who was the {ing} done by?")
    p.add_argument("--instruction", default="Reply with one name only.")
    p.add_argument("--constructions", default="active,object_cleft,passive,by_cleft")
    p.add_argument("--n-per", type=int, default=32)
    p.add_argument("--min-gap", type=float, default=1.0, help="working - failing margin needed (as OTR-09/13)")
    p.add_argument("--min-by-effect", type=float, default=0.5, help="ld_B - ld_F needed to enter the shares")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default=None,
                   help="'auto': split the model over all visible GPUs (70B-class models on 2 x H100)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split", default="dev", choices=SPLITS, help="test0..test3 only after a freeze")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    from relational_lab.instrumentation import chat as _chat
    _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                    assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.experiments import order_role_stimuli as ors
    from relational_lab.instrumentation.hf_scoring import load_for_run, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, _replace_hidden, answer_token, decoder_layers

    prior = json.loads(args.rank_from.read_text())
    heads = [tuple(int(x) for x in k.split(".")) for k in prior["head_scan"]["top20"][:args.k]]

    tok = load_tokenizer(args.model)
    model, device = load_for_run(args.model, dtype=getattr(torch, args.dtype), device_map=args.device_map)
    layers = decoder_layers(model)
    L = len(layers)
    if not 0 <= args.by_layer < L:
        raise SystemExit(f"--by-layer must lie in 0..{L - 1}")
    attn = o9.attn_modules(layers)
    missing = [f"{li}.{h}" for li, h in heads if li not in attn or h >= attn[li][1]]
    if missing:
        raise SystemExit(f"heads not found in this model: {missing}")
    head_layers = sorted({li for li, _ in heads})
    pool = [(li, h) for li in head_layers for h in range(attn[li][1]) if (li, h) not in set(heads)]
    rng = random.Random(args.seed)
    rand_sets = [rng.sample(pool, len(heads)) for _ in range(args.n_random)]
    cap_layers = sorted(head_layers)
    lead = not args.chat_template
    cons = [c.strip() for c in args.constructions.split(",")]
    triples = o9.make_pairs(args.seed, args.base_question, args.source_question, args.instruction, cons, args.n_per,
                            split=args.split)
    triples = [t for t in triples if answer_token(tok, t[0].answer, lead) != answer_token(tok, t[0].distractor, lead)]

    def enc(it):
        ids, offs = _encode_one(tok, it.prompt, args.chat_template)
        return torch.tensor([ids], device=device), offs

    def forward(it, ids, by=None, by_capture=None, head_set=None, capture_heads=False):
        """by: (pos, vec) written at the "by" layer; by_capture: pos to read at the "by" layer;
        head_set: {layer: {head: vec}} at the answer position. Returns (ld, by_vec, {layer: o_proj input})."""
        hs, out = [], {"by": None, "heads": {}}
        if by is not None or by_capture is not None:
            def bh(_m, _i, o):
                h = _hidden(o)
                if by_capture is not None:
                    out["by"] = h[0, by_capture].detach().clone()
                if by is None:
                    return None
                h = h.clone()
                h[0, by[0]] = by[1].to(h.device, h.dtype)
                return _replace_hidden(o, h)
            hs.append(layers[args.by_layer].register_forward_hook(bh))
        if capture_heads:
            for li in cap_layers:
                hs.append(attn[li][0].register_forward_pre_hook(
                    lambda m, a, li=li: out["heads"].__setitem__(li, a[0][0, -1].detach().clone())))
        for li, hv in (head_set or {}).items():
            o, _nh, hd = attn[li]

            def phook(_m, a, hv=hv, hd=hd):
                x = a[0].clone()
                for h, v in hv.items():
                    x[0, -1, h * hd:(h + 1) * hd] = v.to(x.device, x.dtype)
                return (x,) + tuple(a[1:])
            hs.append(o.register_forward_pre_hook(phook))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        a_t, p_t = answer_token(tok, it.answer, lead), answer_token(tok, it.distractor, lead)
        return float(lg[a_t] - lg[p_t]), out["by"], out["heads"]

    def select(caps, hset):
        sel = {}
        for li, h in hset:
            hd = attn[li][2]
            sel.setdefault(li, {})[h] = caps[li][h * hd:(h + 1) * hd]
        return sel

    t0 = time.time()
    recs = []
    for n, (b, s, _d) in enumerate(triples):
        ib, ob = enc(b)
        is_, os_ = enc(s)
        qb = o13.anchors(b, ob, "base", ib.shape[1], 1)["by"]
        qs = o13.anchors(s, os_, "source", is_.shape[1], 1)["by"]
        ld_f, _, h_f = forward(b, ib, capture_heads=True)
        ld_w, by_w, _ = forward(s, is_, by_capture=qs)
        if ld_w - ld_f <= args.min_gap:
            continue
        ld_b, _, h_b = forward(b, ib, by=(qb, by_w), capture_heads=True)
        ld_m, _, _ = forward(b, ib, head_set=select(h_b, heads))
        ld_k, _, _ = forward(b, ib, by=(qb, by_w), head_set=select(h_f, heads))
        rand_m = [forward(b, ib, head_set=select(h_b, rs))[0] for rs in rand_sets]
        recs.append(dict(item_id=b.item_id, family_id=b.family_id, construction=b.construction, ld_failing=ld_f,
                         ld_working=ld_w, ld_by=ld_b, ld_mediated=ld_m, ld_blocked=ld_k, ld_mediated_random=rand_m))
        if n and n % 10 == 0:
            print(f"  {n}/{len(triples)} items, {time.time() - t0:.0f}s", flush=True)
    if not recs:
        raise SystemExit("no analysable items")

    for r in recs:
        eff = r["ld_by"] - r["ld_failing"]
        r["by_repair"] = share(eff, r["ld_working"] - r["ld_failing"])
        r["in_shares"] = eff > args.min_by_effect
        r["mediated"] = share(r["ld_mediated"] - r["ld_failing"], eff) if r["in_shares"] else None
        r["blocked"] = share(r["ld_by"] - r["ld_blocked"], eff) if r["in_shares"] else None
        rs = [share(x - r["ld_failing"], eff) for x in r["ld_mediated_random"]] if r["in_shares"] else []
        r["mediated_random"] = sum(rs) / len(rs) if rs and None not in rs else None

    def mean(key, sub=None):
        xs = [r[key] for r in (sub or recs) if r.get(key) is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {g: {k: mean(k, [r for r in recs if g == "all" or r["construction"] == g])
                   for k in ("by_repair", "mediated", "blocked", "mediated_random")} for g in ["all", *cons]}
    report = {
        "status": split_status(args.split), "split": args.split, "experiment": "OTR-16 by -> heads -> answer mediation",
        "otr16_version": OTR16_VERSION, "generator_version": ors.GENERATOR_VERSION, "model": args.model,
        "chat_template": args.chat_template, "base_question": args.base_question,
        "source_question": args.source_question, "instruction": args.instruction, "n_layers": L,
        "by_layer": args.by_layer, "heads": [f"{li}.{h}" for li, h in heads], "rank_from": str(args.rank_from.name),
        "random_head_sets": [[f"{li}.{h}" for li, h in rs] for rs in rand_sets],
        "min_gap": args.min_gap, "min_by_effect": args.min_by_effect, "n_items": len(recs),
        "n_in_shares": sum(r["in_shares"] for r in recs), "summary": summary,
        "seconds": round(time.time() - t0), "items": recs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr16_{args.model.replace('/', '__')}_{args.split}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    s = summary["all"]
    print(f"by repair {s['by_repair']} | mediated by heads {s['mediated']} (random {s['mediated_random']}) | "
          f"blocked {s['blocked']} | {report['n_in_shares']}/{len(recs)} items in shares")


if __name__ == "__main__":
    main()
