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

Broad English (1.1.0): --stimuli bx-transitive | bx-ditransitive | bx-psych runs the same
analysis on new names, verbs and constructions (failing question for the by-phrase role,
defaults "Who was someone {pp} by?" / "Who was someone {pp} {theme} by?"). --rank-from
OTR09.json takes the head ranking from an earlier run (e.g. on the original materials) and
tests whether THOSE heads repair the new items ("transfer"), next to the fresh scan.

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

from relational_lab.experiments import broad_stimuli as bx  # noqa: E402
from relational_lab.experiments import bx_adapter as bxa  # noqa: E402
from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402
from relational_lab.experiments import xl_adapter as xla  # noqa: E402
from relational_lab.experiments import ud_stimuli as uds  # noqa: E402
from relational_lab.experiments import wiki_roles as wr  # noqa: E402
from relational_lab.experiments.splits import SPLITS, in_split  # noqa: E402
from relational_lab.experiments.splits import status as split_status  # noqa: E402
from relational_lab.experiments.question_templates import fill as fill_template  # noqa: E402

OTR09_VERSION = "1.1.0"


def make_pairs(seed: int, base_q: str, source_q: str, instruction: str, constructions, n_per: int,
               stimuli: str = "otr", wiki_file=None, wiki_names: str = "real", split: str = "dev"):
    """(base item, source item, donor item) triples: same sentence for base/source; donor =
    another dev sentence of the same construction with a disjoint name pair."""
    if split != "dev" and stimuli in ("ud-en", "wiki-en"):
        raise ValueError("UD and WIKI-ROLES have no test split (they are validation sets)")
    if stimuli in xla.STIMULI:
        return _with_donors(xla.repair_pairs(stimuli, seed, constructions, instruction, split), constructions, n_per,
                            seed)
    if stimuli == "wiki-en":
        recs = wr.load_records(wiki_file)
        return _with_donors(wr.generate_pairs(recs, base_q, source_q, instruction, wiki_names, seed),
                            constructions, n_per, seed)
    if stimuli == "ud-en":
        return _with_donors(uds.generate_pairs(base_q, source_q, instruction, seed=seed), constructions, n_per, seed)
    if stimuli != "otr":
        items = bxa.repair_items(stimuli, seed, constructions, split=split)

        def ask(it, template):
            return bxa.ask(it, template, instruction)
        return _triples(items, constructions, n_per, seed, ask, base_q, source_q)
    items = [i for i in ors.generate_items(seed=seed)
             if in_split(i, split) and i.construction in constructions and i.role_asked == "agent"
             and i.question_voice == "active"]

    def ask(it, template):
        q = fill_template(template, it.verb, it.lemma)
        prompt = it.prompt.replace(f"\nQuestion: {it.question}\n", f"\nQuestion: {q}\n")
        assert prompt.count(q) == 1, prompt
        if instruction:
            prompt = prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:")
        return dataclasses.replace(it, question=q, prompt=prompt)

    return _triples(items, constructions, n_per, seed, ask, base_q, source_q)


def _xl_triples(lang, seed, constructions, n_per, instruction):
    """Spanish/Galician: base = the failing passive agent question, source = the active agent
    question on the same sentence; donor = another sentence of the same construction."""
    return _with_donors(xla.repair_pairs(lang, seed, constructions, instruction), constructions, n_per, seed)


def _with_donors(pairs, constructions, n_per, seed):
    rng = random.Random(seed)
    out = []
    for c in constructions:
        pool = [bs for bs in pairs if bs[0].construction == c]
        rng.shuffle(pool)
        for b, s in pool[:n_per]:
            donors = [ds for d, ds in pool if not {d.agent, d.patient} & {b.agent, b.patient}]
            out.append((b, s, rng.choice(donors)))
    return out


def _triples(items, constructions, n_per, seed, ask, base_q, source_q):
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
    p.add_argument("--stimuli", default="otr", choices=["otr", *bxa.STIMULI, *xla.STIMULI, "ud-en", "wiki-en"])
    p.add_argument("--wiki-file", type=Path, default=None, help="WIKI-ROLES jsonl (scripts/wiki_extract_roles.py)")
    p.add_argument("--wiki-names", default="real", choices=["real", "neutral"],
                   help="real Wikipedia names, or two neutral OTR names swapped in (memory control)")
    p.add_argument("--base-question", default=None, help='default "Who was someone {past} by?" (bx: {pp})')
    p.add_argument("--source-question", default=None, help='default "Who was the {ing} done by?"')
    p.add_argument("--split", default="dev", choices=SPLITS, help="test0..test3 only after FREEZE.md")
    p.add_argument("--skip-scan", action="store_true",
                   help="no head scan: the --rank-from heads are the circuit (frozen heads on a test set)")
    p.add_argument("--n-random", type=int, default=1,
                   help="independent random-head draws per k (per-item mean over draws); 1 = original")
    p.add_argument("--controls", action="store_true",
                   help="head controls: patch from another item's failing run; random nudge of the same norm")
    p.add_argument("--controls-extra", action="store_true",
                   help="exploratory controls: the mean head output over other families' failing runs "
                        "(removes item content), and another item's WORKING run")
    p.add_argument("--rank-from", type=Path, default=None,
                   help="OTR-09 result whose head ranking (top20) is tested on these items (transfer)")
    p.add_argument("--instruction", default="")
    p.add_argument("--constructions", default=None,
                   help="default: the four main constructions (bx-psych: the six of both verb classes)")
    p.add_argument("--n-per", type=int, default=32, help="sentences per construction")
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--layers", default=None, metavar="L1,L2,...",
                   help="explicit layer list (overrides --layer-stride); e.g. only the frozen layers on test runs")
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
    p.add_argument("--device-map", default=None,
                   help="'auto': split the model over all visible GPUs (70B-class models on 2 x H100)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    if args.stimuli in xla.STIMULI:
        # questions come from the XL generator (passive vs active agent question)
        args.base_question = "XL default passive agent question"
        args.source_question = "XL default active agent question"
        args.constructions = args.constructions or ",".join(xla.REPAIR_CONSTRUCTIONS[args.stimuli])
    elif args.stimuli == "wiki-en":
        if not args.wiki_file:
            raise SystemExit("--stimuli wiki-en needs --wiki-file")
        args.base_question = args.base_question or "Who was someone {pp} by?"
        args.constructions = args.constructions or ",".join(wr.CONSTRUCTIONS)
    elif args.stimuli == "ud-en":
        args.base_question = args.base_question or "Who was someone {pp} by?"
        args.constructions = args.constructions or ",".join(uds.CONSTRUCTIONS)
    elif args.stimuli == "otr":
        args.base_question = args.base_question or "Who was someone {past} by?"
        args.constructions = args.constructions or "active,object_cleft,passive,by_cleft"
    else:
        block = bxa.block_name(args.stimuli)
        args.base_question = args.base_question or bxa.DEFAULT_BASE_QUESTION[block]
        args.constructions = args.constructions or ",".join(bxa.BASE_CONSTRUCTIONS[block])
    args.source_question = args.source_question or bxa.DEFAULT_SOURCE_QUESTION
    transfer_heads = []
    if args.rank_from:
        transfer_heads = json.loads(args.rank_from.read_text())["head_scan"]["top20"]

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                                  assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.instrumentation.hf_scoring import load_for_run, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, _replace_hidden, answer_token, decoder_layers

    tok = load_tokenizer(args.model)
    model, device = load_for_run(args.model, dtype=getattr(torch, args.dtype), device_map=args.device_map)
    layers = decoder_layers(model)
    L = len(layers)
    if args.layers:
        res_layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        if not res_layers or res_layers[0] < 0 or res_layers[-1] >= L:
            raise SystemExit(f"--layers must lie in 0..{L - 1}")
    else:
        res_layers = sorted(set(range(0, L, args.layer_stride)) | {L - 1})
    attn = attn_modules(layers)
    head_layers = [li for li in sorted(attn) if li >= args.head_from * (L - 1)]
    scan_layers = list(head_layers)
    n_frozen = len(transfer_heads)
    transfer_heads = [k for k in transfer_heads if int(k.split(".")[0]) in attn]
    if args.split != "dev" and len(transfer_heads) != n_frozen:
        raise SystemExit(f"frozen heads dropped ({n_frozen} -> {len(transfer_heads)}); refusing a test run")
    head_layers = sorted(set(head_layers) | {int(k.split(".")[0]) for k in transfer_heads})
    lead = not args.chat_template
    cons = [c.strip() for c in args.constructions.split(",")]
    triples = make_pairs(args.seed, args.base_question, args.source_question, args.instruction, cons, args.n_per,
                         stimuli=args.stimuli, wiki_file=args.wiki_file, wiki_names=args.wiki_names,
                         split=args.split)
    # a bare answer is scored by its first token: skip items whose two names share it
    n_before = len(triples)
    triples = [t for t in triples
               if answer_token(tok, t[0].answer, lead) != answer_token(tok, t[0].distractor, lead)]
    n_shared_first_token = n_before - len(triples)
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
                h[0, -1] = vec.to(h.device, h.dtype)
                return _replace_hidden(out, h)
            hs.append(layers[li].register_forward_hook(rhook))
        for li, heads in (head_patch or {}).items():
            o, nh, hd = attn[li]

            def phook(_m, a, heads=heads, hd=hd):
                x = a[0].clone()
                for h, v in heads.items():
                    x[0, -1, h * hd:(h + 1) * hd] = v.to(x.device, x.dtype)
                return (x,) + tuple(a[1:])
            hs.append(o.register_forward_pre_hook(phook))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids_of(it)).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        # answer / distractor: the agent / patient for OTR agent questions; for bx, the by-phrase role
        a_t, p_t = answer_token(tok, it.answer, lead), answer_token(tok, it.distractor, lead)
        return float(lg[a_t] - lg[p_t]), caps

    t0 = time.time()
    recs = []
    for n, (b, s, d) in enumerate(triples):
        ld_b, cb = run(b, capture=True)
        ld_s, cs = run(s, capture=True)
        _, cd = run(d, capture=True)
        rec = {"item_id": b.item_id, "family_id": b.family_id, "construction": b.construction,
               "ld_base": ld_b, "ld_source": ld_s, "res": {}, "donor": {}, "reverse": {}, "heads_src": cs["head"],
               "heads_base": cb["head"]}
        for li in res_layers:
            rec["res"][li] = run(b, res_patch=(li, cs["res"][li]))[0]
            rec["donor"][li] = run(b, res_patch=(li, cd["res"][li]))[0]
            rec["reverse"][li] = run(s, res_patch=(li, cb["res"][li]))[0]
        recs.append(rec)
        if n and n % 20 == 0:
            print(f"  residual: {n}/{len(triples)} items, {time.time() - t0:.0f}s", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
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
    if args.skip_scan and not transfer_heads:
        raise SystemExit("--skip-scan needs --rank-from")
    for li in ([] if args.skip_scan else scan_layers):
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
    if args.skip_scan:
        ranked = list(transfer_heads)  # frozen heads: the circuit is fixed in advance
        eval_set = [r for r in recs if r["ld_source"] - r["ld_base"] > args.min_gap]  # no ranking fold needed

    ctrl_rng = random.Random(args.seed + 1)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    mean_heads, mean_heads_fam = {}, defaultdict(dict)
    if getattr(args, "controls_extra", False):
        # sums of the failing-run head outputs per layer, overall and per family (for leave-family-out means)
        for r in recs:
            for li, v in r["heads_base"].items():
                v = v.float()
                t, n = mean_heads.get(li, (torch.zeros_like(v), 0))
                mean_heads[li] = (t + v, n + 1)
                ft, fn = mean_heads_fam[r["family_id"]].get(li, (torch.zeros_like(v), 0))
                mean_heads_fam[r["family_id"]][li] = (ft + v, fn + 1)

    def joint(head_keys, items, mode="source"):
        """mode: source = the working run's head outputs (the repair); other_failing = head outputs
        from ANOTHER item's failing run (no transferable state should repair); nudge = the failing
        head output plus a random vector of the same norm as (working - failing)."""
        out = []
        for r in items:
            patch = defaultdict(dict)
            other = None
            if mode in ("other_failing", "other_working"):
                pool = [x for x in recs if x["family_id"] != r["family_id"]]
                other = ctrl_rng.choice(pool)
            for k in head_keys:
                li, h = (int(x) for x in k.split("."))
                hd = attn[li][2]
                sl = slice(h * hd, (h + 1) * hd)
                if mode == "source":
                    patch[li][h] = r["heads_src"][li][sl]
                elif mode == "other_failing":
                    patch[li][h] = other["heads_base"][li][sl]
                elif mode == "other_working":
                    patch[li][h] = other["heads_src"][li][sl]
                elif mode == "mean_failing":
                    tot, n = mean_heads[li]
                    fam_tot, fam_n = mean_heads_fam[r["family_id"]][li]
                    patch[li][h] = ((tot - fam_tot) / max(n - fam_n, 1))[sl].to(r["heads_base"][li].dtype)
                else:
                    base = r["heads_base"][li][sl]
                    diff = (r["heads_src"][li][sl] - base).float()
                    noise = torch.randn(diff.shape, generator=gen).to(diff.device)
                    patch[li][h] = (base.float() + noise / noise.norm() * diff.norm()).to(base.dtype)
            ld = run(item_of[r["item_id"]], head_patch=dict(patch))[0]
            out.append((repair(ld, r["ld_base"], r["ld_source"]), float(ld > 0) if r["ld_base"] < 0 else float("nan")))
        return {"repair": mean([x for x, _ in out]), "flip_to_correct": mean([f for _, f in out]), "n": len(out),
                "heads": list(head_keys),
                # per-item values keyed to families, for paired family-bootstrap CIs (freeze C5)
                "items": [{"item_id": r["item_id"], "family_id": r["family_id"], "repair": x}
                          for r, (x, _) in zip(items, out)]}

    def joint_random(pool, k, items):
        """--n-random independent draws of k heads from ``pool``; per item, the mean over draws."""
        draws = [joint(rng.sample(pool, min(k, len(pool))), items) for _ in range(args.n_random)]
        per = [mean([d["items"][i]["repair"] for d in draws]) for i in range(len(items))]
        return {"repair": mean(per), "flip_to_correct": mean([d["flip_to_correct"] for d in draws]),
                "n": len(items), "n_draws": args.n_random, "draw_heads": [d["heads"] for d in draws],
                "items": [{"item_id": r["item_id"], "family_id": r["family_id"], "repair": x}
                          for r, x in zip(items, per)]}

    rng = random.Random(args.seed)
    pool_layers = sorted(set(scan_layers) | ({int(k.split(".")[0]) for k in ranked} if args.skip_scan else set()))
    all_heads = [f"{li}.{h}" for li in pool_layers for h in range(attn[li][1])]
    circuit = {}
    for k in [int(x) for x in args.top_k.split(",")]:
        k = min(k, len(ranked))
        circuit[str(k)] = {"top": joint(ranked[:k], eval_set),
                           "random": joint_random(all_heads, k, eval_set)}
        if args.controls:
            circuit[str(k)]["top_other_failing"] = joint(ranked[:k], eval_set, "other_failing")
            circuit[str(k)]["top_random_nudge"] = joint(ranked[:k], eval_set, "nudge")
    # transfer: heads ranked on another run (e.g. the original materials) tested on these items
    transfer = None
    if transfer_heads:
        all_gap = [r for r in recs if r["ld_source"] - r["ld_base"] > args.min_gap]
        t_layers = sorted({int(k.split(".")[0]) for k in transfer_heads})
        t_pool = [f"{li}.{h}" for li in t_layers for h in range(attn[li][1])]
        transfer = {"from": str(args.rank_from), "heads": transfer_heads,
                    "overlap_with_fresh_top20": sorted(set(transfer_heads) & set(ranked[:20])),
                    "by_k": {}}
        for k in [int(x) for x in args.top_k.split(",")]:
            k = min(k, len(transfer_heads))
            transfer["by_k"][str(k)] = {
                "eval_items": joint(transfer_heads[:k], eval_set),
                "all_items": joint(transfer_heads[:k], all_gap),
                "random_same_layers": joint_random(t_pool, k, all_gap)}
            if args.controls:
                transfer["by_k"][str(k)]["other_failing"] = joint(transfer_heads[:k], all_gap, "other_failing")
                transfer["by_k"][str(k)]["random_nudge"] = joint(transfer_heads[:k], all_gap, "nudge")
                if args.controls_extra:
                    transfer["by_k"][str(k)]["mean_failing"] = joint(transfer_heads[:k], all_gap, "mean_failing")
                    transfer["by_k"][str(k)]["other_working"] = joint(transfer_heads[:k], all_gap, "other_working")
    named = None
    if args.heads:
        keys = [h.strip() for h in args.heads.split(",") if h.strip()]
        keys = [k for k in keys if int(k.split(".")[0]) in attn]
        named = {"heads": keys, **joint(keys, eval_set)} if keys else None

    report = {
        "experiment": "OTR-09", "status": split_status(args.split), "split": args.split,
        "skip_scan": args.skip_scan, "version": OTR09_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "stimuli": args.stimuli,
        "bx_generator_version": bx.BX_GENERATOR_VERSION if args.stimuli in bxa.STIMULI else None,
        "ud_version": uds.UD_VERSION if args.stimuli == "ud-en" else None,
        "wiki": ({"version": wr.WIKI_VERSION, "file": str(args.wiki_file), "names": args.wiki_names}
                 if args.stimuli == "wiki-en" else None),
        "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "instruction": args.instruction,
        "base_question": args.base_question, "source_question": args.source_question, "n_layers": L,
        "n_skipped_shared_first_token": n_shared_first_token,
        "residual_layers": res_layers, "head_layers": head_layers, "scan_layers": scan_layers,
        "n_items": len(recs),
        "baseline": {c: {"n": len(sub), "base_agent_preferred": mean([float(r["ld_base"] > 0) for r in sub]),
                         "source_agent_preferred": mean([float(r["ld_source"] > 0) for r in sub]),
                         "mean_ld_base": mean([r["ld_base"] for r in sub]),
                         "mean_ld_source": mean([r["ld_source"] for r in sub])}
                     for c in cons for sub in [[r for r in recs if r["construction"] == c]]},
        "residual": residual,
        "head_scan": {"n_rank_items": len(rank), "scores": head_scores, "top20": ranked[:20]},
        "circuit_eval": {"n_eval_items": len(eval_set), "by_k": circuit, "named_heads": named},
        "transfer": transfer,
        "seconds": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stag = "" if args.stimuli == "otr" else f"_{args.stimuli}"
    if args.stimuli == "wiki-en":
        stag += f"-{args.wiki_names}"
    out = args.out_dir / f"otr09_{args.model.replace('/', '__')}_{args.split}{stag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    print("baseline:", {c: (v["base_agent_preferred"], v["source_agent_preferred"]) for c, v in report["baseline"].items()})
    if "all" in residual:
        r = residual["all"]
        i = next((j for j, v in enumerate(r["repair"]) if v is not None and v >= 0.5), None)
        print(f"residual repair (all, n={r['n']}): first layer >= 0.5: {res_layers[i] if i is not None else None}; "
              f"donor max {max(v for v in r['donor_repair'] if v is not None):.2f}")
    print("top heads:", [(k, head_scores.get(k)) for k in ranked[:8]])
    print("circuit (held-out):", {k: (v["top"]["repair"], v["random"]["repair"]) for k, v in circuit.items()})
    if named:
        print("named heads:", named)
    if transfer:
        print("transfer heads (all items):", {k: (v["all_items"]["repair"], v["random_same_layers"]["repair"])
                                              for k, v in transfer["by_k"].items()},
              "overlap with fresh top20:", len(transfer["overlap_with_fresh_top20"]))
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
