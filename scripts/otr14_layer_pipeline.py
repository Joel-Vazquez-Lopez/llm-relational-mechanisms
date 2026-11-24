#!/usr/bin/env python
"""OTR-14: the layer pipeline on one scale, per item, with confidence intervals.

Same items (failing question "Who was someone {past} by?" vs working question "Who was the {ing}
done by?", same sentence), same metric, every item stored. For each layer l:

  read_question_fail  block attention from the answer position to the question tokens at layer l
                      (failing run): share of the failing-working gap closed (repair)
  read_names_fail     block attention from the answer position to the two names at layer l
                      (failing run): change of the margin as a share of the gap
  read_question_work  / read_names_work: the same in the working run, as the share of the
                      working margin lost (damage) -> WHEN the answer reads the question / names
  swap_question       residual of all question tokens from the working run into the failing run
                      at layer l (repair; the OTR-13 "joint" curve, now per item)
  swap_answer         residual of the answer position from the working run (repair; OTR-09)
  by_from_failing     "by" state of the failing question into the working question (damage)
  by_from_patient     "by" state of the PATIENT question "Who was {past} by someone?" into the
                      working question (damage): if "by" after "someone V-ed" encodes a patient
                      question, this should damage the working answer like the failing "by" does
                      (test of account b').
  heads_attention     (clean runs) attention of the --rank-from top heads from the answer position
                      to the agent vs the patient name, failing vs working (eager attention; skipped
                      if the model cannot return attention weights).

Attention blocking acts on the attention mask of ONE layer (pre-hook on self_attn); layers without
standard attention (linear attention / Mamba in hybrid models) are left out. A self-check on the
first item verifies that blocking nothing reproduces the clean logits; if it fails for an
architecture, the blocking curves are skipped and reported as such.

Summaries: mean curve with 95% bootstrap CIs over name-pair families, and bootstrap CIs of the
depth at which each curve first reaches half of its maximum (onset) and of its peak.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import otr09_answer_patching as o9  # noqa: E402

OTR14_VERSION = "1.0.0"
PATIENT_Q = "Who was {past} by someone?"


def char_tokens(offsets, s: int, e: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if a < e and b > s and b > a]


def by_token(item, offsets) -> int:
    q0 = item.prompt.index("\nQuestion: ") + len("\nQuestion: ")
    m = list(re.finditer(r"\bby\b", item.question))
    if len(m) != 1:
        raise ValueError(f"need exactly one 'by' in {item.question!r}")
    toks = char_tokens(offsets, q0 + m[0].start(), q0 + m[0].end())
    return toks[-1]


def question_tokens(item, offsets) -> list[int]:
    q0 = item.prompt.index("\nQuestion: ") + len("\nQuestion: ")
    return char_tokens(offsets, q0, q0 + len(item.question))


def name_tokens(item, offsets) -> dict[str, list[int]]:
    return {("agent" if t.is_agent else "patient"): char_tokens(offsets, t.prompt_char_start, t.prompt_char_end)
            for t in item.names}


def boot_summary(curves: dict[str, list[float]], fam_of: dict[str, str], lays, L, n_boot=1000, seed=0):
    """mean, 95% CI per layer; onset (first layer >= half of the max of the mean curve) and peak
    depth with bootstrap CIs over families. ``curves``: item_id -> list over layers."""
    import numpy as np
    ids = sorted(curves)
    fams = sorted({fam_of[i] for i in ids})
    by_fam = {f: [i for i in ids if fam_of[i] == f] for f in fams}
    M = np.array([curves[i] for i in ids], dtype=float)
    mean = np.nanmean(M, 0)
    depth = np.array(lays) / (L - 1)

    def onset_peak(m):
        mx = np.nanmax(m)
        if not np.isfinite(mx) or mx <= 0:
            return float("nan"), float("nan")
        k = int(np.argmax(m >= mx / 2))
        return float(depth[k]), float(depth[int(np.nanargmax(m))])

    rng = np.random.default_rng(seed)
    boots, ons, pks = [], [], []
    idx = {i: k for k, i in enumerate(ids)}
    for _ in range(n_boot):
        pick = rng.choice(len(fams), len(fams))
        rows = [idx[i] for f in pick for i in by_fam[fams[f]]]
        m = np.nanmean(M[rows], 0)
        boots.append(m)
        o, p = onset_peak(m)
        ons.append(o)
        pks.append(p)
    B = np.array(boots)
    o, p = onset_peak(mean)

    def ci(x):
        x = np.array(x)[np.isfinite(x)]
        return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)] if len(x) else None

    return {"mean": [round(float(x), 4) for x in mean],
            "ci_low": [round(float(x), 4) for x in np.nanpercentile(B, 2.5, 0)],
            "ci_high": [round(float(x), 4) for x in np.nanpercentile(B, 97.5, 0)],
            "onset_depth": round(o, 3), "onset_ci": ci(ons), "peak_depth": round(p, 3), "peak_ci": ci(pks),
            "n_items": len(ids), "n_families": len(fams)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--stimuli", default="otr", choices=["otr", "bx-transitive"])
    p.add_argument("--base-question", default=None)
    p.add_argument("--source-question", default="Who was the {ing} done by?")
    p.add_argument("--patient-question", default=None, help='default "Who was {past} by someone?" (bx: {pp})')
    p.add_argument("--instruction", default="Reply with one name only.")
    p.add_argument("--constructions", default=None)
    p.add_argument("--n-per", type=int, default=16)
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--min-gap", type=float, default=1.0)
    p.add_argument("--rank-from", type=Path, default=None, help="OTR-09 result: its top heads' attention is recorded")
    p.add_argument("--n-heads-attn", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    if args.stimuli == "otr":
        args.base_question = args.base_question or "Who was someone {past} by?"
        args.patient_question = args.patient_question or PATIENT_Q
        args.constructions = args.constructions or "active,object_cleft,passive,by_cleft"
    else:
        args.base_question = args.base_question or "Who was someone {pp} by?"
        args.patient_question = args.patient_question or "Who was {pp} by someone?"
        args.constructions = args.constructions or ",".join(o9.bxa.BASE_CONSTRUCTIONS["transitive"])

    from relational_lab.instrumentation import chat as _chat
    _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg,
                    assistant_prefix=args.assistant_prefix)
    import torch

    from relational_lab.experiments import order_role_stimuli as ors
    from relational_lab.instrumentation.hf_scoring import load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, _replace_hidden, answer_token, decoder_layers

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    layers = decoder_layers(model)
    L = len(layers)
    lays = sorted(set(range(0, L, args.layer_stride)) | {L - 1})
    attn_layers = {li for li in lays if getattr(getattr(layers[li], "self_attn", None), "forward", None) is not None}
    lead = not args.chat_template
    cons = [c.strip() for c in args.constructions.split(",")]
    triples = o9.make_pairs(args.seed, args.base_question, args.source_question, args.instruction, cons, args.n_per,
                            stimuli=args.stimuli)
    triples = [t for t in triples if answer_token(tok, t[0].answer, lead) != answer_token(tok, t[0].distractor, lead)]

    def patient_version(b):
        """same sentence and instruction, the patient question (answer stays the agent for the margin)."""
        if args.stimuli == "otr":
            from relational_lab.experiments.question_templates import fill as fill_template
            q = fill_template(args.patient_question, b.verb, b.lemma)
        else:
            q = o9.bx.fill_template(args.patient_question, b.lemma, o9.bx.theme_of(b))
        import dataclasses
        return dataclasses.replace(b, question=q, prompt=b.prompt.replace(f"\nQuestion: {b.question}\n",
                                                                         f"\nQuestion: {q}\n"))

    def enc(it):
        ids, offs = _encode_one(tok, it.prompt, args.chat_template)
        return torch.tensor([ids], device=device), offs

    def margin(logits, it):
        a, b = answer_token(tok, it.answer, lead), answer_token(tok, it.distractor, lead)
        return float(logits[a] - logits[b])

    def run(it, ids, patches=None, block=None, capture=None, attentions=False):
        """patches {layer: {pos: vec}}; block (layer, [key positions]) masks query=last -> keys at
        that layer; capture [positions] -> {layer: {pos: vec}}."""
        hs, caps = [], {}
        if capture is not None:
            for li in lays:
                def ch(_m, _i, out, li=li):
                    h = _hidden(out)[0]
                    caps[li] = {q: h[q].detach().clone() for q in capture}
                hs.append(layers[li].register_forward_hook(ch))
        for li, pv in (patches or {}).items():
            def ph(_m, _i, out, pv=pv):
                h = _hidden(out).clone()
                for q, v in pv.items():
                    h[0, q] = v.to(h.dtype)
                return _replace_hidden(out, h)
            hs.append(layers[li].register_forward_hook(ph))
        if block is not None:
            li, keys = block
            T = ids.shape[1]

            def pre(_m, a, kw):
                dt = next(model.parameters()).dtype
                neg = torch.finfo(dt).min
                mask = torch.full((T, T), neg, dtype=dt, device=ids.device).triu(1)
                for k in keys:
                    mask[T - 1, k] = neg
                kw["attention_mask"] = mask[None, None]
                return a, kw
            hs.append(layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        orig_impl = getattr(model.config, "_attn_implementation", None)
        try:
            if attentions and orig_impl not in (None, "eager"):
                model.set_attn_implementation("eager")  # sdpa returns no attention weights
            with torch.no_grad():
                out = model(input_ids=ids, output_attentions=attentions)
        finally:
            if attentions and orig_impl not in (None, "eager"):
                model.set_attn_implementation(orig_impl)
            for h in hs:
                h.remove()
        return out, caps

    # ---- self-check: blocking nothing must reproduce the clean logits ----
    b0 = triples[0][0]
    ib0, _ = enc(b0)
    clean = run(b0, ib0)[0].logits[0, -1].float()
    ok_layers = set()
    for li in sorted(attn_layers):
        try:
            noop = run(b0, ib0, block=(li, []))[0].logits[0, -1].float()
            if torch.allclose(noop, clean, atol=1e-2, rtol=1e-3):
                ok_layers.add(li)
        except Exception as exc:  # architecture does not take a 4D mask here
            print(f"layer {li}: attention blocking unavailable ({type(exc).__name__}: {exc})", flush=True)
    print(f"attention blocking verified on {len(ok_layers)}/{len(lays)} sampled layers", flush=True)

    head_list = []
    if args.rank_from:
        head_list = json.loads(args.rank_from.read_text())["head_scan"]["top20"][: args.n_heads_attn]

    t0 = time.time()
    per_item, fam_of, meta = {}, {}, []
    attn_ok = None
    curves_names = ["read_question_fail", "read_names_fail", "read_question_work", "read_names_work",
                    "swap_question", "swap_answer", "by_from_failing", "by_from_patient"]
    for b, s, _d in triples:
        pq = patient_version(b)
        ib, ob = enc(b)
        is_, os_ = enc(s)
        ip, op = enc(pq)
        qb, qs = question_tokens(b, ob), question_tokens(s, os_)
        nb, ns = name_tokens(b, ob), name_tokens(s, os_)
        byb, bys, byp = by_token(b, ob), by_token(s, os_), by_token(pq, op)
        want_att = bool(head_list) and attn_ok is not False
        try:
            out_b, cb = run(b, ib, capture=sorted(set(qb) | {byb, ib.shape[1] - 1}), attentions=want_att)
            out_s, cs = run(s, is_, capture=sorted(set(qs) | {bys, is_.shape[1] - 1}), attentions=want_att)
        except Exception as exc:  # model cannot return attention weights with this backend
            if not want_att:
                raise
            print(f"attention weights unavailable ({type(exc).__name__}); skipping heads_attention", flush=True)
            attn_ok = False
            out_b, cb = run(b, ib, capture=sorted(set(qb) | {byb, ib.shape[1] - 1}))
            out_s, cs = run(s, is_, capture=sorted(set(qs) | {bys, is_.shape[1] - 1}))
        out_p, cp = run(pq, ip, capture=[byp])
        ld_b, ld_s = margin(out_b.logits[0, -1].float(), b), margin(out_s.logits[0, -1].float(), s)
        if ld_s - ld_b <= args.min_gap:
            continue
        gap = ld_s - ld_b
        rec = {k: [] for k in curves_names}
        for li in lays:
            if li in ok_layers:
                ld = margin(run(b, ib, block=(li, qb))[0].logits[0, -1].float(), b)
                rec["read_question_fail"].append((ld - ld_b) / gap)
                ld = margin(run(b, ib, block=(li, nb["agent"] + nb["patient"]))[0].logits[0, -1].float(), b)
                rec["read_names_fail"].append((ld - ld_b) / gap)
                ld = margin(run(s, is_, block=(li, qs))[0].logits[0, -1].float(), s)
                rec["read_question_work"].append((ld_s - ld) / gap)
                ld = margin(run(s, is_, block=(li, ns["agent"] + ns["patient"]))[0].logits[0, -1].float(), s)
                rec["read_names_work"].append((ld_s - ld) / gap)
            else:
                for k in ("read_question_fail", "read_names_fail", "read_question_work", "read_names_work"):
                    rec[k].append(float("nan"))
            # question tokens aligned from the end of the question ("by", "?" ...), rest by order
            qmap = {qb[-1 - k]: cs[li][qs[-1 - k]] for k in range(min(len(qb), len(qs)))}
            ld = margin(run(b, ib, patches={li: qmap})[0].logits[0, -1].float(), b)
            rec["swap_question"].append((ld - ld_b) / gap)
            ld = margin(run(b, ib, patches={li: {ib.shape[1] - 1: cs[li][is_.shape[1] - 1]}})[0].logits[0, -1].float(), b)
            rec["swap_answer"].append((ld - ld_b) / gap)
            ld = margin(run(s, is_, patches={li: {bys: cb[li][byb]}})[0].logits[0, -1].float(), s)
            rec["by_from_failing"].append((ld_s - ld) / gap)
            ld = margin(run(s, is_, patches={li: {bys: cp[li][byp]}})[0].logits[0, -1].float(), s)
            rec["by_from_patient"].append((ld_s - ld) / gap)
        heads = {}
        if head_list and attn_ok is not False:
            att_b, att_s = getattr(out_b, "attentions", None), getattr(out_s, "attentions", None)
            if not att_b or not att_s or any(a is None for a in att_b):
                attn_ok = False
            else:
                attn_ok = True
                for hk in head_list:
                    li, h = (int(x) for x in hk.split("."))
                    if li >= len(att_b) or att_b[li] is None:
                        continue
                    ab, as_ = att_b[li][0, h, -1].float(), att_s[li][0, h, -1].float()
                    heads[hk] = {"fail_agent": float(ab[nb["agent"]].sum()), "fail_patient": float(ab[nb["patient"]].sum()),
                                 "work_agent": float(as_[ns["agent"]].sum()), "work_patient": float(as_[ns["patient"]].sum())}
        per_item[b.item_id] = rec
        fam_of[b.item_id] = b.family_id
        meta.append({"item_id": b.item_id, "family_id": b.family_id, "construction": b.construction,
                     "ld_failing": ld_b, "ld_working": ld_s, "ld_patient_q": margin(out_p.logits[0, -1].float(), b),
                     "heads_attention": heads})
        if len(per_item) % 5 == 1:
            print(f"  {len(per_item)} items, {time.time() - t0:.0f}s", flush=True)
    if not per_item:
        raise SystemExit("no analysable items")

    summary = {}
    for k in curves_names:
        cur = {i: per_item[i][k] for i in per_item}
        if all(all(x != x for x in v) for v in cur.values()):
            summary[k] = None
            continue
        summary[k] = boot_summary(cur, fam_of, lays, L, n_boot=args.n_boot, seed=args.seed)
    for sub in ("wrong", "right"):
        ids = [m["item_id"] for m in meta if (m["ld_failing"] < 0) == (sub == "wrong")]
        if len({fam_of[i] for i in ids}) >= 5:
            summary[f"by_from_patient_{sub}"] = boot_summary({i: per_item[i]["by_from_patient"] for i in ids},
                                                             fam_of, lays, L, n_boot=args.n_boot, seed=args.seed)
    head_summary = None
    if attn_ok:
        head_summary = {}
        for hk in head_list:
            vals = [m["heads_attention"][hk] for m in meta if hk in m["heads_attention"]]
            if vals:
                head_summary[hk] = {k: round(sum(v[k] for v in vals) / len(vals), 4) for k in vals[0]}

    report = {"experiment": "OTR-14 layer pipeline", "otr14_version": OTR14_VERSION,
              "generator_version": ors.GENERATOR_VERSION, "stimuli": args.stimuli, "model": args.model,
              "chat_template": args.chat_template, "base_question": args.base_question,
              "source_question": args.source_question, "patient_question": args.patient_question,
              "instruction": args.instruction, "n_layers": L, "layers": lays,
              "attention_blocking_layers": sorted(ok_layers), "heads_attention_available": attn_ok,
              "n_items": len(per_item), "n_base_wrong": sum(m["ld_failing"] < 0 for m in meta),
              "summary": summary, "heads_attention": head_summary, "items": meta, "curves": per_item,
              "seconds": round(time.time() - t0)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stag = "" if args.stimuli == "otr" else f"_{args.stimuli}"
    out = args.out_dir / f"otr14_{args.model.replace('/', '__')}_dev{stag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    for k, v in summary.items():
        if v:
            print(f"  {k:22} peak {max(v['mean']):+.2f} onset {v['onset_depth']} {v['onset_ci']}")
    _ = random


if __name__ == "__main__":
    main()
