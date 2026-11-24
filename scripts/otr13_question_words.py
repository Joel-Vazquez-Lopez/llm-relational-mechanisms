#!/usr/bin/env python
"""OTR-13: which word of the question tells the answer what to fetch, and at which layer?

OTR-09 showed that the failing question ("Who was someone {past} by?") is repaired by patching
the answer position (last token) from the working question ("Who was the {ing} done by?") from a
model-specific handover layer on. OTR-13 moves one step back: it patches the residual state of
ONE question word (or one prompt-tail token) at ONE layer and measures how much of the gap
between the failing and the working question that closes at the answer.

For every analysed item (same sentence, failing vs working question):

  A  interchange   failing-question run, state of word w at layer l replaced by the state of
                   its counterpart in the working-question run (Who->Who, was->was,
                   someone->the, VERB->ING, by->by, ?->?; tail token k->tail token k, counted
                   from the end, k = 0 is the answer position = OTR-09).
  J  joint         all question words at once at layer l (the question as a whole).
  M  mean-ablate   failing-question run, state of word w at layer l replaced by its mean over
                   all failing-question items: is the word's specific content needed for the
                   error? (repair > 0 = removing it helps).
  R  reverse       working-question run with the failing question's state of w at layer l
                   (does that word carry the "fetch the patient" signal?), reported as the share
                   of the working margin destroyed.

repair = (ld_patched - ld_failing) / (ld_working - ld_failing), ld = logit(agent) - logit(patient)
at the answer position (first answer token, as in OTR-09).

Account (b') predicts the patient signal is written at the end of the local chunk
"who was someone V-ed" (the verb) and carried to the answer through the tail; "by" and "?" fail
to overwrite it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import otr09_answer_patching as o9  # noqa: E402

from relational_lab.experiments.splits import SPLITS  # noqa: E402
from relational_lab.experiments.splits import status as split_status  # noqa: E402

OTR13_VERSION = "1.1.0"
BASE_WORDS = ("Who", "was", "someone", "VERB", "by", "?")
FUNCTION_WORDS = {"Who", "was", "someone", "the", "of", "done", "by", "?"}


def question_words(question: str, kind: str) -> dict[str, tuple[int, int]]:
    """char spans (inside the question) of its words, named by themselves; the one content word
    (the verb form) is named VERB in the failing (base) question and ING in the working one."""
    spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z\-']+|\?", question)]
    content = [w for w, _, _ in spans if w not in FUNCTION_WORDS]
    if len(content) != 1 or len({w for w, _, _ in spans}) != len(spans):
        raise ValueError(f"unexpected question shape: {question!r}")
    return {(("VERB" if kind == "base" else "ING") if w not in FUNCTION_WORDS else w): (s, e) for w, s, e in spans}


def source_of(name: str, source_words) -> str | None:
    """counterpart of a failing-question word in the working question (None = no counterpart)."""
    if name.startswith("tail"):
        return name
    if name == "VERB":
        return "ING"
    if name in source_words:
        return name
    if name == "someone" and "the" in source_words:
        return "the"
    return None


def token_at(offsets, char_end: int) -> int:
    """index of the token covering the character just before ``char_end`` (prompt coordinates)."""
    c = char_end - 1
    hits = [i for i, (a, b) in enumerate(offsets) if a <= c < b]
    if not hits:
        raise ValueError(f"no token covers char {c}")
    return hits[-1]


def anchors(item, offsets, kind: str, n_ids: int, n_tail: int):
    """name -> token index for the question words and the last ``n_tail`` tokens."""
    q0 = item.prompt.index("\nQuestion: ") + len("\nQuestion: ")
    out = {name: token_at(offsets, q0 + e) for name, (s, e) in question_words(item.question, kind).items()}
    for k in range(n_tail):
        out[f"tail{k}"] = n_ids - 1 - k
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--stimuli", default="otr", choices=["otr", "bx-transitive", "bx-psych"])
    p.add_argument("--base-question", default=None, help='default "Who was someone {past} by?" (bx: {pp})')
    p.add_argument("--source-question", default="Who was the {ing} done by?")
    p.add_argument("--instruction", default="Reply with one name only.")
    p.add_argument("--constructions", default=None)
    p.add_argument("--n-per", type=int, default=16)
    p.add_argument("--n-tail", type=int, default=6, help="prompt-tail tokens before and incl. the answer position")
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--min-gap", type=float, default=1.0)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split", default="dev", choices=SPLITS, help="test0..test3 only after FREEZE.md")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    if args.stimuli == "otr":
        args.base_question = args.base_question or "Who was someone {past} by?"
        args.constructions = args.constructions or "active,object_cleft,passive,by_cleft"
    else:
        args.base_question = args.base_question or "Who was someone {pp} by?"
        args.constructions = args.constructions or ",".join(o9.bxa.BASE_CONSTRUCTIONS[o9.bxa.block_name(args.stimuli)])

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
    lead = not args.chat_template
    cons = [c.strip() for c in args.constructions.split(",")]
    triples = o9.make_pairs(args.seed, args.base_question, args.source_question, args.instruction, cons, args.n_per,
                            stimuli=args.stimuli, split=args.split)
    triples = [t for t in triples if answer_token(tok, t[0].answer, lead) != answer_token(tok, t[0].distractor, lead)]

    def enc(it):
        ids, offs = _encode_one(tok, it.prompt, args.chat_template)
        return torch.tensor([ids], device=device), offs

    def forward(it, ids, patches=None, capture_pos=None):
        """patches: {layer: {pos: vec}}; capture_pos: list of positions -> {layer: {pos: vec}}."""
        caps, hs = {}, []
        if capture_pos is not None:
            for li in lays:
                def ch(_m, _i, out, li=li):
                    h = _hidden(out)[0]
                    caps[li] = {q: h[q].detach().clone() for q in capture_pos}
                hs.append(layers[li].register_forward_hook(ch))
        for li, pv in (patches or {}).items():
            def ph(_m, _i, out, pv=pv):
                h = _hidden(out).clone()
                for q, v in pv.items():
                    h[0, q] = v.to(h.dtype)
                return _replace_hidden(out, h)
            hs.append(layers[li].register_forward_hook(ph))
        try:
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        a, b = answer_token(tok, it.answer, lead), answer_token(tok, it.distractor, lead)
        return float(lg[a] - lg[b]), caps

    # ---- pass 1: clean runs, anchors, captures ----
    t0 = time.time()
    data = []
    for b, s, _d in triples:
        ib, ob = enc(b)
        is_, os_ = enc(s)
        ab = anchors(b, ob, "base", ib.shape[1], args.n_tail)
        as_ = anchors(s, os_, "source", is_.shape[1], args.n_tail)
        ld_b, cb = forward(b, ib, capture_pos=sorted(set(ab.values())))
        ld_s, cs = forward(s, is_, capture_pos=sorted(set(as_.values())))
        if ld_s - ld_b <= args.min_gap:
            continue
        toks_b = {n: tok.decode([int(ib[0, q])]) for n, q in ab.items()}
        data.append(dict(b=b, s=s, ib=ib, is_=is_, ab=ab, as_=as_, ld_b=ld_b, ld_s=ld_s, cb=cb, cs=cs, toks=toks_b))
    print(f"{len(data)} items with a working-question gap > {args.min_gap} ({time.time() - t0:.0f}s)", flush=True)
    if not data:
        raise SystemExit("no analysable items")

    names = [*BASE_WORDS, *[f"tail{k}" for k in range(args.n_tail)]]
    # mean failing-question state per (word, layer)
    mean_state = {li: {n: torch.stack([d["cb"][li][d["ab"][n]] for d in data]).mean(0) for n in names} for li in lays}

    src_words = set(question_words(data[0]["s"].question, "source"))

    def src_name(n):
        return source_of(n, src_words)

    def rep(ld, d):
        return (ld - d["ld_b"]) / (d["ld_s"] - d["ld_b"])

    res = {kind: {n: {li: [] for li in lays} for n in names} for kind in ("interchange", "mean_ablate", "reverse")}
    joint = {li: [] for li in lays}
    flips = {n: {li: [] for li in lays} for n in names}
    for k, d in enumerate(data):
        b, s = d["b"], d["s"]
        for li in lays:
            for n in names:
                qb = d["ab"][n]
                ld, _ = forward(b, d["ib"], {li: {qb: mean_state[li][n]}})
                res["mean_ablate"][n][li].append(rep(ld, d))
                if src_name(n) is None:
                    continue
                qs = d["as_"][src_name(n)]
                ld, _ = forward(b, d["ib"], {li: {qb: d["cs"][li][qs]}})
                res["interchange"][n][li].append(rep(ld, d))
                if d["ld_b"] < 0:
                    flips[n][li].append(ld > 0)
                ld, _ = forward(s, d["is_"], {li: {qs: d["cb"][li][qb]}})
                res["reverse"][n][li].append((d["ld_s"] - ld) / (d["ld_s"] - d["ld_b"]))
            qpos = {d["ab"][n]: d["cs"][li][d["as_"][src_name(n)]] for n in BASE_WORDS if src_name(n) is not None}
            joint[li].append(rep(forward(b, d["ib"], {li: qpos})[0], d))
        if k % 5 == 0:
            print(f"  {k + 1}/{len(data)} items, {time.time() - t0:.0f}s", flush=True)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    report = {
        "status": split_status(args.split), "split": args.split, "experiment": "OTR-13 question-word patching", "otr13_version": OTR13_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "stimuli": args.stimuli, "model": args.model,
        "chat_template": args.chat_template,
        "base_question": args.base_question, "source_question": args.source_question,
        "instruction": args.instruction, "n_layers": L, "layers": lays, "n_items": len(data),
        "n_base_wrong": sum(d["ld_b"] < 0 for d in data), "words": names,
        "source_of": {n: src_name(n) for n in names},
        "tokens_example": data[0]["toks"],
        "shared_token_warnings": sorted({f"{n1}={n2}" for d in data for n1 in names for n2 in names
                                         if n1 < n2 and d["ab"][n1] == d["ab"][n2]}),
        "interchange": {n: {li: mean(v) for li, v in res["interchange"][n].items()} for n in names},
        "mean_ablate": {n: {li: mean(v) for li, v in res["mean_ablate"][n].items()} for n in names},
        "reverse_damage": {n: {li: mean(v) for li, v in res["reverse"][n].items()} for n in names},
        "flip_wrong": {n: {li: mean([float(x) for x in v]) for li, v in flips[n].items()} for n in names},
        "joint_question": {li: mean(v) for li, v in joint.items()},
        "seconds": round(time.time() - t0),
        "items": [{"item_id": d["b"].item_id, "construction": d["b"].construction, "ld_failing": d["ld_b"],
                   "ld_working": d["ld_s"]} for d in data],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stag = "" if args.stimuli == "otr" else f"_{args.stimuli}"
    qtag = "" if args.source_question == "Who was the {ing} done by?" else "_src2"
    out = args.out_dir / f"otr13_{args.model.replace('/', '__')}_{args.split}{stag}{qtag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    try:
        _plot(report, out.with_suffix(".png"))
    except Exception as exc:  # plotting must never lose the results
        print("plot failed:", exc)
    print(f"wrote {out}")
    best = max(((n, li, v) for n in names for li, v in report["interchange"][n].items() if v is not None),
               key=lambda x: x[2])
    print("largest single-word repair:", best, "| joint question peak:", max(v for v in report["joint_question"].values()))


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, lays, L = report["words"], report["layers"], report["n_layers"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, key, title in zip(axs, ("interchange", "mean_ablate", "reverse_damage"),
                              ("A. working-question word patched in\n(repair of the failing answer)",
                               "M. word mean-ablated in the failing question\n(repair)",
                               "R. failing-question word patched into the working question\n(share of margin destroyed)")):
        m = [[(lambda v: float("nan") if v is None else v)(
                  report[key][n][str(li)] if isinstance(next(iter(report[key][n])), str) else report[key][n][li])
              for li in lays] for n in names]
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=(lays[0] / (L - 1), lays[-1] / (L - 1), len(names) - 0.5, -0.5))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relative depth")
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels([f"{n}  ({report['tokens_example'].get(n, '')!r})" for n in names], fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(f"OTR-13 {report['model']}  (n = {report['n_items']}, {report['n_base_wrong']} answered wrong)")
    fig.savefig(path, dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
