"""OTR-10: from order to role, word by word and layer by layer.

Question: as a sentence is read, when (at which word) and where (at which layer) does the
model come to represent WHO is the first-mentioned name, WHO is the grammatical subject
and WHO is the agent? Because the constructions decorrelate the three labels, the three
curves can be measured separately; "from order to role" predicts that, at a given word,
the order code is readable earliest in depth, the subject code next and the agent code
last, and that the agent code appears at exactly the word that disambiguates it (the
passive "was V-ed", the cleft verb, "by", ...).

Items: as OTR-08 (dev families x both role assignments x --verbs-per-event verbs x
constructions; English adds the held-out relatives; Spanish/Galician add ovs_clitic).
One forward pass per item. States are read at the LAST TOKEN OF EVERY WORD of the
sentence, at the end of the (default, active agent) question and at the answer cue.
Causal attention means the state at word k has seen exactly words 0..k.

Probe (per layer): ridge regression from the residual stream to a one-hot code of a
target name among the 24 names, for target in {first, subject, agent, patient, recent}
("recent" = the most recently read name: the recency baseline). The prediction is scored
on the item's two names only (target vs the other name), so chance is 0.5 and the probe
generalises to name pairs it never saw. A LINEAR probe trained on constructions where
the agent is sometimes the first and sometimes the second name can only succeed if the
state carries the agent's identity as such, not "identity of the first name" plus "which
construction this is".

Identity confound: a state AT a name is dominated by that name's identity, so any probe
would "pick" the current name. Before probing, each name's mean state over all name
positions (where it is agent and patient equally often) is subtracted from the states at
that name. Recency: where the role is not encoded, a probe trained on a role-balanced
mixture can still lean towards the most recent name; compare with the "recent" row. The
cleanest test is mode "active": at its training positions the recent name is always the
patient and the agent is always first, so an active-trained agent probe that picks the
agent of a passive (agent = second = recent, not subject) can only be reading role. (In
by-clefts the agent is first AND not recent, so there "active" transfer is not diagnostic;
in object clefts it shows role-or-subject.)

  train   states at the second name through the period (both names have been read),
          primary language; mode "main" = the four main constructions (labels exactly
          uncorrelated), mode "active" = actives only (agent = first = subject: any
          success on passives/clefts is transfer of a role code).
  test    every position of every construction (held-out relatives and subject clefts
          included), families in a held-out fold AND the other half of the verbs.
  E3      probes trained on the primary language tested on every position of the other
          languages (held-out families).

Values are P(probe chooses the target name). 0.5 = not represented; < 0.5 = the probe
systematically picks the OTHER name (e.g. an agent probe reading the first name in a
passive). Validity check: at the first name of active vs passive the left context is
identical in kind, so P(agent | active) + P(agent | passive) must be ~1 at every layer.

Memory: states are written into one preallocated float16 tensor [layers, states, d] on
the GPU when it fits next to the model (else on the CPU), with no intermediate lists.

    python scripts/otr10_incremental_roles.py --model google/gemma-4-31B-it --chat-template \\
        --langs en,es,gl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

OTR10_VERSION = "1.0.0"
TARGETS = ("first", "subject", "agent", "patient", "recent")
MODES = ("main", "active")
_WORD = re.compile(r"\w+|[^\w\s]")


def _o8():
    spec = importlib.util.spec_from_file_location("o8", ROOT / "scripts" / "otr08_role_probes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


O8 = _o8()
build_items = O8.build_items
constructions_for = O8.constructions_for


def main_constructions_for(lang: str):
    if lang == "en":
        return ors.MAIN_CONSTRUCTIONS
    from relational_lab.experiments import multilingual_stimuli as xl
    return xl.XL_MAIN_CONSTRUCTIONS


def target_name(it, target: str, recent: str | None = None) -> str | None:
    """'recent' = the most recently read name at the readout position (None before any name)."""
    if target == "recent":
        return recent
    if target == "patient":
        return it.patient
    return O8.target_name(it, target)


def item_positions(it):
    """Readout positions of one item: (label, start, end, is_train, current_name, recent_name).

    One entry per word of the sentence (label 'w<k>'), then 'question_end' and 'answer'.
    Character offsets are in prompt coordinates. is_train marks words from the second name
    through the period; current_name is the name if the word IS a name; recent_name is
    the most recently read name (None before the first name)."""
    off = it.prompt.index(it.sentence)
    words = [(m.group(), m.start() + off, m.end() + off) for m in _WORD.finditer(it.sentence)]
    second = it.names[1].prompt_char_start
    k2 = next(k for k, (_, a, b) in enumerate(words) if a <= second < b)
    out, recent = [], None
    for k, (w, a, b) in enumerate(words):
        cur = w if w in (it.agent, it.patient) else None
        recent = cur or recent
        out.append((f"w{k}", a, b, k >= k2, cur, recent))
    qa, qb = it.span("question_end")
    ca, cb = it.span("answer_cue")
    out.append(("question_end", qa, qb, False, None, recent))
    out.append(("answer", ca, cb, False, None, recent))
    return out


def word_labels(it):
    """Display labels for an item's positions, names shown as A (agent) / P (patient)."""
    labs = []
    for m in _WORD.finditer(it.sentence):
        w = m.group()
        labs.append("A" if w == it.agent else "P" if w == it.patient else w)
    return labs + ["?", "Answer:"]


def last_token(offsets, a: int, b: int) -> int:
    tix = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
    if not tix:
        raise RuntimeError(f"no token overlaps characters [{a}, {b})")
    return tix[-1]


def ridge_scores(Xtr, Ytr, Xte, lam_rel=1.0):
    """Primal ridge with intercept: scores for Xte. Xtr [n, d], Ytr [n, k]."""
    import torch

    mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx
    G = Xc.T @ Xc
    lam = lam_rel * G.diagonal().mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), Xc.T @ (Ytr - my))
    return (Xte - mx) @ W + my


def pick_store_device(n_bytes: int, want: str, margin_gib: float = 8.0) -> str:
    import torch

    if want in ("cpu", "cuda"):
        return want
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free - n_bytes > margin_gib * 2**30:
            return "cuda"
    return "cpu"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--verbs-per-event", type=int, default=4)
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--langs", default="en", help="comma list, first = primary (probes are trained on it)")
    p.add_argument("--lam", type=float, default=1.0, help="ridge strength relative to mean Gram diagonal")
    p.add_argument("--store", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-items", type=int, default=0, help="debug: cap items per language")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg)
    import torch

    from relational_lab.instrumentation.hf_scoring import load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, decoder_layers

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    layers = decoder_layers(model)
    L = len(layers)
    use_layers = sorted(set(list(range(0, L, args.layer_stride)) + [L - 1]))
    D = model.get_input_embeddings().weight.shape[1]

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    prim = langs[0]
    items, lang_of = [], []
    for lg in langs:
        its = build_items(args.seed, args.verbs_per_event, lg)
        if args.max_items:
            its = its[: args.max_items]
        items += its
        lang_of += [lg] * len(its)

    # Tokenise everything first so the state tensor can be allocated once.
    enc, rows, meta = [], [], []
    for i, it in enumerate(items):
        ids, offsets = _encode_one(tok, it.prompt, args.chat_template)
        pos = item_positions(it)
        ix = [last_token(offsets, a, b) for _, a, b, *_ in pos]
        enc.append((ids, ix))
        r0 = len(meta)
        for (lab, _, _, is_tr, cur_name, recent) in pos:
            meta.append((i, lab, is_tr, cur_name, recent))
        rows.append((r0, len(meta)))
    N = len(meta)
    n_bytes = len(use_layers) * N * D * 2
    store_dev = pick_store_device(n_bytes, args.store)
    print(f"{len(items)} items, {N} states, {len(use_layers)} layers x d={D}: "
          f"{n_bytes / 2**30:.1f} GiB on {store_dev}", flush=True)
    feats = torch.empty((len(use_layers), N, D), dtype=torch.float16, device=store_dev)

    cur = {}
    handles = []
    for k, li in enumerate(use_layers):
        def hook(_m, _i, out, k=k):
            h = _hidden(out)
            feats[k, cur["r0"]:cur["r1"]] = h[0, cur["ix"]].detach().to(store_dev, torch.float16)
        handles.append(layers[li].register_forward_hook(hook))
    t0 = time.time()
    try:
        for i, (ids, ix) in enumerate(enc):
            cur["r0"], cur["r1"] = rows[i]
            cur["ix"] = torch.tensor(ix, device=device)
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device))
            if i and i % 500 == 0:
                print(f"  {i}/{len(enc)} items, {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in handles:
            h.remove()
    fwd_seconds = time.time() - t0
    del model, layers
    if device == "cuda":
        torch.cuda.empty_cache()

    # Per-state metadata.
    names = list(ors.NAMES)
    nidx = {n: j for j, n in enumerate(names)}
    fam_ids = sorted({it.family_id for it in items})
    lemmas = {lg: sorted({it.lemma for it, l in zip(items, lang_of) if l == lg}) for lg in langs}
    it_of = torch.tensor([m[0] for m in meta])
    s_lang = [lang_of[m[0]] for m in meta]
    s_cons = [items[m[0]].construction for m in meta]
    s_lab = [m[1] for m in meta]
    is_train_pos = torch.tensor([m[2] for m in meta])
    is_prim = torch.tensor([lg == prim for lg in s_lang])
    fold = torch.tensor([fam_ids.index(items[m[0]].family_id) % 5 for m in meta])
    vhalf = torch.tensor([lemmas[lang_of[m[0]]].index(items[m[0]].lemma) % 2 for m in meta])
    is_main = torch.tensor([items[m[0]].construction in main_constructions_for(lang_of[m[0]]) for m in meta])
    is_act = torch.tensor([items[m[0]].construction == "active" for m in meta])
    tname = {t: [target_name(items[m[0]], t, m[4]) for m in meta] for t in TARGETS}
    defined = {t: torch.tensor([x is not None for x in tname[t]]) for t in TARGETS}
    tgt = {t: torch.tensor([nidx[x] if x is not None else 0 for x in tname[t]]) for t in TARGETS}
    oth = {t: torch.tensor([nidx[items[m[0]].patient if x == items[m[0]].agent else items[m[0]].agent]
                            for m, x in zip(meta, tname[t])]) for t in TARGETS}
    nN = len(names)
    Y = torch.zeros((N, nN * len(TARGETS)))
    for j, t in enumerate(TARGETS):
        Y[torch.arange(N), j * nN + tgt[t]] = defined[t].float()
    # States AT a name are dominated by that name's identity, which would make every probe
    # "pick" the current name. Remove it: subtract each name's mean state over all name
    # positions (roles are balanced there), add back the grand mean of name positions.
    # Done separately per language (the context around a name differs between languages).
    cur = torch.tensor([nidx[m[3]] if m[3] is not None else -1 for m in meta])
    name_groups = []
    for lg in langs:
        in_lg = torch.tensor([s == lg for s in s_lang])
        at = ((cur >= 0) & in_lg).nonzero().squeeze(1)
        name_groups.append((at, [((cur == j) & in_lg).nonzero().squeeze(1) for j in range(nN)]))

    pdev = "cuda" if torch.cuda.is_available() else "cpu"
    Y = Y.to(pdev)
    train_mode = {"main": is_main, "active": is_act}
    # hits[mode][t] : [n_layers', N] bool, tested : [mode] [N] bool (same for all layers)
    hits = {m: {t: torch.zeros((len(use_layers), N), dtype=torch.bool) for t in TARGETS} for m in MODES}
    tested = {m: torch.zeros(N, dtype=torch.bool) for m in MODES}
    t1 = time.time()
    for k in range(len(use_layers)):
        X = feats[k].to(pdev).float()
        for at, groups in name_groups:
            if not len(at):
                continue
            grand = X[at.to(pdev)].mean(0)
            for g in groups:
                if len(g):
                    gd = g.to(pdev)
                    X[gd] -= X[gd].mean(0) - grand
        for mode in MODES:
            splits = []
            for f in range(5):  # primary language: held-out family fold AND other verb half
                for vh in (0, 1):
                    splits.append((train_mode[mode] & is_train_pos & is_prim & (fold != f) & (vhalf == vh),
                                   is_prim & (fold == f) & (vhalf != vh)))
            for lg in langs[1:]:  # other languages: held-out family fold, all verbs
                is_lg = torch.tensor([s == lg for s in s_lang])
                for f in range(5):
                    splits.append((train_mode[mode] & is_train_pos & is_prim & (fold != f), is_lg & (fold == f)))
            for tr, te in splits:
                if tr.sum() < 10 or te.sum() == 0:
                    continue
                trd, ted = tr.to(pdev), te.to(pdev)
                S = ridge_scores(X[trd], Y[trd], X[ted], args.lam)
                te_ix = te.nonzero().squeeze(1)
                for j, t in enumerate(TARGETS):
                    blk = S[:, j * nN:(j + 1) * nN]
                    r = torch.arange(len(te_ix), device=pdev)
                    ok = blk[r, tgt[t][te_ix].to(pdev)] > blk[r, oth[t][te_ix].to(pdev)]
                    hits[mode][t][k, te_ix] = ok.cpu()
                tested[mode] |= te
    probe_seconds = time.time() - t1

    # Aggregate: results[lang][mode][target][construction] = [n_positions][n_layers'].
    results, labels, n_cell = {}, {}, {}
    for lg in langs:
        results[lg], labels[lg], n_cell[lg] = {m: {t: {} for t in TARGETS} for m in MODES}, {}, {}
        for c in constructions_for(lg):
            ex = next(it for it, l in zip(items, lang_of) if l == lg and it.construction == c)
            pos_labs = [lab for lab, *_ in item_positions(ex)]
            labels[lg][c] = word_labels(ex)
            masks = [torch.tensor([sl == lg and sc == c and lb == pl for sl, sc, lb in zip(s_lang, s_cons, s_lab)])
                     for pl in pos_labs]
            n_cell[lg][c] = [int((mk & tested["main"]).sum()) for mk in masks]
            for mode in MODES:
                for t in TARGETS:
                    mat = []
                    for mk in masks:
                        mm = mk & tested[mode] & defined[t]
                        mat.append([round(float(hits[mode][t][k, mm].float().mean()), 4) if mm.any() else None
                                    for k in range(len(use_layers))])
                    results[lg][mode][t][c] = mat

    validity = None
    if prim == "en":
        a = results["en"]["main"]["agent"]
        fa = [lab for lab, *_ in item_positions(next(it for it in items if it.construction == "active"))]
        fp = [lab for lab, *_ in item_positions(next(it for it in items if it.construction == "passive"))]
        ka, kp = fa.index("w0"), fp.index("w0")
        validity = [None if x is None or y is None else round(x + y, 4)
                    for x, y in zip(a["active"][ka], a["passive"][kp])]

    report = {
        "experiment": "OTR-10", "status": "EXPLORATORY", "version": OTR10_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "langs": langs, "n_items": len(items), "n_states": N,
        "n_items_by_lang": {lg: lang_of.count(lg) for lg in langs}, "verbs_per_event": args.verbs_per_event,
        "n_layers": L, "layers": use_layers, "lam_rel": args.lam, "targets": list(TARGETS), "modes": list(MODES),
        "store_device": store_dev, "forward_seconds": round(fwd_seconds, 1), "probe_seconds": round(probe_seconds, 1),
        "validity_first_name_active_plus_passive_agent": validity,
        "labels": labels, "n_tested": n_cell, "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ltag = "" if langs == ["en"] else "_" + "-".join(langs)
    out = args.out_dir / f"otr10_{args.model.replace('/', '__')}_dev{ltag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    r = results[prim]["main"]
    for c in main_constructions_for(prim):
        print(f"  [{prim} main] {c:13s} period, last layer:",
              {t: r[t][c][len(labels[prim][c]) - 3][-1] for t in TARGETS})
    print("  validity (should be ~1):", validity if validity is None else [validity[0], validity[-1]])
    try:
        for path in _plot(report, out):
            print(f"Wrote {path}")
    except Exception as exc:  # plotting must never lose a finished run
        print(f"(no plot: {exc})")


DIVERGING = ("#e34948", "#f0efec", "#2a78d6")  # other name  /  chance  /  target name


def _heat(ax, mat, labels, layers, L, title):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    a = np.array([[np.nan if v is None else v for v in row] for row in mat]).T  # [layers, positions]
    im = ax.imshow(a, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(-0.5, len(labels) - 0.5, -0.5, len(layers) - 0.5))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ticks = [0, len(layers) // 2, len(layers) - 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(layers[t]) for t in ticks], fontsize=7)
    ax.set_title(title, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def _plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    prim, L, layers = report["langs"][0], report["n_layers"], report["layers"]
    cons = list(report["results"][prim]["main"]["agent"])
    fig, axes = plt.subplots(4, len(cons), figsize=(2.3 * len(cons), 11), squeeze=False,
                             gridspec_kw={"hspace": 0.9, "wspace": 0.3})
    for i, t in enumerate(("recent", "first", "subject", "agent")):
        for j, c in enumerate(cons):
            im = _heat(axes[i, j], report["results"][prim]["main"][t][c], report["labels"][prim][c], layers, L,
                       f"{t} | {c}")
        axes[i, 0].set_ylabel("layer", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("P(probe picks the target name)\n0.5 = not encoded, <0.5 = picks the other name", fontsize=8)
    fig.suptitle(f"OTR-10 {report['model']} [{prim}]: probes trained on the four main constructions; "
                 "held-out families and verbs", fontsize=9)
    p1 = out.with_name(out.stem + "_main.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    langs = report["langs"]
    ncol = max(len(report["results"][lg]["active"]["agent"]) for lg in langs)
    fig, axes = plt.subplots(len(langs), ncol, figsize=(2.3 * ncol, 2.9 * len(langs)), squeeze=False,
                             gridspec_kw={"hspace": 0.9, "wspace": 0.3})
    for i, lg in enumerate(langs):
        cs = list(report["results"][lg]["active"]["agent"])
        for j in range(ncol):
            if j >= len(cs):
                axes[i, j].axis("off")
                continue
            im = _heat(axes[i, j], report["results"][lg]["active"]["agent"][cs[j]], report["labels"][lg][cs[j]],
                       layers, L, f"{lg} | {cs[j]}")
        axes[i, 0].set_ylabel("layer", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("P(agent probe picks the agent)", fontsize=8)
    fig.suptitle(f"OTR-10 {report['model']}: agent probe trained on {prim} ACTIVES only "
                 "(agent = first = subject), tested everywhere", fontsize=9)
    p2 = out.with_name(out.stem + "_active_transfer.png")
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)
    return paths


if __name__ == "__main__":
    main()
