"""OTR-08: from order to role, layer by layer (probes on clean prompts).

Question: where in the network do the names of a sentence stop being represented by
position / grammatical function and start being represented by thematic role, and is
that role representation the same across constructions?

Items: every dev family (OTR splits), both role assignments, --verbs-per-event verbs, the
four main constructions plus subject_cleft and the held-out relative_active /
relative_passive. One forward pass per (event, construction). States are read at the
sentence's final period (the whole sentence is read, the question is not yet) and,
optionally, at the question end and the answer position of the item's default agent
question.

Probe: per layer, dual ridge regression from the residual stream to the input embedding
of a target name ("the agent", "the first-mentioned name", "the grammatical subject").
The prediction is scored by cosine against the two names of the item, so the probe
chooses between the two names present and generalises to name pairs it never saw.
Chance = 0.5.

  E1 decodability   train on all constructions; 5-fold grouped by family, and train on
                    half of the verbs / test on the other half. Accuracy per test
                    construction for target in {agent, first, subject}.
  E2 transfer       train an agent probe on ONE construction (actives: agent = first =
                    subject), test on every other construction (held-out families and
                    verbs). P(choose agent) per test construction:
                      passive       agent is second and not subject  -> high = role
                      object_cleft  agent is second and subject      -> role or function
                      by_cleft      agent is first and not subject   -> role or order

    python scripts/otr08_role_probes.py --model google/gemma-4-31B-it --chat-template
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

OTR08_VERSION = "1.0.0"
PROBE_CONSTRUCTIONS = ors.MAIN_CONSTRUCTIONS + ors.CONTROL_CONSTRUCTIONS + ors.HELDOUT_CONSTRUCTIONS
TARGETS = ("agent", "first", "subject")


def constructions_for(lang: str):
    if lang == "en":
        return PROBE_CONSTRUCTIONS
    from relational_lab.experiments import multilingual_stimuli as xl
    return xl.XL_CONSTRUCTIONS


def build_items(seed: int, verbs_per_event: int, lang: str = "en"):
    """Dev families x both orientations x verbs x constructions (default agent question)."""
    splits = ors.split_families(ors.NAMES, seed, 0.15, 4)
    dev = sorted(f for f, (s, _) in splits.items() if s == "dev")
    if lang == "en":
        verbs = [(lemma, past, None) for lemma, past in ors.VERBS]
    else:
        from relational_lab.experiments import multilingual_stimuli as xl
        verbs = list(xl.VERBS[lang])
    items = []
    for fi, fam in enumerate(dev):
        x, y = fam.split("|")
        for oi, (a, p) in enumerate(((x, y), (y, x))):
            chosen = [verbs[(fi * 2 * verbs_per_event + oi * verbs_per_event + k) % len(verbs)]
                      for k in range(verbs_per_event)]
            for lemma, past, stem in chosen:
                for cons in constructions_for(lang):
                    if lang == "en":
                        items.append(ors.build_item(a, p, lemma, past, cons, "agent", "active", family_id=fam))
                    else:
                        items.append(xl.build_item(lang, a, p, lemma, past, stem, cons, "agent", "active",
                                                   family_id=fam))
    return items


def target_name(it, target: str) -> str:
    if target == "agent":
        return it.agent
    if target == "first":
        return it.names[0].name
    if target == "subject":
        return next(n.name for n in it.names if n.is_subject)
    raise ValueError(target)


def ridge_fit_predict(Xtr, Ytr, Xte, lam_rel=1.0):
    """Dual ridge: W = Xtr^T (Xtr Xtr^T + lam I)^-1 Ytr; returns Xte W. Centred on Xtr."""
    import torch

    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    K = Xtr @ Xtr.T
    lam = lam_rel * K.diagonal().mean()
    alpha = torch.linalg.solve(K + lam * torch.eye(K.shape[0], device=K.device, dtype=K.dtype), Ytr)
    return (Xte @ Xtr.T) @ alpha


def choose(pred, emb_a, emb_b):
    """True where pred is closer (cosine) to emb_a than to emb_b."""
    import torch.nn.functional as F

    return F.cosine_similarity(pred, emb_a, dim=-1) > F.cosine_similarity(pred, emb_b, dim=-1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--verbs-per-event", type=int, default=8)
    p.add_argument("--layer-stride", type=int, default=2)
    p.add_argument("--positions", default="period", help="comma list from period,question_end,answer_cue")
    p.add_argument("--langs", default="en", help="comma list, first = primary (E1/E2); others get E3 transfer from it, e.g. en,es,gl")
    p.add_argument("--lam", type=float, default=1.0, help="ridge strength relative to mean kernel diagonal")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-items", type=int, default=0, help="debug: cap the number of items")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    positions = [x.strip() for x in args.positions.split(",") if x.strip()]

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
    emb = model.get_input_embeddings().weight

    def name_vec(name):
        ids = tok.encode(" " + name, add_special_tokens=False)
        return emb[ids].float().mean(0)

    name_emb = {n: name_vec(n) for n in ors.NAMES}

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    items_by_lang = {lg: build_items(args.seed, args.verbs_per_event, lg) for lg in langs}
    if args.max_items:
        items_by_lang = {lg: v[: args.max_items] for lg, v in items_by_lang.items()}
    items = [it for lg in langs for it in items_by_lang[lg]]
    lang_of = [lg for lg in langs for _ in items_by_lang[lg]]
    idx_now: dict = {}
    store = {}
    handles = []
    for li, layer in enumerate(layers):
        if li in use_layers:
            def hook(_m, _i, out, li=li):
                h = _hidden(out)
                store[li] = h[0, idx_now["ix"]].detach().to(torch.float16).cpu()
            handles.append(layer.register_forward_hook(hook))
    feats = {pos: [] for pos in positions}
    t0 = time.time()
    try:
        for it in items:
            ids, offsets = _encode_one(tok, it.prompt, args.chat_template)
            spans = dict((lab, (a, b)) for lab, a, b in it.spans)
            ix = []
            for pos in positions:
                a, b = spans[pos]
                tix = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
                if not tix:
                    raise RuntimeError(f"position {pos} not found for {it.item_id}")
                ix.append(tix[-1])
            idx_now["ix"] = ix
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device))
            for k, pos in enumerate(positions):
                feats[pos].append(torch.stack([store[li][k] for li in use_layers]))  # [L', D]
    finally:
        for h in handles:
            h.remove()
    fwd_seconds = time.time() - t0

    all_items, all_feats, all_lang = items, feats, lang_of
    prim = langs[0]
    keep = [i for i, lg in enumerate(all_lang) if lg == prim]
    items = [all_items[i] for i in keep]
    feats = {pos: [all_feats[pos][i] for i in keep] for pos in positions}
    PROBE_CONS = constructions_for(prim)
    fam_ids = sorted({it.family_id for it in items})
    fold = torch.tensor([fam_ids.index(it.family_id) % 5 for it in items])
    verb_half = torch.tensor([[v for v, _ in ors.VERBS].index(it.lemma) % 2 for it in items])
    cons_of = [it.construction for it in items]
    E = {t: torch.stack([name_emb[target_name(it, t)] for it in items]).to(device) for t in TARGETS}
    other = {t: torch.stack([name_emb[it.patient if target_name(it, t) == it.agent else it.agent] for it in items]).to(device)
             for t in TARGETS}
    emb_agent = E["agent"]
    emb_patient = torch.stack([name_emb[it.patient] for it in items]).to(device)

    results = {}
    for pos in positions:
        F = torch.stack(feats[pos])  # [N, L', D] fp16 cpu
        res_pos = {"E1": {t: {c: [] for c in PROBE_CONS} for t in TARGETS},
                   "E2": {tr: {c: [] for c in PROBE_CONS} for tr in ors.MAIN_CONSTRUCTIONS}}
        for k, li in enumerate(use_layers):
            X = F[:, k].to(device).float()
            # E1: all constructions; family fold != f AND verb half 0 -> test fold f, verb half 1 (and swapped)
            for t in TARGETS:
                ok = torch.zeros(len(items), dtype=torch.bool, device=device)
                tested = torch.zeros(len(items), dtype=torch.bool, device=device)
                for f in range(5):
                    for vh in (0, 1):
                        tr = (fold != f) & (verb_half == vh)
                        te = (fold == f) & (verb_half != vh)
                        if tr.sum() < 10 or te.sum() == 0:
                            continue
                        trd, ted = tr.to(device), te.to(device)
                        pred = ridge_fit_predict(X[trd], E[t][trd], X[ted], args.lam)
                        ok[ted] = choose(pred, E[t][ted], other[t][ted])
                        tested[ted] = True
                for c in PROBE_CONS:
                    m = torch.tensor([cc == c for cc in cons_of], device=device) & tested
                    res_pos["E1"][t][c].append(round(float(ok[m].float().mean()), 4) if m.any() else None)
            # E2: agent probe trained on one construction, tested on all (held-out families and verbs)
            for trc in ors.MAIN_CONSTRUCTIONS:
                agent_hit = torch.zeros(len(items), dtype=torch.bool, device=device)
                tested = torch.zeros(len(items), dtype=torch.bool, device=device)
                is_tr = torch.tensor([cc == trc for cc in cons_of])
                for f in range(5):
                    for vh in (0, 1):
                        tr = is_tr & (fold != f) & (verb_half == vh)
                        te = (fold == f) & (verb_half != vh)
                        if tr.sum() < 10:
                            continue
                        trd, ted = tr.to(device), te.to(device)
                        pred = ridge_fit_predict(X[trd], emb_agent[trd], X[ted], args.lam)
                        agent_hit[ted] = choose(pred, emb_agent[ted], emb_patient[ted])
                        tested[ted] = True
                for c in PROBE_CONS:
                    m = torch.tensor([cc == c for cc in cons_of], device=device) & tested
                    res_pos["E2"][trc][c].append(round(float(agent_hit[m].float().mean()), 4) if m.any() else None)
        results[pos] = res_pos

    # E3: cross-lingual transfer, probes trained on the primary language, tested on each other language
    transfer = {}
    for tgt in langs[1:]:
        ti = [i for i, lg in enumerate(all_lang) if lg == tgt]
        t_items = [all_items[i] for i in ti]
        t_cons = [it.construction for it in t_items]
        t_fold = torch.tensor([fam_ids.index(it.family_id) % 5 if it.family_id in fam_ids else 0 for it in t_items])
        ta = torch.stack([name_emb[it.agent] for it in t_items]).to(device)
        tp = torch.stack([name_emb[it.patient] for it in t_items]).to(device)
        out_t = {}
        for pos in positions:
            Fp = torch.stack(feats[pos])
            Ft = torch.stack([all_feats[pos][i] for i in ti])
            res = {"all_trained": {c: [] for c in constructions_for(tgt)},
                   "active_trained": {c: [] for c in constructions_for(tgt)}}
            for k, li in enumerate(use_layers):
                Xp, Xt = Fp[:, k].to(device).float(), Ft[:, k].to(device).float()
                for mode, trmask in (("all_trained", torch.ones(len(items), dtype=torch.bool)),
                                     ("active_trained", torch.tensor([c == "active" for c in cons_of]))):
                    hit = torch.zeros(len(t_items), dtype=torch.bool, device=device)
                    for f in range(5):  # held-out families: train on folds != f, test target-language fold f
                        tr = (trmask & (fold != f)).to(device)
                        te = (t_fold == f).to(device)
                        if tr.sum() < 10 or te.sum() == 0:
                            continue
                        pred = ridge_fit_predict(Xp[tr], emb_agent[tr], Xt[te], args.lam)
                        hit[te] = choose(pred, ta[te], tp[te])
                    for c in constructions_for(tgt):
                        m = torch.tensor([cc == c for cc in t_cons], device=device)
                        res[mode][c].append(round(float(hit[m].float().mean()), 4) if m.any() else None)
            out_t[pos] = res
        transfer[f"{prim}->{tgt}"] = out_t

    report = {
        "experiment": "OTR-08", "status": "EXPLORATORY", "version": OTR08_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "n_items": len(items), "n_families": len(fam_ids),
        "verbs_per_event": args.verbs_per_event, "constructions": list(PROBE_CONS), "langs": langs,
        "n_items_by_lang": {lg: len(v) for lg, v in items_by_lang.items()}, "transfer": transfer,
        "n_layers": L, "layers": use_layers, "positions": positions, "lam_rel": args.lam,
        "forward_seconds": round(fwd_seconds, 1), "seconds": round(time.time() - t0, 1), "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ltag = "" if langs == ["en"] else "_" + "-".join(langs)
    out = args.out_dir / f"otr08_{args.model.replace('/', '__')}_dev{ltag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    r = results[positions[0]]
    last = -1
    print(f"[{positions[0]}] E1 last layer:", {t: {c[:8]: r['E1'][t][c][last] for c in ors.MAIN_CONSTRUCTIONS} for t in TARGETS})
    print(f"[{positions[0]}] E2 active-trained agent probe, P(agent) at last layer:",
          {c[:8]: r["E2"]["active"][c][last] for c in PROBE_CONS})
    for k, v in transfer.items():
        print(f"E3 {k} [{positions[0]}] all-trained agent probe, P(agent) at last layer:",
              {c[:8]: v[positions[0]]["all_trained"][c][last] for c in v[positions[0]]["all_trained"]})
    try:
        _plot(report, out.with_suffix(".png"))
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = report["positions"][0]
    r = report["results"][pos]
    L = report["n_layers"]
    depth = [l / max(1, L - 1) for l in report["layers"]]
    cols = {"active": "#2a78d6", "passive": "#eb6834", "object_cleft": "#1baf7a", "by_cleft": "#eda100",
            "subject_cleft": "#6250d6", "relative_active": "#e87ba4", "relative_passive": "#008300",
            "ovs_clitic": "#e34948"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, t in zip(axes[:3], TARGETS):
        for c in report["constructions"]:
            ax.plot(depth, r["E1"][t][c], color=cols[c], lw=1.6, label=c)
        ax.set_title(f"E1: decode the {t} name", fontsize=9)
    for c in report["constructions"]:
        axes[3].plot(depth, r["E2"]["active"][c], color=cols[c], lw=1.6, label=c)
    axes[3].set_title("E2: agent probe trained on actives\nP(chooses agent) on each construction", fontsize=9)
    for ax in axes:
        ax.axhline(0.5, color="#898781", lw=0.8, ls="--")
        ax.set_xlabel("relative depth")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"accuracy at '{pos}'")
    axes[3].legend(fontsize=7)
    fig.suptitle(f"OTR-08 order to role: {report['model']} (held-out families and verbs; chance 0.5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
