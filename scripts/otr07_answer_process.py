"""OTR-07: how the answer is produced, layer by layer.

One forward pass per item over  [chat prompt] + [teacher-forced start of the
full-sentence answer, "<patient> was <past> by"]. Causal attention means the pass
contains two readout positions:

  bare  the last prompt token (where a bare-name answer is generated);
  by    the last token of "... was <past> by" (where the full-sentence answer must name
        the agent; Gemma gets this right even when its bare answer is wrong).

At every layer and both positions it records
  * lens     logit difference agent - patient after final norm + unembedding
             (first token of each name; leading space at the by slot);
  * attn     attention mass from the readout position to the agent-name tokens and to
             the patient-name tokens in the sentence, per head (eager attention);
  * probe    (every --probe-every layers) multinomial logistic probes for the identity of
             the agent and of the patient (24-way) from the residual stream, trained with
             5-fold cross-validation grouped by name-pair family; test accuracy is reported
             separately for items the model answers correctly vs wrongly (bare position).

"Correct" = the model's own bare logit difference > 0 for the asked role.
Questions come from --condition ROLE TEMPLATE (as in OTR-06). For patient questions the
lens sign is flipped so that > 0 always means "favours the correct name".

    python scripts/otr07_answer_process.py --model google/gemma-4-31B-it --chat-template \\
        --instruction "Reply with one name only." \\
        --condition agent "Who was someone {past} by?" --condition agent "Who was the {ing} done by?"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

OTR07_VERSION = "1.0.0"
_lens_spec = importlib.util.spec_from_file_location("otr06", ROOT / "scripts" / "otr06_logit_lens.py")
otr06 = importlib.util.module_from_spec(_lens_spec)
_lens_spec.loader.exec_module(otr06)


def span_tokens(offsets, start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def sentence_span(prompt: str, sentence: str, name: str) -> tuple[int, int]:
    """Character span of ``name`` inside the sentence part of the prompt (names occur once
    in the sentence and never in the question)."""
    s0 = prompt.index(sentence)
    k = sentence.index(name)
    return s0 + k, s0 + k + len(name)


def train_probe(X, y, n_classes, groups, folds=5, steps=300, lr=0.05, wd=1e-3):
    """Grouped k-fold multinomial logistic regression on GPU/CPU tensors.
    Returns per-item out-of-fold correctness (bool tensor)."""
    import torch

    uniq = sorted(set(groups))
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    f = torch.tensor([fold_of[g] for g in groups], device=X.device)
    ok = torch.zeros(len(y), dtype=torch.bool, device=X.device)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-5)
    for k in range(folds):
        tr, te = f != k, f == k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        W = torch.zeros(X.shape[1], n_classes, device=X.device, requires_grad=True)
        b = torch.zeros(n_classes, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xs[tr] @ W + b, y[tr])
            loss.backward()
            opt.step()
        with torch.no_grad():
            ok[te] = (Xs[te] @ W + b).argmax(1) == y[te]
    return ok


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--condition", nargs=2, action="append", metavar=("ROLE", "TEMPLATE"), required=True)
    p.add_argument("--instruction", default="")
    p.add_argument("--by-prefix", default="{patient} was {verb} by",
                   help='assistant text whose last token is the "by" slot; placeholders {agent} {patient} {verb}. '
                        'Control: "Someone was {verb} by" (no name given, so the agent must be looked up)')
    p.add_argument("--n-per-construction", type=int, default=None)
    p.add_argument("--probe-every", type=int, default=4, help="probe every k-th layer (0 = no probes)")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split", default="dev", help="dev or test0..test3 (test only after FREEZE.md)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    for role, _ in args.condition:
        if role not in ("agent", "patient"):
            p.error(f"role must be agent or patient, got {role!r}")

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg)
    import torch

    from relational_lab.instrumentation.hf_scoring import load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, answer_token, decoder_layers

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    try:
        model.set_attn_implementation("eager")
    except Exception:
        for cfg in (model.config, getattr(model.config, "text_config", None)):
            if cfg is not None:
                cfg._attn_implementation = "eager"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    layers = decoder_layers(model)
    L = len(layers)
    norm, head, cap = otr06.final_norm(model), model.get_output_embeddings(), otr06.softcap(model)
    probe_layers = list(range(0, L, args.probe_every)) + ([L - 1] if args.probe_every and (L - 1) % args.probe_every else []) \
        if args.probe_every else []
    names = sorted(ors.NAMES)
    name_ix = {n: i for i, n in enumerate(names)}

    prompts = otr06.build_prompts(args.seed, args.condition, args.instruction, args.n_per_construction,
                                  split=args.split)
    store: list = [None] * L
    handles = [layer.register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(i, _hidden(out).detach()))
               for i, layer in enumerate(layers)]
    records = []
    feats = defaultdict(list)  # (condition, layer) -> list of bare-position vectors (fp16)
    H = None
    attn_layers = None  # decoder-layer index of each returned attention map (hybrid models return fewer)
    attn_sum = defaultdict(lambda: None)  # (cond, construction, correct, pos, who) -> [L, H] sums
    attn_n = defaultdict(int)
    t0 = time.time()
    try:
        for ci, it in prompts:
            ids, offsets = _encode_one(tok, it.prompt, args.chat_template)
            prefix = args.by_prefix.format(agent=it.agent, patient=it.patient, verb=it.verb)
            if not args.chat_template:
                prefix = " " + prefix
            pre_ids = tok.encode(prefix, add_special_tokens=False)
            full = torch.tensor([ids + pre_ids], device=device)
            pos = {"bare": len(ids) - 1, "by": len(ids) + len(pre_ids) - 1}
            a_tok = span_tokens(offsets, *sentence_span(it.prompt, it.sentence, it.agent))
            p_tok = span_tokens(offsets, *sentence_span(it.prompt, it.sentence, it.patient))
            if not a_tok or not p_tok:
                raise RuntimeError(f"name tokens not found for {it.item_id}")
            with torch.no_grad():
                out = model(input_ids=full, output_attentions=True)
            att = out.attentions  # L x [1, H, T, T]
            if H is None:
                att = tuple(a for a in att if a is not None)
                H = att[0].shape[1]
                attn_layers = _attention_layers(model, L, len(att))
            else:
                att = tuple(a for a in att if a is not None)
            lead_bare = not args.chat_template
            ids_bare = (answer_token(tok, it.agent, lead_bare), answer_token(tok, it.patient, lead_bare))
            ids_by = (answer_token(tok, it.agent, True), answer_token(tok, it.patient, True))
            sign = 1.0 if it.role_asked == "agent" else -1.0
            rec = {"item_id": it.item_id, "family_id": getattr(it, "family_id", None), "condition": ci,
                   "construction": it.construction, "role": it.role_asked}
            with torch.no_grad():
                for pname, tids, sgn in (("bare", ids_bare, sign), ("by", ids_by, 1.0)):
                    h = torch.stack([store[l][0, pos[pname]] for l in range(L)])
                    lg = head(norm(h.to(head.weight.dtype))).float()
                    if cap:
                        lg = cap * torch.tanh(lg / cap)
                    rec[f"lens_{pname}"] = [round(x, 3) for x in (sgn * (lg[:, tids[0]] - lg[:, tids[1]])).tolist()]
                    am = torch.stack([a[0, :, pos[pname]] for a in att]).float()  # [L, H, T]
                    rec[f"attn_agent_{pname}"] = [round(x, 4) for x in am[:, :, a_tok].sum(-1).mean(-1).tolist()]
                    rec[f"attn_patient_{pname}"] = [round(x, 4) for x in am[:, :, p_tok].sum(-1).mean(-1).tolist()]
                    rec[f"_attn_{pname}"] = (am[:, :, a_tok].sum(-1).cpu(), am[:, :, p_tok].sum(-1).cpu())
                model_ld = sign * float(out.logits[0, pos["bare"], ids_bare[0]].float() - out.logits[0, pos["bare"], ids_bare[1]].float())
                rec["ld_model_bare"] = round(model_ld, 3)
                rec["correct"] = model_ld > 0
                rec["ld_model_by"] = round(float(out.logits[0, pos["by"], ids_by[0]].float() - out.logits[0, pos["by"], ids_by[1]].float()), 3)
                for l in probe_layers:
                    feats[(ci, l)].append(store[l][0, pos["bare"]].to(torch.float16).cpu())
            for pname in ("bare", "by"):
                aa, pp = rec.pop(f"_attn_{pname}")
                key = (ci, it.construction, rec["correct"], pname)
                attn_sum[key + ("agent",)] = aa if attn_sum[key + ("agent",)] is None else attn_sum[key + ("agent",)] + aa
                attn_sum[key + ("patient",)] = pp if attn_sum[key + ("patient",)] is None else attn_sum[key + ("patient",)] + pp
                attn_n[key] += 1
            records.append(rec)
    finally:
        for h_ in handles:
            h_.remove()

    # ---- probes: agent / patient identity at the bare position ----
    probes = {}
    for ci, (role, template) in enumerate(args.condition):
        recs = [r for r in records if r["condition"] == ci]
        if not probe_layers or len(recs) < 50:
            continue
        idx = {r["item_id"]: k for k, r in enumerate(recs)}
        items = {i.item_id: i for _, i in prompts if i.item_id in idx}
        groups = [items[r["item_id"]].family_id for r in recs]
        y_a = torch.tensor([name_ix[items[r["item_id"]].agent] for r in recs], device=device)
        y_p = torch.tensor([name_ix[items[r["item_id"]].patient] for r in recs], device=device)
        correct = torch.tensor([r["correct"] for r in recs], device=device)
        out_l = {}
        for l in probe_layers:
            X = torch.stack(feats[(ci, l)]).to(device).float()
            res = {}
            for who, y in (("agent", y_a), ("patient", y_p)):
                ok = train_probe(X, y, len(names), groups)
                res[who] = {"all": round(float(ok.float().mean()), 3),
                            "model_correct": round(float(ok[correct].float().mean()), 3) if correct.any() else None,
                            "model_wrong": round(float(ok[~correct].float().mean()), 3) if (~correct).any() else None}
            out_l[l] = res
        probes[f"{ci}|{role}|{template}"] = {"layers": out_l, "n_correct": int(correct.sum()), "n_wrong": int((~correct).sum())}

    # ---- aggregate ----
    agg = {}
    for ci, (role, template) in enumerate(args.condition):
        for cons in ors.MAIN_CONSTRUCTIONS:
            for corr in (True, False):
                rs = [r for r in records if r["condition"] == ci and r["construction"] == cons and r["correct"] == corr]
                if not rs:
                    continue
                key = f"{ci}|{role}|{template}|{cons}|{'correct' if corr else 'wrong'}"
                entry = {"n": len(rs)}
                for f in ("lens_bare", "lens_by", "attn_agent_bare", "attn_patient_bare", "attn_agent_by", "attn_patient_by"):
                    n_f = len(rs[0][f])
                    entry[f] = [round(sum(r[f][l] for r in rs) / len(rs), 4) for l in range(n_f)]
                for pname in ("bare", "by"):
                    k2 = (ci, cons, corr, pname)
                    if attn_n[k2]:
                        diff = (attn_sum[k2 + ("agent",)] - attn_sum[k2 + ("patient",)]) / attn_n[k2]
                        flat = [(float(diff[k, h]), attn_layers[k], h) for k in range(diff.shape[0]) for h in range(H)]
                        flat.sort()
                        entry[f"top_agent_heads_{pname}"] = [(l, h, round(v, 3)) for v, l, h in flat[::-1][:10]]
                        entry[f"top_patient_heads_{pname}"] = [(l, h, round(v, 3)) for v, l, h in flat[:10]]
                agg[key] = entry
    report = {
        "experiment": "OTR-07", "status": "EXPLORATORY" if args.split == "dev" else "TEST_SET_OPENED", "split": args.split, "version": OTR07_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "instruction": args.instruction, "by_prefix": args.by_prefix,
        "conditions": [{"role": r, "template": t} for r, t in args.condition], "n_layers": L, "n_heads": H,
        "attn_layers": attn_layers,
        "probe_layers": probe_layers, "seconds": round(time.time() - t0, 1),
        "aggregate": agg, "probes": probes, "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_instr" if args.instruction else ""
    if args.by_prefix != "{patient} was {verb} by":
        tag += "_byctl"
    out = args.out_dir / f"otr07_{args.model.replace('/', '__')}_{args.split}{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    for k, v in agg.items():
        if k.endswith("|active|correct") or k.endswith("|active|wrong"):
            print(f"  {k:70} n={v['n']:4}  lens bare {v['lens_bare'][-1]:+6.2f}  by {v['lens_by'][-1]:+6.2f}")
    for k, v in probes.items():
        last = v["layers"][max(v["layers"])]
        print(f"  probe {k}: last probed layer agent {last['agent']}  patient {last['patient']}")
    try:
        _plot(report, out.with_suffix(".png"))
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


def _attention_layers(model, n_layers, n_maps):
    """Decoder-layer indices of the attention maps a forward pass returns.

    Full-attention models return one map per layer.  Hybrid models (e.g. Qwen3.8 with
    gated-delta linear-attention layers) return maps only for their softmax-attention
    layers; config.layer_types says which those are.
    """
    if n_maps == n_layers:
        return list(range(n_layers))
    cfg = model.config
    types = getattr(cfg, "layer_types", None) or getattr(getattr(cfg, "text_config", None), "layer_types", None)
    if types and len(types) == n_layers:
        full = [i for i, t in enumerate(types) if "linear" not in str(t) and "mamba" not in str(t)
                and "conv" not in str(t)]
        if len(full) == n_maps:
            return full
    raise RuntimeError(f"model returned {n_maps} attention maps for {n_layers} layers and its "
                       f"config.layer_types does not say which layers they belong to: {types}")


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = report["n_layers"]
    depth = [l / max(1, L - 1) for l in range(L)]
    adepth = [l / max(1, L - 1) for l in (report.get("attn_layers") or range(L))]
    conds = report["conditions"]
    fig, axes = plt.subplots(3, len(conds), figsize=(4.4 * len(conds), 9.0), squeeze=False, sharex=True)
    for ci, c in enumerate(conds):
        for corr, ls in (("correct", "-"), ("wrong", "--")):
            key = f"{ci}|{c['role']}|{c['template']}|active|{corr}"
            e = report["aggregate"].get(key)
            if not e:
                continue
            axes[0][ci].plot(depth, e["lens_bare"], ls, color="#2a78d6", label=f"bare answer ({corr}, n={e['n']})")
            axes[0][ci].plot(depth, e["lens_by"], ls, color="#eb6834", label=f"'… by' slot ({corr})")
            axes[1][ci].plot(adepth, e["attn_agent_bare"], ls, color="#1baf7a", label=f"→ agent ({corr})")
            axes[1][ci].plot(adepth, e["attn_patient_bare"], ls, color="#e34948", label=f"→ patient ({corr})")
        pr = report["probes"].get(f"{ci}|{c['role']}|{c['template']}")
        if pr:
            ls_ = sorted(int(k) for k in pr["layers"])
            for who, col in (("agent", "#1baf7a"), ("patient", "#e34948")):
                for grp, st in (("model_correct", "-"), ("model_wrong", "--")):
                    ys = [pr["layers"][l if l in pr["layers"] else str(l)][who][grp] for l in ls_]
                    if all(v is not None for v in ys):
                        axes[2][ci].plot([l / max(1, L - 1) for l in ls_], ys, st, color=col, marker="o", ms=3,
                                         label=f"{who} id, items {grp.replace('model_', '')}")
        axes[0][ci].axhline(0, color="#898781", lw=0.8)
        axes[0][ci].set_title(c["template"].replace("{past}", "V-ed").replace("{ing}", "V-ing") + f" [{c['role']}]", fontsize=8.5)
        axes[2][ci].set_xlabel("relative depth")
    axes[0][0].set_ylabel("lens: correct − other")
    axes[1][0].set_ylabel("attention from bare\nanswer position")
    axes[2][0].set_ylabel("probe accuracy (24-way)")
    for r in range(3):
        axes[r][-1].legend(fontsize=6.5)
    fig.suptitle(f"OTR-07 answer process (active sentences): {report['model']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
