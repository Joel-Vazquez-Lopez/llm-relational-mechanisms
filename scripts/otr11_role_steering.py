"""OTR-11: is "who did it" computed from the SUBJECT code? Causal steering test.

OTR-08/OTR-10 found that at the end of a sentence the abstract, construction- and
language-general code is the grammatical SUBJECT, while the agent becomes readable only
at the answer of an agent question. Hypothesis: the answer to "who did it" is computed at
question time from the subject representation (plus voice). Test: edit the sentence
states along a probe direction and measure the answer.

Fit (families in folds 0-2): one forward pass per OTR-08 item (dev families x both role
assignments x verbs x the four main constructions); states at the non-name words from the
second name through the period ("is the subject now known" positions without the name
identity confound); per layer, ridge readouts from the residual stream to the identity of
the subject / agent / first name (one-hot over the 24 names, as OTR-10).

Steer (held-out families, folds 3-4; OTR-06/07 prompts with --condition questions and
--instruction): at ONE layer, add  beta * v  to the residual stream at the same word
positions, where  v = W[:, X] - W[:, Y]  moves the readout from the name Y that holds the
label to the name X that does not ("make the other name the subject / agent / first").
beta is set so that the readout margin moves by --strengths x (std of fitted margins);
"random" adds a random vector of the same norm as the subject edit.

Outcome: logit(agent) - logit(patient) of the first answer token at the answer position,
change from the unedited run, and the rate at which the chosen name flips.

Predictions for an AGENT question ("Who praised someone?"):
  * subject heuristic (answer = subject):      subject edit moves the answer toward X in
                                                actives AND passives.
  * subject + voice (agent = f(subject, voice)): toward X in actives (X becomes subject =
                                                agent) but toward the OTHER name in passives
                                                (X becomes subject = patient).
  * agent code used directly:                    agent edit moves the answer more than the
                                                subject edit; subject edit ~ random.

    python scripts/otr11_role_steering.py --model google/gemma-4-31B-it --chat-template \\
        --instruction "Reply with one name only."
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402

OTR11_VERSION = "1.0.0"
DIRECTIONS = ("subject", "agent", "first")
_WORD = re.compile(r"\w+|[^\w\s]")
DEFAULT_CONDITIONS = [("agent", "Who {past} someone?"), ("agent", "Who was someone {past} by?"),
                      ("patient", "Who did someone {lemma}?")]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


O8 = _load("o8", "otr08_role_probes.py")
O6 = _load("o6", "otr06_logit_lens.py")


def edit_spans(it):
    """Prompt-character spans of the non-name words from the second name through the period."""
    off = it.prompt.index(it.sentence)
    words = [(m.group(), m.start() + off, m.end() + off) for m in _WORD.finditer(it.sentence)]
    second = it.names[1].prompt_char_start
    k2 = next(k for k, (_, a, b) in enumerate(words) if a <= second < b)
    return [(a, b) for k, (w, a, b) in enumerate(words) if k >= k2 and w not in (it.agent, it.patient)]


def label_holder(it, label: str) -> str:
    return O8.target_name(it, label)


def fold_of(family: str, fams: list) -> int:
    return fams.index(family) % 5


def fit_readout(X, Y, lam_rel=1.0):
    """Ridge with intercept. Returns (W [d, k], mx [d], my [k])."""
    import torch

    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    G = Xc.T @ Xc
    lam = lam_rel * G.diagonal().mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), Xc.T @ (Y - my))
    return W, mx, my


def steer_vector(W, x_idx: int, y_idx: int, shift: float):
    """beta * (W[:, x] - W[:, y]) with beta chosen so the x-vs-y readout margin moves by ``shift``."""
    v = W[:, x_idx] - W[:, y_idx]
    return v * (shift / (v @ v).clamp_min(1e-12))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--condition", nargs=2, action="append", metavar=("ROLE", "TEMPLATE"), default=None)
    p.add_argument("--instruction", default="")
    p.add_argument("--layers", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                   help="comma list of relative depths (<=1) or layer indices (>1)")
    p.add_argument("--strengths", default="2,4,8", help="readout-margin shifts in units of the fitted margin std")
    p.add_argument("--n-per", type=int, default=24, help="test prompts per construction x condition")
    p.add_argument("--verbs-per-event", type=int, default=4)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    conditions = [tuple(c) for c in args.condition] if args.condition else DEFAULT_CONDITIONS
    strengths = [float(x) for x in args.strengths.split(",")]

    from relational_lab.instrumentation import chat as _chat
    chat_kwargs = _chat.configure(no_thinking=args.no_thinking, template_kwargs=args.template_kwarg)
    import torch

    from relational_lab.instrumentation.hf_scoring import load_lm, load_tokenizer
    from relational_lab.instrumentation.patching import _encode_one, _hidden, _replace_hidden, answer_token, decoder_layers

    tok = load_tokenizer(args.model)
    model = load_lm(args.model, dtype=getattr(torch, args.dtype))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    layers = decoder_layers(model)
    L = len(layers)
    steer_layers = sorted({min(L - 1, int(round(float(x) * (L - 1)))) if float(x) <= 1 else int(x)
                           for x in args.layers.split(",")})
    names = list(ors.NAMES)
    nidx = {n: j for j, n in enumerate(names)}
    nN = len(names)

    def token_ix(it):
        ids, offsets = _encode_one(tok, it.prompt, args.chat_template)
        ix = []
        for a, b in edit_spans(it):
            t = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
            ix.append(t[-1])
        return ids, ix

    # ---- fit readouts on folds 0-2 ----
    fit_items = [it for it in O8.build_items(args.seed, args.verbs_per_event, "en") if it.construction in ors.MAIN_CONSTRUCTIONS]
    fams = sorted({it.family_id for it in fit_items})
    fit_items = [it for it in fit_items if fold_of(it.family_id, fams) < 3]
    store = {}
    cur = {}

    def cap_hook(_m, _i, out, li):
        store[li] = _hidden(out)[0, cur["ix"]].detach().float()

    handles = [layers[li].register_forward_hook(lambda m, i, o, li=li: cap_hook(m, i, o, li)) for li in steer_layers]
    feats = {li: [] for li in steer_layers}
    rows = []  # (item, holder idx per direction)
    t0 = time.time()
    try:
        for it in fit_items:
            ids, ix = token_ix(it)
            if not ix:
                continue
            cur["ix"] = torch.tensor(ix, device=device)
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device))
            for li in steer_layers:
                feats[li].append(store[li].cpu())
            rows += [it] * len(ix)
    finally:
        for h in handles:
            h.remove()
    Y = torch.zeros((len(rows), nN * len(DIRECTIONS)))
    for j, d in enumerate(DIRECTIONS):
        Y[torch.arange(len(rows)), j * nN + torch.tensor([nidx[label_holder(it, d)] for it in rows])] = 1.0
    readout, margin_sd = {}, {}
    for li in steer_layers:
        X = torch.cat(feats[li]).to(device)
        W, mx, my = fit_readout(X, Y.to(device), args.lam)
        readout[li] = W
        S = (X - mx) @ W + my
        for j, d in enumerate(DIRECTIONS):
            blk = S[:, j * nN:(j + 1) * nN]
            hold = torch.tensor([nidx[label_holder(it, d)] for it in rows], device=device)
            oth = torch.tensor([nidx[it.patient if label_holder(it, d) == it.agent else it.agent] for it in rows],
                               device=device)
            r = torch.arange(len(rows), device=device)
            margin_sd[(li, d)] = float((blk[r, hold] - blk[r, oth]).std())
    fit_seconds = time.time() - t0
    del feats

    # ---- steer on held-out families (folds 3-4) ----
    prompts = O6.build_prompts(args.seed, conditions, args.instruction, None)
    per = defaultdict(int)
    test = []
    for ci, it in prompts:
        if it.family_id not in fams or fold_of(it.family_id, fams) < 3:
            continue
        key = (ci, it.construction)
        if per[key] >= args.n_per:
            continue
        per[key] += 1
        test.append((ci, it))
    lead = not args.chat_template
    edit = {}

    def add_hook(_m, _i, out):
        if edit.get("vec") is None:
            return out
        h = _hidden(out).clone()
        h[0, edit["ix"]] += edit["vec"].to(h.dtype)
        return _replace_hidden(out, h)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    cells = defaultdict(list)  # (ci, cons, layer, direction, strength) -> [(delta, flipped)]
    base_of = defaultdict(list)
    t1 = time.time()
    for n_done, (ci, it) in enumerate(test):
        ids, ix = token_ix(it)
        a_t, p_t = answer_token(tok, it.agent, lead), answer_token(tok, it.patient, lead)
        inp = torch.tensor([ids], device=device)

        def run():
            with torch.no_grad():
                lg = model(input_ids=inp).logits[0, -1].float()
            return float(lg[a_t] - lg[p_t])

        edit["vec"] = None
        base = run()
        base_of[(ci, it.construction)].append(base)
        edit["ix"] = torch.tensor(ix, device=device)
        for li in steer_layers:
            hnd = layers[li].register_forward_hook(add_hook)
            try:
                for s in strengths:
                    subj_norm = None
                    for d in DIRECTIONS + ("random",):
                        if d == "random":
                            r = torch.randn(readout[li].shape[0], generator=gen)
                            vec = r / r.norm() * subj_norm
                        else:
                            hold = label_holder(it, d)
                            x = it.patient if hold == it.agent else it.agent
                            vec = steer_vector(readout[li], nidx[x], nidx[hold], s * margin_sd[(li, d)]).cpu()
                            if d == "subject":
                                subj_norm = vec.norm()
                        edit["vec"] = vec.to(device)
                        val = run()
                        cells[(ci, it.construction, li, d, s)].append((val - base, (val > 0) != (base > 0)))
            finally:
                hnd.remove()
                edit["vec"] = None
        if n_done and n_done % 50 == 0:
            print(f"  {n_done}/{len(test)} prompts, {time.time() - t1:.0f}s", flush=True)
    steer_seconds = time.time() - t1

    results = {}
    for (ci, cons, li, d, s), v in cells.items():
        results.setdefault(str(ci), {}).setdefault(cons, {}).setdefault(d, {}).setdefault(str(s), {})[str(li)] = {
            "delta_agent_minus_patient": round(sum(x for x, _ in v) / len(v), 3),
            "flip_rate": round(sum(f for _, f in v) / len(v), 3), "n": len(v)}
    baseline = {str(ci): {cons: {"mean_agent_minus_patient": round(sum(v) / len(v), 3),
                                 "agent_preferred": round(sum(x > 0 for x in v) / len(v), 3), "n": len(v)}
                          for (c2, cons), v in base_of.items() if c2 == ci} for ci in range(len(conditions))}
    report = {
        "experiment": "OTR-11", "status": "EXPLORATORY", "version": OTR11_VERSION,
        "generator_version": ors.GENERATOR_VERSION, "model": args.model, "chat_template": args.chat_template,
        "chat_template_kwargs": chat_kwargs, "instruction": args.instruction,
        "conditions": [{"role": r, "template": t} for r, t in conditions], "n_layers": L,
        "steer_layers": steer_layers, "strengths": strengths, "directions": list(DIRECTIONS) + ["random"],
        "n_fit_states": len(rows), "n_test_prompts": len(test),
        "margin_sd": {f"{li}|{d}": round(v, 4) for (li, d), v in margin_sd.items()},
        "fit_seconds": round(fit_seconds, 1), "steer_seconds": round(steer_seconds, 1),
        "baseline": baseline, "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr11_{args.model.replace('/', '__')}_dev_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    smax = str(max(strengths))
    for ci, (role, tpl) in enumerate(conditions):
        print(f"[{role}] {tpl}")
        for cons in ors.MAIN_CONSTRUCTIONS:
            if cons not in results.get(str(ci), {}):
                continue
            r = results[str(ci)][cons]
            best = {d: max((v["delta_agent_minus_patient"] for v in r[d][smax].values()), key=abs) for d in r}
            print(f"   {cons:13s} base {baseline[str(ci)][cons]['mean_agent_minus_patient']:+6.2f}   "
                  f"largest delta(A-P) at strength {smax}: " + "  ".join(f"{d} {v:+6.2f}" for d, v in best.items()))
    try:
        _plot(report, out.with_suffix(".png"))
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


COLORS = {"subject": "#2a78d6", "agent": "#eb6834", "first": "#1baf7a", "random": "#898781"}


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = report["conditions"]
    cons = [c for c in ors.MAIN_CONSTRUCTIONS if c in report["results"]["0"]]
    smax = str(max(report["strengths"]))
    L = report["n_layers"]
    fig, axes = plt.subplots(len(cons), len(conds), figsize=(4.2 * len(conds), 2.6 * len(cons)),
                             squeeze=False, sharex=True, gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    for j, c in enumerate(conds):
        for i, k in enumerate(cons):
            ax = axes[i, j]
            r = report["results"][str(j)][k]
            for d in report["directions"]:
                lis = sorted(r[d][smax], key=int)
                ax.plot([int(l) / (L - 1) for l in lis], [r[d][smax][l]["delta_agent_minus_patient"] for l in lis],
                        color=COLORS[d], lw=2, marker="o", ms=4, label=d)
            ax.axhline(0, color="#c4c3be", lw=0.8)
            ax.set_title(f"{c['role']} q: {c['template']}\n{k} (base {report['baseline'][str(j)][k]['mean_agent_minus_patient']:+.1f})",
                         fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel("Δ logit(agent − patient)", fontsize=8)
            if i == len(cons) - 1:
                ax.set_xlabel("relative depth of the edit", fontsize=8)
    axes[0, 0].legend(fontsize=7, title=f"edit (strength {smax})", title_fontsize=7)
    fig.suptitle(f"OTR-11 {report['model']}: move the subject / agent / first label to the other name "
                 "(sentence words after the second name), held-out families", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
