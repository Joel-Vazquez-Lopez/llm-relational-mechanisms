"""OTR-12: are the OTR-09 heads NECESSARY, and is there one "fetch the agent" vector?

Builds on an OTR-09 result for the same model (``--otr09 results/otr09_<model>_….json``):
the same sentence pairs (base = failing agent question, source = working agent question),
the same family folds (0-2 = fit / ranking, 3-4 = evaluation), the OTR-09 head ranking and
the OTR-09 residual repair curve. All interventions act at the answer (last) position.

A. Knockout (necessity), evaluation items only. Mean-ablate the top-k OTR-09 heads
   (replace each head's output at the answer position by its mean over the fit items'
   runs of the same question) vs k random heads from the same layers, k in --top-k, in
     - the working question ("Who was the {ing} done by?")  and
     - the ordinary agent question ("Who {past} someone?").
   Outcome: logit(agent) - logit(patient) and the share of items still answered with the
   agent. If the heads are necessary, top-k ablation hurts much more than random-k.

B. One vector. At layer l, v_l = mean over FIT items of [resid(source) - resid(base)] at the
   answer position. On EVALUATION items (other families) add alpha * v_l to
     - the failing question: does one fixed vector repair it?   (repair, flips to agent)
     - the patient question "Who did someone {lemma}?": does it turn a patient question
       into an agent question?                                   (shift toward the agent)
   and subtract it from the working question (does -v break it?). Control: a random
   vector of the same norm. Transfer: v fitted on actives only, applied to the other
   constructions. Layers: --vector-layers, "auto" = around the first layer where OTR-09's
   residual repair reaches 0.5.

    python scripts/otr12_knockout_and_role_vector.py --model google/gemma-4-31B-it --chat-template \\
        --instruction "Reply with one name only." --otr09 results/otr09_google__gemma-4-31B-it_dev_20261118T164326.json
"""

from __future__ import annotations

import argparse
import importlib.util
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

OTR12_VERSION = "1.0.0"


def _o9():
    spec = importlib.util.spec_from_file_location("o9", ROOT / "scripts" / "otr09_answer_patching.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


O9 = _o9()


def ask(it, template, instruction):
    """Same item, another question (mirrors otr09.make_pairs)."""
    import dataclasses
    q = fill_template(template, it.verb, it.lemma)
    base_q = it.question
    prompt = it.prompt.replace(f"\nQuestion: {base_q}\n", f"\nQuestion: {q}\n")
    assert prompt.count(q) == 1, prompt
    if instruction and f"\n{instruction}\nAnswer:" not in prompt:
        prompt = prompt.replace("\nAnswer:", f"\n{instruction}\nAnswer:")
    return dataclasses.replace(it, question=q, prompt=prompt)


def auto_layers(o9_report, n_layers):
    res = o9_report["residual"].get("all")
    layers = o9_report["residual_layers"]
    j = next((i for i, v in enumerate(res["repair"]) if v is not None and v >= 0.5), None) if res else None
    if j is None:
        return [int(round(f * (n_layers - 1))) for f in (0.5, 0.6, 0.7, 0.8)]
    l50 = layers[j]
    return sorted({max(0, l50 - 4), max(0, l50 - 2), l50, min(n_layers - 1, l50 + 2), min(n_layers - 1, l50 + 6)})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--otr09", type=Path, required=True, help="OTR-09 result JSON for the same model")
    p.add_argument("--instruction", default=None, help="default: the OTR-09 run's instruction")
    p.add_argument("--normal-question", default="Who {past} someone?")
    p.add_argument("--patient-question", default="Who did someone {lemma}?")
    p.add_argument("--top-k", default="1,3,5,10,20")
    p.add_argument("--vector-layers", default="auto")
    p.add_argument("--alphas", default="1,2")
    p.add_argument("--n-per", type=int, default=None, help="default: as in the OTR-09 run (32)")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--assistant-prefix", default="")
    p.add_argument("--template-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    o9r = json.loads(args.otr09.read_text())
    instruction = o9r.get("instruction", "") if args.instruction is None else args.instruction
    ks = [int(x) for x in args.top_k.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

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
    attn = O9.attn_modules(layers)
    vec_layers = auto_layers(o9r, L) if args.vector_layers == "auto" else [int(x) for x in args.vector_layers.split(",")]
    lead = not args.chat_template

    cons = list(o9r["baseline"].keys())
    n_per = args.n_per or max(v["n"] for v in o9r["baseline"].values())
    triples = O9.make_pairs(args.seed, o9r["base_question"], o9r["source_question"], instruction, cons, n_per)
    fams = sorted({b.family_id for b, _, _ in triples})
    fold = {b.item_id: fams.index(b.family_id) % 5 for b, _, _ in triples}
    fit = [(b, s) for b, s, _ in triples if fold[b.item_id] < 3]
    ev = [(b, s) for b, s, _ in triples if fold[b.item_id] >= 3]
    ranked = [k for k in o9r["head_scan"]["top20"] if int(k.split(".")[0]) in attn]
    pool = [f"{li}.{h}" for li in o9r["head_layers"] if li in attn for h in range(attn[li][1])]
    head_layers = sorted({int(k.split(".")[0]) for k in pool})

    def run(it, res_add=None, head_set=None, cap_heads=False, cap_res=False):
        """res_add: {layer: vector added at the last position}; head_set: {layer: {head: replacement}}."""
        caps = {"head": {}, "res": {}}
        hs = []
        if cap_heads:
            for li in head_layers:
                hs.append(attn[li][0].register_forward_pre_hook(
                    lambda m, a, li=li: caps["head"].__setitem__(li, a[0][0, -1].detach().float().clone())))
        if cap_res:
            for li in vec_layers:
                hs.append(layers[li].register_forward_hook(
                    lambda m, i, o, li=li: caps["res"].__setitem__(li, _hidden(o)[0, -1].detach().float().clone())))
        for li, vec in (res_add or {}).items():
            def rhook(_m, _i, out, vec=vec):
                h = _hidden(out).clone()
                h[0, -1] += vec.to(h.dtype)
                return _replace_hidden(out, h)
            hs.append(layers[li].register_forward_hook(rhook))
        for li, heads in (head_set or {}).items():
            hd = attn[li][2]

            def phook(_m, a, heads=heads, hd=hd):
                x = a[0].clone()
                for h, v in heads.items():
                    x[0, -1, h * hd:(h + 1) * hd] = v.to(x.dtype)
                return (x,) + tuple(a[1:])
            hs.append(attn[li][0].register_forward_pre_hook(phook))
        try:
            ids = torch.tensor([_encode_one(tok, it.prompt, args.chat_template)[0]], device=device)
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        a_t, p_t = answer_token(tok, it.agent, lead), answer_token(tok, it.patient, lead)
        return float(lg[a_t] - lg[p_t]), caps

    t0 = time.time()
    # ---- fit pass: mean head outputs (per question) and the source - base residual difference ----
    qs = {"working": o9r["source_question"], "normal": args.normal_question}
    head_sum = {q: defaultdict(lambda: 0.0) for q in qs}
    diff_sum = {"all": defaultdict(lambda: 0.0), "active": defaultdict(lambda: 0.0)}
    n_diff = {"all": 0, "active": 0}
    for b, s in fit:
        _, cb = run(b, cap_res=True)
        _, cs = run(s, cap_res=True, cap_heads=True)
        for li in head_layers:
            head_sum["working"][li] = head_sum["working"][li] + cs["head"][li]
        _, cn = run(ask(b, qs["normal"], instruction), cap_heads=True)
        for li in head_layers:
            head_sum["normal"][li] = head_sum["normal"][li] + cn["head"][li]
        for g in ("all", "active"):
            if g == "all" or b.construction == "active":
                for li in vec_layers:
                    diff_sum[g][li] = diff_sum[g][li] + (cs["res"][li] - cb["res"][li])
                n_diff[g] += 1
    head_mean = {q: {li: head_sum[q][li] / len(fit) for li in head_layers} for q in qs}
    vec = {g: {li: diff_sum[g][li] / max(n_diff[g], 1) for li in vec_layers} for g in diff_sum}
    fit_seconds = time.time() - t0

    def ablation(keys, q):
        out = defaultdict(dict)
        for k in keys:
            li, h = (int(x) for x in k.split("."))
            hd = attn[li][2]
            out[li][h] = head_mean[q][li][h * hd:(h + 1) * hd].to(device)
        return dict(out)

    rng = random.Random(args.seed)
    knock = {q: {"baseline": [], **{f"top{k}": [] for k in ks}, **{f"rand{k}": [] for k in ks}} for q in qs}
    steer = defaultdict(list)  # key -> list of (ld, ld_base, ld_src, base_item_construction)
    gen = torch.Generator().manual_seed(args.seed)
    t1 = time.time()
    for n, (b, s) in enumerate(ev):
        items = {"working": s, "normal": ask(b, qs["normal"], instruction)}
        for q, it in items.items():
            knock[q]["baseline"].append((run(it)[0], b.construction))
            for k in ks:
                kk = min(k, len(ranked))
                knock[q][f"top{k}"].append((run(it, head_set=ablation(ranked[:kk], q))[0], b.construction))
                knock[q][f"rand{k}"].append((run(it, head_set=ablation(rng.sample(pool, kk), q))[0], b.construction))
        ld_b, ld_s = run(b)[0], run(s)[0]
        pq = ask(b, args.patient_question, instruction)
        ld_p = run(pq)[0]
        steer["base"].append((ld_b, ld_b, ld_s, b.construction))
        steer["patient_q_base"].append((ld_p, ld_p, ld_s, b.construction))
        for li in vec_layers:
            for a in alphas:
                for g in ("all", "active"):
                    v = vec[g][li].to(device) * a
                    steer[f"fail+v[{g}]|{li}|{a}"].append((run(b, res_add={li: v})[0], ld_b, ld_s, b.construction))
                v = vec["all"][li].to(device) * a
                r = torch.randn(v.shape[0], generator=gen).to(device)
                r = r / r.norm() * v.norm()
                steer[f"fail+rand|{li}|{a}"].append((run(b, res_add={li: r})[0], ld_b, ld_s, b.construction))
                steer[f"patient+v|{li}|{a}"].append((run(pq, res_add={li: v})[0], ld_p, ld_s, b.construction))
                steer[f"working-v|{li}|{a}"].append((run(s, res_add={li: -v})[0], ld_s, ld_b, b.construction))
        if n and n % 10 == 0:
            print(f"  eval {n}/{len(ev)}, {time.time() - t1:.0f}s", flush=True)

    def summ(vals, mode):
        """mode 'knock': mean ld and agent share; 'repair': gap-normalised repair + flips."""
        if not vals:
            return None
        if mode == "knock":
            return {"mean_ld": round(sum(v for v, _ in vals) / len(vals), 3),
                    "agent_share": round(sum(v > 0 for v, _ in vals) / len(vals), 3), "n": len(vals)}
        rep = [O9.repair(ld, lb, ls) for ld, lb, ls, _ in vals if abs(ls - lb) > 1.0]
        rep = [x for x in rep if x == x]
        wrong = [(ld, lb) for ld, lb, _, _ in vals if lb < 0]
        return {"repair": round(sum(rep) / len(rep), 3) if rep else None,
                "mean_shift": round(sum(ld - lb for ld, lb, _, _ in vals) / len(vals), 3),
                "agent_share": round(sum(ld > 0 for ld, _, _, _ in vals) / len(vals), 3),
                "flip_to_agent_among_wrong": round(sum(ld > 0 for ld, _ in wrong) / len(wrong), 3) if wrong else None,
                "n": len(vals)}

    knockout = {q: {k: summ(v, "knock") for k, v in d.items()} for q, d in knock.items()}
    steering = {}
    for key, vals in steer.items():
        steering[key] = {"all": summ(vals, "repair")}
        if "[active]" in key:  # transfer: v fitted on actives, evaluated on the other constructions
            steering[key]["non_active"] = summ([v for v in vals if v[3] != "active"], "repair")
    report = {
        "experiment": "OTR-12", "status": "EXPLORATORY", "version": OTR12_VERSION, "model": args.model,
        "otr09_file": args.otr09.name, "chat_template": args.chat_template, "chat_template_kwargs": chat_kwargs,
        "instruction": instruction, "base_question": o9r["base_question"], "source_question": o9r["source_question"],
        "normal_question": args.normal_question, "patient_question": args.patient_question, "n_layers": L,
        "vector_layers": vec_layers, "alphas": alphas, "top_k": ks, "ranked_heads": ranked,
        "n_fit": len(fit), "n_eval": len(ev), "vector_norms": {g: {str(li): round(float(vec[g][li].norm()), 3)
                                                                   for li in vec_layers} for g in vec},
        "knockout": knockout, "steering": steering,
        "fit_seconds": round(fit_seconds, 1), "eval_seconds": round(time.time() - t1, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"otr12_{args.model.replace('/', '__')}_dev_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(report))
    print(f"Wrote {out}")
    for q in qs:
        print(f"knockout [{q}]: base {knockout[q]['baseline']}  " +
              "  ".join(f"k{k}: top {knockout[q][f'top{k}']['agent_share']} / rand {knockout[q][f'rand{k}']['agent_share']}"
                        for k in ks))
    for li in vec_layers:
        a = max(alphas)
        g = lambda key: steering.get(f"{key}|{li}|{a}", {}).get("all") or {}
        print(f"vector L{li} alpha {a}: fail+v repair {g('fail+v[all]').get('repair')} flips {g('fail+v[all]').get('flip_to_agent_among_wrong')} | "
              f"rand {g('fail+rand').get('repair')} | patient+v agent share {g('patient+v').get('agent_share')} "
              f"(base {steering['patient_q_base']['all']['agent_share']}) | working-v agent share {g('working-v').get('agent_share')}")
    try:
        _plot(report, out.with_suffix(".png"))
        print(f"Wrote {out.with_suffix('.png')}")
    except Exception as exc:
        print(f"(no plot: {exc})")


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.3})
    ks = report["top_k"]
    ax = axes[0]
    for q, ls in (("working", "-"), ("normal", "--")):
        kd = report["knockout"][q]
        ax.plot(ks, [kd[f"top{k}"]["agent_share"] for k in ks], ls, color="#e34948", lw=2, marker="o", ms=4,
                label=f"top heads, {q} question")
        ax.plot(ks, [kd[f"rand{k}"]["agent_share"] for k in ks], ls, color="#898781", lw=2, marker="o", ms=4,
                label=f"random heads, {q} question")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("heads mean-ablated at the answer")
    ax.set_ylabel("share answered with the agent")
    ax.set_title("A. knockout (held-out families)", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    a = str(max(report["alphas"]))
    lis = report["vector_layers"]
    xs = [li / (report["n_layers"] - 1) for li in lis]
    for key, col, lab in (("fail+v[all]", "#2a78d6", "failing q + v: share agent"),
                          ("fail+rand", "#898781", "failing q + random: share agent"),
                          ("patient+v", "#eb6834", "patient q + v: share agent"),
                          ("working-v", "#e34948", "working q − v: share agent")):
        ax.plot(xs, [report["steering"][f"{key}|{li}|{float(a)}"]["all"]["agent_share"] for li in lis],
                color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(report["steering"]["base"]["all"]["agent_share"], color="#2a78d6", lw=0.8, ls=":")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("relative depth of the vector")
    ax.set_title(f"B. one averaged vector (alpha {a}); dotted = failing q unedited", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    for x in axes:
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.suptitle(f"OTR-12 {report['model']}", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
