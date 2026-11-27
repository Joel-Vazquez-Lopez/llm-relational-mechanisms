#!/usr/bin/env python3
"""Build the Order-to-Role results section (tables, figures, manifest) from archived result files.

Every number shown in papers/order-to-role/RESULTS.md and the linked pages is produced by this script
from the archived JSON outputs. Nothing is typed in by hand except the frozen settings, which are copied
from docs/order_to_role/FREEZE.md section 4 and checked against freeze_inputs.sha256.

Inputs
  --results DIR   a checkout of results/ from branch order-to-role/results-dev (pinned commit below)
  --code DIR      a checkout of the OTR code (branch order-to-role/stimuli or main after integration);
                  used for the frozen scorer and freeze_inputs.sha256
  --out DIR       papers/order-to-role/results

Usage (from the repository root)
  git worktree add /tmp/otr-results 25c6914e29473e4b1a1063600c48743f68b705c3
  python papers/order-to-role/analysis/build_results.py --results /tmp/otr-results/results \
      --code . --out papers/order-to-role/results \
      --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit <sha>

Requires numpy and matplotlib. Does not load any model and does not open any test split beyond
reading result files that already exist (test0 outputs are read only through the frozen scorer).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VERSION = "1.0.0"

# Frozen per-model settings, FREEZE.md section 4 (dev only; never re-chosen). Physical layer indices.
MODELS = [
    # label, slug, n_layers, role layer, "by" layer, handover
    ("Gemma 4 31B", "google__gemma-4-31B-it", 60, 28, 28, 40),
    ("Qwen3.8 27B", "Qwen__Qwen3.8-27B", 64, 29, 30, 54),
    ("phi-4 14B", "microsoft__phi-4", 40, 15, 8, 22),
    ("OLMo 3.1 32B", "allenai__Olmo-3.1-32B-Instruct", 64, 18, 8, 26),
    ("Granite 4.2 30B", "ibm-granite__granite-4.2-30b", 64, 37, 28, 52),
]
# Dev files the frozen settings were read from (FREEZE.md section 4 and freeze_inputs.sha256).
FREEZE_FILES = {
    "google__gemma-4-31B-it": dict(otr03="otr03_google__gemma-4-31B-it_dev_20261118T015021.json",
                                   otr09="otr09_google__gemma-4-31B-it_dev_20261118T164326.json",
                                   otr13="otr13_google__gemma-4-31B-it_dev_20261123T110732.json",
                                   otr14="otr14_google__gemma-4-31B-it_dev_20261126T070427.json"),
    "Qwen__Qwen3.8-27B": dict(otr03="otr03_Qwen__Qwen3.8-27B_dev_20261116T013154.json",
                              otr09="otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json",
                              otr13="otr13_Qwen__Qwen3.8-27B_dev_20261123T115050.json",
                              otr14="otr14_Qwen__Qwen3.8-27B_dev_20261126T064818.json"),
    "microsoft__phi-4": dict(otr03="otr03_microsoft__phi-4_dev_20261118T221350.json",
                             otr09="otr09_microsoft__phi-4_dev_20261119T093535.json",
                             otr13="otr13_microsoft__phi-4_dev_20261123T095044.json",
                             otr14="otr14_microsoft__phi-4_dev_20261126T062139.json"),
    "allenai__Olmo-3.1-32B-Instruct": dict(otr03="otr03_allenai__Olmo-3.1-32B-Instruct_dev_20261119T021052.json",
                                           otr09="otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json",
                                           otr13="otr13_allenai__Olmo-3.1-32B-Instruct_dev_20261123T102917.json",
                                           otr14="otr14_allenai__Olmo-3.1-32B-Instruct_dev_20261126T063042.json"),
    "ibm-granite__granite-4.2-30b": dict(otr03="otr03_ibm-granite__granite-4.2-30b_dev_20261119T153725.json",
                                         otr09="otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json",
                                         otr13="otr13_ibm-granite__granite-4.2-30b_dev_20261123T102154.json",
                                         otr14="otr14_ibm-granite__granite-4.2-30b_dev_20261126T063320.json"),
}
SHORT = {"google__gemma-4-31B-it": "Gemma", "Qwen__Qwen3.8-27B": "Qwen3.8", "microsoft__phi-4": "phi-4",
         "allenai__Olmo-3.1-32B-Instruct": "OLMo", "ibm-granite__granite-4.2-30b": "Granite"}
SCORER_MODEL = {"google__gemma-4-31B-it": "google/gemma-4-31B-it", "Qwen__Qwen3.8-27B": "Qwen/Qwen3.8-27B",
                "microsoft__phi-4": "microsoft/phi-4", "allenai__Olmo-3.1-32B-Instruct": "allenai/Olmo-3.1-32B-Instruct",
                "ibm-granite__granite-4.2-30b": "ibm-granite/granite-4.2-30b"}

INK, TEAL, MUTED, LIGHT, ORANGE, GRID = "#1F2430", "#0F766E", "#7A808C", "#B9B4AA", "#C2551F", "#ECE8E0"
plt.rcParams.update({
    "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#C9C4BA", "axes.labelcolor": "#4B5262", "xtick.color": "#7A808C", "ytick.color": "#4B5262",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
})

USED: dict[str, str] = {}  # file name -> sha256, every result file read


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(res: str, name: str) -> dict:
    path = os.path.join(res, name)
    USED[name] = sha256(path)
    with open(path) as f:
        return json.load(f)


def pick(res: str, pattern: str, keep=None, last=False) -> str | None:
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(res, pattern)) if p.endswith(".json"))
    if keep:
        files = [f for f in files if keep(f)]
    if not files:
        return None
    return files[-1] if last else files[0]


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def first_crossing(layers, values, thr=0.5):
    for L, v in zip(layers, values):
        if v is not None and v >= thr:
            return L
    return None


# ---------------------------------------------------------------------------------------------------------
def check_freeze_inputs(res: str, code: str) -> dict:
    path = os.path.join(code, "docs/order_to_role/freeze_inputs.sha256")
    out = {"file": "docs/order_to_role/freeze_inputs.sha256", "checked": 0, "matched": 0, "mismatched": []}
    if not os.path.exists(path):
        out["note"] = "not found"
        return out
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        h, name = line.split()
        out["checked"] += 1
        p = os.path.join(res, name)
        if os.path.exists(p) and sha256(p) == h:
            out["matched"] += 1
        else:
            out["mismatched"].append(name)
    return out


def behaviour(res: str) -> list[dict]:
    """Generation scoring (bare-name instruction): ordinary questions vs the failing question."""
    rows = []
    for label, slug, *_ in MODELS:
        core_fail = pick(res, f"otrgen_{slug}_dev_q-who-was-someone-past-by_2026112*")
        core_work = pick(res, f"otrgen_{slug}_dev_q-who-was-the-ing-done-by_2026112*")
        bx_ord = pick(res, f"otrgen_{slug}_dev_bx-transitive_2*")
        bx_fail = pick(res, f"otrgen_{slug}_dev_bx-transitive_q-who-was-someone-pp-by_2*")
        r = {"model": SHORT[slug]}
        if core_fail and core_work:
            r["core_working"] = load(res, core_work)["generation_accuracy"]
            r["core_failing"] = load(res, core_fail)["generation_accuracy"]
            r["core_files"] = f"{core_work}; {core_fail}"
        if bx_ord and bx_fail:
            o, fl = load(res, bx_ord), load(res, bx_fail)
            r["bx_ordinary"] = o["generation_accuracy"]
            r["bx_failing"] = fl["generation_accuracy"]
            r["bx_n"] = fl["n"]
            r["bx_files"] = f"{bx_ord}; {bx_fail}"
        rows.append(r)
    return rows


def role_code(res: str) -> tuple[list[dict], dict]:
    """OTR-03 transplant coefficients at the frozen role layer (core) and nearest grid layer (BX transitive)."""
    rows, curves = [], {}
    for label, slug, nL, role, by, ho in MODELS:
        d = load(res, FREEZE_FILES[slug]["otr03"])
        L = d["layers"]
        p = d["profile"]
        i = L.index(role)
        curves[slug] = {"layers": L, **{k: p[k]["coef"] for k in ("role", "function", "order")}}
        peak_i = int(np.nanargmax(p["role"]["coef"]))
        r = {"model": SHORT[slug], "n_layers": nL, "role_layer": role, "role_depth": role / nL,
             "core_role": p["role"]["coef"][i], "core_role_ci_lo": p["role"]["ci95"][i][0],
             "core_role_ci_hi": p["role"]["ci95"][i][1], "core_function": p["function"]["coef"][i],
             "core_order": p["order"]["coef"][i], "core_peak_layer": L[peak_i],
             "core_validity_null": d.get("left_context_null_role_slope"), "core_file": FREEZE_FILES[slug]["otr03"]}
        bx = pick(res, f"otr03_{slug}_dev_bx-transitive_2*", keep=lambda f: "_aq_" not in f)
        if bx:
            b = load(res, bx)
            bl = b["layers"]
            j = int(np.argmin([abs(x - role) for x in bl]))
            bp = b["profile"]
            r.update({"bx_layer": bl[j], "bx_role": bp["role"]["coef"][j], "bx_function": bp["function"]["coef"][j],
                      "bx_order": bp["order"]["coef"][j], "bx_file": bx})
        rows.append(r)
    return rows, curves


def pipeline(res: str) -> tuple[list[dict], dict]:
    """OTR-09 residual repair, OTR-13 'by' reverse damage, OTR-14 question swap vs answer swap (dev)."""
    rows, curves = [], {}
    for label, slug, nL, role, by, ho in MODELS:
        f = FREEZE_FILES[slug]
        d9 = load(res, f["otr09"])
        d13 = load(res, f["otr13"])
        d14 = load(res, f["otr14"])
        res_layers, rep = d9["residual_layers"], d9["residual"]["all"]["repair"]
        byd = d13["reverse_damage"]["by"]
        by_layers = sorted(int(k) for k in byd)
        s14 = d14["summary"]
        curves[slug] = {
            "residual_layers": res_layers, "residual_repair": rep,
            "by_layers": by_layers, "by_damage": [byd[str(k)] for k in by_layers],
            "p_layers": d14["layers"], "swap_question": s14["swap_question"]["mean"],
            "swap_question_lo": s14["swap_question"]["ci_low"], "swap_question_hi": s14["swap_question"]["ci_high"],
            "swap_answer": s14["swap_answer"]["mean"],
            "swap_answer_lo": s14["swap_answer"]["ci_low"], "swap_answer_hi": s14["swap_answer"]["ci_high"],
        }
        ce = d9["circuit_eval"]["by_k"]["20"]
        rows.append({"model": SHORT[slug], "n_layers": nL, "role_layer": role, "by_layer": by, "handover_layer": ho,
                     "role_depth": role / nL, "by_depth": by / nL, "handover_depth": ho / nL,
                     "handover_recomputed": first_crossing(res_layers, rep),
                     "by_peak_recomputed": by_layers[int(np.argmax([byd[str(k)] for k in by_layers]))],
                     "top20_repair_heldout": ce["top"]["repair"], "random20_repair_heldout": ce["random"]["repair"],
                     "top20_flip_to_correct": ce["top"]["flip_to_correct"], "n_heldout_items": ce["top"]["n"],
                     "otr09_file": f["otr09"], "otr13_file": f["otr13"], "otr14_file": f["otr14"]})
    return rows, curves


def transfer(res: str) -> list[dict]:
    """Frozen English top-20 heads (ranked on the core dev set) applied to other materials (dev)."""
    sets = [("bx-transitive", "English, new names and verbs (BX)"), ("es", "Spanish"), ("gl", "Galician"),
            ("ud-en", "Universal Dependencies (EWT/GUM)"),
            ("wiki-en-real", "Wikipedia, real names (WIKI-ROLES v0.2)"),
            ("wiki-en-neutral", "Wikipedia, neutral names (WIKI-ROLES v0.2)")]
    rows = []
    for label, slug, *_ in MODELS:
        for key, desc in sets:
            def ok(f, key=key):
                if "wiki" not in key:
                    return True
                d = json.load(open(os.path.join(res, f)))
                return (d.get("wiki") or {}).get("version", "").startswith("0.2") and not d.get("skip_scan")
            fname = pick(res, f"otr09_{slug}_dev_{key}_2*", keep=ok)
            if not fname:
                continue
            d = load(res, fname)
            t = d["transfer"]["by_k"]["20"]
            rows.append({"model": SHORT[slug], "set": key, "description": desc,
                         "top20_repair": t["all_items"]["repair"], "random20_repair": t["random_same_layers"]["repair"],
                         "n_items": t["all_items"]["n"], "overlap_with_fresh_top20": len(d["transfer"]["overlap_with_fresh_top20"]),
                         "handover_first_ge_0.5": first_crossing(d["residual_layers"], d["residual"]["all"]["repair"]),
                         "file": fname})
    return rows


def big_models(res: str) -> list[dict]:
    rows = []
    for slug, label in [("meta-llama__Llama-3.3-70B-Instruct", "Llama 3.3 70B"), ("Qwen__Qwen2.5-72B-Instruct", "Qwen2.5 72B")]:
        r = {"model": label, "status": "EXPLORATORY (dev only, not frozen)"}
        f9 = pick(res, f"otr09_{slug}_dev_2*")
        if f9:
            d = load(res, f9)
            r["n_layers"] = d["n_layers"]
            r["handover_first_ge_0.5"] = first_crossing(d["residual_layers"], d["residual"]["all"]["repair"])
            ce = (d.get("circuit_eval") or {}).get("by_k", {}).get("20")
            if ce:
                r["top20_repair_heldout"] = ce["top"]["repair"]
                r["random20_repair_heldout"] = ce["random"]["repair"]
            r["otr09_file"] = f9
        f3 = pick(res, f"otr03_{slug}_dev_2*", keep=lambda f: "_aq_" not in f)
        if f3:
            d = load(res, f3)
            p = d["profile"]
            k = int(np.nanargmax(p["role"]["coef"]))
            r.update({"role_peak_layer": d["layers"][k], "role_peak": p["role"]["coef"][k],
                      "function_at_peak": p["function"]["coef"][k], "order_at_peak": p["order"]["coef"][k],
                      "otr03_file": f3})
        rows.append(r)
    return rows


# Primary estimate and interval reported for each frozen claim (FREEZE.md section 3, FREEZE_XL.md section 3).
PRIMARY = {"C1": ("estimate", "ci"), "C2": ("role", "role_ci"), "C3": ("c2_minus_c3", "c2_minus_c3_ci"),
           "C4": ("first_crossing", None), "C5": ("top_minus_random", "top_minus_random_ci"),
           "C5b": ("top_minus_other_working", "ci"), "C6": ("damage", "ci"), "C7": ("share_agent_after_prefix", "ci"),
           "C8": ("damage", "ci")}


def confirmatory(res: str, code: str, out: str) -> list[dict]:
    """Run the frozen scorer (scripts/otr_freeze_score.py) on test0 for core, BX, es and gl."""
    scorer = os.path.join(code, "scripts/otr_freeze_score.py")
    rows = []
    os.makedirs(os.path.join(out, "confirmatory"), exist_ok=True)
    for stim in (None, "bx-transitive", "es", "gl"):
        tag = stim or "core"
        dest = os.path.join(out, "confirmatory", f"test0_{tag}.json")
        cmd = [sys.executable, scorer, "--results", res, "--split", "test0", "--out", dest]
        if stim:
            cmd += ["--stimuli", stim]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        d = json.load(open(dest))
        for slug in SHORT:
            cells = d["table"].get(SCORER_MODEL[slug], {})
            for claim, c in cells.items():
                if not isinstance(c, dict) or "verdict" not in c:
                    continue
                est_key, ci_key = PRIMARY.get(claim, (None, None))
                ci = c.get(ci_key) if ci_key else None
                rows.append({"set": tag, "claim": claim, "model": SHORT[slug], "verdict": c["verdict"],
                             "measure": est_key or "", "estimate": c.get(est_key) if est_key else None,
                             "ci_low": ci[0] if isinstance(ci, list) else None,
                             "ci_high": ci[1] if isinstance(ci, list) else None,
                             "control_other_failing": c.get("other_failing") if claim == "C5" else None})
    return rows


# --------------------------------------------------------------------------------------------- figures
def fig_behaviour(rows, path):
    rows = sorted(rows, key=lambda r: -r.get("bx_failing", 0))
    names = [r["model"] for r in rows]
    o = [100 * r["bx_ordinary"] for r in rows]
    f = [100 * r["bx_failing"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, o, w, color=TEAL, label="ordinary questions")
    ax.bar(x + w / 2 + 0.01, f, w, color=ORANGE, label="“Who was someone V-ed by?”")
    for i in range(len(rows)):
        ax.text(x[i] - w / 2, o[i] + 1.5, f"{o[i]:.0f}", ha="center", fontsize=10, color=INK)
        ax.text(x[i] + w / 2, f[i] + 1.5, f"{f[i]:.0f}", ha="center", fontsize=10, color=ORANGE)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Answers credited as correct")
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def fig_role_profiles(curves, path):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.3), sharey=True)
    for ax, (label, slug, nL, role, by, ho) in zip(axes, MODELS):
        c = curves[slug]
        dx = [l / nL for l in c["layers"]]
        ax.axhline(0, color="#D8D3CA", lw=1)
        ax.plot(dx, c["function"], color=LIGHT, lw=1.6, ls=(0, (2, 2)), label="grammatical function")
        ax.plot(dx, c["order"], color=MUTED, lw=1.6, label="order / focus")
        ax.plot(dx, c["role"], color=TEAL, lw=2.4, label="thematic role")
        ax.axvline(role / nL, color=TEAL, lw=1, ls=(0, (4, 3)), alpha=0.6)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["in", "50%", "out"])
        ax.grid(axis="y", color=GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Transplant coefficient (logits)")
    axes[2].set_xlabel("Relative depth (dashed: frozen role layer)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_pipeline(curves, path):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for ax, (label, slug, nL, role, by, ho) in zip(axes, MODELS):
        c = curves[slug]
        px = [l / nL for l in c["p_layers"]]
        ax.fill_between(px, c["swap_question_lo"], c["swap_question_hi"], color=MUTED, alpha=0.18, lw=0)
        ax.plot(px, c["swap_question"], color=MUTED, lw=2, label="swap the whole question")
        ax.fill_between(px, c["swap_answer_lo"], c["swap_answer_hi"], color=INK, alpha=0.15, lw=0)
        ax.plot(px, c["swap_answer"], color=INK, lw=2.2, label="patch the answer position")
        bx = [l / nL for l in c["by_layers"]]
        ax.plot(bx, c["by_damage"], color=ORANGE, lw=1.8, label="“by” from the failing question (damage)")
        ax.axvline(ho / nL, color=INK, lw=1, ls=(0, (4, 3)), alpha=0.6)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.65)
        ax.set_xticks([0, 0.5, 1], ["in", "50%", "out"])
        ax.grid(axis="y", color=GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Share of the working–failing gap")
    axes[2].set_xlabel("Relative depth (dashed: frozen handover layer)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_layer_order(rows, path):
    order = ["Gemma", "Qwen3.8", "Granite", "phi-4", "OLMo"]
    rows = sorted(rows, key=lambda r: order.index(r["model"]))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        R, B, H = 100 * r["role_depth"], 100 * r["by_depth"], 100 * r["handover_depth"]
        ax.plot([min(R, B), H], [y, y], color="#D8D3CA", lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([B], [y], s=90, facecolor="white", edgecolor=MUTED, linewidth=2, zorder=3)
        ax.scatter([R], [y], s=100, color=TEAL, zorder=4)
        ax.scatter([H], [y], s=100, color=INK, zorder=4)
        ax.text(H + 2, y, f"handover L{r['handover_layer']} of {r['n_layers']}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows][::-1])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["input", "25%", "50%", "75%", "output"])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("Relative depth (frozen dev layers; in Gemma the role and “by” peaks coincide at L28)")
    ax.scatter([], [], s=90, facecolor="white", edgecolor=MUTED, linewidth=2, label="“by” peak (OTR-13)")
    ax.scatter([], [], s=100, color=TEAL, label="role peak (OTR-03)")
    ax.scatter([], [], s=100, color=INK, label="answer handover (OTR-09)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9.5)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_transfer(pipe_rows, tr_rows, path):
    sets = [("core", "Core English\n(held-out dev families)"), ("bx-transitive", "BX English"), ("es", "Spanish"),
            ("gl", "Galician"), ("ud-en", "UD sentences"), ("wiki-en-real", "Wikipedia\nreal names"),
            ("wiki-en-neutral", "Wikipedia\nneutral names")]
    order = ["Gemma", "phi-4", "Granite", "OLMo", "Qwen3.8"]
    cols = {"Gemma": "#0F766E", "phi-4": "#3E9C8F", "Granite": "#7A808C", "OLMo": "#A86A3D", "Qwen3.8": "#C2551F"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for mi, m in enumerate(order):
        xs, ys, rs = [], [], []
        for si, (key, _) in enumerate(sets):
            if key == "core":
                r = next(r for r in pipe_rows if r["model"] == m)
                top, rnd = r["top20_repair_heldout"], r["random20_repair_heldout"]
            else:
                r = next((r for r in tr_rows if r["model"] == m and r["set"] == key), None)
                if r is None:
                    continue
                top, rnd = r["top20_repair"], r["random20_repair"]
            xs.append(si + (mi - 2) * 0.11)
            ys.append(top)
            rs.append(rnd)
        ax.scatter(xs, ys, s=55, color=cols[m], label=m, zorder=3)
        ax.scatter(xs, rs, s=22, facecolor="white", edgecolor=cols[m], zorder=3)
    ax.axhline(0, color="#D8D3CA", lw=1)
    ax.set_xticks(range(len(sets)), [s[1] for s in sets], fontsize=9.5)
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Repair by 20 heads (share of gap)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9.5)
    fig.text(0.99, -0.06, "filled: frozen top-20 heads · open: 20 random heads from the same layers",
             ha="right", fontsize=9, color=MUTED)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-commit", default="unknown")
    ap.add_argument("--code-commit", default="unknown")
    ap.add_argument("--skip-scorer", action="store_true")
    a = ap.parse_args()
    tables, figs = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figs, exist_ok=True)

    fz = check_freeze_inputs(a.results, a.code)
    beh = behaviour(a.results)
    write_csv(os.path.join(tables, "behaviour_dev.csv"), beh)
    rc, rc_curves = role_code(a.results)
    write_csv(os.path.join(tables, "role_code_dev.csv"), rc)
    pipe, pipe_curves = pipeline(a.results)
    write_csv(os.path.join(tables, "layers_and_heads_dev.csv"), pipe)
    tr = transfer(a.results)
    write_csv(os.path.join(tables, "head_transfer_dev.csv"), tr)
    big = big_models(a.results)
    write_csv(os.path.join(tables, "big_models_dev_exploratory.csv"), big)
    with open(os.path.join(tables, "layer_curves_dev.json"), "w") as f:
        json.dump({"role_code": rc_curves, "pipeline": pipe_curves}, f)
    if not a.skip_scorer:
        conf = confirmatory(a.results, a.code, a.out)
        write_csv(os.path.join(tables, "confirmatory_test0.csv"), conf)

    fig_behaviour(beh, os.path.join(figs, "fig1_failing_question.svg"))
    fig_role_profiles(rc_curves, os.path.join(figs, "fig2_role_code_by_layer.svg"))
    fig_pipeline(pipe_curves, os.path.join(figs, "fig3_question_to_answer.svg"))
    fig_layer_order(pipe, os.path.join(figs, "fig4_layer_order.svg"))
    fig_transfer(pipe, tr, os.path.join(figs, "fig5_head_transfer.svg"))

    manifest = {"script": "papers/order-to-role/analysis/build_results.py", "version": VERSION,
                "results_commit": a.results_commit, "code_commit": a.code_commit,
                "freeze_inputs_check": fz, "inputs_sha256": dict(sorted(USED.items())),
                "status_note": "Tables marked _dev are exploratory development results. confirmatory/ holds the "
                               "frozen scorer's test0 output (scripts/otr_freeze_score.py, FREEZE.md and FREEZE_XL.md)."}
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"freeze_inputs": {k: fz[k] for k in ("checked", "matched")}, "inputs": len(USED)}))


if __name__ == "__main__":
    main()
