#!/usr/bin/env python3
"""Build the Order-to-Role scientific supplement (figures, tables, manifest) from archived result files.

Companion to papers/order-to-role/analysis/build_results.py (same inputs, same frozen settings, same
figure style); it imports that script's frozen settings instead of repeating them. Every number in
papers/order-to-role/supplementary/*.md comes from the tables this script writes.

Inputs
  --results DIR   results/ from branch order-to-role/results-dev (pinned commit in MANIFEST.json)
  --code DIR      repository checkout holding the OTR code (src/relational_lab, docs/order_to_role)
  --main-results DIR  papers/order-to-role/results (holds confirmatory/test0_*.json written by the
                  frozen scorer through build_results.py; read here, never recomputed)
  --out DIR       papers/order-to-role/supplementary

Usage (from the repository root)
  git worktree add /tmp/otr-results 25c6914e29473e4b1a1063600c48743f68b705c3
  python papers/order-to-role/supplementary/analysis/build_supplement.py --results /tmp/otr-results/results \
      --code . --main-results papers/order-to-role/results --out papers/order-to-role/supplementary \
      --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit <sha>

Requires numpy and matplotlib. Loads no model, runs no experiment and reads no test-split output except
the frozen scorer's existing test0 summaries. Stimulus counts come from the generators (design facts:
family and item counts per split, no model outputs).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

VERSION = "1.0.0"

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("build_results", os.path.join(HERE, "..", "..", "analysis", "build_results.py"))
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)  # sets matplotlib to svg and the shared style

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS, FREEZE_FILES, SHORT = br.MODELS, br.FREEZE_FILES, br.SHORT
INK, TEAL, MUTED, LIGHT, ORANGE, GRID = br.INK, br.TEAL, br.MUTED, br.LIGHT, br.ORANGE, br.GRID
MODEL_COL = {"Gemma": "#0F766E", "phi-4": "#3E9C8F", "Granite": "#7A808C", "OLMo": "#A86A3D", "Qwen3.8": "#C2551F"}
CONS = ["active", "passive", "object_cleft", "by_cleft"]
CONS_LABEL = {"active": "active", "passive": "passive", "object_cleft": "object cleft", "by_cleft": "by-cleft",
              "ovs_clitic": "OVS clitic"}
CONS_COL = {"active": INK, "passive": TEAL, "object_cleft": ORANGE, "by_cleft": MUTED}
# Frozen Spanish/Galician handovers (FREEZE_XL.md section 4 / otr_freeze_score.py XL_HANDOVER), checked below.
XL_HANDOVER = {"es": {"Gemma": 46, "Qwen3.8": 50, "phi-4": 22, "OLMo": 24, "Granite": 52},
               "gl": {"Gemma": 46, "Qwen3.8": 50, "phi-4": 22, "OLMo": 26, "Granite": 52}}

USED: dict[str, str] = {}
pick, write_csv, first_crossing = br.pick, br.write_csv, br.first_crossing


def sha256(path):
    return br.sha256(path)


def load_res(res, name):
    d = br.load(res, name)
    USED[name] = br.USED[name]
    return d


plt.rcParams["svg.hashsalt"] = "order-to-role-supplement"  # stable element ids, so rebuilds are byte-identical


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def style(ax, ygrid=True):
    ax.tick_params(length=0)
    if ygrid:
        ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------------- S1 statistics
CLAIM_ORDER = ["C1", "C2", "C3", "C4", "C5", "C5b", "C6", "C7", "C8"]


def confirmatory_full(main_results):
    """Flatten the frozen scorer's test0 summaries: every reported field per set, claim and model."""
    rows, used = [], {}
    for tag in ("core", "bx-transitive", "es", "gl"):
        path = os.path.join(main_results, "confirmatory", f"test0_{tag}.json")
        used[f"papers/order-to-role/results/confirmatory/test0_{tag}.json"] = sha256(path)
        d = json.load(open(path))
        for model, cells in d["table"].items():
            slug = model.replace("/", "__")
            for claim in CLAIM_ORDER:
                c = cells.get(claim)
                if not isinstance(c, dict) or "verdict" not in c:
                    continue
                est_key, ci_key = br.PRIMARY.get(claim, (None, None))
                ci = c.get(ci_key) if ci_key else None
                r = {"set": tag, "claim": claim, "model": SHORT[slug], "verdict": c["verdict"],
                     "primary_measure": est_key, "estimate": c.get(est_key),
                     "ci_low": ci[0] if isinstance(ci, list) else None, "ci_high": ci[1] if isinstance(ci, list) else None,
                     "n_items": c.get("n_items", c.get("n_rows", c.get("n_wrong", c.get("n_wrong_bases")))),
                     "n_families": c.get("n_families"), "layer": c.get("layer", c.get("frozen")),
                     "scorer_version": d["scorer_version"], "n_boot": d["n_boot"], "seed": d["seed"]}
                extra = {k: v for k, v in c.items() if k not in (est_key, ci_key, "verdict", "n_items", "n_families",
                                                                 "layer", "file", "n_rows")}
                r["other_fields"] = json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                                    [round(x, 4) for x in v] if isinstance(v, list) else v)
                                                for k, v in extra.items()}, sort_keys=True)
                rows.append(r)
    return rows, used


def fig_forest(rows, path):
    claims = ["C1", "C2", "C3", "C5", "C5b", "C6", "C8", "C7"]
    titles = {"C1": "C1 working − failing (generation)", "C2": "C2 role coefficient", "C3": "C3 C2 − C3 role",
              "C5": "C5 top-20 − random heads", "C5b": "C5b top-20 − other working (XL)",
              "C6": "C6 “by” damage", "C8": "C8 “by” from patient question", "C7": "C7 share agent after prefix"}
    sets = [("core", "o", "English core"), ("bx-transitive", "s", "English BX"), ("es", "D", "Spanish"), ("gl", "^", "Galician")]
    order = ["Gemma", "Qwen3.8", "phi-4", "OLMo", "Granite"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for ax, claim in zip(axes.flat, claims):
        for si, (tag, marker, _) in enumerate(sets):
            for mi, m in enumerate(order):
                r = next((r for r in rows if r["set"] == tag and r["claim"] == claim and r["model"] == m), None)
                if r is None or r["estimate"] is None:
                    continue
                y = mi + (si - 1.5) * 0.17
                col = MODEL_COL[m]
                if r["ci_low"] is not None:
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=1.4, alpha=0.8)
                filled = r["verdict"] == "PASS"
                ax.scatter([r["estimate"]], [y], marker=marker, s=34, zorder=3,
                           facecolor=col if filled else "white", edgecolor=col, linewidth=1.4)
        ax.axvline(0, color="#D8D3CA", lw=1)
        if claim == "C7":
            ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(range(len(order)), order if claim in ("C1", "C5b") else [""] * len(order))
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_title(titles[claim], loc="left", fontsize=10.5)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    for tag, marker, label in sets:
        axes.flat[0].scatter([], [], marker=marker, s=34, facecolor=INK, edgecolor=INK, label=label)
    axes.flat[0].scatter([], [], marker="o", s=34, facecolor="white", edgecolor=INK, label="open marker: FAIL")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=9.5)
    fig.text(0.99, -0.02, "test0, frozen scorer v1.1.0, 2000 family-bootstrap resamples (seed 2026); bars: 95% CI. "
             "C4 (a layer) is in the table only.", ha="right", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------------------- S2 linguistic controls
QUESTION_SET_DATE = "20261126"  # one battery: same instruction, n, template and scorer for every wording


def question_wording(res):
    """26 Nov question-wording battery (otr_generate_check, dev, 'Reply with one name only.', chat template)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(res, f"otrgen_*_dev_q-*_{QUESTION_SET_DATE}T*.json"))):
        name = os.path.basename(f)
        m = re.match(r"otrgen_(.+?)_dev_q-(.+)_(\d{8}T\d{6})\.json", name)
        slug, qkey, _ = m.groups()
        if slug not in SHORT:
            continue
        d = load_res(res, name)
        r = {"model": SHORT[slug], "question_key": qkey, "question_template": d["question_template"],
             "asks_for": d["question_role"], "instruction": d["instruction"], "n": d["n"],
             "accuracy": d["generation_accuracy"], "named_rate": d.get("named_rate"), "file": name}
        for c in CONS:
            k = f"{c}/{d['question_role']}/custom"
            r[f"acc_{c}"] = d["cells"].get(k)
        rows.append(r)
    return rows


WORDING_ORDER = [
    ("who-was-someone-past-by", "Who was someone V-ed by?  (failing)"),
    ("by-whom-was-someone-past", "By whom was someone V-ed?"),
    ("who-was-the-ing-done-by", "Who was the V-ing done by?  (working)"),
    ("who-past-someone", "Who V-ed someone?"),
    ("who-was-patient-past-by", "Who was <patient> V-ed by?"),
    ("by-whom-was-patient-past", "By whom was <patient> V-ed?"),
    ("who-past-patient", "Who V-ed <patient>?"),
    ("who-was-past-by-agent", "Who was V-ed by <agent>?  (asks patient)"),
    ("who-did-agent-lemma", "Who did <agent> V?  (asks patient)"),
]


def fig_question_wording(rows, path):
    order = ["Gemma", "Granite", "OLMo", "Qwen3.8", "phi-4"]
    keys = [k for k, _ in WORDING_ORDER if any(r["question_key"] == k for r in rows)]
    labels = dict(WORDING_ORDER)
    M = np.full((len(keys), len(order)), np.nan)
    for r in rows:
        if r["question_key"] in keys:
            M[keys.index(r["question_key"]), order.index(r["model"])] = r["accuracy"]
    fail = {r["model"]: r for r in rows if r["question_key"] == "who-was-someone-past-by"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
    cmap = plt.get_cmap("Greys")
    for i in range(len(keys)):
        for j in range(len(order)):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=cmap(0.12 + 0.75 * (1 - v)) if not np.isnan(v) else "white"))
            ax.text(j + 0.48, i + 0.48, f"{100 * v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if v < 0.6 else INK)
    ax.set_xlim(0, len(order))
    ax.set_ylim(len(keys), 0)
    ax.set_xticks(np.arange(len(order)) + 0.48, order)
    ax.set_yticks(np.arange(len(keys)) + 0.46, [labels[k] for k in keys], fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Accuracy (%) by question wording, 4 constructions pooled", loc="left", fontsize=11)
    x = np.arange(len(CONS))
    for j, m in enumerate(order):
        r = fail.get(m)
        if not r:
            continue
        ax2.plot(x, [r[f"acc_{c}"] for c in CONS], marker="o", color=MODEL_COL[m], lw=1.8, label=m)
    ax2.set_xticks(x, [CONS_LABEL[c] for c in CONS])
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax2.set_title("“Who was someone V-ed by?” by construction", loc="left", fontsize=11)
    style(ax2)
    ax2.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, path)


def bx_blocks(res):
    """BX ordinary questions vs the stranded-'by' agent question, per block (dev, exploratory)."""
    blocks = [("bx-transitive", "q-who-was-someone-pp-by"), ("bx-ditransitive", "q-who-was-someone-given-theme-by"),
              ("bx-psych", "q-who-was-someone-pp-by")]
    rows = []
    for _, slug, *_ in MODELS:
        for block, q in blocks:
            o = pick(res, f"otrgen_{slug}_dev_{block}_2*")
            fl = pick(res, f"otrgen_{slug}_dev_{block}_{q}_2*")
            if not (o and fl):
                continue
            do, dfl = load_res(res, o), load_res(res, fl)
            r = {"model": SHORT[slug], "block": block, "ordinary_accuracy": do["generation_accuracy"], "n_ordinary": do["n"],
                 "failing_question": dfl["question_template"], "failing_accuracy": dfl["generation_accuracy"],
                 "n_failing": dfl["n"], "failing_by_construction": json.dumps({k.split("/")[0]: round(v, 3) for k, v in dfl["cells"].items()}),
                 "files": f"{o}; {fl}"}
            rows.append(r)
    return rows


def fig_bx_blocks(rows, path):
    order = ["Gemma", "Granite", "Qwen3.8", "OLMo", "phi-4"]
    blocks = [("bx-transitive", "transitive"), ("bx-ditransitive", "ditransitive"), ("bx-psych", "psych verbs")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (b, title) in zip(axes, blocks):
        x = np.arange(len(order))
        o = [next((r["ordinary_accuracy"] for r in rows if r["model"] == m and r["block"] == b), np.nan) for m in order]
        f = [next((r["failing_accuracy"] for r in rows if r["model"] == m and r["block"] == b), np.nan) for m in order]
        ax.bar(x - 0.19, o, 0.36, color=TEAL, label="ordinary questions")
        ax.bar(x + 0.19, f, 0.36, color=ORANGE, label="stranded “by” agent question")
        ax.set_xticks(x, order, fontsize=9.5)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, path)


def prompt_formats(res):
    rows = []
    for _, slug, *_ in MODELS:
        f = pick(res, f"otr15_{slug}_dev_2*", last=True)
        if not f:
            continue
        d = load_res(res, f)
        for fmt in d["formats"]:
            rows.append({"model": SHORT[slug], "format": fmt, "working": d["accuracy"].get(f"{fmt}|working|all"),
                         "failing": d["accuracy"].get(f"{fmt}|failing|all"), "gap": d["working_minus_failing"].get(fmt),
                         "n_per_question": d["n_per_question"], "chat_template_kwargs": json.dumps(d["chat_template_kwargs"]),
                         "file": f})
    return rows


def fig_prompt_formats(rows, path):
    order = ["Gemma", "Granite", "OLMo", "Qwen3.8", "phi-4"]
    fmts = []
    for r in rows:
        if r["format"] not in fmts:
            fmts.append(r["format"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for fi, fmt in enumerate(fmts):
        for mi, m in enumerate(order):
            r = next((r for r in rows if r["model"] == m and r["format"] == fmt), None)
            if r:
                ax.scatter([fi + (mi - 2) * 0.12], [r["gap"]], s=44, color=MODEL_COL[m], label=m if fi == 0 else None, zorder=3)
    ax.axhline(0, color="#D8D3CA", lw=1)
    ax.set_xticks(range(len(fmts)), fmts)
    ax.set_ylabel("working − failing accuracy")
    style(ax)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9.5)
    save(fig, path)


def stimulus_inventory(code):
    """Design facts from the generators: names, verbs, families and items per split (no model outputs)."""
    sys.path.insert(0, os.path.join(code, "src"))
    from relational_lab.experiments import broad_stimuli as bx
    from relational_lab.experiments import multilingual_stimuli as xl
    from relational_lab.experiments import order_role_stimuli as ors

    rows = []
    items = ors.generate_items()
    corr = ors.label_correlations([i for i in items if i.construction in CONS])

    def split_of(i):
        return i.split if i.split != "test" else f"test{i.test_set}"

    def add(material, version, its, note=""):
        fam = defaultdict(set)
        n_items = Counter()
        for i in its:
            fam[split_of(i)].add(i.family_id)
            n_items[split_of(i)] += 1
        rows.append({"material": material, "generator_version": version,
                     "n_names": len({n.name for i in its for n in i.names}), "n_verbs": len({i.lemma for i in its}),
                     "constructions": " ".join(sorted({i.construction for i in its})),
                     "families_total": len({i.family_id for i in its}),
                     "families_by_split": json.dumps({k: len(v) for k, v in sorted(fam.items())}),
                     "items_by_split": json.dumps(dict(sorted(n_items.items()))), "note": note})

    add("English core (OTR)", ors.GENERATOR_VERSION, items,
        "label correlations over the 4 main constructions: " + json.dumps({k: round(v, 3) for k, v in corr.items()}))
    for lang in ("es", "gl"):
        add({"es": "Spanish (XL)", "gl": "Galician (XL)"}[lang], xl.XL_GENERATOR_VERSION, xl.generate_items(lang))
    for block in bx.BLOCKS:
        add(f"English BX {block}", bx.BX_GENERATOR_VERSION, bx.generate_items(block, split=None))
    return rows


# ------------------------------------------------------------------------------- S3 layerwise
def role_code_layers(res):
    rows = []
    curves = {}
    for label, slug, nL, role, by, ho in MODELS:
        d = load_res(res, FREEZE_FILES[slug]["otr03"])
        p = d["profile"]
        for k, L in enumerate(d["layers"]):
            rows.append({"model": SHORT[slug], "material": "core", "layer": L, "depth": L / nL,
                         "role": p["role"]["coef"][k], "role_ci_lo": p["role"]["ci95"][k][0], "role_ci_hi": p["role"]["ci95"][k][1],
                         "function": p["function"]["coef"][k], "order": p["order"]["coef"][k],
                         "function_ci_lo": p["function"]["ci95"][k][0], "function_ci_hi": p["function"]["ci95"][k][1],
                         "order_ci_lo": p["order"]["ci95"][k][0], "order_ci_hi": p["order"]["ci95"][k][1],
                         "validity_null": (d.get("left_context_null_role_slope") or [None] * len(d["layers"]))[k],
                         "file": FREEZE_FILES[slug]["otr03"]})
        curves[slug] = {"core": (d["layers"], p)}
        bxf = pick(res, f"otr03_{slug}_dev_bx-transitive_2*", keep=lambda f: "_aq_" not in f)
        if bxf:
            b = load_res(res, bxf)
            bp = b["profile"]
            curves[slug]["bx"] = (b["layers"], bp)
            for k, L in enumerate(b["layers"]):
                rows.append({"model": SHORT[slug], "material": "bx-transitive", "layer": L, "depth": L / nL,
                             "role": bp["role"]["coef"][k], "role_ci_lo": bp["role"]["ci95"][k][0], "role_ci_hi": bp["role"]["ci95"][k][1],
                             "function": bp["function"]["coef"][k], "order": bp["order"]["coef"][k],
                             "function_ci_lo": bp["function"]["ci95"][k][0], "function_ci_hi": bp["function"]["ci95"][k][1],
                             "order_ci_lo": bp["order"]["ci95"][k][0], "order_ci_hi": bp["order"]["ci95"][k][1],
                             "validity_null": None, "file": bxf})
    return rows, curves


def fig_role_core_bx(curves, path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey="row")
    for j, (label, slug, nL, role, by, ho) in enumerate(MODELS):
        L, p = curves[slug]["core"]
        x = [l / nL for l in L]
        for row, (k, col, name) in enumerate([("role", TEAL, "thematic role"), ("order", MUTED, "order / focus")]):
            ax = axes[row, j]
            lo = [c[0] for c in p[k]["ci95"]]
            hi = [c[1] for c in p[k]["ci95"]]
            ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(x, p[k]["coef"], color=col, lw=2.2, label=f"core: {name} (95% CI)")
            if "bx" in curves[slug]:
                bL, bp = curves[slug]["bx"]
                ax.plot([l / nL for l in bL], bp[k]["coef"], color=INK, lw=1.4, ls=(0, (3, 2)), marker="o", ms=3,
                        label=f"BX transitive: {name}")
            ax.axhline(0, color="#D8D3CA", lw=1)
            ax.axvline(role / nL, color=TEAL, lw=1, ls=(0, (4, 3)), alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .5, 1], ["in", "50%", "out"])
            style(ax)
            if row == 0:
                ax.set_title(label, loc="left", fontsize=11)
            if j == 0:
                ax.set_ylabel(["role coefficient", "order coefficient"][row])
                ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.5, -0.01, "Relative depth (dashed vertical: frozen role layer). BX transitive uses new names and verbs.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, path)


def residual_by_construction(res):
    rows = []
    curves = {}
    for label, slug, nL, role, by, ho in MODELS:
        d = load_res(res, FREEZE_FILES[slug]["otr09"])
        Ls = d["residual_layers"]
        curves[slug] = {}
        for c in ["all"] + CONS:
            rep = d["residual"][c]["repair"]
            curves[slug][c] = rep
            for L, v in zip(Ls, rep):
                rows.append({"model": SHORT[slug], "construction": c, "layer": L, "depth": L / nL, "repair": v,
                             "n_items": d["residual"][c]["n"], "file": FREEZE_FILES[slug]["otr09"],
                             "base_agent_preferred": None, "source_agent_preferred": None})
        for c in CONS:
            b = d["baseline"][c]
            rows.append({"model": SHORT[slug], "construction": c, "layer": "baseline", "depth": None,
                         "repair": None, "n_items": b["n"], "file": FREEZE_FILES[slug]["otr09"],
                         "base_agent_preferred": b["base_agent_preferred"], "source_agent_preferred": b["source_agent_preferred"]})
        curves[slug]["_layers"] = Ls
    return rows, curves


def fig_residual_by_construction(curves, path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, (label, slug, nL, role, by, ho) in zip(axes, MODELS):
        x = [l / nL for l in curves[slug]["_layers"]]
        for c in CONS:
            ax.plot(x, curves[slug][c], color=CONS_COL[c], lw=1.6, label=CONS_LABEL[c])
        ax.plot(x, curves[slug]["all"], color=TEAL, lw=3, alpha=0.35, label="all items")
        ax.axhline(0.5, color="#D8D3CA", lw=1, ls=(0, (3, 3)))
        ax.axvline(ho / nL, color=INK, lw=1, ls=(0, (4, 3)), alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([0, .5, 1], ["in", "50%", "out"])
        ax.set_title(label, loc="left", fontsize=11)
        style(ax)
    axes[0].set_ylabel("residual repair at the answer position")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.09))
    fig.text(0.5, -0.04, "Relative depth (dashed vertical: frozen handover layer; dotted horizontal: repair = .5)",
             ha="center", fontsize=9.5, color=MUTED)
    save(fig, path)


def heads(res):
    k_rows, h_rows = [], []
    for label, slug, nL, role, by, ho in MODELS:
        f = FREEZE_FILES[slug]["otr09"]
        d = load_res(res, f)
        for k, v in sorted(d["circuit_eval"]["by_k"].items(), key=lambda kv: int(kv[0])):
            k_rows.append({"model": SHORT[slug], "k": int(k), "top_repair": v["top"]["repair"], "random_repair": v["random"]["repair"],
                           "top_flip_to_correct": v["top"]["flip_to_correct"], "random_flip_to_correct": v["random"]["flip_to_correct"],
                           "n_heldout_items": v["top"]["n"], "file": f})
        scores = d["head_scan"]["scores"]
        for rank, h in enumerate(d["head_scan"]["top20"], 1):
            L, H = h.split(".")
            h_rows.append({"model": SHORT[slug], "rank": rank, "head": h, "layer": int(L), "head_index": int(H),
                           "depth": int(L) / nL, "scan_repair": scores.get(f"{int(L)}.{int(H)}", scores.get(h)),
                           "handover_layer": ho, "n_rank_items": d["head_scan"]["n_rank_items"],
                           "scanned_layers": f"{min(d['head_layers'])}-{max(d['head_layers'])}", "file": f})
    return k_rows, h_rows


def fig_heads(k_rows, h_rows, path):
    order = ["Gemma", "phi-4", "Granite", "OLMo", "Qwen3.8"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    for m in order:
        rs = sorted((r for r in k_rows if r["model"] == m), key=lambda r: r["k"])
        ax.plot([r["k"] for r in rs], [r["top_repair"] for r in rs], marker="o", color=MODEL_COL[m], lw=2, label=m)
        ax.plot([r["k"] for r in rs], [r["random_repair"] for r in rs], color=MODEL_COL[m], lw=1, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20], ["1", "3", "5", "10", "20"])
    ax.set_xlabel("number of heads patched (k)")
    ax.set_ylabel("repair on held-out dev families")
    ax.set_title("Top-k heads (solid) vs k random heads (dashed)", loc="left", fontsize=11)
    ax.axhline(0, color="#D8D3CA", lw=1)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    for i, m in enumerate(order):
        rs = [r for r in h_rows if r["model"] == m]
        y = len(order) - 1 - i
        jitter = np.linspace(-0.22, 0.22, len(rs))
        ax2.scatter([r["depth"] for r in rs], y + jitter[np.argsort([r["rank"] for r in rs])], s=[60 - 2 * r["rank"] for r in rs],
                    color=MODEL_COL[m], alpha=0.85, zorder=3)
        ho = rs[0]["handover_layer"]
        nL = next(n for lab, s, n, *_ in MODELS if SHORT[s] == m)
        ax2.plot([ho / nL] * 2, [y - 0.35, y + 0.35], color=INK, lw=1.6)
        lo, hi = (int(v) for v in rs[0]["scanned_layers"].split("-"))
        ax2.add_patch(plt.Rectangle((0, y - 0.4), lo / nL, 0.8, color="#ECE8E0", lw=0, zorder=1))
    ax2.set_yticks(range(len(order)), order[::-1])
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, .25, .5, .75, 1], ["input", "25%", "50%", "75%", "output"])
    ax2.set_title("Frozen top-20 heads (grey: layers not scanned; bar: handover)", loc="left", fontsize=11)
    ax2.tick_params(length=0)
    ax2.grid(axis="x", color=GRID)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, path)


def big_model_curves(res):
    rows, curves = [], {}
    for slug, label in [("meta-llama__Llama-3.3-70B-Instruct", "Llama 3.3 70B"), ("Qwen__Qwen2.5-72B-Instruct", "Qwen2.5 72B")]:
        f9 = pick(res, f"otr09_{slug}_dev_2*")
        if f9:
            d = load_res(res, f9)
            nL = d["n_layers"]
            curves[label] = {"nL": nL, "res": (d["residual_layers"], d["residual"]["all"]["repair"])}
            for L, v in zip(d["residual_layers"], d["residual"]["all"]["repair"]):
                rows.append({"model": label, "measure": "residual_repair", "layer": L, "depth": L / nL, "value": v, "file": f9})
        f3 = pick(res, f"otr03_{slug}_dev_2*", keep=lambda f: "_aq_" not in f)
        if f3:
            d = load_res(res, f3)
            nL = curves[label]["nL"]
            curves[label]["otr03"] = (d["layers"], d["profile"])
            for k, L in enumerate(d["layers"]):
                for meas in ("role", "function", "order"):
                    rows.append({"model": label, "measure": meas, "layer": L, "depth": L / nL,
                                 "value": d["profile"][meas]["coef"][k], "file": f3})
    return rows, curves


def fig_big_models(curves, path):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.6))
    for label, col in [("Llama 3.3 70B", TEAL), ("Qwen2.5 72B", ORANGE)]:
        c = curves.get(label)
        if not c:
            continue
        L, v = c["res"]
        ax.plot([l / c["nL"] for l in L], v, color=col, lw=2.2, label=label)
    ax.axhline(0.5, color="#D8D3CA", lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ["in", "50%", "out"])
    ax.set_title("Residual repair at the answer position", loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False)
    c = curves.get("Llama 3.3 70B")
    if c and "otr03" in c:
        L, p = c["otr03"]
        x = [l / c["nL"] for l in L]
        ax2.plot(x, p["function"]["coef"], color=LIGHT, lw=1.6, ls=(0, (2, 2)), label="grammatical function")
        ax2.plot(x, p["order"]["coef"], color=MUTED, lw=1.6, label="order / focus")
        ax2.plot(x, p["role"]["coef"], color=TEAL, lw=2.4, label="thematic role")
        ax2.axhline(0, color="#D8D3CA", lw=1)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([0, .5, 1], ["in", "50%", "out"])
        ax2.set_title("Llama 3.3 70B: transplant coefficients (OTR-03)", loc="left", fontsize=11)
        style(ax2)
        ax2.legend(frameon=False, fontsize=9)
    fig.text(0.5, -0.04, "Exploratory, dev only, no frozen settings. Qwen2.5 72B OTR-03 has not been run.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------------------- S5 multilingual
def multilingual(res):
    beh, hand, curves = [], [], {}
    for _, slug, nL, role, by, ho in MODELS:
        m = SHORT[slug]
        curves[m] = {"nL": nL}
        d = load_res(res, FREEZE_FILES[slug]["otr09"])
        curves[m]["en"] = (d["residual_layers"], d["residual"]["all"]["repair"])
        hand.append({"model": m, "language": "en", "handover_recomputed": first_crossing(d["residual_layers"], d["residual"]["all"]["repair"]),
                     "handover_frozen": ho, "n_layers": nL, "file": FREEZE_FILES[slug]["otr09"]})
        for lang in ("es", "gl"):
            f9 = pick(res, f"otr09_{slug}_dev_{lang}_20261121*")
            if f9:
                e = load_res(res, f9)
                curves[m][lang] = (e["residual_layers"], e["residual"]["all"]["repair"])
                hand.append({"model": m, "language": lang,
                             "handover_recomputed": first_crossing(e["residual_layers"], e["residual"]["all"]["repair"]),
                             "handover_frozen": XL_HANDOVER[lang][m], "n_layers": nL, "file": f9})
            g = pick(res, f"otrgen_{slug}_dev_{lang}_2*",
                     keep=lambda f: json.load(open(os.path.join(res, f)))["instruction"] in
                     ("Responde con un solo nombre.", "Responde cun só nome."))
            if g:
                gd = load_res(res, g)
                for cons in ("active", "passive", "object_cleft", "by_cleft", "ovs_clitic"):
                    beh.append({"model": m, "language": lang, "construction": cons,
                                "agent_q_active_voice": gd["cells"].get(f"{cons}/agent/active"),
                                "agent_q_passive_voice": gd["cells"].get(f"{cons}/agent/passive"),
                                "patient_q_active_voice": gd["cells"].get(f"{cons}/patient/active"),
                                "patient_q_passive_voice": gd["cells"].get(f"{cons}/patient/passive"),
                                "overall_accuracy": gd["generation_accuracy"], "n_overall": gd["n"],
                                "instruction": gd["instruction"], "file": g})
    return beh, hand, curves


def fig_multilingual(curves, hand, path):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    styles = {"en": (INK, "-", "English"), "es": (TEAL, "-", "Spanish"), "gl": (ORANGE, (0, (3, 2)), "Galician")}
    for ax, (label, slug, *_r) in zip(axes, MODELS):
        m = SHORT[slug]
        c = curves[m]
        for lang, (col, ls, name) in styles.items():
            if lang in c:
                L, v = c[lang]
                ax.plot([l / c["nL"] for l in L], v, color=col, ls=ls, lw=2, label=name)
                h = next(r for r in hand if r["model"] == m and r["language"] == lang)
                ax.scatter([h["handover_frozen"] / c["nL"]], [-0.18], marker="^", color=col, s=36, clip_on=False)
        ax.axhline(0.5, color="#D8D3CA", lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([0, .5, 1], ["in", "50%", "out"])
        ax.set_title(label, loc="left", fontsize=11)
        style(ax)
    axes[0].set_ylabel("residual repair (all items)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.09))
    fig.text(0.5, -0.04, "Relative depth; triangles: frozen handover per language (FREEZE.md, FREEZE_XL.md)",
             ha="center", fontsize=9.5, color=MUTED)
    save(fig, path)


def swedish(res):
    rows = []
    for _, slug, *_ in MODELS:
        o = pick(res, f"otrgen_{slug}_dev_sv_2*")
        if not o:
            continue
        do = load_res(res, o)
        r = {"model": SHORT[slug], "ordinary_accuracy": do["generation_accuracy"], "n": do["n"], "instruction": do["instruction"]}
        for key, q in (("periphrastic_failing", "q-vem-blev-n-gon-pp-av"), ("s_passive_failing", "q-vem-spass-n-gon-av")):
            f = pick(res, f"otrgen_{slug}_dev_sv_{q}_2*",
                     keep=lambda f: json.load(open(os.path.join(res, f)))["instruction"] == "Svara med ett enda namn.")
            if f:
                d = load_res(res, f)
                r[key] = d["generation_accuracy"]
                r[key + "_question"] = d["question_template"]
        r["file"] = o
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------- S4 model cards
def model_cards(conf, k_rows, hand):
    rows = []
    for label, slug, nL, role, by, ho in MODELS:
        m = SHORT[slug]
        passes = Counter()
        totals = Counter()
        for r in conf:
            if r["model"] == m and r["verdict"] in ("PASS", "FAIL"):  # "no claim" (C7 for phi-4, OLMo) is not counted
                totals[r["set"]] += 1
                passes[r["set"]] += r["verdict"] == "PASS"
        k20 = next(r for r in k_rows if r["model"] == m and r["k"] == 20)
        rows.append({"model": m, "full_name": label, "n_layers": nL, "role_layer": role, "by_layer": by,
                     "handover_en": ho, "handover_es": XL_HANDOVER["es"][m], "handover_gl": XL_HANDOVER["gl"][m],
                     "test0_core": f"{passes['core']}/{totals['core']}", "test0_bx": f"{passes['bx-transitive']}/{totals['bx-transitive']}",
                     "test0_es": f"{passes['es']}/{totals['es']}", "test0_gl": f"{passes['gl']}/{totals['gl']}",
                     "failed_claims": "; ".join(f"{r['set']} {r['claim']}" for r in conf if r["model"] == m and r["verdict"] == "FAIL") or "none",
                     "no_claim": "; ".join(f"{r['set']} {r['claim']}" for r in conf if r["model"] == m and r["verdict"] == "no claim") or "",
                     "dev_top20_heldout": k20["top_repair"], "dev_random20_heldout": k20["random_repair"]})
    return rows


def software_versions(res):
    seen = Counter()
    for name in USED:
        if not name.endswith(".json"):
            continue
        d = json.load(open(os.path.join(res, name)))
        v = d.get("versions")
        if isinstance(v, dict):
            seen[json.dumps({k: v[k] for k in sorted(v) if k in ("torch", "transformers", "python", "numpy", "accelerate")})] += 1
    return [{"versions": k, "n_files": n} for k, n in seen.most_common()]


# ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--main-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-commit", default="unknown")
    ap.add_argument("--code-commit", default="unknown")
    a = ap.parse_args()
    T, F = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(T, exist_ok=True)
    os.makedirs(F, exist_ok=True)

    conf, conf_used = confirmatory_full(a.main_results)
    write_csv(os.path.join(T, "st01_confirmatory_test0_full.csv"), conf)
    fig_forest(conf, os.path.join(F, "sfig01_confirmatory_estimates.svg"))

    qw = question_wording(a.results)
    write_csv(os.path.join(T, "st02_question_wording_dev.csv"), qw)
    fig_question_wording(qw, os.path.join(F, "sfig02_question_wording.svg"))
    bxr = bx_blocks(a.results)
    write_csv(os.path.join(T, "st03_bx_blocks_dev.csv"), bxr)
    fig_bx_blocks(bxr, os.path.join(F, "sfig03_bx_blocks.svg"))
    pf = prompt_formats(a.results)
    write_csv(os.path.join(T, "st04_prompt_formats_dev.csv"), pf)
    fig_prompt_formats(pf, os.path.join(F, "sfig04_prompt_formats.svg"))
    write_csv(os.path.join(T, "st05_stimulus_inventory.csv"), stimulus_inventory(a.code))

    rl, rc = role_code_layers(a.results)
    write_csv(os.path.join(T, "st06_role_code_by_layer_dev.csv"), rl)
    fig_role_core_bx(rc, os.path.join(F, "sfig05_role_code_core_vs_bx.svg"))
    rr, rcv = residual_by_construction(a.results)
    write_csv(os.path.join(T, "st07_residual_repair_by_construction_dev.csv"), rr)
    fig_residual_by_construction(rcv, os.path.join(F, "sfig06_residual_repair_by_construction.svg"))
    kr, hr = heads(a.results)
    write_csv(os.path.join(T, "st08_heads_k_curve_dev.csv"), kr)
    write_csv(os.path.join(T, "st09_top20_heads_dev.csv"), hr)
    fig_heads(kr, hr, os.path.join(F, "sfig07_heads.svg"))
    bm, bmc = big_model_curves(a.results)
    write_csv(os.path.join(T, "st10_big_models_curves_dev.csv"), bm)
    fig_big_models(bmc, os.path.join(F, "sfig08_big_models.svg"))

    mb, mh, mc = multilingual(a.results)
    write_csv(os.path.join(T, "st11_multilingual_behaviour_dev.csv"), mb)
    write_csv(os.path.join(T, "st12_handover_by_language_dev.csv"), mh)
    fig_multilingual(mc, mh, os.path.join(F, "sfig09_handover_by_language.svg"))
    sv = swedish(a.results)
    write_csv(os.path.join(T, "st13_swedish_dev_unchecked.csv"), sv)

    cards = model_cards(conf, kr, mh)
    write_csv(os.path.join(T, "st14_model_cards.csv"), cards)
    sw = software_versions(a.results)
    if sw:
        write_csv(os.path.join(T, "st15_software_versions.csv"), sw)

    bad = [r for r in mh if r["handover_recomputed"] != r["handover_frozen"]]
    manifest = {"script": "papers/order-to-role/supplementary/analysis/build_supplement.py", "version": VERSION,
                "results_commit": a.results_commit, "code_commit": a.code_commit,
                "handover_check": {"rows": len(mh), "recomputed_differs_from_frozen": bad},
                "inputs_sha256": dict(sorted(USED.items())), "confirmatory_inputs_sha256": conf_used,
                "status_note": "st01 is the frozen confirmatory test0 result (read from the frozen scorer's output). "
                               "Every other table is development-split, exploratory. st13 (Swedish) uses stimuli "
                               "without a native-speaker check."}
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"inputs": len(USED), "handover_mismatches": len(bad)}))


if __name__ == "__main__":
    main()
