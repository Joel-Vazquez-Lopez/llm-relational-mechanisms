"""Cross-model layer figures for the order-to-role study (dev split, exploratory).

Reads OTR-03 and OTR-04 result JSONs from a checkout of branch order-to-role/results-dev:
    python make_layer_figures.py /path/to/results
Writes:
  layer_role_code.png      OTR-03 role code (% of name-identity effect, 95% CI) by relative depth.
  layer_question_side.png  OTR-04 flip effect at "?", ":" and the period control by relative depth,
                           with the OTR-03 role-code peak marked, one panel per model.
  layer_summary.csv        the per-model peaks used in the README tables.
"""
import ast
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"
Q_COL, C_COL, P_COL = "#2a78d6", "#eb6834", "#a3a29b"
RAMP = {"Qwen2.5": ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"],
        "OLMo-2": ["#f5a882", "#eb6834", "#a53d14"],
        "Other": ["#eda100", "#1baf7a"]}

# label, family, OTR-03 file(s) (averaged), OTR-04 file with flip mode
MODELS = [
    ("Qwen2.5-7B base", "Qwen2.5", ["otr03_Qwen__Qwen2.5-7B_dev_20261114T233555.json"], "otr04_Qwen__Qwen2.5-7B_dev_20261114T232608.json"),
    ("Qwen2.5-1.5B-I", "Qwen2.5", ["otr03_Qwen__Qwen2.5-1.5B-Instruct_dev_20261114T234448.json"], "otr04_Qwen__Qwen2.5-1.5B-Instruct_dev_20261114T233743.json"),
    ("Qwen2.5-7B-I", "Qwen2.5", ["otr03_Qwen__Qwen2.5-7B-Instruct_dev_20261115T000258.json"], "otr04_Qwen__Qwen2.5-7B-Instruct_dev_20261115T000203.json"),
    ("Qwen2.5-14B-I", "Qwen2.5", ["otr03_Qwen__Qwen2.5-14B-Instruct_dev_20261115T011236.json"], "otr04_Qwen__Qwen2.5-14B-Instruct_dev_20261115T012803.json"),
    ("Qwen2.5-32B-I", "Qwen2.5", ["otr03_Qwen__Qwen2.5-32B-Instruct_dev_20261114T182808.json"], "otr04_Qwen__Qwen2.5-32B-Instruct_dev_20261114T192737.json"),
    ("OLMo-2-7B-I", "OLMo-2", ["otr03_allenai__OLMo-2-1124-7B-Instruct_dev_20261114T235848.json"], "otr04_allenai__OLMo-2-1124-7B-Instruct_dev_20261114T235548.json"),
    ("OLMo-2-13B-I", "OLMo-2", ["otr03_allenai__OLMo-2-1124-13B-Instruct_dev_20261115T005114.json"], "otr04_allenai__OLMo-2-1124-13B-Instruct_dev_20261115T003813.json"),
    ("OLMo-2-32B-I", "OLMo-2", ["otr03_allenai__OLMo-2-0325-32B-Instruct_dev_20261115T013059.json"], "otr04_allenai__OLMo-2-0325-32B-Instruct_dev_20261115T012520.json"),
    ("Qwen3.8-27B", "Other", ["otr03_Qwen__Qwen3.8-27B_dev_20261116T013154.json"], "otr04_Qwen__Qwen3.8-27B_dev_20261116T024148.json"),
    ("Gemma 4 31B-it", "Other", ["otr03_google__gemma-4-31B-it_dev_20261114T132626.json", "otr03_google__gemma-4-31B-it_dev_20261114T151524.json"], "otr04_google__gemma-4-31B-it_dev_20261115T011857.json"),
]


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def role_curve(files):
    """Role coefficient as a fraction of |identity effect|, with CI, averaged over files."""
    pct, lo, hi = [], [], []
    for f in files:
        p = load(f)["profile"]
        ident = np.abs(np.array(p["mean_identity_effect"]))
        ident = np.where(ident > 0.5, ident, np.nan)  # last layer: a name-position patch cannot act (identity = 0)
        c = np.array(p["role"]["coef"])
        ci = np.array(p["role"]["ci95"])
        pct.append(c / ident); lo.append(ci[:, 0] / ident); hi.append(ci[:, 1] / ident)
    return np.mean(pct, 0), np.mean(lo, 0), np.mean(hi, 0)


def tail_positions(d):
    tt = d["records"][0]["tail_tokens"]
    tt = tt if isinstance(tt, list) else ast.literal_eval(tt)
    pos = {}
    for i, t in enumerate(tt):
        if "?" in t and "?" not in pos:
            pos["?"] = f"end-{i + 1}"
        if t.strip() == ":" and ":" not in pos:
            pos[":"] = f"end-{i + 1}"
    return pos


def depth(n):
    return np.arange(n) / n  # same convention as the README (layer / n_layers)


plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.7)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


rows, curves = [], {}
for label, fam, f3, f4 in MODELS:
    r, rlo, rhi = role_curve(f3)
    d4 = load(f4)
    flip = d4["aggregate"]["flip/ALL"]
    pos = tail_positions(d4)
    q = np.array(flip[pos["?"]]["ratio_of_means"])
    c = np.array(flip[pos[":"]]["ratio_of_means"])
    if pos[":"] == "end-1":  # base prompt: ':' is the final token = readout position, not a question-side state
        c = np.full_like(c, np.nan)
    per = np.array(flip["period"]["ratio_of_means"])
    curves[label] = dict(fam=fam, r=r, rlo=rlo, rhi=rhi, q=q, c=c, per=per)
    n3, n4 = len(r), len(q)
    best_tok = "?" if np.isnan(c).all() or q.max() >= np.nanmax(c) else ":"
    best = q if best_tok == "?" else c
    rows.append(dict(model=label, n_layers=n4, role_peak_layer=int(np.nanargmax(r)), role_peak_depth=round(float(depth(n3)[np.nanargmax(r)]), 3),
                     role_peak_pct_identity=round(float(np.nanmax(r)) * 100, 2), q_peak=round(float(q.max()), 3), q_peak_layer=int(q.argmax()),
                     colon_peak=None if np.isnan(c).all() else round(float(np.nanmax(c)), 3), colon_peak_layer=None if np.isnan(c).all() else int(np.nanargmax(c)), best_token=best_tok,
                     best_depth=round(float(depth(n4)[best.argmax()]), 3), period_max=round(float(per.max()), 3)))

with open("layer_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# Figure 1: role code by depth, one panel per family group
groups = [("Qwen2.5", "A. Qwen2.5 (darker = larger)"), ("OLMo-2", "B. OLMo-2 (darker = larger)"), ("Other", "C. Qwen3.8 and Gemma 4")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, layout="constrained")
for ax, (fam, title) in zip(axes, groups):
    members = [(lab, v) for lab, v in curves.items() if v["fam"] == fam]
    for (lab, v), col in zip(members, RAMP[fam]):
        x = depth(len(v["r"]))
        ax.fill_between(x, v["rlo"], v["rhi"], color=col, alpha=0.18, lw=0)
        ax.plot(x, v["r"], color=col, lw=2, label=lab)
        k = int(np.nanargmax(v["r"]))
        off = {"Gemma 4 31B-it": (8, -2), "Qwen3.8-27B": (-80, 6)}.get(lab, (4, 4))
        ax.annotate(lab.replace("Qwen2.5-", "").replace("OLMo-2-", ""), (x[k], v["r"][k]), xytext=off,
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK); clean(ax)
    ax.set_xlabel("Relative depth (layer / number of layers)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
axes[0].set_ylabel("Role code on names (% of name-identity effect)")
axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
fig.suptitle("OTR-03: where a portable agent/patient code appears on the name tokens", x=0.0, ha="left", fontsize=11.5, color=INK)
fig.text(0.0, -0.05, "Dev split, exploratory. Shaded = 95% family-bootstrap CI. Gemma = mean of its active- and passive-question runs; "
         "others pooled both question voices. Validity check (identical left context) = 0 in every model.", fontsize=8, color=MUTED)
fig.savefig("layer_role_code.png", dpi=160, bbox_inches="tight", facecolor="white")

# Figure 2: question-side flip by depth, one panel per model, role peak marked
fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True, sharey=True, layout="constrained")
for ax, (lab, v) in zip(axes.flat, curves.items()):
    x = depth(len(v["q"]))
    xr = depth(len(v["r"]))[np.nanargmax(v["r"])]
    ax.axvline(xr, color=INK, lw=1, ls=(0, (2, 2)))
    ax.text(xr, 0.86, " role peak", fontsize=7.5, color=INK, va="top")
    ax.plot(x, v["per"], color=P_COL, lw=1.6)
    ax.plot(x, v["c"], color=C_COL, lw=2)
    ax.plot(x, v["q"], color=Q_COL, lw=2)
    ax.set_title(lab, loc="left", fontsize=10, color=INK); clean(ax)
    if np.isnan(v["c"]).all():
        ax.text(0.02, 0.6, '":" is the final token here\n(readout), not shown', fontsize=7.5, color=MUTED)
    ax.set_ylim(-0.05, 0.9)
for ax in axes[1]:
    ax.set_xlabel("Relative depth")
for ax in axes[:, 0]:
    ax.set_ylabel("Answer moved (ratio of means)")
h = [plt.Line2D([], [], color=Q_COL, lw=2, label='question state at "?"'),
     plt.Line2D([], [], color=C_COL, lw=2, label='question state at ":" of "Answer:"'),
     plt.Line2D([], [], color=P_COL, lw=1.6, label="control: sentence period"),
     plt.Line2D([], [], color=INK, lw=1, ls=(0, (2, 2)), label="OTR-03 role-code peak")]
fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False)
fig.text(0.0, -0.03, "OTR-04 question-side step by layer. Dev split, exploratory. Flip = copy the state from the same sentence asked for the other role; "
         "1 = answer fully moved to the source's answer. Ratio of means over pairs the model gets right.", fontsize=8, color=MUTED)
fig.savefig("layer_question_side.png", dpi=160, bbox_inches="tight", facecolor="white")
for r in rows:
    print(r)
