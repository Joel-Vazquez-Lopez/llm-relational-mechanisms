"""Cross-model summary figures for the order-to-role study (dev split, exploratory).

Numbers are copied from ORDER_TO_ROLE_HANDOFF.md section 2 (OTR-01, OTR-03, OTR-04 results on
branch order-to-role/results-dev). Run:  python make_cross_model_figures.py
Writes cross_model_otr.png (role code vs accuracy; cross-sentence repair vs control) and
question_binding_depth.png (where the question is read vs object-question accuracy).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

C = {"Qwen2.5": "#2a78d6", "OLMo-2": "#eb6834", "Gemma 4": "#1baf7a", "Qwen3.8": "#eda100"}
M = {"Qwen2.5": "o", "OLMo-2": "s", "Gemma 4": "D", "Qwen3.8": "^"}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"

# family, label, role code % of identity (OTR-03 peak), accuracy (OTR-01 overall),
# repair_x, control_x (OTR-04), question-side best token, its depth, "Who did someone V?" accuracy
D = [
    ("Qwen2.5", "7B base", 0.6, .69, .11, .00, "?", 15 / 28, .66),
    ("Qwen2.5", "1.5B-I", 2.1, .75, .25, .01, "?", 15 / 28, .52),
    ("Qwen2.5", "7B-I", 4.2, .70, .45, .04, ":", 20 / 28, .44),
    ("Qwen2.5", "14B-I", 9.9, .76, .44, .03, ":", 32 / 48, .57),
    ("Qwen2.5", "32B-I", 11.2, .79, .49, .03, ":", 47 / 64, .40),
    ("Qwen3.8", "27B", 4.9, .90, .50, .03, "?", 35 / 64, .95),
    ("OLMo-2", "7B-I", 0.8, .65, .21, .01, "?", 14 / 32, .56),
    ("OLMo-2", "13B-I", 2.0, .89, .36, .07, ":", 18 / 40, .86),
    ("OLMo-2", "32B-I", 1.6, .85, .46, .07, ":", 44 / 64, .68),
    ("Gemma 4", "31B-it", 4.9, .98, None, None, "?", 34 / 60, 1.00),
]
NOTE = "Dev split, exploratory (not confirmatory). "

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def legend(fig):
    h = [plt.Line2D([], [], marker=M[f], color=C[f], ls="", markersize=8, label=f) for f in C]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False)


# Figure 1: role code vs accuracy; repair vs control
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
for fam, lab, r, acc, *_ in D:
    a.scatter(r, acc, s=70, color=C[fam], marker=M[fam], edgecolor="white", linewidth=1.5, zorder=3)
    off = {("OLMo-2", "13B-I"): (6, -12), ("Qwen3.8", "27B"): (6, -12), ("Gemma 4", "31B-it"): (6, -12)}.get((fam, lab), (6, 4))
    a.annotate(f"{fam} {lab}" if fam in ("Qwen3.8", "Gemma 4") else lab, (r, acc), xytext=off,
               textcoords="offset points", fontsize=9, color=INK)
a.axhline(0.90, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
a.text(12.3, 0.905, "gate (≥0.90 per cell; approx.)", ha="right", fontsize=8, color=MUTED)
a.set_xlabel("Role code on names (OTR-03 peak, % of name-identity effect)")
a.set_ylabel("Behavioural accuracy (OTR-01, all questions)")
a.set_title("A. A strong role code on names does not predict success", loc="left", fontsize=11, color=INK)
a.set_ylim(0.6, 1.02); a.set_xlim(-0.3, 12.5); a.grid(color=GRID, lw=0.8); clean(a)
rows = [d for d in D if d[4] is not None]
ys = list(range(len(rows)))[::-1]
for y, (fam, lab, _, _, rx, cx, *_) in zip(ys, rows):
    b.plot([cx, rx], [y, y], color=GRID, lw=2, zorder=1)
    b.scatter(cx, y, s=60, facecolor="white", edgecolor=MUTED, linewidth=1.5, zorder=3)
    b.scatter(rx, y, s=70, color=C[fam], marker=M[fam], edgecolor="white", linewidth=1.5, zorder=3)
    b.text(rx + 0.015, y, f"{rx:.0%}", va="center", fontsize=9, color=INK)
b.set_yticks(ys, [f"{d[0]} {d[1]}" for d in rows]); b.set_xlim(-0.02, 0.85)
b.xaxis.set_major_formatter(PercentFormatter(1.0))
b.set_xlabel("Errors fixed by one question-side state from another sentence (best position/layer)")
b.set_title("B. A question state from another sentence repairs errors", loc="left", fontsize=11, color=INK)
b.scatter([], [], s=60, facecolor="white", edgecolor=MUTED, label="control (period position)")
b.scatter([], [], s=60, color=MUTED, label="repair (question-side state)")
b.legend(frameon=False, fontsize=8, loc="upper right"); b.grid(axis="x", color=GRID, lw=0.8); clean(b)
legend(fig)
fig.text(0.0, -0.04, NOTE + "Gemma made too few errors for repair. Gemma role code = mean of its active- and "
         "passive-question runs; others pooled both question voices.", fontsize=8, color=MUTED)
fig.savefig("cross_model_otr.png", dpi=160, bbox_inches="tight", facecolor="white")

# Figure 2: where the question is read vs object-question accuracy
fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
for fam, lab, _, _, _, _, tok, depth, who_did in D:
    filled = tok == "?"
    ax.scatter(depth, who_did, s=80, marker=M[fam], zorder=3, linewidth=1.8,
               color=C[fam] if filled else "white", edgecolor=C[fam] if not filled else "white")
    ax.annotate(f"{fam} {lab}", (depth, who_did), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Depth where the question-side state is strongest (OTR-04 flip, layer / n_layers)")
ax.set_ylabel('Accuracy on "Who did someone V?" (OTR-01)')
ax.set_title("Where the question is read vs object-question accuracy", loc="left", fontsize=11, color=INK)
ax.set_xlim(0.38, 0.82); ax.set_ylim(0.3, 1.05); ax.grid(color=GRID, lw=0.8); clean(ax)
ax.scatter([], [], s=60, color=MUTED, label='read at "?" (filled)')
ax.scatter([], [], s=60, facecolor="white", edgecolor=MUTED, linewidth=1.8, label='read at ":" of "Answer:" (hollow)')
ax.legend(frameon=False, fontsize=8, loc="lower left")
legend(fig)
fig.text(0.0, -0.05, NOTE + "Hypothesis only: 10 models. OLMo-2 13B-I reads at ':' but early and is fairly accurate.",
         fontsize=8, color=MUTED)
fig.savefig("question_binding_depth.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote cross_model_otr.png, question_binding_depth.png")
