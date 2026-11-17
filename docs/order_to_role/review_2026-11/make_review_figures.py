"""Regenerate every figure and table of the November 2026 review package from result files.

    python docs/order_to_role/review_2026-11/make_review_figures.py <results-dir> [<results-dir> ...]

Runs are found by their metadata (model, instruction, question template, stimuli), not by
file name. Generation answers are re-scored with the current extractor
(scripts/otr_generate_check.extract_answer), so full-sentence answers count. Writes PNGs
to figures/ and CSVs to tables/ next to this file. Dev split, exploratory.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from relational_lab.experiments import order_role_stimuli as ors  # noqa: E402
from relational_lab.experiments import stranded_p_stimuli as sp  # noqa: E402

_spec = importlib.util.spec_from_file_location("rescore", ROOT / "scripts" / "otr_rescore_generation.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

# Validated categorical palette (slots 1-4, light mode); text stays in ink colours.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MAIN = ("active", "passive", "object_cleft", "by_cleft")
GEMMA, QWEN = "google/gemma-4-31B-it", "Qwen/Qwen3.8-27B"
STRANDED, PATIENT = "Who was someone {past} by?", "Who was {past} by someone?"
NAME_ONLY, SENTENCE = "Reply with one name only.", "Answer with a full sentence."

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "text.color": INK, "legend.frameon": False,
})


def load(dirs):
    runs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            try:
                runs.append((f, json.loads(f.read_text())))
            except Exception:
                pass
    return runs


def find_gen(runs, items, model, instruction, template=None, stimuli="otr"):
    """Latest generation run with this metadata, re-scored."""
    hits = [(f, r) for f, r in runs if f.name.startswith("otrgen_") and r.get("model") == model
            and (r.get("instruction") or "") == instruction and r.get("question_template") == template
            and r.get("stimuli", "otr") == stimuli and r.get("rows") and "role" in r["rows"][0]
            and (stimuli != "sp" or len(r["rows"]) >= 17712 or template is not None)]
    if not hits:
        return None, None
    f, r = hits[-1]
    return f, rs.rescore(r, items)


def find_otr(runs, prefix, model, **meta):
    hits = [(f, r) for f, r in runs if f.name.startswith(prefix) and r.get("model") == model
            and all(r.get(k) == v for k, v in meta.items())]
    return hits[-1] if hits else (None, None)


def bar_group(ax, cats, series, labels, colors, ylim=(0, 1.08), fmt="{:.2f}"):
    n = len(series)
    w = 0.8 / n
    for i, (vals, lab, col) in enumerate(zip(series, labels, colors)):
        xs = [x + (i - (n - 1) / 2) * w for x in range(len(cats))]
        ax.bar(xs, vals, width=w * 0.92, color=col, label=lab, edgecolor="white", linewidth=1)
        for x, v in zip(xs, vals):
            if v is not None:
                ax.text(x, v + 0.015, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main(argv=None):
    dirs = (argv or sys.argv[1:]) or [ROOT / "results"]
    runs = load(dirs)
    items = {i.item_id: i for i in ors.generate_items()}
    items.update({i.item_id: i for i in sp.generate_items()})
    figs, tabs = HERE / "figures", HERE / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)
    provenance = {}

    # ---------- F1: forced choice vs instructed generation (the artefact) ----------
    models = [("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-I"), ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-I"),
              ("allenai/OLMo-2-1124-13B-Instruct", "OLMo-2-13B-I"), ("allenai/OLMo-2-0325-32B-Instruct", "OLMo-2-32B-I"),
              ("meta-models/Muse-Glimmer-30B", "Glimmer-30B"), ("unsloth/Mistral-Small-3.2-24B-Instruct-2506", "Mistral-3.2-24B")]
    rows, fc_v, gen_v, names = [], [], [], []
    for mid, short in models:
        f1, o1 = find_otr(runs, "otr01_", mid)
        fg, g = find_gen(runs, items, mid, NAME_ONLY)
        if not (o1 and g):
            continue
        sc = [s for s in o1["scores"] if s["item_id"].split(":")[2] in MAIN]
        fc = sum(s["margin"] > 0 for s in sc) / len(sc)
        fc_v.append(fc)
        gen_v.append(g["accuracy"]["all"])
        names.append(short)
        rows.append([short, round(fc, 3), round(g["accuracy"]["all"], 3), len(sc), g["n"], f1.name, fg.name])
    write_csv(tabs / "t1_forced_choice_vs_generation.csv",
              ["model", "forced_choice_acc", "generation_acc_name_only", "n_fc", "n_gen", "otr01_file", "otrgen_file"], rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bar_group(ax, names, [fc_v, gen_v], ["Forced choice (no instruction)", "Generated, 'Reply with one name only.'"], C[:2])
    ax.set_ylabel("accuracy, 4 main constructions")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("F1. Forced choice vs generated answers (dev; see README for item samples)", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "f1_forced_choice_vs_generation.png", dpi=160)
    plt.close(fig)

    # ---------- F2: stranded agent question by construction ----------
    cons_lab = ["active", "object cleft", "by-cleft", "passive"]
    cons = ["active", "object_cleft", "by_cleft", "passive"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    rows = []
    for ax, model, short in zip(axes, (GEMMA, QWEN), ("Gemma 4 31B-it", "Qwen3.8-27B")):
        series, labels = [], []
        f5, o5 = find_otr(runs, "otr05_", model)
        if o5:
            fc = [o5["summary"]["A_who_by"]["by_construction"][c]["accuracy"] for c in cons]
            series.append(fc)
            labels.append("forced choice")
        for instr, lab in ((NAME_ONLY, "generated, name only"), ("", "generated, free"), (SENTENCE, "generated, full sentence")):
            fg, g = find_gen(runs, items, model, instr, STRANDED)
            if g:
                v = [g["cells"].get(f"{c}/agent/custom") for c in cons]
                series.append(v)
                labels.append(lab)
                provenance[f"F2 {short} {lab}"] = fg.name
        bar_group(ax, cons_lab, series, labels, C[:len(series)])
        for lab, v in zip(labels, series):
            rows.append([short, lab] + [round(x, 3) if x is not None else "" for x in v])
        fp, gp = find_gen(runs, items, model, NAME_ONLY, PATIENT)
        title = short
        if gp:
            title += f"   (patient permutation 'Who was V-ed by someone?', name only: {gp['accuracy']['all']:.2f})"
        ax.set_title(title, fontsize=8.5)
        ax.axhline(0.5, color=GRID, lw=1)
    axes[0].set_ylabel("accuracy (agent named)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("F2. 'Who was someone V-ed by?' (agent) on 164 dev events per construction", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figs / "f2_stranded_by_question.png", dpi=160)
    plt.close(fig)
    write_csv(tabs / "t2_stranded_by_question.csv", ["model", "condition"] + cons, rows)

    # ---------- F3: within-item dissociation (name only vs full sentence) ----------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    rows = []
    for ax, model, short in zip(axes, (GEMMA, QWEN), ("Gemma 4 31B-it", "Qwen3.8-27B")):
        _, gb = find_gen(runs, items, model, NAME_ONLY, STRANDED)
        _, gs = find_gen(runs, items, model, SENTENCE, STRANDED)
        if not (gb and gs):
            continue
        b = {r["item_id"]: r["correct"] for r in gb["rows"]}
        tab = defaultdict(int)
        for r in gs["rows"]:
            c = items[r["item_id"]].construction
            if c in ("active", "object_cleft") and r["item_id"] in b:
                tab[(b[r["item_id"]], r["correct"])] += 1
        m = [[tab[(True, True)], tab[(True, False)]], [tab[(False, True)], tab[(False, False)]]]
        ax.imshow([[x / max(1, sum(map(sum, m))) for x in row] for row in m], cmap=SEQ, vmin=0, vmax=0.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i][j]), ha="center", va="center", fontsize=11,
                        color="white" if m[i][j] / max(1, sum(map(sum, m))) > 0.3 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sentence right", "sentence wrong"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["name right", "name wrong"])
        ax.grid(False)
        ax.set_title(f"{short} (n={sum(map(sum, m))})", fontsize=9)
        rows.append([short, m[0][0], m[0][1], m[1][0], m[1][1]])
    fig.suptitle("F3. Same items, two answer formats: active + object-cleft sentences, stranded agent question",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "f3_within_item_formats.png", dpi=160)
    plt.close(fig)
    write_csv(tabs / "t3_within_item_formats.csv",
              ["model", "name_right_sentence_right", "name_right_sentence_wrong", "name_wrong_sentence_right",
               "name_wrong_sentence_wrong"], rows)

    # ---------- F4: logit lens at the answer position ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex=True)
    for row, (model, short) in enumerate(((GEMMA, "Gemma 4 31B-it"), (QWEN, "Qwen3.8-27B"))):
        hits = [(f, r) for f, r in runs if f.name.startswith("otr06_") and r.get("model") == model
                and not r.get("instruction") and r.get("stimuli", "otr") == "otr"]
        f6, o6 = hits[-1] if hits else (None, None)
        if not o6:
            continue
        provenance[f"F4 {short}"] = f6.name
        L = o6["n_layers"]
        depth = [i / (L - 1) for i in range(L)]
        for col, cons_ in enumerate(("active", "object_cleft")):
            ax = axes[row][col]
            for ci, cond in enumerate(o6["conditions"]):
                key = f"{ci}|{cond['role']}|{cond['template']}|{cons_}"
                if key in o6["aggregate"]:
                    lab = cond["template"].replace("{past}", "V-ed") + f" [{cond['role']}]"
                    ax.plot(depth, o6["aggregate"][key]["mean_ld"], color=C[ci], lw=2, label=lab)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{short}: {cons_.replace('_', ' ')} sentences", fontsize=9)
            if col == 0:
                ax.set_ylabel("logit diff, correct − other")
            if row == 1:
                ax.set_xlabel("relative depth")
    axes[0][1].legend(fontsize=7, loc="upper left")
    fig.suptitle("F4. Logit lens at the answer position (no instruction). Above 0 = favours the correct name.",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "f4_logit_lens.png", dpi=160)
    plt.close(fig)

    # ---------- F5: stranded-preposition frames incl. passive questions (heatmaps) ----------
    q_order = list(sp.QUESTION_FORMS)
    col_order = [f"{fr}_{fm}" for fr in sp.FRAMES
                 for fm in (sp.FORMS_ALTERNATING if sp.FRAMES[fr][1] else sp.FORMS_PLAIN)]
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.6))
    for ax, model, short in zip(axes, (GEMMA, QWEN), ("Gemma 4 31B-it", "Qwen3.8-27B")):
        fg, g = find_gen(runs, items, model, NAME_ONLY, None, stimuli="sp")
        if not g:
            continue
        provenance[f"F5 {short}"] = fg.name
        M = []
        for q in q_order:
            role = sp.QUESTION_FORMS[q]
            M.append([g["cells"].get(f"{c}/{role}/{q}") for c in col_order])
            rows.append([short, q] + [("" if x is None else round(x, 3)) for x in M[-1]])
        cm = SEQ.reversed().copy()
        cm.set_bad("#f0efea")
        ax.imshow([[float("nan") if x is None else x for x in r] for r in M], cmap=cm, vmin=0, vmax=1, aspect="auto")
        for i, r in enumerate(M):
            for j, x in enumerate(r):
                ax.text(j, i, "–" if x is None else f"{x:.2f}", ha="center", va="center", fontsize=6.5,
                        color=MUTED if x is None else ("white" if x < 0.55 else INK))
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(q_order)))
        ax.set_yticklabels(q_order, fontsize=7.5)
        ax.grid(False)
        ax.set_title(f"{short}, 'Reply with one name only.' (164 events per cell; dark = errors)", fontsize=9, loc="left")
    fig.suptitle("F5. Stranded-preposition frames (SP 1.1.0): rows = question form, columns = frame_sentence form",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "f5_stranded_p_frames.png", dpi=160)
    plt.close(fig)
    write_csv(tabs / "t5_stranded_p_frames.csv", ["model", "question_form"] + col_order, rows)

    # ---------- F6: per-verb passive-agent questions on active sentences ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    rows = []
    for ax, model, short in zip(axes, (GEMMA, QWEN), ("Gemma 4 31B-it", "Qwen3.8-27B")):
        _, g = find_gen(runs, items, model, NAME_ONLY, None, stimuli="sp")
        if not g:
            continue
        acc = defaultdict(list)
        for r in g["rows"]:
            it = items[r["item_id"]]
            if it.question_voice == "pass_agent_stranded" and it.form in ("po", "do", "cleft_stranded", "cleft_piped"):
                acc[(it.frame, it.lemma)].append(r["correct"])
        keys = sorted(acc, key=lambda k: (list(sp.FRAMES).index(k[0]), k[1]))
        vals = [sum(acc[k]) / len(acc[k]) for k in keys]
        cols = [C[list(sp.FRAMES).index(k[0])] for k in keys]
        ax.bar(range(len(keys)), vals, color=cols, edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k[1] for k in keys], rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_title(short, fontsize=9)
        for i, v in enumerate(vals):
            if v < 0.6:
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
        for k, v in zip(keys, vals):
            rows.append([short, k[0], k[1], round(v, 3), len(acc[k])])
        for fi, fr in enumerate(sp.FRAMES):
            idx = [i for i, k in enumerate(keys) if k[0] == fr]
            ax.text((idx[0] + idx[-1]) / 2, 1.02, f"'{fr}' frame", ha="center", fontsize=7, color=INK)
    axes[0].set_ylabel("accuracy (agent named)")
    fig.suptitle("F6. 'Who was OBJ V-ed P someone by?' on active sentences (po/do/clefts), by verb",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "f6_passive_agent_by_verb.png", dpi=160)
    plt.close(fig)
    write_csv(tabs / "t6_passive_agent_by_verb.csv", ["model", "frame", "verb", "accuracy", "n"], rows)

    # ---------- F7: question wordings x answer format ----------
    variants = [
        ("Who was someone {past} by?", "agent"), ("Who was somebody {past} by?", "agent"),
        ("Who was a person {past} by?", "agent"), ("Who was the other person {past} by?", "agent"),
        ("Who had someone been {past} by?", "agent"), ("Who has someone been {past} by?", "agent"),
        ("Who did someone get {past} by?", "agent"), ("Who was the {ing} done by?", "agent"),
        ("By whom was the {ing} done?", "agent"), ("Who did the {ing}?", "agent"),
        ("Who was the {ing} done to?", "patient"), ("Who was {past} by someone?", "patient"),
    ]
    rows, present = [], []
    for model, short in ((GEMMA, "Gemma 4 31B-it"), (QWEN, "Qwen3.8-27B")):
        for tmpl, role in variants:
            for instr, lab in ((NAME_ONLY, "name"), (SENTENCE, "sentence")):
                fg, g = find_gen(runs, items, model, instr, tmpl)
                if not g:
                    continue
                vals = [g["cells"].get(f"{c}/{role}/custom") for c in cons]
                rows.append([short, tmpl, role, lab] + [("" if v is None else round(v, 3)) for v in vals] + [fg.name])
                present.append((short, tmpl, role, lab, vals))
    write_csv(tabs / "t7_question_variants_by_format.csv",
              ["model", "question", "role", "format"] + cons + ["file"], rows)
    tmpls = [t for t, _ in variants if any(p[1] == t for p in present)]
    if tmpls:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 0.42 * len(tmpls) + 1.8), sharey=True)
        for ax, short in zip(axes, ("Gemma 4 31B-it", "Qwen3.8-27B")):
            M, ylab = [], []
            for t in tmpls:
                byfmt = {p[3]: p[4] for p in present if p[0] == short and p[1] == t}
                M.append([(byfmt.get("name") or [None] * 4)[i] for i in (0, 1)] +
                         [(byfmt.get("sentence") or [None] * 4)[i] for i in (0, 1)])
                ylab.append(t.replace("{past}", "V-ed").replace("{ing}", "V-ing"))
            cm = SEQ.reversed().copy()
            cm.set_bad("#f0efea")
            ax.imshow([[float("nan") if x is None else x for x in r] for r in M], cmap=cm, vmin=0, vmax=1,
                      aspect="auto")
            for i, r in enumerate(M):
                for j, x in enumerate(r):
                    ax.text(j, i, "–" if x is None else f"{x:.2f}", ha="center", va="center", fontsize=7,
                            color=MUTED if x is None else ("white" if x < 0.55 else INK))
            ax.set_xticks(range(4))
            ax.set_xticklabels(["name:\nactive", "name:\nobj. cleft", "sentence:\nactive", "sentence:\nobj. cleft"],
                               fontsize=7)
            ax.set_yticks(range(len(tmpls)))
            ax.set_yticklabels(ylab, fontsize=7.5)
            ax.axvline(1.5, color="white", lw=3)
            ax.grid(False)
            ax.set_title(short, fontsize=9)
        fig.suptitle("F7. Question wordings × answer format (accuracy; dark = errors; grey = not run). Same 656 items per run.",
                     fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(figs / "f7_question_variants_by_format.png", dpi=160)
        plt.close(fig)

    # ---------- T8: three-place passive-agent pockets, name vs sentence ----------
    rows = []
    for model, short in ((GEMMA, "Gemma 4 31B-it"), (QWEN, "Qwen3.8-27B")):
        for instr, lab in ((NAME_ONLY, "name"), (SENTENCE, "sentence")):
            fg, g = find_gen(runs, items, model, instr, None, stimuli="sp")
            if not g:
                continue
            acc = defaultdict(list)
            for r in g["rows"]:
                it = items[r["item_id"]]
                if it.question_voice.startswith("pass_agent") and not it.form.startswith("pass"):
                    key = it.lemma if it.frame == "from" else ("cleft_stranded" if it.form == "cleft_stranded" else "other")
                    acc[(it.frame, key)].append(r["correct"])
            for (fr, key), v in sorted(acc.items()):
                rows.append([short, lab, fr, key, round(sum(v) / len(v), 3), len(v), fg.name])
    write_csv(tabs / "t8_passive_agent_name_vs_sentence.csv",
              ["model", "format", "frame", "verb_or_form", "accuracy", "n", "file"], rows)

    (HERE / "tables" / "provenance.json").write_text(json.dumps(provenance, indent=1))
    print("figures:", sorted(p.name for p in figs.glob("*.png")))
    print("tables:", sorted(p.name for p in tabs.glob("*")))


if __name__ == "__main__":
    main()
