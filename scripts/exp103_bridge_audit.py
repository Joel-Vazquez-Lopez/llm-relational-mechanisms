#!/usr/bin/env python3
"""EXP-103 / bridge-0: descriptive read-only audit of previously saved aggregate reports.

No model loading, data generation, training, or protected evaluation.
Usage (repo root):
    python3 scripts/exp103_bridge_audit.py
    python3 scripts/exp103_bridge_audit.py --write

The --write option creates NEW derived summaries under reports/exp103_bridge/;
original evidence files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

OLD = {
    "ML-080": ("reports/ml080/protected_evaluation.json", "runs", "target_new_pairs_new_exposure_verbs"),
    "ML-084": ("reports/ml084/protected_evaluation.json", "runs", "target_new_pairs_new_exposure_verbs"),
    "ML-085": ("reports/ml085/protected_causal_evaluation.json", "runs", "protected_behavior"),
    "ML-086": ("reports/ml086/protected_evaluation.json", "runs", "full_protected_target"),
    "ML-099": ("reports/ml099_role_q/protected_evaluation.json", "rows", "protected_behavior"),
}
SEEDS = (10401, 10403, 10409, 10413, 10421, 10427, 10429, 10433)
METHODS = ("quartet", "balanced_unpaired")
OLD_MEASURES = ("choice_accuracy", "correct_both_rate", "wrong_role_rate", "wrong_event_rate", "same_answer_rate")
NEW_MEASURES = ("accuracy4", "C_query", "both_correct_first", "both_correct_second", "role_copy_first", "role_copy_second", "correct_event_identity_first", "correct_event_identity_second")


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_json(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}; run from the repository root with all reports available")
    obj = json.loads(p.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), f"Not a JSON object: {p}")
    return obj


def valid_rate(v, label: str) -> float:
    require(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1,
            f"Invalid rate at {label}: {v!r}")
    return float(v)


def mean(rows, metric):
    return statistics.mean(row[metric] for row in rows)


def audit_old(root: Path):
    records = []
    digests = {}
    for cohort, (relpath, rowfield, field) in OLD.items():
        path = root / relpath
        j = load_json(path)
        digests[relpath] = file_sha(path)
        require(j.get("experiment") == cohort, f"Unexpected experiment at {relpath}")
        # These reports were already opened in the historical study. Nothing is generated here.
        require(j.get("protected_evaluation_opened", False) is True,
                f"Historical result was not marked previously opened: {cohort}")
        rows = j[rowfield]
        require(len(rows) == 16, f"Expected 16 historical rows: {cohort}")
        keyed = {(r["seed"], r["arm"]): r for r in rows}
        require(len(keyed) == 16 and set(a for _, a in keyed) == {"familiar", "target"},
                f"Duplicate/missing historical rows: {cohort}")
        seeds = sorted({s for s, a in keyed})
        require(len(seeds) == 8 and all((s, a) in keyed for s in seeds for a in ("familiar", "target")),
                f"Historical paired-seed integrity failure: {cohort}")
        for seed in seeds:
            for arm in ("familiar", "target"):
                m = keyed[seed, arm][field]
                vals = {k: valid_rate(m[k], f"{cohort} {seed} {arm} {k}") for k in OLD_MEASURES}
                require(abs(sum(vals[k] for k in ("choice_accuracy", "wrong_role_rate", "wrong_event_rate")) - 1) < 2e-5,
                        f"Incorrect 4-way error decomposition: {cohort}/{seed}/{arm}")
                require(m["events"] * 2 == m["questions"], f"Incorrect event/question count: {cohort}")
                records.append(dict(cohort=cohort, seed=seed, arm=arm, **vals))
    return records, digests


def audit_new(root: Path):
    records = []
    digests = {}
    data_digests = set()
    for seed in SEEDS:
        for method in METHODS:
            old_rel = f"reports/exp102/v2_familiar_pilot/seed{seed}_{method}.json"
            ext_rel = f"reports/exp102/v2_budget_extension/seed{seed}_{method}_step3200.json"
            p0, p1 = root / old_rel, root / ext_rel
            j0, j1 = load_json(p0), load_json(p1)
            digests[old_rel], digests[ext_rel] = file_sha(p0), file_sha(p1)
            for j in (j0, j1):
                require(j["seed"] == seed and j["method"] == method,
                        f"Wrong seed/policy identity: {seed}/{method}")
                require(j["protected_accessed"] is False,
                        f"Unexpected protected-access status: {seed}/{method}")
                data_digests.add(j["data_digest"])
            require(j0["steps"] == 1600 and j1["start_step"] == 1600 and j1["steps"] == 3200,
                    f"Endpoint mismatch: {seed}/{method}")
            require(j1["source_records"] == j0["records"],
                    f"Extension source trajectory differs from original: {seed}/{method}")
            require(j1["LI_cases_generated"] is False and j1["reserved_set_cases_generated"] is False,
                    f"Protected / composition generation flag set: {seed}/{method}")
            require(j1["evidence_status"] == "EXPLORATORY_ONLY",
                    f"Extension incorrectly marked confirmatory: {seed}/{method}")
            for stage, row in ((1600, j0["records"][-1]), (3200, j1["extension_records"][-1])):
                require(row["step"] == stage, f"Unexpected final milestone: {seed}/{method}/{stage}")
                dev = row["dev"]
                vals = {k: valid_rate(dev[k], f"{seed} {method} {stage} {k}") for k in NEW_MEASURES}
                # Both event-hit fields are question-averaged, not all-or-none rates.
                event_hit = (vals["correct_event_identity_first"] + vals["correct_event_identity_second"]) / 2
                role_copy = (vals["role_copy_first"] + vals["role_copy_second"]) / 2
                both_per_event = (vals["both_correct_first"] + vals["both_correct_second"]) / 2
                wrong_role = event_hit - vals["accuracy4"]
                require(-2e-5 < wrong_role < 1.00002,
                        f"Role-error identity violated: {seed}/{method}/{stage}, gap={wrong_role}")
                require(vals["C_query"] <= min(vals["both_correct_first"], vals["both_correct_second"]) + 2e-5,
                        f"All-four correctness exceeds per-event correctness: {seed}/{method}/{stage}")
                records.append(dict(seed=seed, method=method, step=stage, **vals,
                    wrong_event_rate=max(0., 1-event_hit),
                    wrong_role_rate=max(0., wrong_role),
                    same_answer_rate=role_copy,
                    per_event_both_correct=both_per_event))
    require(len(data_digests) == 1, f"Inconsistent EXP-102 training data digests: {data_digests}")
    return records, digests


def old_summary(rows):
    summary = []
    for cohort in OLD:
        for arm in ("familiar", "target"):
            g = [r for r in rows if r["cohort"] == cohort and r["arm"] == arm]
            assert len(g) == 8
            summary.append(dict(cohort=cohort, arm=arm, **{k: mean(g,k) for k in OLD_MEASURES}))
    return summary


def new_summary(rows):
    summary = []
    for step in (1600, 3200):
        for method in METHODS:
            g = [r for r in rows if r["step"] == step and r["method"] == method]
            assert len(g) == 8
            summary.append(dict(step=step, method=method, **{
                k: mean(g,k) for k in ("accuracy4", "C_query", "per_event_both_correct",
                                   "wrong_event_rate", "wrong_role_rate", "same_answer_rate")
            }))
    return summary


def fmt(v): return f"{100*v:.1f}%"


def report_markdown(old_s, new_s, digests):
    lines = [
        "# EXP-103 / bridge-0: error taxonomy across historical exposure and EXP-102 batching",
        "",
        "**Status:** read-only, descriptive, post hoc. Historical protected results were already opened in the original studies; this audit never generates protected material.",
        "",
        "## Original targeted-exposure cohorts (historical held-out target condition)",
        "",
        "Each row is the **mean across eight paired seeds**. `same` is the rate of identical agent/patient answers for the queried target event; `both` is correctness for its two complementary questions.",
        "",
        "| Cohort | Arm | Question accuracy | Wrong event | Wrong role | Same answer | Both correct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in old_s:
        lines.append(f"| {r['cohort']} | {r['arm']} | {fmt(r['choice_accuracy'])} | {fmt(r['wrong_event_rate'])} | {fmt(r['wrong_role_rate'])} | {fmt(r['same_answer_rate'])} | {fmt(r['correct_both_rate'])} |")
    lines += [
        "", "### Exposure effects (target minus familiar; percentage points)", "",
        "| Cohort | Accuracy | Wrong event | Wrong role | Same answer | Both correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort in OLD:
        f = next(r for r in old_s if r["cohort"] == cohort and r["arm"] == "familiar")
        t = next(r for r in old_s if r["cohort"] == cohort and r["arm"] == "target")
        def delta(k): return f"{100*(t[k]-f[k]):+.1f}"
        lines.append(f"| {cohort} | {delta('choice_accuracy')} | {delta('wrong_event_rate')} | {delta('wrong_role_rate')} | {delta('same_answer_rate')} | {delta('correct_both_rate')} |")
    lines += [
        "", "## EXP-102 v2: familiar-syntax development (not protected)", "",
        "The current experiment measures four queries per sentence. `same` is the mean of first- and second-event role copying; `both/event` is mean per-event two-query correctness, **not** `C_query`.",
        "", "| Step | Sampling policy | Question accuracy | Wrong event | Wrong role | Same answer | Both/event | C_query (all four) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in new_s:
        lines.append(f"| {r['step']} | {r['method']} | {fmt(r['accuracy4'])} | {fmt(r['wrong_event_rate'])} | {fmt(r['wrong_role_rate'])} | {fmt(r['same_answer_rate'])} | {fmt(r['per_event_both_correct'])} | {fmt(r['C_query'])} |")
    lines += [
        "", "## Interpretation", "",
        "- The historical target-exposure cohorts generally reduce **both wrong-event and wrong-role errors**. Do not infer that the primary old bottleneck was identical-answer copying: that metric is not systematically reduced.",
        "- EXP-102 frequently shows nearly perfect event-participant identification alongside high identical-answer copying before the quartet-trained models learn roles. This is a **different observed error profile**, even though both studies involve semantic roles.",
        "- A *shared functional explanation* remains a hypothesis; neither the same circuit nor even an identical behavioral transition has been established.",
        "- Comparisons are **not** head-to-head effect sizes: different grammars, names, held-out regimes, exposure interventions, training schedules, and endpoints. Historical `both` is one event (two queries), current `C_query` is two events (four queries).",
        "- Each cohort contains eight paired initialization seeds. Multiple cohorts share a research lineage; do **not** treat all observations as independent confirmatory replications.",
        "- Ordinary transformer cross-entropy training sums per-example gradients. Co-occurrence in a batch cannot provide direct cross-example attention; quartet effects arise from the update sampling distribution / gradient statistics rather than a model literally reading four examples together.",
        "", "## Next test (not performed)", "",
        "Design a frozen **2×2 study** on a single grammar: targeted versus familiar exposure × quartet versus distributed-complementary-question update policy. Fix a four-update source-question exposure window, source pool, optimizer updates, seed pairing and calibration data. Group 16 source sentences × 4 queries per update in the quartet arm; distribute the same 64 sources × 4 queries across four balanced updates in the comparison arm (Latin-square question rotation), matching all source×question pairs per four-update window. The per-update source diversity still differs inherently; document this. Measure first/second event errors and role copying on the **same development split** in all arms. Freeze before training. Only open reserved evaluation if predeclared competence gates pass.",
        "", "## Input file fingerprints (SHA-256)", "",
    ]
    lines += [f"- `{path}`: `{digest}`" for path, digest in sorted(digests.items())]
    return "\n".join(lines) + "\n"


def save_csv(path: Path, rows):
    keys = list(rows[0])
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    p.add_argument("--write", action="store_true", help="Create derived Markdown and CSV, never edit original reports")
    args=p.parse_args()
    old, hd0 = audit_old(args.root)
    new, hd1 = audit_new(args.root)
    old_s,new_s=old_summary(old),new_summary(new)
    markdown=report_markdown(old_s,new_s,{**hd0,**hd1})
    if args.write:
        out=args.root/"reports/exp103_bridge"
        out.mkdir(parents=True,exist_ok=True)
        targets=[out/"bridge0_error_taxonomy.md",out/"historical_per_seed.csv",out/"current_per_seed.csv"]
        if any(q.exists() for q in targets):
            raise FileExistsError(f"Derived outputs already exist; refusing overwrite: {[str(q) for q in targets if q.exists()]}")
        targets[0].write_text(markdown,encoding="utf-8")
        save_csv(targets[1],old)
        save_csv(targets[2],new)
        print("Saved new derived files:",*(str(t) for t in targets),sep="\n  ")
    else:
        print(markdown.split("## Input file fingerprints")[0])
    print("PASS: 5 historical cohorts (80 rows), 8 v2 pairs at 1600/3200 (32 rows), hash + error decomposition audits")
    print("READ-ONLY: original reports and checkpoints unchanged; no generation, training or evaluation.")

if __name__ == "__main__":
    main()
