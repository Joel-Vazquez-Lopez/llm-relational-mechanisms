#!/usr/bin/env python3
"""Read-only audit of EXP-102's fixed eight-seed exploratory replication.

Reads the saved protocol and 32 original JSON reports. Does not load models,
regenerate examples, train, open protected data, or modify repository files.
If --write is given, creates NEW analysis files without overwriting anything.

Run from repository root:
    python3 scripts/exp102_evidence_audit.py
    python3 scripts/exp102_evidence_audit.py --write
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

SEEDS = (10301, 10303, 10309, 10313, 10321, 10331, 10333, 10337)
METHODS = ("quartet", "balanced_unpaired", "role_pair", "event_pair")
MILESTONES = (400, 800, 1200, 1600)
BOOTSTRAP_SEED = 102102
BOOTSTRAP_N = 20_000


def load(root: Path):
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing {protocol_path}; run the original replication first")
    protocol = json.loads(protocol_path.read_text())
    assert tuple(protocol["fixed_seeds"]) == SEEDS, "Seed list changed"
    assert tuple(protocol["methods"]) == METHODS, "Methods changed"
    assert tuple(protocol["milestones"]) == MILESTONES, "Milestones changed"
    assert protocol["fixed_steps"] == 1600, "Training budget changed"
    assert protocol["no_protected_evaluation"] is True

    rows = {}
    for seed, method in itertools.product(SEEDS, METHODS):
        path = root / f"seed{seed}_{method}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing report: {path}")
        doc = json.loads(path.read_text())
        assert doc["study"] == "EXP-102"
        assert doc["seed"] == seed and doc["method"] == method
        assert doc["steps"] == 1600
        assert doc["data_digest"] == protocol["data_digest"], f"Dataset changed: {path}"
        assert doc["runner_sha256"] == protocol["runner_sha256"], f"Runner changed: {path}"
        assert doc["source_files_sha256"] == protocol["source_files_sha256"], f"Sources changed: {path}"
        records = doc["records"]
        assert [r["step"] for r in records] == list(MILESTONES), f"Unexpected milestones: {path}"
        for r in records:
            for split in ("id", "ood"):
                x = r[split]
                for key in ("accuracy4", "C_query"):
                    assert math.isfinite(x[key]) and 0 <= x[key] <= 1
                for event in ("first", "second"):
                    for key in ("event_member", "role_copy", "both_correct", "role_signal_positive"):
                        v = x[event][key]
                        assert math.isfinite(v) and 0 <= v <= 1
        assert math.isclose(doc["primary_ood_C_query"], records[-1]["ood"]["C_query"], abs_tol=1e-12)
        rows[seed, method] = doc
    return protocol, rows


def percentile(v, q):
    p = (len(v) - 1) * q
    low, high = math.floor(p), math.ceil(p)
    return v[low] + (p - low) * (v[high] - v[low])


def paired_interval(diffs):
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs)
                   for _ in range(BOOTSTRAP_N))
    return (percentile(means, .025), percentile(means, .975))


def exact_two_sided_signflip_p(diffs):
    # Seed-level permutation/sign-flip test, removing exact ties. With five
    # strictly positive and three tied effects, p=2/32=0.0625.
    nonzero = [v for v in diffs if v != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    possible = [abs(sum(sign * val for sign, val in zip(signs, nonzero)))
                for signs in itertools.product((-1, 1), repeat=len(nonzero))]
    return sum(value >= observed - 1e-12 for value in possible) / len(possible)


def get(rows, seed, method, step, key="C_query", split="ood"):
    rec = next(r for r in rows[seed, method]["records"] if r["step"] == step)
    return rec[split][key]


def analyze(protocol, rows):
    final = {m: [get(rows, s, m, 1600) for s in SEEDS] for m in METHODS}
    diffs = [a - b for a, b in zip(final["quartet"], final["balanced_unpaired"])]
    lo, hi = paired_interval(diffs)
    primary = {
        "contrast": "quartet minus balanced_unpaired; OOD C_query at 1600 updates",
        "seed_differences": {str(s): d for s, d in zip(SEEDS, diffs)},
        "mean_paired_difference": statistics.mean(diffs),
        "bootstrap_95_percentile_CI": [lo, hi],
        "bootstrap_resamples": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
        "two_sided_exact_signflip_p_exploratory": exact_two_sided_signflip_p(diffs),
    }
    curves = []
    for step in MILESTONES:
        for m in METHODS:
            vals = [get(rows, s, m, step) for s in SEEDS]
            curves.append({"step": step, "method": m,
                           "mean_C_query": statistics.mean(vals),
                           "min_C_query": min(vals), "max_C_query": max(vals),
                           "seeds_at_least_0_95": sum(v >= .95 for v in vals),
                           "seeds_at_1_0": sum(v == 1.0 for v in vals)})
    individual = []
    for seed in SEEDS:
        for m in METHODS:
            rec = rows[seed, m]["records"][-1]
            x = rec["ood"]
            individual.append({
                "seed": seed, "method": m, "C_query": x["C_query"],
                "accuracy4": x["accuracy4"],
                "first_role_copy": x["first"]["role_copy"],
                "second_role_copy": x["second"]["role_copy"],
                "first_event_member": x["first"]["event_member"],
                "second_event_member": x["second"]["event_member"],
            })
    return {"study": "EXP-102", "status": "POST_HOC_EXPLORATORY_ANALYSIS",
            "primary": primary, "milestone_summaries": curves,
            "seed_final_records": individual,
            "warnings": [
                "The synthetic task/ID/OOD data family was repeatedly explored before this seed panel.",
                "Different methods change unique source count and within-update co-occurrence; not a uniquely identified mechanism.",
                "The statistical replication unit is initialization seed (n=8), not 256 sentences per seed.",
                "Name-set-OOD tests recombination of familiar vocabulary/grammar, NOT unseen syntax.",
                "Performance uses four-name candidate-restricted decoding, not natural-language generation.",
                "Learning-curve milestone comparisons are secondary/descriptive; primary step was 1600.",
            ]}


def render(result):
    p = result["primary"]
    out = ["# EXP-102 — eight-seed evidence audit",
           "", "**Status:** exploratory, existing saved reports only; no new protected evidence.",
           "", "## Primary comparison",
           "", "Quartet minus balanced-unpaired, on four-name-set-OOD `C_query` at step 1600:",
           "", f"- Mean paired difference: **{p['mean_paired_difference']*100:+.2f} percentage points**",
           f"- Seed-bootstrap 95% percentile interval: **[{p['bootstrap_95_percentile_CI'][0]*100:+.2f}, {p['bootstrap_95_percentile_CI'][1]*100:+.2f}] pp**",
           f"- Wins/ties/losses: **{p['wins']}/{p['ties']}/{p['losses']}**",
           f"- Exploratory exact two-sided sign-flip p: **{p['two_sided_exact_signflip_p_exploratory']:.4f}**",
           "", "This p-value should not be used as a preregistered confirmatory significance claim.",
           "", "## Eight-seed learning curves", "",
           "| Updates | Quartet | Balanced unpaired | Role pair | Event pair |",
           "|---:|---:|---:|---:|---:|",]
    for step in MILESTONES:
        row = {r["method"]: r for r in result["milestone_summaries"] if r["step"] == step}
        out.append("| " + str(step) + " | " + " | ".join(
            f"{row[m]['mean_C_query']*100:.1f}%"
            for m in METHODS) + " |")
    out.extend(["", "## Seed-level final OOD C_query", "",
                "| Seed | Quartet | Balanced unpaired | Role pair | Event pair |",
                "|---|---:|---:|---:|---:|"])
    for s in SEEDS:
        found = {r["method"]: r for r in result["seed_final_records"] if r["seed"] == s}
        out.append(f"| {s} | " + " | ".join(
            f"{found[m]['C_query']*100:.1f}%" for m in METHODS) + " |")
    out.extend(["", "## Interpretation safeguards", ""])
    for warning in result["warnings"]:
        out.append(f"- {warning}")
    out.extend(["", "**Next appropriate test:** a separately frozen, syntax-compositional",
                "generalization evaluation with prevalidated labeling and anti-shortcut controls.", ""])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("reports/exp102/fresh_seed_replication"),
                    help="Directory holding original protocol.json and seed JSON reports")
    ap.add_argument("--write", action="store_true", help="Create new summary files; never overwrite")
    args = ap.parse_args()
    protocol, rows = load(args.root)
    result = analyze(protocol, rows)
    report = render(result)
    print(report)
    if args.write:
        out = args.root.parent / "evidence_audit"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite analysis output directory: {out}")
        out.mkdir(parents=True, exist_ok=False)
        (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        (out / "summary.md").write_text(report)
        with (out / "learning_curves.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["milestone_summaries"][0].keys()))
            writer.writeheader()
            writer.writerows(result["milestone_summaries"])
        with (out / "seed_final.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["seed_final_records"][0].keys()))
            writer.writeheader()
            writer.writerows(result["seed_final_records"])
        print(f"\nCREATED NEW FILES IN: {out}")
    else:
        print("\nREAD-ONLY: No files written. Add --write to archive derived summaries.")


if __name__ == "__main__":
    main()
