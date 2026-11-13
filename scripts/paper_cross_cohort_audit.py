#!/usr/bin/env python3
"""Final paper audit: aggregate already archived paired-role outcome reports.

NO model loading, training, checkpoint access, protected-sample generation,
or new evaluation. Reanalyses previously opened, archived protected summaries.
Run from repository root:
    python scripts/paper_cross_cohort_audit.py --phase preflight
    python scripts/paper_cross_cohort_audit.py --phase audit

The five cohorts have comparable *native four-name choice* rates:
- correct_both_rate: fraction of complementary role pairs both right
- wrong_role_rate: fraction of SINGLE questions predicting the opposite
  participant of the correct event; NOT a complete pair-role reversal
- same_answer_rate: fraction of paired questions choosing one identical name
- wrong_event_rate: fraction of SINGLE questions selecting other-event person
The ML-088 protected report does not contain these paired-role measures.
Its calibration-only oracle-restricted role study is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

COHORTS = {
    "ML-080": ("reports/ml080/protected_evaluation.json", "runs", "target_new_pairs_new_exposure_verbs"),
    "ML-084": ("reports/ml084/protected_evaluation.json", "runs", "target_new_pairs_new_exposure_verbs"),
    "ML-085": ("reports/ml085/protected_causal_evaluation.json", "runs", "protected_behavior"),
    "ML-086": ("reports/ml086/protected_evaluation.json", "runs", "full_protected_target"),
    "ML-099": ("reports/ml099_role_q/protected_evaluation.json", "rows", "protected_behavior"),
}
METRICS = (
    "choice_accuracy",
    "correct_both_rate",
    "wrong_role_rate",
    "wrong_event_rate",
    "same_answer_rate",
    "full_vocab_accuracy",
)
OUT = Path("reports/paper_cross_cohort/archived_error_taxonomy.json")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load():
    result = {}
    for cohort, (location, rows_key, metric_key) in COHORTS.items():
        path = Path(location)
        raw = path.read_bytes()
        source = json.loads(raw)
        assert source["experiment"] == cohort, f"{cohort}: wrong experiment ID"
        records = source[rows_key]
        by = {(r["seed"], r["arm"]): r[metric_key] for r in records}
        seeds = sorted({r["seed"] for r in records})
        assert len(seeds) == 8 and len(records) == 16
        assert len(by) == 16 and {a for _, a in by} == {"familiar", "target"}
        for seed in seeds:
            for arm in ("familiar", "target"):
                row = by[(seed, arm)]
                assert row["questions"] == 1536 and row["events"] == 768
                for metric in METRICS:
                    assert 0 <= row[metric] <= 1, (cohort, seed, arm, metric)
                assert abs(row["choice_accuracy"] + row["wrong_role_rate"] +
                           row["wrong_event_rate"] - 1) < 1e-9
                assert row["correct_both_rate"] <= row["choice_accuracy"] + 1e-9
        result[cohort] = {"records": by, "seeds": seeds,
                          "source_path": location, "source_sha256": sha256(raw)}
    return result


def aggregate(loaded):
    answer = {
        "status": "EXPLORATORY_REANALYSIS_OF_PREVIOUSLY_OPENED_ARCHIVED_PROTECTED_SUMMARIES",
        "not_new_protected_evaluation": True,
        "method_note": (
            "Five distinct cohorts, 8 matched seed pairs each, all same synthetic "
            "grammar/architecture. Primary descriptive comparison = paired differences "
            "target minus familiar per cohort. Wrong-role is question-level, NOT "
            "complete pair-role reversal. Same-answer is native four-choice, NOT "
            "oracle-restricted collapse. No significance or cross-cohort independence claim."
        ),
        "metrics": METRICS,
        "cohorts": {},
    }
    totals = {m: {"positive_seed_pairs": 0, "negative_seed_pairs": 0,
                  "zero_seed_pairs": 0} for m in METRICS}
    for cohort, obj in loaded.items():
        by, seeds = obj["records"], obj["seeds"]
        arms = {
            arm: {m: mean(by[(s, arm)][m] for s in seeds) for m in METRICS}
            for arm in ("familiar", "target")
        }
        paired = {}
        for metric in METRICS:
            diffs = [by[(s, "target")][metric] -
                     by[(s, "familiar")][metric] for s in seeds]
            paired[metric] = {
                "mean": mean(diffs),
                "by_seed": {str(seed): diff for seed, diff in zip(seeds, diffs)},
                "positive": sum(x > 0 for x in diffs),
                "negative": sum(x < 0 for x in diffs),
                "zero": sum(x == 0 for x in diffs),
            }
            totals[metric]["positive_seed_pairs"] += paired[metric]["positive"]
            totals[metric]["negative_seed_pairs"] += paired[metric]["negative"]
            totals[metric]["zero_seed_pairs"] += paired[metric]["zero"]
        answer["cohorts"][cohort] = {
            "source_path": obj["source_path"],
            "source_sha256": obj["source_sha256"],
            "n_paired_seeds": len(seeds),
            "means": arms, "paired": paired,
        }
    answer["total_seed_pairs_across_cohorts"] = 40
    answer["paired_sign_totals"] = totals
    return answer


def preflight():
    data = load()
    print("PREFLIGHT PASS: 5 archived cohorts, 8 paired seeds/cohort")
    for cohort, obj in data.items():
        print(cohort, obj["source_path"], obj["source_sha256"])
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("preflight", "audit"), required=True)
    args = p.parse_args()
    data = preflight()
    if args.phase == "preflight":
        return
    summary = aggregate(data)
    assert summary["paired_sign_totals"]["correct_both_rate"]["positive_seed_pairs"] == 40
    assert summary["paired_sign_totals"]["choice_accuracy"]["positive_seed_pairs"] == 40
    assert summary["paired_sign_totals"]["wrong_role_rate"]["negative_seed_pairs"] == 33
    assert summary["paired_sign_totals"]["same_answer_rate"]["positive_seed_pairs"] == 26

    print("\nCOHORT    BOTH Δ   WRONG-ROLE Δ  SAME-PERSON Δ")
    for cohort, details in summary["cohorts"].items():
        p = details["paired"]
        print(f"{cohort:<9} {100*p['correct_both_rate']['mean']:+7.2f} "
              f"{100*p['wrong_role_rate']['mean']:+12.2f} "
              f"{100*p['same_answer_rate']['mean']:+14.2f}")
    print("\nSign counts over 40 matched seeds:",
          json.dumps(summary["paired_sign_totals"], sort_keys=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing audit: {OUT}")
    summary["script_sha256"] = sha256(Path(__file__).read_bytes())
    OUT.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print("Saved:", OUT, "(archived reports only; no new model runs)")


if __name__ == "__main__":
    main()
