
"""ML-052 retrospective question-conditioned selection diagnostic.

Read-only with respect to experimental source data.
No model inference, parameter fitting, or new protected testing.

Measures how changing the requested semantic role affects
the agent-versus-patient log-likelihood contrast.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("qwen7b", "llama3b")
SPLITS = ("calibration", "protected")
OUTPUT = RESULTS / "ml052_question_effect_diagnostic.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(model, split, manifest):
    path = RESULTS / f"ml052_{model}_{split}.json"
    data = json.loads(path.read_text())

    assert data["complete"] is True
    assert data["model"] == model
    assert data["split"] == split
    assert (
        data["manifest_sha256"]
        == manifest["manifest_sha256"]
    )

    assert data["script_sha256"] == sha256(
        ROOT / "scripts/ml052_predict.py"
    )

    expected = manifest["cases"][split]
    records = data["records"]

    assert len(records) == len(expected)

    for row, case in zip(records, expected):
        for key, value in case.items():
            assert row[key] == value, (row["id"], key)

        assert set(row["scores"]) == set(row["pair"])
        assert len(set(row["token_counts"].values())) == 1

    return records


def paired_records(records):
    groups = defaultdict(dict)

    for row in records:
        key = (
            row["split"],
            tuple(row["pair"]),
            row["verb"],
            row["direction"],
            row["construction"],
        )

        assert row["role"] not in groups[key]
        groups[key][row["role"]] = row

    output = []

    for key, roles in groups.items():
        assert set(roles) == {"agent", "patient"}

        agent_q = roles["agent"]
        patient_q = roles["patient"]

        assert agent_q["passage"] == patient_q["passage"]
        assert agent_q["agent"] == patient_q["agent"]
        assert agent_q["patient"] == patient_q["patient"]

        agent = agent_q["agent"]
        patient = agent_q["patient"]

        # Always orient the margin as agent minus patient,
        # regardless of which role the question requests.
        margin_agent_question = (
            agent_q["scores"][agent]
            - agent_q["scores"][patient]
        )

        margin_patient_question = (
            patient_q["scores"][agent]
            - patient_q["scores"][patient]
        )

        if (
            margin_agent_question == 0
            or margin_patient_question == 0
        ):
            raise ValueError(
                f"Exact score tie in paired case {key}"
            )

        delta = (
            margin_agent_question
            - margin_patient_question
        )

        answer_agent_question = (
            agent if margin_agent_question > 0
            else patient
        )

        answer_patient_question = (
            agent if margin_patient_question > 0
            else patient
        )

        first = min(
            (agent, patient),
            key=lambda name: agent_q[
                "passage"
            ].index(name),
        )

        output.append({
            "case": "|".join(map(str, key)),
            "construction": agent_q["construction"],
            "lexical_group": "|".join(
                (*agent_q["pair"], agent_q["verb"])
            ),
            "direction": agent_q["direction"],
            "delta": float(delta),
            "agent_question_margin": float(
                margin_agent_question
            ),
            "patient_question_margin": float(
                margin_patient_question
            ),
            "correct_switch": (
                answer_agent_question == agent
                and answer_patient_question == patient
            ),
            "wrong_switch": (
                answer_agent_question == patient
                and answer_patient_question == agent
            ),
            "same_choice": (
                answer_agent_question
                == answer_patient_question
            ),
            "first_both": (
                answer_agent_question == first
                and answer_patient_question == first
            ),
        })

    assert len(output) * 2 == len(records)
    return output


def summarize(rows):
    if not rows:
        raise ValueError("Empty summary group")

    deltas = [r["delta"] for r in rows]

    return {
        "n_pairs": len(rows),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "positive_delta": sum(
            r["delta"] > 0 for r in rows
        ),
        "negative_delta": sum(
            r["delta"] < 0 for r in rows
        ),
        "correct_switches": sum(
            r["correct_switch"] for r in rows
        ),
        "wrong_switches": sum(
            r["wrong_switch"] for r in rows
        ),
        "same_choice": sum(
            r["same_choice"] for r in rows
        ),
        "first_both": sum(
            r["first_both"] for r in rows
        ),
    }


def main():
    manifest = json.loads(
        (RESULTS / "ml052_design_manifest.json").read_text()
    )

    report = {
        "analysis": "retrospective exploratory diagnostic",
        "experiment": "ML-052",
        "manifest_sha256": manifest["manifest_sha256"],
        "interpretation": (
            "Not a new prospective success criterion. "
            "Pairs within lexical groups are dependent."
        ),
        "models": {},
    }

    for model in MODELS:
        print("\n" + "=" * 68)
        print(model.upper())
        print("=" * 68)

        report["models"][model] = {}

        for split in SPLITS:
            records = load_records(
                model, split, manifest
            )

            pairs = paired_records(records)

            assert len(pairs) == (
                48 if split == "calibration"
                else 60
            )

            constructions = sorted({
                row["construction"] for row in pairs
            })

            result = {
                "overall": summarize(pairs),
                "by_construction": {},
            }

            print(f"\n{split.upper()}")
            print(
                "Construction     Pairs  Mean D    "
                "D>0  Correct switches  Wrong switches"
            )

            for construction in constructions:
                subset = [
                    row for row in pairs
                    if row["construction"] == construction
                ]

                metrics = summarize(subset)
                result["by_construction"][
                    construction
                ] = metrics

                print(
                    f"{construction:16s} "
                    f"{metrics['n_pairs']:3d}  "
                    f"{metrics['mean_delta']:+8.3f} "
                    f"{metrics['positive_delta']:4d} "
                    f"{metrics['correct_switches']:8d} "
                    f"{metrics['wrong_switches']:14d}"
                )

            total = result["overall"]

            print(
                "OVERALL:",
                f"D>0 {total['positive_delta']}/"
                f"{total['n_pairs']},",
                f"correct switches "
                f"{total['correct_switches']},",
                f"wrong switches "
                f"{total['wrong_switches']},",
                f"same choice {total['same_choice']}"
            )

            if split == "protected":
                holdout = [
                    row for row in pairs
                    if row["construction"]
                    == "passive_cleft"
                ]

                by_group = defaultdict(list)

                for row in holdout:
                    by_group[
                        row["lexical_group"]
                    ].append(row)

                print("\nPASSIVE-CLEFT LEXICAL GROUPS")

                group_means = {}

                for group, group_rows in sorted(
                    by_group.items()
                ):
                    assert len(group_rows) == 2

                    group_delta = mean(
                        row["delta"]
                        for row in group_rows
                    )

                    group_means[group] = group_delta

                    print(
                        f"  {group:38s}"
                        f"mean D={group_delta:+9.3f}"
                    )

                result[
                    "holdout_lexical_group_means"
                ] = group_means

            report["models"][model][split] = result

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("\nDIAGNOSTIC COMPLETE")
    print("Saved:", OUTPUT)
    print("No model inference performed.")


if __name__ == "__main__":
    main()
