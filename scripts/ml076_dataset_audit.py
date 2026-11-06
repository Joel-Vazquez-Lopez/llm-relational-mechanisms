
"""ML-076: Read-only dataset integrity and shortcut audit.

Audits ML-075 training, calibration and exposure generators.

Does NOT:
- train models
- load model checkpoints
- generate protected examples
- evaluate protected data
- modify ML-075

Run from repository root:
    python3 scripts/ml076_dataset_audit.py
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import ml075_grammar_benchmark as ml


PROTOCOL = Path(
    "reports/ml075/frozen_protocol.json"
)

OUTPUT = Path(
    "reports/ml076/dataset_audit.json"
)


def tv_distance(a, b):
    """Total variation distance between distributions."""
    keys = set(a) | set(b)

    na = sum(a.values())
    nb = sum(b.values())

    return 0.5 * sum(
        abs(a[k] / na - b[k] / nb)
        for k in keys
    )


def answer_rank(case):
    """Correct answer's rank among four alphabetical names."""
    candidates = sorted(
        case.target_pair + case.distractor_pair
    )

    assert len(candidates) == 4
    assert len(set(candidates)) == 4

    return candidates.index(
        ml.gold(case)
    )


def inspect_integrity(cases):
    assert len(cases) % 2 == 0

    prompts = Counter()
    answers = defaultdict(set)

    for case in cases:
        prompt = tuple(
            ml.prompt(case)
        )

        prompts[prompt] += 1
        answers[prompt].add(
            ml.gold(case)
        )

    role_pairs_valid = all(
        agent.role == "agent"
        and patient.role == "patient"
        and replace(
            agent, role="patient"
        ) == patient
        for agent, patient in zip(
            cases[::2],
            cases[1::2],
        )
    )

    assert role_pairs_valid, (
        "Incorrect agent/patient pairing"
    )

    conflicts = sum(
        len(labels) > 1
        for labels in answers.values()
    )

    assert conflicts == 0, (
        "Identical prompts have conflicting labels"
    )

    events = cases[::2]

    unique_events = {
        replace(c, role="agent")
        for c in events
    }

    return {
        "questions": len(cases),
        "event_instances": len(events),
        "unique_event_instances": len(unique_events),
        "unique_prompts": len(prompts),
        "repeated_prompt_instances": sum(
            count - 1
            for count in prompts.values()
        ),
        "maximum_prompt_multiplicity": max(
            prompts.values()
        ),
        "conflicting_labels": conflicts,
        "role_pairs_valid": role_pairs_valid,
        "distinct_target_pairs": len({
            c.target_pair for c in events
        }),
        "distinct_distractor_pairs": len({
            c.distractor_pair for c in events
        }),
        "distinct_name_quadruples": len({
            frozenset(
                c.target_pair + c.distractor_pair
            )
            for c in events
        }),
    }


def grammar_coverage(cases):
    """Measure coverage within each target grammar cell."""
    groups = defaultdict(list)

    for case in cases[::2]:
        key = (
            f"{case.target_pos}:"
            f"{case.target_voice}"
        )
        groups[key].append(case)

    summary = {}

    for cell, events in sorted(groups.items()):
        pairs = Counter(
            c.target_pair
            for c in events
        )

        names = Counter(
            name
            for c in events
            for name in c.target_pair
        )

        pair_verbs = {
            (c.target_pair, c.target_verb)
            for c in events
        }

        summary[cell] = {
            "events": len(events),
            "distinct_pairs": len(pairs),
            "pair_frequency_min": min(
                pairs.values()
            ),
            "pair_frequency_max": max(
                pairs.values()
            ),
            "target_name_frequency_min": min(
                names.values()
            ),
            "target_name_frequency_max": max(
                names.values()
            ),
            "distinct_pair_verb_cells": len(
                pair_verbs
            ),
            "possible_pair_verb_cells": (
                len(pairs) * len(ml.VERBS)
            ),
            "frame_counts": dict(Counter(
                c.frame for c in events
            )),
            "target_direction_counts": dict(
                Counter(
                    c.target_dir for c in events
                )
            ),
            "distractor_direction_counts": dict(
                Counter(
                    c.distractor_dir for c in events
                )
            ),
            "distractor_voice_counts": dict(
                Counter(
                    c.distractor_voice
                    for c in events
                )
            ),
            "target_verb_counts": dict(
                Counter(
                    c.target_verb
                    for c in events
                )
            ),
        }

    return summary


def shortcut_baseline(train, calibration, features):
    """Train a candidate-rank majority baseline.

    No access to sentence tokens or participant relationships.
    Features are generator metadata only.
    """

    groups = defaultdict(Counter)

    global_counts = Counter(
        answer_rank(c) for c in train
    )

    fallback = global_counts.most_common(1)[0][0]

    for case in train:
        key = tuple(
            getattr(case, feature)
            for feature in features
        )

        groups[key][answer_rank(case)] += 1

    correct = []
    unseen_groups = 0

    for case in calibration:
        key = tuple(
            getattr(case, feature)
            for feature in features
        )

        if key not in groups:
            unseen_groups += 1
            prediction = fallback
        else:
            prediction = (
                groups[key]
                .most_common(1)[0][0]
            )

        correct.append(
            prediction == answer_rank(case)
        )

    both = sum(
        correct[i] and correct[i + 1]
        for i in range(
            0, len(correct), 2
        )
    )

    return {
        "features": list(features),
        "calibration_question_accuracy": (
            sum(correct) / len(correct)
        ),
        "calibration_correct_both_rate": (
            both / (len(correct) // 2)
        ),
        "unseen_feature_groups": unseen_groups,
    }


def load_original_protocol():
    frozen = json.loads(
        PROTOCOL.read_text()
    )

    assert frozen["experiment"] == "ML-075"

    source_hash = hashlib.sha256(
        Path(ml.__file__).read_bytes()
    ).hexdigest()

    assert (
        source_hash == frozen["code_sha256"]
    ), (
        "ML-075 source differs from frozen version"
    )

    expected = ml.sha({
        k: v
        for k, v in frozen.items()
        if k != "sha256"
    })

    assert expected == frozen["sha256"], (
        "Frozen protocol checksum mismatch"
    )

    return frozen, source_hash


def reconstruct_nonprotected_data(frozen):
    train_pairs = [
        tuple(p)
        for p in frozen["training_pair_ids"]
    ]

    calibration_pairs = [
        tuple(p)
        for p in frozen["calibration_pair_ids"]
    ]

    # Inspect only identifiers to verify split integrity.
    # Do not generate protected test cases.
    protected_ids = [
        tuple(p)
        for p in frozen["protected_pair_ids"]
    ]

    assert set(train_pairs).isdisjoint(
        calibration_pairs
    )

    assert set(train_pairs).isdisjoint(
        protected_ids
    )

    assert set(calibration_pairs).isdisjoint(
        protected_ids
    )

    train = ml.balanced_base(
        train_pairs
    )

    calibration = ml.balanced_base(
        calibration_pairs
    )

    doses = {
        "control": ml.exposure(
            train_pairs[:4],
            ml.VERBS[:2],
            train_pairs,
            (0, "passive"),
            broad=False,
        ),
        "sparse": ml.exposure(
            train_pairs[:4],
            ml.VERBS[:2],
            train_pairs,
            ml.TARGET,
            broad=False,
        ),
        "broad": ml.exposure(
            train_pairs[:24],
            ml.VERBS[:4],
            train_pairs,
            ml.TARGET,
            broad=True,
        ),
    }

    assert len(train) == (
        frozen["n_examples"]["base"]
    )

    assert len(calibration) == (
        frozen["n_examples"]["calibration"]
    )

    for arm, cases in doses.items():
        assert len(cases) == (
            frozen["n_examples"][f"dose_{arm}"]
        )

    return (
        train,
        calibration,
        doses,
        len(train_pairs),
        len(calibration_pairs),
        len(protected_ids),
    )


def main():
    frozen, source_hash = load_original_protocol()

    (
        train,
        calibration,
        doses,
        ntrain_pairs,
        ncal_pairs,
        nprotected_pairs,
    ) = reconstruct_nonprotected_data(frozen)

    train_prompts = {
        tuple(ml.prompt(c))
        for c in train
    }

    cal_prompts = {
        tuple(ml.prompt(c))
        for c in calibration
    }

    overlap = len(
        train_prompts & cal_prompts
    )

    assert overlap == 0, (
        "Exact training/calibration prompt overlap"
    )

    train_integrity = inspect_integrity(train)
    cal_integrity = inspect_integrity(calibration)

    train_coverage = grammar_coverage(train)
    cal_coverage = grammar_coverage(calibration)

    steps = frozen["training"]["steps"]
    dose_batch = frozen["training"]["exposure_batch"]

    total_dose_draws = (
        steps * dose_batch
    )

    exposure_stats = {}

    for arm, cases in doses.items():
        integrity = inspect_integrity(cases)

        exposure_stats[arm] = {
            **integrity,
            "expected_draws_per_row": (
                total_dose_draws / len(cases)
            ),
            "expected_draws_per_unique_prompt": (
                total_dose_draws
                / integrity["unique_prompts"]
            ),
            "target_cells": sorted({
                f"{c.target_pos}:{c.target_voice}"
                for c in cases
            }),
            "target_verb_diversity": len({
                c.target_verb for c in cases
            }),
        }

    feature_sets = [
        (),
        ("role",),
        (
            "role",
            "target_pos",
            "target_voice",
            "frame",
        ),
        (
            "role",
            "target_pos",
            "target_voice",
            "frame",
            "target_verb",
        ),
    ]

    shortcuts = [
        shortcut_baseline(
            train, calibration, features
        )
        for features in feature_sets
    ]

    train_ranks = Counter(
        answer_rank(c) for c in train
    )

    cal_ranks = Counter(
        answer_rank(c) for c in calibration
    )

    train_names = Counter(
        ml.gold(c) for c in train
    )

    cal_names = Counter(
        ml.gold(c) for c in calibration
    )

    report = {
        "experiment": "ML-076",
        "status": "OFFLINE_GENERATOR_AUDIT",
        "protected_evaluation_opened": False,
        "protected_cases_generated": False,
        "ml075_protocol_sha256": frozen["sha256"],
        "ml075_script_sha256": source_hash,
        "split_counts": {
            "train_pairs": ntrain_pairs,
            "calibration_pairs": ncal_pairs,
            "protected_pair_ids_not_examined": (
                nprotected_pairs
            ),
            "overlapping_train_calibration_prompts": (
                overlap
            ),
        },
        "train": {
            "integrity": train_integrity,
            "by_grammar": train_coverage,
        },
        "calibration": {
            "integrity": cal_integrity,
            "by_grammar": cal_coverage,
        },
        "exposure": exposure_stats,
        "answer_name_tv_train_cal": tv_distance(
            train_names, cal_names
        ),
        "candidate_rank_tv_train_cal": tv_distance(
            train_ranks, cal_ranks
        ),
        "candidate_rank_distributions": {
            "train": dict(train_ranks),
            "calibration": dict(cal_ranks),
        },
        "metadata_shortcut_baselines": shortcuts,
        "limitations": [
            "No model checkpoints accessed",
            "Protected cases were not generated",
            "Metadata shortcuts are not exhaustive",
            "Sampling expectation differs from actual draw counts",
            "Calibration uses a smaller participant-pair pool",
        ],
    }

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print(
        "ML-076: OFFLINE dataset audit"
    )

    print(
        f"Training: {len(train)} questions; "
        f"{len(train_prompts)} unique prompts"
    )

    print(
        f"Calibration: {len(calibration)} questions; "
        f"{len(cal_prompts)} unique prompts"
    )

    for name, integrity, coverage in [
        ("train", train_integrity, train_coverage),
        ("calibration", cal_integrity, cal_coverage),
    ]:
        print(
            f"\n{name}: repeated prompts="
            f"{integrity['repeated_prompt_instances']}"
        )

        for cell, item in coverage.items():
            print(
                f"  {cell}: "
                f"events={item['events']} "
                f"pairs={item['distinct_pairs']} "
                f"pair-range="
                f"{item['pair_frequency_min']}-"
                f"{item['pair_frequency_max']} "
                f"pair-verb-cells="
                f"{item['distinct_pair_verb_cells']}/"
                f"{item['possible_pair_verb_cells']}"
            )

    print("\nExposure repetitions:")

    for arm, item in exposure_stats.items():
        print(
            f"  {arm}: "
            f"questions={item['questions']} "
            f"unique={item['unique_prompts']} "
            f"expected draws/row="
            f"{item['expected_draws_per_row']:.2f}"
        )

    print(
        "\nCandidate-rank train/cal TV:",
        round(
            report["candidate_rank_tv_train_cal"], 4
        ),
    )

    print(
        "Answer-name train/cal TV:",
        round(
            report["answer_name_tv_train_cal"], 4
        ),
    )

    print("\nMetadata shortcut baselines:")

    for item in shortcuts:
        print(
            f"  {item['features'] or ['constant rank']}: "
            f"accuracy="
            f"{item['calibration_question_accuracy']:.3f} "
            f"both="
            f"{item['calibration_correct_both_rate']:.3f}"
        )

    print("\nSaved:", OUTPUT)
    print(
        "Protected cases NOT generated or evaluated."
    )


if __name__ == "__main__":
    main()
