
"""ML-079: event-cue log-odds on familiar ML-075 calibration sentences.

Read-only diagnostic. No training or ML-075 protected-case generation.
Requires the frozen ML-075 and ML-078 scripts/protocols and 18 local
ML-075 checkpoints.

Run from repository root:
    python3 scripts/ml079_event_logit_diagnostics.py --phase plan
    python3 scripts/ml079_event_logit_diagnostics.py --phase preflight
    python3 scripts/ml079_event_logit_diagnostics.py --phase analyze
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import torch

import ml075_grammar_benchmark as ml
import ml078_event_cue_counterfactual as cf

SEEDS = (211, 223, 227, 229, 233, 239)
ARMS = ("control", "sparse", "broad")


def checksum(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs(path75, path78):
    frozen75 = cf.frozen_ml075(path75)
    assert tuple(frozen75["seeds"]) == SEEDS
    assert tuple(frozen75["arms"]) == ARMS

    frozen78 = json.loads(path78.read_text())
    assert frozen78["experiment"] == "ML-078"
    assert frozen78["code_sha256"] == file_sha(cf.__file__), (
        "ML-078 source changed"
    )
    assert frozen78["ml075_protocol_sha256"] == frozen75["sha256"]
    assert checksum({
        k: v for k, v in frozen78.items()
        if k != "sha256"
    }) == frozen78["sha256"], "ML-078 protocol changed"

    return frozen75, frozen78


def protocol_document(frozen75, frozen78):
    return {
        "experiment": "ML-079",
        "code_sha256": file_sha(__file__),
        "ml075_protocol_sha256": frozen75["sha256"],
        "ml078_protocol_sha256": frozen78["sha256"],
        "data": (
            "The same 1152 paired-role, two-active-event "
            "calibration questions as ML-078"
        ),
        "intervention": (
            "first/second query-cue substitution; "
            "every other prompt token fixed"
        ),
        "arms": ARMS,
        "seeds": SEEDS,
        "primary_diagnostics": [
            "event group log-odds for first-cue and second-cue queries",
            "difference of event log-odds across cues (correct direction: positive)",
            "both cues place >50% of four-name probability in the correct event",
            "wrong-event answers with positive versus nonpositive cue-direction shift",
            "within-target-event correct-name versus wrong-role logit margin",
        ],
        "rule": (
            "Event log-odds = logsumexp(first-event name logits) "
            "- logsumexp(second-event name logits)"
        ),
        "unit": (
            "Question pairs within the same checkpoint, "
            "descriptive seed-level summaries"
        ),
        "limits": (
            "Non-causal logit diagnostics; ML-075 competence gate "
            "failed; no protected evaluation"
        ),
    }


def freeze(path, frozen75, frozen78, create):
    doc = protocol_document(frozen75, frozen78)
    expected = json.loads(json.dumps({
        **doc, "sha256": checksum(doc)
    }))

    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n")

    if not path.exists():
        raise FileNotFoundError(f"Run --phase plan first: {path}")

    assert json.loads(path.read_text()) == expected, (
        "ML-079 code/protocol changed"
    )
    return expected["sha256"]


@torch.inference_mode()
def score_questions(model, cases, device):
    """Original four-name choice plus event-group log odds and target-role margin."""
    result = []
    model.eval()

    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]
        ids, readout, _, choices = ml.encode(batch)
        logits = model(ids.to(device), readout.to(device)).float()

        choices = choices.to(device)
        pred_id = choices.gather(
            1,
            logits.gather(1, choices).argmax(1, keepdim=True),
        )

        first_names = [
            c.target_pair if c.target_pos == 0 else c.distractor_pair
            for c in batch
        ]
        second_names = [
            c.distractor_pair if c.target_pos == 0 else c.target_pair
            for c in batch
        ]

        first_idx = torch.tensor([
            [ml.VOCAB[n] for n in pair]
            for pair in first_names
        ], device=device)
        second_idx = torch.tensor([
            [ml.VOCAB[n] for n in pair]
            for pair in second_names
        ], device=device)

        first_score = logits.gather(1, first_idx).logsumexp(dim=1)
        second_score = logits.gather(1, second_idx).logsumexp(dim=1)
        event_log_odds = first_score - second_score

        other_name = [
            next(n for n in c.target_pair if n != ml.gold(c))
            for c in batch
        ]
        gold_idx = torch.tensor(
            [ml.VOCAB[ml.gold(c)] for c in batch],
            device=device,
        )
        other_idx = torch.tensor(
            [ml.VOCAB[n] for n in other_name],
            device=device,
        )
        role_margin = (
            logits.gather(1, gold_idx[:, None]).flatten()
            - logits.gather(1, other_idx[:, None]).flatten()
        )

        for c, p, event, role in zip(
            batch,
            pred_id.flatten().cpu().tolist(),
            event_log_odds.cpu().tolist(),
            role_margin.cpu().tolist(),
        ):
            name = ml.VOCAB_WORDS[p]
            result.append({
                "name": name,
                "correct": name == ml.gold(c),
                "correct_event_by_top_name": name in c.target_pair,
                "event_log_odds_first_minus_second": event,
                "role_margin_gold_minus_opposite": role,
            })

    assert len(result) == len(cases)
    return result


def describe(pairs, first, second):
    assert len(pairs) == len(first) == len(second)
    records = []

    for (a, b), x, y in zip(pairs, first, second):
        assert a.target_pos == 0 and b.target_pos == 1
        assert a.role == b.role
        assert set(a.target_pair).isdisjoint(b.target_pair)

        efirst = x["event_log_odds_first_minus_second"]
        esecond = y["event_log_odds_first_minus_second"]
        shift = efirst - esecond
        wrong_second_event = not y["correct_event_by_top_name"]

        records.append({
            "role": a.role,
            "frame": b.frame,
            "first_name_correct": x["correct"],
            "second_name_correct": y["correct"],
            "unchanged": x["name"] == y["name"],
            "first_event_group_correct": efirst > 0,
            "second_event_group_correct": esecond < 0,
            "both_top_name_events_correct": (
                x["correct_event_by_top_name"]
                and y["correct_event_by_top_name"]
            ),
            "both_target_names_correct": (
                x["correct"] and y["correct"]
            ),
            "wrong_second_event": wrong_second_event,
            "direction_positive": shift > 0,
            "wrong_event_but_positive_direction": (
                wrong_second_event and shift > 0
            ),
            "first_role_margin_positive": (
                x["role_margin_gold_minus_opposite"] > 0
            ),
            "second_role_margin_positive": (
                y["role_margin_gold_minus_opposite"] > 0
            ),
            "shift": shift,
            "absolute_bias": abs((efirst + esecond) / 2),
        })

    def summarize(subset):
        assert subset
        wrong = sum(r["wrong_second_event"] for r in subset)
        shifts = [r["shift"] for r in subset]

        return {
            "n_paired_questions": len(subset),
            "first_name_accuracy": mean(
                r["first_name_correct"] for r in subset
            ),
            "second_name_accuracy": mean(
                r["second_name_correct"] for r in subset
            ),
            "unchanged_prediction_rate": mean(
                r["unchanged"] for r in subset
            ),
            "both_target_names_correct_rate": mean(
                r["both_target_names_correct"] for r in subset
            ),
            "both_top_name_events_correct_rate": mean(
                r["both_top_name_events_correct"] for r in subset
            ),
            "both_group_events_correct_rate": mean(
                r["first_event_group_correct"]
                and r["second_event_group_correct"]
                for r in subset
            ),
            "first_group_correct_rate": mean(
                r["first_event_group_correct"] for r in subset
            ),
            "second_group_correct_rate": mean(
                r["second_event_group_correct"] for r in subset
            ),
            "positive_cue_direction_rate": mean(
                r["direction_positive"] for r in subset
            ),
            "cue_shift_mean_logits": mean(shifts),
            "cue_shift_median_logits": median(shifts),
            "absolute_event_bias_median_logits": median(
                r["absolute_bias"] for r in subset
            ),
            "first_within_event_role_correct_rate": mean(
                r["first_role_margin_positive"] for r in subset
            ),
            "second_within_event_role_correct_rate": mean(
                r["second_role_margin_positive"] for r in subset
            ),
            "second_wrong_event_count": wrong,
            "second_wrong_event_rate": wrong / len(subset),
            "positive_direction_given_second_wrong_event": (
                sum(
                    r["wrong_event_but_positive_direction"]
                    for r in subset
                ) / wrong if wrong else None
            ),
        }

    overall = summarize(records)
    overall["by_role"] = {
        role: summarize([
            r for r in records if r["role"] == role
        ])
        for role in ("agent", "patient")
    }
    overall["by_frame"] = {
        str(frame): summarize([
            r for r in records if r["frame"] == frame
        ])
        for frame in (0, 1)
    }
    return overall


def parity_check(reference, seed, arm, pairs, first, second):
    if reference is None:
        return

    expected = next((
        r["metrics"]
        for r in reference["results"]
        if r["seed"] == seed and r["arm"] == arm
    ), None)
    assert expected is not None, (
        f"Missing ML-078 reference for {seed}/{arm}"
    )

    first_pred = [
        (x["name"], cf.prediction_kind(a, x["name"]))
        for (a, _), x in zip(pairs, first)
    ]
    second_pred = [
        (y["name"], cf.prediction_kind(b, y["name"]))
        for (_, b), y in zip(pairs, second)
    ]
    actual = cf.summarize(
        pairs, first_pred, second_pred
    )

    for key in (
        "first_correct_rate",
        "second_correct_rate",
        "unchanged_rate",
        "both_cues_correct_rate",
        "second_wrong_event_rate",
        "all_four_queries_correct_rate",
    ):
        assert abs(actual[key] - expected[key]) < 0.003, (
            f"ML-078 parity failure {seed}/{arm}/{key}: "
            f"{actual[key]:.5f} vs {expected[key]:.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=("plan", "preflight", "analyze"),
    )
    parser.add_argument(
        "--ml075-protocol",
        default="reports/ml075/frozen_protocol.json",
    )
    parser.add_argument(
        "--ml078-protocol",
        default="reports/ml078/frozen_protocol.json",
    )
    parser.add_argument(
        "--protocol",
        default="reports/ml079/frozen_protocol.json",
    )
    parser.add_argument(
        "--root", default="results/ml075_grammar",
    )
    parser.add_argument(
        "--reference",
        default="reports/ml078/event_cue_diagnostics.json",
    )
    parser.add_argument(
        "--output",
        default="reports/ml079/event_logit_diagnostics.json",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))

    frozen75, frozen78 = verify_inputs(
        Path(args.ml075_protocol),
        Path(args.ml078_protocol),
    )
    h = freeze(
        Path(args.protocol),
        frozen75, frozen78,
        args.phase == "plan",
    )

    if args.phase == "plan":
        print("Frozen ML-079:", args.protocol, h[:12])
        return

    pairs = cf.calibration_pairs(frozen75)
    cf.check_runs(Path(args.root), frozen75)

    print(
        f"ML-079 preflight: {len(pairs)} paired role "
        "questions, 18 checkpoints"
    )
    print(
        "Protected ML-075 cases NOT generated or evaluated.",
        flush=True,
    )

    if args.phase == "preflight":
        return

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    reference_path = Path(args.reference)
    reference = (
        json.loads(reference_path.read_text())
        if reference_path.is_file() else None
    )

    if reference is not None:
        assert reference["experiment"] == "ML-078"
        assert (
            reference["ml075_protocol_sha256"]
            == frozen75["sha256"]
        )
        assert (
            reference["protocol_sha256"]
            == frozen78["sha256"]
        )

    device = cf.choose_device(args.device)
    first_cases = [a for a, _ in pairs]
    second_cases = [b for _, b in pairs]

    report = {
        "experiment": "ML-079",
        "protocol_sha256": h,
        "ml075_protocol_sha256": frozen75["sha256"],
        "ml078_protocol_sha256": frozen78["sha256"],
        "ml078_reference_checked": reference is not None,
        "protected_evaluation_opened": False,
        "device": device,
        "torch_version": torch.__version__,
        "results": [],
    }

    for seed in SEEDS:
        for arm in ARMS:
            model = ml.load_model(
                seed, arm,
                Path(args.root),
                frozen75["sha256"],
                device,
            )
            first = score_questions(
                model, first_cases, device
            )
            second = score_questions(
                model, second_cases, device
            )

            parity_check(
                reference, seed, arm,
                pairs, first, second,
            )
            stats = describe(pairs, first, second)
            report["results"].append({
                "seed": seed,
                "arm": arm,
                "metrics": stats,
            })

            cond = stats[
                "positive_direction_given_second_wrong_event"
            ]
            cond_text = (
                "N/A" if cond is None else f"{cond:.3f}"
            )

            print(
                f"seed={seed} arm={arm:7s} "
                f"dir+={stats['positive_cue_direction_rate']:.3f} "
                f"both_group={stats['both_group_events_correct_rate']:.3f} "
                f"med_shift={stats['cue_shift_median_logits']:+.2f} "
                f"wrong2={stats['second_wrong_event_count']:3d} "
                f"dir+|wrong2={cond_text}",
                flush=True,
            )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.write_text(
        json.dumps(
            report, indent=2, sort_keys=True
        ) + "\n"
    )

    print("Saved:", output)
    print(
        "Protected ML-075 cases NOT generated or evaluated."
    )


if __name__ == "__main__":
    main()
