
"""ML-077: sealed-split calibration error diagnostics for frozen ML-075.

Uses ONLY ML-075's familiar-construction calibration examples and its
existing final checkpoints/training summaries. No model training, no
protected-example generation, and no protected evaluation.

From repository root:
    python3 scripts/ml077_checkpoint_diagnostics.py --phase plan
    python3 scripts/ml077_checkpoint_diagnostics.py --phase preflight
    python3 scripts/ml077_checkpoint_diagnostics.py --phase analyze
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch

import ml075_grammar_benchmark as ml

SEEDS = (211, 223, 227, 229, 233, 239)
ARMS = ("control", "sparse", "broad")
CATEGORIES = ("correct", "wrong_role_same_event", "distractor_event")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(path):
    frozen = json.loads(Path(path).read_text())
    assert frozen["experiment"] == "ML-075"
    assert file_hash(ml.__file__) == frozen["code_sha256"], "ML-075 source changed"
    assert digest({k: v for k, v in frozen.items() if k != "sha256"}) == frozen["sha256"], (
        "ML-075 frozen protocol hash changed"
    )
    assert tuple(frozen["seeds"]) == SEEDS
    assert tuple(frozen["arms"]) == ARMS
    return frozen


def calibration_only(frozen):
    # DO NOT call ml.build() or ml.target_test(): they generate protected cases.
    pairs = [tuple(pair) for pair in frozen["calibration_pair_ids"]]
    cases = ml.balanced_base(pairs)
    assert len(cases) == frozen["n_examples"]["calibration"] == 6912
    assert len({tuple(ml.prompt(c)) for c in cases}) == len(cases)
    assert all((c.target_pos, c.target_voice) in ml.BASE_CELLS for c in cases)
    assert all(replace(a, role="patient") == b for a, b in zip(cases[::2], cases[1::2]))
    return cases


def training_name_counts(frozen):
    pairs = [tuple(pair) for pair in frozen["training_pair_ids"]]
    base = ml.balanced_base(pairs)
    assert len(base) == frozen["n_examples"]["base"]
    # Aggregate frequency across agent and patient questions.
    return Counter(ml.gold(c) for c in base)


def name_baselines(cases, name_counts):
    """Non-relational and event-aware name-prior baselines.

    Both pick the same name for agent/patient, so their correct-both rate is zero.
    These are diagnostic baselines, not independent trained models.
    """
    out = {}
    for scope in ("four_names", "target_pair_only"):
        hits = 0
        for c in cases:
            candidates = (c.target_pair + c.distractor_pair) if scope == "four_names" else c.target_pair
            best = max(candidates, key=lambda name: (name_counts[name], -ml.NAMES.index(name)))
            hits += best == ml.gold(c)
        out[scope] = {"accuracy": hits / len(cases), "correct_both": 0.0}
    return out


def classify(c, pred):
    if pred == ml.gold(c):
        return "correct"
    if pred in c.target_pair:
        return "wrong_role_same_event"
    assert pred in c.distractor_pair, f"Non-candidate answer: {pred}"
    return "distractor_event"


@torch.inference_mode()
def predict(model, cases, device):
    model.eval()
    rows = []
    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]
        ids, readout, labels, choices = ml.encode(batch)
        logits = model(ids.to(device), readout.to(device)).float()
        choices_device = choices.to(device)
        candidate_logits = logits.gather(1, choices_device)
        pred_ids = choices_device.gather(1, candidate_logits.argmax(1, keepdim=True)).squeeze(1)
        full_ids = logits.argmax(1)
        for case, pred_id, full_id, true_id in zip(
            batch, pred_ids.cpu().tolist(), full_ids.cpu().tolist(), labels.tolist()
        ):
            pred = ml.VOCAB_WORDS[pred_id]
            cat = classify(case, pred)
            rows.append({
                "case": case,
                "gold": ml.gold(case),
                "prediction": pred,
                "category": cat,
                "full_correct": full_id == true_id,
            })
    assert len(rows) == len(cases)
    return rows


def question_summary(rows):
    n = len(rows)
    counts = Counter(r["category"] for r in rows)
    errors = n - counts["correct"]
    return {
        "n_questions": n,
        "accuracy": counts["correct"] / n,
        "wrong_role_rate": counts["wrong_role_same_event"] / n,
        "wrong_event_rate": counts["distractor_event"] / n,
        "error_composition": {
            "wrong_role_given_error": counts["wrong_role_same_event"] / errors if errors else None,
            "wrong_event_given_error": counts["distractor_event"] / errors if errors else None,
        },
    }


def event_summary(rows):
    assert len(rows) % 2 == 0
    counts = Counter()
    for a, b in zip(rows[::2], rows[1::2]):
        assert a["case"].role == "agent" and b["case"].role == "patient"
        assert replace(a["case"], role="patient") == b["case"]
        counts["both_correct"] += a["category"] == b["category"] == "correct"
        counts["same_answer"] += a["prediction"] == b["prediction"]
        counts["roles_swapped"] += (
            a["category"] == b["category"] == "wrong_role_same_event"
        )
        counts["any_wrong_event"] += (
            a["category"] == "distractor_event" or b["category"] == "distractor_event"
        )
        counts["both_wrong_event"] += (
            a["category"] == b["category"] == "distractor_event"
        )
    events = len(rows) // 2
    return {name + "_rate": counts[name] / events for name in (
        "both_correct", "same_answer", "roles_swapped", "any_wrong_event", "both_wrong_event"
    )} | {"n_events": events}


def aggregate(rows, key_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row["case"]))].append(row)
    return {key: question_summary(group) for key, group in sorted(groups.items())}


def examples(rows):
    seen = set()
    out = []
    for row in rows:
        cat = row["category"]
        if cat == "correct" or cat in seen:
            continue
        seen.add(cat)
        out.append({
            "error_category": cat,
            "prompt": " ".join(ml.prompt(row["case"])),
            "gold": row["gold"],
            "predicted": row["prediction"],
        })
    return out


def summarise_run(rows):
    # Note: these are descriptive associations, not identified causal mechanisms.
    pair_groups = aggregate(rows, lambda c: "-".join(c.target_pair))
    worst_pairs = sorted(
        pair_groups.items(),
        key=lambda item: (item[1]["accuracy"], item[0]),
    )[:5]
    q = question_summary(rows)
    return {
        **q,
        "full_vocab_accuracy": sum(r["full_correct"] for r in rows) / len(rows),
        **event_summary(rows),
        "by_grammar": aggregate(rows, lambda c: f"{c.target_pos}:{c.target_voice}"),
        "by_role": aggregate(rows, lambda c: c.role),
        "by_frame": aggregate(rows, lambda c: c.frame),
        "by_distractor_voice": aggregate(rows, lambda c: c.distractor_voice),
        "by_grammar_and_distractor_voice": aggregate(
            rows, lambda c: f"{c.target_pos}:{c.target_voice}/{c.distractor_voice}"
        ),
        "by_verb": aggregate(rows, lambda c: c.target_verb),
        "by_gold_name": aggregate(rows, lambda c: ml.gold(c)),
        "worst_five_target_pairs": [{"pair": pair, **stats} for pair, stats in worst_pairs],
        "illustrative_errors": examples(rows),
    }


def plan_doc(frozen):
    return {
        "experiment": "ML-077",
        "script_sha256": file_hash(__file__),
        "ml075_protocol_sha256": frozen["sha256"],
        "input": "ML-075 calibration examples, final.pt checkpoints, training.json; ML-075 base labels only for name-prior baseline",
        "excluded": "Protected examples, heldout-target grammar, new training, checkpoint modification",
        "seeds": SEEDS,
        "arms": ARMS,
        "error_categories": CATEGORIES,
        "metrics": [
            "four-candidate accuracy", "full-vocabulary accuracy", "correct-both",
            "wrong-role-same-event", "distractor-event selection", "same-answer",
            "role-swapped", "grammar/role/frame/verb/name breakdowns",
            "training-name-prior baselines",
        ],
        "interpretation_note": "Categories describe predicted-name membership, not internal causal operations.",
    }


def freeze(path, frozen, create):
    doc = plan_doc(frozen)
    canonical = json.loads(json.dumps({**doc, "sha256": digest(doc)}))
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(canonical, indent=2) + "\n")
    if not path.exists():
        raise FileNotFoundError(f"Missing ML-077 protocol: {path}; run --phase plan")
    assert json.loads(path.read_text()) == canonical, "ML-077 protocol/source mismatch"
    return canonical["sha256"]


def validate_checkpoints(root, frozen):
    report = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = root / f"seed_{seed}" / arm
            ckpt = folder / "final.pt"
            summary_path = folder / "training.json"
            assert ckpt.is_file() and summary_path.is_file(), f"Missing run: {folder}"
            summary = json.loads(summary_path.read_text())
            assert summary["seed"] == seed and summary["arm"] == arm
            assert summary["protocol_sha256"] == frozen["sha256"]
            report.append((seed, arm, ckpt, summary))
    return report


def choose_device(name):
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return name
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("plan", "preflight", "analyze"), required=True)
    parser.add_argument("--ml075-protocol", default="reports/ml075/frozen_protocol.json")
    parser.add_argument("--protocol", default="reports/ml077/frozen_protocol.json")
    parser.add_argument("--root", default="results/ml075_grammar")
    parser.add_argument("--output", default="reports/ml077/calibration_diagnostics.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    frozen = load_frozen(args.ml075_protocol)
    audit_hash = freeze(Path(args.protocol), frozen, create=(args.phase == "plan"))
    if args.phase == "plan":
        print("ML-077 frozen:", args.protocol, audit_hash[:12]); return

    cases = calibration_only(frozen)
    runs = validate_checkpoints(Path(args.root), frozen)
    print(f"ML-077 preflight: {len(cases)} familiar-calibration questions, {len(runs)} checkpoints")
    print("Protected cases NOT generated or evaluated.", flush=True)
    if args.phase == "preflight":
        return

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    device = choose_device(args.device)
    baseline = name_baselines(cases, training_name_counts(frozen))
    report = {
        "experiment": "ML-077",
        "audit_protocol_sha256": audit_hash,
        "ml075_protocol_sha256": frozen["sha256"],
        "device": device,
        "torch_version": torch.__version__,
        "protected_evaluation_opened": False,
        "calibration_questions": len(cases),
        "name_frequency_baselines": baseline,
        "runs": [],
    }
    for seed, arm, ckpt, previous in runs:
        model = ml.load_model(seed, arm, Path(args.root), frozen["sha256"], device)
        rows = predict(model, cases, device)
        stats = summarise_run(rows)
        # Compare against the original saved checkpoint evaluation as a consistency guard.
        expected = previous["calibration"]
        checks = {
            "pair_accuracy": (stats["accuracy"], expected["pair_accuracy"]),
            "correct_both_rate": (stats["both_correct_rate"], expected["correct_both_rate"]),
            "same_answer_rate": (stats["same_answer_rate"], expected["same_answer_rate"]),
            "full_vocab_accuracy": (stats["full_vocab_accuracy"], expected["full_vocab_accuracy"]),
        }
        for name, (actual, saved) in checks.items():
            assert abs(actual - saved) < 0.002, (
                f"Checkpoint metric mismatch {seed}/{arm}/{name}: {actual:.6f} vs {saved:.6f}"
            )
        report["runs"].append({
            "seed": seed, "arm": arm, "checkpoint_sha256": file_hash(ckpt),
            "saved_training_accuracy": previous["train"]["pair_accuracy"],
            "saved_last_loss": previous["last_loss"],
            "calibration": stats,
        })
        wrong = stats["wrong_role_rate"]
        event = stats["wrong_event_rate"]
        print(
            f"seed={seed} arm={arm:7s} cal={stats['accuracy']:.3f} "
            f"both={stats['both_correct_rate']:.3f} "
            f"wrong_role={wrong:.3f} wrong_event={event:.3f} "
            f"same={stats['same_answer_rate']:.3f}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Name-prior baselines:", baseline)
    print("Saved:", output)
    print("Protected cases NOT generated or evaluated.")


if __name__ == "__main__":
    main()
