
"""ML-078: paired event-cue counterfactuals on familiar ML-075 grammar.

Read-only: no training, no protected target evaluation. Use calibration
sentences where BOTH events are active. The ONLY changed input token is
first <-> second.

Run from repository root:
  python3 scripts/ml078_event_cue_counterfactual.py --phase plan
  python3 scripts/ml078_event_cue_counterfactual.py --phase preflight
  python3 scripts/ml078_event_cue_counterfactual.py --phase analyze
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import ml075_grammar_benchmark as ml

SEEDS = (211, 223, 227, 229, 233, 239)
ARMS = ("control", "sparse", "broad")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def frozen_ml075(path):
    document = json.loads(path.read_text())

    assert document["experiment"] == "ML-075"
    assert file_sha(ml.__file__) == document["code_sha256"], (
        "ML-075 source changed"
    )
    assert digest({
        k: v for k, v in document.items()
        if k != "sha256"
    }) == document["sha256"], "ML-075 protocol altered"

    assert tuple(document["seeds"]) == SEEDS
    assert tuple(document["arms"]) == ARMS
    return document


def counterfactual_first(second):
    """Swap the query target, without changing the sentence."""
    assert (
        second.target_pos,
        second.target_voice,
        second.distractor_voice
    ) == (1, "active", "active")

    first = replace(
        second,
        target_pair=second.distractor_pair,
        distractor_pair=second.target_pair,
        target_verb=second.distractor_verb,
        distractor_verb=second.target_verb,
        target_pos=0,
        target_voice="active",
        distractor_voice="active",
        target_dir=second.distractor_dir,
        distractor_dir=second.target_dir,
    )

    ft = ml.prompt(first)
    st = ml.prompt(second)

    differing = [
        i for i, (a, b) in enumerate(zip(ft, st))
        if a != b
    ]

    assert len(ft) == len(st)
    assert len(differing) == 1, "More than one token changed"

    pos = differing[0]
    assert (ft[pos], st[pos]) == ("first", "second")
    assert pos > ft.index("<sep>")
    assert ft[:pos] == st[:pos]
    assert ft[pos + 1:] == st[pos + 1:]

    assert set(first.target_pair).isdisjoint(second.target_pair)
    assert (first.target_pos, first.target_voice) in ml.BASE_CELLS
    assert (second.target_pos, second.target_voice) in ml.BASE_CELLS

    return first


def calibration_pairs(frozen):
    """Never generate protected examples."""
    cases = ml.balanced_base([
        tuple(p)
        for p in frozen["calibration_pair_ids"]
    ])

    assert len(cases) == frozen["n_examples"]["calibration"]

    pairs = []

    for second in cases:
        if (
            second.target_pos,
            second.target_voice,
            second.distractor_voice
        ) != (1, "active", "active"):
            continue

        first = counterfactual_first(second)
        pairs.append((first, second))

    assert len(pairs) == 1152
    assert all(a.role == b.role for a, b in pairs)
    assert sum(a.role == "agent" for a, _ in pairs) == 576
    assert sum(a.role == "patient" for a, _ in pairs) == 576

    return pairs


def protocol_document(frozen):
    return {
        "experiment": "ML-078",
        "code_sha256": file_sha(__file__),
        "ml075_protocol_sha256": frozen["sha256"],
        "input": (
            "ML-075 calibration pairs only, "
            "using sentences with two active events"
        ),
        "intervention": (
            "Replace exactly one query token: "
            "first <-> second; identical sentence"
        ),
        "paired_role_questions": 1152,
        "models": {
            "seeds": SEEDS,
            "arms": ARMS,
        },
        "metrics": [
            "first/second accuracy",
            "first/second wrong-event selection",
            "first/second wrong-role selection",
            "prediction unchanged across cue",
            "both cues correct for the same role",
            "all four queries correct per sentence",
            "unchanged prediction given second-cue wrong-event output",
        ],
        "analysis": (
            "Within-checkpoint descriptive paired comparisons, "
            "not a new training experiment"
        ),
        "exclusions": (
            "No protected participant-pair split; "
            "no passive-second target queries; no new training"
        ),
    }


def check_protocol(path, frozen, create):
    doc = protocol_document(frozen)
    expected = json.loads(json.dumps({
        **doc,
        "sha256": digest(doc),
    }))

    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(expected, indent=2) + "\n"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen ML-078 protocol: {path}"
        )

    assert json.loads(path.read_text()) == expected, (
        "ML-078 protocol or source changed"
    )

    return expected["sha256"]


def check_runs(root, frozen):
    for seed in SEEDS:
        for arm in ARMS:
            folder = root / f"seed_{seed}" / arm

            assert (folder / "final.pt").is_file(), (
                f"Missing checkpoint: {folder}"
            )

            summary = json.loads(
                (folder / "training.json").read_text()
            )

            assert (summary["seed"], summary["arm"]) == (seed, arm)
            assert summary["protocol_sha256"] == frozen["sha256"]


def prediction_kind(case, prediction):
    if prediction == ml.gold(case):
        return "correct"

    if prediction in case.target_pair:
        return "wrong_role"

    assert prediction in case.distractor_pair, (
        "Model predicted a name outside the four candidates"
    )
    return "wrong_event"


@torch.inference_mode()
def infer(model, cases, device):
    """Use ML-075's original four-candidate scoring rule."""
    output = []
    model.eval()

    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]

        ids, readout, _, choices = ml.encode(batch)

        logits = model(
            ids.to(device),
            readout.to(device),
        ).float()

        candidates = choices.to(device)

        best = candidates.gather(
            1,
            logits.gather(1, candidates).argmax(
                1, keepdim=True
            ),
        )

        for case, pred_id in zip(
            batch,
            best.flatten().cpu().tolist(),
        ):
            pred_name = ml.VOCAB_WORDS[pred_id]
            output.append((
                pred_name,
                prediction_kind(case, pred_name),
            ))

    return output


def average(group, key):
    return sum(item[key] for item in group) / len(group)


def summarize(pairs, first_pred, second_pred):
    assert len(pairs) == len(first_pred) == len(second_pred)

    records = []

    for (first, second), (pa, ka), (pb, kb) in zip(
        pairs, first_pred, second_pred
    ):
        records.append({
            "frame": second.frame,
            "role": second.role,
            "second_verb": second.target_verb,
            "first_verb": first.target_verb,

            "first_correct": int(ka == "correct"),
            "second_correct": int(kb == "correct"),

            "first_wrong_event": int(ka == "wrong_event"),
            "second_wrong_event": int(kb == "wrong_event"),

            "first_wrong_role": int(ka == "wrong_role"),
            "second_wrong_role": int(kb == "wrong_role"),

            "unchanged": int(pa == pb),
            "both_cues_correct": int(
                ka == kb == "correct"
            ),
            "second_wrong_event_and_unchanged": int(
                kb == "wrong_event" and pa == pb
            ),
        })

    numeric = (
        "first_correct",
        "second_correct",
        "first_wrong_event",
        "second_wrong_event",
        "first_wrong_role",
        "second_wrong_role",
        "unchanged",
        "both_cues_correct",
        "second_wrong_event_and_unchanged",
    )

    def stats(group):
        result = {
            key + "_rate": average(group, key)
            for key in numeric
        }

        count_wrong = sum(
            item["second_wrong_event"]
            for item in group
        )

        result["unchanged_given_second_wrong_event"] = (
            sum(
                item["second_wrong_event_and_unchanged"]
                for item in group
            ) / count_wrong
            if count_wrong else None
        )

        result["n_role_questions"] = len(group)
        return result

    assert all(
        a.role == "agent"
        and b.role == "patient"
        and replace(a, role="patient") == b
        for (_, a), (_, b) in zip(
            pairs[::2],
            pairs[1::2],
        )
    )

    all_four = sum(
        all(
            r["both_cues_correct"]
            for r in records[i:i + 2]
        )
        for i in range(0, len(records), 2)
    ) / (len(records) // 2)

    output = stats(records)
    output["all_four_queries_correct_rate"] = all_four
    output["n_sentences"] = len(records) // 2

    output["by_role"] = {
        role: stats([
            r for r in records
            if r["role"] == role
        ])
        for role in ("agent", "patient")
    }

    output["by_frame"] = {
        str(frame): stats([
            r for r in records
            if r["frame"] == frame
        ])
        for frame in (0, 1)
    }

    output["by_second_verb"] = {
        verb: stats([
            r for r in records
            if r["second_verb"] == verb
        ])
        for verb in ml.VERBS
    }

    return output


def choose_device(arg):
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if arg == "mps":
        assert torch.backends.mps.is_available(), (
            "MPS not available"
        )

    if arg == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA not available"
        )

    return arg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("plan", "preflight", "analyze"),
        required=True,
    )
    parser.add_argument(
        "--ml075-protocol",
        default="reports/ml075/frozen_protocol.json",
    )
    parser.add_argument(
        "--protocol",
        default="reports/ml078/frozen_protocol.json",
    )
    parser.add_argument(
        "--root",
        default="results/ml075_grammar",
    )
    parser.add_argument(
        "--output",
        default="reports/ml078/event_cue_diagnostics.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    frozen = frozen_ml075(
        Path(args.ml075_protocol)
    )

    protocol_hash = check_protocol(
        Path(args.protocol),
        frozen,
        create=args.phase == "plan",
    )

    if args.phase == "plan":
        print(
            "Frozen ML-078:",
            args.protocol,
            protocol_hash[:12],
        )
        return

    pairs = calibration_pairs(frozen)
    check_runs(Path(args.root), frozen)

    print(
        f"ML-078 preflight: {len(pairs)} paired role questions, "
        f"{len(pairs)//2} sentences, 18 checkpoints"
    )
    print(
        "Only first/second cue differs. "
        "Both events active; protected results sealed."
    )

    if args.phase == "preflight":
        return

    output_path = Path(args.output)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    device = choose_device(args.device)

    first_cases = [a for a, _ in pairs]
    second_cases = [b for _, b in pairs]

    report = {
        "experiment": "ML-078",
        "protocol_sha256": protocol_hash,
        "ml075_protocol_sha256": frozen["sha256"],
        "protected_evaluation_opened": False,
        "device": device,
        "torch_version": torch.__version__,
        "results": [],
    }

    for seed in SEEDS:
        for arm in ARMS:
            model = ml.load_model(
                seed,
                arm,
                Path(args.root),
                frozen["sha256"],
                device,
            )

            first_pred = infer(
                model, first_cases, device
            )
            second_pred = infer(
                model, second_cases, device
            )

            result = summarize(
                pairs,
                first_pred,
                second_pred,
            )

            report["results"].append({
                "seed": seed,
                "arm": arm,
                "metrics": result,
            })

            print(
                f"seed={seed} arm={arm:7s} "
                f"first={result['first_correct_rate']:.3f} "
                f"second={result['second_correct_rate']:.3f} "
                f"both_cues={result['both_cues_correct_rate']:.3f} "
                f"unchanged={result['unchanged_rate']:.3f} "
                f"second_wrong_event="
                f"{result['second_wrong_event_rate']:.3f}",
                flush=True,
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    print("Saved:", output_path)
    print(
        "Protected cases NOT generated or evaluated."
    )


if __name__ == "__main__":
    main()
