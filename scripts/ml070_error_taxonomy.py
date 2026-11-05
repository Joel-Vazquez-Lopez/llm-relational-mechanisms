
"""ML-070: Diagnose ML-069 compositional generalization failures.

Read-only audit of saved ML-069 checkpoints.
Separates correct role pairs, identical responses, and reversals.

POST-HOC analysis of already evaluated data.
Not a new confirmatory experiment.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import ml069_prospective as m


def classify(cases, predictions):
    events = defaultdict(dict)

    for c, prediction in zip(cases, predictions):
        key = (
            c.pair, c.verb, c.direction,
            c.voice, c.style,
        )

        assert c.role not in events[key]

        events[key][c.role] = (
            prediction,
            m.gold_name(c),
        )

    counts = {
        "correct_both": 0,
        "same_answer": 0,
        "reversed_both": 0,
    }

    for event in events.values():
        assert set(event) == {"agent", "patient"}

        agent_pred, agent_gold = event["agent"]
        patient_pred, patient_gold = event["patient"]

        assert agent_gold != patient_gold

        if (
            agent_pred == agent_gold
            and patient_pred == patient_gold
        ):
            counts["correct_both"] += 1

        elif agent_pred == patient_pred:
            counts["same_answer"] += 1

        else:
            assert (
                agent_pred == patient_gold
                and patient_pred == agent_gold
            )

            counts["reversed_both"] += 1

    n = len(events)
    assert sum(counts.values()) == n

    return {
        "events": n,
        **counts,
        "pair_accuracy": (
            2 * counts["correct_both"]
            + counts["same_answer"]
        ) / (2 * n),
    }


@torch.inference_mode()
def predict(model, cases, device):
    predictions = []

    decoder = {
        value: token
        for token, value in m.VOCAB.items()
    }

    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]

        ids, readout, _, choices = [
            x.to(device)
            for x in m.encode(batch)
        ]

        logits = model(ids, readout).float()

        candidate = logits.gather(
            1, choices
        )

        selection = choices.gather(
            1,
            candidate.argmax(1, keepdim=True),
        ).flatten()

        predictions.extend(
            decoder[int(token)]
            for token in selection.cpu()
        )

    return predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml069_prospective",
    )

    parser.add_argument(
        "--out",
        default="results/ml070_error_taxonomy.json",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()
    root = Path(args.root)

    saved = json.loads(
        (root / "protocol.json").read_text()
    )

    config = argparse.Namespace(
        seeds=list(saved["seeds"]),
        steps=saved["steps"],
        batch_size=saved["batch_size"],
        lr=saved["lr"],
    )

    _, digest, train_pairs, cal_pairs, test_pairs = (
        m.protocol(config)
    )

    assert digest == saved["sha256"], (
        "ML-069 protocol changed"
    )

    _, calibration, tests = m.datasets(
        train_pairs,
        cal_pairs,
        test_pairs,
    )

    reference = json.loads(
        (
            root / "protected_evaluation.json"
        ).read_text()
    )

    assert reference["protocol_sha256"] == digest

    original = {
        row["seed"]: row
        for row in reference["results"]
    }

    assert set(original) == set(config.seeds), (
        "Protected evaluation incomplete"
    )

    device = m.resolve_device(args.device)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    report = {
        "experiment": "ML-070",
        "status": (
            "Post-hoc diagnostic, "
            "not new confirmation"
        ),
        "protocol_sha256": digest,
        "runs": [],
    }

    for seed in config.seeds:
        model = m.load_model(
            root, seed, digest, device
        )

        cohorts = {
            "calibration": calibration,
            **tests,
        }

        results = {}

        for name, cases in cohorts.items():
            predictions = predict(
                model, cases, device
            )

            stats = classify(
                cases, predictions
            )

            if name != "calibration":
                expected = original[
                    seed
                ]["accuracy"][name]["pair_accuracy"]

                assert abs(
                    stats["pair_accuracy"]
                    - expected
                ) < 1e-7, (seed, name)

            results[name] = stats

        report["runs"].append({
            "seed": seed,
            "cohorts": results,
        })

        def show(name):
            x = results[name]

            return (
                f"{name}: "
                f"acc={x['pair_accuracy']:.3f}, "
                f"correct="
                f"{x['correct_both']}/{x['events']}, "
                f"same="
                f"{x['same_answer']}/{x['events']}, "
                f"reversed="
                f"{x['reversed_both']}/{x['events']}"
            )

        print(
            f"seed={seed} | "
            + show("known_pairs_new_composition")
            + " | "
            + show("double_ood"),
            flush=True,
        )

    output = Path(args.out)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}"
        )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
