
"""ML-071: Post-hoc logit audit of role-response collapse in ML-069.

Read-only: frozen checkpoints, same already-examined examples.
A positive role contrast means the question shifts candidate logits
in the correct semantic direction, even when argmax does not change.

This is diagnostic, not new independent confirmation.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import ml069_prospective as m


def aggregate(events):
    n = len(events)
    assert n > 0

    counts = {
        "correct_both": 0,
        "same_answer": 0,
        "reversed_both": 0,
    }

    deltas, biases, same_deltas = [], [], []
    same_agent, same_patient = 0, 0

    for e in events:
        agent, patient = e["agent"], e["patient"]
        da = agent["preference"]
        dp = patient["preference"]

        delta = da - dp
        bias = (da + dp) / 2

        deltas.append(delta)
        biases.append(bias)

        if (
            agent["pred"] == agent["gold"]
            and patient["pred"] == patient["gold"]
        ):
            counts["correct_both"] += 1

        elif agent["pred"] == patient["pred"]:
            counts["same_answer"] += 1
            same_deltas.append(delta)

            if agent["pred"] == e["agent_name"]:
                same_agent += 1
            else:
                same_patient += 1

        else:
            counts["reversed_both"] += 1

    assert sum(counts.values()) == n
    assert same_agent + same_patient == counts["same_answer"]

    return {
        "events": n,
        **counts,
        "same_choose_agent": same_agent,
        "same_choose_patient": same_patient,
        "pair_accuracy": (
            2 * counts["correct_both"] + counts["same_answer"]
        ) / (2 * n),
        "role_signal_positive_fraction": (
            sum(d > 0 for d in deltas) / n
        ),
        "role_signal_negative_fraction": (
            sum(d < 0 for d in deltas) / n
        ),
        "median_role_logit_delta": statistics.median(deltas),
        "median_absolute_role_logit_delta": statistics.median(
            map(abs, deltas)
        ),
        "median_absolute_answer_bias": statistics.median(
            map(abs, biases)
        ),
        "same_answer_role_signal_positive_fraction": (
            sum(d > 0 for d in same_deltas) / len(same_deltas)
            if same_deltas else None
        ),
    }


@torch.inference_mode()
def examine(model, examples, device):
    model.eval()
    event_records = defaultdict(dict)

    for start in range(0, len(examples), 128):
        batch = examples[start:start + 128]

        ids, pos, _, choices = [
            t.to(device) for t in m.encode(batch)
        ]

        logits = model(ids, pos).float().cpu()

        choice_scores = logits.gather(
            1, choices.cpu()
        )

        predicted = choices.cpu().gather(
            1,
            choice_scores.argmax(1, keepdim=True),
        ).flatten()

        for i, c in enumerate(batch):
            agent = (
                c.pair[0]
                if c.direction == 0
                else c.pair[1]
            )

            patient = (
                c.pair[1]
                if c.direction == 0
                else c.pair[0]
            )

            event = (
                c.pair, c.verb, c.direction,
                c.voice, c.style,
            )

            assert c.role not in event_records[event]

            event_records[event][c.role] = {
                "preference": float(
                    logits[i, m.VOCAB[agent]]
                    - logits[i, m.VOCAB[patient]]
                ),
                "pred": m.VOCAB_WORDS[
                    int(predicted[i])
                ],
                "gold": m.gold_name(c),
                "agent_name": agent,
            }

    records = []

    for key, roles in event_records.items():
        assert set(roles) == {"agent", "patient"}

        records.append({
            "pair": key[0],
            "voice": key[3],
            "style": key[4],
            "agent_name": roles["agent"]["agent_name"],
            **roles,
        })

    by_construction = defaultdict(list)
    by_pair = defaultdict(list)

    for e in records:
        by_construction[
            f'{e["voice"]}/{e["style"]}'
        ].append(e)

        by_pair[
            "|".join(e["pair"])
        ].append(e)

    return {
        "all": aggregate(records),
        "by_construction": {
            k: aggregate(v)
            for k, v in sorted(by_construction.items())
        },
        "by_pair": {
            k: aggregate(v)
            for k, v in sorted(by_pair.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml069_prospective",
    )

    parser.add_argument(
        "--out",
        default="results/ml071_role_signal_audit.json",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite {out}"
        )

    saved = json.loads(
        (root / "protocol.json").read_text()
    )

    config = argparse.Namespace(
        seeds=saved["seeds"],
        steps=saved["steps"],
        batch_size=saved["batch_size"],
        lr=saved["lr"],
    )

    _, digest, train_pairs, cal_pairs, test_pairs = (
        m.protocol(config)
    )

    assert digest == saved["sha256"]

    _, calibration, tests = m.datasets(
        train_pairs, cal_pairs, test_pairs
    )

    protected = json.loads(
        (root / "protected_evaluation.json").read_text()
    )

    assert protected["protocol_sha256"] == digest

    expected = {
        x["seed"]: x for x in protected["results"]
    }

    assert set(expected) == set(config.seeds)

    device = m.resolve_device(args.device)

    torch.set_num_threads(
        min(torch.get_num_threads(), 4)
    )

    report = {
        "experiment": "ML-071",
        "status": (
            "Post-hoc diagnostic; "
            "not independent validation"
        ),
        "protocol_sha256": digest,
        "runs": [],
    }

    for seed in config.seeds:
        model = m.load_model(
            root, seed, digest, device
        )

        datasets = {
            "calibration": calibration,
            **tests,
        }

        result = {
            name: examine(model, examples, device)
            for name, examples in datasets.items()
        }

        # Reproduce the original protected accuracies.
        for name in tests:
            got = result[name]["all"]["pair_accuracy"]
            want = expected[seed]["accuracy"][
                name
            ]["pair_accuracy"]

            assert abs(got - want) < 1e-7, (
                seed, name, got, want
            )

        report["runs"].append({
            "seed": seed,
            "cohorts": result,
        })

        good = result[
            "new_pairs_seen_grammar"
        ]["by_construction"]

        bad = result["double_ood"]["all"]

        print(
            f"seed={seed} | "
            f"seen active/alternate role+="
            f"{good['active/alternate']['role_signal_positive_fraction']:.3f} "
            f"passive/standard role+="
            f"{good['passive/standard']['role_signal_positive_fraction']:.3f} | "
            f"doubleOOD same="
            f"{bad['same_answer']}/{bad['events']} "
            f"reversed={bad['reversed_both']} "
            f"role+={bad['role_signal_positive_fraction']:.3f} "
            f"median_delta={bad['median_role_logit_delta']:+.3f} "
            f"median_abs_bias="
            f"{bad['median_absolute_answer_bias']:.3f}",
            flush=True,
        )

    out.parent.mkdir(
        parents=True, exist_ok=True
    )

    out.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("Saved:", out)
    print(
        "Post-hoc logit diagnostic only; "
        "no training or new evaluation data."
    )


if __name__ == "__main__":
    main()
