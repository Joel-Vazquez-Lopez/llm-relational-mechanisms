
"""ML-073: Held-out role-signal and layer-2 causal-factorial audit.

Plan -> validate known constructions -> evaluate a previously unqueried
passive/alternate slice of the ML-072 calibration participant pairs.

No training, no changes to ML-072 code or checkpoints.
This is not an independent model-family or grammar replication.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

import ml072_compositional_exposure as ml


SEEDS = tuple(ml.SEEDS)
ARMS = tuple(ml.ARMS)

GROUPS = {
    "active_alternate_control": (
        ("active", "alternate"),
    ),
    "passive_standard_control": (
        ("passive", "standard"),
    ),
    "target_passive_alternate": (
        ("passive", "alternate"),
    ),
}


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def json_hash(data):
    return hashlib.sha256(
        json.dumps(
            data,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def load_protocol(args):
    root = Path(args.ml072_root)

    frozen = json.loads(
        Path(args.ml072_protocol).read_text()
    )

    assert frozen["experiment"] == "ML-072"

    assert (
        frozen["code_sha256"]
        == file_hash(ml.__file__)
    ), (
        "ML-072 source changed relative "
        "to its frozen protocol"
    )

    digest = frozen["sha256"]

    assert digest == ml.digest({
        k: v
        for k, v in frozen.items()
        if k != "sha256"
    })

    assert frozen["seeds"] == list(SEEDS)
    assert frozen["arms"] == list(ARMS)

    for seed in SEEDS:
        for arm in ARMS:
            path = (
                root
                / f"seed_{seed}"
                / arm
                / "final.pt"
            )

            if not path.exists():
                raise FileNotFoundError(path)

    return frozen


def document(old):
    return {
        "experiment": "ML-073",
        "status": (
            "Predeclared analysis of unused "
            "combinations in established "
            "synthetic ML-072 benchmark"
        ),
        "ml073_code_sha256": file_hash(
            __file__
        ),
        "ml072_protocol_sha256": old[
            "sha256"
        ],
        "ml072_code_sha256": old[
            "code_sha256"
        ],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "groups": GROUPS,
        "previously_unused_target_pairs": old[
            "calibration_pairs"
        ],
        "verbs": list(ml.VERBS),
        "primary_outcome": (
            "Both-role accuracy on calibration-pair "
            "passive/alternate questions"
        ),
        "primary_comparisons": [
            "sparse-control",
            "broad-control",
            "broad-sparse",
        ],
        "predeclared_expectation": (
            "Sparse and broad will outperform "
            "control, based on previously opened "
            "ML-072 results. Not an independent "
            "replication."
        ),
        "mechanistic_measures": [
            "Role-conditioned candidate margin delta",
            "Question-averaged participant bias",
            "Layer-2 attention-output factorial contribution",
            "Layer-2 input-state factorial contribution",
            "Opposite-role attention-only and input-only swaps",
            "Same-role different-verb negative-control swaps",
        ],
        "mechanistic_hypothesis_status": (
            "Exploratory localization; "
            "no predeclared attention/input "
            "dominance direction"
        ),
        "independent_unit": (
            "Random initialization seed; "
            "examples within a model are dependent"
        ),
        "limitations": [
            "Synthetic grammar with familiar vocabulary",
            "Calibration pairs previously evaluated "
            "under other grammatical constructions",
            "Hypotheses motivated by ML-072 results",
            "Parallel final checkpoints do not establish "
            "developmental before/after mechanisms",
            "Factorial hybrids may be off-manifold",
        ],
    }


def assert_frozen(path, data):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Run --phase plan first: {path}"
        )

    saved = json.loads(
        path.read_text()
    )

    assert (
        saved.get("sha256")
        == json_hash(data)
    ), "Frozen protocol hash mismatch"

    expected = json.loads(
        json.dumps(data)
    )

    assert {
        k: v
        for k, v in saved.items()
        if k != "sha256"
    } == expected, (
        "ML-073 source or protocol changed "
        "after freezing"
    )


def model_from_checkpoint(
    root, old, seed, arm, device
):
    path = (
        Path(root)
        / f"seed_{seed}"
        / arm
        / "final.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seed"] == seed
    assert checkpoint["arm"] == arm

    assert (
        checkpoint["protocol_sha256"]
        == old["sha256"]
    )

    model = ml.Transformer().to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    return model


def readout_logits(model, state):
    block = model.blocks[1]

    updated = (
        state
        + block.mlp(
            block.norm2(state)
        )
    )

    return model.output(
        model.norm(updated)
    ).float()


@torch.inference_mode()
def internal_states(model, cases, device):
    ids, readout, _, _ = (
        tensor.to(device)
        for tensor in ml.encode(cases)
    )

    _, length = ids.shape

    positions = torch.arange(
        length,
        device=device,
    )

    causal = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=device,
    ).triu(1)

    x = (
        model.embed(ids)
        + model.pos(positions)
    )

    # Output of the first Transformer block.
    x = model.blocks[0](
        x, causal
    )

    # Attention update generated by block 2.
    z = model.blocks[1].norm1(x)

    a = model.blocks[1].attn(
        z, z, z,
        attn_mask=causal,
        need_weights=False,
    )[0]

    index = torch.arange(
        len(cases),
        device=device,
    )

    return (
        x[index, readout],
        a[index, readout],
        ids,
        readout,
    )


@torch.inference_mode()
def batch_observations(
    model, cases, device, check=False
):
    assert len(cases) % 2 == 0

    assert all(
        cases[i].role == "agent"
        and cases[i + 1].role == "patient"
        and ml.event_key(cases[i])
        == ml.event_key(cases[i + 1])
        for i in range(0, len(cases), 2)
    )

    x, a, ids, pos = internal_states(
        model, cases, device
    )

    x_ag, x_pa = x[0::2], x[1::2]
    a_ag, a_pa = a[0::2], a[1::2]

    # 2x2 factorial intervention:
    # first variable: layer-2 input
    # second variable: layer-2 attention update

    logits_00 = readout_logits(
        model, x_ag + a_ag
    )

    logits_01 = readout_logits(
        model, x_ag + a_pa
    )

    logits_10 = readout_logits(
        model, x_pa + a_ag
    )

    logits_11 = readout_logits(
        model, x_pa + a_pa
    )

    max_difference = None

    if check:
        natural = model(
            ids, pos
        ).float()

        reconstructed = torch.empty_like(
            natural
        )

        reconstructed[0::2] = logits_00
        reconstructed[1::2] = logits_11

        max_difference = float(
            (
                natural - reconstructed
            ).abs().max().item()
        )

        if max_difference > 0.001:
            raise AssertionError(
                "Manual forward reconstruction "
                f"failed: {max_difference}"
            )

    # Negative control:
    # same role and participants, different verb.
    control_cases = [
        ml.Case(
            c.pair,
            ml.VERBS[
                (
                    ml.VERBS.index(c.verb) + 1
                ) % len(ml.VERBS)
            ],
            c.direction,
            c.voice,
            c.style,
            c.role,
        )
        for c in cases
    ]

    _, control_a, _, _ = internal_states(
        model, control_cases, device
    )

    control_logits = readout_logits(
        model,
        x + control_a,
    )

    agents = [
        c.pair[0]
        if c.direction == 0
        else c.pair[1]
        for c in cases[0::2]
    ]

    patients = [
        c.pair[1]
        if c.direction == 0
        else c.pair[0]
        for c in cases[0::2]
    ]

    agent_ids = torch.tensor(
        [ml.VOCAB[t] for t in agents],
        device=device,
    )

    patient_ids = torch.tensor(
        [ml.VOCAB[t] for t in patients],
        device=device,
    )

    rows = torch.arange(
        len(agent_ids),
        device=device,
    )

    def margin(logits):
        return (
            logits[rows, agent_ids]
            - logits[rows, patient_ids]
        )

    m00, m01, m10, m11 = map(
        margin,
        (
            logits_00,
            logits_01,
            logits_10,
            logits_11,
        ),
    )

    verb_ag = margin(
        control_logits[0::2]
    )

    verb_pa = margin(
        control_logits[1::2]
    )

    # Exact factorial decomposition.
    attention = 0.5 * (
        (m00 - m01)
        + (m10 - m11)
    )

    input_state = 0.5 * (
        (m00 - m10)
        + (m01 - m11)
    )

    delta = m00 - m11
    bias = (m00 + m11) / 2

    error = float(
        (
            attention
            + input_state
            - delta
        ).abs().max().item()
    )

    if error > 0.001:
        raise AssertionError(
            "Factorial identity failed: "
            f"{error}"
        )

    results = []

    for i, case in enumerate(cases[0::2]):
        ma = float(m00[i])
        mp = float(m11[i])

        correct = (
            ma > 0
            and mp < 0
        )

        same = (
            (ma > 0 and mp > 0)
            or (ma < 0 and mp < 0)
        )

        reversed_both = (
            ma < 0
            and mp > 0
        )

        if not (
            correct
            or same
            or reversed_both
        ):
            raise AssertionError(
                "Unclassified event or tied logits"
            )

        # Both questions must flip correctly.
        att_flip = (
            float(m01[i]) < 0
            and float(m10[i]) > 0
        )

        input_flip = (
            float(m10[i]) < 0
            and float(m01[i]) > 0
        )

        control_changed_ag = (
            (ma > 0)
            != (float(verb_ag[i]) > 0)
        )

        control_changed_pa = (
            (mp > 0)
            != (float(verb_pa[i]) > 0)
        )

        results.append({
            "pair": list(case.pair),
            "verb": case.verb,
            "direction": case.direction,
            "correct_both": correct,
            "same_answer": same,
            "reversed_both": reversed_both,
            "role_delta": float(delta[i]),
            "abs_answer_bias": abs(
                float(bias[i])
            ),
            "attention_contribution": float(
                attention[i]
            ),
            "input_contribution": float(
                input_state[i]
            ),
            "attention_swap_flips_both": (
                correct and att_flip
            ),
            "input_swap_flips_both": (
                correct and input_flip
            ),
            "negative_control_flip_count": (
                int(control_changed_ag)
                + int(control_changed_pa)
            ),
        })

    return results, max_difference, error


def summarize(records):
    n = len(records)

    eligible = sum(
        r["correct_both"]
        for r in records
    )

    return {
        "events": n,
        "correct_both_rate": (
            eligible / n
        ),
        "same_answer_rate": (
            sum(
                r["same_answer"]
                for r in records
            ) / n
        ),
        "reversed_both_rate": (
            sum(
                r["reversed_both"]
                for r in records
            ) / n
        ),
        "role_signal_positive_fraction": (
            sum(
                r["role_delta"] > 0
                for r in records
            ) / n
        ),
        "median_role_delta": (
            statistics.median(
                r["role_delta"]
                for r in records
            )
        ),
        "median_abs_answer_bias": (
            statistics.median(
                r["abs_answer_bias"]
                for r in records
            )
        ),
        "mean_attention_contribution": (
            statistics.mean(
                r["attention_contribution"]
                for r in records
            )
        ),
        "mean_input_contribution": (
            statistics.mean(
                r["input_contribution"]
                for r in records
            )
        ),
        "attention_swap_flips_both_rate_if_correct": (
            sum(
                r["attention_swap_flips_both"]
                for r in records
            ) / eligible
            if eligible else None
        ),
        "input_swap_flips_both_rate_if_correct": (
            sum(
                r["input_swap_flips_both"]
                for r in records
            ) / eligible
            if eligible else None
        ),
        "verb_control_question_flip_rate": (
            sum(
                r["negative_control_flip_count"]
                for r in records
            ) / (2 * n)
        ),
    }


def make_cohorts(old):
    pairs = [
        tuple(pair)
        for pair in old["calibration_pairs"]
    ]

    train_pairs = {
        tuple(pair)
        for pair in old["train_pairs"]
    }

    test_pairs = {
        tuple(pair)
        for pair in old["test_pairs"]
    }

    assert not (
        set(pairs)
        & (train_pairs | test_pairs)
    )

    return {
        name: ml.make_cases(
            pairs,
            ml.VERBS,
            pattern,
        )
        for name, pattern in GROUPS.items()
    }


def run_model(
    model, cases, device, check=False
):
    records = []
    differences = []
    errors = []

    for offset in range(
        0, len(cases), 64
    ):
        rows, difference, error = (
            batch_observations(
                model,
                cases[offset:offset + 64],
                device,
                check=check,
            )
        )

        records.extend(rows)
        errors.append(error)

        if difference is not None:
            differences.append(difference)

    return (
        records,
        max(differences, default=None),
        max(errors),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "plan",
            "validate",
            "evaluate",
        ),
        required=True,
    )

    parser.add_argument(
        "--ml072-root",
        default="results/ml072_exposure",
    )

    parser.add_argument(
        "--ml072-protocol",
        default="reports/ml072/frozen_protocol.json",
    )

    parser.add_argument(
        "--protocol",
        default="reports/ml073/frozen_protocol.json",
    )

    parser.add_argument(
        "--root",
        default="results/ml073_causal_balance",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    old = load_protocol(args)
    data = document(old)

    output_root = Path(args.root)
    protocol_path = Path(args.protocol)

    if args.phase == "plan":
        if protocol_path.exists():
            assert_frozen(
                protocol_path, data
            )

            print(
                "Protocol already frozen:",
                protocol_path,
            )
            return

        protocol_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        protocol_path.write_text(
            json.dumps(
                {
                    **data,
                    "sha256": json_hash(data),
                },
                indent=2,
            ) + "\n"
        )

        print(
            "ML-073 plan frozen:",
            protocol_path,
        )
        return

    assert_frozen(
        protocol_path, data
    )

    device = ml.resolve_device(
        args.device
    )

    if args.phase == "validate":
        # Already-known calibration constructions.
        # The reserved target is not examined here.
        cases = ml.make_cases(
            [
                tuple(x)
                for x in old[
                    "calibration_pairs"
                ][:2]
            ],
            ml.VERBS[:2],
            (("active", "alternate"),),
        )

        checks = []

        for seed in SEEDS:
            for arm in ARMS:
                model = model_from_checkpoint(
                    args.ml072_root,
                    old,
                    seed,
                    arm,
                    device,
                )

                _, difference, error = run_model(
                    model,
                    cases,
                    device,
                    check=True,
                )

                checks.append({
                    "seed": seed,
                    "arm": arm,
                    "max_logit_difference": difference,
                    "max_factorial_error": error,
                })

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        marker = (
            output_root / "validated.json"
        )

        if marker.exists():
            raise FileExistsError(marker)

        marker.write_text(
            json.dumps(
                {
                    "protocol_sha256": json_hash(
                        data
                    ),
                    "checks": checks,
                },
                indent=2,
            ) + "\n"
        )

        print(
            f"Validated {len(checks)} "
            "checkpoint computations; "
            "max logit error="
            f"{max(x['max_logit_difference'] for x in checks):.6g}; "
            "max factorial error="
            f"{max(x['max_factorial_error'] for x in checks):.6g}"
        )

        print("Saved:", marker)
        return

    marker = (
        output_root / "validated.json"
    )

    if not marker.exists():
        raise FileNotFoundError(
            "Run --phase validate first"
        )

    assert (
        json.loads(
            marker.read_text()
        )["protocol_sha256"]
        == json_hash(data)
    )

    output = (
        output_root / "evaluation.json"
    )

    if output.exists():
        raise FileExistsError(
            "Evaluation already exists"
        )

    cohorts = make_cohorts(old)

    assert len(
        cohorts["target_passive_alternate"]
    ) == 216

    report = {
        "experiment": "ML-073",
        "protocol_sha256": json_hash(data),
        "results": [],
    }

    for seed in SEEDS:
        for arm in ARMS:
            model = model_from_checkpoint(
                args.ml072_root,
                old,
                seed,
                arm,
                device,
            )

            results = {}

            for name, cases in cohorts.items():
                records, _, error = run_model(
                    model,
                    cases,
                    device,
                )

                results[name] = {
                    "summary": summarize(records),
                    "events": records,
                    "max_factorial_error": error,
                }

            report["results"].append({
                "seed": seed,
                "arm": arm,
                "groups": results,
            })

            primary = results[
                "target_passive_alternate"
            ]["summary"]

            print(
                f"seed={seed} arm={arm:7} "
                f"both={primary['correct_both_rate']:.3f} "
                f"role+={primary['role_signal_positive_fraction']:.3f} "
                f"median_delta={primary['median_role_delta']:+.2f} "
                f"med_abs_bias={primary['median_abs_answer_bias']:.2f} "
                f"L2_att={primary['mean_attention_contribution']:+.2f} "
                f"L2_input={primary['mean_input_contribution']:+.2f} "
                f"verb_control_flip={primary['verb_control_question_flip_rate']:.3f}",
                flush=True,
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n"
    )

    print(
        "\nPaired changes in correct-both rate "
        "on reserved target:"
    )

    indexed = {
        (r["seed"], r["arm"]): (
            r["groups"][
                "target_passive_alternate"
            ]["summary"]["correct_both_rate"]
        )
        for r in report["results"]
    }

    comparisons = (
        ("sparse", "control"),
        ("broad", "control"),
        ("broad", "sparse"),
    )

    for a, b in comparisons:
        values = [
            indexed[(seed, a)]
            - indexed[(seed, b)]
            for seed in SEEDS
        ]

        print(
            f"{a}-{b}: "
            + ", ".join(
                f"{v:+.3f}"
                for v in values
            )
            + f" | mean="
            f"{statistics.mean(values):+.3f}"
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
