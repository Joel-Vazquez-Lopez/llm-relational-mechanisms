
"""ML-074: Matched causal controls for grammatical role routing.

Post-hoc mechanistic audit of ML-072/073 checkpoints.
Not an independent replication.

Phases:
    plan -> validate -> evaluate
"""

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

import ml072_compositional_exposure as ml
import ml073_causal_role_balance as prev


GROUPS = {
    "active_alternate": (
        ("active", "alternate"),
    ),
    "passive_standard": (
        ("passive", "standard"),
    ),
    "passive_alternate_target": (
        ("passive", "alternate"),
    ),
}


def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def freeze_doc(old, old73):
    return {
        "experiment": "ML-074",
        "analysis_code_sha256": file_digest(
            __file__
        ),
        "ml072_protocol_sha256": old["sha256"],
        "ml073_protocol_sha256": old73["sha256"],
        "source_sha256": {
            "ml072": file_digest(ml.__file__),
            "ml073": file_digest(prev.__file__),
        },
        "seeds": list(ml.SEEDS),
        "arms": list(ml.ARMS),
        "groups": GROUPS,
        "participant_pairs": old[
            "calibration_pairs"
        ],
        "verbs": list(ml.VERBS),
        "unit_of_comparison": (
            "paired training seed (n=6)"
        ),
        "primary_measure": (
            "all-question mean correct-answer-margin "
            "damage under opposite-role layer-2 "
            "attention update, minus norm-matched "
            "same-role different-verb damage"
        ),
        "secondary_measure": (
            "same contrast versus norm-matched "
            "random-direction damage"
        ),
        "predeclared_comparisons": [
            "sparse-control on passive_alternate_target",
            "broad-control on passive_alternate_target",
            "broad-sparse on passive_alternate_target",
            "same arm across active_alternate, "
            "passive_standard and passive_alternate_target",
        ],
        "interpretation": (
            "Exploratory, post-hoc intervention comparison "
            "on an already examined benchmark. "
            "Off-manifold hybrids and changes in logit "
            "margins limit causal localization claims. "
            "No independent holdout is claimed."
        ),
        "numerical_tolerance": 0.001,
        "random_control": (
            "one fixed random vector per question, "
            "scaled to role-swap delta norm"
        ),
        "verb_control": (
            "next verb cyclically, same pair/role/"
            "voice/style, norm matched"
        ),
    }


def load_old(args):
    old = json.loads(
        Path(args.ml072_protocol).read_text()
    )

    old73 = json.loads(
        Path(args.ml073_protocol).read_text()
    )

    assert old["experiment"] == "ML-072"
    assert old73["experiment"] == "ML-073"

    assert (
        old["code_sha256"]
        == file_digest(ml.__file__)
    )

    assert (
        old73["ml073_code_sha256"]
        == file_digest(prev.__file__)
    )

    assert (
        old73["ml072_protocol_sha256"]
        == old["sha256"]
    )

    assert digest({
        k: v for k, v in old.items()
        if k != "sha256"
    }) == old["sha256"]

    assert digest({
        k: v for k, v in old73.items()
        if k != "sha256"
    }) == old73["sha256"]

    return old, old73


def check_frozen(args, plan):
    saved = json.loads(
        Path(args.protocol).read_text()
    )

    assert saved == json.loads(json.dumps({
        **plan,
        "sha256": digest(plan),
    })), (
        "ML-074 source or plan changed "
        "after freezing"
    )


@torch.inference_mode()
def analyze_batch(
    model, cases, device,
    random_seed, validate=False
):
    assert len(cases) % 2 == 0

    assert all(
        cases[i].role == "agent"
        and cases[i + 1].role == "patient"
        and ml.event_key(cases[i])
        == ml.event_key(cases[i + 1])
        for i in range(0, len(cases), 2)
    )

    x, a, ids, readout = prev.internal_states(
        model, cases, device
    )

    n = len(cases)
    index = torch.arange(n, device=device)

    # Opposite-role attention donor.
    opposite = index ^ 1

    role_delta = a[opposite] - a

    target_norm = role_delta.norm(
        dim=1, keepdim=True
    )

    # Same role, same participants, different verb.
    verb_donors = [
        replace(
            c,
            verb=ml.VERBS[
                (
                    ml.VERBS.index(c.verb) + 1
                ) % len(ml.VERBS)
            ],
        )
        for c in cases
    ]

    _, donor_a, _, _ = prev.internal_states(
        model, verb_donors, device
    )

    verb_delta = donor_a - a

    vnorm = verb_delta.norm(
        dim=1, keepdim=True
    )

    degenerate = vnorm < 1e-8

    scale = (
        target_norm / vnorm.clamp_min(1e-8)
    )

    scaled_verb_delta = torch.where(
        degenerate,
        torch.zeros_like(verb_delta),
        verb_delta * scale,
    )

    # Fixed random-direction control.
    # Generated on CPU for reproducible sampling.
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(random_seed)

    noise = torch.randn(
        (n, a.shape[1]),
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    noise = noise.to(a.dtype)

    random_delta = (
        noise
        / noise.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        * target_norm
    )

    # Four downstream computations using
    # the same layer-2 receiver input state.
    base = prev.readout_logits(
        model, x + a
    )

    role = prev.readout_logits(
        model, x + a + role_delta
    )

    verb = prev.readout_logits(
        model, x + a + scaled_verb_delta
    )

    rand = prev.readout_logits(
        model, x + a + random_delta
    )

    if validate:
        original = model(
            ids, readout
        ).float()

        error = float(
            (original - base).abs().max().item()
        )

        if error > 0.001:
            raise AssertionError(
                "Natural forward mismatch: "
                f"{error}"
            )

        controls = (
            ("random", random_delta),
            ("verb", scaled_verb_delta),
        )

        for label, delta in controls:
            if label == "verb":
                mask = ~degenerate.squeeze(1)
            else:
                mask = torch.ones(
                    n,
                    dtype=torch.bool,
                    device=device,
                )

            if mask.any():
                mismatch = (
                    delta.norm(dim=1)[mask]
                    - target_norm.squeeze(1)[mask]
                ).abs().max().item()

                if mismatch > 0.001:
                    raise AssertionError(
                        f"{label} norm mismatch: "
                        f"{mismatch}"
                    )
    else:
        error = None

    # Convert both question types into
    # the same correct-answer margin convention.
    names_agent = [
        c.pair[0]
        if c.direction == 0
        else c.pair[1]
        for c in cases
    ]

    names_patient = [
        c.pair[1]
        if c.direction == 0
        else c.pair[0]
        for c in cases
    ]

    agents = torch.tensor(
        [ml.VOCAB[t] for t in names_agent],
        device=device,
    )

    patients = torch.tensor(
        [ml.VOCAB[t] for t in names_patient],
        device=device,
    )

    signs = torch.tensor(
        [
            1 if c.role == "agent" else -1
            for c in cases
        ],
        device=device,
    )

    def correct_margin(logits):
        return (
            logits[index, agents]
            - logits[index, patients]
        ) * signs

    natural_margin = correct_margin(base)

    damages = {
        "role": (
            natural_margin - correct_margin(role)
        ),
        "verb": (
            natural_margin - correct_margin(verb)
        ),
        "random": (
            natural_margin - correct_margin(rand)
        ),
    }

    result = []

    for i, c in enumerate(cases):
        result.append({
            "pair": list(c.pair),
            "verb": c.verb,
            "direction": c.direction,
            "voice": c.voice,
            "style": c.style,
            "role": c.role,
            "correct_natural": bool(
                natural_margin[i] > 0
            ),
            "natural_margin": float(
                natural_margin[i]
            ),
            "role_damage": float(
                damages["role"][i]
            ),
            "verb_damage": float(
                damages["verb"][i]
            ),
            "random_damage": float(
                damages["random"][i]
            ),
            "role_delta_norm": float(
                target_norm[i]
            ),
            "verb_scale": (
                float(scale[i])
                if not bool(degenerate[i])
                else None
            ),
            "verb_degenerate": bool(
                degenerate[i]
            ),
        })

    return result, error


def summarize(records):
    def avg(field):
        return statistics.mean(
            r[field] for r in records
        )

    scales = [
        r["verb_scale"]
        for r in records
        if r["verb_scale"] is not None
    ]

    return {
        "questions": len(records),
        "correct_question_rate": (
            statistics.mean(
                r["correct_natural"]
                for r in records
            )
        ),
        "mean_role_damage": avg(
            "role_damage"
        ),
        "mean_verb_damage": avg(
            "verb_damage"
        ),
        "mean_random_damage": avg(
            "random_damage"
        ),
        "role_minus_verb": (
            statistics.mean(
                r["role_damage"]
                - r["verb_damage"]
                for r in records
            )
        ),
        "role_minus_random": (
            statistics.mean(
                r["role_damage"]
                - r["random_damage"]
                for r in records
            )
        ),
        "fraction_role_damage_positive": (
            statistics.mean(
                r["role_damage"] > 0
                for r in records
            )
        ),
        "mean_role_delta_norm": avg(
            "role_delta_norm"
        ),
        "median_verb_norm_scale": (
            statistics.median(scales)
            if scales else None
        ),
        "degenerate_verb_count": sum(
            r["verb_degenerate"]
            for r in records
        ),
    }


def run_group(
    model, cases, device,
    seed, arm_index, group_index,
    validate=False
):
    rows = []
    errors = []

    for offset in range(
        0, len(cases), 64
    ):
        batch, err = analyze_batch(
            model,
            cases[offset:offset + 64],
            device,
            7401
            + seed * 1000
            + group_index * 10
            + offset,
            validate=validate,
        )

        rows.extend(batch)

        if err is not None:
            errors.append(err)

    return rows, max(errors, default=None)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=("plan", "validate", "evaluate"),
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
        "--ml073-protocol",
        default="reports/ml073/frozen_protocol.json",
    )

    parser.add_argument(
        "--protocol",
        default="reports/ml074/frozen_protocol.json",
    )

    parser.add_argument(
        "--root",
        default="results/ml074_matched_controls",
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

    old, old73 = load_old(args)
    plan = freeze_doc(old, old73)

    protocol_path = Path(args.protocol)
    outroot = Path(args.root)

    if args.phase == "plan":
        if protocol_path.exists():
            check_frozen(args, plan)
            print(
                "Already frozen:",
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
                    **plan,
                    "sha256": digest(plan),
                },
                indent=2,
            ) + "\n"
        )

        print(
            "Frozen:",
            protocol_path,
        )
        return

    check_frozen(args, plan)

    device = ml.resolve_device(
        args.device
    )

    pairs = [
        tuple(p)
        for p in old["calibration_pairs"]
    ]

    assert len(pairs) == 9

    assert not (
        set(pairs)
        & set(
            map(
                tuple,
                old["train_pairs"],
            )
        )
    )

    outroot.mkdir(
        parents=True, exist_ok=True
    )

    if args.phase == "validate":
        # Validate on familiar grammar only.
        # Do not inspect target examples.
        cases = ml.make_cases(
            pairs[:2],
            ml.VERBS[:2],
            GROUPS["active_alternate"],
        )

        errors = []

        for seed in ml.SEEDS:
            for arm_idx, arm in enumerate(ml.ARMS):
                model = prev.model_from_checkpoint(
                    args.ml072_root,
                    old,
                    seed,
                    arm,
                    device,
                )

                _, err = run_group(
                    model,
                    cases,
                    device,
                    seed,
                    arm_idx,
                    0,
                    validate=True,
                )

                errors.append(err)

        check = {
            "protocol_sha256": digest(plan),
            "max_reconstruction_error": (
                max(errors)
            ),
            "models_validated": len(errors),
        }

        path = outroot / "validated.json"

        if path.exists():
            assert (
                json.loads(path.read_text())
                == check
            )
        else:
            path.write_text(
                json.dumps(
                    check, indent=2
                ) + "\n"
            )

        print(
            f"Validated {len(errors)} models; "
            "max reconstruction error="
            f"{max(errors):.3g}"
        )

        print("Saved:", path)
        return

    path = outroot / "validated.json"

    if (
        not path.exists()
        or json.loads(
            path.read_text()
        )["protocol_sha256"] != digest(plan)
    ):
        raise RuntimeError(
            "Run --phase validate first"
        )

    output = outroot / "evaluation.json"

    if output.exists():
        raise FileExistsError(output)

    report = {
        "experiment": "ML-074",
        "protocol_sha256": digest(plan),
        "results": [],
    }

    for seed in ml.SEEDS:
        for arm_idx, arm in enumerate(ml.ARMS):
            model = prev.model_from_checkpoint(
                args.ml072_root,
                old,
                seed,
                arm,
                device,
            )

            cohorts = {}

            for group_idx, (
                group, grammar
            ) in enumerate(GROUPS.items()):

                cases = ml.make_cases(
                    pairs,
                    ml.VERBS,
                    grammar,
                )

                rows, _ = run_group(
                    model,
                    cases,
                    device,
                    seed,
                    arm_idx,
                    group_idx,
                )

                cohorts[group] = {
                    "summary": summarize(rows),
                    "questions": rows,
                }

            report["results"].append({
                "seed": seed,
                "arm": arm,
                "cohorts": cohorts,
            })

            for group in GROUPS:
                s = cohorts[group]["summary"]

                print(
                    f"seed={seed} "
                    f"arm={arm:7s} "
                    f"grammar={group:24s} "
                    f"accuracy={s['correct_question_rate']:.3f} "
                    f"role-verb={s['role_minus_verb']:+.3f} "
                    f"role-random={s['role_minus_random']:+.3f}",
                    flush=True,
                )

    indexed = {
        (x["seed"], x["arm"]): x
        for x in report["results"]
    }

    print(
        "\nMatched-seed comparisons on target "
        "(role damage minus norm-matched "
        "verb damage):"
    )

    comparisons = (
        ("sparse", "control"),
        ("broad", "control"),
        ("broad", "sparse"),
    )

    for arm_a, arm_b in comparisons:
        diffs = [
            indexed[
                (seed, arm_a)
            ]["cohorts"][
                "passive_alternate_target"
            ]["summary"]["role_minus_verb"]
            - indexed[
                (seed, arm_b)
            ]["cohorts"][
                "passive_alternate_target"
            ]["summary"]["role_minus_verb"]
            for seed in ml.SEEDS
        ]

        print(
            f"{arm_a}-{arm_b}: "
            + ", ".join(
                f"{v:+.3f}"
                for v in diffs
            )
            + f" | mean="
            f"{statistics.mean(diffs):+.3f}"
        )

    output.write_text(
        json.dumps(
            report, indent=2
        ) + "\n"
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
