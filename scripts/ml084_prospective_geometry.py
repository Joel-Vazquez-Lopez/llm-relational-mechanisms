
"""ML-084: fresh paired replication and layer-2 geometry prediction.

Run from repository root:

python3 scripts/ml084_prospective_geometry.py --phase plan
python3 scripts/ml084_prospective_geometry.py --phase preflight

# Commit script and frozen protocol BEFORE training.

python3 scripts/ml084_prospective_geometry.py --phase train --until 3200
python3 scripts/ml084_prospective_geometry.py --phase summarize
python3 scripts/ml084_prospective_geometry.py --phase train --until 12800
python3 scripts/ml084_prospective_geometry.py --phase summarize

python3 scripts/ml084_prospective_geometry.py --phase calibrate

# Commit calibration predictions BEFORE protected evaluation.

python3 scripts/ml084_prospective_geometry.py \
    --phase evaluate --confirm-protected

Protected examples are generated only after the competence gate,
prediction freeze, and explicit unlock.

Fresh names, participant-pair partitions, and seeds.
Same synthetic grammar and Transformer architecture as ML-080.
"""

import argparse
import hashlib
import itertools
import json
import math
import random

from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml081_layerwise_event_role as m81


NEW_NAMES = (
    "Aster", "Beryl", "Caden", "Doria",
    "Elian", "Freya", "Galen", "Hester",
    "Imani", "Jorah", "Keira", "Loren",
    "Maia", "Nolan", "Orla", "Perrin",
)

NEW_SEEDS = (
    401, 409, 419, 421,
    431, 433, 439, 443,
)

MATCHING_SEED = 84084
GEOMETRY_SENTENCES = 64
GEOMETRY_LAYER = 1
MILESTONES = (3200, 12800)


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def source_sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def configure_generator():
    """Reuse the frozen ML-080 architecture and training code.

    Only the dataset vocabulary and training seeds change.
    The original source file remains untouched.
    """
    assert source_sha(ml.__file__) == (
        "827a22367605dc0db93165d53c022de4e007ff00729de75a9216e10c56a438e8"
    ), "ML-080 source changed; stop."

    assert len(NEW_NAMES) == len(set(NEW_NAMES)) == 16
    assert not set(NEW_NAMES) & set(ml.NAMES)

    common = (
        "<pad>", "<bos>", "<sep>", "<answer>",
        "While", "while", "was", "by",
        ",", ".", "In", "the", "first",
        "second", "event", "who",
        "agent", "patient", "?",
    )

    vocab = tuple(dict.fromkeys(
        common + NEW_NAMES + tuple(ml.VERBS)
    ))

    assert len(vocab) == (
        len(common) + len(NEW_NAMES) + len(ml.VERBS)
    )

    ml.NAMES = NEW_NAMES
    ml.SEEDS = NEW_SEEDS
    ml.VOCAB_WORDS = vocab
    ml.VOCAB = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    return vocab


def fresh_matchings():
    """A new balanced K16 pair partition."""

    players = list(range(len(NEW_NAMES)))
    rounds = []

    for _ in range(len(NEW_NAMES) - 1):
        pairs = [
            tuple(sorted((
                NEW_NAMES[players[i]],
                NEW_NAMES[players[-1-i]],
            )))
            for i in range(len(NEW_NAMES) // 2)
        ]

        rounds.append(pairs)

        players = [
            players[0],
            players[-1],
            *players[1:-1],
        ]

    observed = {
        pair
        for rd in rounds
        for pair in rd
    }

    expected = set(
        itertools.combinations(NEW_NAMES, 2)
    )

    assert observed == expected

    rng = random.Random(MATCHING_SEED)
    rng.shuffle(rounds)

    for rd in rounds:
        rng.shuffle(rd)

    return rounds[:9], rounds[9:12], rounds[12:]


def calibration_grid(cal):
    """Four question combinations from familiar calibration."""

    second = [
        c for c in cal
        if c.target_pos == 1
        and c.target_voice == "active"
        and c.distractor_voice == "active"
        and c.role == "agent"
    ]

    assert len(second) == 576

    second.sort(
        key=lambda c: hashlib.sha256(
            " ".join(ml.prompt(c)).encode()
        ).hexdigest()
    )

    selected = second[:GEOMETRY_SENTENCES]

    grid = {
        k: []
        for k in ("00", "01", "10", "11")
    }

    for s_agent in selected:
        f_agent = m81.swap_event(s_agent)

        f_patient = replace(
            f_agent,
            role="patient",
        )

        s_patient = replace(
            s_agent,
            role="patient",
        )

        assert m81.changed_tokens(
            f_agent, f_patient
        ) == [("agent", "patient")]

        assert m81.changed_tokens(
            f_agent, s_agent
        ) == [("first", "second")]

        assert len({
            ml.answer(c)
            for c in (
                f_agent,
                f_patient,
                s_agent,
                s_patient,
            )
        }) == 4

        for k, c in zip(
            grid,
            (
                f_agent,
                f_patient,
                s_agent,
                s_patient,
            ),
        ):
            grid[k].append(c)

    assert all(
        len(v) == GEOMETRY_SENTENCES
        for v in grid.values()
    )

    assert len({
        tuple(ml.prompt(c))
        for c in grid["00"]
    }) == GEOMETRY_SENTENCES

    return grid


def prepare():
    vocab = configure_generator()

    train, cal, held = fresh_matchings()

    base = ml.base_cases(train)
    calibration = ml.base_cases(cal)
    dose = ml.exposure_pair(train)

    ml.verify_cases(
        base,
        calibration,
        dose,
        train,
        cal,
        held,
    )

    grid = calibration_grid(calibration)

    assert all(
        len(ml.prompt(c)) <= 64
        for cases in grid.values()
        for c in cases
    )

    return (
        vocab,
        train,
        cal,
        held,
        base,
        calibration,
        dose,
        grid,
    )


def manifest(vocab, train, cal, held, grid):
    return {
        "experiment": "ML-084",
        "script_sha256": source_sha(__file__),
        "ml080_source_sha256": source_sha(ml.__file__),
        "ml081_source_sha256": source_sha(m81.__file__),
        "new_names": NEW_NAMES,
        "verbs": tuple(ml.VERBS),
        "vocab": vocab,
        "new_seeds": NEW_SEEDS,
        "matching_seed": MATCHING_SEED,
        "splits": {
            "training_pairs": train,
            "calibration_pairs": cal,
            "protected_pair_ids": held,
        },
        "split_rule":
            "Fresh K16 one-factorization, 9/3/3 rounds; pair-disjoint",
        "training_rule":
            "Same frozen ML-080 architecture, optimizer, "
            "minibatch, dose and steps",
        "familiar_exposure":
            "64 first-position passive questions, repeatedly sampled",
        "target_exposure":
            "64 second-position passive questions, repeatedly sampled",
        "magnitude_note":
            "64 distinct questions, not 64 optimization examples; "
            "exposure repeats for 12800 steps",
        "architecture": {
            "layers": ml.LAYERS,
            "width": ml.WIDTH,
            "heads": ml.HEADS,
        },
        "steps": ml.TOTAL_STEPS,
        "milestones": MILESTONES,
        "geometry": {
            "site": "L2 attention output, final answer token",
            "sentences": GEOMETRY_SENTENCES,
            "sample_sha256": digest({
                k: [ml.prompt(c) for c in v]
                for k, v in grid.items()
            }),
            "score":
                "Median ||h11-h10-h01+h00|| / ||h11-h00||",
            "additional": [
                "absolute interaction norm",
                "joint norm",
                "candidate gold margin",
            ],
            "initialization":
                "torch.manual_seed(seed); freshly initialized model",
            "stage":
                "Before protected evaluation; calibration-only "
                "active/active familiar questions",
        },
        "competence_gate":
            "At least 6/8 seeds in each arm have >=0.95 "
            "familiar calibration choice and >=0.90 correct-both "
            "at step 12800",
        "primary_behavioral_test":
            "Protected target choice accuracy paired "
            "target-minus-familiar mean >0.20 AND "
            "positive in >=6/8 paired seeds",
        "secondary_training_prediction":
            "L2 relative interaction increases from initialization "
            "to step 12800 in >=6/8 seeds for EACH arm",
        "exploratory_arm_prediction":
            "L2 relative interaction lower in target than familiar "
            "(negative paired mean); uncertain sign/size",
        "prospective_ranking": {
            "feature":
                "Negative target-minus-familiar L2 relative "
                "interaction at final calibration",
            "target":
                "Paired protected target-choice accuracy gap",
            "hypothesis":
                "Greater reduction in relative interaction ranks "
                "higher in protected benefit",
            "metric":
                "Spearman rank correlation across 8 paired seeds; "
                "positive expected",
            "baseline_feature":
                "Target-minus-familiar mean 4-choice "
                "gold-vs-best-competitor calibration logit margin",
            "important":
                "Weak exploratory association with n=8; "
                "no protected-dependent site selection",
        },
        "evaluation_guard":
            "Programmatic final competence gate, frozen calibration "
            "predictions, explicit unlock and no result overwrite",
        "caveats":
            "Same synthetic grammar and architecture; fresh names, "
            "splits and seeds are not natural-language validation",
    }


def frozen(path, candidate, create):
    expected = json.loads(json.dumps({
        **candidate,
        "sha256": digest(candidate),
    }))

    if create and not path.exists():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(expected, indent=2) + "\n"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Freeze manifest first: --phase plan ({path})"
        )

    assert json.loads(path.read_text()) == expected, (
        "ML-084 source or protocol changed since freeze"
    )

    return expected["sha256"]


def rows_at(root, h, strict=True):
    rows = ml.all_metrics(
        root,
        h,
        12800,
        complete=strict,
    )

    if strict:
        assert len(rows) == 16

    return rows


def gate(root, h):
    rows = rows_at(root, h)

    report = {}

    for arm in ml.ARMS:
        count = sum(
            rows[(seed, arm)]["calibration"]["choice_accuracy"]
                >= 0.95
            and
            rows[(seed, arm)]["calibration"]["correct_both_rate"]
                >= 0.90
            for seed in NEW_SEEDS
        )

        report[arm] = count

    passed = all(
        c >= 6
        for c in report.values()
    )

    return passed, report


def inputs(cases, device):
    x, readout, lab, choices = ml.encode(cases)

    return (
        x.to(device),
        readout.to(device),
        lab.to(device),
        choices.to(device),
    )


@torch.inference_mode()
def geometry(model, grid, device):
    traces = {}

    for corner, cases in grid.items():
        x, ro, _, _ = inputs(cases, device)

        _, trace = m81.forward_trace(
            model,
            x,
            ro,
        )

        traces[corner] = (
            trace[GEOMETRY_LAYER]["attn"]
        )

    a, b, c, d = (
        traces[k]
        for k in ("00", "01", "10", "11")
    )

    interaction = (
        d - c - b + a
    ).norm(dim=1)

    joint = (d - a).norm(dim=1)

    return {
        "relative_interaction": float(
            (
                interaction / joint.clamp_min(1e-12)
            ).median()
        ),
        "absolute_interaction": float(
            interaction.median()
        ),
        "joint_change": float(
            joint.median()
        ),
        "small_joint_rate": float(
            (joint < 1e-8).float().mean()
        ),
    }


@torch.inference_mode()
def candidate_margin(model, grid, device):
    scores = []

    for cases in grid.values():
        x, ro, gold, choices = inputs(
            cases, device
        )

        logits = model(x, ro).float()

        scores.append(
            m81.margin(
                logits,
                gold,
                choices,
            )
        )

    return float(
        torch.cat(scores).mean()
    )


def freeze_predictions(path, payload):
    content = json.loads(json.dumps({
        **payload,
        "sha256": digest(payload),
    }))

    if path.exists():
        raise FileExistsError(
            f"Predictions already frozen: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(content, indent=2) + "\n"
    )

    return content


def read_predictions(path, h):
    obj = json.loads(path.read_text())

    assert obj["sha256"] == digest({
        k: v
        for k, v in obj.items()
        if k != "sha256"
    }), "Prediction freeze modified"

    assert obj["protocol_sha256"] == h
    assert len(obj["seeds"]) == 8

    assert set(map(int, obj["seeds"].keys())) == set(
        NEW_SEEDS
    )

    return obj


def ranking(values):
    """Midranks with exact equality."""

    sorted_idx = sorted(
        range(len(values)),
        key=lambda k: values[k],
    )

    ranks = [0.0] * len(values)
    i = 0

    while i < len(values):
        j = i + 1

        while (
            j < len(values)
            and values[sorted_idx[j]]
            == values[sorted_idx[i]]
        ):
            j += 1

        rank = (i + j + 1) / 2.0

        for k in range(i, j):
            ranks[sorted_idx[k]] = rank

        i = j

    return ranks


def spearman(a, b):
    x = ranking(a)
    y = ranking(b)

    mx, my = mean(x), mean(y)

    xx = sum(
        (v - mx) ** 2
        for v in x
    )

    yy = sum(
        (v - my) ** 2
        for v in y
    )

    if xx * yy <= 0:
        return None

    return sum(
        (u - mx) * (v - my)
        for u, v in zip(x, y)
    ) / math.sqrt(xx * yy)


def bootstrap_gap(delta):
    rng = random.Random(840843)

    dist = sorted(
        mean(
            delta[rng.randrange(len(delta))]
            for _ in delta
        )
        for _ in range(20000)
    )

    return [dist[500], dist[19500]]


def do_calibrate(root, h, grid, device, out):
    passed, counts = gate(root, h)

    if not passed:
        print(
            "FAMILIAR COMPETENCE GATE FAILED",
            counts,
        )
        print(
            "Do not open protected examples "
            "or select successful seeds."
        )
        return

    if out.exists():
        raise FileExistsError(
            f"Refusing overwrite of frozen calibration: {out}"
        )

    rows = {}

    for seed in NEW_SEEDS:
        torch.manual_seed(seed)

        init = ml.Model().to(device).eval()
        init_geo = geometry(
            init, grid, device
        )

        arms = {}

        for arm in ml.ARMS:
            checkpoint = (
                root
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint,
                seed,
                arm,
                h,
                device,
            )

            arms[arm] = {
                "geometry": geometry(
                    model, grid, device
                ),
                "calibration_margin": candidate_margin(
                    model, grid, device
                ),
            }

        delta_geo = (
            arms["target"]["geometry"]["relative_interaction"]
            - arms["familiar"]["geometry"]["relative_interaction"]
        )

        delta_margin = (
            arms["target"]["calibration_margin"]
            - arms["familiar"]["calibration_margin"]
        )

        rows[str(seed)] = {
            "initial": init_geo,
            "familiar": arms["familiar"],
            "target": arms["target"],
            "delta_target_minus_familiar_relative": delta_geo,
            "predicted_protected_benefit_rank_score": -delta_geo,
            "baseline_margin_rank_score": delta_margin,
        }

        print(
            f"seed={seed}: "
            f"initial_R={init_geo['relative_interaction']:.3f} "
            f"familiar_R="
            f"{arms['familiar']['geometry']['relative_interaction']:.3f} "
            f"target_R="
            f"{arms['target']['geometry']['relative_interaction']:.3f} "
            f"prediction_score={-delta_geo:+.3f}",
            flush=True,
        )

    payload = {
        "experiment": "ML-084",
        "protocol_sha256": h,
        "calibration_only": True,
        "gate_counts": counts,
        "prediction":
            "Higher -delta_relative_L2 predicts higher "
            "paired protected target benefit; rank correlation "
            "tested after unlock",
        "baseline_prediction":
            "Higher delta candidate logit margin predicts "
            "higher paired protected target benefit",
        "seeds": rows,
    }

    doc = freeze_predictions(
        out,
        payload,
    )

    print(
        "FROZEN CALIBRATION/PREDICTIONS:",
        out,
        doc["sha256"][:12],
    )

    print(
        "Commit this file BEFORE opening "
        "the protected evaluation."
    )


def do_evaluate(
    root,
    h,
    held,
    predictions_path,
    out,
    device,
    confirm,
):
    if not confirm:
        raise PermissionError(
            "Use --confirm-protected explicitly."
        )

    if out.exists():
        raise FileExistsError(
            f"Protected results already exist: {out}"
        )

    passed, counts = gate(root, h)

    if not passed:
        raise PermissionError(
            f"Competence gate failed: {counts}"
        )

    predictions = read_predictions(
        predictions_path,
        h,
    )

    # Protected examples are generated ONLY after
    # all checks above have passed.
    familiar = ml.base_cases(held)
    target = ml.protected_target_cases(held)

    results = []

    for seed in NEW_SEEDS:
        for arm in ml.ARMS:
            checkpoint = (
                root
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint,
                seed,
                arm,
                h,
                device,
            )

            results.append({
                "seed": seed,
                "arm": arm,
                "familiar_new_pairs": ml.measure(
                    model, familiar, device
                ),
                "target_new_pairs_new_exposure_verbs": ml.measure(
                    model, target, device
                ),
            })

            score = results[-1][
                "target_new_pairs_new_exposure_verbs"
            ]["choice_accuracy"]

            print(
                f"seed={seed} arm={arm} "
                f"protected target choice={score:.3f}",
                flush=True,
            )

    lookup = {
        (r["seed"], r["arm"]): r
        for r in results
    }

    gaps = [
        lookup[(s, "target")][
            "target_new_pairs_new_exposure_verbs"
        ]["choice_accuracy"]
        - lookup[(s, "familiar")][
            "target_new_pairs_new_exposure_verbs"
        ]["choice_accuracy"]
        for s in NEW_SEEDS
    ]

    predictor = [
        predictions["seeds"][str(s)][
            "predicted_protected_benefit_rank_score"
        ]
        for s in NEW_SEEDS
    ]

    baseline = [
        predictions["seeds"][str(s)][
            "baseline_margin_rank_score"
        ]
        for s in NEW_SEEDS
    ]

    rho = spearman(predictor, gaps)
    baseline_rho = spearman(baseline, gaps)

    ci = bootstrap_gap(gaps)

    summary = {
        "paired_mean_target_minus_familiar": mean(gaps),
        "paired_bootstrap_95pct_ci": ci,
        "positive_seed_pairs": sum(
            v > 0 for v in gaps
        ),
        "per_seed_gaps": dict(
            zip(
                map(str, NEW_SEEDS),
                gaps,
            )
        ),
        "behavioral_success_rule_met":
            mean(gaps) > 0.20
            and sum(v > 0 for v in gaps) >= 6,
        "geometry_rank_spearman": rho,
        "margin_baseline_rank_spearman": baseline_rho,
        "n_independent_seed_pairs": len(NEW_SEEDS),
    }

    doc = {
        "experiment": "ML-084",
        "protocol_sha256": h,
        "calibration_predictions_sha256":
            predictions["sha256"],
        "protected_evaluation_opened": True,
        "runs": results,
        "summary": summary,
    }

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(doc, indent=2) + "\n"
    )

    print("\nPROTECTED SUMMARY")
    print(json.dumps(summary, indent=2))

    print("Saved:", out)
    print(
        "One-time protected test. "
        "Do not rerun, tune predictors or discard seeds."
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan",
            "preflight",
            "train",
            "summarize",
            "calibrate",
            "evaluate",
        ),
    )

    ap.add_argument(
        "--protocol",
        default="reports/ml084/frozen_protocol.json",
    )

    ap.add_argument(
        "--root",
        default="results/ml084_fresh_learning_curve",
    )

    ap.add_argument(
        "--predictions",
        default="reports/ml084/calibration_predictions.json",
    )

    ap.add_argument(
        "--output",
        default="reports/ml084/protected_evaluation.json",
    )

    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    ap.add_argument(
        "--until",
        type=int,
        choices=MILESTONES,
        default=3200,
    )

    ap.add_argument(
        "--seed",
        type=int,
        choices=NEW_SEEDS,
    )

    ap.add_argument(
        "--arm",
        choices=ml.ARMS,
    )

    ap.add_argument(
        "--save-every",
        type=int,
        default=400,
    )

    ap.add_argument(
        "--confirm-protected",
        action="store_true",
    )

    args = ap.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    (
        vocab,
        train,
        cal,
        held,
        base,
        calibration,
        dose,
        grid,
    ) = prepare()

    h = frozen(
        Path(args.protocol),
        manifest(
            vocab,
            train,
            cal,
            held,
            grid,
        ),
        args.phase == "plan",
    )

    if args.phase == "plan":
        print(
            f"ML-084 frozen: {args.protocol} "
            f"sha256={h[:12]}"
        )
        return

    if args.phase == "preflight":
        torch.manual_seed(84084)

        model = ml.Model().eval()

        small, readout, _, _ = inputs(
            grid["00"][:4], "cpu"
        )

        with torch.inference_mode():
            raw, _ = m81.forward_trace(
                model,
                small,
                readout,
            )

            parity = float(
                (raw - model(small, readout))
                .abs().max()
            )

            assert parity < 1e-3

            test = geometry(
                model,
                {
                    k: v[:4]
                    for k, v in grid.items()
                },
                "cpu",
            )

            assert math.isfinite(
                test["relative_interaction"]
            )

        print(
            "PREFLIGHT PASS: 16 fresh names, "
            "72/24/24 disjoint pairs, "
            f"{len(base)} training, "
            f"{len(calibration)} calibration questions"
        )

        print(
            "Exposure=64 distinct questions/arm; "
            f"{GEOMETRY_SENTENCES} geometry sentences; "
            f"parity={parity:.2e}"
        )

        print(
            "Protected case generator not invoked. "
            "No trained checkpoints accessed."
        )
        return

    root = Path(args.root)

    if args.phase == "train":
        if args.save_every <= 0:
            raise ValueError(
                "--save-every must be positive"
            )

        device = ml.select_device(
            args.device
        )

        for seed in (
            NEW_SEEDS
            if args.seed is None
            else (args.seed,)
        ):
            for arm in (
                ml.ARMS
                if args.arm is None
                else (args.arm,)
            ):
                ml.train_one(
                    seed,
                    arm,
                    args.until,
                    args.save_every,
                    root,
                    h,
                    base,
                    calibration,
                    dose,
                    device,
                )

        print(
            "Training checkpoints saved. "
            "Protected questions remain sealed."
        )
        return

    if args.phase == "summarize":
        ml.summarize(root, h)
        return

    if args.phase == "calibrate":
        do_calibrate(
            root,
            h,
            grid,
            ml.select_device(args.device),
            Path(args.predictions),
        )
        return

    do_evaluate(
        root,
        h,
        held,
        Path(args.predictions),
        Path(args.output),
        ml.select_device(args.device),
        args.confirm_protected,
    )


if __name__ == "__main__":
    main()
