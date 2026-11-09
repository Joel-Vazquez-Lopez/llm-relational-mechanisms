
"""ML-085: prospective causal transfer on withheld passive-second grammar.

Run from repository root:
  python3 scripts/ml085_causal_transfer.py --phase plan
  python3 scripts/ml085_causal_transfer.py --phase preflight

Freeze and commit BEFORE training.

  python3 scripts/ml085_causal_transfer.py --phase train --until 3200
  python3 scripts/ml085_causal_transfer.py --phase summarize
  python3 scripts/ml085_causal_transfer.py --phase train --until 12800
  python3 scripts/ml085_causal_transfer.py --phase summarize

Only after final competence gate passes:
  python3 scripts/ml085_causal_transfer.py --phase evaluate --confirm-protected

Fresh models, names, splits and paired seeds.
Original ML-080 architecture and training implementation reused.
Protected questions are generated only after the gate and explicit unlock.
"""

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean

import torch

import ml080_learning_curve as ml
import ml081_layerwise_event_role as m81
import ml084_prospective_geometry as m84


NAMES = (
    "Althea", "Boris", "Celia", "Darwin",
    "Eamon", "Felix", "Greta", "Hugo",
    "Ivy", "Jasper", "Kiran", "Leona",
    "Mabel", "Nikos", "Ophelia", "Quinn",
)

SEEDS = (503, 509, 521, 523, 541, 547, 557, 563)

MATCHING_SEED = 85085
N_PROBE_EVENTS = 192
SITE = (1, "attn")  # L2 attention, zero-based
RANDOM_REPS = 3
CHUNK_EVENTS = 24


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def sha_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def setup():
    """Configure fresh data; never edit previous source files."""
    assert len(NAMES) == len(set(NAMES)) == 16
    assert not set(NAMES) & set(m84.NEW_NAMES)
    assert not set(NAMES) & set(ml.NAMES)

    m84.NEW_NAMES = NAMES
    m84.NEW_SEEDS = SEEDS
    m84.MATCHING_SEED = MATCHING_SEED

    vocab, tr, ca, held, base, cal, dose, grid = m84.prepare()

    assert tuple(ml.SEEDS) == SEEDS
    assert tuple(ml.NAMES) == NAMES
    assert len(base) == 20736
    assert len(cal) == 6912
    assert len(dose["target"]) == 64
    assert len(dose["familiar"]) == 64

    return vocab, tr, ca, held, base, cal, dose, grid


def manifest(vocab, tr, ca, held):
    return {
        "experiment": "ML-085",
        "script_sha256": sha_file(__file__),
        "ml080_sha256": sha_file(ml.__file__),
        "ml081_sha256": sha_file(m81.__file__),
        "ml084_sha256": sha_file(m84.__file__),
        "seeds": SEEDS,
        "arms": tuple(ml.ARMS),
        "names": NAMES,
        "verbs": tuple(ml.VERBS),
        "vocab_sha256": digest(vocab),
        "matching_seed": MATCHING_SEED,
        "splits": {
            "train": tr,
            "calibration": ca,
            "protected": held,
        },
        "training": {
            "architecture": "frozen ML-080 3-layer 96-wide causal Transformer",
            "steps": 12800,
            "milestones": [3200, 12800],
            "base_batch": 56,
            "repeated_exposure_batch": 8,
            "exposure_questions_per_arm": 64,
            "contrast": "first-position passive vs second-position passive exposure",
        },
        "competence_gate": {
            "min_passing_seeds_per_arm": 6,
            "min_calibration_accuracy": 0.95,
            "min_calibration_correct_both": 0.90,
            "evaluated_at_step": 12800,
        },
        "protected_behavior": (
            "Four-candidate accuracy on all protected second-passive "
            "questions; paired target-minus-familiar across eight seeds"
        ),
        "causal_probe": {
            "sample": (
                "192 SHA256-sorted protected second-passive agent "
                "event instances, two roles each"
            ),
            "site": "L2 attention output at final <answer> token",
            "reference": (
                "Recipient is a protected target question; "
                "other-role donor differs only in agent/patient cue"
            ),
            "main_patch": (
                "Replace recipient L2 attention update with "
                "same-sentence other-role donor update"
            ),
            "structured_control": (
                "Same-sentence first-event, same-role attention "
                "delta normalized to role-delta norm"
            ),
            "shuffled_role_control": (
                "Role-cue delta from another selected protected "
                "probe sentence, shifted by one event pair and norm matched"
            ),
            "random_control": (
                "Three isotropic Gaussian directions individually "
                "matched to role-delta norm"
            ),
            "lexical_control": (
                "Answer-preserving distractor verb replacement; "
                "not norm matched"
            ),
            "native_filter": (
                "Primary uses all selected questions without conditioning; "
                "secondary is restricted to native both-role-correct "
                "event pairs, with counts"
            ),
            "primary_arm": "target exposure",
            "primary_metric": (
                "Per-seed unconditional donor-gold hit rate after "
                "role patch minus same donor-gold hit after structured "
                "event control, averaged across all eight seeds"
            ),
            "positive_rule": (
                "Mean primary difference >0.15 and positive in >=6/8 "
                "seeds; all eight seeds included, no posthoc filtering"
            ),
            "secondary": (
                "Norm-random and shuffled-role excess, native both-correct "
                "conditional transfer, full-vocab switches, lexical "
                "corruption, familiar-arm results"
            ),
        },
        "scope": (
            "Fresh models, names and pairs; same synthetic grammar and "
            "architecture. Successful patching does NOT alone establish "
            "native necessity or mediation of exposure benefit."
        ),
        "guard": (
            "No protected target case generation before final gate and "
            "explicit unlock; refuse overwrite"
        ),
    }


def freeze(path, plan, create):
    wrapped = json.loads(
        json.dumps({
            **plan,
            "sha256": digest(plan),
        })
    )

    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(wrapped, indent=2) + "\n"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Run --phase plan first: {path}"
        )

    assert json.loads(path.read_text()) == wrapped, (
        "ML-085 code or frozen protocol changed"
    )

    return wrapped["sha256"]


def gate(root, h):
    rows = ml.all_metrics(
        root, h, 12800, complete=True
    )

    counts = {}

    for arm in ml.ARMS:
        counts[arm] = sum(
            rows[(seed, arm)]["calibration"]["choice_accuracy"] >= 0.95
            and
            rows[(seed, arm)]["calibration"]["correct_both_rate"] >= 0.90
            for seed in SEEDS
        )

    return (
        all(v >= 6 for v in counts.values()),
        counts,
    )


def native_pred(logits, candidates):
    scores = logits.gather(1, candidates)
    index = scores.argmax(1, keepdim=True)

    return candidates.gather(
        1, index
    ).flatten()


def norm_match(direction, target):
    scale = target.norm(
        dim=1, keepdim=True
    )

    norm = direction.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    return direction / norm * scale


def protected_sample(protected):
    """Never call this before final gate and unlock."""
    agents = [
        c for c in protected
        if c.role == "agent"
    ]

    assert len(agents) == 768

    agents.sort(
        key=lambda c: hashlib.sha256(
            " ".join(ml.prompt(c)).encode()
        ).hexdigest()
    )

    selected = agents[:N_PROBE_EVENTS]

    assert len({
        tuple(ml.prompt(c))
        for c in selected
    }) == N_PROBE_EVENTS

    recipients = []

    for c in selected:
        assert c.target_pos == 1
        assert c.target_voice == "passive"

        recipients.extend((
            c,
            replace(c, role="patient"),
        ))

    role_donors = [
        replace(
            c,
            role="patient"
            if c.role == "agent"
            else "agent",
        )
        for c in recipients
    ]

    event_donors = [
        m81.swap_event(c)
        for c in recipients
    ]

    lexical_donors = [
        m81.lexical_donor(c)
        for c in recipients
    ]

    for c, r, e, l in zip(
        recipients,
        role_donors,
        event_donors,
        lexical_donors,
    ):
        assert len(m81.changed_tokens(c, r)) == 1
        assert len(m81.changed_tokens(c, e)) == 1
        assert len(m81.changed_tokens(c, l)) == 1

        assert ml.answer(c) != ml.answer(r)
        assert ml.answer(c) != ml.answer(e)
        assert ml.answer(c) == ml.answer(l)

    return (
        selected,
        recipients,
        role_donors,
        event_donors,
        lexical_donors,
    )


def encode(batch, device):
    x, ro, gold, candidates = ml.encode(batch)

    return (
        x.to(device),
        ro.to(device),
        gold.to(device),
        candidates.to(device),
    )


@torch.inference_mode()
def causal_score(
    model,
    recipients,
    role_donors,
    event_donors,
    lexical_donors,
    device,
    seed,
    arm,
):
    model.eval()

    keys = (
        "events",
        "both_correct_events",
        "recipient_correct",
        "eligible_questions",
        "role_hit_all",
        "role_hit_eligible",
        "event_hit_all",
        "event_hit_eligible",
        "shuffled_role_hit_all",
        "shuffled_role_hit_eligible",
        "random_hit_all",
        "random_hit_eligible",
        "role_full_hit_all",
        "role_full_hit_eligible",
        "lexical_flip_all",
        "lexical_flip_eligible",
        "role_not_donor_or_gold_eligible",
        "norm_sum",
        "norm_count",
    )

    totals = {k: 0.0 for k in keys}

    assert len(recipients) % 2 == 0

    for start in range(
        0,
        len(recipients),
        CHUNK_EVENTS * 2,
    ):
        end = min(
            len(recipients),
            start + CHUNK_EVENTS * 2,
        )

        chunk = recipients[start:end]

        x, ro, gold, candidates = encode(
            chunk, device
        )

        native_logits, trace = m81.forward_trace(
            model, x, ro
        )

        n_pred = native_pred(
            native_logits, candidates
        )

        both_ok = (
            (n_pred == gold)
            .reshape(-1, 2)
            .all(dim=1)
        )

        eligible = both_ok.repeat_interleave(2)

        alts = [
            role_donors[start:end],
            event_donors[start:end],
            lexical_donors[start:end],
        ]

        donor_trace = []

        for batch in alts:
            dx, dr, _, _ = encode(batch, device)

            _, dt = m81.forward_trace(
                model, dx, dr
            )

            donor_trace.append(
                dt[SITE[0]][SITE[1]]
            )

        original = trace[SITE[0]][SITE[1]]

        role_delta = donor_trace[0] - original

        event_delta = norm_match(
            donor_trace[1] - original,
            role_delta,
        )

        alt_gold = encode(alts[0], device)[2]

        assert bool(torch.all(
            alt_gold != gold
        ))

        def score_patch(delta):
            patched, _ = m81.forward_trace(
                model,
                x,
                ro,
                patch=(
                    SITE[0],
                    SITE[1],
                    original + delta,
                ),
            )

            p = native_pred(
                patched, candidates
            )

            hit = (p == alt_gold).float()
            full = (
                patched.argmax(1) == alt_gold
            ).float()

            flip = (p != gold).float()

            return hit, full, flip, p

        hit, full_hit, _, preds = score_patch(
            role_delta
        )

        decoy, _, _, _ = score_patch(
            event_delta
        )

        # Same role direction, different sentence.
        shuffled = norm_match(
            role_delta.roll(shifts=2, dims=0),
            role_delta,
        )

        shuffled_hit, _, _, _ = score_patch(
            shuffled
        )

        lex_pred = score_patch(
            donor_trace[2] - original
        )[3]

        random_rates = []

        for rep in range(RANDOM_REPS):
            key = (
                seed * 1_000_000
                + (0 if arm == "familiar" else 1) * 100_000
                + start * 10
                + rep
            )

            rng = torch.Generator(
                device="cpu"
            ).manual_seed(key)

            noise = torch.randn(
                role_delta.shape,
                generator=rng,
                device="cpu",
            ).to(device)

            random_rates.append(
                score_patch(
                    norm_match(noise, role_delta)
                )[0]
            )

        random_hit = torch.stack(
            random_rates
        ).mean(dim=0)

        other_wrong = (
            (preds != alt_gold)
            & (preds != gold)
        ).float()

        totals["events"] += len(chunk) // 2

        totals["both_correct_events"] += float(
            both_ok.sum()
        )

        totals["recipient_correct"] += float(
            (n_pred == gold).float().sum()
        )

        totals["eligible_questions"] += float(
            eligible.sum()
        )

        measures = (
            ("role_hit", hit),
            ("event_hit", decoy),
            ("shuffled_role_hit", shuffled_hit),
            ("random_hit", random_hit),
            ("role_full_hit", full_hit),
        )

        for name, values in measures:
            totals[name + "_all"] += float(
                values.sum()
            )

            totals[name + "_eligible"] += float(
                values[eligible].sum()
            )

        lex_flip = (
            lex_pred != gold
        ).float()

        totals["lexical_flip_all"] += float(
            lex_flip.sum()
        )

        totals["lexical_flip_eligible"] += float(
            lex_flip[eligible].sum()
        )

        totals["role_not_donor_or_gold_eligible"] += float(
            other_wrong[eligible].sum()
        )

        totals["norm_sum"] += float(
            role_delta.norm(dim=1).sum()
        )

        totals["norm_count"] += len(chunk)

    n = 2 * totals["events"]
    eligible_n = totals["eligible_questions"]

    out = {
        "seed": seed,
        "arm": arm,
        "n_events": int(totals["events"]),
        "n_native_both_correct_events": int(
            totals["both_correct_events"]
        ),
        "native_both_correct_rate": (
            totals["both_correct_events"]
            / totals["events"]
        ),
        "native_question_accuracy": (
            totals["recipient_correct"] / n
        ),
        "n_eligible_questions": int(eligible_n),
        "mean_role_delta_norm": (
            totals["norm_sum"]
            / totals["norm_count"]
        ),
    }

    names = (
        "role_hit",
        "event_hit",
        "shuffled_role_hit",
        "random_hit",
        "role_full_hit",
        "lexical_flip",
    )

    for name in names:
        out[name + "_all"] = (
            totals[name + "_all"] / n
        )

        out[name + "_eligible"] = (
            totals[name + "_eligible"] / eligible_n
            if eligible_n
            else None
        )

    out["non_donor_other_eligible"] = (
        totals["role_not_donor_or_gold_eligible"]
        / eligible_n
        if eligible_n
        else None
    )

    out["primary_role_minus_event"] = (
        out["role_hit_all"]
        - out["event_hit_all"]
    )

    out["conditional_role_minus_event"] = (
        out["role_hit_eligible"]
        - out["event_hit_eligible"]
        if eligible_n
        else None
    )

    out["role_minus_shuffled_all"] = (
        out["role_hit_all"]
        - out["shuffled_role_hit_all"]
    )

    out["role_minus_random_all"] = (
        out["role_hit_all"]
        - out["random_hit_all"]
    )

    out["role_minus_random_eligible"] = (
        out["role_hit_eligible"]
        - out["random_hit_eligible"]
        if eligible_n
        else None
    )

    return out


def bootstrap(values):
    rng = random.Random(850853)

    draws = sorted(
        mean(
            values[rng.randrange(len(values))]
            for _ in values
        )
        for _ in range(20000)
    )

    return [draws[500], draws[19500]]


def evaluate(root, h, held, output, device, confirm):
    if not confirm:
        raise PermissionError(
            "Protected evaluation requires --confirm-protected"
        )

    if output.exists():
        raise FileExistsError(
            f"Refusing overwrite: {output}"
        )

    passed, counts = gate(root, h)

    if not passed:
        raise PermissionError(
            f"Frozen competence gate failed: {counts}"
        )

    # No protected questions exist before this point.
    protected = ml.protected_target_cases(held)

    selected, recipient, roles, events, lexical = (
        protected_sample(protected)
    )

    report = {
        "experiment": "ML-085",
        "protocol_sha256": h,
        "protected_evaluation_opened": True,
        "gate": counts,
        "probe_sample_sha256": digest(
            [ml.prompt(c) for c in selected]
        ),
        "runs": [],
    }

    for seed in SEEDS:
        for arm in ml.ARMS:
            path = (
                root
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                path,
                seed,
                arm,
                h,
                device,
            )

            overall = ml.measure(
                model, protected, device
            )

            causal = causal_score(
                model,
                recipient,
                roles,
                events,
                lexical,
                device,
                seed,
                arm,
            )

            report["runs"].append({
                "seed": seed,
                "arm": arm,
                "protected_behavior": overall,
                "causal": causal,
            })

            print(
                f"seed={seed} arm={arm:8s} "
                f"protected_acc={overall['choice_accuracy']:.3f} "
                f"both={causal['native_both_correct_rate']:.3f} "
                f"eligible_events={causal['n_native_both_correct_events']} "
                f"role-minus-event={causal['primary_role_minus_event']:+.3f}",
                flush=True,
            )

    lookup = {
        (r["seed"], r["arm"]): r
        for r in report["runs"]
    }

    gaps = [
        (
            lookup[(s, "target")]["protected_behavior"]["choice_accuracy"]
            -
            lookup[(s, "familiar")]["protected_behavior"]["choice_accuracy"]
        )
        for s in SEEDS
    ]

    prim = [
        lookup[(s, "target")]["causal"]["primary_role_minus_event"]
        for s in SEEDS
    ]

    min_events = min(
        lookup[(s, "target")]["causal"]["n_native_both_correct_events"]
        for s in SEEDS
    )

    evaluable = all(
        x is not None
        for x in prim
    )

    positive = sum(
        x > 0
        for x in prim
        if x is not None
    )

    report["summary"] = {
        "paired_behavioral_gap_mean": mean(gaps),
        "paired_behavioral_gap_ci": bootstrap(gaps),
        "positive_behavioral_pairs": sum(
            x > 0 for x in gaps
        ),
        "per_seed_behavioral_gaps": {
            str(s): x
            for s, x in zip(SEEDS, gaps)
        },
        "primary_per_seed": {
            str(s): x
            for s, x in zip(SEEDS, prim)
        },
        "minimum_target_native_both_correct_events": min_events,
        "all_questions_primary_unconditional": True,
        "primary_evaluable": evaluable,
        "primary_mean": (
            mean(prim) if evaluable else None
        ),
        "primary_ci": (
            bootstrap(prim) if evaluable else None
        ),
        "primary_positive_seeds": positive,
        "primary_success_rule_met": (
            evaluable
            and mean(prim) > 0.15
            and positive >= 6
        ),
        "interpretation": (
            "Unconditional primary plus conditional secondary, "
            "with structured donor controls; does not prove "
            "native necessity, grammar specificity or "
            "mediation of exposure effect."
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("PROTECTED SUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print("Saved", output)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan",
            "preflight",
            "train",
            "summarize",
            "evaluate",
        ),
    )

    p.add_argument(
        "--protocol",
        default="reports/ml085/frozen_protocol.json",
    )

    p.add_argument(
        "--root",
        default="results/ml085_fresh_models",
    )

    p.add_argument(
        "--output",
        default="reports/ml085/protected_causal_evaluation.json",
    )

    p.add_argument(
        "--until",
        type=int,
        choices=(3200, 12800),
        default=3200,
    )

    p.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
    )

    p.add_argument(
        "--arm",
        choices=ml.ARMS,
    )

    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=400,
    )

    p.add_argument(
        "--confirm-protected",
        action="store_true",
    )

    a = p.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    vocab, tr, ca, held, base, cal, dose, grid = setup()

    h = freeze(
        Path(a.protocol),
        manifest(vocab, tr, ca, held),
        a.phase == "plan",
    )

    if a.phase == "plan":
        print("ML-085 frozen", a.protocol, h[:12])
        return

    if a.phase == "preflight":
        torch.manual_seed(85085)

        model = ml.Model().eval()

        x, ro, _, choices = encode(
            grid["00"][:4], "cpu"
        )

        with torch.inference_mode():
            original, trace = m81.forward_trace(
                model, x, ro
            )

            parity = float(
                (original - model(x, ro))
                .abs().max()
            )

            assert parity < 1e-3

            same, _ = m81.forward_trace(
                model,
                x,
                ro,
                patch=(
                    SITE[0],
                    SITE[1],
                    trace[SITE[0]][SITE[1]],
                ),
            )

            assert float(
                (same - original)
                .abs().max()
            ) < 1e-3

            assert native_pred(
                original, choices
            ).shape == (4,)

        # Calibration-only smoke test of the full patch routine.
        demo = []

        for case in grid["10"][:4]:
            demo.extend((
                case,
                replace(case, role="patient"),
            ))

        r = [
            replace(
                c,
                role="patient"
                if c.role == "agent"
                else "agent",
            )
            for c in demo
        ]

        e = [
            m81.swap_event(c)
            for c in demo
        ]

        l = [
            m81.lexical_donor(c)
            for c in demo
        ]

        pilot = causal_score(
            model,
            demo,
            r,
            e,
            l,
            "cpu",
            SEEDS[0],
            "target",
        )

        assert pilot["n_events"] == 4
        assert 0 <= pilot["role_hit_all"] <= 1

        print(
            "PREFLIGHT PASS: 16 fresh names, "
            "72/24/24 disjoint pairs, 16 planned checkpoints, "
            f"L2 attention patch parity={parity:.2e}"
        )

        print(
            "Protected questions NOT generated; "
            "trained models NOT loaded."
        )
        return

    root = Path(a.root)

    if a.phase == "train":
        if a.save_every <= 0:
            raise ValueError(
                "--save-every must be positive"
            )

        device = ml.select_device(a.device)

        for seed in (
            SEEDS if a.seed is None
            else (a.seed,)
        ):
            for arm in (
                ml.ARMS if a.arm is None
                else (a.arm,)
            ):
                ml.train_one(
                    seed,
                    arm,
                    a.until,
                    a.save_every,
                    root,
                    h,
                    base,
                    cal,
                    dose,
                    device,
                )

        print(
            "Training completed to requested milestone; "
            "protected test remains sealed."
        )
        return

    if a.phase == "summarize":
        ml.summarize(root, h)
        return

    evaluate(
        root,
        h,
        held,
        Path(a.output),
        ml.select_device(a.device),
        a.confirm_protected,
    )


if __name__ == "__main__":
    main()
