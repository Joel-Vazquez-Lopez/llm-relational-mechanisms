
"""ML-086: prospective calibration-derived role-subspace erasure.

Run from repository root:
  python3 scripts/ml086_role_subspace_necessity.py --phase plan
  python3 scripts/ml086_role_subspace_necessity.py --phase preflight

Freeze and commit before training:
  python3 scripts/ml086_role_subspace_necessity.py --phase train --until 3200
  python3 scripts/ml086_role_subspace_necessity.py --phase summarize
  python3 scripts/ml086_role_subspace_necessity.py --phase train --until 12800
  python3 scripts/ml086_role_subspace_necessity.py --phase summarize

Freeze calibration subspaces BEFORE protected evaluation:
  python3 scripts/ml086_role_subspace_necessity.py --phase calibrate

After committing calibration files:
  python3 scripts/ml086_role_subspace_necessity.py --phase evaluate --confirm-protected

No donor answer is supplied to the protected intervention.
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




NAMES = tuple(sorted((
    "Rhea", "Soren", "Talia", "Ulric",
    "Vesta", "Wren", "Xander", "Yara",
    "Zane", "Alden", "Brisa", "Corin",
    "Delia", "Evren", "Farah", "Gio",
)))


SEEDS = (601, 607, 613, 617, 619, 631, 641, 643)

MATCHING_SEED = 86086
N_EVENTS = 192
RANK = 8
SITE = (1, "attn")
RANDOM_REPS = 3
CHUNK = 48


def sha_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def setup():
    assert len(NAMES) == len(set(NAMES)) == 16
    assert not set(NAMES) & set(m84.NEW_NAMES)
    assert not set(NAMES) & set(ml.NAMES)

    m84.NEW_NAMES = NAMES
    m84.NEW_SEEDS = SEEDS
    m84.MATCHING_SEED = MATCHING_SEED

    result = m84.prepare()

    vocab, tr, ca, held, base, calibration, dose, grid = result

    assert len(base) == 20736
    assert len(calibration) == 6912
    assert len(dose["familiar"]) == 64
    assert len(dose["target"]) == 64

    return result


def protocol(vocab, tr, ca, held, grid):
    return {
        "experiment": "ML-086",
        "script_sha256": sha_file(__file__),
        "ml080_source_sha256": sha_file(ml.__file__),
        "ml081_source_sha256": sha_file(m81.__file__),
        "ml084_source_sha256": sha_file(m84.__file__),
        "names": NAMES,
        "seeds": SEEDS,
        "arms": ml.ARMS,
        "matching_seed": MATCHING_SEED,
        "vocabulary_sha256": digest(vocab),
        "splits": {
            "train": tr,
            "calibration": ca,
            "protected": held,
        },
        "training": (
            "Exact ML-080 architecture and optimization; "
            "56 base + 8 exposure questions per step; "
            "64 distinct exposure questions per arm; "
            "12800 steps"
        ),
        "competence_gate": (
            "At least 6/8 seeds per arm with "
            "calibration accuracy >= .95 and "
            "correct-both >= .90 at step 12800"
        ),
        "subspace": {
            "site": "L2 attention, final answer token",
            "rank": RANK,
            "calibration": (
                "64 familiar active/active calibration "
                "sentences, four event-role questions each"
            ),
            "sample_sha256": digest({
                k: [ml.prompt(c) for c in cases]
                for k, cases in grid.items()
            }),
            "role_basis": (
                "Top-8 right singular vectors of "
                "patient-minus-agent differences "
                "at both event positions"
            ),
            "event_basis": (
                "Top-8 right singular vectors of "
                "second-minus-first differences "
                "for both roles"
            ),
            "center": (
                "Mean of all four calibration "
                "corner activations"
            ),
            "intervention": (
                "h <- h - B B^T(h-center)"
            ),
            "event_control": (
                "Event-subspace erasure norm matched "
                "per question to role erasure"
            ),
            "random_control": (
                "Three independent random orthonormal "
                "rank-8 subspace erasures; individually "
                "norm matched to role erasure"
            ),
        },
        "protected": {
            "selection": (
                "192 SHA256-sorted second-passive "
                "agent event instances; both roles"
            ),
            "unfamiliar": (
                "384 second-passive questions"
            ),
            "matched_familiar": (
                "Same event semantics, participants, verbs, "
                "voice, frame and role; target event first"
            ),
            "behavior": (
                "All 1536 protected second-passive "
                "questions per model"
            ),
            "metrics": [
                "four-candidate accuracy",
                "full-vocabulary accuracy",
                "native prediction",
                "role-subspace erasure",
                "event-subspace erasure",
                "random-subspace erasure",
                "intervention norms",
            ],
        },
        "primary": {
            "per_seed": (
                "[(A_random - A_role) target-arm] - "
                "[(A_random - A_role) familiar-arm], "
                "unfamiliar protected probe questions"
            ),
            "purpose": (
                "Test whether role-subspace removal "
                "selectively reduces the target-exposure "
                "advantage beyond matched random removal"
            ),
            "success": (
                "Mean > +.05 accuracy units "
                "and positive in >=6/8 seeds"
            ),
            "uncertainty": (
                "Paired seed bootstrap 95% interval"
            ),
            "limitation": (
                "No unique-circuit, grammar-specificity, "
                "or causal-mediation identification"
            ),
        },
        "secondary": [
            "Target-arm role erasure versus random",
            "Role erasure versus event control",
            "Familiar construction",
            "Protected behavioral replication",
            "Full-vocabulary accuracy",
            "Per-seed results",
        ],
        "evaluation_rule": (
            "Protected cases generated only after "
            "final competence gate, frozen calibration "
            "bases, and explicit unlock; no overwrite"
        ),
    }


def frozen(path, doc, create):
    expected = json.loads(json.dumps({
        **doc,
        "sha256": digest(doc),
    }))

    if create and not path.exists():
        path.parent.mkdir(
            parents=True, exist_ok=True
        )
        path.write_text(
            json.dumps(expected, indent=2) + "\n"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Run --phase plan first: {path}"
        )

    assert json.loads(path.read_text()) == expected, (
        "ML-086 code or frozen protocol changed"
    )

    return expected["sha256"]


def gate(root, h):
    rows = ml.all_metrics(
        root, h, 12800, complete=True
    )

    counts = {}

    for arm in ml.ARMS:
        counts[arm] = sum(
            rows[(seed, arm)]["calibration"][
                "choice_accuracy"
            ] >= 0.95
            and rows[(seed, arm)]["calibration"][
                "correct_both_rate"
            ] >= 0.90
            for seed in SEEDS
        )

    return (
        all(n >= 6 for n in counts.values()),
        counts,
    )


def enc(cases, device):
    x, ro, gold, options = ml.encode(cases)

    return (
        x.to(device),
        ro.to(device),
        gold.to(device),
        options.to(device),
    )


def fit_basis(model, grid, device):
    trace = {}

    with torch.inference_mode():
        for corner in ("00", "01", "10", "11"):
            x, ro, _, _ = enc(
                grid[corner], device
            )

            _, t = m81.forward_trace(
                model, x, ro
            )

            trace[corner] = (
                t[SITE[0]][SITE[1]]
                .detach()
                .cpu()
                .double()
            )

    a, b, c, d = (
        trace[k]
        for k in ("00", "01", "10", "11")
    )

    role_data = torch.cat(
        (b - a, d - c), dim=0
    )

    event_data = torch.cat(
        (c - a, d - b), dim=0
    )

    center = torch.cat(
        (a, b, c, d), dim=0
    ).mean(0)

    assert role_data.shape == (
        128, ml.WIDTH
    )
    assert event_data.shape == role_data.shape

    def top_basis(matrix):
        _, singular, vh = torch.linalg.svd(
            matrix, full_matrices=False
        )

        basis = (
            vh[:RANK]
            .T.contiguous()
            .float()
        )

        assert torch.allclose(
            basis.T @ basis,
            torch.eye(RANK),
            atol=1e-4,
        )

        fraction = float(
            (singular[:RANK] ** 2).sum()
            / (singular ** 2).sum().clamp_min(1e-12)
        )

        return basis, fraction

    rb, rv = top_basis(role_data)
    eb, ev = top_basis(event_data)

    return {
        "role": rb,
        "event": eb,
        "center": center.float(),
        "role_variance_fraction": rv,
        "event_variance_fraction": ev,
    }


def validate_basis(basis):
    for key in ("role", "event"):
        b = basis[key]

        assert b.shape == (
            ml.WIDTH, RANK
        )

        assert bool(torch.isfinite(b).all())

        assert torch.allclose(
            b.T @ b,
            torch.eye(RANK),
            atol=1e-4,
        )

    assert basis["center"].shape == (
        ml.WIDTH,
    )
    assert bool(
        torch.isfinite(basis["center"]).all()
    )


def tensor_digest(bank):
    h = hashlib.sha256()

    for key in sorted(bank):
        h.update(key.encode())

        for name in ("center", "event", "role"):
            t = (
                bank[key][name]
                .detach()
                .cpu()
                .contiguous()
                .float()
            )

            h.update(name.encode())
            h.update(t.numpy().tobytes())

    return h.hexdigest()


def calibrate(
    root, h, grid, device,
    basis_path, manifest_path
):
    passed, counts = gate(root, h)

    if not passed:
        raise PermissionError(
            f"Competence gate FAILED: {counts}"
        )

    if basis_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Calibration bases already exist"
        )

    bank, metrics = {}, {}

    for seed in SEEDS:
        for arm in ml.ARMS:
            key = f"{seed}/{arm}"

            checkpoint = (
                root
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint, seed, arm, h, device
            )

            b = fit_basis(
                model, grid, device
            )

            validate_basis(b)

            bank[key] = {
                name: b[name]
                for name in (
                    "role", "event", "center"
                )
            }

            metrics[key] = {
                name: b[name]
                for name in (
                    "role_variance_fraction",
                    "event_variance_fraction",
                )
            }

            print(
                f"calibrate {key}: "
                f"role_var={b['role_variance_fraction']:.3f} "
                f"event_var={b['event_variance_fraction']:.3f}",
                flush=True,
            )

    basis_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save(bank, basis_path)

    doc = {
        "experiment": "ML-086",
        "protocol_sha256": h,
        "gate_counts": counts,
        "calibration_only": True,
        "basis_rank": RANK,
        "content_sha256": tensor_digest(bank),
        "bases_file_sha256": sha_file(basis_path),
        "per_model": metrics,
    }

    doc["sha256"] = digest(doc)

    manifest_path.write_text(
        json.dumps(doc, indent=2) + "\n"
    )

    print("FROZEN bases:", basis_path)
    print("FROZEN manifest:", manifest_path)
    print(
        "Commit BOTH files before "
        "protected evaluation."
    )


def load_bases(basis_path, manifest_path, h):
    doc = json.loads(
        manifest_path.read_text()
    )

    assert doc["sha256"] == digest({
        k: v
        for k, v in doc.items()
        if k != "sha256"
    })

    assert doc["protocol_sha256"] == h
    assert doc["basis_rank"] == RANK

    assert sha_file(basis_path) == doc[
        "bases_file_sha256"
    ]

    bank = torch.load(
        basis_path,
        map_location="cpu",
        weights_only=True,
    )

    assert tensor_digest(bank) == doc[
        "content_sha256"
    ]

    assert set(bank) == {
        f"{s}/{a}"
        for s in SEEDS
        for a in ml.ARMS
    }

    for basis in bank.values():
        validate_basis(basis)

    return bank, doc


def protected_cases(held):
    # Called only after all evaluation guards.
    target_all = ml.protected_target_cases(held)

    agents = [
        c for c in target_all
        if c.role == "agent"
    ]

    assert len(agents) == 768

    agents.sort(
        key=lambda c: hashlib.sha256(
            " ".join(ml.prompt(c)).encode()
        ).hexdigest()
    )

    selected = agents[:N_EVENTS]

    target = [
        item
        for c in selected
        for item in (
            c,
            replace(c, role="patient"),
        )
    ]

    familiar = [
        replace(c, target_pos=0)
        for c in target
    ]

    assert len(target) == 2 * N_EVENTS
    assert len(familiar) == len(target)

    assert all(
        ml.answer(t) == ml.answer(f)
        for t, f in zip(target, familiar)
    )

    assert all(
        (c.target_pos, c.target_voice)
        == (1, "passive")
        for c in target
    )

    assert all(
        (c.target_pos, c.target_voice)
        == (0, "passive")
        for c in familiar
    )

    return (
        target_all,
        target,
        familiar,
        digest([ml.prompt(c) for c in target]),
    )


def erase_delta(site_state, basis, center):
    return -(
        (site_state - center) @ basis
    ) @ basis.T


def match_norm(delta, reference):
    norm = delta.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)

    magnitude = reference.norm(
        dim=1, keepdim=True
    )

    return delta / norm * magnitude


def random_basis(seed, arm, construction, rep):
    code = (
        seed * 100000
        + (0 if arm == "familiar" else 1) * 10000
        + (0 if construction == "target" else 1) * 1000
        + rep
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(code)

    matrix = torch.randn(
        ml.WIDTH,
        RANK,
        generator=generator,
    )

    q, _ = torch.linalg.qr(
        matrix, mode="reduced"
    )

    return q


@torch.inference_mode()
def score_patches(
    model, cases, bank, device,
    seed, arm, construction
):
    rb = bank["role"].to(device)
    eb = bank["event"].to(device)
    center = bank["center"].to(device)

    random_bases = [
        random_basis(
            seed, arm, construction, rep
        ).to(device)
        for rep in range(RANDOM_REPS)
    ]

    keys = (
        "native", "role", "event", "random",
        "native_full", "role_full",
        "event_full", "random_full",
        "role_delta_norm", "event_delta_norm",
    )

    totals = {key: 0.0 for key in keys}

    for start in range(0, len(cases), CHUNK):
        x, ro, gold, options = enc(
            cases[start:start + CHUNK],
            device,
        )

        raw, trace = m81.forward_trace(
            model, x, ro
        )

        original = trace[SITE[0]][SITE[1]]

        role_delta = erase_delta(
            original, rb, center
        )

        event_delta = match_norm(
            erase_delta(
                original, eb, center
            ),
            role_delta,
        )

        def scores(logits):
            choice = options.gather(
                1,
                logits.gather(
                    1, options
                ).argmax(
                    1, keepdim=True
                ),
            ).flatten()

            return (
                (choice == gold).float(),
                (logits.argmax(1) == gold).float(),
            )

        n_choice, n_full = scores(raw)

        role_logits, _ = m81.forward_trace(
            model, x, ro,
            patch=(
                SITE[0],
                SITE[1],
                original + role_delta,
            ),
        )

        event_logits, _ = m81.forward_trace(
            model, x, ro,
            patch=(
                SITE[0],
                SITE[1],
                original + event_delta,
            ),
        )

        r_choice, r_full = scores(role_logits)
        e_choice, e_full = scores(event_logits)

        random_choice, random_full = [], []

        for basis in random_bases:
            delta = match_norm(
                erase_delta(
                    original, basis, center
                ),
                role_delta,
            )

            logits, _ = m81.forward_trace(
                model, x, ro,
                patch=(
                    SITE[0],
                    SITE[1],
                    original + delta,
                ),
            )

            c, f = scores(logits)

            random_choice.append(c)
            random_full.append(f)

        values = {
            "native": n_choice,
            "role": r_choice,
            "event": e_choice,
            "random": torch.stack(
                random_choice
            ).mean(0),
            "native_full": n_full,
            "role_full": r_full,
            "event_full": e_full,
            "random_full": torch.stack(
                random_full
            ).mean(0),
        }

        for key, value in values.items():
            totals[key] += float(value.sum())

        totals["role_delta_norm"] += float(
            role_delta.norm(dim=1).sum()
        )

        totals["event_delta_norm"] += float(
            event_delta.norm(dim=1).sum()
        )

    n = len(cases)

    result = {
        "n": n,
        **{
            key: value / n
            for key, value in totals.items()
        },
    }

    result["role_selective_damage"] = (
        result["random"] - result["role"]
    )

    result["role_vs_event_damage"] = (
        result["event"] - result["role"]
    )

    return result


def bootstrap(values):
    rng = random.Random(860863)

    dist = sorted(
        mean(
            values[rng.randrange(len(values))]
            for _ in values
        )
        for _ in range(20000)
    )

    return [dist[500], dist[19500]]


def evaluate(
    root, h, held, basis_path,
    manifest_path, output, device, unlock
):
    if not unlock:
        raise PermissionError(
            "Must supply --confirm-protected"
        )

    if output.exists():
        raise FileExistsError(
            f"Protected result already exists: {output}"
        )

    passed, counts = gate(root, h)

    if not passed:
        raise PermissionError(
            f"Competence gate FAILED: {counts}"
        )

    bank, cal_doc = load_bases(
        basis_path, manifest_path, h
    )

    target_all, target, familiar, sample_hash = (
        protected_cases(held)
    )

    report = {
        "experiment": "ML-086",
        "protocol_sha256": h,
        "calibration_manifest_sha256":
            cal_doc["sha256"],
        "protected_evaluation_opened": True,
        "gate_counts": counts,
        "target_probe_sha256": sample_hash,
        "runs": [],
    }

    for seed in SEEDS:
        for arm in ml.ARMS:
            checkpoint = (
                root
                / f"seed_{seed}"
                / arm
                / "model_step_12800.pt"
            )

            model = ml.load_checkpoint(
                checkpoint, seed, arm, h, device
            )

            basis = bank[f"{seed}/{arm}"]

            full_behavior = ml.measure(
                model, target_all, device
            )

            scores = {
                "target": score_patches(
                    model, target, basis, device,
                    seed, arm, "target",
                ),
                "familiar": score_patches(
                    model, familiar, basis, device,
                    seed, arm, "familiar",
                ),
            }

            report["runs"].append({
                "seed": seed,
                "arm": arm,
                "full_protected_target": full_behavior,
                "patches": scores,
            })

            print(
                f"seed={seed} arm={arm:8s} "
                f"target_native={scores['target']['native']:.3f} "
                f"target_role={scores['target']['role']:.3f} "
                f"target_random={scores['target']['random']:.3f} "
                f"fam_native={scores['familiar']['native']:.3f}",
                flush=True,
            )

    lookup = {
        (r["seed"], r["arm"]): r
        for r in report["runs"]
    }

    differences = []
    behavioral = []

    for seed in SEEDS:
        t = lookup[
            seed, "target"
        ]["patches"]["target"]

        f = lookup[
            seed, "familiar"
        ]["patches"]["target"]

        differences.append(
            t["role_selective_damage"]
            - f["role_selective_damage"]
        )

        behavioral.append(
            lookup[seed, "target"][
                "full_protected_target"
            ]["choice_accuracy"]
            -
            lookup[seed, "familiar"][
                "full_protected_target"
            ]["choice_accuracy"]
        )

    summary = {
        "primary_per_seed": {
            str(seed): diff
            for seed, diff in zip(
                SEEDS, differences
            )
        },
        "primary_mean": mean(differences),
        "primary_ci": bootstrap(differences),
        "primary_positive_seeds": sum(
            x > 0 for x in differences
        ),
        "primary_rule_met": (
            mean(differences) > 0.05
            and sum(
                x > 0 for x in differences
            ) >= 6
        ),
        "behavioral_per_seed": {
            str(seed): diff
            for seed, diff in zip(
                SEEDS, behavioral
            )
        },
        "behavioral_mean": mean(behavioral),
        "behavioral_ci": bootstrap(behavioral),
        "behavioral_positive_pairs": sum(
            x > 0 for x in behavioral
        ),
        "interpretation": (
            "Role-subspace erasure tests dependence on a "
            "calibration-derived activation subspace. "
            "Does not establish a unique native grammatical "
            "circuit or complete mediation."
        ),
    }

    report["summary"] = summary

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("PROTECTED SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        required=True,
        choices=(
            "plan", "preflight",
            "train", "summarize",
            "calibrate", "evaluate",
        ),
    )

    ap.add_argument(
        "--protocol",
        default="reports/ml086/frozen_protocol.json",
    )

    ap.add_argument(
        "--root",
        default="results/ml086_fresh_models",
    )

    ap.add_argument(
        "--bases",
        default="reports/ml086/calibration_bases.pt",
    )

    ap.add_argument(
        "--calibration-manifest",
        default="reports/ml086/calibration_manifest.json",
    )

    ap.add_argument(
        "--output",
        default="reports/ml086/protected_evaluation.json",
    )

    ap.add_argument(
        "--until",
        choices=(3200, 12800),
        type=int,
        default=3200,
    )

    ap.add_argument(
        "--seed",
        choices=SEEDS,
        type=int,
    )

    ap.add_argument(
        "--arm",
        choices=ml.ARMS,
    )

    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
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
        vocab, tr, ca, held,
        base, calibration, dose, grid,
    ) = setup()

    h = frozen(
        Path(args.protocol),
        protocol(
            vocab, tr, ca, held, grid
        ),
        args.phase == "plan",
    )

    if args.phase == "plan":
        print(
            f"ML-086 frozen: {args.protocol} "
            f"sha256={h[:12]}"
        )
        return

    if args.phase == "preflight":
        torch.manual_seed(86086)

        model = ml.Model().eval()

        basis = fit_basis(
            model, grid, "cpu"
        )

        validate_basis(basis)

        x, ro, _, _ = enc(
            grid["00"][:4], "cpu"
        )

        with torch.inference_mode():
            original, trace = m81.forward_trace(
                model, x, ro
            )

            error = float(
                (
                    original
                    - model(x, ro)
                ).abs().max()
            )

            assert error < 1e-3

            state = trace[SITE[0]][SITE[1]]

            unchanged, _ = m81.forward_trace(
                model, x, ro,
                patch=(
                    SITE[0],
                    SITE[1],
                    state,
                ),
            )

            assert float(
                (
                    unchanged
                    - original
                ).abs().max()
            ) < 1e-3

            delta = erase_delta(
                state,
                basis["role"],
                basis["center"],
            )

            modified, _ = m81.forward_trace(
                model, x, ro,
                patch=(
                    SITE[0],
                    SITE[1],
                    state + delta,
                ),
            )

            assert bool(
                torch.isfinite(modified).all()
            )

            test = score_patches(
                model,
                grid["00"][:4],
                basis,
                "cpu",
                SEEDS[0],
                "target",
                "familiar",
            )

            assert test["n"] == 4

        print(
            "PREFLIGHT PASS: 16 fresh names, "
            "72/24/24 pair splits, "
            f"rank-{RANK} calibration subspaces, "
            f"forward parity={error:.2e}"
        )

        print(
            "Protected cases NOT generated. "
            "Trained models NOT accessed."
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
            SEEDS
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
            "Training to requested milestone completed. "
            "Protected cases remain sealed."
        )
        return

    if args.phase == "summarize":
        ml.summarize(root, h)
        return

    if args.phase == "calibrate":
        calibrate(
            root, h, grid,
            ml.select_device(args.device),
            Path(args.bases),
            Path(args.calibration_manifest),
        )
        return

    evaluate(
        root,
        h,
        held,
        Path(args.bases),
        Path(args.calibration_manifest),
        Path(args.output),
        ml.select_device(args.device),
        args.confirm_protected,
    )


if __name__ == "__main__":
    main()
