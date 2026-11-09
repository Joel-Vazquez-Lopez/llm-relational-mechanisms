"""ML-087: frozen exploratory calibration-only factorial ablation.

Reuses ML-086's 16 saved final checkpoints. No new training.
Uses 64 previously unused calibration events to fit and 64 separate
calibration events for evaluation. Never creates protected cases.

python3 scripts/ml087_factorial_necessity.py --phase plan
python3 scripts/ml087_factorial_necessity.py --phase preflight
# Commit script and reports/ml087/frozen_protocol.json BEFORE evaluation.
python3 scripts/ml087_factorial_necessity.py --phase evaluate
# Commit reports/ml087/calibration_evaluation.json after evaluation.
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
import ml086_role_subspace_necessity as m86

FIT_SLICE = (64, 128)       # ML-086 used first 64; exclude those
EVAL_SLICE = (128, 192)    # unseen by ML-086 geometry/ablation fit
RANK = 8
SITE = (1, "attn")
RANDOM_REPS = 3
CHUNK = 64
ROOT = Path("results/ml086_fresh_models")
PROTOCOL = Path("reports/ml087/frozen_protocol.json")
OUTPUT = Path("reports/ml087/calibration_evaluation.json")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def sample_grid(cal, start, end):
    agents = [
        c for c in cal
        if c.target_pos == 1
        and c.target_voice == "active"
        and c.distractor_voice == "active"
        and c.role == "agent"
    ]
    assert len(agents) == 576
    agents.sort(key=lambda c: hashlib.sha256(
        " ".join(ml.prompt(c)).encode()
    ).hexdigest())
    selected = agents[start:end]
    assert len(selected) == 64

    grid = {k: [] for k in ("00", "01", "10", "11")}
    for second_agent in selected:
        first_agent = m81.swap_event(second_agent)
        first_patient = replace(first_agent, role="patient")
        second_patient = replace(second_agent, role="patient")
        assert m81.changed_tokens(first_agent, second_agent) == [
            ("first", "second")
        ]
        assert m81.changed_tokens(first_agent, first_patient) == [
            ("agent", "patient")
        ]
        for k, c in zip(grid, (
            first_agent, first_patient, second_agent, second_patient
        )):
            grid[k].append(c)
    assert all(len(x) == 64 for x in grid.values())
    return grid


def grid_prompts(grid):
    return [
        ml.prompt(grid[key][i])
        for i in range(64)
        for key in ("00", "01", "10", "11")
    ]


def prepare():
    # Strictly reconstruct frozen ML-086 data and protocol.
    vocab, tr, ca, held, base, cal, dose, original_grid = m86.setup()
    old_hash = m86.frozen(
        Path("reports/ml086/frozen_protocol.json"),
        m86.protocol(vocab, tr, ca, held, original_grid),
        False,
    )
    fit = sample_grid(cal, *FIT_SLICE)
    evaluation = sample_grid(cal, *EVAL_SLICE)
    old = set(map(tuple, grid_prompts(original_grid)))
    fitted = set(map(tuple, grid_prompts(fit)))
    evaluated = set(map(tuple, grid_prompts(evaluation)))
    assert len(old) == len(fitted) == len(evaluated) == 256
    assert not (old & fitted or old & evaluated or fitted & evaluated)
    return old_hash, fit, evaluation


def spec(old_hash, fit, evaluation):
    return {
        "experiment": "ML-087",
        "status": "EXPLORATORY_SAME_ML086_MODELS_CALIBRATION_ONLY",
        "script_sha256": sha_file(__file__),
        "ml080_source_sha256": sha_file(ml.__file__),
        "ml081_source_sha256": sha_file(m81.__file__),
        "ml086_source_sha256": sha_file(m86.__file__),
        "ml086_protocol_sha256": old_hash,
        "checkpoint_step": 12800,
        "seeds": list(m86.SEEDS),
        "arms": list(ml.ARMS),
        "site": "L2 final answer token attention output",
        "rank": RANK,
        "fit_events": 64,
        "evaluation_events": 64,
        "four_queries_per_event": True,
        "fit_slice_in_sorted_576_calibration_events": list(FIT_SLICE),
        "evaluation_slice_in_sorted_576_calibration_events": list(EVAL_SLICE),
        "fit_prompt_sha256": digest(grid_prompts(fit)),
        "evaluation_prompt_sha256": digest(grid_prompts(evaluation)),
        "basis": {
            "main": "SVD top-8 of stacked average role and event effects",
            "gamma": "SVD top-8 of factorial interaction",
            "gamma_unique": "SVD top-8 of gamma after projection out of pooled main-8",
            "center": "mean of all four fit-corner attention outputs",
        },
        "interventions": {
            "primary": "exact erasure of gamma_unique projected component",
            "random": "three random rank-8 complement-of-main erasure directions, each norm-matched to gamma_unique erasure",
            "secondary_main": "pooled main-8 direction, norm-matched to gamma_unique erasure",
            "secondary_gamma": "raw gamma-8 direction, norm-matched to gamma_unique erasure",
        },
        "primary": {
            "population": "eight target-exposure ML-086 models",
            "unit": "seed (all 256 evaluation questions per seed)",
            "statistic": "mean(random-complement accuracy - gamma_unique accuracy)",
            "success": "mean>0.05 and >=6/8 positive and paired-seed bootstrap 95% lower>0",
            "comparison": "same per-example perturbation L2 norms",
        },
        "secondary": [
            "familiar arm primary contrast",
            "pooled main vs gamma and unique matched perturbations",
            "full vocabulary correctness",
            "gamma residual energy after removing main subspace",
        ],
        "limitations": (
            "Reuses same models and grammar family as ML-086; calibration "
            "examples may be exposed to model selection and were not "
            "a separately protected set. Purely exploratory and NOT a "
            "fresh-cohort validation of the unfamiliar-grammar mechanism."
        ),
        "safety": "No protected cases generated and no protected evaluation run",
    }


def freeze(doc, create):
    wrapped = {**doc, "sha256": digest(doc)}
    if create and not PROTOCOL.exists():
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL.write_text(json.dumps(wrapped, indent=2) + "\n")
    if not PROTOCOL.exists():
        raise FileNotFoundError("Run --phase plan first")
    assert json.loads(PROTOCOL.read_text()) == wrapped, (
        "Frozen protocol or script mismatch; stop."
    )
    return wrapped["sha256"]


def basis_top8(matrix):
    _, _, vt = torch.linalg.svd(
        matrix.double(), full_matrices=False
    )
    b = vt[:RANK].T.contiguous().float()
    assert bool(torch.allclose(
        b.T @ b, torch.eye(RANK), atol=1e-4
    ))
    return b


@torch.inference_mode()
def fit_basis(model, grid):
    state = {}
    for k, cases in grid.items():
        x, ro, _, _ = m86.enc(cases, "cpu")
        _, trace = m81.forward_trace(model, x, ro)
        state[k] = trace[SITE[0]][SITE[1]].cpu().double()
    a, b, c, d = [state[k] for k in ("00", "01", "10", "11")]
    role_main = ((b - a) + (d - c)) / 2
    event_main = ((c - a) + (d - b)) / 2
    gamma = d - c - b + a
    main = basis_top8(torch.cat([role_main, event_main]))
    raw_gamma = basis_top8(gamma)
    residual = gamma - (gamma @ main.double()) @ main.double().T
    unique = basis_top8(residual)
    # Numerical orthogonality, without using protected data.
    assert (main.T @ unique).abs().max() < 1e-4
    fraction = float(
        residual.square().sum() / gamma.square().sum().clamp_min(1e-12)
    )
    center = torch.cat([a, b, c, d]).mean(0).float()
    return {
        "main": main,
        "gamma": raw_gamma,
        "unique": unique,
        "center": center,
        "gamma_residual_energy_fraction": fraction,
    }


def delta(state, basis, center):
    return -((state - center) @ basis) @ basis.T


def match_norm(other, reference):
    return other / other.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12) * reference.norm(dim=1, keepdim=True)


def random_complement(main, seed, arm, rep):
    code = 870870 + 100 * seed + 10 * (arm == "target") + rep
    g = torch.Generator().manual_seed(code)
    x = torch.randn(ml.WIDTH, RANK, generator=g)
    x -= main @ (main.T @ x)
    q, _ = torch.linalg.qr(x, mode="reduced")
    assert (q.T @ main).abs().max() < 1e-4
    return q


@torch.inference_mode()
def score(model, cases, fitted, seed, arm):
    main, gamma, unique, center = [
        fitted[k] for k in ("main", "gamma", "unique", "center")
    ]
    controls = [
        random_complement(main, seed, arm, rep)
        for rep in range(RANDOM_REPS)
    ]
    keys = ("native", "unique", "random", "main_matched", "gamma_matched")
    sums = {k: 0.0 for k in keys}
    full = {k: 0.0 for k in keys}
    norm_sum = 0.0

    for start in range(0, len(cases), CHUNK):
        x, ro, gold, choices = m86.enc(cases[start:start + CHUNK], "cpu")
        logits, trace = m81.forward_trace(model, x, ro)
        original = trace[SITE[0]][SITE[1]]
        ud = delta(original, unique, center)
        norm_sum += float(ud.norm(dim=1).sum())

        def record(name, scores):
            choice_ids = choices.gather(
                1, scores.gather(1, choices).argmax(1, keepdim=True)
            ).flatten()
            sums[name] += float((choice_ids == gold).float().sum())
            full[name] += float((scores.argmax(1) == gold).float().sum())

        def patched(d):
            out, _ = m81.forward_trace(
                model, x, ro,
                patch=(SITE[0], SITE[1], original + d),
            )
            return out

        record("native", logits)
        record("unique", patched(ud))
        record("main_matched", patched(match_norm(
            delta(original, main, center), ud
        )))
        record("gamma_matched", patched(match_norm(
            delta(original, gamma, center), ud
        )))

        random_acc, random_full = [], []
        for rb in controls:
            out = patched(match_norm(delta(original, rb, center), ud))
            choice_ids = choices.gather(
                1, out.gather(1, choices).argmax(1, keepdim=True)
            ).flatten()
            random_acc.append(float((choice_ids == gold).float().sum()))
            random_full.append(float((out.argmax(1) == gold).float().sum()))
        sums["random"] += mean(random_acc)
        full["random"] += mean(random_full)

    n = len(cases)
    assert n == 256
    result = {k: sums[k] / n for k in keys}
    result.update({f"{k}_full": full[k] / n for k in keys})
    result["unique_delta_mean_norm"] = norm_sum / n
    result["random_minus_unique"] = result["random"] - result["unique"]
    return result


def bootstrap(values):
    rng = random.Random(870871)
    draws = sorted(mean(
        values[rng.randrange(len(values))] for _ in values
    ) for _ in range(20000))
    return [draws[500], draws[19500]]


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    old_hash, fit, evaluation = prepare()
    protocol_hash = freeze(spec(old_hash, fit, evaluation), False)
    cases = grid_prompts(evaluation)  # only used to validate selection
    assert len(cases) == 256
    evaluation_cases = [
        evaluation[k][i]
        for i in range(64)
        for k in ("00", "01", "10", "11")
    ]
    report = {
        "experiment": "ML-087",
        "status": "EXPLORATORY_CALIBRATION_ONLY",
        "protocol_sha256": protocol_hash,
        "ml086_protocol_sha256": old_hash,
        "protected_cases_generated": False,
        "runs": [],
    }
    for seed in m86.SEEDS:
        for arm in ml.ARMS:
            checkpoint = (ROOT / f"seed_{seed}" / arm / "model_step_12800.pt")
            model = ml.load_checkpoint(checkpoint, seed, arm, old_hash, "cpu")
            fitted = fit_basis(model, fit)
            metrics = score(model, evaluation_cases, fitted, seed, arm)
            report["runs"].append({
                "seed": seed, "arm": arm,
                "gamma_residual_energy_fraction": fitted["gamma_residual_energy_fraction"],
                "metrics": metrics,
            })
            print(
                f"seed={seed} arm={arm:<8} "
                f"native={metrics['native']:.3f} "
                f"unique={metrics['unique']:.3f} "
                f"random={metrics['random']:.3f} "
                f"excess={metrics['random_minus_unique']:+.3f} "
                f"gamma_residual={fitted['gamma_residual_energy_fraction']:.3f}",
                flush=True,
            )
    target = [r for r in report["runs"] if r["arm"] == "target"]
    familiar = [r for r in report["runs"] if r["arm"] == "familiar"]
    prim = [r["metrics"]["random_minus_unique"] for r in target]
    ci = bootstrap(prim)
    report["summary"] = {
        "primary_per_seed": {str(r["seed"]): r["metrics"]["random_minus_unique"] for r in target},
        "primary_mean": mean(prim),
        "primary_ci": ci,
        "primary_positive_seeds": sum(v > 0 for v in prim),
        "primary_rule_met": mean(prim) > 0.05 and sum(v > 0 for v in prim) >= 6 and ci[0] > 0,
        "familiar_arm_mean_excess": mean(r["metrics"]["random_minus_unique"] for r in familiar),
        "target_mean_gamma_residual_energy_fraction": mean(r["gamma_residual_energy_fraction"] for r in target),
        "familiar_mean_gamma_residual_energy_fraction": mean(r["gamma_residual_energy_fraction"] for r in familiar),
        "limitation": "Exploratory, same ML-086 models, familiar calibration only; not generalization proof",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("\nCALIBRATION-ONLY SUMMARY")
    print(json.dumps(report["summary"], indent=2))
    print("Saved", OUTPUT)
    print("No protected evaluation executed.")


def preflight(fit, evaluation):
    torch.manual_seed(87087)
    model = ml.Model().eval()
    fitted = fit_basis(model, fit)
    basis = fitted["unique"]
    main = fitted["main"]
    assert basis.shape == main.shape == (96, 8)
    cases = [evaluation[k][i] for i in range(2) for k in ("00", "01", "10", "11")]
    x, ro, _, _ = m86.enc(cases, "cpu")
    with torch.inference_mode():
        logits, trace = m81.forward_trace(model, x, ro)
        assert float((logits - model(x, ro)).abs().max()) < 1e-3
        s = trace[SITE[0]][SITE[1]]
        unchanged, _ = m81.forward_trace(model, x, ro, patch=(1, "attn", s))
        assert float((logits - unchanged).abs().max()) < 1e-3
        d = delta(s, basis, fitted["center"])
        changed, _ = m81.forward_trace(model, x, ro, patch=(1, "attn", s + d))
        assert bool(torch.isfinite(changed).all())
    print("PREFLIGHT PASS: disjoint fit/evaluation grids; "
          "rank-8 orthogonal interaction residual; patch parity verified")
    print("No trained checkpoint or protected example accessed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("plan", "preflight", "evaluate"), required=True)
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    old_hash, fit, evaluation = prepare()
    h = freeze(spec(old_hash, fit, evaluation), args.phase == "plan")
    if args.phase == "plan":
        print("ML-087 protocol frozen", PROTOCOL, h[:12])
    elif args.phase == "preflight":
        preflight(fit, evaluation)
    else:
        run()


if __name__ == "__main__":
    main()
