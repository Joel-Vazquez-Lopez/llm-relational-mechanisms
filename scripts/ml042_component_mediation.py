
"""ML-042: Mechanistic accounting and component-level causal mediation.

Exploratory: ML-041 CALIBRATION pairs only. Never touches protected pairs.
Uses training-only donor directions for block-1 attention, MLP, and residual.
Performs norm-matched interventions and replay of the natural MLP output.
Positive control must reproduce ML-041's frozen calibration effect.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from ml037_scratch_transformer_pilot import encode_batch
from ml041_prospective_transfer import (
    SEEDS, SPLIT, DESIGN_HASH as ML041_HASH, RESULTS,
    quartets, load_model, donor_from_training, control_directions,
)

BUDGET = 0.20
CONTROLS = 4
SIGNS = (1., -1., -1., 1.)
MODES = ("residual", "attention", "mlp", "both", "attention_fixed_mlp")
SPEC = {
    "experiment": "ML-042",
    "source": "ML-041 frozen trained model checkpoints",
    "source_design_hash": ML041_HASH,
    "data": "calibration only; protected data forbidden",
    "block": 1,
    "budget_fraction_of_block1_residual": BUDGET,
    "modes": MODES,
    "matched_controls": CONTROLS,
    "primary_integrity_check": "reproduce ML-041 frozen calibration harm",
    "status": "exploratory mechanistic diagnosis, not prospective confirmation",
}
HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def factorial(x):
    return x[3] - x[2] - x[1] + x[0]


@torch.no_grad()
def capture(model, examples, device):
    ids, pos, labels, choices = encode_batch(examples)
    ids, pos, labels, choices = (
        x.to(device) for x in (ids, pos, labels, choices)
    )
    idx = torch.arange(4, device=device)
    captured = {}

    def att_capture(_module, _inputs, output):
        captured["a"] = output[0][idx, pos].detach().clone()

    def mlp_capture(_module, _inputs, output):
        captured["m"] = output[idx, pos].detach().clone()

    hooks = [
        model.blocks[0].attn.register_forward_hook(att_capture),
        model.blocks[0].mlp.register_forward_hook(mlp_capture),
    ]
    try:
        logits, states = model(ids, pos, return_states=True)
    finally:
        for h in hooks:
            h.remove()

    h0, h1 = states[0], states[1]
    a, m = captured["a"], captured["m"]
    reconstruction = (h1 - (h0 + a + m)).norm(dim=1).max()
    relative = float(reconstruction / h1.norm(dim=1).mean().clamp_min(1e-8))
    if relative > 0.0005:
        raise RuntimeError(f"Residual identity check failed: {relative:.3g}")

    delta0, delta_a, delta_m, delta1 = map(
        factorial, (h0, a, m, h1)
    )
    d_error = float(
        (delta1 - delta0 - delta_a - delta_m).norm()
        / delta1.norm().clamp_min(1e-6)
    )
    if d_error > 0.001:
        raise RuntimeError(f"Factorial identity check failed: {d_error:.3g}")

    # The <answer> token position depends on voice, not question role.
    if float(delta0.norm()) > 0.001:
        raise RuntimeError("Unexpected interaction in input readout")

    other = torch.where(
        choices[:, 0] == labels, choices[:, 1], choices[:, 0]
    )
    margin = (
        logits.gather(1, labels[:, None]).squeeze(1)
        - logits.gather(1, other[:, None]).squeeze(1)
    )
    return {
        "ids": ids, "pos": pos, "labels": labels, "other": other,
        "h0": h0, "h1": h1, "a": a, "m": m,
        "delta_a": delta_a, "delta_m": delta_m, "delta_h": delta1,
        "margin": margin, "identity_error": d_error,
    }


def donor_vectors(model, device, seed):
    vectors = {"attention": [], "mlp": [], "residual": []}
    norm_stats = {
        "a_over_h": [], "m_over_h": [],
        "cos_a_h": [], "cos_m_h": []
    }

    for _pair, _verb, _direction, examples in quartets(SPLIT["train"]):
        c = capture(model, examples, device)
        da, dm, dh = c["delta_a"], c["delta_m"], c["delta_h"]

        for name, v in (
            ("attention", da), ("mlp", dm), ("residual", dh)
        ):
            if float(v.norm()) < 1e-7:
                raise RuntimeError(f"Degenerate {name} training delta")
            vectors[name].append(v / v.norm())

        norm_stats["a_over_h"].append(float(da.norm() / dh.norm()))
        norm_stats["m_over_h"].append(float(dm.norm() / dh.norm()))
        norm_stats["cos_a_h"].append(float(
            torch.nn.functional.cosine_similarity(da[None], dh[None])
        ))
        norm_stats["cos_m_h"].append(float(
            torch.nn.functional.cosine_similarity(dm[None], dh[None])
        ))

    donors, coherence = {}, {}
    for name, vs in vectors.items():
        avg = torch.stack(vs).mean(dim=0)
        coh = float(avg.norm())
        if coh < 1e-6:
            raise RuntimeError(f"Undefined {name} donor")
        donors[name] = avg / avg.norm()
        coherence[name] = coh

    reference_donor, ref_coh = donor_from_training(model, device)
    if float(
        (reference_donor - donors["residual"]).abs().max()
    ) > 1e-4:
        raise RuntimeError("New residual donor disagrees with ML-041")
    if abs(ref_coh - coherence["residual"]) > 1e-4:
        raise RuntimeError("New donor coherence disagrees with ML-041")

    return donors, coherence, {
        k: mean(v) for k, v in norm_stats.items()
    }


@torch.no_grad()
def intervene(model, c, seed, donors, device):
    signs = torch.tensor(
        SIGNS, dtype=c["h1"].dtype, device=device
    )[:, None]
    hnorm = c["h1"].norm(dim=1).mean()
    shift = BUDGET * hnorm
    controls = {
        k: control_directions(v, seed)
        for k, v in donors.items()
    }

    variants = [("baseline", "self", -1)]
    for mode in MODES:
        variants.append((mode, "actual", -1))
        variants.extend(
            (mode, "control", j) for j in range(CONTROLS)
        )

    n = len(variants)
    da = torch.zeros(n, 4, c["h1"].shape[1], device=device)
    dm = torch.zeros_like(da)
    dh = torch.zeros_like(da)
    clamp_m = torch.zeros(n, dtype=torch.bool, device=device)

    for v, (mode, typ, j) in enumerate(variants):
        if mode == "baseline":
            continue
        if mode == "residual":
            category = "residual"
        elif mode in ("attention", "attention_fixed_mlp"):
            category = "attention"
        elif mode == "mlp":
            category = "mlp"
        else:
            category = "both"

        def direction(k):
            return (
                donors[k] if typ == "actual"
                else controls[k][j]
            )

        if category == "residual":
            dh[v] = -signs * shift * direction("residual")
        if category in ("attention", "both"):
            da[v] = -signs * shift * direction("attention")
        if category in ("mlp", "both"):
            dm[v] = -signs * shift * direction("mlp")
        if mode == "attention_fixed_mlp":
            da[v] = -signs * shift * direction("attention")
            clamp_m[v] = True

    ids = c["ids"].repeat(n, 1)
    pos = c["pos"].repeat(n)
    index = torch.arange(n * 4, device=device)
    da = da.reshape(n * 4, -1)
    dm = dm.reshape(n * 4, -1)
    dh = dh.reshape(n * 4, -1)
    clamp_m = clamp_m.repeat_interleave(4)
    natural_m = c["m"].repeat(n, 1)

    def att_hook(_module, _inputs, output):
        values, *rest = output
        updated = values.clone()
        updated[index, pos] = (
            updated[index, pos] + da.to(updated.dtype)
        )
        return (updated, *rest)

    def mlp_hook(_module, _inputs, output):
        updated = output.clone()
        original = updated[index, pos]
        updated[index, pos] = torch.where(
            clamp_m[:, None],
            natural_m.to(output.dtype),
            original + dm.to(output.dtype),
        )
        return updated

    def block_hook(_module, _inputs, output):
        updated = output.clone()
        updated[index, pos] = (
            updated[index, pos] + dh.to(output.dtype)
        )
        return updated

    hooks = [
        model.blocks[0].attn.register_forward_hook(att_hook),
        model.blocks[0].mlp.register_forward_hook(mlp_hook),
        model.blocks[0].register_forward_hook(block_hook),
    ]
    try:
        logits = model(ids, pos).reshape(n, 4, -1)
    finally:
        for h in hooks:
            h.remove()

    target = c["labels"][None, :, None].expand(n, -1, -1)
    wrong = c["other"][None, :, None].expand(n, -1, -1)
    margins = (
        logits.gather(2, target)
        - logits.gather(2, wrong)
    ).squeeze(-1)

    diff = float(
        (margins[0] - c["margin"]).abs().max()
    )
    if diff > 0.05:
        raise RuntimeError(f"Baseline patch mismatch: {diff:.4f}")

    def find(mode, typ, j=-1):
        return margins[variants.index((mode, typ, j))]

    original = find("baseline", "self")
    actual = {
        name: find(name, "actual")
        for name in MODES
    }

    row = {
        "original_accuracy": float(
            (original > 0).float().mean()
        ),
        "delta_norms": {
            "attention": float(c["delta_a"].norm()),
            "mlp": float(c["delta_m"].norm()),
            "residual": float(c["delta_h"].norm()),
        },
        "delta_a_h_cosine": float(
            torch.nn.functional.cosine_similarity(
                c["delta_a"][None], c["delta_h"][None]
            )
        ),
        "delta_m_h_cosine": float(
            torch.nn.functional.cosine_similarity(
                c["delta_m"][None], c["delta_h"][None]
            )
        ),
        "delta_identity_error": c["identity_error"],
        "modes": {},
        "nonadditivity": float(
            (
                actual["both"]
                - actual["attention"]
                - actual["mlp"]
                + original
            ).mean()
        ),
        "mlp_response_to_attention": float(
            (
                actual["attention"]
                - actual["attention_fixed_mlp"]
            ).mean()
        ),
    }

    for mode in MODES:
        control = torch.stack([
            find(mode, "control", j)
            for j in range(CONTROLS)
        ])
        candidate = actual[mode]
        row["modes"][mode] = {
            "harm": float(
                (control.mean(dim=0) - candidate).mean()
            ),
            "actual_minus_baseline": float(
                (candidate - original).mean()
            ),
            "actual_accuracy": float(
                (candidate > 0).float().mean()
            ),
            "control_accuracy": float(
                (control > 0).float().mean()
            ),
        }
    return row


def run(seed, phase, device):
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"ml042_component_mediation_seed{seed}.json"
    model, digest = load_model(seed, device)

    frozen_path = RESULTS / "ml041_frozen_predictions.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("design_hash") != ML041_HASH:
        raise RuntimeError("Wrong ML-041 prediction manifest")

    entry = frozen["seeds"][str(seed)]
    if digest != entry["checkpoint_sha256"]:
        raise RuntimeError(
            "Checkpoint differs from frozen ML-041 checkpoint"
        )

    if phase == "audit":
        _, _, _, q = quartets(SPLIT["train"])[0]
        c = capture(model, q, device)
        print(
            f"seed {seed} audit PASS, "
            f"delta identity error={c['identity_error']:.3g}"
        )
        return

    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")

    donors, coherence, donor_stats = donor_vectors(
        model, device, seed
    )
    rows = []

    for i, (pair, verb, direction, examples) in enumerate(
        quartets(SPLIT["calibration"]), 1
    ):
        c = capture(model, examples, device)
        r = intervene(model, c, seed, donors, device)
        r.update(
            pair=list(pair),
            verb=verb,
            direction=direction,
        )
        rows.append(r)
        if i % 8 == 0:
            print(
                f"seed {seed}: analyzed "
                f"{i}/56 calibration quartets"
            )

    if len(rows) != 56:
        raise RuntimeError("Expected 56 calibration quartets")

    summaries = {}
    for mode in MODES:
        summaries[mode] = {
            "harm": mean(
                r["modes"][mode]["harm"] for r in rows
            ),
            "actual_accuracy": mean(
                r["modes"][mode]["actual_accuracy"]
                for r in rows
            ),
            "control_accuracy": mean(
                r["modes"][mode]["control_accuracy"]
                for r in rows
            ),
            "actual_margin_change": mean(
                r["modes"][mode]["actual_minus_baseline"]
                for r in rows
            ),
        }

    expected = float(
        entry["calibration"]["mean_harm"]
    )
    observed = summaries["residual"]["harm"]

    if abs(expected - observed) > 0.035:
        raise RuntimeError(
            f"ML-041 residual positive control FAILED "
            f"expected {expected:+.4f} "
            f"got {observed:+.4f}"
        )

    result = {
        "experiment": "ML-042",
        "design_hash": HASH,
        "source_checkpoint_sha256": digest,
        "seed": seed,
        "eligible_ml041": entry["eligible"],
        "coherence": coherence,
        "donor_stats": donor_stats,
        "calibration_accuracy": mean(
            r["original_accuracy"] for r in rows
        ),
        "positive_control_ml041_expected": expected,
        "positive_control_observed": observed,
        "summaries": summaries,
        "mean_nonadditivity": mean(
            r["nonadditivity"] for r in rows
        ),
        "mean_mlp_response_to_attention": mean(
            r["mlp_response_to_attention"] for r in rows
        ),
        "records": rows,
        "complete": True,
    }
    output.write_text(json.dumps(result, indent=2))

    print(
        f"\n=== ML-042 seed {seed} | "
        f"{'ELIGIBLE' if entry['eligible'] else 'INELIGIBLE'} ==="
    )
    print(
        f"calibration accuracy "
        f"{result['calibration_accuracy']:.3f}"
        f" | donor coherence {coherence}"
    )
    print(
        f"ML-041 reproduction: {observed:+.3f} "
        f"vs frozen {expected:+.3f} PASS"
    )
    print(
        "mode                  harm     actual_acc  "
        "control_acc   margin_change"
    )

    for name, s in summaries.items():
        print(
            f"{name:<21} {s['harm']:+8.3f} "
            f"{s['actual_accuracy']:11.3f} "
            f"{s['control_accuracy']:12.3f} "
            f"{s['actual_margin_change']:+13.3f}"
        )

    print(
        "A x M intervention nonadditivity: "
        f"{result['mean_nonadditivity']:+.3f}"
    )
    print(
        "MLP response to attention intervention: "
        f"{result['mean_mlp_response_to_attention']:+.3f}"
    )
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=("audit", "collect"),
        required=True,
    )
    p.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        required=True,
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    device = (
        "mps"
        if args.device == "auto"
        and torch.backends.mps.is_available()
        else (
            "cpu" if args.device == "auto"
            else args.device
        )
    )
    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    print(
        f"ML-042 {args.phase} "
        f"seed={args.seed} device={device} "
        f"design={HASH}"
    )
    run(args.seed, args.phase, device)


if __name__ == "__main__":
    main()
