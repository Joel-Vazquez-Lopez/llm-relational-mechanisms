
"""ML-040: magnitude-matched factorial perturbations.

Uses ML-037 checkpoints and ML-039 held-out quartets.
Compares own-interaction, sign-flipped and other-pair directions
at fixed perturbation magnitudes relative to residual-state norm.

Exploratory; does not establish distinct algorithms or circuits.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from ml037_scratch_transformer_pilot import encode_batch
from ml039_factorial_residualization import (
    quartets, load_trained, signflip
)

BUDGETS = (0.05, 0.10, 0.20, 0.40)
N_CONTROLS = 4
SIGNS = (1.0, -1.0, -1.0, 1.0)

SPEC = {
    "experiment": "ML-040",
    "seeds": (7, 17),
    "block": 1,
    "budgets_state_fraction": BUDGETS,
    "controls_per_type": N_CONTROLS,
    "control_types": (
        "signflip", "other_pair_interaction"
    ),
    "patch": (
        "h_i - s_i * budget * mean(||h_j||) "
        "* unit_direction"
    ),
    "status": "exploratory; reused held-out examples",
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


@torch.no_grad()
def collect_states(model, cases, device):
    data = []

    for base, examples in cases:
        ids, pos, labels, choices = encode_batch(examples)

        ids, pos = ids.to(device), pos.to(device)
        labels = labels.to(device)
        choices = choices.to(device)

        logits, states = model(
            ids, pos, return_states=True
        )

        h = states[1]

        delta = h[3] - h[2] - h[1] + h[0]

        if float(delta.norm()) <= 1e-9:
            raise RuntimeError(
                f"Degenerate interaction: {base}"
            )

        other = torch.where(
            choices[:, 0] == labels,
            choices[:, 1],
            choices[:, 0],
        )

        margin = (
            logits.gather(
                1, labels[:, None]
            ).squeeze(1)
            - logits.gather(
                1, other[:, None]
            ).squeeze(1)
        )

        data.append({
            "base": base,
            "ids": ids,
            "pos": pos,
            "labels": labels,
            "other": other,
            "h": h,
            "delta": delta,
            "mean_h_norm": float(
                h.norm(dim=1).mean()
            ),
            "original_margin": margin,
        })

    return data


@torch.no_grad()
def evaluate_one(model, data, seed, index, device):
    row = data[index]

    h = row["h"]
    delta = row["delta"]
    base = row["base"]
    hnorm = row["mean_h_norm"]

    unit = delta / delta.norm()

    signs = torch.tensor(
        SIGNS, device=device, dtype=h.dtype
    ).unsqueeze(1)

    # Seven sorted groups of eight quartets.
    # Choose donors from four different name pairs.
    donor_indices = [
        (
            (index // 8 + offset) % 7
        ) * 8 + (index % 8)
        for offset in (1, 2, 3, 4)
    ]

    assert all(
        data[j]["base"][0] != base[0]
        for j in donor_indices
    )

    donor_units = [
        data[j]["delta"] / data[j]["delta"].norm()
        for j in donor_indices
    ]

    sign_units = [
        unit * signflip(
            seed, base, 0, j,
            unit.numel(), device, unit.dtype
        )
        for j in range(N_CONTROLS)
    ]

    replacements = [h]
    keys = [("self", 0.0, -1)]

    for budget in BUDGETS:
        shift = budget * hnorm

        for kind, directions in (
            ("actual", [unit]),
            ("signflip", sign_units),
            ("other_pair", donor_units),
        ):
            for j, direction in enumerate(directions):
                replacements.append(
                    h - signs * shift * direction
                )
                keys.append((kind, budget, j))

    n = len(replacements)

    vectors = torch.stack(
        replacements
    ).reshape(
        4 * n, h.shape[-1]
    )

    big_ids = row["ids"].repeat(n, 1)
    big_pos = row["pos"].repeat(n)

    def hook(_module, _inputs, output):
        out = output.clone()

        batch_indices = torch.arange(
            out.shape[0], device=device
        )

        out[
            batch_indices, big_pos
        ] = vectors.to(out.dtype)

        return out

    handle = (
        model.blocks[0]
        .register_forward_hook(hook)
    )

    try:
        logits = model(
            big_ids, big_pos
        ).reshape(n, 4, -1)
    finally:
        handle.remove()

    target = row["labels"][
        None, :, None
    ].expand(n, -1, 1)

    other = row["other"][
        None, :, None
    ].expand(n, -1, 1)

    margins = (
        logits.gather(2, target)
        - logits.gather(2, other)
    ).squeeze(-1)

    baseline = row["original_margin"]

    discrepancy = float(
        (margins[0] - baseline).abs().max()
    )

    if discrepancy > 0.05:
        raise RuntimeError(
            f"Self-patch mismatch: "
            f"{base}, {discrepancy}"
        )

    output = []

    for budget in BUDGETS:
        def get(kind):
            indices = [
                k for k, key in enumerate(keys)
                if key[0] == kind
                and key[1] == budget
            ]

            return margins[
                indices
            ].cpu().tolist()

        output.append({
            "base": [
                list(base[0]),
                base[1],
                base[2],
            ],
            "budget": budget,
            "equivalent_alpha": float(
                4 * budget * hnorm / delta.norm()
            ),
            "natural_per_cell_fraction": float(
                delta.norm() / (4 * hnorm)
            ),
            "other_pair_cosine": [
                float((unit * u).sum())
                for u in donor_units
            ],
            "baseline": baseline.cpu().tolist(),
            "actual": get("actual")[0],
            "signflip": get("signflip"),
            "other_pair": get("other_pair"),
        })

    return output


def describe(records, seed):
    print(
        f"\n=== ML-040 SEED {seed} "
        "| BLOCK 1 ==="
    )
    print(
        "Each budget: perturbation norm / "
        "mean quartet state norm"
    )

    for budget in BUDGETS:
        rr = [
            r for r in records
            if r["budget"] == budget
        ]

        if len(rr) != 56 or any(
            len(r["signflip"]) != 4
            or len(r["other_pair"]) != 4
            for r in rr
        ):
            raise RuntimeError(
                "Incomplete results"
            )

        baseline = [
            x for r in rr
            for x in r["baseline"]
        ]

        actual = [
            x for r in rr
            for x in r["actual"]
        ]

        sf = [
            [
                x for r in rr
                for x in r["signflip"][j]
            ]
            for j in range(4)
        ]

        op = [
            [
                x for r in rr
                for x in r["other_pair"][j]
            ]
            for j in range(4)
        ]

        def accuracy(values):
            return mean(x > 0 for x in values)

        def effect(values):
            return mean(
                x - y
                for x, y in zip(values, baseline)
            )

        equivalent_alpha = mean(
            r["equivalent_alpha"]
            for r in rr
        )

        mean_cosine = mean(
            x for r in rr
            for x in r["other_pair_cosine"]
        )

        print(
            f"\nBudget={budget:.0%}, "
            f"equivalent alpha={equivalent_alpha:.2f}"
        )

        print(
            f"  Baseline margin={mean(baseline):+.3f}"
            f", other-pair cosine={mean_cosine:+.3f}"
        )

        print(
            f"  Accuracy: "
            f"baseline={accuracy(baseline):.3f} "
            f"actual={accuracy(actual):.3f} "
            f"signflip={mean(accuracy(x) for x in sf):.3f} "
            f"other_pair={mean(accuracy(x) for x in op):.3f}"
        )

        print(
            f"  Margin change: "
            f"actual={effect(actual):+.3f} "
            f"signflip={mean(effect(x) for x in sf):+.3f} "
            f"other_pair={mean(effect(x) for x in op):+.3f}"
        )

        if budget in (0.10, 0.20):
            affected = sum(
                any(x <= 0 for x in r["actual"])
                for r in rr
            )

            print(
                f"  Quartets with errors: "
                f"{affected}/56"
            )

    print(
        "\nEquivalent alpha >1 means overshooting "
        "complete interaction removal."
    )

    print(
        "Exploratory sensitivity comparison; "
        "not independent confirmation."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int,
        choices=(7, 17), required=True
    )

    parser.add_argument(
        "--phase",
        choices=("audit", "collect", "summarize"),
        default="audit",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    torch.set_num_threads(
        min(torch.get_num_threads(), 4)
    )

    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    cases, heldout = quartets()

    if len(cases) != 56:
        raise RuntimeError(
            "Unexpected number of quartets"
        )

    output = Path(
        f"results/ml040_scale_matched_seed{args.seed}.json"
    )

    print(
        f"ML-040 {args.phase}, "
        f"seed={args.seed}, "
        f"device={device}, "
        f"design={DESIGN_HASH}"
    )

    if args.phase == "summarize":
        record = json.loads(output.read_text())

        if (
            not record.get("complete")
            or record.get("design_hash") != DESIGN_HASH
        ):
            raise RuntimeError(
                "Wrong or incomplete saved result"
            )

        describe(
            record["records"], args.seed
        )
        return

    model, checkpoint = load_trained(
        args.seed, heldout, device
    )

    print(
        f"Checkpoint: {checkpoint}; "
        f"blocks={len(model.blocks)}"
    )

    if args.phase == "audit":
        return

    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )

    data = collect_states(
        model, cases, device
    )

    records = []

    for index in range(len(data)):
        records.extend(
            evaluate_one(
                model, data,
                args.seed, index, device
            )
        )

        if (index + 1) % 8 == 0:
            print(
                f"Completed {index+1}/56 quartets"
            )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-040",
            "complete": True,
            "seed": args.seed,
            "source_checkpoint": str(checkpoint),
            "design_hash": DESIGN_HASH,
            "design": SPEC,
            "records": records,
        }, indent=2)
    )

    print("Saved:", output)

    describe(records, args.seed)


if __name__ == "__main__":
    main()
