
"""ML-038: Functional test of learned interaction vectors.

For each held-out event quartet:
  delta = h11 - h10 - h01 + h00

Subtract delta from the natural passive-patient readout state
at block 1 or block 2, with alpha = 0.5 or 1.0.

Controls are deterministic coordinate sign flips of the same
delta, preserving its Euclidean norm.

This is exploratory, not proof that a circuit is necessary.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

import torch

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    VOICES,
    ROLES,
    VOCAB,
    answer,
    encode_batch,
    split_examples,
)

ALPHAS = (0.5, 1.0)
N_CONTROLS = 4
LAYERS = (0, 1)
ROOT = Path("results")


def deterministic_signs(
    seed, base_key, block, control,
    width, device, dtype
):
    payload = repr((
        "ML-038", seed, base_key, block, control
    )).encode()

    number = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big"
    ) % (2**63 - 1)

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(number)

    bits = torch.randint(
        0, 2, (width,), generator=generator
    )

    return (2 * bits - 1).to(
        device=device, dtype=dtype
    )


def get_quartets():
    _, test, heldout = split_examples()

    grouped = defaultdict(dict)

    for example in test:
        base = (
            example.pair,
            example.verb,
            example.direction,
        )

        key = (
            example.voice,
            example.role,
        )

        if key in grouped[base]:
            raise RuntimeError(
                f"Duplicate example: {base} {key}"
            )

        grouped[base][key] = example

    expected = set(product(VOICES, ROLES))

    assert len(grouped) == 56

    for base, cells in grouped.items():
        if set(cells) != expected:
            raise RuntimeError(
                f"Incomplete quartet: {base}"
            )

    quartets = [
        (
            base,
            [
                grouped[base][v, q]
                for v, q in product(VOICES, ROLES)
            ]
        )
        for base in sorted(grouped)
    ]

    return quartets, heldout


def load_model(seed, device, heldout):
    path = ROOT / (
        f"ml037_scratch_seed{seed}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if saved["vocab"] != VOCAB:
        raise RuntimeError(
            "Vocabulary differs from ML-037"
        )

    saved_pairs = [
        tuple(x)
        for x in saved["result"][
            "heldout_unordered_pairs"
        ]
    ]

    if saved_pairs != heldout:
        raise RuntimeError(
            "Test split differs from checkpoint"
        )

    model = ScratchTransformer()

    model.load_state_dict(
        saved["state_dict"],
        strict=True,
    )

    model.to(device).eval()

    return model, path


@torch.no_grad()
def measure(
    model, seed, base_key,
    quartet, block, device
):
    ids, positions, _, _ = encode_batch(
        quartet
    )

    original_logits, states = model(
        ids.to(device),
        positions.to(device),
        return_states=True,
    )

    # Quartet order:
    # active-agent, active-patient,
    # passive-agent, passive-patient.
    h00, h01, h10, h11 = states[block + 1]

    delta = (
        h11 - h10 - h01 + h00
    )

    delta_norm = torch.linalg.vector_norm(
        delta
    )

    h_norm = torch.linalg.vector_norm(
        h11
    ).clamp_min(1e-12)

    if (
        not torch.isfinite(delta_norm)
        or float(delta_norm) < 1e-10
    ):
        raise RuntimeError(
            f"Degenerate delta: {base_key}"
        )

    target_example = quartet[3]

    pair = target_example.pair

    correct_name = answer(
        target_example
    )

    target = VOCAB[correct_name]

    other = VOCAB[
        next(
            name for name in pair
            if name != correct_name
        )
    ]

    baseline = float(
        original_logits[3, target]
        - original_logits[3, other]
    )

    signflipped = [
        delta * deterministic_signs(
            seed,
            base_key,
            block,
            j,
            delta.numel(),
            delta.device,
            delta.dtype,
        )
        for j in range(N_CONTROLS)
    ]

    replacements = [h11]
    indices = {}

    for alpha in ALPHAS:
        start = len(replacements)

        replacements.append(
            h11 - alpha * delta
        )

        replacements.extend(
            h11 - alpha * d
            for d in signflipped
        )

        indices[alpha] = (
            start,
            list(
                range(
                    start + 1,
                    start + 1 + N_CONTROLS
                )
            ),
        )

    replacements = torch.stack(
        replacements
    )

    target_ids, target_pos, _, _ = (
        encode_batch([target_example])
    )

    input_ids = target_ids.to(
        device
    ).repeat(
        len(replacements), 1
    )

    readout = target_pos.to(
        device
    ).repeat(
        len(replacements)
    )

    position = int(readout[0])

    def inject(_module, _inputs, output):
        if output.shape[0] != len(
            replacements
        ):
            raise RuntimeError(
                "Unexpected patch batch size"
            )

        patched = output.clone()

        patched[:, position, :] = (
            replacements.to(
                dtype=output.dtype
            )
        )

        return patched

    handle = (
        model.blocks[block]
        .register_forward_hook(inject)
    )

    try:
        patched_logits = model(
            input_ids, readout
        )
    finally:
        handle.remove()

    margins = (
        patched_logits[:, target]
        - patched_logits[:, other]
    ).detach().cpu()

    self_effect = (
        float(margins[0]) - baseline
    )

    if abs(self_effect) > 0.05:
        raise RuntimeError(
            f"Self-patch mismatch: "
            f"{base_key} block={block} "
            f"effect={self_effect:+.4f}"
        )

    rows = []

    for alpha, (
        actual_idx, ctrl_indices
    ) in indices.items():

        actual_margin = float(
            margins[actual_idx]
        )

        controls = [
            float(margins[j])
            for j in ctrl_indices
        ]

        rows.append({
            "seed": seed,
            "base": [
                list(base_key[0]),
                base_key[1],
                base_key[2],
            ],
            "block": block + 1,
            "alpha": alpha,
            "baseline_margin": baseline,
            "actual_margin": actual_margin,
            "control_margins": controls,
            "actual_effect": (
                actual_margin - baseline
            ),
            "control_effects": [
                x - baseline
                for x in controls
            ],
            "delta_over_state": float(
                delta_norm / h_norm
            ),
        })

    return rows


def summarize(rows, seed):
    print(
        f"\n=== ML-038 | SEED {seed} "
        "| HELD-OUT QUARTETS ==="
    )

    for layer, alpha in product(
        (1, 2), ALPHAS
    ):
        rr = [
            r for r in rows
            if (
                r["block"] == layer
                and r["alpha"] == alpha
            )
        ]

        if (
            len(rr) != 56
            or any(
                len(r["control_margins"])
                != N_CONTROLS
                for r in rr
            )
        ):
            raise RuntimeError(
                "Incomplete results"
            )

        actual = [
            r["actual_effect"]
            for r in rr
        ]

        controls = [
            mean(r["control_effects"])
            for r in rr
        ]

        excess = [
            a - c
            for a, c in zip(
                actual, controls
            )
        ]

        accuracy_before = mean(
            r["baseline_margin"] > 0
            for r in rr
        )

        accuracy_after = mean(
            r["actual_margin"] > 0
            for r in rr
        )

        accuracy_control = mean(
            mean(
                m > 0
                for m in r["control_margins"]
            )
            for r in rr
        )

        relative_norm = mean(
            r["delta_over_state"]
            for r in rr
        )

        print(
            f"\nBlock={layer} "
            f"alpha={alpha:.2f}"
        )

        print(
            f"  Delta/state norm: "
            f"{relative_norm:.3f}"
        )

        print(
            f"  Actual margin effect: "
            f"{mean(actual):+.3f}"
        )

        print(
            f"  Control margin effect: "
            f"{mean(controls):+.3f}"
        )

        print(
            f"  Actual minus control: "
            f"{mean(excess):+.3f}"
        )

        print(
            f"  Negative excess: "
            f"{sum(x < 0 for x in excess)}/56"
        )

        print(
            "  Pairwise accuracy: "
            f"original={accuracy_before:.3f}, "
            f"delta-subtracted={accuracy_after:.3f}, "
            f"controls={accuracy_control:.3f}"
        )

    print(
        "\nRemoving an interaction vector "
        "does not establish circuit necessity."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        choices=(7, 17),
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
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
        raise RuntimeError(
            "MPS unavailable; use --device cpu"
        )

    quartets, heldout = get_quartets()

    model, source = load_model(
        args.seed, device, heldout
    )

    print(
        f"ML-038 | seed={args.seed} "
        f"device={device}"
    )

    print(
        "Checkpoint:",
        source,
    )

    output = ROOT / (
        f"ml038_interaction_ablation_seed"
        f"{args.seed}.json"
    )

    rows = []

    for i, (
        base, quartet
    ) in enumerate(quartets, 1):

        for block in LAYERS:
            rows.extend(
                measure(
                    model,
                    args.seed,
                    base,
                    quartet,
                    block,
                    device,
                )
            )

        if (
            i % 8 == 0
            or i == len(quartets)
        ):
            print(
                f"Completed {i}/"
                f"{len(quartets)} quartets"
            )

    result = {
        "experiment": "ML-038",
        "seed": args.seed,
        "checkpoint": str(source),
        "complete": True,
        "design": {
            "alphas": ALPHAS,
            "controls": N_CONTROLS,
            "blocks": [1, 2],
            "recipient": "passive patient",
        },
        "records": rows,
    }

    output.write_text(
        json.dumps(result, indent=2)
    )

    print(
        "Saved:",
        output,
    )

    summarize(rows, args.seed)


if __name__ == "__main__":
    main()
