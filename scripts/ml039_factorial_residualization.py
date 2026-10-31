
"""ML-039: Balanced causal removal of voice x question interactions.

For each held-out quartet, compute:
    delta = h11 - h10 - h01 + h00

Remove the factorial interaction component from all four cells:
    h_i' = h_i - alpha * sign_i * delta / 4

Controls use deterministic sign-flipped deltas of equal norm.

This is exploratory. Functional sensitivity does not establish a
uniquely necessary grammar circuit.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    VOICES,
    ROLES,
    VOCAB,
    encode_batch,
    split_examples,
)

ALPHAS = (0.5, 1.0)
N_CONTROLS = 4
BLOCKS = (0, 1)
RESULTS = Path("results")

# Quartet order:
# active-agent, active-patient, passive-agent, passive-patient
SIGNS = (1.0, -1.0, -1.0, 1.0)

SPEC = {
    "experiment": "ML-039",
    "seeds": (7, 17),
    "blocks": (1, 2),
    "alphas": ALPHAS,
    "controls": N_CONTROLS,
    "patch": "h_i - alpha * sign_i * delta / 4",
    "source": "ML-037 trained scratch Transformer checkpoints",
    "status": "exploratory follow-up after ML-038",
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def quartets():
    _, test, heldout = split_examples()
    grouped = defaultdict(dict)

    for example in test:
        key = (
            example.pair,
            example.verb,
            example.direction,
        )
        sub = (example.voice, example.role)

        if sub in grouped[key]:
            raise RuntimeError(f"Duplicate: {key}, {sub}")

        grouped[key][sub] = example

    expected = set(product(VOICES, ROLES))

    if len(grouped) != 56 or any(
        set(cells) != expected
        for cells in grouped.values()
    ):
        raise RuntimeError("Expected 56 complete quartets")

    return [
        (
            key,
            [
                grouped[key][v, r]
                for v, r in product(VOICES, ROLES)
            ],
        )
        for key in sorted(grouped)
    ], heldout


def load_trained(seed, heldout, device):
    path = RESULTS / f"ml037_scratch_seed{seed}.pt"

    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if saved["vocab"] != VOCAB:
        raise RuntimeError("Vocabulary mismatch")

    original_pairs = [
        tuple(p)
        for p in saved["result"]["heldout_unordered_pairs"]
    ]

    if original_pairs != heldout:
        raise RuntimeError("Held-out split mismatch")

    model = ScratchTransformer()
    model.load_state_dict(
        saved["state_dict"],
        strict=True,
    )
    model.to(device).eval()

    return model, path


def signflip(seed, base, block, index, width, device, dtype):
    payload = json.dumps([
        "ML-039",
        seed,
        [list(base[0]), base[1], base[2]],
        block,
        index,
    ])

    number = int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8],
        "big",
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(number)

    bits = torch.randint(
        0, 2, (width,),
        generator=generator,
    )

    return (2 * bits - 1).to(
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def run_quartet(model, seed, base, examples, block, device):
    ids, readout, labels, choices = encode_batch(examples)

    ids = ids.to(device)
    readout = readout.to(device)
    labels = labels.to(device)
    choices = choices.to(device)

    original, states = model(
        ids,
        readout,
        return_states=True,
    )

    h = states[block + 1]

    # h00, h01, h10, h11
    delta = h[3] - h[2] - h[1] + h[0]

    signs = torch.tensor(
        SIGNS,
        device=device,
        dtype=h.dtype,
    ).unsqueeze(1)

    delta_norm = float(delta.norm())

    if delta_norm < 1e-9:
        raise RuntimeError(
            f"Degenerate interaction: {base}"
        )

    other = torch.where(
        choices[:, 0] == labels,
        choices[:, 1],
        choices[:, 0],
    )

    base_margin = (
        original.gather(1, labels[:, None]).squeeze(1)
        - original.gather(1, other[:, None]).squeeze(1)
    )

    base_nll = F.cross_entropy(
        original,
        labels,
        reduction="none",
    )

    replacements = [h]
    specs = [("self", 0.0, -1)]

    controls = [
        delta * signflip(
            seed,
            base,
            block,
            j,
            delta.numel(),
            device,
            delta.dtype,
        )
        for j in range(N_CONTROLS)
    ]

    for alpha in ALPHAS:
        replacements.append(
            h - alpha * signs * delta / 4.0
        )
        specs.append(("actual", alpha, -1))

        for j, control in enumerate(controls):
            replacements.append(
                h - alpha * signs * control / 4.0
            )
            specs.append(("control", alpha, j))

    # Verify the balanced intervention removes the
    # factorial interaction at alpha=1.
    exact = replacements[
        specs.index(("actual", 1.0, -1))
    ]

    residual = (
        exact[3] - exact[2] - exact[1] + exact[0]
    )

    residual_ratio = (
        float(residual.norm()) / delta_norm
    )

    if residual_ratio > 1e-3:
        raise RuntimeError(
            f"Factorial identity failed: {residual_ratio}"
        )

    count = len(replacements)

    replacements = torch.stack(replacements)

    big_ids = ids.repeat(count, 1)
    big_readout = readout.repeat(count)

    replacement_vectors = replacements.reshape(
        count * 4,
        h.shape[-1],
    )

    def patch_hook(_module, _inputs, output):
        patched = output.clone()

        batch_ids = torch.arange(
            patched.shape[0],
            device=patched.device,
        )

        patched[
            batch_ids,
            big_readout,
            :
        ] = replacement_vectors.to(patched.dtype)

        return patched

    handle = model.blocks[block].register_forward_hook(
        patch_hook
    )

    try:
        patched_logits = model(
            big_ids,
            big_readout,
        ).reshape(count, 4, -1)
    finally:
        handle.remove()

    targets = labels[
        None, :, None
    ].expand(count, -1, 1)

    wrong = other[
        None, :, None
    ].expand(count, -1, 1)

    margins = (
        patched_logits.gather(2, targets).squeeze(-1)
        - patched_logits.gather(2, wrong).squeeze(-1)
    )

    nll = -F.log_softmax(
        patched_logits,
        dim=-1,
    ).gather(2, targets).squeeze(-1)

    if float(
        (margins[0] - base_margin).abs().max()
    ) > 0.05:
        raise RuntimeError(
            f"Self-patch margin mismatch: {base}"
        )

    if float(
        (nll[0] - base_nll).abs().max()
    ) > 0.05:
        raise RuntimeError(
            f"Self-patch loss mismatch: {base}"
        )

    rows = []

    for alpha in ALPHAS:
        actual_idx = specs.index(
            ("actual", alpha, -1)
        )

        control_indices = [
            specs.index(("control", alpha, j))
            for j in range(N_CONTROLS)
        ]

        rows.append({
            "base": [
                list(base[0]),
                base[1],
                base[2],
            ],
            "block": block + 1,
            "alpha": alpha,
            "delta_over_state": (
                delta_norm
                / float(h.norm(dim=1).mean().clamp_min(1e-12))
            ),
            "residual_ratio_at_full_removal": residual_ratio,
            "baseline_margins": base_margin.cpu().tolist(),
            "actual_margins": margins[
                actual_idx
            ].cpu().tolist(),
            "control_margins": margins[
                control_indices
            ].cpu().tolist(),
            "baseline_nll": base_nll.cpu().tolist(),
            "actual_nll": nll[
                actual_idx
            ].cpu().tolist(),
            "control_nll": nll[
                control_indices
            ].cpu().tolist(),
        })

    return rows


def summarize(rows, seed):
    print(
        f"\n=== ML-039 | SEED {seed} "
        "| BALANCED INTERACTION REMOVAL ==="
    )

    print(
        "Per condition: 56 quartets x 4 questions = 224 answers"
    )

    names = (
        "active-agent",
        "active-patient",
        "passive-agent",
        "passive-patient",
    )

    for block, alpha in product((1, 2), ALPHAS):
        rr = [
            r for r in rows
            if r["block"] == block
            and r["alpha"] == alpha
        ]

        if len(rr) != 56:
            raise RuntimeError("Missing quartets")

        baseline = [
            m for r in rr
            for m in r["baseline_margins"]
        ]

        actual = [
            m for r in rr
            for m in r["actual_margins"]
        ]

        controls = [
            [
                m for r in rr
                for m in r["control_margins"][j]
            ]
            for j in range(N_CONTROLS)
        ]

        base_acc = mean(m > 0 for m in baseline)
        actual_acc = mean(m > 0 for m in actual)
        control_acc = mean(
            mean(m > 0 for m in c)
            for c in controls
        )

        actual_changes = [
            a - b
            for a, b in zip(actual, baseline)
        ]

        control_changes = [
            mean(
                c[i] - baseline[i]
                for c in controls
            )
            for i in range(len(baseline))
        ]

        actual_nll_change = [
            a - b
            for r in rr
            for a, b in zip(
                r["actual_nll"],
                r["baseline_nll"],
            )
        ]

        control_nll_change = [
            mean(
                r["control_nll"][j][k]
                - r["baseline_nll"][k]
                for j in range(N_CONTROLS)
            )
            for r in rr
            for k in range(4)
        ]

        print(
            f"\nBLOCK {block} | alpha={alpha:.2f}"
        )

        print(
            f"Accuracy: original={base_acc:.3f} "
            f"actual={actual_acc:.3f} "
            f"controls={control_acc:.3f}"
        )

        print(
            "Margin change: "
            f"actual={mean(actual_changes):+.3f} "
            f"controls={mean(control_changes):+.3f}"
        )

        print(
            "Gold NLL change: "
            f"actual={mean(actual_nll_change):+.3f} "
            f"controls={mean(control_nll_change):+.3f}"
        )

        for k, name in enumerate(names):
            a = mean(
                r["actual_margins"][k] > 0
                for r in rr
            )

            c = mean(
                mean(
                    r["control_margins"][j][k] > 0
                    for j in range(N_CONTROLS)
                )
                for r in rr
            )

            print(
                f"  {name}: "
                f"actual={a:.3f} controls={c:.3f}"
            )

    print(
        "\nThis is a factorial intervention, "
        "not proof of a unique necessary circuit."
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
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    cases, heldout = quartets()

    print(
        f"ML-039 {args.phase} | seed={args.seed} "
        f"device={device} quartets={len(cases)} "
        f"hash={DESIGN_HASH}"
    )

    if args.phase == "audit":
        model, path = load_trained(
            args.seed, heldout, device
        )
        print(
            "Verified checkpoint:", path,
            "| blocks:", len(model.blocks)
        )
        return

    output = RESULTS / (
        f"ml039_factorial_residualization_seed{args.seed}.json"
    )

    if args.phase == "summarize":
        data = json.loads(output.read_text())

        if (
            not data.get("complete")
            or data.get("design_hash") != DESIGN_HASH
        ):
            raise RuntimeError("Wrong or incomplete results")

        summarize(
            data["records"], args.seed
        )
        return

    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing results: {output}"
        )

    model, source = load_trained(
        args.seed, heldout, device
    )

    rows = []

    for i, (base, examples) in enumerate(cases, 1):
        for block in BLOCKS:
            rows.extend(
                run_quartet(
                    model, args.seed, base,
                    examples, block, device
                )
            )

        if i % 8 == 0 or i == len(cases):
            print(
                f"DONE {i}/{len(cases)} "
                f"records={len(rows)}"
            )

    result = {
        "experiment": "ML-039",
        "design_hash": DESIGN_HASH,
        "complete": True,
        "seed": args.seed,
        "checkpoint": str(source),
        "alphas": list(ALPHAS),
        "controls": N_CONTROLS,
        "records": rows,
    }

    output.write_text(
        json.dumps(result, indent=2)
    )

    print("Saved:", output)

    summarize(rows, args.seed)


if __name__ == "__main__":
    main()
