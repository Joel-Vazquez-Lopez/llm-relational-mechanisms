
"""ML-043: Per-head attention decomposition and causal interventions.

Training pairs nominate attention heads.
Calibration pairs measure causal responses.
Already exposed ML-041 protected pairs are NOT reused.

This is exploratory mechanistic diagnosis.
"""
import argparse
import hashlib
import json
import math
from statistics import mean

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import encode_batch
from ml041_prospective_transfer import (
    DESIGN_HASH as SOURCE_HASH,
    RESULTS,
    SEEDS,
    SPLIT,
    quartets,
    load_model,
)

CONTROLS = 4
SIGNS = (1.0, -1.0, -1.0, 1.0)

SPEC = {
    "experiment": "ML-043",
    "source_hash": SOURCE_HASH,
    "scope": (
        "ML-041 training for head nomination; "
        "calibration for causal intervention"
    ),
    "block": 1,
    "intervention": (
        "Remove each head's readout factorial interaction: "
        "h_i - sign_i * delta_head / 4"
    ),
    "controls": CONTROLS,
    "mlp_conditions": ("live", "replay_natural"),
    "nomination": (
        "Maximum positive training-mean head projection "
        "onto total attention interaction"
    ),
    "status": "exploratory",
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def factorial(states):
    return states[3] - states[2] - states[1] + states[0]


@torch.no_grad()
def get_head_contributions(model, ids, pos):
    """Projected per-head contributions at the readout."""
    attn = model.blocks[0].attn

    if (
        not attn._qkv_same_embed_dim
        or attn.bias_k is not None
        or attn.add_zero_attn
    ):
        raise RuntimeError("Unexpected attention architecture")

    batch, length = ids.shape
    dim = attn.embed_dim
    heads = attn.num_heads
    head_dim = dim // heads

    x = (
        model.embed(ids)
        + model.pos(
            torch.arange(length, device=ids.device)
        )
    )

    z = model.blocks[0].norm1(x)

    q, k, v = F.linear(
        z,
        attn.in_proj_weight,
        attn.in_proj_bias,
    ).split(dim, dim=-1)

    def reshape(t):
        return t.reshape(
            batch, length, heads, head_dim
        ).transpose(1, 2)

    q, k, v = map(reshape, (q, k, v))

    scores = (
        q @ k.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    mask = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=ids.device,
    ).triu(1)

    probs = scores.masked_fill(
        mask[None, None],
        float("-inf"),
    ).softmax(-1)

    values = probs @ v

    batch_idx = torch.arange(
        batch, device=ids.device
    )

    context = values[
        batch_idx, :, pos, :
    ]

    projected = torch.stack([
        F.linear(
            context[:, head, :],
            attn.out_proj.weight[
                :,
                head * head_dim:(head + 1) * head_dim,
            ],
        )
        for head in range(heads)
    ], dim=1)

    natural, _ = attn(
        z, z, z,
        attn_mask=mask,
        need_weights=False,
    )

    reconstructed = (
        projected.sum(dim=1)
        + attn.out_proj.bias
    )

    natural_readout = natural[
        batch_idx, pos, :
    ]

    error = natural_readout - reconstructed

    max_abs = float(error.abs().max())
    relative = float(
        error.norm()
        / natural_readout.norm().clamp_min(1e-8)
    )

    if max_abs > 0.01 or relative > 0.002:
        raise RuntimeError(
            "Head reconstruction failed: "
            f"abs={max_abs}, relative={relative}"
        )

    return projected, max_abs, relative


def signflip(seed, key, head, index, width, device, dtype):
    payload = json.dumps([
        "ML043",
        seed,
        list(key[0]),
        key[1],
        key[2],
        head,
        index,
    ])

    number = int.from_bytes(
        hashlib.sha256(
            payload.encode()
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)

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
def capture_case(model, examples, device):
    ids, pos, labels, choices = encode_batch(examples)

    ids, pos, labels, choices = (
        x.to(device)
        for x in (ids, pos, labels, choices)
    )

    projected, max_err, rel_err = (
        get_head_contributions(
            model, ids, pos
        )
    )

    captures = {}
    indices = torch.arange(4, device=device)

    def hook(_module, _args, output):
        captures["mlp"] = (
            output[indices, pos]
            .detach()
            .clone()
        )

    handle = (
        model.blocks[0].mlp
        .register_forward_hook(hook)
    )

    try:
        logits, states = model(
            ids,
            pos,
            return_states=True,
        )
    finally:
        handle.remove()

    other = torch.where(
        choices[:, 0] == labels,
        choices[:, 1],
        choices[:, 0],
    )

    margins = (
        logits.gather(
            1, labels[:, None]
        ).squeeze(1)
        - logits.gather(
            1, other[:, None]
        ).squeeze(1)
    )

    return {
        "ids": ids,
        "pos": pos,
        "labels": labels,
        "other": other,
        "heads": projected,
        "mlp": captures["mlp"],
        "states": states,
        "margins": margins,
        "head_max_error": max_err,
        "head_rel_error": rel_err,
    }


@torch.no_grad()
def training_ranking(model, device):
    head_deltas = []

    for _, _, _, examples in quartets(
        SPLIT["train"]
    ):
        data = capture_case(
            model, examples, device
        )

        head_deltas.append(
            factorial(data["heads"])
        )

    head_deltas = torch.stack(head_deltas)

    average = head_deltas.mean(dim=0)
    total = average.sum(dim=0)

    denominator = (
        total.square()
        .sum()
        .clamp_min(1e-10)
    )

    contributions = (
        (average * total).sum(dim=1)
        / denominator
    )

    winner = int(contributions.argmax())

    if not torch.isclose(
        contributions.sum(),
        torch.ones((), device=device),
        atol=1e-3,
    ):
        raise RuntimeError(
            "Head contributions do not sum to one"
        )

    norms = (
        head_deltas.norm(dim=2)
        .mean(dim=0)
    )

    return {
        "winner": winner,
        "fractions": contributions.cpu().tolist(),
        "head_mean_interaction_norm": norms.cpu().tolist(),
        "total_mean_interaction_norm": float(
            total.norm()
        ),
    }


@torch.no_grad()
def intervene(model, data, key, seed, device):
    h = data["states"][1]

    dheads = torch.stack([
        factorial(data["heads"][:, j, :])
        for j in range(data["heads"].shape[1])
    ])

    signs = torch.tensor(
        SIGNS,
        device=device,
        dtype=h.dtype,
    )[:, None]

    alternatives = [
        (
            "baseline", -1, -1,
            "live", torch.zeros_like(h)
        )
    ]

    for head in range(dheads.shape[0]):
        delta = dheads[head]

        if not bool(torch.isfinite(delta).all()):
            raise RuntimeError(
                f"Non-finite head interaction: {head}"
            )

        # A zero interaction is a valid scientific observation.
        # Removing it produces a zero perturbation; retain the
        # head and quartet rather than censoring the result.

        vectors = [delta] + [
            delta * signflip(
                seed,
                key,
                head,
                j,
                delta.numel(),
                device,
                delta.dtype,
            )
            for j in range(CONTROLS)
        ]

        for j, vec in enumerate(vectors):
            for state in (
                "live", "replay_natural"
            ):
                alternatives.append((
                    "head",
                    head,
                    j,
                    state,
                    -signs * vec / 4,
                ))

    n = len(alternatives)

    all_vectors = torch.stack([
        item[4] for item in alternatives
    ]).reshape(n * 4, -1)

    fixed = torch.tensor([
        item[3] == "replay_natural"
        for item in alternatives
    ], device=device)

    fixed = fixed.repeat_interleave(4)

    natural_m = data["mlp"].repeat(n, 1)

    ids = data["ids"].repeat(n, 1)
    pos = data["pos"].repeat(n)

    indices = torch.arange(
        n * 4, device=device
    )

    def attn_hook(_module, _inputs, output):
        values, *rest = output
        updated = values.clone()

        updated[indices, pos] += (
            all_vectors.to(updated.dtype)
        )

        return (updated, *rest)

    def mlp_hook(_module, _inputs, output):
        updated = output.clone()
        original = updated[indices, pos]

        updated[indices, pos] = torch.where(
            fixed[:, None],
            natural_m.to(updated.dtype),
            original,
        )

        return updated

    hooks = [
        model.blocks[0].attn.register_forward_hook(
            attn_hook
        ),
        model.blocks[0].mlp.register_forward_hook(
            mlp_hook
        ),
    ]

    try:
        logits = model(
            ids, pos
        ).reshape(n, 4, -1)
    finally:
        for hook in hooks:
            hook.remove()

    target = data["labels"][
        None, :, None
    ].expand(n, -1, -1)

    other = data["other"][
        None, :, None
    ].expand(n, -1, -1)

    margins = (
        logits.gather(2, target)
        - logits.gather(2, other)
    ).squeeze(-1)

    if float(
        (
            margins[0] - data["margins"]
        ).abs().max()
    ) > 0.05:
        raise RuntimeError(
            "Self-patch control mismatch"
        )

    baseline = margins[0]
    rows = []

    for head in range(dheads.shape[0]):
        for mlp_state in (
            "live", "replay_natural"
        ):
            def lookup(j):
                index = next(
                    k
                    for k, item in enumerate(alternatives)
                    if item[:4] == (
                        "head",
                        head,
                        j,
                        mlp_state,
                    )
                )

                return margins[index]

            actual = lookup(0)

            controls = torch.stack([
                lookup(j)
                for j in range(1, CONTROLS + 1)
            ])

            rows.append({
                "head": head,
                "mlp_state": mlp_state,
                "head_delta_norm": float(
                    dheads[head].norm()
                ),
                "actual_margin_change": float(
                    (actual - baseline).mean()
                ),
                "control_margin_change": float(
                    (
                        controls.mean(0) - baseline
                    ).mean()
                ),
                "harm": float(
                    (
                        controls.mean(0) - actual
                    ).mean()
                ),
                "actual_accuracy": float(
                    (actual > 0).float().mean()
                ),
                "control_accuracy": float(
                    (controls > 0).float().mean()
                ),
            })

    return rows


def run(seed, phase, device):
    model, digest = load_model(
        seed, device
    )

    frozen = json.loads(
        (
            RESULTS / "ml041_frozen_predictions.json"
        ).read_text()
    )

    entry = frozen["seeds"][str(seed)]

    if (
        frozen["design_hash"] != SOURCE_HASH
        or digest != entry["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "ML-041 provenance mismatch"
        )

    ranking = training_ranking(
        model, device
    )

    print(
        "TRAINING-only head ranking:",
        ranking
    )

    cases = quartets(
        SPLIT[
            "train" if phase == "audit"
            else "calibration"
        ]
    )

    if phase == "audit":
        c = capture_case(
            model, cases[0][3], device
        )

        print(
            "HEAD RECONSTRUCTION PASS | "
            f"max_abs={c['head_max_error']:.2g} "
            f"relative={c['head_rel_error']:.2g}"
        )

        checks = intervene(
            model,
            c,
            cases[0][:3],
            seed,
            device,
        )

        assert len(checks) == (
            model.blocks[0].attn.num_heads * 2
        )

        print(
            "PATCH SMOKE PASS | "
            f"{len(checks)} head/MLP conditions"
        )
        return

    output = (
        RESULTS / f"ml043_heads_seed{seed}.json"
    )

    if output.exists():
        raise RuntimeError(
            f"Refusing overwrite: {output}"
        )

    rows = []

    for i, (
        pair, verb, direction, examples
    ) in enumerate(cases, 1):

        c = capture_case(
            model, examples, device
        )

        for r in intervene(
            model,
            c,
            (pair, verb, direction),
            seed,
            device,
        ):
            rows.append({
                "pair": list(pair),
                "verb": verb,
                "direction": direction,
                **r,
            })

        if i % 8 == 0:
            print(
                f"seed {seed}: "
                f"{i}/{len(cases)} quartets"
            )

    summary = {}

    for head in range(
        model.blocks[0].attn.num_heads
    ):
        summary[str(head)] = {}

        for mlp_state in (
            "live", "replay_natural"
        ):
            subset = [
                r for r in rows
                if r["head"] == head
                and r["mlp_state"] == mlp_state
            ]

            summary[str(head)][mlp_state] = {
                k: mean(x[k] for x in subset)
                for k in (
                    "harm",
                    "actual_margin_change",
                    "control_margin_change",
                    "actual_accuracy",
                    "control_accuracy",
                    "head_delta_norm",
                )
            }

    result = {
        "experiment": "ML-043",
        "complete": True,
        "design_hash": HASH,
        "seed": seed,
        "eligible_ml041": entry["eligible"],
        "source_checkpoint_sha256": digest,
        "training_ranking": ranking,
        "summary": summary,
        "records": rows,
    }

    output.write_text(
        json.dumps(result, indent=2)
    )

    print(
        f"\n=== ML-043 seed {seed} | "
        f"TRAINING-NOMINATED HEAD "
        f"{ranking['winner']} ==="
    )

    for head, details in summary.items():
        live = details["live"]
        fixed = details["replay_natural"]

        print(
            f"head {head}: "
            f"train_frac="
            f"{ranking['fractions'][int(head)]:+.3f} "
            f"live_harm={live['harm']:+.3f} "
            f"fixed_mlp_harm={fixed['harm']:+.3f} "
            f"live_acc={live['actual_accuracy']:.3f} "
            f"fixed_acc={fixed['actual_accuracy']:.3f}"
        )

    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=("audit", "collect"),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=SEEDS,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    device = (
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if args.device == "auto"
        else args.device
    )

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    print(
        f"ML-043 {args.phase} "
        f"seed={args.seed} "
        f"device={device} "
        f"design={HASH}"
    )

    run(seed=args.seed, phase=args.phase, device=device)


if __name__ == "__main__":
    main()
