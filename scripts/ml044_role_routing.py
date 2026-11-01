
"""ML-044: Question-conditioned grammatical role routing.

Training examples nominate a block-2 attention head.
Calibration examples test agent/patient attention routing and the
effect of swapping their attention weights at the answer readout.

Previously exposed ML-041 protected examples are never accessed.
Exploratory, not confirmatory.
"""

import argparse
import hashlib
import json
import math
from statistics import mean

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import encode_batch, prompt
from ml041_prospective_transfer import (
    DESIGN_HASH as SOURCE_HASH,
    RESULTS,
    SEEDS,
    SPLIT,
    quartets,
    load_model,
)

CONTROLS = 4

SPEC = {
    "experiment": "ML-044",
    "source_design_hash": SOURCE_HASH,
    "data": "ML-041 training for nomination; calibration for evaluation",
    "linguistic_contrast": "agent vs patient question, active vs passive",
    "nomination": "highest mean block-2 agent-question routing contrast on training",
    "intervention": "swap agent/patient attention weights within one head at readout",
    "controls": "four equal-norm coordinate signflips per quartet cell",
    "status": "exploratory linguistic mechanism study",
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def participant_positions(examples):
    positions = []

    for ex in examples:
        a, b = (
            ex.pair if ex.direction == 0
            else ex.pair[::-1]
        )

        tokens = prompt(ex)

        if ex.voice == "active":
            agent, patient = 1, 3
        else:
            agent, patient = 5, 1

        assert tokens[agent] == a
        assert tokens[patient] == b

        positions.append((agent, patient))

    return torch.tensor(
        positions, dtype=torch.long
    )


def question_contrast(logratio):
    # Quartet order:
    # active-agent, active-patient,
    # passive-agent, passive-patient.
    return torch.stack([
        logratio[0] - logratio[1],
        logratio[2] - logratio[3],
    ])


@torch.no_grad()
def capture(model, examples, device):
    ids, pos, labels, choices = encode_batch(examples)

    ids, pos, labels, choices = (
        x.to(device)
        for x in (ids, pos, labels, choices)
    )

    participants = participant_positions(
        examples
    ).to(device)

    captured = {}
    hooks = []

    for block_index in (0, 1):

        def pre_hook(_module, args, index=block_index):
            captured[index, "input"] = (
                args[0].detach().clone()
            )

        def post_hook(
            _module, _args, output, index=block_index
        ):
            captured[index, "output"] = (
                output[0].detach().clone()
            )

        attention = model.blocks[block_index].attn

        hooks.append(
            attention.register_forward_pre_hook(pre_hook)
        )

        hooks.append(
            attention.register_forward_hook(post_hook)
        )

    try:
        logits, states = model(
            ids, pos, return_states=True
        )
    finally:
        for hook in hooks:
            hook.remove()

    result = {
        "ids": ids,
        "pos": pos,
        "labels": labels,
        "choices": choices,
        "participants": participants,
        "logits": logits,
        "states": states,
        "layers": [],
    }

    batch_indices = torch.arange(
        4, device=device
    )

    for block in (0, 1):
        attn = model.blocks[block].attn

        if (
            not attn._qkv_same_embed_dim
            or attn.bias_k is not None
            or attn.add_zero_attn
        ):
            raise RuntimeError(
                "Unexpected attention architecture"
            )

        z = captured[block, "input"]
        batch, length, width = z.shape

        heads = attn.num_heads
        head_width = width // heads

        q, k, v = F.linear(
            z,
            attn.in_proj_weight,
            attn.in_proj_bias,
        ).split(width, dim=-1)

        def split_heads(x):
            return x.reshape(
                batch, length, heads, head_width
            ).transpose(1, 2)

        q, k, v = map(
            split_heads, (q, k, v)
        )

        q_readout = q[
            batch_indices, :, pos, :
        ]

        scores = torch.einsum(
            "bhd,bhld->bhl",
            q_readout, k,
        ) / math.sqrt(head_width)

        masked = (
            torch.arange(length, device=device)[
                None, None, :
            ] > pos[:, None, None]
        )

        weights = scores.masked_fill(
            masked, float("-inf")
        ).softmax(-1)

        combined = torch.einsum(
            "bhl,bhld->bhd",
            weights, v,
        )

        reconstructed = F.linear(
            combined.reshape(batch, width),
            attn.out_proj.weight,
            attn.out_proj.bias,
        )

        natural_output = captured[
            block, "output"
        ][batch_indices, pos]

        error = float(
            (reconstructed - natural_output)
            .abs().max()
        )

        if error > 0.003:
            raise RuntimeError(
                f"Block {block + 1} reconstruction "
                f"failed: {error}"
            )

        agent = participants[:, 0]
        patient = participants[:, 1]

        score_agent = scores[
            batch_indices, :, agent
        ]

        score_patient = scores[
            batch_indices, :, patient
        ]

        logratio = score_agent - score_patient

        wa = weights[
            batch_indices, :, agent
        ]

        wp = weights[
            batch_indices, :, patient
        ]

        va = v[
            batch_indices, :, agent, :
        ]

        vp = v[
            batch_indices, :, patient, :
        ]

        # Swapping the two weights produces this
        # exact change to the projected head output.
        swap_deltas = []

        for head in range(heads):
            W = attn.out_proj.weight[
                :,
                head * head_width:
                (head + 1) * head_width,
            ]

            delta = (
                (wp[:, head] - wa[:, head])[:, None]
                * F.linear(
                    va[:, head] - vp[:, head],
                    W,
                )
            )

            swap_deltas.append(delta)

        swap_deltas = torch.stack(
            swap_deltas, dim=1
        )

        # Independently verify the swap algebra
        # using head zero and explicit reweighting.
        reweighted = weights[:, 0].clone()

        reweighted[
            batch_indices, agent
        ] = wp[:, 0]

        reweighted[
            batch_indices, patient
        ] = wa[:, 0]

        direct_change = F.linear(
            torch.einsum(
                "bl,bld->bd",
                reweighted - weights[:, 0],
                v[:, 0],
            ),
            attn.out_proj.weight[
                :, :head_width
            ],
        )

        if float(
            (
                direct_change
                - swap_deltas[:, 0]
            ).abs().max()
        ) > 0.001:
            raise RuntimeError(
                "Attention weight-swap identity failed"
            )

        result["layers"].append({
            "logratio": logratio,
            "weights_agent": wa,
            "weights_patient": wp,
            "swap_deltas": swap_deltas,
            "reconstruction_error": error,
        })

    # Question-conditioned participant ratios cannot
    # change in block 1 for this architecture.
    initial = question_contrast(
        result["layers"][0]["logratio"]
    )

    invariance_error = float(
        initial.abs().max()
    )

    if invariance_error > 5e-5:
        raise RuntimeError(
            "First-block invariance failed: "
            f"{invariance_error}"
        )

    result["first_block_invariance_error"] = (
        invariance_error
    )

    return result


def signflip(
    seed, key, head, control,
    device, width, dtype,
):
    payload = json.dumps([
        "ML-044",
        seed,
        list(key[0]),
        key[1],
        key[2],
        head,
        control,
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
        0, 2, (4, width),
        generator=generator,
    )

    return (2 * bits - 1).to(
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def intervene(model, data, seed, key, device):
    deltas = data["layers"][1]["swap_deltas"]

    heads, width = deltas.shape[1:]

    conditions = [(
        "baseline", -1, -1,
        torch.zeros_like(deltas[:, 0]),
    )]

    for head in range(heads):
        delta = deltas[:, head]

        conditions.append((
            "swap", head, -1, delta,
        ))

        for j in range(CONTROLS):
            controls = delta * signflip(
                seed, key, head, j,
                device, width, delta.dtype,
            )

            conditions.append((
                "control", head, j, controls,
            ))

    count = len(conditions)

    offsets = torch.stack([
        item[3] for item in conditions
    ]).reshape(count * 4, width)

    ids = data["ids"].repeat(count, 1)
    pos = data["pos"].repeat(count)

    batch_indices = torch.arange(
        count * 4, device=device
    )

    def patch_attention(_module, _inputs, output):
        values, *remaining = output

        modified = values.clone()

        modified[
            batch_indices, pos
        ] = (
            modified[batch_indices, pos]
            + offsets.to(modified.dtype)
        )

        return (modified, *remaining)

    handle = (
        model.blocks[1].attn
        .register_forward_hook(patch_attention)
    )

    try:
        logits = model(
            ids, pos
        ).reshape(count, 4, -1)
    finally:
        handle.remove()

    labels = data["labels"]
    choices = data["choices"]

    other = torch.where(
        choices[:, 0] == labels,
        choices[:, 1],
        choices[:, 0],
    )

    targets = labels[
        None, :, None
    ].expand(count, -1, -1)

    alternatives = other[
        None, :, None
    ].expand(count, -1, -1)

    margins = (
        logits.gather(2, targets)
        - logits.gather(2, alternatives)
    ).squeeze(-1)

    base = (
        data["logits"].gather(
            1, labels[:, None]
        ).squeeze(1)
        - data["logits"].gather(
            1, other[:, None]
        ).squeeze(1)
    )

    if float(
        (margins[0] - base).abs().max()
    ) > 0.05:
        raise RuntimeError(
            "Self-patch baseline mismatch"
        )

    rows = []

    for head in range(heads):

        def get(kind, j):
            index = next(
                i for i, item in enumerate(conditions)
                if item[:3] == (kind, head, j)
            )
            return margins[index]

        actual = get("swap", -1)

        control_values = torch.stack([
            get("control", j)
            for j in range(CONTROLS)
        ])

        rows.append({
            "head": head,
            "swap_norm_over_residual": float(
                deltas[:, head]
                .norm(dim=-1).mean()
                / data["states"][2]
                .norm(dim=-1).mean()
                .clamp_min(1e-8)
            ),
            "harm": float(
                (
                    control_values.mean(0)
                    - actual
                ).mean()
            ),
            "margin_change": float(
                (actual - base).mean()
            ),
            "baseline_accuracy": float(
                (base > 0).float().mean()
            ),
            "swap_accuracy": float(
                (actual > 0).float().mean()
            ),
            "controls_accuracy": float(
                (control_values > 0)
                .float().mean()
            ),
            "question_harm": [
                float(
                    control_values.mean(0)[j]
                    - actual[j]
                )
                for j in range(4)
            ],
        })

    return rows


def nominate(model, device):
    contrasts = []

    for _, _, _, examples in quartets(
        SPLIT["train"]
    ):
        captured = capture(
            model, examples, device
        )

        contrasts.append(
            question_contrast(
                captured["layers"][1]["logratio"]
            )
        )

    scores = torch.stack(
        contrasts
    ).mean(dim=(0, 1))

    winner = int(scores.argmax())

    return winner, scores.cpu().tolist()


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
            "Checkpoint provenance mismatch"
        )

    if phase == "audit":
        example_quartet = quartets(
            SPLIT["train"]
        )[0][3]

        captured = capture(
            model, example_quartet, device
        )

        print(
            "AUDIT PASS: head reconstruction, "
            "weight-swap identity and "
            "block-1 question invariance.",
            captured[
                "first_block_invariance_error"
            ],
        )
        return

    output = RESULTS / (
        f"ml044_role_routing_seed{seed}.json"
    )

    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )

    winner, train_scores = nominate(
        model, device
    )

    print(
        f"Seed {seed}: training-nominated "
        f"head={winner}, scores={train_scores}"
    )

    records = []

    for i, (
        pair, verb, direction, examples
    ) in enumerate(
        quartets(SPLIT["calibration"]), 1
    ):
        captured = capture(
            model, examples, device
        )

        effects = intervene(
            model,
            captured,
            seed,
            (pair, verb, direction),
            device,
        )

        records.append({
            "pair": list(pair),
            "verb": verb,
            "direction": direction,
            "block1_routing": (
                question_contrast(
                    captured["layers"][0]["logratio"]
                ).cpu().tolist()
            ),
            "block2_routing": (
                question_contrast(
                    captured["layers"][1]["logratio"]
                ).cpu().tolist()
            ),
            "effects": effects,
        })

        if i % 8 == 0:
            print(
                f"Seed {seed}: "
                f"{i}/56 calibration quartets"
            )

    summary = {}

    for head in range(
        model.blocks[1].attn.num_heads
    ):
        effects = [
            r["effects"][head]
            for r in records
        ]

        summary[str(head)] = {
            "active_routing": mean(
                r["block2_routing"][0][head]
                for r in records
            ),
            "passive_routing": mean(
                r["block2_routing"][1][head]
                for r in records
            ),
            "harm": mean(
                e["harm"] for e in effects
            ),
            "baseline_accuracy": mean(
                e["baseline_accuracy"]
                for e in effects
            ),
            "swap_accuracy": mean(
                e["swap_accuracy"]
                for e in effects
            ),
            "control_accuracy": mean(
                e["controls_accuracy"]
                for e in effects
            ),
            "swap_norm_over_residual": mean(
                e["swap_norm_over_residual"]
                for e in effects
            ),
        }

    result = {
        "experiment": "ML-044",
        "complete": True,
        "design_hash": DESIGN_HASH,
        "seed": seed,
        "eligible_ml041": entry["eligible"],
        "checkpoint_sha256": digest,
        "training_nominated_head": winner,
        "training_routing_scores": train_scores,
        "summary": summary,
        "records": records,
    }

    output.write_text(
        json.dumps(result, indent=2)
    )

    print(
        f"\nML-044 seed={seed} "
        f"eligible={entry['eligible']} "
        f"nominated_head={winner}"
    )

    for head, stats in summary.items():
        print(
            f"head={head} "
            f"active_routing="
            f"{stats['active_routing']:+.3f} "
            f"passive_routing="
            f"{stats['passive_routing']:+.3f} "
            f"swap_harm={stats['harm']:+.3f} "
            f"swap_acc={stats['swap_accuracy']:.3f} "
            f"control_acc={stats['control_accuracy']:.3f}"
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

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    ) if args.device == "auto" else args.device

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    print(
        f"ML-044 phase={args.phase} "
        f"seed={args.seed} "
        f"device={device} "
        f"design={DESIGN_HASH}"
    )

    run(args.seed, args.phase, device)


if __name__ == "__main__":
    main()
