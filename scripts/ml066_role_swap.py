
"""ML-066: Counterfactual role transfer through layer-1 attention.

Frozen ML-057 step-300 models.
Uses historically examined ML-037 held-out participant pairs.

Interchanges only a layer-1 attention output, keeping token
embeddings and the rest of the target prompt unchanged.

Tests whether restoring layer-2 Q or V mediates the effect.
Same-role, different-verb interchange is a control.

Exploratory: the evaluation pairs are not pristine.
"""

import argparse
import json
from pathlib import Path

import torch

from ml037_scratch_transformer_pilot import (
    Example,
    ScratchTransformer,
    VERBS,
    VOCAB,
    encode_batch,
    split_examples,
)


SEEDS = (23, 29, 31, 37, 41, 43)
SITES = ("readout", "role_cue")


def swap_role(e):
    role = (
        "patient"
        if e.role == "agent"
        else "agent"
    )

    return Example(
        e.pair,
        e.verb,
        e.direction,
        e.voice,
        role,
    )


def swap_verb(e):
    index = VERBS.index(e.verb)
    verb = VERBS[
        (index + 1) % len(VERBS)
    ]

    return Example(
        e.pair,
        verb,
        e.direction,
        e.voice,
        e.role,
    )


def token_positions(ids, readout):
    positions = torch.arange(
        ids.size(1),
        device=ids.device,
    )[None, :]

    role = (
        (ids == VOCAB["acted"])
        | (ids == VOCAB["received"])
    ) & (
        positions < readout[:, None]
    )

    assert torch.all(role.sum(1) == 1)

    return {
        "role_cue": role.long().argmax(1),
        "readout": readout,
    }


def capture(model, ids, readout):
    captured = {}

    def attn1_hook(_module, _inputs, result):
        captured["attn1"] = (
            result[0].detach().clone()
        )

    def attn2_pre(_module, args):
        captured["attn2_inputs"] = tuple(
            tensor.detach().clone()
            for tensor in args[:3]
        )

    h1 = (
        model.blocks[0]
        .attn.register_forward_hook(attn1_hook)
    )

    h2 = (
        model.blocks[1]
        .attn.register_forward_pre_hook(attn2_pre)
    )

    try:
        logits = model(ids, readout)
    finally:
        h2.remove()
        h1.remove()

    assert set(captured) == {
        "attn1",
        "attn2_inputs",
    }

    return logits, captured


def intervene(
    model,
    ids,
    readout,
    donor_update,
    site,
    pos,
    original_l2,
    repair,
):
    batch = torch.arange(
        ids.size(0),
        device=ids.device,
    )

    def l1_patch(_module, _inputs, result):
        attention_output = result[0].clone()

        attention_output[batch, pos] = (
            donor_update[batch, pos]
        )

        return (
            attention_output,
            *result[1:],
        )

    def l2_repair(_module, args):
        q, k, v = args[:3]

        clean_q, clean_k, clean_v = original_l2

        q, k, v = (
            q.clone(),
            k.clone(),
            v.clone(),
        )

        if repair == "Q":
            q[batch, pos] = clean_q[batch, pos]

        elif repair == "V":
            v[batch, pos] = clean_v[batch, pos]

        elif repair == "KV":
            k[batch, pos] = clean_k[batch, pos]
            v[batch, pos] = clean_v[batch, pos]

        elif repair != "none":
            raise ValueError(repair)

        return (q, k, v) + tuple(args[3:])

    h1 = (
        model.blocks[0]
        .attn.register_forward_hook(l1_patch)
    )

    h2 = (
        model.blocks[1]
        .attn.register_forward_pre_hook(l2_repair)
    )

    try:
        logits = model(ids, readout)
    finally:
        h2.remove()
        h1.remove()

    return logits


def answer_margins(logits, labels, choices):
    candidate_scores = (
        logits.float().gather(1, choices)
    )

    gold = (
        choices == labels[:, None]
    ).long().argmax(1)

    assert torch.all(
        choices.gather(
            1,
            gold[:, None],
        ).flatten() == labels
    )

    correct = candidate_scores.gather(
        1,
        gold[:, None],
    ).flatten()

    incorrect = candidate_scores.gather(
        1,
        (1 - gold)[:, None],
    ).flatten()

    return (
        correct - incorrect
    ).detach().cpu()


def summarize(original, affected, eligible, pairs):
    delta = original - affected

    assert eligible.any()

    by_pair = {}

    for name in sorted(set(pairs)):
        keep = (
            torch.tensor([
                pair == name
                for pair in pairs
            ])
            & eligible
        )

        by_pair[name] = {
            "eligible": int(keep.sum()),
            "mean_target_margin_drop": (
                float(delta[keep].mean())
                if keep.any()
                else None
            ),
            "role_flip_rate": (
                float(
                    (affected[keep] < 0)
                    .float().mean()
                )
                if keep.any()
                else None
            ),
        }

    return {
        "eligible": int(eligible.sum()),
        "mean_target_margin_drop": float(
            delta[eligible].mean()
        ),
        "role_flip_rate": float(
            (affected[eligible] < 0)
            .float().mean()
        ),
        "by_pair": by_pair,
    }


@torch.inference_mode()
def evaluate(model, examples, device):
    target_ids, target_readout, labels, choices = [
        x.to(device)
        for x in encode_batch(examples)
    ]

    opposite_examples = [
        swap_role(e)
        for e in examples
    ]

    control_examples = [
        swap_verb(e)
        for e in examples
    ]

    opposite_ids, opposite_readout, opposite_labels, _ = [
        x.to(device)
        for x in encode_batch(opposite_examples)
    ]

    control_ids, control_readout, control_labels, _ = [
        x.to(device)
        for x in encode_batch(control_examples)
    ]

    assert (
        target_ids.shape
        == opposite_ids.shape
        == control_ids.shape
    )

    assert torch.equal(
        target_readout,
        opposite_readout,
    )

    assert torch.equal(
        target_readout,
        control_readout,
    )

    assert torch.all(
        labels != opposite_labels
    )

    assert torch.equal(
        labels,
        control_labels,
    )

    logits, clean = capture(
        model,
        target_ids,
        target_readout,
    )

    opposite_logits, opposite = capture(
        model,
        opposite_ids,
        opposite_readout,
    )

    control_logits, control = capture(
        model,
        control_ids,
        control_readout,
    )

    target_margin = answer_margins(
        logits,
        labels,
        choices,
    )

    donor_margin = answer_margins(
        opposite_logits,
        labels,
        choices,
    )

    ctrl_margin = answer_margins(
        control_logits,
        labels,
        choices,
    )

    # Analyze examples where both the target and its
    # opposite-role donor originally answer correctly.
    eligible = (
        (target_margin > 0)
        & (donor_margin < 0)
    )

    assert int(eligible.sum()) >= 10, (
        "Insufficient jointly correct cases"
    )

    positions = token_positions(
        target_ids,
        target_readout,
    )

    pairs = [
        "|".join(e.pair)
        for e in examples
    ]

    result = {
        "baseline_accuracy": float(
            (target_margin > 0)
            .float().mean()
        ),
        "opposite_role_accuracy": float(
            (donor_margin < 0)
            .float().mean()
        ),
        "same_role_other_verb_accuracy": float(
            (ctrl_margin > 0)
            .float().mean()
        ),
        "jointly_correct_cases": int(
            eligible.sum()
        ),
        "sites": {},
    }

    for site in SITES:
        pos = positions[site]

        repair = (
            "Q"
            if site == "readout"
            else "V"
        )

        # Identity control: swapping an attention output
        # with itself must preserve predictions.
        same_logits = intervene(
            model,
            target_ids,
            target_readout,
            clean["attn1"],
            site,
            pos,
            clean["attn2_inputs"],
            "none",
        )

        error = float(
            (same_logits - logits).abs().max()
        )

        assert error < 2e-3, (
            f"Identity patch failed at {site}: {error}"
        )

        per_site = {}

        for name, donor in (
            ("role_flip", opposite),
            ("verb_control", control),
        ):
            shifted = intervene(
                model,
                target_ids,
                target_readout,
                donor["attn1"],
                site,
                pos,
                clean["attn2_inputs"],
                "none",
            )

            repaired = intervene(
                model,
                target_ids,
                target_readout,
                donor["attn1"],
                site,
                pos,
                clean["attn2_inputs"],
                repair,
            )

            shifted_margin = answer_margins(
                shifted,
                labels,
                choices,
            )

            repaired_margin = answer_margins(
                repaired,
                labels,
                choices,
            )

            stats = summarize(
                target_margin,
                shifted_margin,
                eligible,
                pairs,
            )

            stats["after_route_repair"] = summarize(
                target_margin,
                repaired_margin,
                eligible,
                pairs,
            )

            stats[
                "mean_margin_restored_by_route_repair"
            ] = float(
                (
                    repaired_margin[eligible]
                    - shifted_margin[eligible]
                ).mean()
            )

            per_site[name] = stats

        result["sites"][site] = {
            "identity_control_max_abs_logit_error": (
                error
            ),
            "route_repair": repair,
            **per_site,
        }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/ml057_development",
    )

    parser.add_argument(
        "--frozen",
        default="reports/ml065/frozen_predictions.json",
    )

    parser.add_argument(
        "--out",
        default="results/ml066_role_swap.json",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    args = parser.parse_args()

    frozen = json.loads(
        Path(args.frozen).read_text()
    )

    assert set(
        map(int, frozen["predictions"])
    ) == set(SEEDS)

    assert frozen["checkpoint_step"] == 300

    _, historical, pairs = split_examples()

    assert len(historical) == 224
    assert len(pairs) == 7

    device = args.device

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    if device == "mps":
        assert torch.backends.mps.is_available()

    if device == "cuda":
        assert torch.cuda.is_available()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    report = {
        "experiment": "ML-066",
        "data_status": (
            "ML-037 historical holdout; not pristine"
        ),
        "purpose": (
            "Counterfactual interchange of L1 attention "
            "updates, controlling embedding differences"
        ),
        "seeds": list(SEEDS),
        "step": 300,
        "runs": [],
    }

    output = Path(args.out)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing "
            f"ML-066 results: {output}"
        )

    for seed in SEEDS:
        checkpoint = torch.load(
            Path(args.root)
            / f"seed_{seed}"
            / "checkpoint_0300.pt",
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["seed"] == seed
        assert checkpoint["step"] == 300

        assert (
            checkpoint["split_hash"]
            == frozen["source_manifest_hash"]
        )

        model = ScratchTransformer(
            width=96,
            heads=4,
            layers=2,
        ).to(device)

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        model.eval()

        result = evaluate(
            model,
            historical,
            device,
        )

        report["runs"].append({
            "seed": seed,
            "previous_route": (
                frozen["predictions"][
                    str(seed)
                ]["prediction"]
            ),
            **result,
        })

        output.write_text(
            json.dumps(report, indent=2) + "\n"
        )

        parts = []

        for site in SITES:
            row = result["sites"][site]

            transfer = row["role_flip"]
            control = row["verb_control"]

            parts.append(
                f"{site}: "
                f"flip={transfer['role_flip_rate']:.3f} "
                f"margin_drop="
                f"{transfer['mean_target_margin_drop']:+.2f} "
                f"rescue="
                f"{transfer['mean_margin_restored_by_route_repair']:+.2f} "
                f"verb_ctrl_flip="
                f"{control['role_flip_rate']:.3f}"
            )

        print(
            f"seed={seed} "
            f"prior="
            f"{frozen['predictions'][str(seed)]['prediction']} "
            f"acc={result['baseline_accuracy']:.3f} "
            f"eligible={result['jointly_correct_cases']} | "
            + " | ".join(parts),
            flush=True,
        )

    print("\nSaved:", output)
    print(
        "Historically examined evaluation set; "
        "exploratory causal interchange."
    )


if __name__ == "__main__":
    main()
