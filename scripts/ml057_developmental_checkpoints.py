
"""ML-057: Developmental checkpoints of a scratch Transformer.

Uses the architecture and examples from ML-037.
Creates a new calibration split exclusively from ML-037's old
training data. Does NOT evaluate ML-037's historical holdout.

Descriptive study only; no causal mechanism is assumed.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import (
    ScratchTransformer,
    encode_batch,
    interaction_trajectory,
    split_examples,
)

CHECKPOINTS = (0, 10, 25, 50, 75, 100, 150, 200, 250, 300)
SPLIT_SEED = 5701


def case_key(e):
    return "|".join((
        *e.pair, e.verb, str(e.direction), e.voice, e.role
    ))


def make_split():
    old_train, _old_holdout, _ = split_examples()

    pairs = sorted({e.pair for e in old_train})
    assert len(pairs) == 21

    random.Random(SPLIT_SEED).shuffle(pairs)
    calibration_pairs = set(pairs[:4])

    train = [
        e for e in old_train
        if e.pair not in calibration_pairs
    ]
    calibration = [
        e for e in old_train
        if e.pair in calibration_pairs
    ]

    assert len(train) == 544
    assert len(calibration) == 128

    assert (
        {e.pair for e in train}
        .isdisjoint({e.pair for e in calibration})
    )

    return train, calibration, sorted(calibration_pairs)


@torch.inference_mode()
def assess(model, examples, device):
    model.eval()
    records = []

    for start in range(0, len(examples), 128):
        batch = examples[start:start + 128]
        ids, pos, targets, choices = encode_batch(batch)

        logits = model(
            ids.to(device), pos.to(device)
        ).float().cpu()

        choice_logits = logits.gather(1, choices)

        predictions = choices.gather(
            1, choice_logits.argmax(1, keepdim=True)
        ).squeeze(1)

        for i, example in enumerate(batch):
            gold = int(targets[i])
            competitor = int(choices[i, 0])

            if competitor == gold:
                competitor = int(choices[i, 1])

            assert gold != competitor

            records.append({
                "case": case_key(example),
                "pair_correct": bool(
                    predictions[i].item() == gold
                ),
                "full_vocab_correct": bool(
                    logits[i].argmax().item() == gold
                ),
                "signed_margin": float(
                    (logits[i, gold] -
                     logits[i, competitor]).item()
                ),
                "voice": example.voice,
                "role": example.role,
            })

    def accuracy(subset):
        return sum(r["pair_correct"] for r in subset) / len(subset)

    return {
        "n": len(records),
        "pair_accuracy": accuracy(records),
        "full_vocab_accuracy": (
            sum(r["full_vocab_correct"] for r in records)
            / len(records)
        ),
        "mean_signed_margin": (
            sum(r["signed_margin"] for r in records)
            / len(records)
        ),
        "by_voice_role": {
            f"{voice}_{role}": accuracy([
                r for r in records
                if r["voice"] == voice and r["role"] == role
            ])
            for voice in ("active", "passive")
            for role in ("agent", "patient")
        },
        "cases": records,
    }


def run_seed(seed, train, calibration, cal_pairs, args, split_hash):
    torch.manual_seed(seed)

    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use --device cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    model = ScratchTransformer(
        width=96, heads=4, layers=2
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    ids, positions, labels, _ = encode_batch(train)

    generator = torch.Generator(device="cpu").manual_seed(
        seed + 100_000
    )

    output = Path(args.out) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    history = []
    last_loss = None

    def snapshot(step):
        train_stats = assess(model, train, device)
        cal_stats = assess(model, calibration, device)

        # Descriptive activation interaction, NOT a causal circuit.
        interaction = interaction_trajectory(
            model, cal_pairs, device
        )

        record = {
            "step": step,
            "train": train_stats,
            "calibration": cal_stats,
            "interaction_relative_norm_by_depth": interaction,
            "last_train_loss": last_loss,
        }
        history.append(record)

        torch.save({
            "step": step,
            "seed": seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "split_hash": split_hash,
            "architecture": "ML037 ScratchTransformer",
        }, output / f"checkpoint_{step:04d}.pt")

        (output / "trajectory.json").write_text(
            json.dumps({
                "experiment": "ML-057",
                "seed": seed,
                "split_hash": split_hash,
                "training_steps": args.steps,
                "history": history,
                "claim_level": (
                    "Descriptive only; no causal mechanism established"
                ),
            }, indent=2)
        )

        print(
            f"seed={seed} step={step:4d} "
            f"train={train_stats['pair_accuracy']:.3f} "
            f"cal={cal_stats['pair_accuracy']:.3f} "
            f"margin={cal_stats['mean_signed_margin']:+.3f} "
            f"interaction={interaction}",
            flush=True,
        )

    snapshot(0)

    for step in range(1, args.steps + 1):
        model.train()

        picks = torch.randint(
            len(train),
            (args.batch_size,),
            generator=generator,
        )

        loss = F.cross_entropy(
            model(
                ids[picks].to(device),
                positions[picks].to(device),
            ),
            labels[picks].to(device),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu())

        if step in CHECKPOINTS:
            snapshot(step)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 17]
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto"
    )
    parser.add_argument(
        "--out", default="results/ml057_development"
    )

    args = parser.parse_args()

    if args.steps != 300:
        parser.error(
            "First experiment uses a fixed 300-step protocol."
        )

    torch.set_num_threads(min(torch.get_num_threads(), 4))

    train, calibration, cal_pairs = make_split()

    manifest = {
        "experiment": "ML-057",
        "source": "ML-037 original training examples only",
        "split_seed": SPLIT_SEED,
        "train_n": len(train),
        "calibration_n": len(calibration),
        "train_case_keys": sorted(map(case_key, train)),
        "calibration_case_keys": sorted(
            map(case_key, calibration)
        ),
        "checkpoints": CHECKPOINTS,
        "architecture": {
            "layers": 2, "heads": 4, "width": 96
        },
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    split_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    (output / "manifest.json").write_text(
        json.dumps({
            **manifest, "sha256": split_hash
        }, indent=2)
    )

    print(
        f"ML-057 | train={len(train)} "
        f"calibration={len(calibration)} "
        f"manifest={split_hash[:12]}"
    )
    print("Historical ML-037 holdout is NOT evaluated.")

    for seed in args.seeds:
        run_seed(
            seed, train, calibration,
            cal_pairs, args, split_hash
        )

    print(
        "Finished. Send manifest.json and trajectory.json "
        "for each seed. Keep the .pt checkpoints locally."
    )


if __name__ == "__main__":
    main()
