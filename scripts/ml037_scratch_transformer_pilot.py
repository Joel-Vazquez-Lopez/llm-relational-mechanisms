
"""ML-037: minimal from-scratch causal Transformer relational-learning pilot.

No pretrained weights/tokenizer, no role-specific neural modules.
Train on active/passive sentences; evaluate unseen unordered name pairs.
Track voice x question readout activation interaction at initialization
and after training. Descriptive interactions are NOT causal circuits.
"""
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

NAMES = ("Maya", "Lena", "Omar", "Rosa", "Nora", "Clara", "Felix", "Iris")
VERBS = ("helped", "praised", "thanked", "followed")
VOICES = ("active", "passive")
ROLES = ("agent", "patient")

TOKENS = tuple(dict.fromkeys(
    ("<pad>", "<bos>", "<sep>", "<answer>", ".", "was", "by",
     "who", "acted", "received", "?") + NAMES + VERBS
))
VOCAB = {token: i for i, token in enumerate(TOKENS)}
PAD = VOCAB["<pad>"]


@dataclass(frozen=True)
class Example:
    pair: tuple[str, str]
    verb: str
    direction: int
    voice: str
    role: str


def prompt(example):
    a, b = (
        example.pair if example.direction == 0
        else example.pair[::-1]
    )

    if example.voice == "active":
        event = [a, example.verb, b, "."]
    else:
        event = [b, "was", example.verb, "by", a, "."]

    query = "acted" if example.role == "agent" else "received"

    return (
        ["<bos>"] + event
        + ["<sep>", "who", query, "?", "<answer>"]
    )


def answer(example):
    a, b = (
        example.pair if example.direction == 0
        else example.pair[::-1]
    )
    return a if example.role == "agent" else b


def split_examples():
    # Split is fixed independently of the training seed.
    pairs = [
        (NAMES[i], NAMES[j])
        for i in range(len(NAMES))
        for j in range(i + 1, len(NAMES))
    ]

    random.Random(1729).shuffle(pairs)

    heldout = set(pairs[:7])
    train, test = [], []

    for pair in pairs:
        for verb in VERBS:
            for direction in (0, 1):
                for voice in VOICES:
                    for role in ROLES:
                        example = Example(
                            pair, verb, direction, voice, role
                        )
                        (test if pair in heldout else train).append(example)

    return train, test, pairs[:7]


def encode_batch(examples):
    seqs = [
        [VOCAB[t] for t in prompt(e)]
        for e in examples
    ]

    max_len = max(map(len, seqs))

    ids = torch.full(
        (len(seqs), max_len), PAD, dtype=torch.long
    )

    readout = torch.tensor(
        [len(seq) - 1 for seq in seqs],
        dtype=torch.long
    )

    labels = torch.tensor(
        [VOCAB[answer(e)] for e in examples],
        dtype=torch.long
    )

    choices = torch.tensor(
        [[VOCAB[n] for n in e.pair] for e in examples],
        dtype=torch.long
    )

    for idx, seq in enumerate(seqs):
        ids[idx, :len(seq)] = torch.tensor(seq)

    # Padding comes after the readout.
    # Causal masking prevents it from influencing the answer.
    return ids, readout, labels, choices


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(width)

        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x, causal_mask):
        z = self.norm1(x)

        x = x + self.attn(
            z, z, z,
            attn_mask=causal_mask,
            need_weights=False
        )[0]

        return x + self.mlp(self.norm2(x))


class ScratchTransformer(nn.Module):
    def __init__(
        self, width=96, heads=4,
        layers=2, max_len=32
    ):
        super().__init__()

        self.embed = nn.Embedding(len(VOCAB), width)
        self.pos = nn.Embedding(max_len, width)

        self.blocks = nn.ModuleList([
            Block(width, heads)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, len(VOCAB))

    def forward(
        self, ids, readout, return_states=False
    ):
        batch, length = ids.shape

        x = self.embed(ids) + self.pos(
            torch.arange(length, device=ids.device)
        )

        idx = torch.arange(batch, device=ids.device)

        states = (
            [x[idx, readout]]
            if return_states else None
        )

        mask = torch.ones(
            length, length,
            dtype=torch.bool,
            device=ids.device
        ).triu(1)

        for block in self.blocks:
            x = block(x, mask)

            if return_states:
                states.append(x[idx, readout])

        logits = self.output(
            self.norm(x[idx, readout])
        )

        if return_states:
            return logits, torch.stack(states)

        return logits


@torch.no_grad()
def evaluate(model, examples, device):
    model.eval()

    buckets = {
        (v, r): [0, 0]
        for v in VOICES
        for r in ROLES
    }

    total_full = 0
    total_pair = 0

    for start in range(0, len(examples), 128):
        chunk = examples[start:start + 128]

        ids, pos, targets, choices = encode_batch(chunk)

        scores = model(
            ids.to(device),
            pos.to(device)
        ).cpu()

        full = scores.argmax(-1) == targets

        allowed = scores.gather(1, choices)

        candidate = choices.gather(
            1,
            allowed.argmax(-1, keepdim=True)
        ).squeeze(1)

        correct = candidate == targets

        total_full += int(full.sum())
        total_pair += int(correct.sum())

        for j, example in enumerate(chunk):
            cell = buckets[example.voice, example.role]
            cell[0] += int(correct[j])
            cell[1] += 1

    return {
        "full_vocab_accuracy": round(
            total_full / len(examples), 4
        ),
        "pairwise_accuracy": round(
            total_pair / len(examples), 4
        ),
        "by_voice_role": {
            f"{v}_{r}": round(ok / n, 4)
            for (v, r), (ok, n) in buckets.items()
        },
    }


@torch.no_grad()
def interaction_trajectory(
    model, heldout_pairs, device
):
    model.eval()
    trajectories = []

    # Eight event identities from held-out name pairs.
    for pair in heldout_pairs[:2]:
        for verb in VERBS[:2]:
            for direction in (0, 1):

                quartet = [
                    Example(
                        pair, verb, direction,
                        voice, role
                    )
                    for voice in VOICES
                    for role in ROLES
                ]

                ids, positions, _, _ = encode_batch(quartet)

                _, states = model(
                    ids.to(device),
                    positions.to(device),
                    return_states=True
                )

                # Order:
                # active-agent, active-patient,
                # passive-agent, passive-patient
                delta = (
                    states[:, 3]
                    - states[:, 2]
                    - states[:, 1]
                    + states[:, 0]
                )

                base_norm = (
                    states.norm(dim=-1)
                    .mean(dim=1)
                    .clamp_min(1e-8)
                )

                trajectories.append(
                    (delta.norm(dim=-1) / base_norm).cpu()
                )

    return [
        round(x, 5)
        for x in torch.stack(
            trajectories
        ).mean(dim=0).tolist()
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
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

    train, test, heldout = split_examples()

    assert len(train) == 672
    assert len(test) == 224

    print(
        f"ML-037 | random-init Transformer | "
        f"device={device} | seed={args.seed}"
    )

    print(
        f"Train: {len(train)} / "
        f"test unseen pairs: {len(test)}"
    )

    print("Held-out unordered pairs:", heldout)

    model = ScratchTransformer().to(device)

    train_ids, train_pos, train_y, _ = encode_batch(train)

    init_train = evaluate(model, train, device)
    init_test = evaluate(model, test, device)

    init_interaction = interaction_trajectory(
        model, heldout, device
    )

    history = [{
        "step": 0,
        "train": init_train,
        "heldout": init_test,
        "interaction": init_interaction,
    }]

    print(
        f"step=   0 "
        f"heldout_pairwise={init_test['pairwise_accuracy']:.3f} "
        f"interaction={init_interaction}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    for step in range(1, args.steps + 1):
        model.train()

        picks = torch.randint(
            len(train),
            (args.batch_size,)
        )

        ids = train_ids[picks].to(device)
        pos = train_pos[picks].to(device)
        target = train_y[picks].to(device)

        loss = F.cross_entropy(
            model(ids, pos),
            target
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (
            step in (25, 50, 100, 150, 200)
            or step == args.steps
        ):
            tr = evaluate(model, train, device)
            te = evaluate(model, test, device)

            inter = interaction_trajectory(
                model, heldout, device
            )

            history.append({
                "step": step,
                "train": tr,
                "heldout": te,
                "interaction": inter,
            })

            print(
                f"step={step:4} "
                f"loss={loss.item():.3f} "
                f"heldout_pairwise={te['pairwise_accuracy']:.3f} "
                f"interaction={inter}"
            )

    final_train = evaluate(model, train, device)
    final_test = evaluate(model, test, device)

    final_interaction = interaction_trajectory(
        model, heldout, device
    )

    result = {
        "experiment": "ML-037",
        "seed": args.seed,
        "steps": args.steps,
        "architecture": {
            "causal": True,
            "layers": 2,
            "heads": 4,
            "width": 96,
        },
        "split_seed": 1729,
        "heldout_unordered_pairs": heldout,
        "init_train": init_train,
        "init_test": init_test,
        "trained_train": final_train,
        "trained_test": final_test,
        "training_history": history,
        "interaction_relative_norm_at_init": init_interaction,
        "interaction_relative_norm_after_training": final_interaction,
        "interpretation": (
            "Descriptive activation geometry; "
            "no causal mechanism established"
        ),
    }

    Path("results").mkdir(exist_ok=True)

    output = Path(
        f"results/ml037_scratch_seed{args.seed}.json"
    )

    output.write_text(
        json.dumps(result, indent=2)
    )

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "result": result,
            "vocab": VOCAB,
        },
        f"results/ml037_scratch_seed{args.seed}.pt"
    )

    print(
        "\nFinal heldout pairwise accuracy:",
        final_test["pairwise_accuracy"]
    )

    print(
        "Heldout accuracy by condition:",
        final_test["by_voice_role"]
    )

    print(
        "Interaction norm "
        "(embeddings, block 1, block 2):"
    )

    print("  init   :", init_interaction)
    print("  trained:", final_interaction)

    print("Saved:", output)

    print(
        "Descriptive pilot only: successful accuracy "
        "and growing interaction do not identify "
        "a grammatical circuit."
    )


if __name__ == "__main__":
    main()
