
"""ML-072: Controlled compositional exposure.

Three phases:
    plan -> train -> evaluate

Paired initializations and identical base minibatches across arms.
No neural-mechanism claims are made in this experiment.
"""

import argparse
import hashlib
import itertools
import json
import random
import statistics

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


NAMES = (
    "Ada", "Ben", "Cora", "Davi", "Ema",
    "Finn", "Gia", "Hugo", "Inez", "Jules",
)

VERBS = (
    "helped", "praised", "thanked",
    "followed", "noticed", "visited",
)

SEEDS = (211, 223, 227, 229, 233, 239)
ARMS = ("control", "sparse", "broad")

KNOWN = (
    ("active", "standard"),
    ("active", "alternate"),
    ("passive", "standard"),
)

TARGET = (("passive", "alternate"),)

VOCAB_WORDS = tuple(dict.fromkeys(
    (
        "<pad>", "<bos>", "<sep>", "<answer>",
        ".", "was", "by", "who", "acted",
        "received", "?", "identify", "agent",
        "patient",
    ) + NAMES + VERBS
))

VOCAB = {
    word: i
    for i, word in enumerate(VOCAB_WORDS)
}


@dataclass(frozen=True)
class Case:
    pair: tuple[str, str]
    verb: str
    direction: int
    voice: str
    style: str
    role: str


def make_cases(pairs, verbs, constructions):
    return [
        Case(pair, verb, direction, voice, style, role)
        for pair in pairs
        for verb in verbs
        for direction in (0, 1)
        for voice, style in constructions
        for role in ("agent", "patient")
    ]


def event_key(case):
    return (
        case.pair,
        case.verb,
        case.direction,
        case.voice,
        case.style,
    )


def sentence(case):
    agent, patient = (
        case.pair
        if case.direction == 0
        else case.pair[::-1]
    )

    if case.voice == "active":
        event = [
            agent, case.verb, patient, "."
        ]
    else:
        event = [
            patient, "was", case.verb,
            "by", agent, "."
        ]

    if case.style == "standard":
        cue = (
            "acted"
            if case.role == "agent"
            else "received"
        )
        query = ["who", cue, "?"]
    else:
        query = [
            "identify", case.role, "?"
        ]

    return (
        ["<bos>"]
        + event
        + ["<sep>"]
        + query
        + ["<answer>"]
    )


def encode(cases):
    prompts = [
        sentence(case)
        for case in cases
    ]

    readout = torch.tensor(
        [len(p) - 1 for p in prompts],
        dtype=torch.long,
    )

    ids = torch.full(
        (
            len(cases),
            max(map(len, prompts)),
        ),
        VOCAB["<pad>"],
        dtype=torch.long,
    )

    for i, prompt in enumerate(prompts):
        ids[i, :len(prompt)] = torch.tensor(
            [VOCAB[token] for token in prompt]
        )

    labels = []

    for case in cases:
        agent, patient = (
            case.pair
            if case.direction == 0
            else case.pair[::-1]
        )

        answer = (
            agent
            if case.role == "agent"
            else patient
        )

        labels.append(VOCAB[answer])

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    choices = torch.tensor(
        [
            [VOCAB[name] for name in case.pair]
            for case in cases
        ],
        dtype=torch.long,
    )

    return ids, readout, labels, choices


class Block(nn.Module):
    def __init__(self, width=96, heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(width)

        self.attn = nn.MultiheadAttention(
            width,
            heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(width)

        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x, mask):
        z = self.norm1(x)

        x = x + self.attn(
            z, z, z,
            attn_mask=mask,
            need_weights=False,
        )[0]

        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(
            len(VOCAB), 96
        )

        self.pos = nn.Embedding(32, 96)

        self.blocks = nn.ModuleList([
            Block(),
            Block(),
        ])

        self.norm = nn.LayerNorm(96)

        self.output = nn.Linear(
            96, len(VOCAB)
        )

    def forward(self, ids, readout):
        batch, length = ids.shape

        positions = torch.arange(
            length,
            device=ids.device,
        )

        x = (
            self.embed(ids)
            + self.pos(positions)
        )

        causal = torch.ones(
            length,
            length,
            device=ids.device,
            dtype=torch.bool,
        ).triu(1)

        for block in self.blocks:
            x = block(x, causal)

        selected = x[
            torch.arange(
                batch,
                device=ids.device,
            ),
            readout,
        ]

        return self.output(
            self.norm(selected)
        )


def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def experiment(args):
    pairs = list(
        itertools.combinations(NAMES, 2)
    )

    random.Random(7201).shuffle(pairs)

    train_pairs = pairs[:27]
    cal_pairs = pairs[27:36]
    test_pairs = pairs[36:]

    sparse_pairs = train_pairs[:4]
    broad_pairs = train_pairs[:18]

    sparse_verbs = VERBS[:2]
    broad_verbs = VERBS[:4]
    novel_verbs = VERBS[4:]

    base = make_cases(
        train_pairs,
        VERBS,
        KNOWN,
    )

    calibration = make_cases(
        cal_pairs,
        VERBS,
        KNOWN,
    )

    exposure = {
        "control": make_cases(
            sparse_pairs,
            sparse_verbs,
            (("passive", "standard"),),
        ),
        "sparse": make_cases(
            sparse_pairs,
            sparse_verbs,
            TARGET,
        ),
        "broad": make_cases(
            broad_pairs,
            broad_verbs,
            TARGET,
        ),
    }

    tests = {
        "known_grammar_new_pairs": make_cases(
            test_pairs, VERBS, KNOWN
        ),
        "target_exposure_cells": make_cases(
            sparse_pairs, sparse_verbs, TARGET
        ),
        "target_train_pairs_new_verbs": make_cases(
            train_pairs, novel_verbs, TARGET
        ),
        "target_new_pairs_seen_verbs": make_cases(
            test_pairs, broad_verbs, TARGET
        ),
        "target_double_unseen": make_cases(
            test_pairs, novel_verbs, TARGET
        ),
    }

    assert len(base) == 1944
    assert len(calibration) == 648

    assert [
        len(exposure[arm])
        for arm in ARMS
    ] == [32, 32, 288]

    assert len(
        tests["target_double_unseen"]
    ) == 72

    assert not set(train_pairs) & set(cal_pairs)
    assert not set(train_pairs) & set(test_pairs)
    assert not set(cal_pairs) & set(test_pairs)

    assert {
        name
        for pair in train_pairs
        for name in pair
    } == set(NAMES)

    assert all(
        not (
            c.voice == "passive"
            and c.style == "alternate"
        )
        for c in base
    )

    assert all(
        c.verb not in novel_verbs
        for pool in exposure.values()
        for c in pool
        if (
            c.voice == "passive"
            and c.style == "alternate"
        )
    )

    assert all(
        c.pair not in train_pairs
        for c in tests["target_double_unseen"]
    )

    document = {
        "experiment": "ML-072",
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "names": list(NAMES),
        "verbs": list(VERBS),
        "train_pairs": train_pairs,
        "calibration_pairs": cal_pairs,
        "test_pairs": test_pairs,
        "sparse_exposure_pairs": sparse_pairs,
        "broad_exposure_pairs": broad_pairs,
        "sparse_exposure_verbs": sparse_verbs,
        "broad_exposure_verbs": broad_verbs,
        "target_never_exposed_verbs": novel_verbs,
        "base_constructions": KNOWN,
        "target_construction": TARGET,
        "steps": args.steps,
        "batch_size": 64,
        "base_per_batch": 56,
        "exposure_per_batch": 8,
        "learning_rate": 0.002,
        "primary_outcome": (
            "target_double_unseen: "
            "correct_both_rate and pair_accuracy"
        ),
        "predeclared_comparisons": [
            "sparse-control",
            "broad-control",
            "broad-sparse",
        ],
        "hypotheses": [
            "Sparse exposure may repair familiar "
            "composition cells without broad transfer",
            "Broad exposure may improve generalization "
            "to unseen pairs and target-composition verbs",
            "Role-signal direction may differ from "
            "pairwise answer accuracy",
        ],
        "controls": (
            "Both question styles have equal length "
            "and the role cue at the same position"
        ),
        "limits": (
            "Synthetic grammar; six paired seeds; "
            "designed after exploratory ML-069 findings"
        ),
    }

    return (
        document,
        digest(document),
        base,
        calibration,
        exposure,
        tests,
    )


def check_plan(path, document, sha):
    if not path.exists():
        raise FileNotFoundError(
            f"Protocol not frozen: {path}. "
            "Run --phase plan first."
        )

    saved = json.loads(path.read_text())

    assert saved.get("sha256") == sha, (
        "Frozen protocol changed"
    )

    core = dict(saved)
    core.pop("sha256")

    assert core == json.loads(
        json.dumps(document)
    ), "Protocol or script changed"


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA unavailable")

    if (
        device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS unavailable")

    return device


@torch.inference_mode()
def measure(model, cases, device):
    model.eval()

    events = defaultdict(dict)
    pair_correct = 0
    full_correct = 0

    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]

        ids, readout, labels, choices = [
            tensor.to(device)
            for tensor in encode(batch)
        ]

        logits = model(
            ids, readout
        ).float()

        candidate_logits = logits.gather(
            1, choices
        )

        predicted = choices.gather(
            1,
            candidate_logits.argmax(
                1, keepdim=True
            ),
        ).flatten()

        pair_correct += int(
            (predicted == labels).sum()
        )

        full_correct += int(
            (logits.argmax(1) == labels).sum()
        )

        for i, case in enumerate(batch):
            agent, patient = (
                case.pair
                if case.direction == 0
                else case.pair[::-1]
            )

            preference = float(
                (
                    logits[i, VOCAB[agent]]
                    - logits[i, VOCAB[patient]]
                ).item()
            )

            key = event_key(case)

            assert case.role not in events[key]

            events[key][case.role] = (
                preference,
                int(predicted[i].item()),
                VOCAB[agent],
                VOCAB[patient],
            )

    counts = {
        "correct_both": 0,
        "same_answer": 0,
        "reversed_both": 0,
    }

    role_positive = 0
    role_deltas = []

    for roles in events.values():
        assert set(roles) == {
            "agent", "patient"
        }

        agent = roles["agent"]
        patient = roles["patient"]

        assert agent[2:] == patient[2:]

        delta = (
            agent[0] - patient[0]
        )

        role_deltas.append(delta)

        role_positive += delta > 0

        if (
            agent[1] == agent[2]
            and patient[1] == agent[3]
        ):
            counts["correct_both"] += 1

        elif agent[1] == patient[1]:
            counts["same_answer"] += 1

        else:
            assert (
                agent[1] == agent[3]
                and patient[1] == agent[2]
            )

            counts["reversed_both"] += 1

    n = len(events)

    assert 2 * n == len(cases)
    assert sum(counts.values()) == n

    expected_accuracy = (
        2 * counts["correct_both"]
        + counts["same_answer"]
    ) / (2 * n)

    assert abs(
        pair_correct / len(cases)
        - expected_accuracy
    ) < 1e-10

    return {
        "questions": len(cases),
        "events": n,
        "pair_accuracy": (
            pair_correct / len(cases)
        ),
        "full_vocab_accuracy": (
            full_correct / len(cases)
        ),
        "correct_both_rate": (
            counts["correct_both"] / n
        ),
        "same_answer_rate": (
            counts["same_answer"] / n
        ),
        "reversed_both_rate": (
            counts["reversed_both"] / n
        ),
        "positive_role_signal_fraction": (
            role_positive / n
        ),
        "median_role_logit_delta": (
            statistics.median(role_deltas)
        ),
        **counts,
    }


def train_one(
    seed, arm, root, protocol_sha,
    base, calibration, exposure,
    device, steps,
):
    folder = (
        root / f"seed_{seed}" / arm
    )

    checkpoint_path = folder / "final.pt"

    summary_path = (
        folder / "train_summary.json"
    )

    if (
        checkpoint_path.exists()
        and summary_path.exists()
    ):
        print(
            f"seed={seed} arm={arm} already complete; skipped",
            flush=True,
        )
        return

    if folder.exists():
        raise FileExistsError(
            f"Incomplete training directory: {folder}"
        )

    folder.mkdir(parents=True)

    torch.manual_seed(seed)

    model = Transformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.01,
    )

    base_ids, base_pos, base_labels, _ = (
        encode(base)
    )

    dose_ids, dose_pos, dose_labels, _ = (
        encode(exposure[arm])
    )

    # Identical base minibatches for the same seed
    # across the three exposure conditions.
    g_base = torch.Generator(
        device="cpu"
    ).manual_seed(seed + 100000)

    g_dose = torch.Generator(
        device="cpu"
    ).manual_seed(seed + 200000)

    for _ in range(steps):
        model.train()

        bi = torch.randint(
            len(base),
            (56,),
            generator=g_base,
        )

        di = torch.randint(
            len(exposure[arm]),
            (8,),
            generator=g_dose,
        )

        base_x = base_ids[bi]
        dose_x = dose_ids[di]

        # Pad variable-length active/passive prompts.
        length = max(
            base_x.size(1),
            dose_x.size(1),
        )

        bx = F.pad(
            base_x,
            (0, length - base_x.size(1)),
            value=VOCAB["<pad>"],
        )

        dx = F.pad(
            dose_x,
            (0, length - dose_x.size(1)),
            value=VOCAB["<pad>"],
        )

        ids = torch.cat(
            [bx, dx],
            dim=0,
        ).to(device)

        readout = torch.cat([
            base_pos[bi],
            dose_pos[di],
        ]).to(device)

        labels = torch.cat([
            base_labels[bi],
            dose_labels[di],
        ]).to(device)

        loss = F.cross_entropy(
            model(ids, readout),
            labels,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

    summary = {
        "seed": seed,
        "arm": arm,
        "protocol_sha256": protocol_sha,
        "base_training": measure(
            model, base, device
        ),
        "calibration_known_grammar": measure(
            model, calibration, device
        ),
        "last_minibatch_loss": float(
            loss.detach().cpu()
        ),
    }

    torch.save({
        "seed": seed,
        "arm": arm,
        "protocol_sha256": protocol_sha,
        "model": model.cpu().state_dict(),
    }, checkpoint_path)

    summary_path.write_text(
        json.dumps(
            summary, indent=2
        ) + "\n"
    )

    print(
        f"seed={seed} arm={arm} "
        f"train={summary['base_training']['pair_accuracy']:.3f} "
        f"cal={summary['calibration_known_grammar']['pair_accuracy']:.3f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("plan", "train", "evaluate"),
        required=True,
    )

    parser.add_argument(
        "--root",
        default="results/ml072_exposure",
    )

    parser.add_argument(
        "--protocol",
        default="reports/ml072/frozen_protocol.json",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    assert args.steps > 0

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    device = resolve_device(args.device)

    doc, sha, base, cal, exposure, tests = (
        experiment(args)
    )

    protocol_path = Path(args.protocol)
    root = Path(args.root)

    if args.phase == "plan":
        if protocol_path.exists():
            check_plan(
                protocol_path, doc, sha
            )

            print(
                f"Frozen plan already exists: "
                f"{protocol_path} sha={sha[:12]}"
            )
            return

        protocol_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        protocol_path.write_text(
            json.dumps(
                {**doc, "sha256": sha},
                indent=2,
            ) + "\n"
        )

        print(
            f"ML-072 plan frozen: "
            f"{protocol_path} sha={sha[:12]}"
        )
        return

    check_plan(protocol_path, doc, sha)

    root.mkdir(
        parents=True, exist_ok=True
    )

    if args.phase == "train":
        for seed in SEEDS:
            for arm in ARMS:
                train_one(
                    seed, arm, root, sha,
                    base, cal, exposure,
                    device, args.steps,
                )

        print(
            "Training finished; "
            "evaluation has not been opened."
        )
        return

    output = root / "evaluation.json"

    ordered = [
        (seed, arm)
        for seed in SEEDS
        for arm in ARMS
    ]

    if output.exists():
        report = json.loads(
            output.read_text()
        )

        assert (
            report["experiment"] == "ML-072"
            and report["protocol_sha256"] == sha
        )

        completed = [
            (r["seed"], r["arm"])
            for r in report["results"]
        ]

        assert completed == ordered[
            :len(completed)
        ]

        if len(completed) == len(ordered):
            raise FileExistsError(
                "Evaluation already complete"
            )

        print(
            f"Resuming after "
            f"{len(completed)} completed models"
        )

    else:
        report = {
            "experiment": "ML-072",
            "protocol_sha256": sha,
            "status": "post-freeze evaluation",
            "results": [],
        }

    for seed, arm in ordered[
        len(report["results"]):
    ]:
        folder = (
            root / f"seed_{seed}" / arm
        )

        checkpoint = torch.load(
            folder / "final.pt",
            map_location="cpu",
            weights_only=True,
        )

        assert checkpoint["seed"] == seed
        assert checkpoint["arm"] == arm

        assert (
            checkpoint["protocol_sha256"]
            == sha
        )

        model = Transformer().to(device)

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        measured = {
            name: measure(
                model, cases, device
            )
            for name, cases in tests.items()
        }

        report["results"].append({
            "seed": seed,
            "arm": arm,
            "cohorts": measured,
        })

        output.write_text(
            json.dumps(
                report, indent=2
            ) + "\n"
        )

        primary = measured[
            "target_double_unseen"
        ]

        print(
            f"seed={seed} arm={arm:7s} "
            f"doubleOOD={primary['pair_accuracy']:.3f} "
            f"both={primary['correct_both_rate']:.3f} "
            f"same={primary['same_answer_rate']:.3f} "
            f"role+={primary['positive_role_signal_fraction']:.3f}",
            flush=True,
        )

    print(
        "\nPaired double-OOD differences "
        "(correct_both_rate):"
    )

    result_map = {
        (r["seed"], r["arm"]): (
            r["cohorts"][
                "target_double_unseen"
            ]["correct_both_rate"]
        )
        for r in report["results"]
    }

    comparisons = (
        ("sparse", "control"),
        ("broad", "control"),
        ("broad", "sparse"),
    )

    for arm_a, arm_b in comparisons:
        changes = [
            result_map[(seed, arm_a)]
            - result_map[(seed, arm_b)]
            for seed in SEEDS
        ]

        print(
            f"{arm_a}-{arm_b}: "
            + ", ".join(
                f"{value:+.3f}"
                for value in changes
            )
            + f" | mean="
            f"{statistics.mean(changes):+.3f}"
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
