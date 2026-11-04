
"""ML-069: Prospective replication on a fresh relational-language task.

Three phases:
    train -> freeze -> evaluate

Freeze and commit predictions BEFORE protected evaluation.

The architecture matches ML-037, but the vocabulary, training
distribution, participant pairs, and evaluation combinations are new.

This is a synthetic benchmark, not natural-language validation.
"""

import argparse
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import Block


NAMES = (
    "Ada", "Ben", "Cora", "Davi", "Ema",
    "Finn", "Gia", "Hugo", "Inez", "Jules",
)

VERBS = (
    "helped", "praised", "thanked",
    "followed", "noticed", "visited",
)

SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)

KNOWN = (
    "active/standard",
    "active/alternate",
    "passive/standard",
)

NOVEL = "passive/alternate"

VOCAB_WORDS = tuple(dict.fromkeys(
    (
        "<pad>", "<bos>", "<sep>", "<answer>",
        ".", "was", "by", "who", "acted",
        "received", "?", "identify", "the",
        "agent", "patient",
    ) + NAMES + VERBS
))

VOCAB = {
    token: i for i, token in enumerate(VOCAB_WORDS)
}


@dataclass(frozen=True)
class Case:
    pair: tuple
    verb: str
    direction: int
    voice: str
    role: str
    style: str


def case_key(c):
    return "|".join((
        *c.pair,
        c.verb,
        str(c.direction),
        c.voice,
        c.role,
        c.style,
    ))


def make_cases(pairs, constructions):
    return [
        Case(pair, verb, direction, voice, role, style)
        for pair in pairs
        for verb in VERBS
        for direction in (0, 1)
        for voice, style in constructions
        for role in ("agent", "patient")
    ]


def sentence(c):
    a, b = (
        c.pair
        if c.direction == 0
        else c.pair[::-1]
    )

    if c.voice == "active":
        event = [a, c.verb, b, "."]
    else:
        event = [b, "was", c.verb, "by", a, "."]

    if c.style == "standard":
        cue = (
            "acted"
            if c.role == "agent"
            else "received"
        )
        query = ["who", cue, "?"]
    else:
        query = ["identify", "the", c.role]

    return (
        ["<bos>"]
        + event
        + ["<sep>"]
        + query
        + ["<answer>"]
    )


def gold_name(c):
    a, b = (
        c.pair
        if c.direction == 0
        else c.pair[::-1]
    )

    return a if c.role == "agent" else b


def encode(cases):
    sequences = [sentence(c) for c in cases]
    lengths = [len(s) for s in sequences]

    ids = torch.full(
        (len(cases), max(lengths)),
        VOCAB["<pad>"],
        dtype=torch.long,
    )

    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(
            [VOCAB[t] for t in seq]
        )

    readout = torch.tensor([
        n - 1 for n in lengths
    ])

    labels = torch.tensor([
        VOCAB[gold_name(c)] for c in cases
    ])

    choices = torch.tensor([
        [VOCAB[name] for name in c.pair]
        for c in cases
    ])

    return ids, readout, labels, choices


class Transformer(nn.Module):
    """Same architecture family as ML-037; new vocabulary."""

    def __init__(self):
        super().__init__()

        width = 96

        self.embed = nn.Embedding(
            len(VOCAB), width
        )

        self.pos = nn.Embedding(32, width)

        self.blocks = nn.ModuleList([
            Block(width, 4)
            for _ in range(2)
        ])

        self.norm = nn.LayerNorm(width)

        self.output = nn.Linear(
            width, len(VOCAB)
        )

    def forward(self, ids, readout):
        batch, length = ids.shape

        x = (
            self.embed(ids)
            + self.pos(
                torch.arange(
                    length, device=ids.device
                )
            )
        )

        causal_mask = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=ids.device,
        ).triu(1)

        for block in self.blocks:
            x = block(x, causal_mask)

        selected = x[
            torch.arange(
                batch, device=ids.device
            ),
            readout,
        ]

        return self.output(
            self.norm(selected)
        )


def sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def protocol(args):
    pairs = list(
        itertools.combinations(NAMES, 2)
    )

    assert len(pairs) == 45

    random.Random(6901).shuffle(pairs)

    train_pairs = pairs[:27]
    cal_pairs = pairs[27:36]
    test_pairs = pairs[36:]

    assert not (
        set(train_pairs) & set(cal_pairs)
    )

    assert not (
        set(train_pairs) & set(test_pairs)
    )

    assert not (
        set(cal_pairs) & set(test_pairs)
    )

    assert {
        name
        for pair in train_pairs
        for name in pair
    } == set(NAMES)

    document = {
        "experiment": "ML-069",
        "names": NAMES,
        "verbs": VERBS,
        "vocabulary": VOCAB_WORDS,
        "train_constructions": KNOWN,
        "novel_construction": NOVEL,
        "train_pairs": train_pairs,
        "calibration_pairs": cal_pairs,
        "test_pairs": test_pairs,
        "seeds": args.seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": {
            "layers": 2,
            "heads": 4,
            "width": 96,
        },
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "primary_outcomes": [
            "Calibration-to-double-OOD pathway transfer",
            "Double-OOD pairwise answer accuracy",
        ],
        "route_rule": {
            "min_eligible": 64,
            "min_flip_rate": 0.20,
            "min_abs_margin_drop": 2.0,
            "dominance_threshold": 0.35,
        },
    }

    return (
        document,
        sha(document),
        train_pairs,
        cal_pairs,
        test_pairs,
    )


def datasets(train_pairs, cal_pairs, test_pairs):
    known = tuple(
        tuple(c.split("/"))
        for c in KNOWN
    )

    novel = (
        ("passive", "alternate"),
    )

    train = make_cases(train_pairs, known)
    cal = make_cases(cal_pairs, known)

    tests = {
        "new_pairs_seen_grammar":
            make_cases(test_pairs, known),

        "known_pairs_new_composition":
            make_cases(train_pairs, novel),

        "double_ood":
            make_cases(test_pairs, novel),
    }

    assert len(train) == 1944
    assert len(cal) == 648

    assert len(
        tests["new_pairs_seen_grammar"]
    ) == 648

    assert len(
        tests["known_pairs_new_composition"]
    ) == 648

    assert len(
        tests["double_ood"]
    ) == 216

    train_keys = set(map(case_key, train))
    cal_keys = set(map(case_key, cal))

    for test_cases in tests.values():
        test_keys = set(
            map(case_key, test_cases)
        )

        assert not (train_keys & test_keys)
        assert not (cal_keys & test_keys)

    return train, cal, tests


@torch.inference_mode()
def accuracy(model, cases, device):
    model.eval()

    pair_correct = 0
    full_correct = 0

    for start in range(0, len(cases), 256):
        batch = cases[start:start + 256]

        ids, pos, labels, choices = [
            tensor.to(device)
            for tensor in encode(batch)
        ]

        logits = model(ids, pos)

        candidate_scores = logits.gather(
            1, choices
        )

        predictions = choices.gather(
            1,
            candidate_scores.argmax(
                1, keepdim=True
            ),
        ).flatten()

        pair_correct += int(
            (predictions == labels).sum()
        )

        full_correct += int(
            (logits.argmax(1) == labels).sum()
        )

    return {
        "n": len(cases),
        "pair_accuracy": (
            pair_correct / len(cases)
        ),
        "full_accuracy": (
            full_correct / len(cases)
        ),
    }


def capture(model, ids, readout, donor_l1=None):
    state = {}

    rows = torch.arange(
        len(ids), device=ids.device
    )

    def record_l1(_module, _args, result):
        state["l1"] = (
            result[0].detach().clone()
        )

    def alter_l1(_module, _args, result):
        output = result[0].clone()

        output[rows, readout] = (
            donor_l1[rows, readout]
        )

        return (output, *result[1:])

    def record_x(_module, args):
        state["x"] = (
            args[0].detach().clone()
        )

    def record_a(_module, _args, result):
        state["a"] = (
            result[0].detach().clone()
        )

    handles = [
        model.blocks[0].attn.register_forward_hook(
            record_l1
        ),
        model.blocks[1].register_forward_pre_hook(
            record_x
        ),
        model.blocks[1].attn.register_forward_hook(
            record_a
        ),
    ]

    if donor_l1 is not None:
        handles.append(
            model.blocks[0].attn.register_forward_hook(
                alter_l1
            )
        )

    try:
        logits = model(ids, readout)
    finally:
        for handle in reversed(handles):
            handle.remove()

    assert set(state) == {"l1", "x", "a"}

    return logits, state


def hybrid(
    model, ids, readout,
    clean, swapped,
    x_source, a_source,
):
    rows = torch.arange(
        len(ids), device=ids.device
    )

    x_state = (
        clean if x_source == "clean"
        else swapped
    )

    a_state = (
        clean if a_source == "clean"
        else swapped
    )

    def inject_x(_module, args):
        x = args[0].clone()

        x[rows, readout] = (
            x_state["x"][rows, readout]
        )

        return (x, *args[1:])

    def inject_a(_module, _args, result):
        attn = result[0].clone()

        attn[rows, readout] = (
            a_state["a"][rows, readout]
        )

        return (attn, *result[1:])

    h1 = model.blocks[1].register_forward_pre_hook(
        inject_x
    )

    h2 = model.blocks[1].attn.register_forward_hook(
        inject_a
    )

    try:
        return model(ids, readout)
    finally:
        h2.remove()
        h1.remove()


def margin(logits, labels, choices):
    scores = logits.float().gather(
        1, choices
    )

    is_gold = (
        choices == labels[:, None]
    )

    assert torch.all(is_gold.sum(1) == 1)

    gold_slot = is_gold.long().argmax(1)

    gold_score = scores.gather(
        1, gold_slot[:, None]
    )

    other_score = scores.gather(
        1, (1 - gold_slot)[:, None]
    )

    return (
        gold_score - other_score
    ).flatten().cpu()


@torch.inference_mode()
def mechanism(model, cases, device):
    model.eval()

    ids, pos, labels, choices = [
        tensor.to(device)
        for tensor in encode(cases)
    ]

    clean_logits, clean = capture(
        model, ids, pos
    )

    clean_margin = margin(
        clean_logits, labels, choices
    )

    opposite = [
        Case(
            c.pair,
            c.verb,
            c.direction,
            c.voice,
            (
                "patient"
                if c.role == "agent"
                else "agent"
            ),
            c.style,
        )
        for c in cases
    ]

    donor_ids, donor_pos, _, _ = [
        tensor.to(device)
        for tensor in encode(opposite)
    ]

    assert donor_ids.shape == ids.shape
    assert torch.equal(donor_pos, pos)

    donor_logits, donor = capture(
        model,
        donor_ids,
        donor_pos,
    )

    donor_margin = margin(
        donor_logits, labels, choices
    )

    eligible = (
        (clean_margin > 0)
        & (donor_margin < 0)
    )

    n_eligible = int(eligible.sum())

    if n_eligible == 0:
        return {
            "n": len(cases),
            "eligible": 0,
            "baseline_accuracy": float(
                (clean_margin > 0).float().mean()
            ),
            "route": "indeterminate",
            "reason": "No jointly correct cases",
        }

    swapped_logits, swapped = capture(
        model,
        ids,
        pos,
        donor_l1=donor["l1"],
    )

    conditions = {}

    for x_source, a_source in itertools.product(
        ("clean", "swapped"),
        repeat=2,
    ):
        logits = hybrid(
            model,
            ids,
            pos,
            clean,
            swapped,
            x_source,
            a_source,
        )

        conditions[(x_source, a_source)] = (
            margin(logits, labels, choices)
        )

    m00 = conditions[("clean", "clean")]
    m01 = conditions[("clean", "swapped")]
    m10 = conditions[("swapped", "clean")]
    m11 = conditions[("swapped", "swapped")]

    assert float(
        (m00 - clean_margin).abs().max()
    ) < 0.003

    assert float(
        (
            m11 - margin(
                swapped_logits,
                labels,
                choices,
            )
        ).abs().max()
    ) < 0.003

    attention = 0.5 * (
        (m00 - m01)
        + (m10 - m11)
    )

    input_state = 0.5 * (
        (m00 - m10)
        + (m01 - m11)
    )

    total = m00 - m11

    assert torch.allclose(
        attention + input_state,
        total,
        atol=0.003,
    )

    # Same-role, different-verb control.
    other_verb = [
        Case(
            c.pair,
            VERBS[
                (VERBS.index(c.verb) + 1)
                % len(VERBS)
            ],
            c.direction,
            c.voice,
            c.role,
            c.style,
        )
        for c in cases
    ]

    verb_ids, verb_pos, _, _ = [
        tensor.to(device)
        for tensor in encode(other_verb)
    ]

    assert verb_ids.shape == ids.shape
    assert torch.equal(verb_pos, pos)

    _, verb_state = capture(
        model,
        verb_ids,
        verb_pos,
    )

    verb_logits, _ = capture(
        model,
        ids,
        pos,
        donor_l1=verb_state["l1"],
    )

    verb_margin = margin(
        verb_logits, labels, choices
    )

    def stats(mask):
        n = int(mask.sum())

        if n == 0:
            return None

        a = float(
            attention[mask].mean()
        )

        x = float(
            input_state[mask].mean()
        )

        t = float(
            total[mask].mean()
        )

        ratio = (
            (a - x)
            / (abs(a) + abs(x) + 1e-9)
        )

        flips = float(
            (m11[mask] < 0).float().mean()
        )

        route = "indeterminate"

        if (
            n >= 64
            and flips >= 0.20
            and abs(t) >= 2.0
        ):
            if ratio > 0.35:
                route = "attention"
            elif ratio < -0.35:
                route = "input"

        return {
            "eligible": n,
            "attention_margin_contribution": a,
            "input_margin_contribution": x,
            "total_margin_drop": t,
            "dominance_ratio": ratio,
            "flip_rate": flips,
            "attention_only_flip_rate": float(
                (m01[mask] < 0).float().mean()
            ),
            "input_only_flip_rate": float(
                (m10[mask] < 0).float().mean()
            ),
            "different_verb_control_flip_rate": float(
                (verb_margin[mask] < 0).float().mean()
            ),
            "route": route,
        }

    aggregate = stats(eligible)

    by_pair = {}

    for pair in sorted({
        c.pair for c in cases
    }):
        mask = torch.tensor([
            c.pair == pair for c in cases
        ]) & eligible

        by_pair["|".join(pair)] = stats(mask)

    by_construction = {}

    for voice, style in sorted({
        (c.voice, c.style)
        for c in cases
    }):
        mask = torch.tensor([
            c.voice == voice and c.style == style
            for c in cases
        ]) & eligible

        by_construction[
            f"{voice}/{style}"
        ] = stats(mask)

    return {
        "n": len(cases),
        "baseline_accuracy": float(
            (clean_margin > 0).float().mean()
        ),
        **aggregate,
        "by_pair": by_pair,
        "by_construction": by_construction,
    }


def resolve_device(requested):
    if requested != "auto":
        if requested == "mps":
            assert torch.backends.mps.is_available()
        if requested == "cuda":
            assert torch.cuda.is_available()
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(root, seed, digest, device):
    checkpoint = torch.load(
        root / f"seed_{seed}" / "final.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seed"] == seed
    assert checkpoint["protocol_sha256"] == digest

    model = Transformer().to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("train", "freeze", "evaluate"),
        required=True,
    )

    parser.add_argument(
        "--root",
        default="results/ml069_prospective",
    )

    parser.add_argument(
        "--frozen",
        default="reports/ml069/frozen_predictions.json",
    )

    parser.add_argument(
        "--steps", type=int, default=800
    )

    parser.add_argument(
        "--batch-size", type=int, default=64
    )

    parser.add_argument(
        "--lr", type=float, default=0.002
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    assert len(set(args.seeds)) == len(args.seeds)

    device = resolve_device(args.device)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    document, digest, train_pairs, cal_pairs, test_pairs = (
        protocol(args)
    )

    train, calibration, tests = datasets(
        train_pairs,
        cal_pairs,
        test_pairs,
    )

    root = Path(args.root)

    manifest_path = root / "protocol.json"

    if manifest_path.exists():
        saved = json.loads(
            manifest_path.read_text()
        )

        claimed = saved.pop("sha256")

        assert claimed == digest
        assert sha(saved) == digest

    elif args.phase == "train":
        root.mkdir(
            parents=True, exist_ok=True
        )

        manifest_path.write_text(
            json.dumps(
                {
                    **document,
                    "sha256": digest,
                },
                indent=2,
            ) + "\n"
        )

    else:
        raise FileNotFoundError(
            "Training protocol does not exist"
        )

    print(
        f"ML-069 phase={args.phase} "
        f"protocol={digest[:12]} "
        f"device={device}",
        flush=True,
    )

    print(
        f"train={len(train)} "
        f"calibration={len(calibration)}",
        flush=True,
    )

    if args.phase == "train":
        ids, positions, labels, _ = encode(train)

        for seed in args.seeds:
            seed_folder = (
                root / f"seed_{seed}"
            )

            if seed_folder.exists():
                checkpoint_file = (
                    seed_folder / "final.pt"
                )

                summary_file = (
                    seed_folder / "train_summary.json"
                )

                if (
                    checkpoint_file.exists()
                    and summary_file.exists()
                ):
                    print(
                        f"seed={seed} already complete; "
                        "preserving checkpoint"
                    )
                    continue

                raise FileExistsError(
                    f"Incomplete seed directory: {seed_folder}"
                )

            seed_folder.mkdir(
                parents=True, exist_ok=True
            )

            torch.manual_seed(seed)

            model = Transformer().to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=0.01,
            )

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(100000 + seed)

            for step in range(
                1, args.steps + 1
            ):
                model.train()

                picks = torch.randint(
                    len(train),
                    (args.batch_size,),
                    generator=generator,
                )

                logits = model(
                    ids[picks].to(device),
                    positions[picks].to(device),
                )

                loss = F.cross_entropy(
                    logits,
                    labels[picks].to(device),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()
                optimizer.step()

            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "step": args.steps,
                    "protocol_sha256": digest,
                },
                seed_folder / "final.pt",
            )

            train_score = accuracy(
                model, train, device
            )

            cal_score = accuracy(
                model, calibration, device
            )

            summary = {
                "seed": seed,
                "train": train_score,
                "calibration": cal_score,
                "last_minibatch_loss": float(
                    loss.detach().cpu()
                ),
            }

            (
                seed_folder / "train_summary.json"
            ).write_text(
                json.dumps(
                    summary, indent=2
                ) + "\n"
            )

            print(
                f"seed={seed} "
                f"train={train_score['pair_accuracy']:.3f} "
                f"cal={cal_score['pair_accuracy']:.3f}",
                flush=True,
            )

        print(
            "\nTraining complete. "
            "Next phase: freeze."
        )

        return

    frozen_path = Path(args.frozen)

    if args.phase == "freeze":
        if frozen_path.exists():
            raise FileExistsError(
                "Predictions already frozen"
            )

        frozen = {
            "experiment": "ML-069",
            "protocol_sha256": digest,
            "seeds": args.seeds,
            "status": "Pre-protected evaluation",
            "predictions": {},
        }

        for seed in args.seeds:
            model = load_model(
                root, seed, digest, device
            )

            cal_score = accuracy(
                model, calibration, device
            )

            mech = mechanism(
                model, calibration, device
            )

            frozen["predictions"][str(seed)] = {
                "calibration": cal_score,
                "mechanism": mech,
                "predicted_ood_route": mech["route"],
            }

            print(
                f"seed={seed} "
                f"cal={cal_score['pair_accuracy']:.3f} "
                f"eligible={mech['eligible']} "
                f"flip={mech.get('flip_rate', 0):.3f} "
                f"route={mech['route']}",
                flush=True,
            )

        frozen_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        frozen_path.write_text(
            json.dumps(
                frozen, indent=2
            ) + "\n"
        )

        print(
            "\nPredictions frozen:",
            frozen_path,
        )

        return

    # Protected evaluation phase.
    if not frozen_path.exists():
        raise FileNotFoundError(
            "Freeze predictions first"
        )

    frozen = json.loads(
        frozen_path.read_text()
    )

    assert frozen["protocol_sha256"] == digest
    assert frozen["seeds"] == args.seeds

    output = (
        root / "protected_evaluation.json"
    )

    expected = {
        "experiment": "ML-069",
        "protocol_sha256": digest,
        "frozen_sha256": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest(),
        "status": "Protected evaluation",
    }

    if output.exists():
        report = json.loads(
            output.read_text()
        )

        for key, value in expected.items():
            assert report[key] == value

        completed = [
            row["seed"]
            for row in report["results"]
        ]

        assert completed == args.seeds[
            :len(completed)
        ]

        if len(completed) == len(args.seeds):
            raise FileExistsError(
                "Evaluation already complete"
            )

        print(
            "Resuming evaluation after:",
            completed,
        )

    else:
        report = {
            **expected,
            "results": [],
        }
        completed = []

    for seed in args.seeds[len(completed):]:
        model = load_model(
            root, seed, digest, device
        )

        performance = {
            name: accuracy(
                model, cases, device
            )
            for name, cases in tests.items()
        }

        mechanisms = {
            name: mechanism(
                model, cases, device
            )
            for name, cases in tests.items()
        }

        predicted = frozen[
            "predictions"
        ][str(seed)]["predicted_ood_route"]

        observed = mechanisms[
            "double_ood"
        ]["route"]

        match = (
            predicted == observed
            if predicted != "indeterminate"
            else None
        )

        report["results"].append({
            "seed": seed,
            "predicted_route": predicted,
            "double_ood_route": observed,
            "matches": match,
            "accuracy": performance,
            "mechanisms": mechanisms,
        })

        output.write_text(
            json.dumps(
                report, indent=2
            ) + "\n"
        )

        print(
            f"seed={seed} "
            f"pairOOD="
            f"{performance['new_pairs_seen_grammar']['pair_accuracy']:.3f} "
            f"compositionOOD="
            f"{performance['known_pairs_new_composition']['pair_accuracy']:.3f} "
            f"doubleOOD="
            f"{performance['double_ood']['pair_accuracy']:.3f} "
            f"predicted={predicted} "
            f"observed={observed}",
            flush=True,
        )

    informative = [
        row
        for row in report["results"]
        if row["matches"] is not None
    ]

    matches = sum(
        row["matches"]
        for row in informative
    )

    print(
        f"\nRoute matches: "
        f"{matches}/{len(informative)} "
        "informative models"
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
