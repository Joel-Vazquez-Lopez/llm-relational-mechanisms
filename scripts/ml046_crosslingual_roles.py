
"""ML-046: English/Spanish prospective comparison of semantic-role routing.

Two independently initialized scratch Transformer cohorts, one per language,
with paired seeds, a shared UNION vocabulary and identical semantic events.
Train/calibration name pairs are used to nominate heads and set competence.
The protected pairs are evaluated only after the nomination manifest is frozen.

Primary endpoint: in each language, training-nominated block-2 head shows
positive (agent minus patient) attention-swap harm in both voices on novel pairs.
The task remains synthetic; this does not establish natural language ability.
"""

import argparse
import hashlib
import json
import math
import random

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from ml037_scratch_transformer_pilot import ScratchTransformer


NAMES = (
    "Maya", "Lena", "Omar", "Rosa",
    "Nora", "Clara", "Felix", "Iris",
)

GENDER = {
    "Maya": "f",
    "Lena": "f",
    "Omar": "m",
    "Rosa": "f",
    "Nora": "f",
    "Clara": "f",
    "Felix": "m",
    "Iris": "f",
}

# English active, English passive,
# Spanish active, Spanish participle stem.
VERBS = (
    ("helped", "helped", "ayudó", "ayudad"),
    ("praised", "praised", "elogió", "elogiad"),
    ("visited", "visited", "visitó", "visitad"),
    ("greeted", "greeted", "saludó", "saludad"),
)

LANGUAGES = ("en", "es")
VOICES = ("active", "passive")
ROLES = ("agent", "patient")

SEEDS = (201, 211, 223, 227)

STEPS = 300
BATCH = 64
LEARNING_RATE = 0.002

SPLIT_SEED = 4601
CONTROL_COUNT = 4

RESULTS = Path("results")


@dataclass(frozen=True)
class Case:
    pair: tuple[str, str]
    verb: int
    direction: int
    voice: str
    role: str


def participants(case):
    if case.direction == 0:
        return case.pair
    return case.pair[::-1]


def prompt(case, language):
    agent, patient = participants(case)

    en, en_pass, es, stem = VERBS[case.verb]

    if language == "en":
        if case.voice == "active":
            event = [agent, en, patient, "."]
        else:
            event = [
                patient, "was", en_pass,
                "by", agent, ".",
            ]

        question = [
            "who",
            (
                "performed"
                if case.role == "agent"
                else "received"
            ),
            "the", "action", "?",
        ]

    elif language == "es":
        if case.voice == "active":
            event = [
                agent, es, "a", patient, ".",
            ]
        else:
            ending = (
                "a" if GENDER[patient] == "f"
                else "o"
            )

            event = [
                patient,
                "fue",
                stem + ending,
                "por",
                agent,
                ".",
            ]

        question = [
            "¿",
            "quién",
            (
                "realizó"
                if case.role == "agent"
                else "recibió"
            ),
            "la", "acción", "?",
        ]

    else:
        raise ValueError(language)

    return (
        ["<bos>"]
        + event
        + ["<sep>"]
        + question
        + ["<answer>"]
    )


def split():
    pairs = [
        (NAMES[i], NAMES[j])
        for i in range(len(NAMES))
        for j in range(i + 1, len(NAMES))
    ]

    random.Random(SPLIT_SEED).shuffle(pairs)

    result = {
        "train": pairs[:14],
        "calibration": pairs[14:21],
        "protected": pairs[21:],
    }

    assert tuple(
        map(len, result.values())
    ) == (14, 7, 7)

    assert {
        name
        for pair in result["train"]
        for name in pair
    } == set(NAMES)

    return result


SPLIT = split()


def quartets(pairs, verb_ids=range(len(VERBS))):
    for pair in pairs:
        for verb in verb_ids:
            for direction in (0, 1):
                yield (
                    pair,
                    verb,
                    direction,
                    [
                        Case(
                            pair, verb, direction,
                            voice, role,
                        )
                        for voice in VOICES
                        for role in ROLES
                    ],
                )


def all_cases(pairs):
    return [
        case
        for _, _, _, quartet in quartets(pairs)
        for case in quartet
    ]


TOKENS = ("<pad>",) + tuple(sorted({
    token
    for language in LANGUAGES
    for _, _, _, quartet in quartets([
        pair
        for group in SPLIT.values()
        for pair in group
    ])
    for case in quartet
    for token in prompt(case, language)
} - {"<pad>"}))

VOCAB = {
    token: i
    for i, token in enumerate(TOKENS)
}


SPEC = {
    "experiment": "ML-046",
    "languages": LANGUAGES,
    "seeds": SEEDS,
    "steps": STEPS,
    "batch": BATCH,
    "lr": LEARNING_RATE,
    "split_seed": SPLIT_SEED,
    "split": SPLIT,
    "vocab": TOKENS,
    "model": {
        "layers": 2,
        "heads": 4,
        "width": 96,
    },
    "nomination": (
        "training-only maximum agent-minus-patient "
        "directional harm, first two verbs"
    ),
    "primary": (
        "positive agent-minus-patient harm in BOTH "
        "voices for each language, protected pairs"
    ),
    "gate": (
        "calibration pairwise accuracy >= 0.95"
    ),
    "intervention": (
        "swap block-2 head attention weights "
        "between semantic agent and patient"
    ),
    "controls": (
        "4 coordinate sign-flipped equal-norm offsets"
    ),
    "status": (
        "prospective new models; prior English "
        "observations motivated hypothesis"
    ),
}

DESIGN_HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def encode(cases, language):
    sequences = [
        prompt(c, language)
        for c in cases
    ]

    lengths = [
        len(seq) for seq in sequences
    ]

    ids = torch.full(
        (
            len(cases),
            max(lengths),
        ),
        VOCAB["<pad>"],
        dtype=torch.long,
    )

    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor([
            VOCAB[t] for t in seq
        ])

    readout = torch.tensor([
        n - 1 for n in lengths
    ])

    labels = torch.tensor([
        VOCAB[
            participants(c)[
                0 if c.role == "agent" else 1
            ]
        ]
        for c in cases
    ])

    choices = torch.tensor([
        [VOCAB[n] for n in c.pair]
        for c in cases
    ])

    locations = torch.tensor([
        [
            seq.index(participants(c)[0]),
            seq.index(participants(c)[1]),
        ]
        for c, seq in zip(cases, sequences)
    ])

    assert all(
        seq.count(participants(c)[0]) == 1
        and seq.count(participants(c)[1]) == 1
        for c, seq in zip(cases, sequences)
    )

    return ids, readout, labels, choices, locations


class PairedModel(ScratchTransformer):
    def __init__(self):
        super().__init__(
            width=96,
            heads=4,
            layers=2,
            max_len=40,
        )

        self.embed = nn.Embedding(
            len(VOCAB), 96
        )

        self.output = nn.Linear(
            96, len(VOCAB)
        )


def path_for(language, seed):
    return RESULTS / (
        f"ml046_{language}_seed{seed}.pt"
    )


def checksum(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_model(
    language, seed, device,
    expected_hash=None,
):
    path = path_for(language, seed)
    digest = checksum(path)

    if (
        expected_hash is not None
        and expected_hash != digest
    ):
        raise RuntimeError(
            f"Checkpoint changed: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        data["language"],
        data["seed"],
        data["design_hash"],
        data["vocab"],
    ) != (
        language,
        seed,
        DESIGN_HASH,
        VOCAB,
    ):
        raise RuntimeError(
            f"Checkpoint metadata mismatch {path}"
        )

    model = PairedModel().to(device)

    model.load_state_dict(
        data["state_dict"],
        strict=True,
    )

    model.eval()

    return model, digest


@torch.no_grad()
def accuracy(model, cases, language, device):
    correct = 0

    for start in range(0, len(cases), 128):
        batch = cases[start:start + 128]

        ids, pos, y, choices, _ = encode(
            batch, language
        )

        logits = model(
            ids.to(device),
            pos.to(device),
        ).cpu()

        candidate_scores = logits.gather(
            1, choices
        )

        guess = choices.gather(
            1,
            candidate_scores.argmax(
                1, keepdim=True
            ),
        ).squeeze(1)

        correct += int(
            (guess == y).sum()
        )

    return correct / len(cases)


def train(language, seed, device):
    path = path_for(language, seed)

    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite {path}"
        )

    torch.manual_seed(seed)

    model = PairedModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    cases = all_cases(SPLIT["train"])

    ids, pos, y, _, _ = encode(
        cases, language
    )

    for step in range(1, STEPS + 1):
        model.train()

        pick = torch.randint(
            len(cases), (BATCH,)
        )

        logits = model(
            ids[pick].to(device),
            pos[pick].to(device),
        )

        loss = F.cross_entropy(
            logits,
            y[pick].to(device),
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        if step in (100, 200, 300):
            print(
                f"{language} seed={seed} "
                f"step={step} "
                f"loss={loss.item():.4f}"
            )

    model.eval()

    training = accuracy(
        model, cases, language, device
    )

    calibration = accuracy(
        model,
        all_cases(SPLIT["calibration"]),
        language,
        device,
    )

    torch.save({
        "language": language,
        "seed": seed,
        "design_hash": DESIGN_HASH,
        "vocab": VOCAB,
        "train_accuracy": training,
        "calibration_accuracy": calibration,
        "state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
    }, path)

    print(
        f"TRAINED {language} seed={seed}: "
        f"train={training:.3f} "
        f"cal={calibration:.3f} "
        f"saved={path}"
    )


@torch.no_grad()
def capture(model, quartet, language, device):
    ids, pos, labels, choices, locations = encode(
        quartet, language
    )

    ids, pos, labels, choices, locations = (
        x.to(device)
        for x in (
            ids, pos, labels,
            choices, locations,
        )
    )

    saved = {}

    def prehook(_module, inputs):
        saved["z"] = inputs[0].detach()

    def posthook(_module, _inputs, output):
        saved["attention_output"] = (
            output[0].detach()
        )

    hook = (
        model.blocks[1].attn
        .register_forward_pre_hook(prehook)
    )

    hook2 = (
        model.blocks[1].attn
        .register_forward_hook(posthook)
    )

    try:
        logits, states = model(
            ids, pos,
            return_states=True,
        )
    finally:
        hook.remove()
        hook2.remove()

    mha = model.blocks[1].attn

    z = saved["z"]

    batch, length, width = z.shape

    heads = mha.num_heads
    dim = width // heads

    q, k, v = F.linear(
        z,
        mha.in_proj_weight,
        mha.in_proj_bias,
    ).split(width, -1)

    def divide(x):
        return x.reshape(
            batch,
            length,
            heads,
            dim,
        ).transpose(1, 2)

    q, k, v = (
        divide(q),
        divide(k),
        divide(v),
    )

    idx = torch.arange(
        batch, device=device
    )

    qread = q[idx, :, pos]

    scores = torch.einsum(
        "bhd,bhld->bhl",
        qread, k,
    ) / math.sqrt(dim)

    scores = scores.masked_fill(
        (
            torch.arange(
                length, device=device
            )[None, None]
            > pos[:, None, None]
        ),
        float("-inf"),
    )

    weights = scores.softmax(-1)

    context = torch.einsum(
        "bhl,bhld->bhd",
        weights, v,
    )

    recon = F.linear(
        context.reshape(batch, width),
        mha.out_proj.weight,
        mha.out_proj.bias,
    )

    # Compare reconstructed attention with
    # the actual attention module output.
    natural = saved["attention_output"][
        idx, pos
    ]

    reconstruction_error = float(
        (recon - natural).abs().max()
    )

    if reconstruction_error > 0.005:
        raise RuntimeError(
            "Attention reconstruction failed: "
            f"{reconstruction_error}"
        )

    agent = locations[:, 0]
    patient = locations[:, 1]

    wa = weights[
        idx, :, agent
    ]

    wp = weights[
        idx, :, patient
    ]

    va = v[
        idx, :, agent
    ]

    vp = v[
        idx, :, patient
    ]

    offsets = []

    for h in range(heads):
        W = mha.out_proj.weight[
            :,
            h * dim:(h + 1) * dim,
        ]

        delta = (
            (wp[:, h] - wa[:, h])[:, None]
            * F.linear(
                va[:, h] - vp[:, h],
                W,
            )
        )

        offsets.append(delta)

    offsets = torch.stack(
        offsets, dim=1
    )

    other = torch.where(
        choices[:, 0] == labels,
        choices[:, 1],
        choices[:, 0],
    )

    baseline = (
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
        "offsets": offsets,
        "baseline": baseline,
        "logits": logits,
        "heads": heads,
        "mean_margin": float(
            baseline.mean()
        ),
        "weight_mass": float(
            (wa + wp).mean()
        ),
        "h_norm": float(
            states[2].norm(
                dim=-1
            ).mean()
        ),
    }


def control_signs(
    seed, language, key,
    head, index, width,
    device, dtype,
):
    payload = json.dumps([
        "ML046",
        seed,
        language,
        list(key[0]),
        key[1],
        key[2],
        head,
        index,
    ])

    num = int.from_bytes(
        hashlib.sha256(
            payload.encode()
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(num)

    bits = torch.randint(
        0, 2,
        (4, width),
        generator=generator,
    )

    return (2 * bits - 1).to(
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def effects(
    model, data, language,
    seed, key, device,
):
    offsets = data["offsets"]

    variants = [
        (
            "baseline",
            -1,
            -1,
            torch.zeros_like(
                offsets[:, 0]
            ),
        )
    ]

    for head in range(data["heads"]):
        delta = offsets[:, head]

        variants.append((
            "actual",
            head,
            -1,
            delta,
        ))

        for j in range(CONTROL_COUNT):
            ctrl = delta * control_signs(
                seed,
                language,
                key,
                head,
                j,
                delta.shape[-1],
                device,
                delta.dtype,
            )

            variants.append((
                "control",
                head,
                j,
                ctrl,
            ))

    n = len(variants)

    patches = torch.stack([
        row[3] for row in variants
    ]).reshape(n * 4, -1)

    ids = data["ids"].repeat(n, 1)
    pos = data["pos"].repeat(n)

    idx = torch.arange(
        n * 4, device=device
    )

    def hook(_module, _inputs, output):
        attn, *rest = output

        updated = attn.clone()

        updated[idx, pos] = (
            updated[idx, pos]
            + patches.to(updated.dtype)
        )

        return (updated, *rest)

    handle = (
        model.blocks[1].attn
        .register_forward_hook(hook)
    )

    try:
        logits = model(
            ids, pos
        ).reshape(n, 4, -1)
    finally:
        handle.remove()

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
            margins[0] - data["baseline"]
        ).abs().max()
    ) > 0.05:
        raise RuntimeError(
            "No-op intervention mismatch"
        )

    rows = []

    for head in range(data["heads"]):
        actual_index = (
            1 + head * (CONTROL_COUNT + 1)
        )

        actual = margins[actual_index]

        start = actual_index + 1

        ctrl = margins[
            start:start + CONTROL_COUNT
        ]

        if ctrl.shape[0] != CONTROL_COUNT:
            raise RuntimeError(
                "Wrong number of controls"
            )

        harm = (
            ctrl.mean(0) - actual
        )

        rows.append({
            "head": head,
            "harm": harm.cpu().tolist(),
            "baseline_accuracy": (
                (margins[0] > 0)
                .float().cpu().tolist()
            ),
            "swap_accuracy": (
                (actual > 0)
                .float().cpu().tolist()
            ),
            "control_accuracy": (
                (ctrl > 0)
                .float().mean(0)
                .cpu().tolist()
            ),
            "offset_norm_fraction": (
                float(
                    offsets[:, head]
                    .norm(dim=-1).mean()
                )
                / data["h_norm"]
            ),
        })

    return rows


def summarize(rows, head):
    selected = [
        r["head_results"][head]
        for r in rows
    ]

    harms = [
        r["harm"]
        for r in selected
    ]

    acc = [
        r["swap_accuracy"]
        for r in selected
    ]

    control_acc = [
        r["control_accuracy"]
        for r in selected
    ]

    baseline_acc = [
        r["baseline_accuracy"]
        for r in selected
    ]

    # Quartet order:
    # active agent, active patient,
    # passive agent, passive patient.
    means = [
        mean(z[j] for z in harms)
        for j in range(4)
    ]

    return {
        "harm_by_voice_role": dict(zip(
            (
                "active_agent",
                "active_patient",
                "passive_agent",
                "passive_patient",
            ),
            means,
        )),
        "agent_minus_patient_active": (
            means[0] - means[1]
        ),
        "agent_minus_patient_passive": (
            means[2] - means[3]
        ),
        "swap_accuracy": mean(
            x for z in acc for x in z
        ),
        "control_accuracy": mean(
            x for z in control_acc for x in z
        ),
        "baseline_accuracy": mean(
            x for z in baseline_acc for x in z
        ),
        "mean_offset_norm_fraction": mean(
            r["offset_norm_fraction"]
            for r in selected
        ),
    }


def score_head(rows, head):
    s = summarize(rows, head)

    return 0.5 * (
        s["agent_minus_patient_active"]
        + s["agent_minus_patient_passive"]
    )


def evaluate_cases(
    model, language, seed,
    cases, device,
):
    rows = []

    for i, (
        pair, verb, direction, q
    ) in enumerate(cases, 1):

        captured = capture(
            model, q, language, device
        )

        head_results = effects(
            model,
            captured,
            language,
            seed,
            (pair, verb, direction),
            device,
        )

        rows.append({
            "pair": list(pair),
            "verb": verb,
            "direction": direction,
            "head_results": head_results,
            "baseline_mean_margin": (
                captured["mean_margin"]
            ),
            "attention_mass_on_participants": (
                captured["weight_mass"]
            ),
        })

        if i % 16 == 0:
            print(
                f"{language} seed={seed} "
                f"causal quartets "
                f"{i}/{len(cases)}"
            )

    return rows


def freeze(device):
    output = RESULTS / (
        "ml046_frozen_head_nomination.json"
    )

    if output.exists():
        raise RuntimeError(
            "Refusing to overwrite frozen manifest"
        )

    manifest = {
        "experiment": "ML-046",
        "design_hash": DESIGN_HASH,
        "spec": SPEC,
        "models": {},
        "complete": True,
    }

    for language in LANGUAGES:
        for seed in SEEDS:
            model, digest = load_model(
                language, seed, device
            )

            cal_acc = accuracy(
                model,
                all_cases(
                    SPLIT["calibration"]
                ),
                language,
                device,
            )

            # Nomination only uses training pairs
            # and the first two verbs.
            rows = evaluate_cases(
                model,
                language,
                seed,
                list(quartets(
                    SPLIT["train"],
                    verb_ids=(0, 1),
                )),
                device,
            )

            scores = [
                score_head(rows, j)
                for j in range(4)
            ]

            chosen = max(
                range(4),
                key=lambda j: scores[j],
            )

            key = f"{language}:{seed}"

            manifest["models"][key] = {
                "checkpoint_sha256": digest,
                "calibration_accuracy": cal_acc,
                "eligible": (
                    cal_acc >= 0.95
                ),
                "nominated_head": chosen,
                "training_scores": scores,
            }

            print(
                f"FROZEN {key}: "
                f"cal={cal_acc:.3f} "
                f"eligible={cal_acc >= .95} "
                f"head={chosen} "
                f"training_scores="
                f"{[round(x, 3) for x in scores]}"
            )

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "FROZEN:", output,
        "sha256:", checksum(output)
    )

    print(
        "Commit manifest BEFORE --phase test. "
        "No protected outcomes accessed."
    )


def test(device):
    manifest_path = RESULTS / (
        "ml046_frozen_head_nomination.json"
    )

    output = RESULTS / (
        "ml046_protected_comparison.json"
    )

    if output.exists():
        raise RuntimeError(
            "Refusing to overwrite protected results"
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    expected_models = {
        f"{lang}:{seed}"
        for lang in LANGUAGES
        for seed in SEEDS
    }

    if (
        manifest.get("design_hash") != DESIGN_HASH
        or not manifest.get("complete")
        or set(manifest["models"]) != expected_models
    ):
        raise RuntimeError(
            "Frozen manifest mismatch"
        )

    results = []

    for lang in LANGUAGES:
        for seed in SEEDS:
            name = f"{lang}:{seed}"

            record = manifest["models"][name]

            model, digest = load_model(
                lang,
                seed,
                device,
                record["checkpoint_sha256"],
            )

            rows = evaluate_cases(
                model,
                lang,
                seed,
                list(quartets(
                    SPLIT["protected"]
                )),
                device,
            )

            nominated = record["nominated_head"]

            summary = summarize(
                rows, nominated
            )

            result = {
                "language": lang,
                "seed": seed,
                "eligible": record["eligible"],
                "selected_head": nominated,
                "summary": summary,
                "records": rows,
            }

            results.append(result)

            print(
                f"TEST {name} "
                f"eligible={record['eligible']} "
                f"H{nominated}: "
                f"active selectivity="
                f"{summary['agent_minus_patient_active']:+.3f} "
                f"passive selectivity="
                f"{summary['agent_minus_patient_passive']:+.3f} "
                f"base_acc="
                f"{summary['baseline_accuracy']:.3f} "
                f"swap_acc="
                f"{summary['swap_accuracy']:.3f}"
            )

    output.write_text(
        json.dumps({
            "experiment": "ML-046",
            "complete": True,
            "design_hash": DESIGN_HASH,
            "manifest_sha256": (
                checksum(manifest_path)
            ),
            "results": results,
        }, indent=2, ensure_ascii=False)
    )

    for lang in LANGUAGES:
        good = [
            r for r in results
            if r["language"] == lang
            and r["eligible"]
        ]

        print(
            f"{lang}: "
            f"{len(good)}/{len(SEEDS)} "
            "competent seeds"
        )

        if good:
            active_mean = mean(
                r["summary"][
                    "agent_minus_patient_active"
                ]
                for r in good
            )

            passive_mean = mean(
                r["summary"][
                    "agent_minus_patient_passive"
                ]
                for r in good
            )

            successes = sum(
                (
                    r["summary"][
                        "agent_minus_patient_active"
                    ] > 0
                    and r["summary"][
                        "agent_minus_patient_passive"
                    ] > 0
                )
                for r in good
            )

            print(
                "  agent selectivity, "
                "active/passive:",
                round(active_mean, 3),
                round(passive_mean, 3),
            )

            print(
                "  both positive:",
                successes,
                "/",
                len(good),
            )

    print("Saved", output)


def audit(device):
    print(
        "DESIGN", DESIGN_HASH
    )

    print(
        "TRAIN/CAL/PROTECTED",
        {
            k: len(v)
            for k, v in SPLIT.items()
        },
    )

    print(
        "VOCAB",
        len(VOCAB),
        "same for both language models",
    )

    for lang in LANGUAGES:
        examples = [
            Case(
                SPLIT["train"][0],
                1,
                0,
                voice,
                "agent",
            )
            for voice in VOICES
        ]

        for case in examples:
            print(
                lang,
                case.voice,
                " ".join(
                    prompt(case, lang)
                ),
            )

            if (
                lang == "es"
                and case.voice == "passive"
            ):
                _, patient = participants(case)

                expected = (
                    VERBS[case.verb][3]
                    + (
                        "a"
                        if GENDER[patient] == "f"
                        else "o"
                    )
                )

                assert expected in prompt(
                    case, lang
                )

        torch.manual_seed(42)

        model = PairedModel().to(device).eval()

        quartet = [
            Case(
                SPLIT["train"][0],
                1,
                0,
                voice,
                role,
            )
            for voice in VOICES
            for role in ROLES
        ]

        captured = capture(
            model,
            quartet,
            lang,
            device,
        )

        res = effects(
            model,
            captured,
            lang,
            42,
            (
                SPLIT["train"][0],
                1,
                0,
            ),
            device,
        )

        assert len(res) == 4

        print(
            "AUDIT PASS",
            lang,
            "causal attention swap, "
            "4 heads, 4 controls",
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--phase",
        choices=(
            "audit",
            "train",
            "freeze",
            "test",
        ),
        required=True,
    )

    p.add_argument(
        "--language",
        choices=LANGUAGES,
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=None,
    )

    p.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "mps",
        ),
        default="auto",
    )

    args = p.parse_args()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
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
            "MPS unavailable"
        )

    RESULTS.mkdir(exist_ok=True)

    print(
        f"ML-046 phase={args.phase} "
        f"device={device} "
        f"design={DESIGN_HASH}"
    )

    if args.phase == "audit":
        audit(device)

    elif args.phase == "train":
        if (
            args.language is None
            or args.seed is None
        ):
            p.error(
                "--phase train needs "
                "--language and --seed"
            )

        train(
            args.language,
            args.seed,
            device,
        )

    elif args.phase == "freeze":
        freeze(device)

    elif args.phase == "test":
        test(device)


if __name__ == "__main__":
    main()
