"""ML-080: prospective learning curves for paired target-vs-familiar exposure.

A NEW two-event grammar dataset using 16 new participant names and balanced
perfect-matching pair splits. The ML-075 protected cases are never generated.

Phases from repository root:
  python3 scripts/ml080_learning_curve.py --phase plan
  python3 scripts/ml080_learning_curve.py --phase preflight
  python3 scripts/ml080_learning_curve.py --phase train --until 3200
  python3 scripts/ml080_learning_curve.py --phase summarize
  python3 scripts/ml080_learning_curve.py --phase train --until 12800
  python3 scripts/ml080_learning_curve.py --phase summarize

Protected evaluation is an OPTIONAL separately confirmed action and is
blocked unless the frozen competence gate passes in BOTH arms.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

NAMES = ("Aria", "Bram", "Cleo", "Dax", "Esme", "Faye", "Glen", "Hale",
         "Isla", "Juno", "Kaya", "Luca", "Mira", "Nell", "Omar", "Pia")
VERBS = ("helped", "praised", "thanked", "followed", "noticed", "visited")
SEEDS = (307, 311, 313, 317, 331, 337, 347, 349)
ARMS = ("familiar", "target")
BASE_CELLS = ((0, "active"), (0, "passive"), (1, "active"))
TARGET_CELL = (1, "passive")
MILESTONES = (3200, 12800)
TOTAL_STEPS = 12800
TRAIN_BATCH = 56
DOSE_BATCH = 8
WIDTH, HEADS, LAYERS = 96, 4, 3
VOCAB_WORDS = tuple(dict.fromkeys((
    "<pad>", "<bos>", "<sep>", "<answer>", "While", "while", "was", "by",
    ",", ".", "In", "the", "first", "second", "event", "who", "agent",
    "patient", "?",
) + NAMES + VERBS))
VOCAB = {word: idx for idx, word in enumerate(VOCAB_WORDS)}


@dataclass(frozen=True)
class Case:
    target_pair: tuple[str, str]
    distractor_pair: tuple[str, str]
    target_verb: str
    distractor_verb: str
    target_pos: int
    target_voice: str
    distractor_voice: str
    target_dir: int
    distractor_dir: int
    frame: int
    role: str


def hash_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matchings():
    """One-factorization of K_16: each of 15 rounds covers all 16 names once."""
    players = list(range(len(NAMES)))
    rounds = []
    for _ in range(len(NAMES) - 1):
        pairs = [tuple(sorted((NAMES[players[i]], NAMES[players[-1-i]])))
                 for i in range(len(NAMES) // 2)]
        rounds.append(pairs)
        players = [players[0], players[-1], *players[1:-1]]
    flat = [pair for round_pairs in rounds for pair in round_pairs]
    assert len(flat) == len(set(flat)) == 120
    assert set(flat) == set(itertools.combinations(NAMES, 2))
    rng = random.Random(80080)
    rng.shuffle(rounds)
    for round_pairs in rounds:
        rng.shuffle(round_pairs)
    return rounds[:9], rounds[9:12], rounds[12:]


def clause(pair, verb, voice, direction):
    agent, patient = pair if direction == 0 else pair[::-1]
    if voice == "active":
        return [agent, verb, patient]
    assert voice == "passive"
    return [patient, "was", verb, "by", agent]


def prompt(c):
    target = clause(c.target_pair, c.target_verb, c.target_voice, c.target_dir)
    distractor = clause(c.distractor_pair, c.distractor_verb, c.distractor_voice, c.distractor_dir)
    left, right = (target, distractor) if c.target_pos == 0 else (distractor, target)
    sentence = (["While"] + left + [","] + right + ["."] if c.frame == 0
                else left + ["while"] + right + ["."])
    question = ["In", "the", "first" if c.target_pos == 0 else "second",
                "event", ",", "who", "was", "the", c.role, "?"]
    return ["<bos>"] + sentence + ["<sep>"] + question + ["<answer>"]


def answer(c):
    agent, patient = c.target_pair if c.target_dir == 0 else c.target_pair[::-1]
    return agent if c.role == "agent" else patient


def cases_for_event(**kw):
    return [Case(**kw, role=role) for role in ("agent", "patient")]


def base_cases(rounds):
    """Every pair x verb x familiar cell x frame x distractor voice x target direction.

    A cyclic rotation within each perfect matching balances distractor-name
    appearances as well as target-name appearances.
    """
    cases = []
    for matching in rounds:
        for pair_idx, pair in enumerate(matching):
            for verb_idx, verb in enumerate(VERBS):
                for target_pos, target_voice in BASE_CELLS:
                    voice_idx = int(target_voice == "passive")
                    for frame in (0, 1):
                        for distractor_voice_idx, distractor_voice in enumerate(("active", "passive")):
                            for target_dir in (0, 1):
                                salt = verb_idx * 3 + frame * 5 + distractor_voice_idx * 2 + target_dir + target_pos + voice_idx
                                shift = 1 + salt % 7
                                distractor_pair = matching[(pair_idx + shift) % 8]
                                distractor_dir = (verb_idx + frame + distractor_voice_idx + target_dir + target_pos) % 2
                                distractor_verb = VERBS[(verb_idx + 1 + (pair_idx + frame) % 5) % len(VERBS)]
                                cases.extend(cases_for_event(
                                    target_pair=pair, distractor_pair=distractor_pair,
                                    target_verb=verb, distractor_verb=distractor_verb,
                                    target_pos=target_pos, target_voice=target_voice,
                                    distractor_voice=distractor_voice, target_dir=target_dir,
                                    distractor_dir=distractor_dir, frame=frame,
                                ))
    return cases


def exposure_pair(rounds):
    """64 exactly matched examples/arm, differing ONLY in target-event position.

    One source event occupies the first (familiar) or second (target) position.
    Participants, verbs, roles, directions, frame, and distractor voice are
    otherwise identical. Clausal order and query cue necessarily change.
    """
    matching = rounds[0]
    familiar, target = [], []
    for pair_idx in range(4):
        for verb_idx in (0, 1):
            for frame in (0, 1):
                for dv_idx, distractor_voice in enumerate(("active", "passive")):
                    target_dir = (pair_idx + verb_idx + frame) % 2
                    distractor_dir = (pair_idx + verb_idx + dv_idx) % 2
                    shift = 1 + (verb_idx * 3 + frame * 5 + dv_idx * 2) % 7
                    distractor_pair = matching[(pair_idx + shift) % 8]
                    distractor_verb = VERBS[(verb_idx + 2 + pair_idx % 3) % len(VERBS)]
                    shared = dict(
                        target_pair=matching[pair_idx], distractor_pair=distractor_pair,
                        target_verb=VERBS[verb_idx], distractor_verb=distractor_verb,
                        target_voice="passive", distractor_voice=distractor_voice,
                        target_dir=target_dir, distractor_dir=distractor_dir,
                        frame=frame,
                    )
                    familiar.extend(cases_for_event(target_pos=0, **shared))
                    target.extend(cases_for_event(target_pos=1, **shared))
    assert len(familiar) == len(target) == 64
    for a, b in zip(familiar, target):
        assert a.target_pos == 0 and b.target_pos == 1
        assert replace(a, target_pos=1) == b
    return {"familiar": familiar, "target": target}


def protected_target_cases(rounds):
    """Only called after final competence gate and explicit user unlock."""
    examples = []
    for matching in rounds:
        for pi, pair in enumerate(matching):
            for vi in range(2, len(VERBS)):
                for frame in (0, 1):
                    for dv_idx, dv in enumerate(("active", "passive")):
                        for tdir in (0, 1):
                            shift = 1 + (vi * 3 + frame * 5 + dv_idx * 2 + tdir) % 7
                            examples.extend(cases_for_event(
                                target_pair=pair,
                                distractor_pair=matching[(pi + shift) % 8],
                                target_verb=VERBS[vi],
                                distractor_verb=VERBS[(vi + 1 + pi % 4) % 6],
                                target_pos=1, target_voice="passive",
                                distractor_voice=dv, target_dir=tdir,
                                distractor_dir=(vi + pi + frame + dv_idx + tdir) % 2,
                                frame=frame,
                            ))
    assert len(examples) == 1536
    return examples


def encode(cases):
    seqs = [prompt(c) for c in cases]
    lengths = [len(seq) for seq in seqs]
    assert max(lengths) <= 64
    ids = torch.full((len(cases), max(lengths)), VOCAB["<pad>"], dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor([VOCAB[token] for token in seq], dtype=torch.long)
    readout = torch.tensor([length - 1 for length in lengths], dtype=torch.long)
    labels = torch.tensor([VOCAB[answer(c)] for c in cases], dtype=torch.long)
    choices = torch.tensor([[VOCAB[name] for name in sorted(c.target_pair + c.distractor_pair)]
                            for c in cases], dtype=torch.long)
    return ids, readout, labels, choices


def verify_cases(train, cal, dose, train_rounds, cal_rounds, protected_rounds):
    assert len(train) == 20736 and len(cal) == 6912
    assert all(len(v) == 64 for v in dose.values())
    assert len(NAMES) == len(set(NAMES)) == 16
    for rounds, n in ((train_rounds, 9), (cal_rounds, 3), (protected_rounds, 3)):
        assert len(rounds) == n
        assert all(len(matching) == 8 for matching in rounds)
        assert all(Counter(name for pair in matching for name in pair)
                   == Counter({name: 1 for name in NAMES}) for matching in rounds)
    ps = [set(pair for matching in rounds for pair in matching)
          for rounds in (train_rounds, cal_rounds, protected_rounds)]
    assert not (ps[0] & ps[1] or ps[0] & ps[2] or ps[1] & ps[2])
    assert len(ps[0]) == 72 and len(ps[1]) == len(ps[2]) == 24
    for cases, cells in ((train, set(BASE_CELLS)), (cal, set(BASE_CELLS))):
        assert len(set(tuple(prompt(c)) for c in cases)) == len(cases)
        assert all((c.target_pos, c.target_voice) in cells for c in cases)
        assert all(len(set(c.target_pair + c.distractor_pair)) == 4 and c.target_verb != c.distractor_verb for c in cases)
        assert all(a.role == "agent" and b.role == "patient" and replace(a, role="patient") == b
                   for a, b in zip(cases[::2], cases[1::2]))
        assert len(set(answer(c) for c in cases)) == 16
        names = Counter(answer(c) for c in cases)
        assert len(set(names.values())) == 1, f"Answer-name imbalance: {names}"
        assert all(c.target_pair in (ps[0] if cases is train else ps[1]) for c in cases)
    assert not ({tuple(prompt(c)) for c in train} & {tuple(prompt(c)) for c in cal})
    assert all((c.target_pos, c.target_voice) != TARGET_CELL for c in train + cal)
    assert all((c.target_pos, c.target_voice) == TARGET_CELL for c in dose["target"])
    assert Counter(answer(c) for c in dose["familiar"]) == Counter(answer(c) for c in dose["target"])
    assert all(len(set(tuple(prompt(c)) for c in rows)) == 64 for rows in dose.values())
    assert all(c.target_pair in ps[0] for cases in dose.values() for c in cases)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x, mask):
        normed = self.n1(x)
        x = x + self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), WIDTH)
        self.pos = nn.Embedding(64, WIDTH)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, len(VOCAB))

    def forward(self, ids, readout):
        batch, n = ids.shape
        assert n <= 64
        positions = torch.arange(n, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        mask = torch.ones(n, n, dtype=torch.bool, device=ids.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(self.norm(x[torch.arange(batch, device=ids.device), readout]))


@torch.inference_mode()
def measure(model, cases, device):
    model.eval()
    totals = Counter()
    for start in range(0, len(cases), 128):
        group = cases[start:start + 128]
        ids, readout, labels, choices = encode(group)
        logits = model(ids.to(device), readout.to(device)).float()
        cand = choices.to(device)
        pred = cand.gather(1, logits.gather(1, cand).argmax(1, keepdim=True)).flatten()
        full_pred = logits.argmax(1)
        preds = pred.cpu().tolist()
        all_pred = full_pred.cpu().tolist()
        lab = labels.tolist()
        for index, c in enumerate(group):
            p = VOCAB_WORDS[preds[index]]
            if p == answer(c):
                totals["correct"] += 1
            elif p in c.target_pair:
                totals["wrong_role"] += 1
            elif p in c.distractor_pair:
                totals["wrong_event"] += 1
            else:
                raise AssertionError("Four-choice decoding failed")
            totals["full_correct"] += all_pred[index] == lab[index]
            totals["n"] += 1
            # Cases are constructed in agent,patient adjacent order across batches.
            if index % 2 == 0:
                prev_prediction = p
                prev_correct = preds[index] == lab[index]
            else:
                totals["both"] += prev_correct and (preds[index] == lab[index])
                totals["same"] += prev_prediction == p
                totals["events"] += 1
    n = totals["n"]
    assert n == len(cases) and totals["events"] * 2 == n
    assert totals["correct"] + totals["wrong_role"] + totals["wrong_event"] == n
    return {
        "questions": n, "events": totals["events"],
        "choice_accuracy": totals["correct"] / n,
        "full_vocab_accuracy": totals["full_correct"] / n,
        "correct_both_rate": totals["both"] / totals["events"],
        "wrong_role_rate": totals["wrong_role"] / n,
        "wrong_event_rate": totals["wrong_event"] / n,
        "same_answer_rate": totals["same"] / totals["events"],
    }


def manifest(train_rounds, cal_rounds, protected_rounds):
    return {
        "experiment": "ML-080",
        "code_sha256": sha_file(__file__),
        "seeds": SEEDS, "arms": ARMS,
        "names": NAMES, "verbs": VERBS,
        "splits": {
            "method": "one-factorization K16, shuffled with seed 80080; 9/3/3 perfect matchings",
            "train_pairs": train_rounds,
            "calibration_pairs": cal_rounds,
            "protected_pairs_identifiers_only": protected_rounds,
        },
        "counts": {"train_questions": 20736, "calibration_questions": 6912,
                   "dose_questions_per_arm": 64, "protected_familiar_questions": 6912,
                   "protected_target_questions": 1536},
        "training": {"total_steps": TOTAL_STEPS, "milestones": MILESTONES,
                     "batch_size": TRAIN_BATCH + DOSE_BATCH, "base_batch": TRAIN_BATCH,
                     "exposure_batch": DOSE_BATCH, "learning_rate": 0.002,
                     "weight_decay": 0.01, "architecture": "3-layer causal attention + MLP",
                     "width": WIDTH, "heads": HEADS, "dropout": 0},
        "exposure": {
            "familiar": "64 first-event passive target questions",
            "target": "64 second-event passive target questions",
            "matching": "same target/distractor pairs, verbs, frame, directions, roles and distractor voice",
            "sole_assigned_difference": "target position 0 versus 1 (with corresponding clause order/query cue)",
            "sampling": "same dose index RNG stream and same base RNG stream within paired seeds",
        },
        "primary": "target-minus-familiar choice_accuracy on disjoint-pair familiar calibration at 12800 steps",
        "secondary": ["3200-step calibration gap", "correct_both_rate", "base train fit",
                      "wrong_event_rate", "wrong_role_rate", "full_vocab_accuracy",
                      "post-gate protected familiar and target construction generalization"],
        "competence_gate": {
            "min_passing_seeds_per_arm": 6,
            "min_final_calibration_choice_accuracy": 0.95,
            "min_final_calibration_correct_both_rate": 0.90,
            "protected_requires_explicit_flag": True,
        },
        "interpretation": {
            "material_harm": "paired mean target-familiar calibration gap < -0.05",
            "negligible_gap": "paired mean gap >= -0.03",
            "uncertainty": "use 8 paired seeds and 95% bootstrap CI; outcomes between or straddling thresholds inconclusive",
            "note": "cannot infer specific circuit from behavior; training regimes differ in target-event position",
        },
        "protected_rule": "NO protected case generation until final gate passes and --confirm-protected supplied",
        "device_note": "MPS/GPU kernels may not be bitwise deterministic across machines",
    }


def frozen_protocol(path, doc, create):
    canonical = json.loads(json.dumps({**doc, "sha256": hash_json(doc)}))
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(canonical, indent=2) + "\n")
    if not path.is_file():
        raise FileNotFoundError(f"Run --phase plan first: {path}")
    if json.loads(path.read_text()) != canonical:
        raise AssertionError("ML-080 source or frozen protocol changed")
    return canonical["sha256"]


def split_data():
    train_rounds, cal_rounds, protected_rounds = matchings()
    base = base_cases(train_rounds)
    cal = base_cases(cal_rounds)
    dose = exposure_pair(train_rounds)
    verify_cases(base, cal, dose, train_rounds, cal_rounds, protected_rounds)
    return train_rounds, cal_rounds, protected_rounds, base, cal, dose


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    return name


def load_checkpoint(path, seed, arm, protocol_hash, device):
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert (data["seed"], data["arm"], data["protocol_sha256"]) == (seed, arm, protocol_hash)
    model = Model().to(device)
    model.load_state_dict(data["model"], strict=True)
    return model.eval()


def atomic_save(object_to_save, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(object_to_save, temp)
    os.replace(temp, path)


def train_one(seed, arm, until, save_every, root, h, base, cal, dose, device):
    folder = root / f"seed_{seed}" / arm
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "state.pt"
    torch.manual_seed(seed)
    model = Model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    gbase = torch.Generator(device="cpu").manual_seed(seed + 800000)
    gdose = torch.Generator(device="cpu").manual_seed(seed + 810000)
    step, last_loss = 0, None
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        assert (saved["seed"], saved["arm"], saved["protocol_sha256"]) == (seed, arm, h)
        model.load_state_dict(saved["model"], strict=True)
        optim.load_state_dict(saved["optimizer"])
        gbase.set_state(saved["base_rng"])
        gdose.set_state(saved["dose_rng"])
        step = saved["step"]
        last_loss = saved["last_loss"]
    if step > until:
        print(f"seed={seed} arm={arm} already at {step} > until={until}; skipped", flush=True)
        return

    base_enc, dose_enc = encode(base), encode(dose[arm])

    def save_state():
        atomic_save({
            "seed": seed, "arm": arm, "protocol_sha256": h, "step": step,
            "last_loss": last_loss,
            "model": model.state_dict(), "optimizer": optim.state_dict(),
            "base_rng": gbase.get_state(), "dose_rng": gdose.get_state(),
        }, state_path)

    def milestone():
        metrics_path = folder / f"metrics_step_{step:05d}.json"
        snapshot_path = folder / f"model_step_{step:05d}.pt"
        if metrics_path.exists() and snapshot_path.exists():
            existing = json.loads(metrics_path.read_text())
            assert existing["protocol_sha256"] == h
            return
        if metrics_path.exists() != snapshot_path.exists():
            raise RuntimeError(f"Incomplete milestone files in {folder}")
        scores = {
            "seed": seed, "arm": arm, "step": step, "protocol_sha256": h,
            "last_training_loss": last_loss,
            "train": measure(model, base, device),
            "calibration": measure(model, cal, device),
            "exposure": measure(model, dose[arm], device),
        }
        # Save a model-only milestone snapshot, preserving both endpoints.
        atomic_save({"seed": seed, "arm": arm, "step": step,
                     "protocol_sha256": h, "model": model.state_dict()}, snapshot_path)
        metrics_path.write_text(json.dumps(scores, indent=2) + "\n")
        print(f"seed={seed} arm={arm:8s} step={step:5d} "
              f"train={scores['train']['choice_accuracy']:.3f} "
              f"cal={scores['calibration']['choice_accuracy']:.3f} "
              f"both={scores['calibration']['correct_both_rate']:.3f} "
              f"exposure={scores['exposure']['choice_accuracy']:.3f} "
              f"wrong_event={scores['calibration']['wrong_event_rate']:.3f}", flush=True)

    if step in MILESTONES:
        milestone()
    while step < until:
        model.train()
        bidx = torch.randint(len(base), (TRAIN_BATCH,), generator=gbase)
        didx = torch.randint(len(dose[arm]), (DOSE_BATCH,), generator=gdose)
        bx, br, by, _ = base_enc
        dx, dr, dy, _ = dose_enc
        a, b = bx[bidx], dx[didx]
        m = max(a.shape[1], b.shape[1])
        ids = torch.cat([
            F.pad(a, (0, m - a.shape[1]), value=VOCAB["<pad>"]),
            F.pad(b, (0, m - b.shape[1]), value=VOCAB["<pad>"]),
        ], dim=0).to(device)
        readout = torch.cat([br[bidx], dr[didx]], dim=0).to(device)
        labels = torch.cat([by[bidx], dy[didx]], dim=0).to(device)
        loss = F.cross_entropy(model(ids, readout), labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1
        last_loss = float(loss.detach())
        if step % save_every == 0 or step == until or step in MILESTONES:
            save_state()
        if step in MILESTONES:
            milestone()
    print(f"seed={seed} arm={arm:8s} checkpoint={step}/{TOTAL_STEPS}", flush=True)


def all_metrics(root, h, step, complete=False):
    rows = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = root / f"seed_{seed}" / arm / f"metrics_step_{step:05d}.json"
            if not path.exists():
                if complete:
                    raise FileNotFoundError(f"Missing milestone: {path}")
                continue
            obj = json.loads(path.read_text())
            assert (obj["protocol_sha256"], obj["seed"], obj["arm"], obj["step"]) == (h, seed, arm, step)
            rows[(seed, arm)] = obj
    return rows


def bootstrap_gap(values):
    rng = random.Random(800803)
    dist = [mean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(20000)]
    dist.sort()
    return (dist[500], dist[19500])


def summarize(root, h):
    for step in MILESTONES:
        rows = all_metrics(root, h, step)
        print(f"STEP {step}: {len(rows)}/16 milestone reports")
        if not rows:
            continue
        for arm in ARMS:
            matched = [row["calibration"] for (seed, a), row in rows.items() if a == arm]
            if matched:
                print(f"  {arm:8s} ({len(matched)}/8) mean_cal={mean(x['choice_accuracy'] for x in matched):.3f} "
                      f"both={mean(x['correct_both_rate'] for x in matched):.3f}")
        paired = [rows[(seed, "target")]["calibration"]["choice_accuracy"]
                  - rows[(seed, "familiar")]["calibration"]["choice_accuracy"]
                  for seed in SEEDS if (seed, "target") in rows and (seed, "familiar") in rows]
        if len(paired) == len(SEEDS):
            lo, hi = bootstrap_gap(paired)
            print(f"  target-minus-familiar cal gap={mean(paired):+.3f} "
                  f"95% paired-bootstrap CI=[{lo:+.3f},{hi:+.3f}]")
    final = all_metrics(root, h, TOTAL_STEPS)
    if len(final) != len(SEEDS) * len(ARMS):
        print("PROTECTED GATE: NOT READY (all 16 final milestone reports required)")
        return False
    gate_ok = True
    for arm in ARMS:
        passing = sum(
            final[(seed, arm)]["calibration"]["choice_accuracy"] >= 0.95
            and final[(seed, arm)]["calibration"]["correct_both_rate"] >= 0.90
            for seed in SEEDS
        )
        gate_ok &= passing >= 6
        print(f"PROTECTED GATE {arm}: {passing}/8 pass (need >=6)")
    print("PROTECTED GATE:", "PASS" if gate_ok else "FAIL -- protected evaluation forbidden")
    return gate_ok


def evaluate(root, h, protected_rounds, device, unlock):
    if not unlock:
        raise PermissionError("Protected evaluation requires explicit --confirm-protected")
    if not summarize(root, h):
        raise PermissionError("Frozen competence gate has not passed in both arms")
    path = root / "protected_evaluation.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite protected results: {path}")
    # CRITICAL: protected cases are generated ONLY after all gate checks pass.
    familiar_protected = base_cases(protected_rounds)
    target_protected = protected_target_cases(protected_rounds)
    result = {"experiment": "ML-080", "protocol_sha256": h,
              "protected_evaluation_opened": True, "runs": []}
    for seed in SEEDS:
        for arm in ARMS:
            snapshot = root / f"seed_{seed}" / arm / f"model_step_{TOTAL_STEPS:05d}.pt"
            model = load_checkpoint(snapshot, seed, arm, h, device)
            result["runs"].append({
                "seed": seed, "arm": arm,
                "familiar_new_pairs": measure(model, familiar_protected, device),
                "target_new_pairs_new_exposure_verbs": measure(model, target_protected, device),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Protected evaluation saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=("plan", "preflight", "train", "summarize", "evaluate"))
    parser.add_argument("--protocol", default="reports/ml080/frozen_protocol.json")
    parser.add_argument("--root", default="results/ml080_learning_curve")
    parser.add_argument("--until", type=int, default=3200)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--confirm-protected", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not (0 < args.until <= TOTAL_STEPS):
        raise ValueError(f"--until must be in 1..{TOTAL_STEPS}")
    if args.save_every <= 0:
        raise ValueError("--save-every must be >0")
    train_rounds, cal_rounds, protected_rounds, base, cal, dose = split_data()
    doc = manifest(train_rounds, cal_rounds, protected_rounds)
    protocol_hash = frozen_protocol(Path(args.protocol), doc, args.phase == "plan")
    if args.phase == "plan":
        print("ML-080 frozen:", args.protocol, protocol_hash[:12]); return
    if args.phase == "preflight":
        print("ML-080 preflight PASS", f"protocol={protocol_hash[:12]}")
        print(f"new_names={len(NAMES)} train_pairs=72 calibration_pairs=24 protected_pair_IDs=24")
        print(f"training_questions={len(base)} calibration_questions={len(cal)} exposure_per_arm=64")
        print("Every name has exactly equal training and calibration gold-label frequency.")
        print("Exposure arms match on content except event position and query cue.")
        print("ML-075 protected data NOT accessed; ML-080 protected questions NOT generated.")
        return
    root = Path(args.root)
    if args.phase == "summarize":
        summarize(root, protocol_hash); return
    device = select_device(args.device)
    if args.phase == "train":
        for seed in (SEEDS if args.seed is None else (args.seed,)):
            for arm in (ARMS if args.arm is None else (args.arm,)):
                train_one(seed, arm, args.until, args.save_every, root,
                          protocol_hash, base, cal, dose, device)
        print("Training session complete. Protected results still sealed.")
        return
    evaluate(root, protocol_hash, protected_rounds, device, args.confirm_protected)


if __name__ == "__main__":
    main()
