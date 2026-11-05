
"""ML-075: Two-event grammatical role binding and compositional exposure.

Phases: plan -> preflight -> train -> evaluate.
The primary heldout target is the PASSIVE event in the SECOND clause.
Run all phases from the repository root. Only PyTorch is required.
"""
import argparse
import hashlib
import itertools
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

NAMES = ("Ada", "Ben", "Cora", "Davi", "Ema", "Finn",
         "Gia", "Hugo", "Inez", "Jules", "Kira", "Leon")
VERBS = ("helped", "praised", "thanked", "followed", "noticed", "visited")
SEEDS = (211, 223, 227, 229, 233, 239)
ARMS = ("control", "sparse", "broad")
BASE_CELLS = ((0, "active"), (0, "passive"), (1, "active"))
TARGET = (1, "passive")
VOCAB_WORDS = tuple(dict.fromkeys(
    ("<pad>", "<bos>", "<sep>", "<answer>", "While", "while", "was",
     "by", ",", ".", "In", "the", "first", "second", "event", "who",
     "agent", "patient", "?", "first", "second") + NAMES + VERBS
))
VOCAB = {s: i for i, s in enumerate(VOCAB_WORDS)}


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


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def two_roles(**kwargs):
    return [Case(**kwargs, role=r) for r in ("agent", "patient")]


def clause(pair, verb, voice, direction):
    a, p = pair if direction == 0 else pair[::-1]
    if voice == "active":
        return [a, verb, p]
    return [p, "was", verb, "by", a]


def prompt(c):
    target = clause(c.target_pair, c.target_verb, c.target_voice, c.target_dir)
    dist = clause(c.distractor_pair, c.distractor_verb, c.distractor_voice, c.distractor_dir)
    left, right = (target, dist) if c.target_pos == 0 else (dist, target)
    if c.frame == 0:
        sentence = ["While"] + left + [","] + right + ["."]
    else:
        sentence = left + ["while"] + right + ["."]
    question = ["In", "the", "first" if c.target_pos == 0 else "second",
                "event", ",", "who", "was", "the", c.role, "?"]
    return ["<bos>"] + sentence + ["<sep>"] + question + ["<answer>"]


def gold(c):
    agent, patient = c.target_pair if c.target_dir == 0 else c.target_pair[::-1]
    return agent if c.role == "agent" else patient


def encode(cases):
    sequences = [prompt(c) for c in cases]
    lengths = [len(s) for s in sequences]
    maxlen = max(lengths)
    ids = torch.full((len(cases), maxlen), VOCAB["<pad>"], dtype=torch.long)
    for i, s in enumerate(sequences):
        ids[i, :len(s)] = torch.tensor([VOCAB[t] for t in s])
    readout = torch.tensor([n-1 for n in lengths])
    labels = torch.tensor([VOCAB[gold(c)] for c in cases])
    choices = torch.tensor([[VOCAB[n] for n in sorted(c.target_pair + c.distractor_pair)]
                            for c in cases])
    return ids, readout, labels, choices


def select_distractor(target_pair, other_pairs, salt):
    eligible = [p for p in other_pairs if not set(p) & set(target_pair)]
    assert eligible
    return eligible[salt % len(eligible)]


def balanced_base(pairs):
    """Balanced by target cell, verb, frame, distractor voice and both directions."""
    cases = []
    for cell_index, (pos, voice) in enumerate(BASE_CELLS):
        for vi, verb in enumerate(VERBS):
            for frame in range(2):
                for dv in ("active", "passive"):
                    for tdir in range(2):
                        for ddir in range(2):
                            for rep in range(12):
                                pair = pairs[(rep * 7 + vi * 3 + frame * 5 +
                                              cell_index * 11 + tdir * 13 + ddir) % len(pairs)]
                                dist = select_distractor(pair, pairs,
                                    rep * 17 + vi * 7 + frame * 3 + tdir * 5 + ddir)
                                dverb = VERBS[(vi + 1 + rep % (len(VERBS)-1)) % len(VERBS)]
                                if dverb == verb:
                                    dverb = VERBS[(vi + 1) % len(VERBS)]
                                cases.extend(two_roles(
                                    target_pair=pair, distractor_pair=dist,
                                    target_verb=verb, distractor_verb=dverb,
                                    target_pos=pos, target_voice=voice,
                                    distractor_voice=dv, target_dir=tdir,
                                    distractor_dir=ddir, frame=frame))
    return cases


def exposure(pairs, verbs, pool, target_cell, broad):
    cases = []
    for pi, pair in enumerate(pairs):
        for vi, verb in enumerate(verbs):
            for frame in range(2):
                for dv in ("active", "passive"):
                    directions = tuple(itertools.product(range(2), repeat=2)) if broad else (
                        ((pi + vi + frame) % 2, (pi + vi + int(dv == "passive")) % 2),)
                    for tdir, ddir in directions:
                        dist = select_distractor(pair, pool,
                            pi * 43 + vi * 29 + frame * 7 + tdir * 3 + ddir)
                        dverb = VERBS[(VERBS.index(verb) + 2 + pi % 3) % len(VERBS)]
                        if dverb == verb:
                            dverb = VERBS[(VERBS.index(verb) + 1) % len(VERBS)]
                        cases.extend(two_roles(
                            target_pair=pair, distractor_pair=dist,
                            target_verb=verb, distractor_verb=dverb,
                            target_pos=target_cell[0], target_voice=target_cell[1],
                            distractor_voice=dv, target_dir=tdir,
                            distractor_dir=ddir, frame=frame))
    return cases


def target_test(pairs, pool, verbs):
    """All test target pairs, both lexical verbs, frames, distractor voices and target directions."""
    cases = []
    for pi, pair in enumerate(pairs):
        for verb in verbs:
            vi = VERBS.index(verb)
            for frame in range(2):
                for dv in ("active", "passive"):
                    for tdir in range(2):
                        dist = select_distractor(pair, pool,
                            pi * 31 + vi * 13 + frame * 7 + tdir)
                        dverb = VERBS[(vi + 1 + pi % 3) % len(VERBS)]
                        if dverb == verb:
                            dverb = VERBS[(vi + 1) % len(VERBS)]
                        cases.extend(two_roles(
                            target_pair=pair, distractor_pair=dist,
                            target_verb=verb, distractor_verb=dverb,
                            target_pos=TARGET[0], target_voice=TARGET[1],
                            distractor_voice=dv, target_dir=tdir,
                            distractor_dir=(pi + frame + tdir) % 2, frame=frame))
    return cases


def build(args):
    pairs = list(itertools.combinations(NAMES, 2))
    random.Random(7501).shuffle(pairs)
    train_pairs, cal_pairs, test_pairs = pairs[:40], pairs[40:53], pairs[53:]
    assert all(set(train_pairs).isdisjoint(x) for x in (cal_pairs, test_pairs))
    assert set(cal_pairs).isdisjoint(test_pairs)
    assert {n for pair in train_pairs for n in pair} == set(NAMES)
    base = balanced_base(train_pairs)
    doses = {
        "control": exposure(train_pairs[:4], VERBS[:2], train_pairs,
                            (0, "passive"), broad=False),
        "sparse": exposure(train_pairs[:4], VERBS[:2], train_pairs,
                           TARGET, broad=False),
        "broad": exposure(train_pairs[:24], VERBS[:4], train_pairs,
                          TARGET, broad=True),
    }
    calibration = balanced_base(cal_pairs)
    test = {
        "known_grammar_test_pairs": balanced_base(test_pairs),
        "target_seen_pairs_new_verbs": target_test(train_pairs[:16], train_pairs, VERBS[4:]),
        "target_new_pairs_seen_verbs": target_test(test_pairs, train_pairs, VERBS[:4]),
        "target_double_unseen": target_test(test_pairs, train_pairs, VERBS[4:]),
    }
    assert all((c.target_pos, c.target_voice) != TARGET for c in base + calibration)
    assert len(base) == 3 * 6 * 2 * 2 * 2 * 2 * 12 * 2
    assert len(doses["control"]) == len(doses["sparse"]) == 64
    assert len(doses["broad"]) == 24 * 4 * 2 * 2 * 4 * 2
    assert all(len({*c.target_pair, *c.distractor_pair}) == 4 for seq in
               [base, calibration, *doses.values(), *test.values()] for c in seq)
    assert all(c.target_verb != c.distractor_verb for seq in
               [base, calibration, *doses.values(), *test.values()] for c in seq)
    assert all(c.target_pair not in train_pairs for c in test["target_double_unseen"])
    assert all(c.target_verb in VERBS[4:] for c in test["target_double_unseen"])
    # Every event has one agent question and one patient question, adjacent.
    for seq in [base, calibration, *doses.values(), *test.values()]:
        assert all(seq[i].role == "agent" and seq[i+1].role == "patient" and
                   seq[i].__dict__ | {"role": "patient"} == seq[i+1].__dict__
                   for i in range(0, len(seq), 2))
    doc = {
        "experiment": "ML-075", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": SEEDS, "arms": ARMS, "names": NAMES, "verbs": VERBS,
        "training_pair_ids": train_pairs, "calibration_pair_ids": cal_pairs,
        "protected_pair_ids": test_pairs, "base_cells": BASE_CELLS,
        "heldout_target_cell": TARGET, "sparse_pairs": train_pairs[:4],
        "broad_pairs": train_pairs[:24], "target_exposure_verbs": VERBS[:4],
        "target_new_verbs": VERBS[4:],
        "n_examples": {"base": len(base), "calibration": len(calibration),
                       **{f"dose_{a}": len(doses[a]) for a in ARMS},
                       **{k: len(v) for k,v in test.items()}},
        "training": {"steps": args.steps, "batch_size": 64, "base_batch": 56,
                     "exposure_batch": 8, "lr": 0.002, "layers": 3,
                     "width": 96, "heads": 4},
        "primary": "target_double_unseen.correct_both_rate",
        "comparisons": ("sparse-control", "broad-control", "broad-sparse"),
        "gate": "Before protected evaluation require at least 5 of 6 seeds in EACH arm to reach calibration pair_accuracy >= 0.95 and correct_both_rate >= 0.90 on the known constructions. If gate fails, report failures and stop; do not retune within this protocol.",
        "limits": "Templated English; heldout pair/verb recombinations, not unseen vocabulary or natural-language transfer",
    }
    return doc, sha(doc), base, calibration, doses, test


class Block(nn.Module):
    def __init__(self, width=96, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))

    def forward(self, x, mask):
        z = self.n1(x)
        x = x + self.attn(z, z, z, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), 96)
        self.pos = nn.Embedding(64, 96)
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.out = nn.Linear(96, len(VOCAB))

    def forward(self, ids, readout):
        batch, n = ids.shape
        assert n <= 64
        positions = torch.arange(n, device=ids.device)
        x = self.embed(ids) + self.pos(positions)
        mask = torch.ones(n, n, device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(self.norm(x[torch.arange(batch, device=ids.device), readout]))


@torch.inference_mode()
def measure(model, cases, device):
    model.eval()
    results = {}
    for start in range(0, len(cases), 128):
        batch = cases[start:start+128]
        ids, pos, labels, choices = [x.to(device) for x in encode(batch)]
        logits = model(ids, pos).float()
        candidate = logits.gather(1, choices)
        prediction = choices.gather(1, candidate.argmax(1, keepdim=True)).flatten()
        full = logits.argmax(1)
        for c, g, p, f in zip(batch, labels.tolist(), prediction.tolist(), full.tolist()):
            k = (c.target_pair, c.distractor_pair, c.target_verb, c.distractor_verb,
                 c.target_pos, c.target_voice, c.distractor_voice,
                 c.target_dir, c.distractor_dir, c.frame)
            results.setdefault(k, {})[c.role] = (p == g, f == g, p)
    counts = {"both": 0, "pair": 0, "full": 0, "same": 0}
    for e in results.values():
        assert set(e) == {"agent", "patient"}
        a, p = e["agent"], e["patient"]
        counts["both"] += a[0] and p[0]
        counts["pair"] += int(a[0]) + int(p[0])
        counts["full"] += int(a[1]) + int(p[1])
        counts["same"] += a[2] == p[2]
    n = len(results)
    assert 2*n == len(cases), "Duplicate event keys: generation integrity failure"
    return {"events": n, "questions": 2*n, "correct_both_rate": counts["both"]/n,
            "pair_accuracy": counts["pair"]/(2*n), "full_vocab_accuracy": counts["full"]/(2*n),
            "same_answer_rate": counts["same"]/n}


def freeze_check(path, doc, h):
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen protocol {path}; run --phase plan first")
    assert json.loads(path.read_text()) == json.loads(json.dumps({**doc, "sha256":h})), "Protocol/source mismatch"


def train_one(seed, arm, root, h, base, cal, doses, device, steps):
    folder = root / f"seed_{seed}" / arm
    ckpt = folder / "final.pt"
    summary_path = folder / "training.json"
    if ckpt.exists() and summary_path.exists():
        print(f"seed={seed} arm={arm} complete; skipped", flush=True)
        return
    if folder.exists():
        raise FileExistsError(f"Incomplete run: {folder}")
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    base_enc = encode(base)
    dose_enc = encode(doses[arm])
    gbase = torch.Generator(device="cpu").manual_seed(seed+750000)
    gdose = torch.Generator(device="cpu").manual_seed(seed+760000)
    for step in range(steps):
        model.train()
        bi = torch.randint(len(base), (56,), generator=gbase)
        di = torch.randint(len(doses[arm]), (8,), generator=gdose)
        ids0, pos0, label0, _ = base_enc
        ids1, pos1, label1, _ = dose_enc
        batch0, batch1 = ids0[bi], ids1[di]
        maxlen = max(batch0.size(1), batch1.size(1))
        batch = torch.cat([F.pad(batch0, (0, maxlen-batch0.size(1)), value=VOCAB["<pad>"]),
                           F.pad(batch1, (0, maxlen-batch1.size(1)), value=VOCAB["<pad>"])], dim=0).to(device)
        readout = torch.cat([pos0[bi], pos1[di]]).to(device)
        labels = torch.cat([label0[bi], label1[di]]).to(device)
        loss = F.cross_entropy(model(batch, readout), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    summary = {"seed":seed, "arm":arm, "protocol_sha256":h,
               "train":measure(model, base, device), "calibration":measure(model, cal, device),
               "last_loss":float(loss.detach())}
    torch.save({"seed":seed,"arm":arm,"protocol_sha256":h,"model":model.cpu().state_dict()}, ckpt)
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    print(f"seed={seed} arm={arm:7s} train={summary['train']['pair_accuracy']:.3f} "
          f"cal={summary['calibration']['pair_accuracy']:.3f} "
          f"cal_both={summary['calibration']['correct_both_rate']:.3f}", flush=True)


def load_model(seed, arm, root, h, device):
    d = torch.load(root/f"seed_{seed}"/arm/"final.pt", map_location="cpu", weights_only=True)
    assert d["seed"] == seed and d["arm"] == arm and d["protocol_sha256"] == h
    model = Model().to(device)
    model.load_state_dict(d["model"], strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("plan", "preflight", "train", "evaluate"), required=True)
    p.add_argument("--root", default="results/ml075_grammar")
    p.add_argument("--protocol", default="reports/ml075/frozen_protocol.json")
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    assert args.steps > 0
    doc, h, base, cal, doses, tests = build(args)
    protocol = Path(args.protocol)
    root = Path(args.root)
    if args.phase == "plan":
        if protocol.exists():
            freeze_check(protocol, doc, h)
        else:
            protocol.parent.mkdir(parents=True, exist_ok=True)
            protocol.write_text(json.dumps({**doc, "sha256":h}, indent=2)+"\n")
        print(f"ML-075 frozen {protocol} protocol={h[:12]}")
        return
    freeze_check(protocol, doc, h)
    if args.phase == "preflight":
        print(f"ML-075 preflight OK, protocol={h[:12]}")
        print(f"base={len(base)} calibration={len(cal)} "
              f"dose_control={len(doses['control'])} dose_sparse={len(doses['sparse'])} "
              f"dose_broad={len(doses['broad'])}")
        print("Example sentence:", " ".join(prompt(base[0])))
        print("Example target role answer:", gold(base[0]))
        print("Protected results NOT evaluated")
        return
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        for seed in SEEDS:
            for arm in ARMS:
                train_one(seed, arm, root, h, base, cal, doses, device, args.steps)
        print("Training complete. Protected outcomes untouched.")
        return
    out = root/"protected_evaluation.json"
    if out.exists():
        raise FileExistsError(f"Already evaluated: {out}")
    report = {"experiment":"ML-075", "protocol_sha256":h, "results":[]}
    for seed in SEEDS:
        for arm in ARMS:
            model = load_model(seed, arm, root, h, device)
            groups = {k:measure(model, c, device) for k,c in tests.items()}
            report["results"].append({"seed":seed,"arm":arm,"cohorts":groups})
            x = groups["target_double_unseen"]
            print(f"seed={seed} arm={arm:7s} doubleOOD_both={x['correct_both_rate']:.3f} "
                  f"choice_acc={x['pair_accuracy']:.3f} full_acc={x['full_vocab_accuracy']:.3f} "
                  f"same={x['same_answer_rate']:.3f}", flush=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    print("Paired double-OOD differences (both-role correct):")
    index={(r["seed"],r["arm"]):r["cohorts"]["target_double_unseen"]["correct_both_rate"]
           for r in report["results"]}
    for a,b in (("sparse","control"),("broad","control"),("broad","sparse")):
        diffs=[index[(s,a)]-index[(s,b)] for s in SEEDS]
        print(f"{a}-{b}: "+", ".join(f"{d:+.3f}" for d in diffs)+
              f" mean={statistics.mean(diffs):+.3f}")
    print("Saved:",out)


if __name__ == "__main__":
    main()
