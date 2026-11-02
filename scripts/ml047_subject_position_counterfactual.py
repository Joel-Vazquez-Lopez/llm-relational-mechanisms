
"""ML-047: Does causal participant routing follow first mention or subjecthood?

Spanish synthetic event grammar (one-name-per-token):
    active:   Nora elogió a Clara .                (subject=agent; first=agent)
    passive:  Clara fue elogiada por Nora .        (subject=patient; first=patient)
    fronted:  A Clara , Nora la elogió .          (subject=agent; first=patient)

The new training cohort sees active/passive for ALL verbs and object-fronting
for three verbs. Protected tests use a novel unordered name pair AND the verb
never seen in object-fronting. Every name and verb token is seen in training.

Training-only nomination selects a block-2 attention head showing first-subject
sensitivity in BOTH active/passive reference constructions. The head identity,
checkpoint SHA256s and eligibility decisions are frozen before protected tests.

On fronted-object probes the hypotheses diverge:
  subject hypothesis: agent harm > patient harm
  first-mention hypothesis: patient harm > agent harm

These are hypotheses about *causal sensitivity*, not direct proof of parsing.
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

NAMES = ("Maya", "Lena", "Omar", "Rosa", "Nora", "Clara", "Felix", "Iris")
GENDER = {"Maya": "f", "Lena": "f", "Omar": "m", "Rosa": "f", "Nora": "f", "Clara": "f", "Felix": "m", "Iris": "f"}
# Active, passive participle stem (add gender), participle's following clitic.
VERBS = (("ayudó", "ayudad"), ("elogió", "elogiad"),
         ("visitó", "visitad"), ("saludó", "saludad"))
VOICES = ("active", "passive", "fronted")
ROLES = ("agent", "patient")
SEEDS = (301, 307, 311, 313)
SPLIT_SEED = 4701
STEPS = 500
BATCH = 64
LR = 0.002
CONTROLS = 4
TRAIN_FRONTED_VERBS = (0, 1, 2)
HELDOUT_FRONTED_VERB = 3
CAL_THRESHOLD = 0.90  # Both ordinary and seen-verb fronting generalization.
MIN_NOMINATION_SCORE = 0.50  # Correct-minus-control logit units.
RESULTS = Path("results")


@dataclass(frozen=True)
class Case:
    pair: tuple[str, str]
    verb: int
    direction: int
    construction: str
    role: str


def participants(c):
    return c.pair if c.direction == 0 else c.pair[::-1]


def sentence(c):
    agent, patient = participants(c)
    v, stem = VERBS[c.verb]
    if c.construction == "active":
        return [agent, v, "a", patient, "."]
    if c.construction == "passive":
        return [patient, "fue", stem + ("a" if GENDER[patient] == "f" else "o"), "por", agent, "."]
    if c.construction == "fronted":
        # Left-dislocated topicalized object with obligatory clitic resumption.
        return ["A", patient, ",", agent, "la" if GENDER[patient] == "f" else "lo", v, "."]
    raise ValueError(c.construction)


def prompt(c):
    return ["<bos>"] + sentence(c) + ["<sep>", "¿", "quién",
            "realizó" if c.role == "agent" else "recibió", "la", "acción", "?", "<answer>"]


def split():
    pairs = [(NAMES[i], NAMES[j]) for i in range(len(NAMES)) for j in range(i + 1, len(NAMES))]
    random.Random(SPLIT_SEED).shuffle(pairs)
    result = {"train": pairs[:14], "calibration": pairs[14:21], "protected": pairs[21:]}
    assert [len(result[k]) for k in ("train", "calibration", "protected")] == [14, 7, 7]
    assert {n for p in result["train"] for n in p} == set(NAMES)
    return result


SPLIT = split()


def cases(pairs, constructions=VOICES, verbs=range(4)):
    return [Case(pair, verb, direction, construction, role)
            for pair in pairs for verb in verbs for direction in (0, 1)
            for construction in constructions for role in ROLES]


# Tokens are union of all prompts, including protected syntax but NOT new lexical
# items. Audit verifies every token has appeared in a training prompt.
ALL_CASES = cases([p for group in SPLIT.values() for p in group])
TRAIN_CASES = (cases(SPLIT["train"], ("active", "passive"))
               + cases(SPLIT["train"], ("fronted",), TRAIN_FRONTED_VERBS))
TOKENS = ("<pad>",) + tuple(sorted({t for c in ALL_CASES for t in prompt(c)} - {"<pad>"}))
VOCAB = {t: i for i, t in enumerate(TOKENS)}

SPEC = {
    "id": "ML-047", "source_model": "ml037 ScratchTransformer", "language": "es",
    "seeds": SEEDS, "split_seed": SPLIT_SEED, "split": SPLIT,
    "train_fronted_verbs": TRAIN_FRONTED_VERBS,
    "protected_fronted_verb": HELDOUT_FRONTED_VERB,
    "steps": STEPS, "batch": BATCH, "lr": LR, "controls": CONTROLS,
    "calibration_threshold": CAL_THRESHOLD, "nomination_threshold": MIN_NOMINATION_SCORE,
    "vocab": TOKENS,
    "architecture": {"layers": 2, "heads": 4, "width": 96},
    "nomination": "training-only max of min(active agent-patient harm, passive patient-agent harm); verbs 0,1",
    "main_endpoint": "natural swap on fronted verb 3 protected pairs: agent-harm minus patient-harm (positive=subject-compatible, negative=first-compatible)",
    "secondary": "same agent-question-derived vector applied to both role questions; inspect agent-v-patient logit-shift interaction",
    "protected_baseline_interpretability_threshold": 0.85,
    "reference_endpoint": "active agent-patient and passive patient-agent harm on protected pairs",
    "eligibility": "both active/passive and seen-verb fronted pair accuracy >= .90 on calibration; nomination score >= .50",
    "data_limit": "name pairs repeated; protected examples are clustered by pair and seed",
}
HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


class Model(ScratchTransformer):
    def __init__(self):
        super().__init__(width=96, heads=4, layers=2, max_len=40)
        self.embed = nn.Embedding(len(VOCAB), 96)
        self.output = nn.Linear(96, len(VOCAB))


def encode(batch):
    sequences = [prompt(c) for c in batch]
    ids = torch.full((len(batch), max(map(len, sequences))), VOCAB["<pad>"], dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor([VOCAB[t] for t in seq])
    pos = torch.tensor([len(seq) - 1 for seq in sequences])
    labels = torch.tensor([VOCAB[participants(c)[0 if c.role == "agent" else 1]] for c in batch])
    other = torch.tensor([VOCAB[participants(c)[1 if c.role == "agent" else 0]] for c in batch])
    source = torch.tensor([[seq.index(participants(c)[0]), seq.index(participants(c)[1])]
                           for c, seq in zip(batch, sequences)])
    return ids, pos, labels, other, source


@torch.no_grad()
def accuracy(model, examples, device):
    n = 0
    for i in range(0, len(examples), 128):
        ids, pos, labels, other, _ = encode(examples[i:i+128])
        logits = model(ids.to(device), pos.to(device))
        labels, other = labels.to(device), other.to(device)
        correct = logits.gather(1, labels[:, None]) > logits.gather(1, other[:, None])
        n += int(correct.sum().item())
    return n / len(examples)


def checkpoint(seed):
    return RESULTS / f"ml047_es_seed{seed}.pt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(seed, device):
    path = checkpoint(seed)
    if path.exists():
        raise RuntimeError(f"Checkpoint exists; refusing overwrite: {path}")
    torch.manual_seed(seed)
    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    ids, pos, labels, _, _ = encode(TRAIN_CASES)
    for step in range(1, STEPS + 1):
        model.train()
        sample = torch.randint(len(TRAIN_CASES), (BATCH,))
        logits = model(ids[sample].to(device), pos[sample].to(device))
        loss = F.cross_entropy(logits, labels[sample].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"seed={seed} step={step} loss={loss.item():.4f}", flush=True)
    model.eval()
    ref = accuracy(model, cases(SPLIT["calibration"], ("active", "passive")), device)
    topical = accuracy(model, cases(SPLIT["calibration"], ("fronted",), TRAIN_FRONTED_VERBS), device)
    torch.save({"seed": seed, "hash": HASH, "vocab": VOCAB,
                "cal_reference_accuracy": ref, "cal_fronted_seen_accuracy": topical,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
    print(f"TRAIN seed={seed} cal_active_passive={ref:.3f} cal_fronted_seen={topical:.3f} file={path}")


def load(seed, device, expected=None):
    path = checkpoint(seed)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f"Frozen checkpoint changed: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["hash"] != HASH or data["seed"] != seed or data["vocab"] != VOCAB:
        raise RuntimeError("Incompatible checkpoint")
    m = Model().to(device)
    m.load_state_dict(data["state"], strict=True)
    m.eval()
    return m, digest


@torch.no_grad()
def capture(model, batch, device):
    ids, pos, correct, other, locations = (t.to(device) for t in encode(batch))
    saved = {}
    mha = model.blocks[1].attn
    h1 = mha.register_forward_pre_hook(lambda mod, inputs: saved.update(z=inputs[0].detach()))
    h2 = mha.register_forward_hook(lambda mod, inputs, output: saved.update(out=output[0].detach()))
    try:
        logits = model(ids, pos)
    finally:
        h1.remove()
        h2.remove()
    z = saved["z"]
    b, length, width = z.shape
    heads, dim = mha.num_heads, width // mha.num_heads
    q, k, v = F.linear(z, mha.in_proj_weight, mha.in_proj_bias).split(width, -1)
    def reshape(x):
        return x.reshape(b, length, heads, dim).transpose(1, 2)
    q, k, v = map(reshape, (q, k, v))
    idx = torch.arange(b, device=device)
    scores = torch.einsum("bhd,bhld->bhl", q[idx, :, pos], k) / math.sqrt(dim)
    scores = scores.masked_fill(torch.arange(length, device=device)[None, None] > pos[:, None, None], float("-inf"))
    weights = scores.softmax(-1)
    context = torch.einsum("bhl,bhld->bhd", weights, v).reshape(b, width)
    reconstructed = F.linear(context, mha.out_proj.weight, mha.out_proj.bias)
    mismatch = (reconstructed - saved["out"][idx, pos]).abs().max().item()
    if mismatch > .005:
        raise RuntimeError(f"Attention reconstruction mismatch {mismatch}")
    agent, patient = locations[:, 0], locations[:, 1]
    wa, wp = weights[idx, :, agent], weights[idx, :, patient]
    va, vp = v[idx, :, agent], v[idx, :, patient]
    offsets = []
    for h in range(heads):
        W = mha.out_proj.weight[:, h*dim:(h+1)*dim]
        offsets.append((wp[:, h] - wa[:, h])[:, None] * F.linear(va[:, h] - vp[:, h], W))
    offsets = torch.stack(offsets, dim=1)
    margin = (logits.gather(1, correct[:, None]) - logits.gather(1, other[:, None])).squeeze(-1)
    return {"ids": ids, "pos": pos, "correct": correct, "other": other,
            "offsets": offsets, "baseline": margin, "heads": heads}


def random_signs(seed, key, head, control, width, device, dtype):
    # Intentionally deterministic across runs/hardware.
    payload = json.dumps([HASH, seed, key, head, control])
    number = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)
    g = torch.Generator(device="cpu").manual_seed(number)
    return (2 * torch.randint(0, 2, (width,), generator=g) - 1).to(device=device, dtype=dtype)


@torch.no_grad()
def effect(model, batch, seed, key, device, shared=False):
    cap = capture(model, batch, device)
    offsets = cap["offsets"]
    k = len(batch)
    variants = [torch.zeros_like(offsets[:, 0])]
    for head in range(cap["heads"]):
        delta = offsets[:, head].clone()
        if shared:
            # Each construction is represented by (agent question, patient
            # question), about exactly the same sentence. Apply the same
            # donor vector in both questions to control perturbation dose.
            assert k % 2 == 0
            delta[1::2] = delta[0::2]
        variants.append(delta)
        variants.extend(delta * random_signs(seed, key, head, j, delta.shape[-1], device, delta.dtype)
                        for j in range(CONTROLS))
    n = len(variants)
    edits = torch.stack(variants).reshape(n*k, -1)
    ids = cap["ids"].repeat(n, 1)
    pos = cap["pos"].repeat(n)
    idx = torch.arange(n*k, device=device)

    def hook(_module, _inputs, output):
        x, *rest = output
        updated = x.clone()
        updated[idx, pos] = updated[idx, pos] + edits.to(updated.dtype)
        return (updated, *rest)

    handle = model.blocks[1].attn.register_forward_hook(hook)
    try:
        logits = model(ids, pos).reshape(n, k, -1)
    finally:
        handle.remove()
    correct = cap["correct"][None, :, None].expand(n, -1, -1)
    other = cap["other"][None, :, None].expand(n, -1, -1)
    margins = (logits.gather(2, correct) - logits.gather(2, other)).squeeze(-1)
    if (margins[0] - cap["baseline"]).abs().max().item() > .05:
        raise RuntimeError("No-op intervention mismatch")
    result = []
    for head in range(cap["heads"]):
        j = 1 + head*(CONTROLS+1)
        swap = margins[j]
        controls = margins[j+1:j+1+CONTROLS]
        harm = controls.mean(0) - swap
        # Question-independent agent-minus-patient logit direction.
        # correct-other margin is negated for patient questions.
        d_shift = torch.where(
            torch.tensor([c.role == "agent" for c in batch], device=device),
            -harm, harm
        )
        result.append({"head": head, "harm": harm.cpu().tolist(),
                       "agent_patient_logit_shift": d_shift.cpu().tolist(),
                       "baseline_acc": (margins[0] > 0).float().cpu().tolist(),
                       "swap_acc": (swap > 0).float().cpu().tolist(),
                       "control_acc": (controls > 0).float().mean(0).cpu().tolist(),
                       "offset_norm": offsets[:, head].norm(dim=-1).cpu().tolist()})
    return result


def reference_quartet(pair, verb, direction):
    return [Case(pair, verb, direction, construction, role)
            for construction in ("active", "passive") for role in ROLES]


def fronted_pair(pair, verb, direction):
    return [Case(pair, verb, direction, "fronted", role) for role in ROLES]


def collect(model, seed, pairs, verbs, device, fronted=False, shared=False):
    rows = []
    for pair in pairs:
        for verb in verbs:
            for direction in (0, 1):
                batch = (fronted_pair(pair, verb, direction) if fronted
                         else reference_quartet(pair, verb, direction))
                rows.append({"pair": list(pair), "verb": verb, "direction": direction,
                             "results": effect(model, batch, seed, [list(pair), verb, direction, fronted, shared], device, shared=shared)})
    return rows


def aggregate(rows, head, order):
    head_rows = [r["results"][head] for r in rows]
    result = {}
    for j, name in enumerate(order):
        result[name] = {
            "harm": mean(r["harm"][j] for r in head_rows),
            "baseline_accuracy": mean(r["baseline_acc"][j] for r in head_rows),
            "swap_accuracy": mean(r["swap_acc"][j] for r in head_rows),
            "control_accuracy": mean(r["control_acc"][j] for r in head_rows),
            "offset_norm": mean(r["offset_norm"][j] for r in head_rows),
            "agent_patient_logit_shift": mean(r["agent_patient_logit_shift"][j] for r in head_rows),
        }
    return result


def nomination_score(summary):
    # First participant is subject in both reference constructions.
    active = summary["active_agent"]["harm"] - summary["active_patient"]["harm"]
    passive = summary["passive_patient"]["harm"] - summary["passive_agent"]["harm"]
    return min(active, passive), (active, passive)


def freeze(device):
    path = RESULTS / "ml047_frozen_nomination.json"
    if path.exists():
        raise RuntimeError("Frozen manifest exists; will not overwrite")
    manifest = {"design_hash": HASH, "spec": SPEC, "models": {}, "complete": True}
    for seed in SEEDS:
        model, digest = load(seed, device)
        cal_ref = accuracy(model, cases(SPLIT["calibration"], ("active", "passive")), device)
        cal_front = accuracy(model, cases(SPLIT["calibration"], ("fronted",), TRAIN_FRONTED_VERBS), device)
        rows = collect(model, seed, SPLIT["train"], (0, 1), device)
        summaries = [aggregate(rows, h, ("active_agent", "active_patient", "passive_agent", "passive_patient")) for h in range(4)]
        scores = [nomination_score(s)[0] for s in summaries]
        selected = max(range(4), key=lambda h: scores[h])
        eligible = cal_ref >= CAL_THRESHOLD and cal_front >= CAL_THRESHOLD and scores[selected] >= MIN_NOMINATION_SCORE
        manifest["models"][str(seed)] = {
            "checkpoint_sha256": digest, "head": selected,
            "cal_ref_accuracy": cal_ref, "cal_front_seen_accuracy": cal_front,
            "training_scores": scores,
            "training_voice_contrasts": nomination_score(summaries[selected])[1],
            "eligible": eligible,
        }
        print(f"FREEZE seed={seed} head={selected} cal_ref={cal_ref:.3f} cal_front={cal_front:.3f} score={scores[selected]:+.3f} eligible={eligible}", flush=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("FROZEN", path, "sha256", sha(path), "COMMIT THIS BEFORE TEST")


def test(device):
    mp = RESULTS / "ml047_frozen_nomination.json"
    op = RESULTS / "ml047_protected_results.json"
    if op.exists():
        raise RuntimeError("Protected results exist; refusing overwrite")
    frozen = json.loads(mp.read_text())
    if frozen["design_hash"] != HASH or not frozen["complete"] or set(frozen["models"]) != set(map(str, SEEDS)):
        raise RuntimeError("Frozen manifest mismatch")
    results = []
    for seed in SEEDS:
        record = frozen["models"][str(seed)]
        model, _ = load(seed, device, expected=record["checkpoint_sha256"])
        head = record["head"]
        # Primary: protected name pairs x never-fronted verb3.
        topical = collect(model, seed, SPLIT["protected"], (HELDOUT_FRONTED_VERB,), device, fronted=True)
        s = aggregate(topical, head, ("fronted_agent", "fronted_patient"))
        contrast = s["fronted_agent"]["harm"] - s["fronted_patient"]["harm"]
        # Secondary: identical agent-question-derived attention offset is
        # applied to both question recipients. This is NOT a literal swap
        # in the patient-question recipient; it tests downstream sensitivity.
        shared_rows = collect(model, seed, SPLIT["protected"], (HELDOUT_FRONTED_VERB,), device,
                              fronted=True, shared=True)
        shared = aggregate(shared_rows, head, ("fronted_agent", "fronted_patient"))
        question_interaction = (
            shared["fronted_agent"]["agent_patient_logit_shift"]
            - shared["fronted_patient"]["agent_patient_logit_shift"]
        )
        # Secondary: same protected events, unfronted voice-reference conditions.
        references = collect(model, seed, SPLIT["protected"], (HELDOUT_FRONTED_VERB,), device)
        ref = aggregate(references, head, ("active_agent", "active_patient", "passive_agent", "passive_patient"))
        protected_competent = (
            s["fronted_agent"]["baseline_accuracy"] >= .85
            and s["fronted_patient"]["baseline_accuracy"] >= .85
        )
        record_out = {"seed": seed, "eligible": record["eligible"], "head": head,
                      "protected_fronted_baseline_competent": protected_competent,
                      "fronted_agent_minus_patient_harm": contrast,
                      "interpretation_if_competent": (
                          "not-interpretable" if not protected_competent else
                          "subject-compatible" if contrast >= .5 else
                          "first-mention-compatible" if contrast <= -.5 else "undetermined"),
                      "shared_question_interaction": question_interaction,
                      "fronted": s, "fronted_shared_donor": shared, "reference": ref,
                      "raw_fronted": topical, "raw_fronted_shared_donor": shared_rows,
                      "raw_reference": references}
        results.append(record_out)
        print(f"TEST seed={seed} eligible={record['eligible']} H{head} fronted(agent-patient)={contrast:+.3f} "
              f"agent_acc={s['fronted_agent']['baseline_accuracy']:.3f} "
              f"patient_acc={s['fronted_patient']['baseline_accuracy']:.3f} "
              f"swap_agent_acc={s['fronted_agent']['swap_accuracy']:.3f} "
              f"swap_patient_acc={s['fronted_patient']['swap_accuracy']:.3f} "
              f"shared_question_interaction={question_interaction:+.3f} "
              f"protected_competent={protected_competent}", flush=True)
    op.write_text(json.dumps({"design_hash": HASH, "manifest_sha256": sha(mp),
                              "complete": True, "results": results}, indent=2, ensure_ascii=False))
    eligible = [r for r in results if r["eligible"]]
    print(f"ELIGIBLE {len(eligible)}/{len(SEEDS)}")
    if eligible:
        print("Mean eligible fronted contrast:", mean(r["fronted_agent_minus_patient_harm"] for r in eligible))
        print("Interpretations:", {x: sum(r["interpretation_if_competent"] == x for r in eligible)
                                   for x in ("subject-compatible", "first-mention-compatible", "undetermined", "not-interpretable")})
    print("SAVED", op)


def audit(device):
    print("DESIGN", HASH)
    print("SPLIT", {k: len(v) for k, v in SPLIT.items()})
    print("TRAIN", len(TRAIN_CASES), "VOCAB", len(VOCAB))
    seen = {t for c in TRAIN_CASES for t in prompt(c)}
    assert seen == set(TOKENS) - {"<pad>"}, "Unseen lexical token in protected constructions"
    for con in VOICES:
        c = Case(("Nora", "Clara"), 1, 0, con, "agent")
        print(con, " ".join(prompt(c)))
    # The three conditions genuinely dissociate subject and first mention.
    assert sentence(Case(("Nora", "Clara"), 1, 0, "active", "agent"))[0] == "Nora"
    assert sentence(Case(("Nora", "Clara"), 1, 0, "passive", "agent"))[0] == "Clara"
    assert sentence(Case(("Nora", "Clara"), 1, 0, "fronted", "agent"))[1] == "Clara"
    torch.manual_seed(43)
    model = Model().to(device).eval()
    e = effect(model, reference_quartet(("Nora", "Clara"), 1, 0), 43, ["audit-ref"], device)
    f = effect(model, fronted_pair(("Nora", "Clara"), 1, 0), 43, ["audit-front"], device)
    fs = effect(model, fronted_pair(("Nora", "Clara"), 1, 0), 43, ["audit-front-shared"], device, shared=True)
    assert len(e) == len(f) == len(fs) == 4 and len(e[0]["harm"]) == 4 and len(f[0]["harm"]) == len(fs[0]["harm"]) == 2
    print("AUDIT PASS: lexical coverage, grammar, attention reconstruction, head swaps and matched controls")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=("audit", "train", "freeze", "test"))
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    a = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if a.device == "auto" else a.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    RESULTS.mkdir(exist_ok=True)
    print(f"ML-047 {a.phase} device={device} hash={HASH}", flush=True)
    if a.phase == "audit": audit(device)
    elif a.phase == "train":
        if a.seed is None: p.error("--phase train requires --seed")
        train(a.seed, device)
    elif a.phase == "freeze": freeze(device)
    else: test(device)


if __name__ == "__main__":
    main()
