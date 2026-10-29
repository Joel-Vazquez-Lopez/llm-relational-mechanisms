
"""ML-031: Donor voice x event x question, L20 readout.

Reuse completed ML-029 opposite-voice results; collect same-voice donors.
This is a within-stimulus exploratory causal control, not held-out validation.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

from ml029_answer_event_disentangle import (
    PAIRS, VERBS, VOICES, QUESTIONS, gold, make_prompt, positions,
    answer_scores, gold_margin, capture, patched_scores,
    DESIGN_HASH as ML029_HASH,
)

LAYER = 20
DESIGN = {
    "experiment": "ML-031", "layer": LAYER, "site": "readout",
    "stimuli": "ML-029 reused (exploratory)",
    "donor_voice": "same vs opposite",
    "factors": ["event_reversed", "question_reversed"],
    "opposite_voice_source_hash": ML029_HASH,
}
HASH = hashlib.sha256(json.dumps(DESIGN, sort_keys=True).encode()).hexdigest()
RESULTS = Path("results")


def get_file(pattern):
    paths = sorted(RESULTS.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[-1]


def signature(row):
    return (
        row["pair_id"], row["verb"], row["recipient_direction"],
        row["recipient_voice"], row["recipient_question"],
        int(row["event_reversed"]), int(row["question_reversed"]),
    )


def load_records(path, hash_value, count):
    data = json.loads(path.read_text())
    if (not data.get("complete") or data.get("design_hash") != hash_value
            or len(data.get("records", [])) != count):
        raise RuntimeError(f"Incomplete/wrong-design results: {path}")
    out = {}
    for row in data["records"]:
        key = signature(row)
        if key in out:
            raise RuntimeError(f"Duplicate case: {key}")
        out[key] = row["effect"]
    if len(out) != count:
        raise RuntimeError("Unique-record count mismatch")
    return out


def factorial(e):
    return {
        "event": (e[1, 0] + e[1, 1] - e[0, 0] - e[0, 1]) / 2,
        "question": (e[0, 1] + e[1, 1] - e[0, 0] - e[1, 0]) / 2,
        "answer_mismatch_interaction": (
            e[1, 0] + e[0, 1] - e[0, 0] - e[1, 1]
        ) / 2,
    }


def summarize(model):
    opposite_path = get_file(f"ml029_{model}_*.json")
    same_path = get_file(f"ml031_{model}_samevoice_*.json")
    opposite = load_records(opposite_path, ML029_HASH, 128)
    same = load_records(same_path, HASH, 128)
    if set(opposite) != set(same):
        raise RuntimeError("ML-029 and ML-031 condition sets differ")
    print("Opposite-voice data:", opposite_path)
    print("Same-voice data:", same_path)

    cases = defaultdict(dict)
    for key in same:
        cases[key[:5]][key[5:]] = (same[key], opposite[key])
    if len(cases) != 32 or any(len(v) != 4 for v in cases.values()):
        raise RuntimeError("Incomplete 2x2x2 factorial")

    for voice_index, label in enumerate(("SAME", "OPPOSITE")):
        print(f"\n=== DONOR VOICE {label} ===")
        for ev, qu in product((0, 1), repeat=2):
            values = [e[ev, qu][voice_index] for e in cases.values()]
            print(f"event={ev} question={qu} mean={mean(values):+.3f} "
                  f"negative={sum(x < 0 for x in values)}/32")
        for role in ("ALL", "agent", "patient"):
            relevant = [e for k, e in cases.items() if role == "ALL" or k[4] == role]
            values = [factorial({k: v[voice_index] for k, v in e.items()})
                      for e in relevant]
            print(f"  recipient={role} (n={len(values)})")
            for factor in ("event", "question", "answer_mismatch_interaction"):
                xs = [v[factor] for v in values]
                print(f"    {factor}: mean={mean(xs):+.3f} "
                      f"neg={sum(x < 0 for x in xs)}/{len(xs)}")

    print("\n=== EFFECT OF CHANGING DONOR VOICE ===")
    for ev, qu in product((0, 1), repeat=2):
        delta = [e[ev, qu][1] - e[ev, qu][0] for e in cases.values()]
        print(f"event={ev} question={qu}: opposite-minus-same="
              f"{mean(delta):+.3f}")

    for factor in ("event", "question", "answer_mismatch_interaction"):
        differences = []
        for e in cases.values():
            same_f = factorial({k: v[0] for k, v in e.items()})
            opp_f = factorial({k: v[1] for k, v in e.items()})
            differences.append(opp_f[factor] - same_f[factor])
        print(f"voice x {factor}: mean={mean(differences):+.3f} "
              f"positive={sum(x > 0 for x in differences)}/32")


def collect(model_name, audit_only):
    from mlx_lm import load
    from ml003_role_margins import MODELS, log_likelihood
    from ml012_cue_patch import encode_prompt

    model, tokenizer = load(str(MODELS / model_name))
    prompts = {}
    for key in product(range(len(PAIRS)), VERBS, (0, 1), VOICES, QUESTIONS):
        pid, verb, direction, voice, question = key
        prompt = make_prompt(tokenizer, PAIRS[pid], verb, direction, voice, question)
        position = positions(tokenizer, prompt, encode_prompt)["readout"]
        prompts[key] = (prompt, position)
    if len(prompts) != 32:
        raise RuntimeError("Expected 32 audited prompts")
    print("Audited prompts:", len(prompts))
    if audit_only:
        return

    opposite_path = get_file(f"ml029_{model_name}_*.json")
    opposite = load_records(opposite_path, ML029_HASH, 128)
    print("Validated existing opposite-voice data:", opposite_path)

    states = {}
    for key, (prompt, position) in prompts.items():
        states[key] = capture(model, tokenizer, prompt, LAYER, position, encode_prompt)

    output = RESULTS / f"ml031_{model_name}_samevoice_{datetime.now():%Y%m%dT%H%M%S%f}.json"
    data = {"design_hash": HASH, "complete": False, "model": model_name,
            "ml029_source": str(opposite_path), "records": []}
    for key, (prompt, position) in prompts.items():
        pid, verb, direction, voice, question = key
        pair = PAIRS[pid]
        expected = gold(pair, direction, question)
        baseline_scores = answer_scores(model, tokenizer, prompt, log_likelihood, pair)
        baseline = gold_margin(baseline_scores, expected, pair)
        for ev, qu in product((0, 1), repeat=2):
            donor_direction = 1 - direction if ev else direction
            donor_question = next(q for q in QUESTIONS if q != question) if qu else question
            donor_key = (pid, verb, donor_direction, voice, donor_question)
            scores = patched_scores(
                model, tokenizer, prompt, pair, LAYER,
                position, states[donor_key], log_likelihood,
            )
            effect = gold_margin(scores, expected, pair) - baseline
            if ev == 0 and qu == 0 and abs(effect) > 0.05:
                raise RuntimeError(f"Self patch failed: {key}: {effect}")
            row = dict(pair_id=pid, verb=verb, recipient_direction=direction,
                       recipient_voice=voice, recipient_question=question,
                       event_reversed=bool(ev), question_reversed=bool(qu),
                       effect=effect)
            if signature(row) not in opposite:
                raise RuntimeError(f"Cannot match ML-029: {signature(row)}")
            data["records"].append(row)
        output.write_text(json.dumps(data, indent=2))
        print("DONE", key, "records:", len(data["records"]))
    data["complete"] = True
    output.write_text(json.dumps(data, indent=2))
    print("Saved:", output)
    summarize(model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llama3b", "qwen7b"), default="qwen7b")
    parser.add_argument("--phase", choices=("design", "audit", "collect", "summarize"), default="design")
    args = parser.parse_args()
    print("ML-031", args.phase, args.model, "hash:", HASH)
    if args.phase == "design":
        print(json.dumps(DESIGN, indent=2))
    elif args.phase == "summarize":
        summarize(args.model)
    else:
        collect(args.model, args.phase == "audit")


if __name__ == "__main__":
    main()
