
"""ML-035: Trace causal influence of Qwen voice x question interaction vectors.

At selected layers, build delta = h11 - h10 - h01 + h00 for the
same-event, two-voice, two-question factorial. Add +/- a scaled delta
to natural h00. Compare with two norm-matched signflip directions.

Exploratory on the original 16 ML-033/034 recipients, NOT held-out.
"""
import argparse
import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

LAYERS = (3, 7, 13, 20)
RELATIVE_NORM = 0.20
N_CONTROLS = 2
RESULTS = Path("results")

from ml028_event_interchange import (
    PAIRS, VOICES, QUESTIONS, gold, capture,
    patched_scores, answer_scores, gold_margin,
)
from ml033_causal_compatibility import (
    recipient_cases, validate_audit,
)
from ml034_additivity import HASH as ML034_HASH

SPEC = {
    "experiment": "ML-035",
    "model": "qwen7b",
    "site": "readout",
    "layers": LAYERS,
    "recipient_cases": "frozen 16 cases from ML-033",
    "delta": "h11-h10-h01+h00 at fixed event and verb",
    "strength": "scale each delta to 20% of local recipient state norm",
    "controls": "two deterministic signflips with same coordinate magnitudes",
    "effects": "paired odd response [f(h+v)-f(h-v)]/2",
    "primary": "L20 true-minus-control odd response mean >=+1, positive >=12/16",
    "source_ml034": ML034_HASH,
    "status": "exploratory causal localization, not a circuit proof",
}
HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def source_check():
    matches = []
    for path in RESULTS.glob("ml034_qwen7b_additivity_*.json"):
        data = json.loads(path.read_text())
        if data.get("hash") == ML034_HASH and data.get("complete"):
            if len(data.get("records", [])) == 32:
                matches.append(path)
    if not matches:
        raise RuntimeError("Missing complete 32-record ML-034 source")
    return sorted(matches)[-1]


def seed_for(case, layer, control):
    blob = json.dumps([HASH, list(case), layer, control]).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def summarize(data):
    cases = recipient_cases()
    if not data.get("complete") or data.get("hash") != HASH:
        raise RuntimeError("Incomplete or wrong design hash")
    rows = data["records"]
    if len(rows) != len(cases) * len(LAYERS) * (1 + N_CONTROLS) * 2:
        raise RuntimeError("Unexpected record count")
    by_key = {}
    for r in rows:
        case = tuple(r["case"])
        key = (case, int(r["layer"]), r["kind"], int(r["control"]), int(r["sign"]))
        if key in by_key:
            raise RuntimeError(f"Duplicate record: {key}")
        by_key[key] = float(r["effect"])
    print("\n=== ML-035 INTERACTION VECTOR UPSTREAM TRACE ===")
    for layer in LAYERS:
        odd_true, odd_control, excess, geometry = [], [], [], []
        for case in cases:
            def odd(kind, control):
                return (
                    by_key[case, layer, kind, control, 1]
                    - by_key[case, layer, kind, control, -1]
                ) / 2
            t = odd("true", -1)
            c = mean(odd("flip", j) for j in range(N_CONTROLS))
            odd_true.append(t)
            odd_control.append(c)
            excess.append(t - c)
            geometry.append(next(r["delta_over_h"] for r in rows
                                 if tuple(r["case"]) == case and r["layer"] == layer))
        print(
            f"L{layer:02d} delta/h={mean(geometry):.3f} "
            f"true_odd={mean(odd_true):+.3f} "
            f"positive={sum(v > 0 for v in odd_true)}/16 "
            f"control_odd={mean(odd_control):+.3f} "
            f"excess={mean(excess):+.3f} "
            f"excess_pos={sum(v > 0 for v in excess)}/16"
        )
        if layer == 20:
            passed = (mean(excess) >= 1.0 and sum(v > 0 for v in excess) >= 12)
            print("FROZEN L20 TRANSFER GATE:", "PASS" if passed else "FAIL")
    print("Exploratory layers, shared stimuli; not evidence of creation in a specific block.")


def collect():
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from ml003_role_margins import MODELS, log_likelihood
    from ml012_cue_patch import encode_prompt

    source = source_check()
    print("Historical checkpoint:", source)
    model, tokenizer = load(str(MODELS / "qwen7b"))
    prompts = validate_audit(tokenizer, encode_prompt)
    print("Audited prompts:", len(prompts))
    if max(LAYERS) >= len(model.model.layers):
        raise RuntimeError("Layer index out of range")
    RESULTS.mkdir(exist_ok=True)
    matching = []
    for path in sorted(RESULTS.glob("ml035_qwen7b_trace_*.json")):
        obj = json.loads(path.read_text())
        if obj.get("hash") == HASH:
            matching.append((path, obj))
    if any(obj.get("complete") for _, obj in matching):
        raise RuntimeError("A complete matching ML-035 run already exists")
    if matching:
        output, data = matching[-1]
        if data.get("source") != str(source):
            raise RuntimeError("Historical source changed")
        per_case = len(LAYERS) * (1 + N_CONTROLS) * 2
        if len(data["records"]) % per_case:
            raise RuntimeError("Incomplete recipient checkpoint")
        done = len(data["records"]) // per_case
        for idx in range(done):
            for row in data["records"][idx*per_case:(idx+1)*per_case]:
                if tuple(row["case"]) != recipient_cases()[idx]:
                    raise RuntimeError("Checkpoint case order mismatch")
        print(f"Resuming after {done}/16 recipients")
    else:
        done = 0
        output = RESULTS / f"ml035_qwen7b_trace_{datetime.now():%Y%m%dT%H%M%S%f}.json"
        data = {"hash": HASH, "complete": False, "source": str(source), "records": []}
        output.write_text(json.dumps(data, indent=2))
    group = None
    states = {}
    for count, case in enumerate(recipient_cases()[done:], done + 1):
        pid, verb, direction, voice, role = case
        if (pid, verb) != group:
            group = (pid, verb)
            states = {}
            for layer in LAYERS:
                for d, v, q in product((0, 1), VOICES, QUESTIONS):
                    prompt, position = prompts[pid, verb, d, v, q]
                    states[layer, d, v, q] = capture(
                        model, tokenizer, prompt, layer, position, encode_prompt
                    )
        pair = PAIRS[pid]
        prompt, position = prompts[case]
        correct = gold(pair, direction, role)
        baseline = gold_margin(
            answer_scores(model, tokenizer, prompt, log_likelihood, pair), correct, pair
        )
        other_voice = next(v for v in VOICES if v != voice)
        other_role = next(q for q in QUESTIONS if q != role)
        new_rows = []
        for layer in LAYERS:
            h00 = states[layer, direction, voice, role]
            h10 = states[layer, direction, other_voice, role]
            h01 = states[layer, direction, voice, other_role]
            h11 = states[layer, direction, other_voice, other_role]
            delta = h11 - h10 - h01 + h00
            dnorm = float(mx.sqrt(mx.sum(delta * delta)))
            hnorm = float(mx.sqrt(mx.sum(h00 * h00)))
            if dnorm < 1e-9 or hnorm < 1e-9:
                raise RuntimeError(f"Degenerate norm in {case} L{layer}")
            direction_vector = delta * (RELATIVE_NORM * hnorm / dnorm)
            mx.eval(direction_vector)
            self_scores = patched_scores(
                model, tokenizer, prompt, pair, layer, position, h00, log_likelihood
            )
            self_effect = gold_margin(self_scores, correct, pair) - baseline
            if abs(self_effect) > 0.05:
                raise RuntimeError(f"Self patch failed: {case} L{layer} {self_effect:+.4f}")
            vectors = [("true", -1, direction_vector)]
            for j in range(N_CONTROLS):
                rng = np.random.default_rng(seed_for(case, layer, j))
                signs = mx.array(rng.choice((-1.0, 1.0), size=(delta.shape[-1],)).astype(np.float32))
                signs = signs.astype(delta.dtype)
                vectors.append(("flip", j, direction_vector * signs))
            for kind, ci, vec in vectors:
                for sign in (1, -1):
                    state = h00 + sign * vec
                    mx.eval(state)
                    scores = patched_scores(
                        model, tokenizer, prompt, pair, layer, position, state, log_likelihood
                    )
                    effect = gold_margin(scores, correct, pair) - baseline
                    new_rows.append({
                        "case": list(case), "layer": layer, "kind": kind,
                        "control": ci, "sign": sign, "effect": effect,
                        "delta_over_h": dnorm / hnorm,
                        "baseline_margin": baseline,
                    })
        if len(new_rows) != len(LAYERS) * (1 + N_CONTROLS) * 2:
            raise RuntimeError("Missing case measurements")
        data["records"].extend(new_rows)
        output.write_text(json.dumps(data, indent=2))
        print(f"DONE {count}/16 {case}, records={len(data['records'])}")
    data["complete"] = True
    output.write_text(json.dumps(data, indent=2))
    print("Saved:", output)
    summarize(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("design", "audit", "collect", "summarize"), default="design")
    args = parser.parse_args()
    print("ML-035", args.phase, "qwen7b", HASH)
    if args.phase == "design":
        print(json.dumps(SPEC, indent=2))
        print("Cases:", *recipient_cases(), sep="\n  ")
        return
    source = source_check()
    print("Prior complete ML-034:", source)
    if args.phase == "summarize":
        paths = sorted(RESULTS.glob("ml035_qwen7b_trace_*.json"), reverse=True)
        for path in paths:
            data = json.loads(path.read_text())
            if data.get("hash") == HASH and data.get("complete"):
                print("Reading:", path)
                summarize(data)
                return
        raise RuntimeError("No complete ML-035 result")
    if args.phase == "audit":
        from mlx_lm import load
        from ml003_role_margins import MODELS
        from ml012_cue_patch import encode_prompt
        model, tokenizer = load(str(MODELS / "qwen7b"))
        prepared = validate_audit(tokenizer, encode_prompt)
        print("Audited prompts:", len(prepared))
        print("Available blocks:", len(model.model.layers))
        if max(LAYERS) >= len(model.model.layers):
            raise RuntimeError("Selected layer does not exist")
        return
    collect()


if __name__ == "__main__":
    main()
