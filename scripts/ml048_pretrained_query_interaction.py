
"""ML-048 — Prospective question-conditioned causal response in pretrained LLMs.

Test one nonzero, fixed donor-direction intervention per voice, applied
identically at the readout state to agent and patient questions about a
new event.

Calibrate the sign of a question interaction, freeze predictions,
then test held-out names and verbs.

Uses existing ML-028, ML-003, ML-012 utilities and MLX-LM.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

SPEC = {
    "experiment": "ML-048",
    "question": (
        "Does the same nonzero readout perturbation have "
        "question-dependent causal effects?"
    ),
    "models": {
        "qwen7b": 20,
        "llama3b": 14,
    },
    "donor": {
        "pair": ("Nora", "Clara"),
        "verb": "praised",
        "direction": 0,
    },
    "calibration": {
        "pairs_verbs": [
            ("Felix", "Rosa", "helped"),
        ],
    },
    "protected": {
        "pairs_verbs": [
            ("Maya", "Omar", "thanked"),
            ("Lena", "Iris", "followed"),
        ],
    },
    "voices": ("active", "passive"),
    "questions": ("agent", "patient"),
    "directions": (0, 1),
    "relative_state_norm": 0.10,
    "signflip_controls": 2,
    "gate": {
        "min_cal_accuracy": 0.75,
        "min_abs_cal_interaction": 0.20,
    },
    "primary": (
        "Sign of mean (I_true - mean I_signflip) "
        "on protected events, per voice"
    ),
    "sign_success": (
        "Predicted sign on >=3/4 protected events "
        "and mean sign agrees"
    ),
    "cautions": (
        "Calibration and protected data share a prompt template. "
        "Readout interventions can transfer answer preferences "
        "without establishing grammatical parsing."
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()

OUT = Path("results")


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def paths(model):
    return {
        phase: OUT / f"ml048_{model}_{phase}.json"
        for phase in (
            "calibration",
            "frozen",
            "protected",
        )
    }


def events(split):
    source = SPEC[split]["pairs_verbs"]

    return [
        {
            "pair": (left, right),
            "verb": verb,
            "direction": direction,
            "voice": voice,
        }
        for left, right, verb in source
        for direction in (0, 1)
        for voice in SPEC["voices"]
    ]


def choice(event, role):
    a, b = (
        event["pair"]
        if event["direction"] == 0
        else event["pair"][::-1]
    )

    return a if role == "agent" else b


def reference_prompt(tokenizer, event, role):
    from ml028_event_interchange import make_prompt

    return make_prompt(
        tokenizer,
        event["pair"],
        event["verb"],
        event["direction"],
        event["voice"],
        role,
    )


def safe_name_boundaries(tokenizer, prompt, pair):
    from ml012_cue_patch import encode_prompt

    before = encode_prompt(tokenizer, prompt)

    for name in pair:
        after = encode_prompt(
            tokenizer, prompt + name
        )

        if (
            after[:len(before)] != before
            or len(after) <= len(before)
        ):
            raise RuntimeError(
                "Answer tokenization boundary shifted "
                f"for {name!r}"
            )


def readout_state(model, tokenizer, prompt, layer):
    from ml028_event_interchange import (
        capture,
        positions,
    )
    from ml012_cue_patch import encode_prompt

    pos = positions(
        tokenizer,
        prompt,
        encode_prompt,
    )["readout"]

    state = capture(
        model,
        tokenizer,
        prompt,
        layer,
        pos,
        encode_prompt,
    )

    return pos, state


def scores(
    model,
    tokenizer,
    prompt,
    pair,
    layer=None,
    pos=None,
    replacement=None,
):
    from ml003_role_margins import log_likelihood
    from ml028_event_interchange import (
        answer_scores,
        patched_scores,
    )

    if replacement is None:
        return answer_scores(
            model,
            tokenizer,
            prompt,
            log_likelihood,
            pair,
        )

    return patched_scores(
        model,
        tokenizer,
        prompt,
        pair,
        layer,
        pos,
        replacement,
        log_likelihood,
    )


def margin(scores_by_name, event):
    agent = choice(event, "agent")
    patient = choice(event, "patient")

    return (
        scores_by_name[agent]
        - scores_by_name[patient]
    )


def norm(vector):
    import mlx.core as mx

    return float(
        mx.sqrt(
            mx.sum(
                vector.astype(mx.float32) ** 2
            )
        ).item()
    )


def signflip(direction, model_name, event, j):
    import numpy as np
    import mlx.core as mx

    payload = json.dumps(
        [HASH, model_name, event, j],
        sort_keys=True,
    ).encode()

    seed = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )

    generator = np.random.default_rng(seed)

    signs = (
        generator.integers(
            0,
            2,
            size=(int(direction.shape[0]),),
            dtype=np.int8,
        )
        * 2
        - 1
    )

    return (
        direction
        * mx.array(signs).astype(direction.dtype)
    )


def donor_direction(
    model,
    tokenizer,
    model_name,
    voice,
):
    import mlx.core as mx

    donor = SPEC["donor"]

    event = {
        "pair": tuple(donor["pair"]),
        "verb": donor["verb"],
        "direction": donor["direction"],
        "voice": voice,
    }

    layer = SPEC["models"][model_name]
    states = {}

    for role in SPEC["questions"]:
        prompt = reference_prompt(
            tokenizer, event, role
        )

        _, states[role] = readout_state(
            model, tokenizer, prompt, layer
        )

    direction = (
        states["agent"] - states["patient"]
    )

    mx.eval(direction)

    size = norm(direction)

    if size <= 1e-8:
        raise RuntimeError(
            "Zero donor direction for "
            f"{model_name} {voice}"
        )

    print(
        f"  donor {voice}: norm={size:.4f}",
        flush=True,
    )

    return direction


def probe_event(
    model,
    tokenizer,
    model_name,
    event,
    direction,
    check_identity=False,
):
    import mlx.core as mx

    layer = SPEC["models"][model_name]

    natural = {}
    readout = {}
    prompts = {}

    for role in SPEC["questions"]:
        prompt = reference_prompt(
            tokenizer,
            event,
            role,
        )

        safe_name_boundaries(
            tokenizer,
            prompt,
            event["pair"],
        )

        pos, state = readout_state(
            model,
            tokenizer,
            prompt,
            layer,
        )

        prompts[role] = prompt
        readout[role] = (pos, state)

        natural[role] = margin(
            scores(
                model,
                tokenizer,
                prompt,
                event["pair"],
            ),
            event,
        )

        if check_identity:
            nop = scores(
                model,
                tokenizer,
                prompt,
                event["pair"],
                layer,
                pos,
                state,
            )

            difference = abs(
                margin(nop, event)
                - natural[role]
            )

            if difference > 0.05:
                raise RuntimeError(
                    "No-op patch changed behavior: "
                    f"{model_name}, {event}, {role}"
                )

    # Normalize the donor direction to 10%
    # of the average native readout-state norm.
    # The SAME edit is used for both questions.
    strength = (
        SPEC["relative_state_norm"]
        * mean(
            norm(readout[r][1])
            for r in SPEC["questions"]
        )
    )

    edit = direction * (
        strength / norm(direction)
    )

    mx.eval(edit)

    if norm(edit) <= 1e-5:
        raise RuntimeError(
            "Normalized intervention unexpectedly zero"
        )

    interventions = [
        ("actual", edit),
    ]

    for j in range(SPEC["signflip_controls"]):
        interventions.append(
            (
                f"control_{j}",
                signflip(
                    edit,
                    model_name,
                    event,
                    j,
                ),
            )
        )

    effects = {}

    for label, vector in interventions:
        edited = {}

        for role in SPEC["questions"]:
            pos, state = readout[role]

            changed = scores(
                model,
                tokenizer,
                prompts[role],
                event["pair"],
                layer,
                pos,
                state + vector,
            )

            edited[role] = (
                margin(changed, event)
                - natural[role]
            )

        # Primary interaction:
        # identical vector, different question.
        effects[label] = (
            edited["agent"]
            - edited["patient"]
        )

    controls = [
        effects[f"control_{j}"]
        for j in range(
            SPEC["signflip_controls"]
        )
    ]

    return {
        "event": event,
        "baseline_margins": natural,
        "baseline_correct": {
            r: (
                natural[r] > 0
                if r == "agent"
                else natural[r] < 0
            )
            for r in SPEC["questions"]
        },
        "donor_edit_norm": norm(edit),
        "native_state_norm": (
            strength
            / SPEC["relative_state_norm"]
        ),
        "interactions": effects,
        "excess": (
            effects["actual"]
            - mean(controls)
        ),
    }


def collect(model_name, split, output):
    from mlx_lm import load
    from ml003_role_margins import MODELS

    location = MODELS / model_name

    if not location.exists():
        raise FileNotFoundError(
            f"Model not found: {location}"
        )

    cases = events(split)
    records = []

    if output.exists():
        previous = json.loads(
            output.read_text()
        )

        if (
            previous.get("design_hash") != HASH
            or previous.get("model") != model_name
            or previous.get("split") != split
            or previous.get("complete")
        ):
            raise RuntimeError(
                "Existing result is complete "
                f"or incompatible: {output}"
            )

        records = previous.get(
            "records", []
        )

        if len(records) > len(cases):
            raise RuntimeError(
                "Partial result contains too many cases"
            )

        for i, row in enumerate(records):
            expected = json.loads(
                json.dumps(cases[i])
            )

            if row["event"] != expected:
                raise RuntimeError(
                    f"Resume sequence mismatch at {i}"
                )

        print(
            f"RESUMING {split}: "
            f"{len(records)}/{len(cases)} complete",
            flush=True,
        )

    model, tokenizer = load(str(location))

    layers = len(
        model.model.layers
    )

    if SPEC["models"][model_name] >= layers:
        raise RuntimeError(
            f"Layer out of range ({layers} blocks)"
        )

    directions = {
        voice: donor_direction(
            model,
            tokenizer,
            model_name,
            voice,
        )
        for voice in SPEC["voices"]
    }

    remaining = cases[len(records):]

    for index, event in enumerate(
        remaining,
        start=len(records),
    ):
        row = probe_event(
            model,
            tokenizer,
            model_name,
            event,
            directions[event["voice"]],
            check_identity=(index < 2),
        )

        records.append(row)

        print(
            f"  {split} {index+1}/{len(cases)} "
            f"{event['voice']} "
            f"{event['pair']} "
            f"{event['direction']} "
            f"excess={row['excess']:+.4f}",
            flush=True,
        )

        # Save after every event, so an
        # interrupted run can safely resume.
        output.write_text(
            json.dumps(
                {
                    "design_hash": HASH,
                    "model": model_name,
                    "split": split,
                    "complete": False,
                    "records": records,
                },
                indent=2,
            )
        )

    output.write_text(
        json.dumps(
            {
                "design_hash": HASH,
                "model": model_name,
                "split": split,
                "complete": True,
                "records": records,
            },
            indent=2,
        )
    )

    print("SAVED", output)


def check_data(path, model, split):
    data = json.loads(
        path.read_text()
    )

    if (
        data.get("design_hash") != HASH
        or data.get("model") != model
        or data.get("split") != split
        or not data.get("complete")
    ):
        raise RuntimeError(
            f"Wrong/incomplete data: {path}"
        )

    if len(data.get("records", [])) != len(
        events(split)
    ):
        raise RuntimeError(
            f"Unexpected record count: {path}"
        )

    return data


def freeze(model_name):
    ps = paths(model_name)

    if ps["frozen"].exists():
        raise RuntimeError(
            "Already frozen: refusing overwrite"
        )

    data = check_data(
        ps["calibration"],
        model_name,
        "calibration",
    )

    predictions = {}

    for voice in SPEC["voices"]:
        rows = [
            x
            for x in data["records"]
            if x["event"]["voice"] == voice
        ]

        evidence = mean(
            x["excess"] for x in rows
        )

        baseline = mean(
            float(yes)
            for x in rows
            for yes in x[
                "baseline_correct"
            ].values()
        )

        eligible = (
            baseline >= SPEC["gate"][
                "min_cal_accuracy"
            ]
            and abs(evidence) >= SPEC["gate"][
                "min_abs_cal_interaction"
            ]
        )

        predictions[voice] = {
            "calibration_excess": evidence,
            "calibration_accuracy": baseline,
            "predicted_sign": (
                1 if evidence > 0 else -1
            ),
            "eligible": eligible,
        }

        print(
            f"FREEZE {voice}: "
            f"cal_excess={evidence:+.4f} "
            f"acc={baseline:.3f} "
            f"eligible={eligible}"
        )

    ps["frozen"].write_text(
        json.dumps(
            {
                "design_hash": HASH,
                "model": model_name,
                "calibration_sha256": sha(
                    ps["calibration"]
                ),
                "predictions": predictions,
            },
            indent=2,
        )
    )

    print(
        "FROZEN",
        ps["frozen"],
        "SHA256",
        sha(ps["frozen"]),
    )

    print(
        "Commit calibration and frozen files "
        "BEFORE the protected test."
    )


def test(model_name):
    import subprocess

    ps = paths(model_name)

    # Require calibration and predictions
    # to be committed before testing.
    for kind in (
        "calibration",
        "frozen",
    ):
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                str(ps[kind]),
            ],
            capture_output=True,
            text=True,
        )

        if tracked.returncode:
            raise RuntimeError(
                "Must commit "
                f"{ps[kind]} before protected test"
            )

        dirty = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                str(ps[kind]),
            ]
        )

        if dirty.returncode:
            raise RuntimeError(
                "Frozen input differs "
                f"from committed HEAD: {ps[kind]}"
            )

    if not ps["frozen"].exists():
        raise RuntimeError(
            "Missing freeze manifest"
        )

    frozen = json.loads(
        ps["frozen"].read_text()
    )

    if (
        frozen.get("design_hash") != HASH
        or frozen.get("model") != model_name
        or frozen.get(
            "calibration_sha256"
        ) != sha(ps["calibration"])
    ):
        raise RuntimeError(
            "Frozen design/calibration mismatch"
        )

    collect(
        model_name,
        "protected",
        ps["protected"],
    )

    protected = check_data(
        ps["protected"],
        model_name,
        "protected",
    )

    for voice in SPEC["voices"]:
        rec = [
            x
            for x in protected["records"]
            if x["event"]["voice"] == voice
        ]

        pred = frozen[
            "predictions"
        ][voice]

        sign = pred["predicted_sign"]

        positives = sum(
            x["excess"] * sign > 0
            for x in rec
        )

        tested_mean = mean(
            x["excess"] for x in rec
        )

        baseline = mean(
            float(correct)
            for x in rec
            for correct in x[
                "baseline_correct"
            ].values()
        )

        passed = (
            pred["eligible"]
            and positives >= 3
            and tested_mean * sign > 0
        )

        status = (
            "PASS"
            if passed
            else "FAIL/INELIGIBLE"
        )

        print(
            f"TEST {voice}: "
            f"predicted_sign={sign:+d} "
            f"heldout_mean={tested_mean:+.4f} "
            f"sign_match={positives}/{len(rec)} "
            f"baseline_acc={baseline:.3f} "
            f"eligible={pred['eligible']} "
            f"gate={status}"
        )

    print(
        "All protected results retained, "
        "including ineligible conditions."
    )


def audit():
    print(
        "ML-048 design SHA256:",
        HASH,
    )

    for name, layer in SPEC["models"].items():
        print(
            "MODEL",
            name,
            "readout block",
            layer,
        )

    for split in (
        "calibration",
        "protected",
    ):
        for event in events(split):
            print(
                split.upper(),
                event,
                "agent=",
                choice(event, "agent"),
                "patient=",
                choice(event, "patient"),
            )

    pairs = [
        SPEC["donor"]["pair"],
    ] + [
        (a, b)
        for split in (
            "calibration",
            "protected",
        )
        for a, b, _ in SPEC[split][
            "pairs_verbs"
        ]
    ]

    assert len({
        name
        for pair in pairs
        for name in pair
    }) == 8

    assert len(events("calibration")) == 4
    assert len(events("protected")) == 8

    print(
        "AUDIT PASS: disjoint names, "
        "four calibration events, "
        "eight protected events"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "audit",
            "calibrate",
            "freeze",
            "test",
        ),
    )

    parser.add_argument(
        "--model",
        choices=tuple(
            SPEC["models"]
        ),
        default="qwen7b",
    )

    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)

    if args.phase == "audit":
        audit()

    elif args.phase == "calibrate":
        collect(
            args.model,
            "calibration",
            paths(args.model)[
                "calibration"
            ],
        )

    elif args.phase == "freeze":
        freeze(args.model)

    elif args.phase == "test":
        test(args.model)


if __name__ == "__main__":
    main()
