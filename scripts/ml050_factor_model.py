
"""ML-050: shared linguistic factors, independently fitted model-specific weights.

A behavioral prediction experiment, NOT a mechanistic or causal demonstration.
Qwen/Llama separately: calibrate on English relational events involving three
participants, freeze additive and context-conditioned linear factor models,
then predict candidate log-likelihood preferences on disjoint names and verbs.

Three candidates make semantic, position and distractor heuristics distinguishable.
All factor definitions, splits, regularization and scoring are fixed in advance.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

RESULTS = Path("results")
SCRIPT = Path("scripts/ml050_factor_model.py")
MODELS = ("qwen7b", "llama3b")
CONSTRUCTIONS = ("active", "passive", "object_cleft")
ORDERS = ("main_first", "distractor_first")
ROLES = ("agent", "patient")

# Every calibration name and verb is absent from protected contexts.
CALIBRATION = (
    ("Alice", "Bob", "Carol", "praised"),
    ("Grace", "Henry", "Jack", "helped"),
)

PROTECTED = (
    ("Maya", "Omar", "Lena", "thanked"),
    ("Sophie", "Daniel", "Iris", "followed"),
)

ADDITIVE = (
    "semantic",
    "first",
    "last",
    "main_subject",
    "distractor",
    "cleft_focus",
)

INTERACTIONS = (
    "semantic_passive",
    "semantic_cleft",
    "first_passive",
    "first_cleft",
    "first_agent_question",
    "distractor_first",
)

FAMILIES = {
    "semantic_only": ("semantic",),
    "additive": ADDITIVE,
    "contextual": ADDITIVE + INTERACTIONS,
}

LAMBDA = 8.0

SPEC = {
    "experiment": "ML-050",
    "models": MODELS,
    "calibration": CALIBRATION,
    "protected": PROTECTED,
    "constructions": CONSTRUCTIONS,
    "orders": ORDERS,
    "roles": ROLES,
    "candidate_count": 3,
    "families": FAMILIES,
    "ridge_lambda": LAMBDA,
    "output": (
        "centered candidate log-likelihood, "
        "from exact complete-name continuations"
    ),
    "primary": (
        "protected mean squared error "
        "of candidate-centered log-likelihood"
    ),
    "comparison": (
        "contextual vs additive vs semantic_only; "
        "report all, no posthoc selection"
    ),
    "interpretation": (
        "shared analyst-defined behavioral features; "
        "model-specific coefficients; not a neural circuit"
    ),
    "candidate_length": (
        "report unequal answer-token counts, "
        "do not filter posthoc"
    ),
    "success_gate": (
        "contextual protected MSE at most "
        "0.90 * additive protected MSE and "
        "0.90 * semantic_only protected MSE"
    ),
    "scope": (
        "distinct names and verbs; templates and "
        "3 construction families repeated"
    ),
}

HASH = hashlib.sha256(
    json.dumps(SPEC, sort_keys=True).encode()
).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def paths(model):
    return {
        key: RESULTS / f"ml050_{model}_{key}.json"
        for key in (
            "calibration",
            "frozen",
            "protected",
        )
    }


def make_cases(split):
    source = (
        CALIBRATION
        if split == "calibration"
        else PROTECTED
    )

    out = []

    for a, b, distractor, verb in source:
        for direction in (0, 1):
            agent, patient = (
                (a, b)
                if direction == 0
                else (b, a)
            )

            for construction in CONSTRUCTIONS:
                if construction == "active":
                    main = (
                        f"At dawn, {agent} "
                        f"{verb} {patient}."
                    )
                    main_order = (agent, patient)
                    subject = agent
                    cleft_focus = None

                elif construction == "passive":
                    main = (
                        f"At dawn, {patient} was "
                        f"{verb} by {agent}."
                    )
                    main_order = (patient, agent)
                    subject = patient
                    cleft_focus = None

                else:
                    main = (
                        f"At dawn, it was {patient} "
                        f"that {agent} {verb}."
                    )
                    main_order = (patient, agent)
                    subject = agent
                    cleft_focus = patient

                side = (
                    f"At dusk, {distractor} "
                    "waited nearby."
                )

                for order in ORDERS:
                    if order == "main_first":
                        passage = main + " " + side
                        mentions = (
                            *main_order,
                            distractor,
                        )
                    else:
                        passage = side + " " + main
                        mentions = (
                            distractor,
                            *main_order,
                        )

                    for role in ROLES:
                        question = (
                            "At dawn, who performed "
                            "the action?"
                            if role == "agent"
                            else
                            "At dawn, who received "
                            "the action?"
                        )

                        out.append({
                            "id": (
                                f"{split}_{a}_{b}_{verb}_"
                                f"{direction}_{construction}_"
                                f"{order}_{role}"
                            ),
                            "split": split,
                            "agent": agent,
                            "patient": patient,
                            "distractor": distractor,
                            "verb": verb,
                            "construction": construction,
                            "order": order,
                            "role": role,
                            "subject": subject,
                            "cleft_focus": cleft_focus,
                            "mentions": list(mentions),
                            "candidates": [
                                a, b, distractor
                            ],
                            "gold": (
                                agent
                                if role == "agent"
                                else patient
                            ),
                            "passage": passage,
                            "question": question,
                        })

    return out


def feature(case, candidate):
    semantic = float(
        candidate == case["gold"]
    )

    first = float(
        candidate == case["mentions"][0]
    )

    last = float(
        candidate == case["mentions"][-1]
    )

    subject = float(
        candidate == case["subject"]
    )

    distractor = float(
        candidate == case["distractor"]
    )

    cleft = float(
        candidate == case["cleft_focus"]
        and case["cleft_focus"] is not None
    )

    passive = float(
        case["construction"] == "passive"
    )

    is_cleft = float(
        case["construction"] == "object_cleft"
    )

    agent_q = float(
        case["role"] == "agent"
    )

    side_first = float(
        case["order"] == "distractor_first"
    )

    return {
        "semantic": semantic,
        "first": first,
        "last": last,
        "main_subject": subject,
        "distractor": distractor,
        "cleft_focus": cleft,
        "semantic_passive": (
            semantic * passive
        ),
        "semantic_cleft": (
            semantic * is_cleft
        ),
        "first_passive": (
            first * passive
        ),
        "first_cleft": (
            first * is_cleft
        ),
        "first_agent_question": (
            first * agent_q
        ),
        "distractor_first": (
            distractor * side_first
        ),
    }


def features(case, names):
    X = np.array(
        [
            [
                feature(case, candidate)[f]
                for f in names
            ]
            for candidate in case["candidates"]
        ],
        dtype=np.float64,
    )

    return X - X.mean(
        axis=0,
        keepdims=True,
    )


def labels(record):
    values = np.array(
        [
            record["logprobs"][candidate]
            for candidate in record["candidates"]
        ],
        dtype=np.float64,
    )

    return values - values.mean()


def fit(records, names):
    XX = []
    yy = []

    for row in records:
        XX.append(
            features(row, names)
        )
        yy.append(
            labels(row)
        )

    X = np.concatenate(XX)
    y = np.concatenate(yy)

    coef = np.linalg.solve(
        X.T @ X
        + LAMBDA * np.eye(len(names)),
        X.T @ y,
    )

    return coef


def predict(case, names, coef):
    return (
        features(case, names)
        @ np.asarray(
            coef,
            dtype=np.float64,
        )
    )


def metrics(rows, predictions):
    mse = []
    choice_matches = []
    gold_matches = []

    for row, prediction in zip(
        rows, predictions
    ):
        values = labels(row)

        mse.extend(
            ((prediction - values) ** 2).tolist()
        )

        predicted_choice = row["candidates"][
            int(np.argmax(prediction))
        ]

        observed_choice = row["candidates"][
            int(np.argmax(values))
        ]

        choice_matches.append(
            predicted_choice == observed_choice
        )

        gold_matches.append(
            observed_choice == row["gold"]
        )

    return {
        "mse": float(np.mean(mse)),
        "model_choice_match": float(
            np.mean(choice_matches)
        ),
        "observed_model_gold_accuracy": float(
            np.mean(gold_matches)
        ),
        "n": len(rows),
    }


def make_prompt(tokenizer, case):
    content = (
        "Answer with exactly one person's name.\n"
        f"Passage: {case['passage']}\n"
        f"Question: {case['question']}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def ensure_committed(path):
    commands = (
        [
            "git",
            "ls-files",
            "--error-unmatch",
            str(path),
        ],
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            str(path),
        ],
    )

    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode:
            raise RuntimeError(
                "File must be tracked and "
                f"clean in HEAD: {path}"
            )


def validate_calibration(data, model):
    if (
        data.get("hash") != HASH
        or data.get("model") != model
        or not data.get("complete")
    ):
        raise RuntimeError(
            "Missing or invalid complete calibration"
        )

    target = make_cases("calibration")

    if [
        r["id"] for r in data["records"]
    ] != [
        r["id"] for r in target
    ]:
        raise RuntimeError(
            "Calibration sequence mismatch"
        )

    return data["records"]


def summarize_records(rows):
    mismatches = sum(
        len(set(
            r["token_counts"].values()
        )) > 1
        for r in rows
    )

    print(
        "Token count mismatch events:",
        mismatches,
        "/",
        len(rows),
    )

    for cons in CONSTRUCTIONS:
        for order in ORDERS:
            group = [
                r for r in rows
                if r["construction"] == cons
                and r["order"] == order
            ]

            if group:
                acc = sum(
                    max(
                        r["logprobs"],
                        key=r["logprobs"].get,
                    ) == r["gold"]
                    for r in group
                )

                print(
                    f"  {cons:12s} "
                    f"{order:16s} "
                    f"observed gold "
                    f"{acc}/{len(group)}"
                )


def run_inference(model_name, split):
    ensure_committed(SCRIPT)

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS as MODEL_DIRS,
        log_likelihood,
    )

    out = paths(model_name)[split]

    if split == "protected":
        for key in ("calibration", "frozen"):
            ensure_committed(
                paths(model_name)[key]
            )

        frozen = json.loads(
            paths(model_name)[
                "frozen"
            ].read_text()
        )

        if (
            frozen.get("hash") != HASH
            or frozen.get(
                "calibration_sha256"
            ) != file_hash(
                paths(model_name)[
                    "calibration"
                ]
            )
        ):
            raise RuntimeError(
                "Frozen input mismatch"
            )

    all_cases = make_cases(split)
    records = []

    if out.exists():
        old = json.loads(
            out.read_text()
        )

        if (
            old.get("hash") != HASH
            or old.get("model") != model_name
            or old.get("split") != split
            or old.get("complete")
        ):
            raise RuntimeError(
                "Existing output complete "
                "or incompatible"
            )

        records = old.get("records", [])

        if [
            r["id"] for r in records
        ] != [
            r["id"]
            for r in all_cases[:len(records)]
        ]:
            raise RuntimeError(
                "Cannot safely resume: "
                "cases changed"
            )

        print(
            f"RESUME {len(records)}"
            f"/{len(all_cases)}"
        )

    model_path = MODEL_DIRS / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing local checkpoint: {model_path}"
        )

    model, tokenizer = load(
        str(model_path)
    )

    for index in range(
        len(records),
        len(all_cases),
    ):
        case = all_cases[index]

        prompt = make_prompt(
            tokenizer,
            case,
        )

        scores = {}
        lengths = {}

        for name in case["candidates"]:
            score, count = log_likelihood(
                model,
                tokenizer,
                prompt,
                name,
            )

            scores[name] = score
            lengths[name] = count

        row = {
            **case,
            "logprobs": scores,
            "token_counts": lengths,
        }

        records.append(row)

        out.write_text(
            json.dumps(
                {
                    "hash": HASH,
                    "model": model_name,
                    "split": split,
                    "complete": False,
                    "records": records,
                },
                indent=2,
            )
        )

        chosen = max(
            scores,
            key=scores.get,
        )

        print(
            f"{split} "
            f"{index+1:02}/{len(all_cases)} "
            f"{case['construction']:12s} "
            f"{case['order']:16s} "
            f"{case['role']:7s} "
            f"gold={case['gold']:6s} "
            f"chosen={chosen:6s} "
            f"tokens={list(lengths.values())}",
            flush=True,
        )

    out.write_text(
        json.dumps(
            {
                "hash": HASH,
                "model": model_name,
                "split": split,
                "complete": True,
                "records": records,
            },
            indent=2,
        )
    )

    summarize_records(records)

    print("SAVED", out)


def freeze(model_name):
    p = paths(model_name)

    if p["frozen"].exists():
        raise RuntimeError(
            "Freeze manifest exists, "
            "refusing overwrite"
        )

    data = json.loads(
        p["calibration"].read_text()
    )

    records = validate_calibration(
        data,
        model_name,
    )

    frozen = {
        "hash": HASH,
        "model": model_name,
        "calibration_sha256": file_hash(
            p["calibration"]
        ),
        "coefficients": {},
        "calibration_metrics": {},
        "protected_predictions": {},
    }

    protected = make_cases("protected")

    for family, names in FAMILIES.items():
        coef = fit(
            records,
            names,
        )

        fitted = [
            predict(r, names, coef)
            for r in records
        ]

        projected = [
            predict(r, names, coef)
            for r in protected
        ]

        frozen["coefficients"][family] = {
            n: float(v)
            for n, v in zip(names, coef)
        }

        frozen[
            "calibration_metrics"
        ][family] = metrics(
            records,
            fitted,
        )

        frozen[
            "protected_predictions"
        ][family] = {
            r["id"]: {
                "candidate_centered_scores": {
                    c: float(z)
                    for c, z in zip(
                        r["candidates"],
                        pred,
                    )
                },
                "predicted_choice": (
                    r["candidates"][
                        int(np.argmax(pred))
                    ]
                ),
            }
            for r, pred in zip(
                protected,
                projected,
            )
        }

        x = frozen[
            "calibration_metrics"
        ][family]

        print(
            f"FREEZE {family:13s} "
            f"calibration MSE={x['mse']:.3f} "
            f"choice_match="
            f"{x['model_choice_match']:.3f}"
        )

        print(
            "  coefficients:",
            ", ".join(
                f"{n}={v:+.2f}"
                for n, v in zip(
                    names,
                    coef,
                )
            ),
        )

    p["frozen"].write_text(
        json.dumps(
            frozen,
            indent=2,
        )
    )

    print(
        f"FROZEN: {p['frozen']} "
        f"SHA256={file_hash(p['frozen'])}"
    )

    print(
        "Commit the calibration + frozen "
        "predictions before testing."
    )


def evaluate(model_name):
    p = paths(model_name)

    frozen = json.loads(
        p["frozen"].read_text()
    )

    data = json.loads(
        p["protected"].read_text()
    )

    if (
        frozen["hash"] != HASH
        or data["hash"] != HASH
        or data["model"] != model_name
        or not data["complete"]
    ):
        raise RuntimeError(
            "Protected data mismatch"
        )

    rows = data["records"]
    target = make_cases("protected")

    if [
        x["id"] for x in rows
    ] != [
        x["id"] for x in target
    ]:
        raise RuntimeError(
            "Protected case mismatch"
        )

    result = {}

    for family in FAMILIES:
        pred = frozen[
            "protected_predictions"
        ][family]

        vectors = []

        for case in target:
            rec = pred[case["id"]]

            vectors.append(
                np.array(
                    [
                        rec[
                            "candidate_centered_scores"
                        ][c]
                        for c in case["candidates"]
                    ],
                    dtype=np.float64,
                )
            )

        result[family] = metrics(
            rows,
            vectors,
        )

    print(
        f"\nML-050: {model_name} "
        f"| protected {len(rows)} contexts"
    )

    print(
        "Family          MSE       "
        "Model-choice match   "
        "Observed gold accuracy"
    )

    for family, m in result.items():
        print(
            f"{family:13s} "
            f"{m['mse']:8.3f}       "
            f"{m['model_choice_match']:7.3f}"
            f"            "
            f"{m['observed_model_gold_accuracy']:7.3f}"
        )

    a, b, c = [
        result[k]["mse"]
        for k in (
            "semantic_only",
            "additive",
            "contextual",
        )
    ]

    passed = (
        c <= 0.90 * b
        and c <= 0.90 * a
    )

    print(
        "PREDECLARED CONTEXTUAL GATE:",
        "PASS" if passed else "FAIL",
    )

    summarize_records(rows)

    return result


def audit():
    print("ML-050 DESIGN", HASH)

    for split in (
        "calibration",
        "protected",
    ):
        cases = make_cases(split)

        print(
            split,
            "contexts",
            len(cases),
        )

        assert len(cases) == 48

        assert len({
            c["id"] for c in cases
        }) == len(cases)

        for c in cases:
            assert (
                c["gold"]
                in c["candidates"]
            )

            assert len(set(
                c["candidates"]
            )) == 3

            assert len(set(
                c["mentions"]
            )) == 3

            assert all(
                c["passage"].count(n) == 1
                for n in c["candidates"]
            )

            assert features(
                c,
                FAMILIES["contextual"],
            ).shape == (
                3,
                len(FAMILIES["contextual"]),
            )

    cal_names = {
        n
        for x in CALIBRATION
        for n in x[:3]
    }

    test_names = {
        n
        for x in PROTECTED
        for n in x[:3]
    }

    assert not (
        cal_names & test_names
    )

    assert not (
        {x[3] for x in CALIBRATION}
        &
        {x[3] for x in PROTECTED}
    )

    sample = next(
        c
        for c in make_cases("protected")
        if (
            c["construction"] == "object_cleft"
            and c["order"] == "distractor_first"
        )
    )

    print(
        "EXAMPLE",
        sample["passage"],
        sample["question"],
    )

    print("Features", FAMILIES)

    # Synthetic regression smoke test.
    cal = make_cases("calibration")

    test_weights = np.array(
        [7., -2., 1., 1.5, -1., 0.5]
    )

    synthetic = [
        {
            **c,
            "logprobs": {
                k: float(v)
                for k, v in zip(
                    c["candidates"],
                    features(
                        c, ADDITIVE
                    ) @ test_weights,
                )
            },
        }
        for c in cal
    ]

    coef = fit(
        synthetic,
        ADDITIVE,
    )

    assert np.isfinite(coef).all()

    assert metrics(
        synthetic,
        [
            predict(r, ADDITIVE, coef)
            for r in synthetic
        ],
    )["mse"] < 1.0

    print(
        "AUDIT PASS: splits, factors, "
        "feature centering, ridge fitting, "
        "no model inference"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        required=True,
        choices=(
            "audit",
            "calibrate",
            "freeze",
            "test",
            "summarize",
        ),
    )

    ap.add_argument(
        "--model",
        choices=MODELS,
        default="qwen7b",
    )

    a = ap.parse_args()

    RESULTS.mkdir(
        exist_ok=True
    )

    if a.phase == "audit":
        audit()

    elif a.phase == "calibrate":
        run_inference(
            a.model,
            "calibration",
        )

    elif a.phase == "freeze":
        freeze(a.model)

    elif a.phase == "test":
        run_inference(
            a.model,
            "protected",
        )
        evaluate(a.model)

    else:
        evaluate(a.model)


if __name__ == "__main__":
    main()
