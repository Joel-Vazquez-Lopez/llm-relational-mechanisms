
"""ML-054A: retrospective measurement audit.

Compares ML-053 candidate likelihood, completed-answer likelihood,
and unrestricted greedy generation. Does not modify ML-053 files.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/ml053_design_manifest.json"
PREDICTOR = ROOT / "scripts/ml053_predict.py"

TARGETS = {
    "llama3b": ("active", "verb_explicit"),
    "qwen7b": ("object_cleft", "paraphrase"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(model):
    manifest = json.loads(MANIFEST.read_text())
    claimed = manifest.pop("manifest_sha256")

    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    if claimed != actual:
        raise RuntimeError("ML-053 manifest hash mismatch")

    source = ROOT / f"results/ml053_{model}_protected.json"
    data = json.loads(source.read_text())

    if not (
        data["complete"]
        and data["model"] == model
        and data["split"] == "protected"
        and data["manifest_sha256"] == claimed
        and data["script_sha256"] == sha(PREDICTOR)
    ):
        raise RuntimeError("Protected record integrity failed")

    cases = manifest["cases"]["protected"]

    if len(cases) != 288 or len(data["records"]) != 288:
        raise RuntimeError("Unexpected protected case count")

    for case, record in zip(cases, data["records"]):
        if any(
            record.get(k) != value
            for k, value in case.items()
        ):
            raise RuntimeError(f"Case mismatch: {case['id']}")

        if set(record["scores"]) != set(case["pair"]):
            raise RuntimeError("Candidate mismatch")

    construction, style = TARGETS[model]

    rows = [
        row for row in data["records"]
        if row["construction"] == construction
        and row["style"] == style
    ]

    if (
        len(rows) != 32
        or len({tuple(r["pair"]) for r in rows}) != 8
    ):
        raise RuntimeError("Expected 32 prompts, eight lexical groups")

    provenance = {
        "manifest_sha256": claimed,
        "protected_sha256": sha(source),
        "predictor_sha256": sha(PREDICTOR),
    }

    return rows, provenance


def user_content(case):
    return (
        "Answer with exactly one person's name.\n"
        f"Passage: {case['passage']}\n"
        f"Question: {case['question']}\n"
        "Answer:"
    )


def score_name_and_turn(
    model, tokenizer, prompt, case, candidate
):
    import mlx.core as mx

    bos = tokenizer.bos_token
    add_special = (
        bos is None or not prompt.startswith(bos)
    )

    prefix = tokenizer.encode(
        prompt, add_special_tokens=add_special
    )

    full = tokenizer.encode(
        prompt + candidate,
        add_special_tokens=add_special,
    )

    if (
        not prefix
        or full[:len(prefix)] != prefix
        or len(full) == len(prefix)
    ):
        raise RuntimeError("Prompt/candidate token boundary mismatch")

    whole_turn = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content(case)},
            {"role": "assistant", "content": candidate},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )

    if whole_turn[:len(full)] != full:
        raise RuntimeError(
            "Completed chat turn does not match scorer prefix"
        )

    end_tokens = whole_turn[len(full):]

    if not 1 <= len(end_tokens) <= 4:
        raise RuntimeError(
            f"Unexpected end-of-turn tokens: {end_tokens}"
        )

    name_tokens = full[len(prefix):]
    targets = name_tokens + end_tokens

    inputs = mx.array(
        [full + end_tokens[:-1]],
        dtype=mx.int32,
    )

    logits = model(inputs)

    start = len(prefix) - 1

    selected = logits[
        0,
        start:start + len(targets),
        :,
    ].astype(mx.float32)

    log_probs = (
        selected
        - mx.logsumexp(
            selected,
            axis=-1,
            keepdims=True,
        )
    )

    values = mx.take_along_axis(
        log_probs,
        mx.array(targets, dtype=mx.int32)[:, None],
        axis=-1,
    ).reshape((-1,))

    name_score = float(
        mx.sum(values[:len(name_tokens)]).item()
    )

    turn_score = float(mx.sum(values).item())

    mx.clear_cache()

    return name_score, turn_score, end_tokens


def exact_generated_name(text, pair):
    if not isinstance(text, str):
        return None

    match = re.fullmatch(
        r"\s*['\"`]*([A-Za-z]+)[.!?\s'\"`]*",
        text,
    )

    if match and match.group(1) in pair:
        return match.group(1)

    return None


def classify(rows, field):
    paired = {}

    for row in rows:
        key = (tuple(row["pair"]), row["direction"])
        paired.setdefault(key, {})[row["role"]] = row

    if (
        len(paired) != 16
        or any(
            set(group) != {"agent", "patient"}
            for group in paired.values()
        )
    ):
        raise RuntimeError("Missing paired questions")

    result = {
        "correct_switch": 0,
        "wrong_switch": 0,
        "same_choice": 0,
        "unresolved": 0,
    }

    for group in paired.values():
        a = group["agent"]
        p = group["patient"]

        if field == "generated_name":
            winner_a = a.get(field)
            winner_p = p.get(field)

        else:
            def preferred(row):
                scores = row[field]

                if (
                    scores[row["agent"]]
                    == scores[row["patient"]]
                ):
                    return None

                return max(scores, key=scores.get)

            winner_a = preferred(a)
            winner_p = preferred(p)

        if not winner_a or not winner_p:
            result["unresolved"] += 1

        elif (
            winner_a == a["agent"]
            and winner_p == p["patient"]
        ):
            result["correct_switch"] += 1

        elif (
            winner_a == a["patient"]
            and winner_p == p["agent"]
        ):
            result["wrong_switch"] += 1

        elif winner_a == winner_p:
            result["same_choice"] += 1

        else:
            raise RuntimeError("Unclassified preference pair")

    return result


def summarize(rows):
    result = {}

    for field in (
        "scores",
        "name_score",
        "turn_score",
        "generated_name",
    ):
        if field != "scores" and any(
            field not in r for r in rows
        ):
            continue

        result[field] = classify(rows, field)

    if all("name_score" in r for r in rows):
        result["max_original_score_discrepancy"] = max(
            abs(
                r["scores"][name]
                - r["name_score"][name]
            )
            for r in rows
            for name in r["pair"]
        )

        result["generations_with_exact_name"] = sum(
            r.get("generated_name") is not None
            for r in rows
        )

        result[
            "generation_matches_original_preference"
        ] = sum(
            r.get("generated_name")
            == max(r["scores"], key=r["scores"].get)
            for r in rows
            if r.get("generated_name") is not None
        )

    return result


def run(model_name):
    rows, provenance = load_rows(model_name)

    destination = (
        ROOT
        / f"results/ml054_{model_name}_measurement_audit.json"
    )

    report = {
        "experiment": "ML-054A",
        "interpretation": "retrospective measurement audit",
        "model": model_name,
        "condition": TARGETS[model_name],
        "provenance": provenance,
        "complete": False,
        "records": [],
    }

    if destination.exists():
        report = json.loads(destination.read_text())

        if (
            report.get("complete")
            or report.get("provenance") != provenance
            or report.get("model") != model_name
            or report.get("condition")
            != list(TARGETS[model_name])
        ):
            raise RuntimeError(
                "Existing audit complete or incompatible"
            )

        if len(report["records"]) > len(rows):
            raise RuntimeError("Partial audit too long")

        for old, case in zip(report["records"], rows):
            if old["id"] != case["id"]:
                raise RuntimeError(
                    "Partial audit record mismatch"
                )

    from mlx_lm import load, generate

    from ml003_role_margins import (
        MODELS as MODEL_ROOT,
    )

    from ml053_design import render

    model, tokenizer = load(
        str(MODEL_ROOT / model_name)
    )

    for i in range(len(report["records"]), len(rows)):
        case = rows[i]
        prompt = render(tokenizer, case)

        names = {}
        turns = {}
        endings = {}

        for name in case["pair"]:
            (
                names[name],
                turns[name],
                endings[name],
            ) = score_name_and_turn(
                model, tokenizer, prompt, case, name
            )

        max_diff = max(
            abs(names[n] - case["scores"][n])
            for n in case["pair"]
        )

        if max_diff > 0.1:
            raise RuntimeError(
                f"Archived scorer differs by {max_diff:.4f}; "
                "inspect before continuing"
            )

        # MLX-LM defaults to greedy decoding without a sampler.
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=14,
            verbose=False,
        )

        record = {
            **case,
            "name_score": names,
            "turn_score": turns,
            "end_tokens": endings,
            "generated_text": generated,
            "generated_name": exact_generated_name(
                generated, case["pair"]
            ),
        }

        report["records"].append(record)

        temp = destination.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(report, indent=2) + "\n"
        )
        temp.replace(destination)

        print(
            f"{model_name}: {i+1:02d}/{len(rows)} "
            f"{case['role']:7s} "
            f"generated={generated!r}",
            flush=True,
        )

    report["complete"] = True
    report["summary"] = summarize(report["records"])

    temp = destination.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(destination)

    print(json.dumps(report["summary"], indent=2))
    print("SAVED", destination)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=("audit", "run", "summarize"),
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=tuple(TARGETS),
        default="llama3b",
    )

    args = parser.parse_args()

    rows, provenance = load_rows(args.model)

    if args.phase == "audit":
        print(
            "AUDIT PASS:",
            args.model,
            TARGETS[args.model],
            len(rows),
            "protected prompts",
        )

        print(
            "Original switches:",
            summarize(rows)["scores"],
        )

        print("Retrospective diagnostic only")

    elif args.phase == "run":
        run(args.model)

    else:
        path = (
            ROOT
            / f"results/ml054_{args.model}_measurement_audit.json"
        )

        report = json.loads(path.read_text())

        if (
            not report["complete"]
            or report["provenance"] != provenance
        ):
            raise RuntimeError("Audit incomplete or changed")

        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
