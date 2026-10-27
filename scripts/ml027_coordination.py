
"""ML-027: Predicting counterfactual answer-score changes.

Tests whether grammatical role relevance adds predictive
value beyond surface and candidate-accessibility features.

Behavioral prediction experiment; not a neural mechanism test.
"""

import argparse
import hashlib
import json

from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path


NAMES = ("Leo", "Felix", "Nora", "Clara")

ASSIGNMENTS = (
    ("Leo", "Felix", "Nora", "Clara"),
    ("Felix", "Leo", "Clara", "Nora"),
    ("Nora", "Clara", "Leo", "Felix"),
    ("Clara", "Nora", "Felix", "Leo"),
)

PACKS = {
    "calibration": {
        "a": ("found", "lost", "key"),
        "b": ("opened", "closed", "box"),
        "verb": "warned",
        "constructions": ("active", "passive"),
    },
    "holdout": {
        "a": ("bought", "sold", "painting"),
        "b": ("lifted", "dropped", "bag"),
        "verb": "photographed",
        "constructions": (
            "active",
            "passive",
            "subject_cleft",
            "object_cleft",
        ),
    },
}

ORDERS = ("a_first", "b_first")
QUESTIONS = ("agent", "patient")
STYLES = ("direct", "role")
EDITS = ("base", "a", "b")

RIDGE_ALPHA = 8.0
MIN_CLEFT_MAE_GAIN = 0.10
MAX_SEEN_MAE_DEGRADATION = 0.05


def a_is_agent(order, construction):
    first_is_agent = construction in (
        "active", "subject_cleft"
    )

    return first_is_agent == (order == "a_first")


def passage(split, assignment, order, construction, edit):
    pack = PACKS[split]

    a0, a1, b0, b1 = assignment
    aw0, aw1, anoun = pack["a"]
    bw0, bw1, bnoun = pack["b"]

    aword = aw1 if edit == "a" else aw0
    bword = bw1 if edit == "b" else bw0

    a = f"the person who {aword} the {anoun}"
    b = f"the person who {bword} the {bnoun}"

    if order == "a_first":
        first, second = a, b
    else:
        first, second = b, a

    verb = pack["verb"]

    if construction == "active":
        event = (
            f"At dusk, {first} {verb} {second}."
        )
    elif construction == "passive":
        event = (
            f"At dusk, {first} was {verb} by {second}."
        )
    elif construction == "subject_cleft":
        event = (
            f"At dusk, it was {first} who "
            f"{verb} {second}."
        )
    elif construction == "object_cleft":
        event = (
            f"At dusk, it was {first} whom "
            f"{second} {verb}."
        )
    else:
        raise ValueError(construction)

    facts = (
        f"{a0} {aw0} the {anoun}. "
        f"{a1} {aw1} the {anoun}. "
        f"{b0} {bw0} the {bnoun}. "
        f"{b1} {bw1} the {bnoun}."
    )

    return event + " " + facts


def question_text(split, role, style):
    verb = PACKS[split]["verb"]

    if style == "direct":
        if role == "agent":
            return f"Who {verb} the other person?"
        return f"Who was {verb}?"

    if role == "agent":
        return (
            "In the main event, who acted "
            "on the other person?"
        )

    return (
        "In the main event, who was acted "
        "on by the other person?"
    )


def gold(assignment, order, construction, role, edit):
    a0, a1, b0, b1 = assignment

    a = a1 if edit == "a" else a0
    b = b1 if edit == "b" else b0

    requested_a = (
        a_is_agent(order, construction)
        == (role == "agent")
    )

    return a if requested_a else b


def case_keys(split):
    return product(
        range(len(ASSIGNMENTS)),
        ORDERS,
        PACKS[split]["constructions"],
        QUESTIONS,
        STYLES,
    )


def design_hash(split):
    spec = {
        "names": NAMES,
        "assignments": ASSIGNMENTS,
        "pack": PACKS[split],
        "orders": ORDERS,
        "questions": QUESTIONS,
        "styles": STYLES,
        "edits": EDITS,
        "ridge_alpha": RIDGE_ALPHA,
        "min_cleft_mae_gain": MIN_CLEFT_MAE_GAIN,
        "max_seen_mae_degradation":
            MAX_SEEN_MAE_DEGRADATION,
        "outcome": (
            "centered four-name log likelihood delta"
        ),
        "models": (
            "zero", "surface", "coordination"
        ),
    }

    encoded = json.dumps(
        spec, sort_keys=True
    ).encode()

    return hashlib.sha256(encoded).hexdigest()


def make_prompt(
    tokenizer, split, assignment,
    order, construction, role, style, edit
):
    statement = passage(
        split, assignment, order, construction, edit
    )

    question = question_text(
        split, role, style
    )

    content = (
        "Answer with exactly one person's name.\n\n"
        f"Passage: {statement}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def base_features(case, cue, name, assignment):
    aid, order, construction, role, style = case

    a0, a1, b0, b1 = assignment

    pair = (
        (a0, a1)
        if cue == "a"
        else (b0, b1)
    )

    if name == pair[0]:
        sign = -1.0
    elif name == pair[1]:
        sign = 1.0
    else:
        sign = 0.0

    first = (
        (order == "a_first")
        == (cue == "a")
    )

    passive = construction == "passive"
    agent_q = role == "agent"

    # Surface-only model: lexical identity,
    # cue location, question and syntax markers.
    features = {
        "bias": 1.0,
        f"name:{name}": 1.0,
        f"cue:{cue}": 1.0,
        "signed": sign,
        f"signed:name:{name}": sign,
        f"signed:cue:{cue}": sign,
        "signed:first": sign * first,
        "signed:passive": sign * passive,
        "signed:agent_q": sign * agent_q,
        "signed:role_style":
            sign * (style == "role"),
        "signed:first:agent_q":
            sign * first * agent_q,
        "signed:passive:agent_q":
            sign * passive * agent_q,
        "signed:first:passive":
            sign * first * passive,
        f"signed:cue:{cue}:first":
            sign * first,
    }

    return features, sign


def row_features(
    case, cue, name, assignment, coordinated
):
    features, sign = base_features(
        case, cue, name, assignment
    )

    if coordinated:
        aid, order, construction, role, style = case

        a_agent = a_is_agent(order, construction)

        cue_agent = (
            a_agent if cue == "a" else not a_agent
        )

        relevant = cue_agent == (role == "agent")

        # The only additional theory feature:
        # Does the edited dependency supply the
        # grammatical role requested by the question?
        features["signed:semantic_query_match"] = (
            sign * relevant
        )

    return features


def centered_delta(base, changed):
    delta = {
        name: changed[name] - base[name]
        for name in NAMES
    }

    average = sum(delta.values()) / len(delta)

    return {
        name: delta[name] - average
        for name in NAMES
    }


def grouped_records(records):
    groups = defaultdict(dict)

    for row in records:
        case = (
            row["assignment_id"],
            row["order"],
            row["construction"],
            row["role"],
            row["style"],
        )

        if row["edit"] in groups[case]:
            raise ValueError(
                f"Duplicate record: {case}"
            )

        groups[case][row["edit"]] = row

    for case, rows in groups.items():
        if set(rows) != set(EDITS):
            raise ValueError(
                f"Incomplete case: {case}"
            )

    return groups


def regression_data(groups, coordinated):
    raw = []

    for case, rows in groups.items():
        assignment = ASSIGNMENTS[case[0]]
        baseline = rows["base"]["scores"]

        for cue in ("a", "b"):
            target = centered_delta(
                baseline, rows[cue]["scores"]
            )

            for name in NAMES:
                features = row_features(
                    case, cue, name,
                    assignment, coordinated
                )

                raw.append((
                    features, target[name]
                ))

    vocabulary = tuple(sorted(
        set().union(*(
            features.keys()
            for features, _ in raw
        ))
    ))

    import numpy as np

    X = np.array([
        [
            float(features.get(key, 0.0))
            for key in vocabulary
        ]
        for features, _ in raw
    ], dtype=float)

    y = np.array([
        value for _, value in raw
    ], dtype=float)

    return X, y, vocabulary


def train_model(groups, coordinated):
    import numpy as np

    X, y, vocabulary = regression_data(
        groups, coordinated
    )

    regularization = (
        RIDGE_ALPHA * np.eye(X.shape[1])
    )

    coefficients = np.linalg.solve(
        X.T @ X + regularization,
        X.T @ y,
    )

    return {
        "vocabulary": vocabulary,
        "coefficients": coefficients,
        "coordinated": coordinated,
    }


def predict_delta(model, case, cue):
    import numpy as np

    assignment = ASSIGNMENTS[case[0]]

    features_by_name = [
        row_features(
            case, cue, name, assignment,
            model["coordinated"],
        )
        for name in NAMES
    ]

    X = np.array([
        [
            float(features.get(key, 0.0))
            for key in model["vocabulary"]
        ]
        for features in features_by_name
    ], dtype=float)

    values = X @ model["coefficients"]
    values -= values.mean()

    return dict(zip(
        NAMES,
        map(float, values),
    ))


def evaluate(calibration, holdout):
    train = grouped_records(
        calibration["records"]
    )

    test = grouped_records(
        holdout["records"]
    )

    if len(train) != len(tuple(case_keys("calibration"))):
        raise ValueError("Missing calibration cases")

    if len(test) != len(tuple(case_keys("holdout"))):
        raise ValueError("Missing holdout cases")

    fitted = {
        "surface": train_model(train, False),
        "coordination": train_model(train, True),
    }

    families = (
        "seen_constructions",
        "unseen_clefts",
    )

    results = {
        family: {
            model: {
                "errors": [],
                "gold_correct": 0,
                "winner_agreement": 0,
                "winner_changes": 0,
                "n": 0,
            }
            for model in (
                "zero", "surface", "coordination"
            )
        }
        for family in families
    }

    for case, rows in test.items():
        family = (
            "unseen_clefts"
            if "cleft" in case[2]
            else "seen_constructions"
        )

        assignment = ASSIGNMENTS[case[0]]
        baseline = rows["base"]["scores"]

        base_winner = max(
            baseline, key=baseline.get
        )

        for cue in ("a", "b"):
            actual = rows[cue]["scores"]

            actual_delta = centered_delta(
                baseline, actual
            )

            actual_winner = max(
                actual, key=actual.get
            )

            expected = gold(
                assignment,
                case[1],
                case[2],
                case[3],
                cue,
            )

            for name in (
                "zero", "surface", "coordination"
            ):
                delta = (
                    {n: 0.0 for n in NAMES}
                    if name == "zero"
                    else predict_delta(
                        fitted[name], case, cue
                    )
                )

                metrics = results[family][name]

                metrics["errors"].extend(
                    abs(actual_delta[n] - delta[n])
                    for n in NAMES
                )

                predicted_scores = {
                    n: baseline[n] + delta[n]
                    for n in NAMES
                }

                winner = max(
                    predicted_scores,
                    key=predicted_scores.get,
                )

                metrics["gold_correct"] += (
                    winner == expected
                )

                metrics["winner_agreement"] += (
                    winner == actual_winner
                )

                metrics["winner_changes"] += (
                    winner != base_winner
                )

                metrics["n"] += 1

    def mae(family, name):
        errors = results[family][name]["errors"]
        return sum(errors) / len(errors)

    print("\n=== PROSPECTIVE HELD-OUT TEST ===")

    for family in families:
        print(f"\n{family.upper()}")

        for name in (
            "zero", "surface", "coordination"
        ):
            row = results[family][name]

            print(
                f"{name:12s} "
                f"delta_MAE={mae(family, name):.4f} "
                f"gold={row['gold_correct']}/{row['n']} "
                f"winner_agreement="
                f"{row['winner_agreement']}/{row['n']} "
                f"winner_changes="
                f"{row['winner_changes']}/{row['n']}"
            )

    surface_cleft = mae(
        "unseen_clefts", "surface"
    )
    coord_cleft = mae(
        "unseen_clefts", "coordination"
    )

    surface_seen = mae(
        "seen_constructions", "surface"
    )
    coord_seen = mae(
        "seen_constructions", "coordination"
    )

    gain = (
        (surface_cleft - coord_cleft) / surface_cleft
        if surface_cleft > 0
        else float("-inf")
    )

    degradation = (
        (coord_seen - surface_seen) / surface_seen
        if surface_seen > 0
        else float("inf")
    )

    cleft_agreement = results[
        "unseen_clefts"
    ]["coordination"]["winner_agreement"]

    surface_agreement = results[
        "unseen_clefts"
    ]["surface"]["winner_agreement"]

    passes = (
        gain >= MIN_CLEFT_MAE_GAIN
        and degradation <= MAX_SEEN_MAE_DEGRADATION
        and cleft_agreement >= surface_agreement
    )

    print("\n=== FROZEN DEVELOPMENT GATE ===")
    print(f"Unseen-cleft MAE gain: {gain:+.1%}")
    print(f"Seen-construction degradation: {degradation:+.1%}")
    print(
        "Winner agreement:",
        cleft_agreement,
        "vs",
        surface_agreement,
    )
    print("Gate:", "PASS" if passes else "FAIL")

    print(
        "\nNote: Semantic role information was "
        "supplied from the experimental design."
    )
    print(
        "A pass demonstrates predictive value "
        "for this feature, not a neural mechanism."
    )


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "design", "audit",
            "collect", "evaluate"
        ),
        default="design",
    )

    parser.add_argument(
        "--split",
        choices=tuple(PACKS),
        default="calibration",
    )

    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )

    args = parser.parse_args()

    print(
        "ML-027",
        args.phase,
        args.split,
        args.model,
    )

    for split in PACKS:
        print(
            f"{split} SHA-256:",
            design_hash(split),
        )

    if args.phase == "design":
        for split in PACKS:
            print(f"\n=== {split.upper()} ===")

            for construction in PACKS[split]["constructions"]:
                print(
                    construction,
                    passage(
                        split,
                        ASSIGNMENTS[0],
                        "a_first",
                        construction,
                        "base",
                    ),
                )

            total = (
                len(tuple(case_keys(split)))
                * len(EDITS)
            )

            print("Prompt count:", total)

        print("\nNo inference performed.")
        return

    if args.phase == "evaluate":
        data = {}

        for split in PACKS:
            paths = sorted(
                Path("results").glob(
                    f"ml027_{args.model}_{split}_*.json"
                )
            )

            if not paths:
                raise FileNotFoundError(split)

            source = paths[-1]

            result = json.loads(
                source.read_text()
            )

            if (
                not result.get("complete")
                or result.get("design_hash")
                != design_hash(split)
            ):
                raise ValueError(
                    f"Invalid results: {source}"
                )

            data[split] = result
            print("Using:", source)

        evaluate(
            data["calibration"],
            data["holdout"],
        )
        return

    from mlx_lm import load

    from ml003_role_margins import (
        MODELS,
        log_likelihood,
    )

    model, tokenizer = load(
        str(MODELS / args.model)
    )

    prepared = []

    for case in case_keys(args.split):
        aid, order, construction, role, style = case

        prompts = {
            edit: make_prompt(
                tokenizer,
                args.split,
                ASSIGNMENTS[aid],
                order,
                construction,
                role,
                style,
                edit,
            )
            for edit in EDITS
        }

        expected = {
            edit: gold(
                ASSIGNMENTS[aid],
                order,
                construction,
                role,
                edit,
            )
            for edit in EDITS
        }

        if any(
            name not in NAMES
            for name in expected.values()
        ):
            raise ValueError(
                f"Invalid expected answer: {case}"
            )

        prepared.append((
            case, prompts, expected
        ))

    token_lengths = {
        name: len(
            tokenizer.encode(
                name, add_special_tokens=False
            )
        )
        for name in NAMES
    }

    print(
        "Candidate token lengths:",
        token_lengths,
    )

    print(
        "Validated cases:",
        len(prepared),
    )

    if args.phase == "audit":
        print("No likelihood scoring performed.")
        return

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml027_{args.model}_{args.split}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    data = {
        "experiment": "ML-027",
        "model": args.model,
        "split": args.split,
        "design_hash": design_hash(args.split),
        "complete": False,
        "records": [],
    }

    def save():
        output.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    for case, prompts, expected in prepared:
        aid, order, construction, role, style = case

        for edit in EDITS:
            scores = {}

            for name in NAMES:
                lp, _ = log_likelihood(
                    model,
                    tokenizer,
                    prompts[edit],
                    name,
                )

                scores[name] = float(lp)

            record = {
                "assignment_id": aid,
                "order": order,
                "construction": construction,
                "role": role,
                "style": style,
                "edit": edit,
                "scores": scores,
                "gold": expected[edit],
                "forced": max(
                    scores, key=scores.get
                ),
            }

            data["records"].append(record)

        save()

        print(
            "DONE",
            case,
            "records:",
            len(data["records"]),
        )

    data["complete"] = True
    save()

    print("\nSaved:", output)


if __name__ == "__main__":
    run()
