
import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from itertools import product
from pathlib import Path


MODELS_TO_TEST = ("llama3b", "qwen7b")
PROBES = ("agent", "patient")
COMPLEXITY = tuple(product((0, 1), repeat=2))


def configurations():
    configs = []

    for flip_agent, flip_patient in product(
        (False, True), repeat=2
    ):
        writer, carrier = (
            ("Leo", "Felix")
            if not flip_agent
            else ("Felix", "Leo")
        )

        folder, painter = (
            ("Nora", "Clara")
            if not flip_patient
            else ("Clara", "Nora")
        )

        configs.append({
            "writer": writer,
            "carrier": carrier,
            "folder": folder,
            "painter": painter,
        })

    return configs


def make_passage(config, agent_rel, patient_rel):
    agent = (
        "the person who wrote the letter"
        if agent_rel
        else config["writer"]
    )

    patient = (
        "the person who folded the map"
        if patient_rel
        else config["folder"]
    )

    event = f"At dusk, {agent} helped {patient}."

    facts = " ".join([
        f"{config['writer']} wrote the letter.",
        f"{config['carrier']} carried the letter.",
        f"{config['folder']} folded the map.",
        f"{config['painter']} painted the map.",
    ])

    return f"{event} {facts}"


def question_for(probe):
    if probe == "agent":
        return "Who helped the other person?"

    return "Who was helped?"


def gold_answer(config, probe):
    if probe == "agent":
        return config["writer"]

    return config["folder"]


def symbolic_baseline(passage, question):
    """
    Deterministic lexical/syntactic baseline.

    Uses the passage and question only.
    Does not receive the gold answer or model configuration.
    """

    event = passage.split(". ", 1)[0] + "."

    match = re.fullmatch(
        r"At dusk, "
        r"(?P<agent>[A-Z][a-z]+|"
        r"the person who wrote the letter) "
        r"helped "
        r"(?P<patient>[A-Z][a-z]+|"
        r"the person who folded the map)\.",
        event,
    )

    if match is None:
        raise ValueError(f"Unrecognized event: {event}")

    role = (
        "patient"
        if question == "Who was helped?"
        else "agent"
    )

    expression = match.group(role)

    if expression == "the person who wrote the letter":
        pattern = r"\b([A-Z][a-z]+) wrote the letter\."

    elif expression == "the person who folded the map":
        pattern = r"\b([A-Z][a-z]+) folded the map\."

    else:
        return expression

    candidates = re.findall(pattern, passage)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one match: {candidates}"
        )

    return candidates[0]


def make_trials():
    trials = []

    for context_id, config in enumerate(configurations()):
        for (agent_rel, patient_rel), probe in product(
            COMPLEXITY, PROBES
        ):
            passage = make_passage(
                config, agent_rel, patient_rel
            )
            question = question_for(probe)
            gold = gold_answer(config, probe)

            baseline = symbolic_baseline(
                passage, question
            )

            assert baseline == gold

            trials.append({
                "context_id": context_id,
                "config": config,
                "agent_rel": agent_rel,
                "patient_rel": patient_rel,
                "probe": probe,
                "passage": passage,
                "question": question,
                "gold": gold,
                "symbolic_baseline": baseline,
            })

    assert len(trials) == 32

    for context_id in range(4):
        group = [
            t for t in trials
            if t["context_id"] == context_id
        ]

        assert len(group) == 8

        combinations = {
            (
                t["agent_rel"],
                t["patient_rel"],
                t["probe"],
            )
            for t in group
        }

        assert len(combinations) == 8

    return trials


def make_prompt(tokenizer, passage, question):
    content = (
        "Answer using exactly one person's name.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_native(raw, names):
    cleaned = raw.strip().strip(" \"'.,!?;:\n\t")

    for name in names:
        if cleaned.casefold() == name.casefold():
            return name

    return None


def score_trial(model, tokenizer, trial, log_likelihood, generate):
    config = trial["config"]

    names = (
        config["writer"],
        config["carrier"],
        config["folder"],
        config["painter"],
    )

    prompt = make_prompt(
        tokenizer,
        trial["passage"],
        trial["question"],
    )

    scores = {}
    token_counts = {}

    for name in names:
        logprob, count = log_likelihood(
            model, tokenizer, prompt, name
        )

        scores[name] = float(logprob)
        token_counts[name] = int(count)

    forced = max(scores, key=scores.get)

    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=12,
        verbose=False,
    )

    native = parse_native(raw, names)
    gold = trial["gold"]

    if trial["probe"] == "agent":
        target = config["writer"]
        competitor = config["carrier"]
    else:
        target = config["folder"]
        competitor = config["painter"]

    within_pair_margin = (
        scores[target] - scores[competitor]
    )

    global_margin = (
        scores[gold]
        - max(
            scores[name]
            for name in names
            if name != gold
        )
    )

    return {
        **trial,
        "prompt": prompt,
        "scores": scores,
        "token_counts": token_counts,
        "forced": forced,
        "native": native,
        "native_raw": raw,
        "within_pair_margin": within_pair_margin,
        "global_margin": global_margin,
        "within_pair_correct": within_pair_margin > 0,
        "forced_correct": forced == gold,
        "native_correct": native == gold,
    }


def calculate_interactions(records):
    """
    For each context and probe, calculate the factorial
    interaction on the same correct-answer margin.

    I = M11 - M10 - M01 + M00

    Also calculate averaged main effects of making the
    agent or patient relational.
    """

    results = []

    for context_id in range(4):
        for probe in PROBES:
            subset = {
                (r["agent_rel"], r["patient_rel"]): r
                for r in records
                if (
                    r["context_id"] == context_id
                    and r["probe"] == probe
                )
            }

            assert len(subset) == 4

            for metric in (
                "within_pair_margin",
                "global_margin",
            ):
                m00 = subset[(0, 0)][metric]
                m10 = subset[(1, 0)][metric]
                m01 = subset[(0, 1)][metric]
                m11 = subset[(1, 1)][metric]

                interaction = (
                    m11 - m10 - m01 + m00
                )

                agent_effect = (
                    (m10 - m00) + (m11 - m01)
                ) / 2

                patient_effect = (
                    (m01 - m00) + (m11 - m10)
                ) / 2

                results.append({
                    "context_id": context_id,
                    "probe": probe,
                    "metric": metric,
                    "m00": m00,
                    "m10": m10,
                    "m01": m01,
                    "m11": m11,
                    "agent_complexity_effect": agent_effect,
                    "patient_complexity_effect": patient_effect,
                    "interaction": interaction,
                })

    return results


def summarize(records, interactions):
    print("\n=== ACCURACY BY CONDITION ===")

    for probe in PROBES:
        for agent_rel, patient_rel in COMPLEXITY:
            group = [
                r for r in records
                if (
                    r["probe"] == probe
                    and r["agent_rel"] == agent_rel
                    and r["patient_rel"] == patient_rel
                )
            ]

            forced = sum(
                r["forced_correct"] for r in group
            )

            native = sum(
                r["native_correct"] for r in group
            )

            within_pair = sum(
                r["within_pair_correct"] for r in group
            )

            avg_margin = statistics.mean(
                r["within_pair_margin"] for r in group
            )

            print(
                f"probe={probe:7s} "
                f"complexity={agent_rel}{patient_rel} "
                f"forced={forced}/4 "
                f"native={native}/4 "
                f"within_pair={within_pair}/4 "
                f"pair_margin={avg_margin:+.3f}"
            )

    print("\n=== FACTORIAL INTERACTIONS ===")

    for probe in PROBES:
        for metric in (
            "within_pair_margin",
            "global_margin",
        ):
            group = [
                r for r in interactions
                if (
                    r["probe"] == probe
                    and r["metric"] == metric
                )
            ]

            agent_effect = statistics.mean(
                r["agent_complexity_effect"]
                for r in group
            )

            patient_effect = statistics.mean(
                r["patient_complexity_effect"]
                for r in group
            )

            interaction = statistics.mean(
                r["interaction"]
                for r in group
            )

            signs = [
                "+" if r["interaction"] > 0
                else "-" if r["interaction"] < 0
                else "0"
                for r in group
            ]

            print(
                f"{probe:7s} {metric:19s} "
                f"agent_effect={agent_effect:+.3f} "
                f"patient_effect={patient_effect:+.3f} "
                f"interaction={interaction:+.3f} "
                f"signs={''.join(signs)}"
            )

    print("\n=== INTERPRETATION LIMITS ===")
    print(
        "Interactions are descriptive, based on four "
        "name assignments and one construction."
    )
    print(
        "A nonzero interaction does not establish "
        "a neural binding mechanism."
    )
    print(
        "Global margins can be influenced by "
        "unequal candidate tokenization."
    )
    print(
        "The symbolic parser solves the task; "
        "lexical and syntactic heuristics remain baselines."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=MODELS_TO_TEST,
        default="llama3b",
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
    )

    args = parser.parse_args()

    trials = make_trials()

    design_string = json.dumps(
        trials,
        sort_keys=True,
        ensure_ascii=False,
    )

    design_hash = hashlib.sha256(
        design_string.encode("utf-8")
    ).hexdigest()

    print("Experiment: ML-021")
    print("Model:", args.model)
    print("Contexts:", len(configurations()))
    print("Complexity conditions:", len(COMPLEXITY))
    print("Questions:", len(PROBES))
    print("Trials:", len(trials))
    print("Symbolic baseline: 32/32")
    print("Design SHA-256:", design_hash)

    if args.audit_only:
        print("\n=== DESIGN PREVIEW ===")

        for trial in trials[:8]:
            print(
                f"\nComplexity: "
                f"{trial['agent_rel']}{trial['patient_rel']}"
                f" | Probe: {trial['probe']}"
            )
            print(trial["passage"])
            print(trial["question"])
            print("Gold:", trial["gold"])

        print("\nDesign audit passed.")
        print("No model inference performed.")
        return

    from mlx_lm import load, generate
    from ml003_role_margins import MODELS, log_likelihood

    model_path = MODELS / args.model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    print("\nLoading:", model_path)
    model, tokenizer = load(str(model_path))

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output = results_dir / (
        f"ml021_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    records = []

    def save(complete, interactions=None):
        payload = {
            "experiment": "ML-021",
            "model": args.model,
            "model_path": str(model_path),
            "design_sha256": design_hash,
            "complete": complete,
            "records": records,
        }

        if interactions is not None:
            payload["interactions"] = interactions

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for trial in trials:
        result = score_trial(
            model,
            tokenizer,
            trial,
            log_likelihood,
            generate,
        )

        records.append(result)

        print(
            f"ctx={result['context_id']:02d} "
            f"complexity="
            f"{result['agent_rel']}{result['patient_rel']} "
            f"probe={result['probe']:7s} "
            f"gold={result['gold']:5s} "
            f"forced={result['forced']:5s} "
            f"native={str(result['native']):5s} "
            f"pair={result['within_pair_margin']:+.3f} "
            f"global={result['global_margin']:+.3f}"
        )

        save(complete=False)

    interactions = calculate_interactions(records)

    summarize(records, interactions)

    save(
        complete=True,
        interactions=interactions,
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
