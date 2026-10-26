
import argparse
import hashlib
import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path

FORMS = ("direct", "identity", "relation")
PROBES = ("agent", "patient")

C = {
    "writer": "Leo",
    "carrier": "Felix",
    "folder": "Nora",
    "painter": "Clara",
}


def description(role, form):
    name = C["writer"] if role == "agent" else C["folder"]

    if form == "direct":
        return name

    if form == "identity":
        return f"the person named {name}"

    if role == "agent":
        return "the person who wrote the letter"

    return "the person who folded the map"


def make_passage(agent_form, patient_form):
    agent = description("agent", agent_form)
    patient = description("patient", patient_form)

    event = f"At dusk, {agent} helped {patient}."

    facts = " ".join([
        f"{C['writer']} wrote the letter.",
        f"{C['carrier']} carried the letter.",
        f"{C['folder']} folded the map.",
        f"{C['painter']} painted the map.",
    ])

    return f"{event} {facts}"


def symbolic_answer(passage, question):
    event = passage.split(". ", 1)[0] + "."

    match = re.fullmatch(
        r"At dusk, (.*?) helped (.*?)\.",
        event,
    )
    assert match is not None

    expression = (
        match.group(1)
        if question.startswith("Who helped")
        else match.group(2)
    )

    if expression.startswith("the person named "):
        return expression.removeprefix("the person named ")

    if expression == "the person who wrote the letter":
        pattern = r"\b([A-Z][a-z]+) wrote the letter\."

    elif expression == "the person who folded the map":
        pattern = r"\b([A-Z][a-z]+) folded the map\."

    else:
        assert re.fullmatch(r"[A-Z][a-z]+", expression)
        return expression

    matches = re.findall(pattern, passage)
    assert len(matches) == 1
    return matches[0]


def make_trials():
    trials = []

    for agent_form, patient_form, probe in product(
        FORMS, FORMS, PROBES
    ):
        passage = make_passage(agent_form, patient_form)

        question = (
            "Who helped the other person?"
            if probe == "agent"
            else "Who was helped?"
        )

        gold = (
            C["writer"] if probe == "agent"
            else C["folder"]
        )

        assert symbolic_answer(passage, question) == gold

        trials.append({
            "agent_form": agent_form,
            "patient_form": patient_form,
            "probe": probe,
            "passage": passage,
            "question": question,
            "gold": gold,
        })

    assert len(trials) == 18
    return trials


def factorial_interaction(records, probe, metric, low, high):
    rows = {
        (r["agent_form"], r["patient_form"]): r
        for r in records
        if r["probe"] == probe
    }

    def score(a, p):
        return rows[(a, p)][metric]

    return (
        score(high, high)
        - score(high, low)
        - score(low, high)
        + score(low, low)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        default="llama3b",
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    trials = make_trials()
    digest = hashlib.sha256(
        json.dumps(
            trials, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    print("Experiment: ML-022")
    print("Model:", args.model)
    print("Design: 3x3 reference forms x 2 questions")
    print("Trials:", len(trials))
    print("Symbolic baseline: 18/18")
    print("Design SHA-256:", digest)

    if args.audit_only:
        for t in trials:
            print(
                f"{t['agent_form']:8s} "
                f"{t['patient_form']:8s} "
                f"{t['probe']:7s} "
                f"gold={t['gold']}"
            )
        print("No inference performed.")
        return

    from mlx_lm import load, generate
    from ml003_role_margins import MODELS, log_likelihood
    from ml021_complexity import make_prompt

    model, tokenizer = load(str(MODELS / args.model))

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml022_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    names = tuple(C.values())
    records = []

    def save(complete):
        output.write_text(
            json.dumps({
                "experiment": "ML-022",
                "model": args.model,
                "design_sha256": digest,
                "complete": complete,
                "records": records,
            }, indent=2),
            encoding="utf-8",
        )

    for t in trials:
        prompt = make_prompt(
            tokenizer, t["passage"], t["question"]
        )

        scores = {}
        token_counts = {}

        for name in names:
            lp, count = log_likelihood(
                model, tokenizer, prompt, name
            )
            scores[name] = float(lp)
            token_counts[name] = int(count)

        gold = t["gold"]

        same_role_rival = (
            C["carrier"] if t["probe"] == "agent"
            else C["painter"]
        )

        cross_role_rival = (
            C["folder"] if t["probe"] == "agent"
            else C["writer"]
        )

        raw = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=16,
            verbose=False,
        )

        record = {
            **t,
            "scores": scores,
            "token_counts": token_counts,
            "forced": max(scores, key=scores.get),
            "native_raw": raw,
            "within_pair_margin": (
                scores[gold] - scores[same_role_rival]
            ),
            "fixed_cross_role_margin": (
                scores[gold] - scores[cross_role_rival]
            ),
            "global_margin": (
                scores[gold]
                - max(
                    scores[n] for n in names if n != gold
                )
            ),
        }

        records.append(record)

        print(
            f"agent={t['agent_form']:8s} "
            f"patient={t['patient_form']:8s} "
            f"q={t['probe']:7s} "
            f"forced={record['forced']:5s} "
            f"pair={record['within_pair_margin']:+.3f} "
            f"fixed={record['fixed_cross_role_margin']:+.3f} "
            f"native={raw!r}"
        )

        save(False)

    print("\n=== FACTORIAL INTERACTIONS ===")

    for probe in PROBES:
        for metric in (
            "within_pair_margin",
            "fixed_cross_role_margin",
        ):
            for low, high in (
                ("direct", "identity"),
                ("direct", "relation"),
                ("identity", "relation"),
            ):
                value = factorial_interaction(
                    records, probe, metric, low, high
                )

                print(
                    f"{probe:7s} {metric:23s} "
                    f"{low:8s} -> {high:8s} "
                    f"I={value:+.4f}"
                )

    print("\nInterpretation:")
    print("One development context; no statistical generalization.")
    print("Identity descriptions are not fully matched lexical controls.")
    print("Interactions do not identify neural mechanisms.")

    save(True)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
