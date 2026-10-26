
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate

from ml003_role_margins import MODELS, log_likelihood
from ml020_double_binding import make_prompt, parse_native


NAMES = ("Leo", "Felix", "Nora", "Clara")


def evaluate(model, tokenizer, passage, question, gold,
             group, condition, agent, patient):
    prompt = make_prompt(tokenizer, passage, question)

    scores = {
        name: float(log_likelihood(
            model, tokenizer, prompt, name
        )[0])
        for name in NAMES
    }

    forced = max(scores, key=scores.get)
    raw = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=12,
        verbose=False,
    )
    native = parse_native(raw, NAMES)

    best_patient = max(scores[n] for n in patient)
    best_agent = max(scores[n] for n in agent)

    return {
        "group": group,
        "condition": condition,
        "passage": passage,
        "question": question,
        "gold": gold,
        "scores": scores,
        "forced": forced,
        "native": native,
        "native_raw": raw,
        "patient_minus_agent": best_patient - best_agent,
        "gold_margin": scores[gold] - max(
            scores[n] for n in NAMES if n != gold
        ),
    }


def main():
    paths = sorted(Path("results").glob("ml020_dev_*.json"))
    assert paths, "Missing ML-020 development results"

    source = paths[-1]
    data = json.loads(source.read_text())

    assert data["complete"]
    original = [
        r for r in data["records"]
        if r["probe"] == "patient"
    ]
    assert len(original) == 4

    jobs = []

    # Same complex passages, alternative patient questions.
    for r in original:
        c = r["config"]
        selector = r["agent_selector"]

        variants = {
            "active_object": (
                f"Who did the person who {selector} "
                "the letter norp?"
            ),
            "grammatical_object": (
                "In the dusk sentence, who is the direct "
                "object of the verb 'norped'?"
            ),
        }

        for condition, question in variants.items():
            jobs.append({
                "passage": r["passage"],
                "question": question,
                "gold": r["gold"],
                "group": "complex",
                "condition": condition,
                "agent": (c["writer"], c["carrier"]),
                "patient": (c["folder"], c["painter"]),
            })

    # Simple transitive sentence controls.
    for agent, patient in (
        ("Leo", "Nora"),
        ("Nora", "Leo"),
    ):
        for verb, base in (
            ("norped", "norp"),
            ("praised", "praise"),
        ):
            passage = f"At dusk, {agent} {verb} {patient}."

            variants = {
                f"{verb}_passive": f"Who was {verb}?",
                f"{verb}_active": (
                    f"Who did {agent} {base}?"
                ),
            }

            for condition, question in variants.items():
                jobs.append({
                    "passage": passage,
                    "question": question,
                    "gold": patient,
                    "group": "simple",
                    "condition": condition,
                    "agent": (agent,),
                    "patient": (patient,),
                })

    assert len(jobs) == 16
    print("Source:", source)
    print("New diagnostic prompts:", len(jobs))
    print("Held-out evaluation: untouched")

    print("\nLoading Llama 3B")
    model, tokenizer = load(str(MODELS / "llama3b"))

    output = Path("results") / (
        "ml020_patient_diagnostic_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    records = []

    for job in jobs:
        result = evaluate(
            model, tokenizer,
            **job
        )
        records.append(result)

        print(
            f"{result['group']:7s} "
            f"{result['condition']:19s} "
            f"gold={result['gold']:5s} "
            f"forced={result['forced']:5s} "
            f"native={str(result['native']):5s} "
            f"patient-agent="
            f"{result['patient_minus_agent']:+.3f}"
        )

        output.write_text(
            json.dumps({
                "experiment": "ML-020D",
                "source": str(source),
                "complete": False,
                "records": records,
            }, indent=2),
            encoding="utf-8",
        )

    print("\n=== SUMMARY ===")
    groups = defaultdict(list)

    for record in records:
        groups[
            (record["group"], record["condition"])
        ].append(record)

    for key, items in sorted(groups.items()):
        forced_correct = sum(
            r["forced"] == r["gold"] for r in items
        )
        native_correct = sum(
            r["native"] == r["gold"] for r in items
        )

        print(
            f"{key[0]:7s} {key[1]:19s} "
            f"forced={forced_correct}/{len(items)} "
            f"native={native_correct}/{len(items)}"
        )

    output.write_text(
        json.dumps({
            "experiment": "ML-020D",
            "source": str(source),
            "complete": True,
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
