
import argparse
import json
from datetime import datetime
from pathlib import Path

from mlx_lm import load
from ml003_role_margins import MODELS, log_likelihood


def make_cases(verb):
    return [
        {
            "id": "event_by_leo",
            "passage": f"Mira was {verb} by Leo near Nora.",
            "agent": "Leo",
        },
        {
            "id": "note_by_leo",
            "passage": (
                f"Mira was {verb} near Nora, "
                "according to a note by Leo."
            ),
            "agent": None,
        },
        {
            "id": "event_by_nora_note_by_leo",
            "passage": (
                f"Mira was {verb} by Nora, "
                "according to a note by Leo."
            ),
            "agent": "Nora",
        },
        {
            "id": "event_by_leo_note_by_nora",
            "passage": (
                f"Mira was {verb} by Leo, "
                "according to a note by Nora."
            ),
            "agent": "Leo",
        },
    ]


def score_yes_no(model, tokenizer, passage, question):
    content = (
        "Answer only Yes or No. "
        "If the passage does not explicitly identify "
        "the person as performing the action, answer No.\n\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        "Answer:"
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    yes_lp, yes_tokens = log_likelihood(
        model, tokenizer, prompt, "Yes"
    )
    no_lp, no_tokens = log_likelihood(
        model, tokenizer, prompt, "No"
    )

    margin = yes_lp - no_lp

    return {
        "prompt": prompt,
        "margin": margin,
        "yes_lp": yes_lp,
        "no_lp": no_lp,
        "yes_tokens": yes_tokens,
        "no_tokens": no_tokens,
        "predicted": "Yes" if margin > 0 else "No",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    args = parser.parse_args()

    model, tokenizer = load(str(MODELS / args.model))

    records = []

    for verb in ("praised", "tavved"):
        print(f"\n=== {verb.upper()} ===")

        for case in make_cases(verb):
            print(f"\n{case['id']}")
            print(case["passage"])

            for person in ("Leo", "Nora"):
                question = (
                    f"Does the passage explicitly identify {person} "
                    "as performing the action on Mira?"
                )

                scored = score_yes_no(
                    model,
                    tokenizer,
                    case["passage"],
                    question,
                )

                expected = (
                    "Yes" if case["agent"] == person else "No"
                )

                record = {
                    "model": args.model,
                    "verb": verb,
                    "case_id": case["id"],
                    "passage": case["passage"],
                    "target_person": person,
                    "question": question,
                    "expected": expected,
                    **scored,
                }

                records.append(record)

                print(
                    f"  {person:5s} "
                    f"Yes-No={scored['margin']:+8.3f} "
                    f"pred={scored['predicted']:3s} "
                    f"gold={expected}"
                )

    correct = sum(
        r["predicted"] == r["expected"]
        for r in records
    )

    print("\n=== SUMMARY ===")
    print(f"Actor-attribution probes: {correct}/{len(records)}")

    print("\n=== CRITICAL LEXICAL TRAP ===")
    for record in records:
        if (
            record["case_id"] == "note_by_leo"
            and record["target_person"] == "Leo"
        ):
            print(
                record["verb"],
                f"Leo actor margin={record['margin']:+.3f}",
                f"prediction={record['predicted']}",
                "expected=No",
            )

    metadata = json.loads(
        (MODELS / "RESTORATION.json").read_text()
    )
    checkpoint = next(
        item for item in metadata
        if item["model"] == args.model
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml008_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps(
            {
                "experiment": "ML-008",
                "model": args.model,
                "revision": checkpoint["revision"],
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
