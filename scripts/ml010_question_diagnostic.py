
import json
import re
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate
from ml003_role_margins import MODELS, log_likelihood


MODEL = "llama3b"
NAMES = ("Leo", "Felix", "Nora", "Clara")

QUESTIONS = {
    "original": "Who wrote the note?",
    "note_by": "Who is the note by?",
    "note_author": "Who is identified as the author of the note?",
    "letter_control": "Who wrote the letter?",
}


def parse_answer(text):
    cleaned = text.strip().strip("\"'")
    cleaned = re.sub(r"[.!?,;:\s]+$", "", cleaned).strip()
    return next(
        (name for name in NAMES if name.casefold() == cleaned.casefold()),
        None,
    )


def main():
    files = sorted(
        Path("results").glob("ml010_llama3b_active_bridge_*.json")
    )

    if not files:
        raise FileNotFoundError("ML-010 active-bridge results not found")

    source = files[-1]
    data = json.loads(source.read_text())

    trials = [
        record for record in data["records"]
        if record["role"] == "note"
    ]

    assert len(trials) == 4

    print("Source:", source)
    print("Loading:", MODEL)

    model, tokenizer = load(str(MODELS / MODEL))

    results = []

    for trial in trials:
        actor = trial["actor"]
        writer = trial["writer"]

        print(f"\n=== Letter writer: {actor} | Note author: {writer} ===")

        for condition, question in QUESTIONS.items():
            gold = actor if condition == "letter_control" else writer

            content = (
                "Answer with exactly one person's name.\n\n"
                f"Passage: {trial['passage']}\n"
                f"Question: {question}\n"
                "Answer:"
            )

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )

            if condition == "original":
                assert prompt == trial["prompt"], (
                    "Original prompt does not match the saved experiment"
                )

            scores = {}
            for name in NAMES:
                score, _ = log_likelihood(
                    model, tokenizer, prompt, name
                )
                scores[name] = score

            preferred = max(scores, key=scores.get)

            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=12,
                verbose=False,
            )
            native = parse_answer(raw)

            margin = scores[gold] - max(
                score for name, score in scores.items()
                if name != gold
            )

            results.append({
                "actor": actor,
                "writer": writer,
                "condition": condition,
                "passage": trial["passage"],
                "question": question,
                "prompt": prompt,
                "gold": gold,
                "scores": scores,
                "preferred": preferred,
                "native_raw": raw,
                "native": native,
                "gold_margin": margin,
            })

            print(
                f"{condition:14s} "
                f"gold={gold:5s} "
                f"forced={preferred:5s} "
                f"native={str(native):5s} "
                f"margin={margin:+.3f}"
            )

    print("\n=== SUMMARY ===")

    for condition in QUESTIONS:
        group = [
            r for r in results
            if r["condition"] == condition
        ]
        forced = sum(r["preferred"] == r["gold"] for r in group)
        native = sum(r["native"] == r["gold"] for r in group)

        print(
            f"{condition:14s} "
            f"forced={forced}/4 "
            f"native={native}/4"
        )

    output = Path("results") / (
        "ml010_question_diagnostic_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-010 question diagnostic",
            "source": str(source),
            "model": MODEL,
            "records": results,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
