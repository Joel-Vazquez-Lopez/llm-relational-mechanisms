
import argparse
import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path

from mlx_lm import load, generate
from ml003_role_margins import MODELS, log_likelihood


NAMES = ("Leo", "Felix", "Nora", "Clara")
LINKS = ("wrote", "carried")

QUESTIONS = {
    "event": "Who norped Mira?",
    "writer": "Who wrote the letter?",
    "carrier": "Who carried the letter?",
    "note": "Who wrote the note?",
}


def build_passage(writer, carrier, link, order):
    bridge = (
        f"At dawn, the person who {link} "
        "the letter norped Mira."
    )

    facts = (
        f"{writer} wrote the letter. "
        f"{carrier} carried the letter."
    )

    distractors = "Nora wrote the note. Clara watched."

    if order == "bridge_first":
        passage = f"{bridge} {facts} {distractors}"
    else:
        passage = f"{facts} {bridge} {distractors}"

    assert all(passage.count(name) == 1 for name in NAMES)
    return passage


def parse_answer(raw):
    answer = re.sub(
        r"""^[\s"']+|[\s"'.,!?]+$""",
        "",
        raw,
    )
    return next(
        (n for n in NAMES if n.casefold() == answer.casefold()),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("llama3b", "qwen7b"),
        required=True,
    )
    parser.add_argument(
        "--full",
        action="store_true",
    )
    args = parser.parse_args()

    assignments = (
        (("Leo", "Felix"), ("Felix", "Leo"))
        if args.full else
        (("Leo", "Felix"),)
    )
    orders = (
        ("bridge_first", "facts_first")
        if args.full else
        ("bridge_first",)
    )

    print("Loading:", args.model)
    model, tokenizer = load(str(MODELS / args.model))

    records = []
    lookup = {}

    for (writer, carrier), order, link, probe in product(
        assignments, orders, LINKS, QUESTIONS
    ):
        passage = build_passage(writer, carrier, link, order)

        gold = {
            "event": writer if link == "wrote" else carrier,
            "writer": writer,
            "carrier": carrier,
            "note": "Nora",
        }[probe]

        content = (
            "Answer with exactly one person's name.\n\n"
            f"Passage: {passage}\n"
            f"Question: {QUESTIONS[probe]}\n"
            "Answer:"
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        scores = {}
        counts = {}

        for name in NAMES:
            lp, count = log_likelihood(
                model, tokenizer, prompt, name
            )
            scores[name] = lp
            counts[name] = count

        forced = max(NAMES, key=lambda n: scores[n])
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=12,
            verbose=False,
        )
        native = parse_answer(raw)

        record = {
            "model": args.model,
            "writer": writer,
            "carrier": carrier,
            "order": order,
            "link": link,
            "probe": probe,
            "passage": passage,
            "question": QUESTIONS[probe],
            "prompt": prompt,
            "gold": gold,
            "scores": scores,
            "token_counts": counts,
            "forced": forced,
            "native_raw": raw,
            "native": native,
            "correct_margin": (
                scores[gold]
                - max(scores[n] for n in NAMES if n != gold)
            ),
        }

        records.append(record)
        lookup[(writer, carrier, order, link, probe)] = record

        print(
            f"{link:7s} {probe:7s} "
            f"gold={gold:5s} "
            f"forced={forced:5s} "
            f"native={str(native):5s} "
            f"margin={record['correct_margin']:+.3f}"
        )

    print("\n=== SELECTIVE DEPENDENCY SWITCH ===")

    for (writer, carrier), order in product(assignments, orders):
        def preference(link, probe):
            scores = lookup[
                (writer, carrier, order, link, probe)
            ]["scores"]
            return scores[writer] - scores[carrier]

        target = (
            preference("wrote", "event")
            - preference("carried", "event")
        )

        control = (
            preference("wrote", "writer")
            - preference("carried", "writer")
        )

        print(
            f"writer={writer} carrier={carrier} "
            f"order={order}"
        )
        print(f"  Event switch: {target:+.3f}")
        print(f"  Unchanged-fact control: {control:+.3f}")
        print(f"  Selective effect: {target-control:+.3f}")

    forced_correct = sum(
        r["forced"] == r["gold"] for r in records
    )
    native_correct = sum(
        r["native"] == r["gold"] for r in records
    )

    print("\n=== SUMMARY ===")
    print(f"Forced correct: {forced_correct}/{len(records)}")
    print(f"Native correct: {native_correct}/{len(records)}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / (
        f"ml011_{args.model}_"
        f"{datetime.now():%Y%m%dT%H%M%S%f}.json"
    )

    output.write_text(
        json.dumps({
            "experiment": "ML-011",
            "model": args.model,
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
