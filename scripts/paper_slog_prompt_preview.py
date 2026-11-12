#!/usr/bin/env python3
"""Preview SLOG agent/theme QA prompts; public development data only.

Reads the manually reviewable SLOG dev-sample JSONL from
paper_slog_label_audit.py. Creates a local prompt preview that includes
gold answers FOR HUMAN LABEL VALIDATION ONLY; it must NEVER be used as
a scored model input or protected evaluation. No model is imported or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from paper_slog_label_audit import EXPECTED_DEV_SHA256
from paper_slog_preflight import extract


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mark_verb(sentence: str, index: int, expected: str) -> str:
    tokens = sentence.split()
    if not 0 <= index < len(tokens):
        raise ValueError("Predicate index outside sentence")
    if tokens[index] != expected:
        raise ValueError(f"Predicate index alignment failed: {tokens[index]!r} != {expected!r}")
    tokens[index] = f"[EVENT]{tokens[index]}[/EVENT]"
    return " ".join(tokens)


def make_question(sentence: str, event: dict, role: str, order: int) -> dict:
    if role not in {"agent", "theme"} or order not in (0, 1):
        raise ValueError("Unexpected role or option-order")
    agent = event["agent_surface"]
    theme = event["theme_surface"]
    candidates = sorted([agent, theme], key=lambda s: (s.casefold(), s))
    if order:
        candidates.reverse()
    if len(set(x.casefold() for x in candidates)) != 2:
        raise ValueError("Candidate labels not distinct")
    gold = event[f"{role}_surface"]
    label = "A" if candidates[0] == gold else "B"
    target = mark_verb(sentence, event["verb_position_0based"], event["verb_surface"])
    if role == "agent":
        question = "Which entity is the AGENT (doer or performer) of the marked event?"
    else:
        question = "Which entity is the THEME (the participant acted upon, perceived, or affected) of the marked event?"
    prompt = (
        "Read this sentence carefully. [EVENT]...[/EVENT] marks the event to answer about.\n"
        f"Sentence: {target}\n"
        f"Question: {question}\n"
        f"A. {candidates[0]}\nB. {candidates[1]}\n"
        "Answer with only A or B."
    )
    assert label in {"A", "B"}
    return {
        "role": role, "order": order, "candidates": candidates,
        "gold_option": label, "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", type=Path, default=Path("data/external/slog/cogs_LF/dev.tsv"))
    ap.add_argument("--review", type=Path,
                    default=Path("reports/paper_slog/local_dev_label_review.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/paper_slog/local_dev_prompt_preview.jsonl"))
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    if args.dev.name != "dev.tsv" or args.dev.parent.name != "cogs_LF":
        raise ValueError("Only public SLOG development file allowed")
    if h(args.dev.read_bytes()) != EXPECTED_DEV_SHA256:
        raise ValueError("SLOG dev SHA256 mismatch")
    original = args.dev.read_text(encoding="utf-8").splitlines()
    if args.out.exists():
        raise FileExistsError("Preview exists; refusing to overwrite: " + str(args.out))
    review = [json.loads(x) for x in args.review.read_text(encoding="utf-8").splitlines()]
    if len(review) != 50:
        raise ValueError(f"Unexpected review size: {len(review)}; expected 50")
    if len(set(r["row_1based"] for r in review)) != len(review):
        raise ValueError("Duplicate selected rows")

    cases = []
    counts = Counter()
    for row in review:
        line_num = row["row_1based"]
        if not 1 <= line_num <= len(original):
            raise ValueError(f"Bad dev row index: {line_num}")
        sentence, lf, category = original[line_num - 1].split("\t")
        if row["sentence"] != sentence or row["logical_form"] != lf or row["category"] != category:
            raise ValueError(f"JSONL/dev mismatch on line {line_num}")
        _, valid, _ = extract(sentence, lf)
        if valid != row["eligible_events"]:
            raise ValueError(f"Extractor mismatch on line {line_num}")

        counts[category] += 1
        # Preview every eligible event in a chosen sentence, not just the
        # first event, to test whether predicate marking remains unambiguous.
        for i, event in enumerate(valid):
            variants = [make_question(sentence, event, role, order)
                        for role in ("agent", "theme") for order in (0, 1)]
            assert [v["gold_option"] for v in variants[:2]] in (["A", "B"], ["B", "A"])
            assert [v["gold_option"] for v in variants[2:]] in (["A", "B"], ["B", "A"])
            cases.append({
                "row_1based": line_num, "category": category,
                "event_index": i, "sentence": sentence,
                "event": event, "preview_only": True,
                "variants": variants,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    print("SLOG PUBLIC-DEV QUESTION PREVIEW / NO MODEL INFERENCE")
    print("Selected sentences:", len(review))
    print("Eligible selected predicate events:", len(cases))
    print("Balanced agent/theme × candidate-order question variants:", 4 * len(cases))
    print("Categories:", dict(sorted(counts.items())))
    print("Local preview:", args.out)
    for case in cases[:args.show]:
        print(f"\nrow={case['row_1based']} category={case['category']} event={case['event']['verb_surface']}")
        print(case["variants"][0]["prompt"])
        print("DEVELOPMENT GOLD (not a scored model input):",
              case["variants"][0]["gold_option"],
              case["variants"][2]["gold_option"])
    print("\nManual review: verify marked verb and agent/theme heads for every event.")
    print("Do not commit the preview JSONL: it contains original benchmark sentences and gold labels.")


if __name__ == "__main__":
    main()
