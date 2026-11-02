
"""Tokenizer-only name-pair repair for the UNSEALED ML-055 design.

No model weights, logits, generation, or past outcomes are inspected.
Default is preview; --apply updates NAMES only after full validation.
"""

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

from transformers import AutoTokenizer

import ml055_design as design
from ml003_role_margins import MODELS
from ml053_design import (
    TRAIN_NAMES,
    TEST_NAMES,
    answer_tokens,
    render,
)


# Additional fresh name candidates.
# Their deterministic order is intentional.
RESERVE = """
Alex Alice Allison Amber Andrea April Arthur Audrey
Bailey Barbara Beth Blake Bridget Bruce
Cara Caroline Casey Cedric Charles Charlotte Chloe Christina
Damon Daniel David Dean Derek Diana Dominic Dorothy
Elena Elizabeth Emily Emma Eric Esther Ethan Eve
Faith Florence Forrest Frances Gabriel Gloria Gordon Graham
Harper Henry Holly Hugo Ian Irene Isabelle Ivan
Jacob Jade James Jane Jasmine Jason Jeremy Jessica Joan Joseph
Jude Julia Justin Kara Kieran Kim Laura Lawrence Leah Lewis
Lilian Linda Logan Lucas Lydia Malcolm Margaret Marion Martin
Mason Matthew Max Meredith Mia Michael Mila Miriam Molly Monica
Naomi Nathan Nina Noah Nora Oliver Olivia Owen Paige Paula
Peter Philip Quentin Rachel Raymond Richard Riley Robert Robin
Ruth Sara Sasha Scott Serena Simon Sophia Stella Steven Susan
Theodore Thomas Timothy Tony Ursula Vera Victor Vincent Violet
Walter Wendy Whitney Yasmin Zachary
""".split()


def digest(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()


def valid_pair(index, first, second, tokenizers):
    past, base = design.VERBS[index]

    for direction in (0, 1):
        agent, patient = (
            (first, second)
            if direction == 0
            else (second, first)
        )

        passage = f"At dawn, {agent} {past} {patient}."

        for style in design.STYLES:
            for role in design.ROLES:
                row = {
                    "passage": passage,
                    "question": design.question(
                        past, base, style, role
                    ),
                }

                for tokenizer in tokenizers.values():
                    prompt = render(tokenizer, row)

                    a = answer_tokens(
                        tokenizer, prompt, first
                    )

                    b = answer_tokens(
                        tokenizer, prompt, second
                    )

                    if (
                        a is None
                        or b is None
                        or a == 0
                        or a != b
                    ):
                        return False

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
    )

    args = parser.parse_args()

    original_cases = design.design()
    original_hash = digest(original_cases)

    expected = (
        "32df691e8750fbf26b60ed39d7520c575cc4b7b3086fe"
        "307b4434fec48a9b22d"
    )

    if original_hash != expected:
        raise RuntimeError(
            "ML-055 design changed since the preview "
            f"({original_hash}). Inspect before repairing."
        )

    if (
        len(design.NAMES) != 32
        or len(design.VERBS) != 16
    ):
        raise RuntimeError(
            "Unexpected ML-055 design dimensions"
        )

    tokenizers = {
        model: AutoTokenizer.from_pretrained(
            str(MODELS / model),
            local_files_only=True,
        )
        for model in ("llama3b", "qwen7b")
    }

    original_pairs = [
        tuple(design.NAMES[2*i:2*i+2])
        for i in range(16)
    ]

    selected = [None] * 16
    occupied = set()
    failures = []

    # Preserve all pairs that are already compatible.
    for i, (a, b) in enumerate(original_pairs):
        if valid_pair(i, a, b, tokenizers):
            selected[i] = (a, b)
            occupied.update((a, b))
        else:
            failures.append(i)

    forbidden = (
        set(TRAIN_NAMES)
        | set(TEST_NAMES)
    )

    pool = [
        name
        for name in dict.fromkeys(
            [*design.NAMES, *RESERVE]
        )
        if name not in occupied
        and name not in forbidden
    ]

    # Replace incompatible pairs using tokenizer
    # compatibility alone, never model predictions.
    for i in failures:
        match = next(
            (
                (a, b)
                for a, b in combinations(pool, 2)
                if valid_pair(
                    i, a, b, tokenizers
                )
            ),
            None,
        )

        if match is None:
            raise RuntimeError(
                f"No compatible unused pair for group {i}; "
                "no files changed"
            )

        selected[i] = match

        pool = [
            name for name in pool
            if name not in match
        ]

    flattened = tuple(
        name
        for pair in selected
        for name in pair
    )

    if (
        len(flattened) != 32
        or len(set(flattened)) != 32
    ):
        raise RuntimeError(
            "Names reused; no files changed"
        )

    # Validate all final conditions with both tokenizers.
    # This changes the imported design only in memory.
    design.NAMES = flattened

    cases = design.design()

    if len(cases) != 192:
        raise RuntimeError(
            "Incorrect number of cases"
        )

    if not all(
        valid_pair(
            i, *selected[i], tokenizers
        )
        for i in range(16)
    ):
        raise RuntimeError(
            "Repair did not pass all tokenizer checks"
        )

    print("ML-055 TOKENIZER-ONLY REPAIR: VALID")
    print("Original hash:", original_hash)
    print("Repaired hash:", digest(cases))
    print(
        "Preserved pairs:",
        16 - len(failures),
    )
    print("Repaired group indices:", failures)

    for i in failures:
        print(
            f"  {i:02d}: "
            f"{original_pairs[i]} -> {selected[i]}"
        )

    print(
        "Matched candidate lengths: "
        "192/192 prompts, BOTH models"
    )

    if not args.apply:
        print(
            "PREVIEW ONLY. "
            "Re-run with --apply to update NAMES."
        )
        return

    # Write only the validated NAMES block.
    source = Path(design.__file__)
    raw = source.read_text()

    replacement = (
        "NAMES = (\n"
        + "".join(
            f'    "{a}", "{b}",\n'
            for a, b in selected
        )
        + ")\n\nVERBS = ("
    )

    updated, count = re.subn(
        r"NAMES = \(\n.*?\n\)\n\nVERBS = \(",
        lambda _: replacement,
        raw,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise RuntimeError(
            "Couldn't locate the NAMES block; "
            "no files changed"
        )

    source.write_text(updated)

    print("UPDATED", source)
    print(
        "Run python scripts/ml055_design.py "
        "and repeat the tokenizer audit."
    )


if __name__ == "__main__":
    main()
