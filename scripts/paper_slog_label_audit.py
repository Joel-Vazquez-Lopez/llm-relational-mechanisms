#!/usr/bin/env python3
"""SLOG public-development label audit; NO model inference or held-out data reads.

This program samples annotation projections for human inspection. It uses only
the already downloaded cogs_LF/dev.tsv file and creates a *local* JSONL with
sentences, candidate event-verb identifiers and gold agent/theme surface heads.
Do not commit the raw review file to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

from paper_slog_preflight import extract, selftest

EXPECTED_DEV_SHA256 = "177128c3f64be56732a102db119139bbec33a771c485a525ea0f9f4e45926a5c"
ALLOWED_CATEGORIES = {
    "in_distribution",
    "object_modifying_RC",
    "center_embed_2",
    "center_embed_4",
    "cp_4",
    "pp_4",
    "wh_Q_simple_trans",
}
PER_CATEGORY = 8
EXTRA_TWO_EVENT = 8
EXTRA_FOUR_DISTINCT = 8


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_four(event_a: dict, event_b: dict) -> bool:
    names = [
        event_a["agent_surface"], event_a["theme_surface"],
        event_b["agent_surface"], event_b["theme_surface"]
    ]
    return len({name.casefold() for name in names}) == 4


def choose(pool: list[dict], n: int, salt: str) -> list[dict]:
    # Fixed seed-free cryptographic ordering. Never use outcomes.
    return sorted(
        pool,
        key=lambda r: sha(f"{salt}|{r['row_1based']}|{r['sentence']}".encode())
    )[:n]


def preflight_hardware() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = "not_installed"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not_installed"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path,
                        default=Path("data/external/slog/cogs_LF/dev.tsv"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/paper_slog/local_dev_label_review.jsonl"))
    args = parser.parse_args()
    selftest()

    if args.dev.name != "dev.tsv" or args.dev.parent.name != "cogs_LF":
        raise ValueError("Only official public SLOG cogs_LF/dev.tsv is permitted")
    raw = args.dev.read_bytes()
    digest = sha(raw)
    if digest != EXPECTED_DEV_SHA256:
        raise ValueError(f"Dev dataset SHA mismatch: {digest}")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite review: {args.out}")

    categories = Counter()
    pools = defaultdict(list)
    multi = []
    disjoint = []
    for i, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        cells = line.split("\t")
        if len(cells) != 3:
            raise ValueError(f"Bad TSV structure on row {i}")
        sentence, lf, category = cells
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unexpected category {category!r} on row {i}")
        categories[category] += 1
        events, valid, reasons = extract(sentence, lf)
        if len(events) < 2 or not valid:
            continue
        obj = {
            "row_1based": i,
            "category": category,
            "sentence": sentence,
            "logical_form": lf,
            "eligible_events": valid,
            "total_annotated_predicates": len(events),
            "projection_note": "Answer labels are surface HEADS, not full NPs.",
        }
        pools[category].append(obj)
        if len(valid) >= 2:
            multi.append(obj)
        if any(exact_four(a, b) for j, a in enumerate(valid)
               for b in valid[j+1:]):
            disjoint.append(obj)

    selected = {}
    for c in sorted(ALLOWED_CATEGORIES):
        for row in choose(pools[c], PER_CATEGORY, f"category:{c}"):
            selected[row["row_1based"]] = row
    for pool, n, salt in [
        (multi, EXTRA_TWO_EVENT, "two_eligible"),
        (disjoint, EXTRA_FOUR_DISTINCT, "four_distinct"),
    ]:
        for row in choose(pool, n, salt):
            selected[row["row_1based"]] = row

    rows = sorted(selected.values(), key=lambda r: r["row_1based"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    print("SLOG LABEL AUDIT — PUBLIC DEV ONLY / NO MODEL EVALUATION")
    print("SHA256:", digest)
    print("Category counts:", dict(sorted(categories.items())))
    print("Eligible multi-event category counts:",
          {c: len(v) for c, v in sorted(pools.items())})
    print("Multiple eligible role events:", len(multi))
    print("Four-distinct-participant sentences:", len(disjoint))
    print("Selected unique sentences for manual audit:", len(rows))
    print("Local review file:", args.out)
    print("Hardware snapshot:", json.dumps(preflight_hardware(), sort_keys=True))

    print("\nA FEW AUDIT EXAMPLES — examine full JSONL before freezing:")
    for row in rows[:14]:
        ev = row["eligible_events"][0]
        print(
            f"row={row['row_1based']:4} cat={row['category']:<20} "
            f"verb={ev['verb_surface']!r:<12} "
            f"agent={ev['agent_surface']!r:<14} "
            f"theme={ev['theme_surface']!r:<14}"
        )
        print("    " + row["sentence"])
    print("\nDo not interpret as model accuracy; no checkpoint or protected test read.")
    print("Review the local JSONL for incorrect LF-to-surface projections.")


if __name__ == "__main__":
    main()
