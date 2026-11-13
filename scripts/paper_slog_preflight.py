#!/usr/bin/env python3
"""Read-only SLOG development-split feasibility audit for event agent/theme QA.

Never reads SLOG's generalization archive or any held-out/test split.
Only emits dataset-eligibility summaries and SHA256 fingerprints, not model scores.

Usage:
    python scripts/paper_slog_preflight.py --phase selftest
    python scripts/paper_slog_preflight.py --phase preflight \
        --dev data/external/slog/cogs_LF/dev.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# SLOG cogs_LF contains clauses like:
# draw . agent ( x _ 5 , Isabella ) AND draw . theme ( x _ 5 , x _ 7 )
ROLE_RE = re.compile(
    r"([A-Za-z][A-Za-z-]*)\s*\.\s*(agent|theme)\s*\(\s*"
    r"x\s*_\s*(\d+)\s*,\s*(x\s*_\s*\d+|[A-Za-z][A-Za-z'-]*)\s*\)"
)
VAR_RE = re.compile(r"^x\s*_\s*(\d+)$")
TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_arg(arg: str, tokens: list[str]) -> str | None:
    match = VAR_RE.fullmatch(arg)
    if match is None:
        return arg if TOKEN_RE.fullmatch(arg) else None
    idx = int(match.group(1))
    return tokens[idx] if 0 <= idx < len(tokens) else None


def extract(sentence: str, logical_form: str):
    tokens = sentence.split()
    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for m in ROLE_RE.finditer(logical_form):
        lemma, role, pos, argument = m.groups()
        key = lemma, int(pos)
        item = indexed.setdefault(key, {})
        if role in item and item[role] != argument:
            item["duplicate_role"] = "1"
        item[role] = argument

    valid: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for (lemma, idx), roles in sorted(indexed.items(), key=lambda kv: kv[0][1]):
        if not {"agent", "theme"} <= roles.keys():
            reasons["missing_agent_or_theme"] += 1
            continue
        if "duplicate_role" in roles:
            reasons["multiple_arguments_same_role"] += 1
            continue
        if not 0 <= idx < len(tokens):
            reasons["predicate_index_out_of_range"] += 1
            continue
        verb = tokens[idx]
        agent = resolve_arg(roles["agent"], tokens)
        theme = resolve_arg(roles["theme"], tokens)
        if not agent or not theme or not TOKEN_RE.fullmatch(verb):
            reasons["unresolvable_or_nonlexical"] += 1
            continue
        if agent.casefold() == theme.casefold():
            reasons["same_surface_answer"] += 1
            continue
        occurrences = lambda word: sum(t.casefold() == word.casefold() for t in tokens)
        if occurrences(agent) != 1 or occurrences(theme) != 1:
            reasons["ambiguous_answer_mentions"] += 1
            continue
        if occurrences(verb) != 1:
            reasons["ambiguous_predicate_mention"] += 1
            continue
        valid.append({
            "lemma": lemma,
            "verb_position_0based": idx,
            "verb_surface": verb,
            "agent_surface": agent,
            "theme_surface": theme,
        })
    return indexed, valid, reasons


def selftest() -> None:
    sentence = "The dog noticed that Isabella drew the girl ."
    lf = (
        "* dog ( x _ 1 ) ; * girl ( x _ 7 ) ; "
        "notice . agent ( x _ 2 , x _ 1 ) AND "
        "notice . ccomp ( x _ 2 , x _ 5 ) AND "
        "draw . agent ( x _ 5 , Isabella ) AND "
        "draw . theme ( x _ 5 , x _ 7 )"
    )
    all_events, valid, reasons = extract(sentence, lf)
    assert len(all_events) == 2
    assert len(valid) == 1
    assert valid[0]["verb_surface"] == "drew"
    assert valid[0]["agent_surface"] == "Isabella"
    assert valid[0]["theme_surface"] == "girl"
    assert not reasons.get("same_surface_answer")
    passive = "A raisin was missed by the girl ."
    lf2 = "miss . theme ( x _ 3 , x _ 1 ) AND miss . agent ( x _ 3 , x _ 6 )"
    ev, v, _ = extract(passive, lf2)
    assert len(ev) == len(v) == 1
    assert (v[0]["agent_surface"], v[0]["theme_surface"]) == ("girl", "raisin")
    assert resolve_arg("x _ 88", ["A"]) is None
    print("SELFTEST PASS: event-token indexing, agent/theme alignment, passive ordering")


def preflight(path: Path, out: Path) -> None:
    if path.name != "dev.tsv" or path.parent.name != "cogs_LF":
        raise ValueError("Only the public SLOG cogs_LF/dev.tsv is allowed in preflight")
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines()
    counters: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    eligible_by_category: Counter[str] = Counter()
    disjoint_by_category: Counter[str] = Counter()
    for line_idx, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"TSV row {line_idx}: expected sentence, LF, category")
        sentence, lf, category = fields
        if not category.strip():
            raise ValueError(f"Missing development category on row {line_idx}")
        # Public SLOG development files include structural categories, not
        # just 'in_distribution'. Treat their labels as metadata, not errors.
        categories[category] += 1
        events, eligible, why = extract(sentence, lf)
        counters["sentences"] += 1
        counters["all_predicate_events_with_agent_or_theme"] += len(events)
        counters["valid_agent_theme_events_any_sentence"] += len(eligible)
        counters["sentences_with_at_least_two_predicates"] += len(events) >= 2
        rejects.update(why)
        if len(events) < 2:
            continue
        if eligible:
            counters["eligible_multi_event_sentences"] += 1
            eligible_by_category[category] += 1
            counters["eligible_multi_event_role_pairs"] += len(eligible)
            if len(eligible) >= 2:
                counters["multi_event_sentences_with_two_eligible_events"] += 1
                if any(
                    len({
                        *[a["agent_surface"].casefold(), a["theme_surface"].casefold()],
                        *[b["agent_surface"].casefold(), b["theme_surface"].casefold()]
                    }) == 4
                    for i, a in enumerate(eligible)
                    for b in eligible[i+1:]
                ):
                    counters["multi_event_sentences_with_disjoint_four_participants"] += 1
                    disjoint_by_category[category] += 1
            counters["multi_event_with_passive_surface_cue"] += bool(
                re.search(r"\bwas\b.*\bby\b", sentence, flags=re.I)
            )
            counters["multi_event_with_that_who_which"] += bool(
                re.search(r"\b(that|who|which)\b", sentence, flags=re.I)
            )
    report = {
        "experiment": "SLOG external validation: public-development eligibility preflight",
        "status": "NO_MODEL_SCORING_NO_HELDOUT_DATA_READ",
        "official_source": "https://github.com/bingzhilee/SLOG",
        "dataset": "data/cogs_LF/dev.tsv",
        "dataset_sha256": sha256(data),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "metrics": dict(sorted(counters.items())),
        "development_categories": dict(sorted(categories.items())),
        "eligible_multi_event_by_category": dict(sorted(eligible_by_category.items())),
        "four_participant_multi_event_by_category": dict(sorted(disjoint_by_category.items())),
        "rejection_reasons": dict(sorted(rejects.items())),
        "eligibility_note": (
            "Unique surface agent/theme and predicate token, >=2 LF predicates. "
            "SLOG theme is not identical to synthetic patient. "
            "This is only an annotation projection feasibility check; lexical and "
            "structural validity require separate auditing before any held-out run."
        ),
        "not_tested": [
            "Structural generalization archive", "any pretrained checkpoint",
            "out-of-distribution accuracy", "semantic-role model predictions"
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight report: {out}")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SLOG PUBLIC DEVELOPMENT PREFLIGHT")
    print("dataset SHA256:", report["dataset_sha256"])
    for k, v in report["metrics"].items():
        print(f"{k}: {v}")
    print("Development categories:", report["development_categories"])
    print("Eligible multi-event cases by category:", report["eligible_multi_event_by_category"])
    print("Four-participant cases by category:", report["four_participant_multi_event_by_category"])
    print("No protected generalization archive/test split or trained-model outputs read.")
    print("Saved:", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("selftest", "preflight"))
    p.add_argument("--dev", type=Path, default=Path("data/external/slog/cogs_LF/dev.tsv"))
    p.add_argument("--out", type=Path, default=Path("reports/paper_slog/dev_preflight.json"))
    args = p.parse_args()
    selftest()
    if args.phase == "preflight":
        preflight(args.dev, args.out)


if __name__ == "__main__":
    main()
