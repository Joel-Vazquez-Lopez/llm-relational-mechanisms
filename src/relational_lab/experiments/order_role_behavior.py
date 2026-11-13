"""Behavioral evaluation for the order-to-role stimuli.

Dependency-free: the model-specific part is a *scorer* that returns, for each
item, the log-probability of the correct name and of the distractor name as a
continuation of the prompt, and whether the correct name's first token is the
model's top-1 token over the full vocabulary. Everything else (accuracy per
cell, event-level both-correct, cross-construction consistency, bootstrap
intervals over name-pair families and the pre-registered gate) lives here.

Candidate-restricted choice and full-vocabulary correctness are reported
separately, as the repository's evidence policy requires.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from relational_lab.experiments.order_role_stimuli import (
    MAIN_CONSTRUCTIONS,
    QUESTION_VOICES,
    ROLES,
    Item,
)

GATE_THRESHOLD = 0.90  # pre-registered: candidate-restricted accuracy per construction x role


@dataclass(frozen=True)
class Score:
    item_id: str
    logprob_answer: float
    logprob_distractor: float
    answer_is_top1: bool  # full vocabulary, first token of the answer

    @property
    def margin(self) -> float:
        return self.logprob_answer - self.logprob_distractor

    @property
    def choice_correct(self) -> bool:
        return self.margin > 0


Scorer = Callable[[Sequence[Item]], list[Score]]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(
    values_by_family: dict[str, list[float]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of the mean, resampling families (all items of one
    family stay together)."""
    families = sorted(values_by_family)
    if not families:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in families:
            sample.extend(values_by_family[rng.choice(families)])
        means.append(_mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def _cell_summary(rows: list[tuple[Item, Score]], n_boot: int, seed: int) -> dict:
    by_family: dict[str, list[float]] = defaultdict(list)
    for it, sc in rows:
        by_family[it.family_id].append(float(sc.choice_correct))
    lo, hi = bootstrap_ci(by_family, n_boot=n_boot, seed=seed)
    return {
        "n": len(rows),
        "choice_accuracy": _mean([float(sc.choice_correct) for _, sc in rows]),
        "choice_accuracy_ci95": [lo, hi],
        "full_vocab_top1": _mean([float(sc.answer_is_top1) for _, sc in rows]),
        "mean_margin": _mean([sc.margin for _, sc in rows]),
    }


def summarize(
    items: Sequence[Item],
    scores: Iterable[Score],
    n_boot: int = 2000,
    seed: int = 0,
    gate_threshold: float = GATE_THRESHOLD,
) -> dict:
    by_id = {s.item_id: s for s in scores}
    missing = [it.item_id for it in items if it.item_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} items have no score, e.g. {missing[0]}")
    pairs = [(it, by_id[it.item_id]) for it in items]

    cells: dict[str, dict] = {}
    groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for it, sc in pairs:
        groups[(it.construction, it.role_asked, it.question_voice)].append((it, sc))
    for (c, r, q), rows in sorted(groups.items()):
        cells[f"{c}/{r}/{q}"] = _cell_summary(rows, n_boot, seed)

    by_construction_role: dict[str, dict] = {}
    cr_groups: dict[tuple[str, str], list] = defaultdict(list)
    for it, sc in pairs:
        cr_groups[(it.construction, it.role_asked)].append((it, sc))
    for (c, r), rows in sorted(cr_groups.items()):
        by_construction_role[f"{c}/{r}"] = _cell_summary(rows, n_boot, seed)

    # Event-level: both role questions right for the same sentence and question voice.
    both: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    for it, sc in pairs:
        both[(it.event_id, it.construction, it.question_voice)][it.role_asked] = sc.choice_correct
    both_by_construction: dict[str, list[float]] = defaultdict(list)
    for (_, c, _), d in both.items():
        if set(d) == set(ROLES):
            both_by_construction[c].append(float(all(d.values())))
    both_correct = {c: _mean(v) for c, v in sorted(both_by_construction.items())}

    # Consistency: same answer chosen across all four main constructions,
    # for the same event, role and question voice. Agreement is reported
    # separately from accuracy: answers can agree and still be wrong.
    chosen: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for it, sc in pairs:
        if it.construction in MAIN_CONSTRUCTIONS:
            pick = it.answer if sc.choice_correct else it.distractor
            chosen[(it.event_id, it.role_asked, it.question_voice)][it.construction] = pick
    complete = [d for d in chosen.values() if set(d) == set(MAIN_CONSTRUCTIONS)]
    consistency = _mean([float(len(set(d.values())) == 1) for d in complete])

    # Question-wording effect: active minus passive question, per construction x role.
    wording = {}
    for c in sorted({it.construction for it in items}):
        for r in ROLES:
            acc = {}
            for q in QUESTION_VOICES:
                key = f"{c}/{r}/{q}"
                if key in cells:
                    acc[q] = cells[key]["choice_accuracy"]
            if len(acc) == 2:
                wording[f"{c}/{r}"] = acc["active"] - acc["passive"]

    gate_cells = {
        k: v["choice_accuracy"] >= gate_threshold
        for k, v in by_construction_role.items()
        if k.split("/")[0] in MAIN_CONSTRUCTIONS
    }
    expected = {f"{c}/{r}" for c in MAIN_CONSTRUCTIONS for r in ROLES}
    gate = {
        "threshold": gate_threshold,
        "metric": "candidate-restricted choice accuracy per construction x role",
        "cells": gate_cells,
        "complete": set(gate_cells) == expected,
        "passed": set(gate_cells) == expected and all(gate_cells.values()),
    }

    return {
        "n_items": len(pairs),
        "n_families": len({it.family_id for it in items}),
        "overall_choice_accuracy": _mean([float(sc.choice_correct) for _, sc in pairs]),
        "by_construction_role": by_construction_role,
        "cells": cells,
        "both_roles_correct_by_construction": both_correct,
        "cross_construction_consistency": consistency,
        "question_wording_effect": wording,
        "gate": gate,
    }


def score_to_json(score: Score) -> dict:
    d = asdict(score)
    d["margin"] = score.margin
    return d
