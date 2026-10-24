"""Local candidate discrimination versus global answer competition.

A measurement utility only. Native generated answers must be supplied; never
infer generation from a candidate-constrained logit ranking.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class QueryResult:
    gold: str
    logits: Mapping[str, float]
    generated_answer: str | None = None

    def validate(self) -> None:
        if self.gold not in self.logits:
            raise ValueError("Candidate logits must include the gold answer")
        if len(self.logits) < 2:
            raise ValueError("At least two candidate names required")
        for name, value in self.logits.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Candidate names must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("All candidate logits must be finite numbers")

    def global_gold_margin(self) -> float:
        self.validate()
        return float(self.logits[self.gold] - max(
            val for name, val in self.logits.items() if name != self.gold
        ))

    def unique_candidate_winner(self) -> str | None:
        """Winner among listed candidates, not a native generation."""
        self.validate()
        peak = max(self.logits.values())
        winners = [name for name, value in self.logits.items() if value == peak]
        return winners[0] if len(winners) == 1 else None


@dataclass(frozen=True)
class PairDiagnostic:
    old_gold: str
    new_gold: str
    old_vs_new_before: float
    new_vs_old_after: float
    change_in_new_vs_old: float
    global_gold_margin_before: float
    global_gold_margin_after: float
    local_pair_correct_both: bool
    candidate_winner_before: str | None
    candidate_winner_after: str | None
    generated_answer_before: str | None
    generated_answer_after: str | None
    generated_correct_before: bool | None
    generated_correct_after: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_pair(before: QueryResult, after: QueryResult) -> PairDiagnostic:
    """Compare a targeted role-filler replacement under the same query.

    Caller must independently verify relation/question identity and provenance.
    """
    before.validate()
    after.validate()
    if set(before.logits) != set(after.logits):
        raise ValueError("Paired candidate sets differ")
    if before.gold == after.gold:
        raise ValueError("Counterfactual must change the relevant gold filler")
    old_name, new_name = before.gold, after.gold
    old_over_new = float(before.logits[old_name] - before.logits[new_name])
    new_over_old = float(after.logits[new_name] - after.logits[old_name])

    def correct(rec: QueryResult) -> bool | None:
        return None if rec.generated_answer is None else rec.generated_answer == rec.gold

    return PairDiagnostic(
        old_gold=old_name,
        new_gold=new_name,
        old_vs_new_before=old_over_new,
        new_vs_old_after=new_over_old,
        change_in_new_vs_old=float(
            (after.logits[new_name] - after.logits[old_name]) -
            (before.logits[new_name] - before.logits[old_name])
        ),
        global_gold_margin_before=before.global_gold_margin(),
        global_gold_margin_after=after.global_gold_margin(),
        local_pair_correct_both=old_over_new > 0 and new_over_old > 0,
        candidate_winner_before=before.unique_candidate_winner(),
        candidate_winner_after=after.unique_candidate_winner(),
        generated_answer_before=before.generated_answer,
        generated_answer_after=after.generated_answer,
        generated_correct_before=correct(before),
        generated_correct_after=correct(after),
    )
