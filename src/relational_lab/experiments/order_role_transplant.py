"""OTR-03 (E3): code transplant — which code does a name's state carry, layer by layer?

A *transplant* copies the residual-stream state of one name token (slot j) from
a SOURCE sentence into one name slot (k) of a BASE sentence, at the same layer.
Source and base share the verb but use disjoint names, so the transplanted
vector cannot steer the answer through name identity: the candidates are the
base's own two names, and the source's names are neither.

Readout at the base's final position:

    Y = [logit(name_k) - logit(other base name)]_patched - [same]_unpatched

i.e. how far the transplant moved the answer toward slot k.

Each slot has three binary labels: is_first (order), is_subject (function),
is_agent (role). The question asks for the agent or the patient. For each code
c we form the predictor

    x_c = [label_c(source slot j) == q] - [label_c(base slot k) == q]

where "== q" means the label matches the canonical answer to question q
(first / subject / agent for an agent question; their negations for a patient
question). x_c in {-1, 0, +1} says whether, IF the model read code c at slot k,
the transplant should make slot k a better (+1) or worse (-1) answer. Across
the four main constructions, both slots and both questions, the three
predictors are uncorrelated by design (tested), so the per-layer regression

    Y = b0 + b_order x_order + b_function x_function + b_role x_role

gives three curves over layers: the code profile.

Caveat: in the four main constructions the focused (clefted / subject)
participant is always the first-mentioned one, so b_order measures
order-or-focus. The held-out pseudo-clefts separate the two later.

The sweep needs torch (lazy import); everything else is dependency-free.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from relational_lab.experiments.order_role_stimuli import (
    MAIN_CONSTRUCTIONS,
    NAMES,
    Item,
    build_item,
)

CODES: tuple[str, ...] = ("order", "function", "role")
SLOTS: tuple[str, ...] = ("first_name", "second_name")


def _slot_labels(item: Item, slot: str) -> dict[str, bool]:
    tok = item.names[0] if slot == "first_name" else item.names[1]
    return {"order": tok.is_first, "function": tok.is_subject, "role": tok.is_agent}


def predictors(base: Item, k: str, source: Item, j: str) -> dict[str, int]:
    """x_c for each code, relative to the base question (see module doc)."""
    want = base.role_asked == "agent"  # canonical answer: first / subject / agent
    lb, ls = _slot_labels(base, k), _slot_labels(source, j)
    return {c: int(ls[c] == want) - int(lb[c] == want) for c in CODES}


@dataclass(frozen=True)
class Transplant:
    base: Item
    source: Item
    k: str  # base slot receiving the state
    j: str  # source slot providing it

    @property
    def x(self) -> dict[str, int]:
        return predictors(self.base, self.k, self.source, self.j)


def _foreign_pair(base: Item, rng: random.Random, names: Sequence[str]) -> tuple[str, str]:
    pool = [n for n in names if n not in (base.agent, base.patient)]
    a, b = rng.sample(pool, 2)
    return a, b


def make_transplants(
    bases: Sequence[Item],
    source_constructions: Sequence[str] = MAIN_CONSTRUCTIONS,
    seed: int = 0,
    names: Sequence[str] = NAMES,
) -> list[Transplant]:
    """For every base: both base slots x every source construction x both source
    slots x BOTH assignments of one foreign name pair (U as agent and V as
    patient, then V as agent and U as patient). The counterbalancing makes each
    foreign name contribute equally to every role, order and function value, so
    name identity cancels exactly instead of only in expectation."""
    rng = random.Random(seed)
    out: list[Transplant] = []
    for base in bases:
        u, v = _foreign_pair(base, rng, names)
        for fa, fp in ((u, v), (v, u)):
            for cs in source_constructions:
                source = build_item(fa, fp, base.lemma, base.verb, cs, base.role_asked, base.question_voice,
                                    family_id=base.family_id, split=base.split, test_set=base.test_set)
                for k in SLOTS:
                    for j in SLOTS:
                        out.append(Transplant(base, source, k, j))
    return out


def left_context_null(records: Sequence[TransplantRecord]) -> list[float]:
    """Validity check, per layer: the role slope among first-name sources from
    active vs passive sentences. Their left context ("Sentence: X") is identical,
    so a causal model's state cannot differ by role there, and with counterbalanced
    names the slope must be ~0. A clearly non-zero value flags a design problem."""
    sub = [r for r in records if r.j == "first_name" and r.source_construction in ("active", "passive")]
    if not sub:
        return []
    out = []
    for li in range(len(sub[0].y)):
        rows = _within_rows(sub, li, codes=("role",))
        out.append(ols(rows)[1] if rows else float("nan"))
    return out


def predictor_correlations(transplants: Sequence[Transplant]) -> dict[str, float]:
    xs = [t.x for t in transplants]

    def corr(a: str, b: str) -> float:
        u = [x[a] for x in xs]
        v = [x[b] for x in xs]
        mu, mv = sum(u) / len(u), sum(v) / len(v)
        cov = sum((p - mu) * (q - mv) for p, q in zip(u, v))
        su = math.sqrt(sum((p - mu) ** 2 for p in u))
        sv = math.sqrt(sum((q - mv) ** 2 for q in v))
        return cov / (su * sv) if su and sv else float("nan")

    return {f"{a}_{b}": corr(a, b) for i, a in enumerate(CODES) for b in CODES[i + 1:]}


# ---------------------------------------------------------------- regression

def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small square system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular design matrix")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(rows: Sequence[tuple[Sequence[float], float]]) -> list[float]:
    """Least squares with intercept: rows of (features, y) -> [b0, b1, ...]."""
    p = len(rows[0][0]) + 1
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for feats, y in rows:
        z = [1.0, *feats]
        for i in range(p):
            xty[i] += z[i] * y
            for j in range(p):
                xtx[i][j] += z[i] * z[j]
    return _solve(xtx, xty)


@dataclass
class TransplantRecord:
    base_id: str
    family_id: str
    base_construction: str
    source_construction: str
    k: str
    j: str
    role_asked: str
    x: dict[str, int]
    ld_base: float  # logit(name_k) - logit(other base name), unpatched
    y: list[float] = field(default_factory=list)  # per layer: patched - unpatched


def _within_rows(recs, li, codes=CODES):
    """Rows for a within-(base sentence, base slot) regression: y and x are centred
    inside each group, so everything that is constant for a base slot (its own
    labels, whether it holds the answer, the size of the identity effect) drops out
    and only the source state varies."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in recs:
        if not math.isnan(r.y[li]):
            groups[(r.base_id, r.k)].append(r)
    rows = []
    for g in groups.values():
        if len(g) < 2:
            continue
        my = sum(r.y[li] for r in g) / len(g)
        mx = {c: sum(r.x[c] for r in g) / len(g) for c in codes}
        rows.extend(([r.x[c] - mx[c] for c in codes], r.y[li] - my) for r in g)
    return rows


def fit_profile(records: Sequence[TransplantRecord], n_boot: int = 200, seed: int = 0) -> dict:
    """Per layer: within-(base, slot) OLS coefficients for each code, with a
    family-bootstrap 95% interval. Also reports the mean identity effect (the
    average y, i.e. how much any foreign-name transplant moves the answer)."""
    n_layers = len(records[0].y)
    by_family: dict[str, list[TransplantRecord]] = defaultdict(list)
    for r in records:
        by_family[r.family_id].append(r)
    families = sorted(by_family)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = []
        for _f in families:
            sample.extend(by_family[rng.choice(families)])
        boots.append(sample)

    def coefs(recs, li):
        return ols(_within_rows(recs, li))

    out: dict[str, dict] = {c: {"coef": [], "ci95": []} for c in CODES}
    out["mean_identity_effect"] = []
    for li in range(n_layers):
        b = coefs(records, li)
        vals = [r.y[li] for r in records if not math.isnan(r.y[li])]
        out["mean_identity_effect"].append(sum(vals) / len(vals))
        bs = [coefs(s, li) for s in boots] if n_boot else []
        for ci, c in enumerate(CODES, start=1):
            out[c]["coef"].append(b[ci])
            if bs:
                vals = sorted(v[ci] for v in bs)
                out[c]["ci95"].append([vals[int(0.025 * n_boot)], vals[min(n_boot - 1, int(0.975 * n_boot))]])
    out["n_records"] = len(records)
    out["n_families"] = len(families)
    return out


# ---------------------------------------------------------------- the sweep (torch)

def _single_token(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"name {name!r} is not a single token")
    return ids[0]


def sweep(
    model,
    tokenizer,
    transplants: Sequence[Transplant],
    layers: Sequence[int] | None = None,
    batch_size: int = 32,
    require_base_correct: bool = True,
) -> tuple[list[TransplantRecord], dict[str, int]]:
    """Run every transplant at every layer. Returns records and skip counts."""
    import torch

    from relational_lab.instrumentation.patching import ResidualPatcher, encode_batch, token_index

    patcher = ResidualPatcher(model)
    layer_list = list(range(patcher.n_layers)) if layers is None else list(layers)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    skipped: dict[str, int] = defaultdict(int)
    records: list[TransplantRecord] = []

    for start in range(0, len(transplants), batch_size):
        batch = list(transplants[start : start + batch_size])
        b_ids, b_mask, b_len, b_off = encode_batch(tokenizer, [t.base.prompt for t in batch], pad)
        s_ids, s_mask, _, s_off = encode_batch(tokenizer, [t.source.prompt for t in batch], pad)
        src_store, _ = patcher.capture(s_ids, s_mask)
        with torch.no_grad():
            base_logits = model(input_ids=b_ids.to(patcher.device), attention_mask=b_mask.to(patcher.device)).logits

        rows, k_pos, j_pos, toks = [], [], [], []
        for r, t in enumerate(batch):
            last = b_len[r] - 1
            ans, dis = _single_token(tokenizer, t.base.answer), _single_token(tokenizer, t.base.distractor)
            if require_base_correct and not base_logits[r, last, ans] > base_logits[r, last, dis]:
                skipped["model_wrong_on_base"] += 1
                continue
            name_k = t.base.names[0].name if t.k == "first_name" else t.base.names[1].name
            other = t.base.agent if name_k == t.base.patient else t.base.patient
            nk, no = _single_token(tokenizer, name_k), _single_token(tokenizer, other)
            rows.append(r)
            k_pos.append(token_index(b_off[r], *t.base.span(t.k)))
            j_pos.append(token_index(s_off[r], *t.source.span(t.j)))
            toks.append((nk, no))
            records.append(TransplantRecord(
                base_id=t.base.item_id, family_id=t.base.family_id,
                base_construction=t.base.construction, source_construction=t.source.construction,
                k=t.k, j=t.j, role_asked=t.base.role_asked, x=t.x,
                ld_base=float(base_logits[r, last, nk] - base_logits[r, last, no]),
                y=[float("nan")] * len(layer_list),
            ))
        if not rows:
            continue
        recs = records[-len(rows):]
        row_t = torch.tensor(rows)
        sub_ids, sub_mask = b_ids[row_t], b_mask[row_t]
        for li, layer in enumerate(layer_list):
            store = src_store[layer]
            vecs = store[row_t.to(store.device), torch.tensor(j_pos, device=store.device)]
            logits = patcher.run_patched(sub_ids, sub_mask, layer, k_pos, vecs)
            for i, r in enumerate(rows):
                last = b_len[r] - 1
                nk, no = toks[i]
                recs[i].y[li] = float(logits[i, last, nk] - logits[i, last, no]) - recs[i].ld_base
    return records, dict(skipped)
