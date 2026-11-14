"""E2 (OTR-02): where is the answer decided? Position x layer patching.

For each base prompt, the source is its role-swapped twin: same verb,
construction and question, agent and patient exchanged, so the correct answer
flips while the token sequence keeps the same length and layout. Patching the
residual stream at one (position, layer) from source into base measures how
much of the answer is carried there:

    LD(x)  = logit(base answer) - logit(base distractor) at the final position
    effect = (LD_base - LD_patched) / (LD_base - LD_source)

0 = nothing moved, 1 = the patched run behaves like the source. ``flipped``
records whether the patched run now prefers the source's answer.

Patching at a *name* position also moves that token's identity, so early-layer
name effects are partly trivial; the informative pattern is how the effect
migrates from name positions to later positions across layers.

The sweep needs torch (lazy import); aggregation is dependency-free.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from relational_lab.experiments.order_role_stimuli import POSITION_LABELS, Item, role_swapped


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


@dataclass
class PairRecord:
    item_id: str
    family_id: str
    construction: str
    role_asked: str
    ld_base: float
    ld_source: float
    # effects[label][layer] and flips[label][layer]; labels absent in this construction are missing
    effects: dict[str, list[float]] = field(default_factory=dict)
    flips: dict[str, list[bool]] = field(default_factory=dict)
    ld_patched: dict[str, list[float]] = field(default_factory=dict)


def normalized_effect(ld_base: float, ld_source: float, ld_patched: float) -> float:
    denom = ld_base - ld_source
    if denom == 0:
        return float("nan")
    return (ld_base - ld_patched) / denom


def sweep(
    model,
    tokenizer,
    items: Sequence[Item],
    labels: Sequence[str] = POSITION_LABELS,
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    require_correct: bool = True,
    chat_template: bool = False,
) -> tuple[list[PairRecord], dict[str, int]]:
    """Patch every (label, layer) for every base item. Returns records and skip counts.

    ``chat_template``: wrap each prompt as one user turn (as ``HFScorer`` does);
    item spans are mapped into the templated sequence, and the answer token is
    the name without a leading space."""
    import torch

    from relational_lab.instrumentation.patching import ResidualPatcher, answer_token, encode_batch, token_index

    patcher = ResidualPatcher(model)
    layer_list = list(range(patcher.n_layers)) if layers is None else list(layers)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    skipped: dict[str, int] = defaultdict(int)
    records: list[PairRecord] = []

    for start in range(0, len(items), batch_size):
        bases = list(items[start : start + batch_size])
        sources = [role_swapped(b) for b in bases]
        b_ids, b_mask, b_len, b_off = encode_batch(tokenizer, [b.prompt for b in bases], pad, chat_template)
        s_ids, s_mask, s_len, s_off = encode_batch(tokenizer, [s.prompt for s in sources], pad, chat_template)
        src_store, src_logits = patcher.capture(s_ids, s_mask)
        with torch.no_grad():
            base_logits = model(input_ids=b_ids.to(patcher.device), attention_mask=b_mask.to(patcher.device)).logits

        keep, pos_by_label, ans, dis = [], defaultdict(dict), [], []
        for r, (b, s) in enumerate(zip(bases, sources)):
            if b_len[r] != s_len[r]:
                skipped["length_mismatch"] += 1
                continue
            positions = {}
            ok = True
            for lab in labels:
                bs, ss = b.span(lab), s.span(lab)
                if bs is None:
                    continue
                bi, si = token_index(b_off[r], *bs), token_index(s_off[r], *ss)
                if bi != si:
                    ok = False
                    break
                positions[lab] = bi
            if not ok:
                skipped["position_mismatch"] += 1
                continue
            space = not chat_template
            a_id, d_id = answer_token(tokenizer, b.answer, space), answer_token(tokenizer, b.distractor, space)
            last = b_len[r] - 1
            ld_b = float(base_logits[r, last, a_id] - base_logits[r, last, d_id])
            ld_s = float(src_logits[r, last, a_id] - src_logits[r, last, d_id])
            if require_correct and not (ld_b > 0 and ld_s < 0):
                skipped["model_wrong_on_base_or_source"] += 1
                continue
            keep.append(r)
            ans.append(a_id)
            dis.append(d_id)
            for lab, idx in positions.items():
                pos_by_label[lab][r] = idx
            records.append(PairRecord(b.item_id, b.family_id, b.construction, b.role_asked, ld_b, ld_s))

        if not keep:
            continue
        recs = records[-len(keep):]
        for lab in labels:
            rows = [r for r in keep if r in pos_by_label[lab]]
            if not rows:
                continue
            row_t = torch.tensor(rows)
            sub_ids, sub_mask = b_ids[row_t], b_mask[row_t]
            positions = [pos_by_label[lab][r] for r in rows]
            for layer in layer_list:
                vecs = src_store[layer][row_t.to(src_store[layer].device), torch.tensor(positions, device=src_store[layer].device)]
                logits = patcher.run_patched(sub_ids, sub_mask, layer, positions, vecs)
                for j, r in enumerate(rows):
                    k = keep.index(r)
                    rec = recs[k]
                    last = b_len[r] - 1
                    ld_p = float(logits[j, last, ans[k]] - logits[j, last, dis[k]])
                    rec.effects.setdefault(lab, [float("nan")] * len(layer_list))
                    rec.flips.setdefault(lab, [False] * len(layer_list))
                    rec.ld_patched.setdefault(lab, [float("nan")] * len(layer_list))
                    li = layer_list.index(layer)
                    rec.effects[lab][li] = normalized_effect(rec.ld_base, rec.ld_source, ld_p)
                    rec.flips[lab][li] = ld_p < 0
                    rec.ld_patched[lab][li] = ld_p
    return records, dict(skipped)


def aggregate(records: Sequence[PairRecord]) -> dict:
    """Per construction x label x layer (and pooled): mean and median of the
    per-pair normalized effect, the ratio of means
    sum(LD_base - LD_patched) / sum(LD_base - LD_source), which is robust to
    pairs with tiny denominators, and the flip rate."""
    cells: dict[tuple[str, str], list[PairRecord]] = defaultdict(list)
    for rec in records:
        for lab in rec.effects:
            cells[(rec.construction, lab)].append(rec)
            cells[("ALL", lab)].append(rec)
    out: dict[str, dict] = {}
    for (c, lab), recs in sorted(cells.items()):
        n_layers = len(recs[0].effects[lab])
        mean_effect, median_effect, ratio_of_means, flip_rate = [], [], [], []
        denom = sum(r.ld_base - r.ld_source for r in recs)
        for li in range(n_layers):
            vals = [r.effects[lab][li] for r in recs if not math.isnan(r.effects[lab][li])]
            mean_effect.append(sum(vals) / len(vals) if vals else float("nan"))
            median_effect.append(_median(vals))
            num = sum(r.ld_base - r.ld_patched[lab][li] for r in recs)
            ratio_of_means.append(num / denom if denom else float("nan"))
            flip_rate.append(sum(r.flips[lab][li] for r in recs) / len(recs))
        out.setdefault(c, {})[lab] = {
            "n": len(recs),
            "n_families": len({r.family_id for r in recs}),
            "mean_effect": mean_effect,
            "median_effect": median_effect,
            "ratio_of_means": ratio_of_means,
            "flip_rate": flip_rate,
        }
    return out
