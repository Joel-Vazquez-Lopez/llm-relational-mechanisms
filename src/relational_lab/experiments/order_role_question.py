"""OTR-04: question-side patching. Where is the question linked to the roles?

Base and source share the SAME sentence and differ only in the question, so
every position before the question is identical (patching there must have
exactly zero effect: a built-in check). We patch the question's first word
(``q_first``: "Who" or "By"), the last sentence token (``period``, the
control) and every position of the shared tail, aligned from the end
(``?``, ``Answer:``, and in chat mode the template's turn markers), at every
layer.

Two modes:

flip
    Source asks for the OTHER role in the same voice ("Who praised someone?"
    <-> "Who did someone praise?"; "By whom was someone praised?" <-> "Who was
    praised by someone?"). Only pairs the model answers correctly in both
    directions. Effect 1 = the base now answers the source's question: shows
    where and when the question's role demand is carried.

repair
    Base is a question the model gets WRONG; the source asks for the SAME role
    in the other voice and is answered correctly (e.g. "Who did someone
    praise?" <- "Who was praised by someone?"). Both have the same correct
    answer. Effect 1 = the patched base now answers like the source; a repair
    rate > 0 at some (position, layer) means the sentence-side role code was
    intact and the failure was in reading the question.

For both, effect = (LD_base - LD_patched) / (LD_base - LD_source) with LD the
base answer's logit minus the base distractor's, and "flip" = the patched LD
has the sign of the source LD.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from relational_lab.experiments.order_role_stimuli import Item, build_item

MODES: tuple[str, ...] = ("flip", "repair")
CONTROL_LABEL = "period"  # last sentence token: identical prefix -> zero effect
WH_LABEL = "q_first"  # first word of the question ("Who" / "By")
MAX_TAIL = 16  # cap on shared-tail positions patched (chat templates add ~5-8)


def partner(item: Item, mode: str) -> Item:
    """The source prompt for ``item`` in ``mode`` (same sentence, other question)."""
    if mode == "flip":
        role, voice = ("patient" if item.role_asked == "agent" else "agent"), item.question_voice
    elif mode == "repair":
        role, voice = item.role_asked, ("passive" if item.question_voice == "active" else "active")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return build_item(item.agent, item.patient, item.lemma, item.verb, item.construction, role, voice,
                      family_id=item.family_id, split=item.split, test_set=item.test_set)


def shared_tail(a: Sequence[int], b: Sequence[int], cap: int = MAX_TAIL) -> int:
    """Length of the longest common suffix of two id sequences (at most ``cap``)."""
    n = 0
    while n < min(len(a), len(b), cap) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def tail_label(k: int) -> str:
    return f"end-{k}"  # end-1 is the final position (the one the answer is read from)


@dataclass
class QuestionRecord:
    item_id: str
    family_id: str
    construction: str
    role_asked: str
    question_voice: str
    mode: str
    source_id: str
    ld_base: float
    ld_source: float
    tail_tokens: list[str] = field(default_factory=list)  # base tokens for end-1, end-2, ...
    ld_patched: dict[str, list[float]] = field(default_factory=dict)


def effect(ld_base: float, ld_source: float, ld_patched: float) -> float:
    d = ld_base - ld_source
    return float("nan") if d == 0 else (ld_base - ld_patched) / d


def _wants(mode: str, ld_b: float, ld_s: float) -> bool:
    if mode == "flip":  # source answer = base distractor; both correct
        return ld_b > 0 and ld_s < 0
    return ld_b < 0 and ld_s > 0  # repair: base wrong, source right, same answer


def sweep(
    model,
    tokenizer,
    items: Sequence[Item],
    mode: str = "flip",
    layers: Sequence[int] | None = None,
    batch_size: int = 16,
    require_selection: bool = True,
    chat_template: bool = False,
    max_tail: int = MAX_TAIL,
) -> tuple[list[QuestionRecord], dict[str, int]]:
    """Patch q_first / period / every shared-tail position at every layer."""
    import torch

    from relational_lab.instrumentation.patching import ResidualPatcher, answer_token, encode_batch, token_index

    patcher = ResidualPatcher(model)
    layer_list = list(range(patcher.n_layers)) if layers is None else list(layers)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    space = not chat_template
    skipped: dict[str, int] = defaultdict(int)
    records: list[QuestionRecord] = []

    for start in range(0, len(items), batch_size):
        bases = list(items[start:start + batch_size])
        sources = [partner(b, mode) for b in bases]
        b_ids, b_mask, b_len, b_off = encode_batch(tokenizer, [b.prompt for b in bases], pad, chat_template)
        s_ids, s_mask, s_len, s_off = encode_batch(tokenizer, [s.prompt for s in sources], pad, chat_template)
        src_store, src_logits = patcher.capture(s_ids, s_mask)
        with torch.no_grad():
            base_logits = model(input_ids=b_ids.to(patcher.device), attention_mask=b_mask.to(patcher.device)).logits

        # per kept row: {label: (base_pos, source_pos)}
        kept: list[tuple[int, QuestionRecord, dict[str, tuple[int, int]], int, int]] = []
        for r, (b, s) in enumerate(zip(bases, sources)):
            bi = b_ids[r, : b_len[r]].tolist()
            si = s_ids[r, : s_len[r]].tolist()
            a_id, d_id = answer_token(tokenizer, b.answer, space), answer_token(tokenizer, b.distractor, space)
            lb, ls = b_len[r] - 1, s_len[r] - 1
            ld_b = float(base_logits[r, lb, a_id] - base_logits[r, lb, d_id])
            ld_s = float(src_logits[r, ls, a_id] - src_logits[r, ls, d_id])
            if require_selection and not _wants(mode, ld_b, ld_s):
                skipped["not_selected_" + mode] += 1
                continue
            pos: dict[str, tuple[int, int]] = {}
            per, pes = b.span(CONTROL_LABEL), s.span(CONTROL_LABEL)
            pos[CONTROL_LABEL] = (token_index(b_off[r], *per), token_index(s_off[r], *pes))
            qb, qs = b.prompt.index(b.question), s.prompt.index(s.question)
            wb = b.question.split()[0]
            ws = s.question.split()[0]
            pos[WH_LABEL] = (token_index(b_off[r], qb, qb + len(wb)), token_index(s_off[r], qs, qs + len(ws)))
            n_tail = shared_tail(bi, si, max_tail)
            for k in range(1, n_tail + 1):
                pos[tail_label(k)] = (b_len[r] - k, s_len[r] - k)
            rec = QuestionRecord(
                b.item_id, b.family_id, b.construction, b.role_asked, b.question_voice, mode, s.item_id,
                ld_b, ld_s, tail_tokens=[tokenizer.decode([bi[-k]]) for k in range(1, n_tail + 1)],
            )
            kept.append((r, rec, pos, a_id, d_id))
            records.append(rec)
        if not kept:
            continue

        labels = sorted({lab for _, _, pos, _, _ in kept for lab in pos})
        for lab in labels:
            rows = [(r, rec, pos[lab], a, d) for r, rec, pos, a, d in kept if lab in pos]
            row_t = torch.tensor([r for r, *_ in rows])
            sub_ids, sub_mask = b_ids[row_t], b_mask[row_t]
            b_pos = [p[0] for _, _, p, _, _ in rows]
            s_pos = torch.tensor([p[1] for _, _, p, _, _ in rows])
            for li, layer in enumerate(layer_list):
                st = src_store[layer]
                vecs = st[row_t.to(st.device), s_pos.to(st.device)]
                logits = patcher.run_patched(sub_ids, sub_mask, layer, b_pos, vecs)
                for j, (r, rec, _, a, d) in enumerate(rows):
                    last = b_len[r] - 1
                    rec.ld_patched.setdefault(lab, [float("nan")] * len(layer_list))
                    rec.ld_patched[lab][li] = float(logits[j, last, a] - logits[j, last, d])
    return records, dict(skipped)


def _crossed(ld_patched: float, ld_source: float) -> bool:
    """The patched LD strictly has the source LD's sign (ties do not count)."""
    return ld_patched < 0 if ld_source < 0 else ld_patched > 0


def aggregate(records: Sequence[QuestionRecord]) -> dict:
    """Per group (mode/role/voice, plus mode/ALL) x label x layer: ratio of means
    sum(LD_base - LD_patched) / sum(LD_base - LD_source), median per-pair effect,
    and flip rate (patched LD takes the source LD's sign)."""
    cells: dict[tuple[str, str], list[QuestionRecord]] = defaultdict(list)
    for rec in records:
        for g in (f"{rec.mode}/{rec.role_asked}/{rec.question_voice}", f"{rec.mode}/ALL"):
            for lab in rec.ld_patched:
                cells[(g, lab)].append(rec)
    out: dict[str, dict] = {}
    for (g, lab), recs in sorted(cells.items()):
        n_layers = len(recs[0].ld_patched[lab])
        den = sum(r.ld_base - r.ld_source for r in recs)
        rom, med, flip = [], [], []
        for li in range(n_layers):
            num = sum(r.ld_base - r.ld_patched[lab][li] for r in recs)
            rom.append(num / den if den else float("nan"))
            vals = sorted(effect(r.ld_base, r.ld_source, r.ld_patched[lab][li]) for r in recs)
            vals = [v for v in vals if not math.isnan(v)]
            med.append(vals[len(vals) // 2] if vals else float("nan"))
            flip.append(sum(_crossed(r.ld_patched[lab][li], r.ld_source) for r in recs) / len(recs))
        out.setdefault(g, {})[lab] = {
            "n": len(recs), "n_families": len({r.family_id for r in recs}),
            "ratio_of_means": rom, "median_effect": med, "flip_rate": flip,
        }
    return out


def label_order(labels: Sequence[str]) -> list[str]:
    """Display order: control, q_first, then the tail from earliest to final position."""
    tails = sorted((lab for lab in labels if lab.startswith("end-")), key=lambda s: -int(s[4:]))
    return [lab for lab in (CONTROL_LABEL, WH_LABEL) if lab in labels] + tails
