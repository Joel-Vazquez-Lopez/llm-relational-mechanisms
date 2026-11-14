# EXP-105 — prospective PA/AP update-schedule audit v3

**Status: UNFROZEN design-only candidate. No training, checkpoints, reserved examples or protected evaluation.**

Fixed candidate schedule: 3200 optimizer updates, 8 fresh paired seeds, 64 labels/update.
Each update: 7 shared AA + 7 shared PA + 2 matched-dose PA/AP sources; all four questions per source.
Each arm receives 9 passive-event occurrences per update; the dose changes passive *event position*.

| Per-update invariant | Status |
|---|---|
| Matched shared-base source IDs | PASS |
| Matched paired dose semantic cores and four gold labels | PASS |
| 16 sources / 64 labels / 16 of each query | PASS |
| 9 passive-event occurrences per arm | PASS |
| Aggregate token multiset in each individual update | PASS |
| Source input-length histogram in each individual update | NOT GUARANTEED (documented) |
| Source input-length histogram per complete 45-update window | PASS |
| Nine modifier cells × five frame types and layout parity balanced over each full 45-update block | PASS |

| Seed | SHA256 prefix | Updates audited | Token matches | Length matches (per update) |
|---:|---|---:|---:|---:|
| 10701 | `aff779e1cc7bde8f` | 3200 | 3200/3200 | 2489/3200 |
| 10703 | `5c8b13a01b528b77` | 3200 | 3200/3200 | 2490/3200 |
| 10709 | `73ef9e37d82752a7` | 3200 | 3200/3200 | 2488/3200 |
| 10713 | `f9719ae080a003bd` | 3200 | 3200/3200 | 2488/3200 |
| 10721 | `f26225f207a64e5b` | 3200 | 3200/3200 | 2488/3200 |
| 10727 | `20ad814850986578` | 3200 | 3200/3200 | 2490/3200 |
| 10729 | `25cf6009c1961742` | 3200 | 3200/3200 | 2488/3200 |
| 10733 | `7366a790cb2c58d3` | 3200 | 3200/3200 | 2490/3200 |

**PASS:** 25,600 proposed paired updates audited. No model trained.

## Limits before training
- PA and AP differ in the event position of the passive construction; this is the intervention.
- Individual-update source length histograms can differ for L/I and I/L modifier doses; exact length matching is only over full 45-update windows.
- Absolute token positions and syntactic contexts are not identical; same token/length marginals do not equate positional frequencies.
- This is a proposed quartet-only scheduling regime; no quartet-vs-distributed contrast is included.
- Train/development grammar belongs to the existing synthetic family; independent replication remains needed.
- The complete candidate training protocol (model, loss, optimizer, gate, analysis) is not yet frozen.
- No learning or transfer result is claimed by a schedule audit.

**Next:** peer-review the full pilot protocol and code for label leakage, source ordering, and optimizer setup before any training.
