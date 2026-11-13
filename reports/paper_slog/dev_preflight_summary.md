# SLOG public development preflight: archived summary

**User-run preflight:** 12 November 2026, from the official `data/cogs_LF/dev.tsv`; `paper_slog_preflight.py` on the paper branch, corrected to accept SLOG development structural-category labels.

**Dataset SHA256:** `177128c3f64be56732a102db119139bbec33a771c485a525ea0f9f4e45926a5c`

## Reported totals

| Metric | Value |
|---|---:|
| Sentences | 4,046 |
| Events with an agent or theme role | 5,420 |
| Sentences with at least two annotated predicates | 1,135 |
| Valid agent–theme events, any sentence | 3,190 |
| Eligible role events in multi-event sentences | 1,345 |
| Eligible multi-event sentences | 812 |
| Multi-event sentences with two eligible events | 462 |
| Multi-event sentences with four distinct participant surface heads | 53 |
| Eligible multi-event sentences with passive surface cue | 175 |
| Eligible multi-event sentences with that/who/which | 812 |

## Eligibility by public development category

| Category | Total dev rows | Eligible multi-event sentences |
|---|---:|---:|
| `in_distribution` | 3,000 | 149 |
| `object_modifying_RC` | 635 | 613 |
| `wh_Q_simple_trans` | 292 | 0 |
| `pp_4` | 56 | 1 |
| `center_embed_2` | 29 | 29 |
| `cp_4` | 20 | 6 |
| `center_embed_4` | 14 | 14 |

Disjoint four-participant examples: `center_embed_2` 29; `center_embed_4` 14; `object_modifying_RC` 10.

**Evidence status:** counts copied from a completed user-supplied local preflight log, not reconstructed from model predictions. The complete JSON with script SHA and breakdown remains at the user’s local `reports/paper_slog/dev_preflight.json`. These counts were also cross-checked against the official public development TSV. No SLOG protected generalization archive or model output has been examined.

**Limitations:** single-token surface heads, unique verb surfaces, `theme` rather than `patient`, dominated by relative clauses, and only 53 four-distinct cases. Eligibility does not imply correct QA projection for every sample; label audit is mandatory before model scoring.
