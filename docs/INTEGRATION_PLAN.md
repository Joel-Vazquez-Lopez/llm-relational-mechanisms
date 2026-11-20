[← Research overview](../README.md) · [Repository map](../papers/MAP.md)

# Integration plan: two paper tracks, stable scientific evidence

This is a **plan**, not a statement that experimental branches are already merged. The documentation PR should introduce navigation without moving scientific files or changing frozen protocol content.

## Reference snapshot — 21 November 2026

| Reference | Head at inspection | Interpretation |
|:--|:--|:--|
| `main` | `4423683a` | Reader guide merged November 18 |
| `experiments/exp102-clean-design` | `df82ab3a` | Post-PR #8 scratch experiments |
| `order-to-role/stimuli` | `16e13d47` | OTR code, new constructions and tests |
| `order-to-role/results-dev` | Verify live SHA before integration | OTR development result archive |

Branch heads can advance. Verify current refs, local uncommitted work, merge bases and existing releases immediately before each code or data PR.

## Phase A · Documentation only

- Main README points to both study hubs and clearly identifies the separate research scopes.
- Each hub links to evidence, experiments and reproducibility, with branch-only paths labeled.
- No frozen scripts, raw results, model weights, protected items or existing research data modified.
- Run Markdown/link checks and review scientific wording before merging.

## Phase B · Integrate scratch-trained work

- Create an integration branch from the **current** `main`, not from an old experiment tree.
- Audit the remaining EXP-109–116 and PHASE2 changes against their parent protocols and manifests.
- Import only the intended additions, preserving canonical frozen paths and bytes.
- Check tests, protocol integrity, seed/result completeness and the manuscript's existing claim boundaries.
- Avoid silently merging exploratory findings into the manuscript text.

## Phase C · Integrate OTR code and documentation

- Create a separate integration branch from current `main`.
- Inventory code, tests, dependencies, notebooks, figures and rights-cleared review material from `order-to-role/stimuli`.
- Integrate OTR-specific implementations without replacing newer shared code from `main`.
- Reconcile CI and test dependencies in a dedicated, reviewable diff; validate standard offline tests.
- Update paper navigation links from the code branch to canonical paths **only after integration**.

## Phase D · Results stewardship

- Retain `order-to-role/results-dev` as source provenance until raw outputs have been audited and released appropriately.
- Generate curated rights-cleared summaries, figures and a manifest of exact source commits, checksums and analysis versions.
- Prefer a versioned research-data release for large artifacts; never push model weights, protected test material or restricted content by accident.
- Confirm whether any results were committed to `order-to-role/stimuli` before declaring the archive complete.

## Merge gates

Every research PR should disclose its source SHA(s), file ownership, tests run, effect on frozen paths, evidence status, and any omitted outputs. Reject unrelated formatting churn, force-pushes to long-lived research branches, unreviewed changes to protocols, and silent reassignment of historical ML experiments between papers.

**Order:** documentation PR → scratch PR → OTR code PR → reviewed data/release workflow → README links and cross-study review.
