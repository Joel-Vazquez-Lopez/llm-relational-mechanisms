[← Research overview](../README.md) · [Repository map](../papers/MAP.md)

# Integration plan: two paper tracks, stable scientific evidence

This is a **plan**, not a statement that experimental branches are already merged. The documentation PR should introduce navigation without moving scientific files or changing frozen protocol content.

## Reference snapshot — 27 November 2026

| Reference | Head at inspection | Interpretation |
|:--|:--|:--|
| `main` | `5e8ab46c` | PR #12 merged the November 21 OTR results snapshot |
| `experiments/exp102-clean-design` | `df82ab3a` | Post-PR #8 scratch experiments |
| [Scratch study branch inventory](../papers/learning-relations/EXPERIMENTS.md) | Pinned per-study SHAs | Current surface/diversity/collapse/mechanism/repair/origins studies, including ongoing local training |
| `order-to-role/stimuli` | `9c437928` | OTR-01–15, English/XL freezes, large-model extension |
| `order-to-role/results-dev` | `25c6914e` | OTR development, test0 and 70B exploratory result archive |

Branch heads can advance. Verify current refs, local uncommitted work, merge bases and existing releases immediately before each code or data PR.

## Phase A · Documentation only

- Main README introduces the shared Model Linguistics concept and has three primary entrances: historical development, Paper I, and Paper II.
- Historical navigation includes a complete tracked ML-series inventory; later transition studies are marked without inventing an exact split cutoff. Paper I uses a provisional navigation title, not a manuscript rename.
- Each hub links to evidence, experiments and reproducibility, with branch-only paths labeled. Paper I covers the 15 studies from the 27 November handover plus position controls; preserve the manuscript and disclose archive gaps.
- No frozen scripts, raw results, model weights, protected items or existing research data modified.
- Run Markdown/link checks and review scientific wording before merging.

## Phase B · Integrate scratch-trained work

- Create an integration branch from the **current** `main`, not from an old experiment tree.
- Audit the remaining EXP-109–116 and PHASE2 changes and later position, surface/diversity, recovery, mechanism, repair and origins branches against their own protocols, manifests and source SHAs. Do not merge scientific branches in PR #11.
- ROLE-LEARNING-ORIGINS-v0 is running locally. Never interfere with the active scratch worktree, training processes or checkpoints. Any scientific integration needs separate authorisation and an isolated environment.
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

PR #12 already imported an earlier results snapshot into `main`. Audit that import for size, rights and provenance; do not rewrite history or delete tracked results as part of this navigation PR. The name `results-dev` is historical: later commits contain held-out test0 outputs as well.

- Retain `order-to-role/results-dev` as source provenance until raw outputs have been audited and released appropriately.
- Generate curated rights-cleared summaries, figures and a manifest of exact source commits, checksums and analysis versions.
- Prefer a versioned research-data release for large artifacts; never push model weights, protected test material or restricted content by accident.
- Confirm whether any results were committed to `order-to-role/stimuli` before declaring the archive complete.

## Documentation merge gates

Review all three entrances, verify major scratch studies are discoverable, distinguish ongoing/selected/exploratory/confirmatory/failed work, validate links and CI, and verify the PR is documentation-only. The diagnostic archive gap is disclosed; its dedicated source is still needed for full provenance.

## Merge gates

Every research PR should disclose its source SHA(s), file ownership, tests run, effect on frozen paths, evidence status, and any omitted outputs. Reject unrelated formatting churn, force-pushes to long-lived research branches, unreviewed changes to protocols, and silent reassignment of historical ML experiments between papers.

**Order:** documentation PR → scratch PR → OTR code PR → reviewed data/release workflow → README links and cross-study review.
