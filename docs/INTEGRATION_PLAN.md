[← Research overview](../README.md) · [Repository map](../papers/MAP.md)

# Integration plan: two paper tracks, stable scientific evidence

This plan records completed Paper II integration alongside open Paper I scientific and preservation work. Navigation updates do not move scientific files or change frozen protocols.

## Reference snapshot — 28 November 2026

| Reference | Head at inspection | Interpretation |
|:--|:--|:--|
| `main` | `e6c37eb2` | OTR code (stimuli merge `59d64a3` and PR #19), complete tracked results (PR #25), OTR-12 knockout (PR #26), Paper I branch preservation register (PR #23) |
| `experiments/exp102-clean-design` | `df82ab3a` | Post-PR #8 scratch experiments |
| [Scratch study branch inventory](../papers/learning-relations/EXPERIMENTS.md) | Pinned per-study SHAs | Current surface/diversity/collapse/mechanism/repair/origins studies, including ongoing local training |
| `order-to-role/stimuli` | `9c437928` | **Merged into `main`** with frozen history preserved (PR #19); retained for historical provenance |
| `order-to-role/results-dev` | `25c6914e` | Original result files imported into `main` (PR #25), except four Wikipedia-derived files; original branch retained for pinned history |

Branch heads can advance. Verify current refs, local uncommitted work, merge bases and existing releases immediately before each code or data PR.

## Phase A · Documentation only

**Status: complete (PR #11 and follow-ups).** This section records the documentation requirements fulfilled during the initial integration.

- Main README introduces the shared Model Linguistics concept and has three primary entrances: historical development, Paper I, and Paper II.
- Historical navigation includes a complete tracked ML-series inventory; later transition studies are marked without inventing an exact split cutoff. Paper I uses a provisional navigation title, not a manuscript rename.
- Each hub links to evidence, experiments and reproducibility, with branch-only paths labeled. Paper I covers the 15 studies from the 27 November handover plus position controls; preserve the manuscript and disclose archive gaps.
- No frozen scripts, raw results, model weights, protected items or existing research data modified.
- Run Markdown/link checks and review scientific wording before merging.

## Phase B · Integrate scratch-trained work

**Status: open.** Paper I source and report integrations remain subject to their separate scientific and archival gates.

- Create an integration branch from the **current** `main`, not from an old experiment tree.
- Audit the remaining EXP-109–116 and PHASE2 changes and later position, surface/diversity, recovery, mechanism, repair and origins branches against their own protocols, manifests and source SHAs. Do not merge scientific branches in PR #11.
- ROLE-LEARNING-ORIGINS-v0 is running locally. Never interfere with the active scratch worktree, training processes or checkpoints. Any scientific integration needs separate authorisation and an isolated environment.
- Import only the intended additions, preserving canonical frozen paths and bytes.
- Check tests, protocol integrity, seed/result completeness and the manuscript's existing claim boundaries.
- Avoid silently merging exploratory findings into the manuscript text.

## Phase C · Integrate OTR code and documentation

**Status: complete for Paper II.** `order-to-role/stimuli` was merged through `59d64a3` (PR #19); shared navigation now points to the canonical `main` paths.

- Create a separate integration branch from current `main`.
- Inventory code, tests, dependencies, notebooks, figures and rights-cleared review material from `order-to-role/stimuli`.
- Integrate OTR-specific implementations without replacing newer shared code from `main`.
- Reconcile CI and test dependencies in a dedicated, reviewable diff; validate standard offline tests.
- Update the remaining shared navigation from branch-only references to canonical `main` paths (this documentation cleanup).

## Phase D · Results stewardship

**Status: tracked raw outputs imported (PR #25).** Curated results, supplementary tables and figures, and a checksummed source-archive inventory are under `papers/order-to-role/`. The `results-dev` branch stays for historical provenance. Four Wikipedia-derived text files remain excluded from tracked Git pending licensing and archival decisions.

PR #12 brought an earlier snapshot to `main`; PR #25 imported the remaining 454 authors' output files. The name `results-dev` is historical: later commits also contain frozen test0 outputs. Publication, rights clearance, externally deposited archives and independent retrieval checks remain open; do not rewrite history or delete tracked results as part of this navigation PR.

- Retain `order-to-role/results-dev` as source provenance until raw outputs have been audited and released appropriately.
- Maintain the existing curated summaries, figures and manifests of exact source commits, checksums and analysis versions; complete outstanding publication-rights clearance before an external release.
- Prefer a versioned research-data release for large artifacts; never push model weights, protected test material or restricted content by accident.
- Historical results committed to `order-to-role/stimuli` are preserved in `main`'s merge history; preserve those commits as part of the full archive record.

## Documentation merge gates

Review all three entrances, verify major scratch studies are discoverable, distinguish ongoing/selected/exploratory/confirmatory/failed work, validate links and CI, and verify the PR is documentation-only. The diagnostic archive gap is disclosed; its dedicated source is still needed for full provenance.

## Merge gates

Every research PR should disclose its source SHA(s), file ownership, tests run, effect on frozen paths, evidence status, and any omitted outputs. Reject unrelated formatting churn, force-pushes to long-lived research branches, unreviewed changes to protocols, and silent reassignment of historical ML experiments between papers.

**Order:** documentation PR → scratch PR → OTR code PR → reviewed data/release workflow → README links and cross-study review.
