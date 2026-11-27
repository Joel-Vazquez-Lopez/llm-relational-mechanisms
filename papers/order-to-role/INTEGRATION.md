[← Order-to-Role](README.md) · [Reproduce](REPRODUCE.md) · [Shared integration plan](../../docs/INTEGRATION_PLAN.md)

# Integration of Paper II into `main`

An audit of every branch that holds Order-to-Role material, against `main` at `b170ac8` (after PR #11, and after Paper I's PRs #13 and #16, which touch only `papers/learning-relations/`). It also sets out the plan to develop future work from a unified `main`. This page implements **Phase C** (OTR code) and the Paper II part of **Phase D** (results stewardship) of the [shared integration plan](../../docs/INTEGRATION_PLAN.md). Paper I and the shared entrance are out of scope.

## 1. Branch audit (27 November 2026)

| Branch | Head | Ahead of `main` | Paper II content not on `main` | Treatment |
|:--|:--|--:|:--|:--|
| `order-to-role/stimuli` | `9c437928` | 72 commits | 131 files absent, 8 files newer (see below) | **Merge** (PR B) |
| `order-to-role/results-dev` | `25c6914e` | 13 commits | 458 result files, 206 MB (22–27 Nov: WIKI v0.2, OTR-13/14, test0 English and es/gl, 70B dev) | **Keep as provenance branch**; curated summaries on `main` (PR A); full archive as a data release |
| `docs/two-paper-research-navigation` | `90b92b9` | 7 commits | none (source of the squash-merged PR #11) | Kept. The coordinating review decides when historical branches are retired |
| Every other branch | – | – | no OTR files | Not Paper II |

**`order-to-role/stimuli` in detail.** Merge base with `main`: `67ae853c`. Of the 232 files the branch changed since then:

- **93 are byte-identical on `main`** (result files rescued by PR #9 and PR #12).
- **131 are absent from `main`**: 49 scripts (OTR-04 to OTR-15, scoring, generation, UD and WIKI extraction, 23 Slurm jobs), 19 source modules (generators, instrumentation helpers), 28 test modules, 42 documentation files (freezes, freeze input checksums, review package, state snapshots, 70B plan), and the 16 Nov handoff note.
- **8 differ**, and in every case `main` still has the branch-point version, so there is no conflict: `otr01`/`otr02` scripts, `order_role_stimuli.py`, `order_role_patching.py`, `instrumentation/hf_scoring.py`, `instrumentation/patching.py`, `setup_pelle.sh`, `test_order_role_stimuli.py`.

**`order-to-role/results-dev` in detail.** It is based on the commit that PR #12 merged (`c3e07b3`), plus 13 commits of new outputs. The largest files are test0 OTR-07 and OTR-03 outputs (4–6 MB each). The canonical test0 attempt logs (`test0_attempts.md`, `test0_attempts_pelle.md`) are here, not on the code branch.

## 2. The two pull requests

### PR A · Paper II research section (documentation and curated results)

Only files under `papers/order-to-role/` change:

- The section pages: README, Results, Methods, Layers, Answers and interventions, Generality, Evidence ledger, Development, Experiment map, Manuscript map, Reproduce, this page.
- `analysis/build_results.py`, which regenerates every number from pinned result files.
- `results/`: five figures (SVG), development tables (CSV/JSON), frozen-scorer output for test0 (JSON) and `MANIFEST.json` with input checksums. Total about 0.5 MB.

No scripts, protocols or result files elsewhere are touched. PR A links to the pinned code snapshot, so it does not depend on PR B.

### PR B · OTR code integration (`paper2/integrate-otr-code`)

A **true merge** (not a squash) of `order-to-role/stimuli`, so that the frozen-protocol commits keep their hashes in `main`'s history. FREEZE.md and FREEZE_XL.md cite those commits (`f6ea0e1`, `3811d7c`, `c40a1f7`) as the record of what was frozen when.

1. Merge commit: 131 files added, 8 updated, no conflicts, no result files. The frozen commits `f6ea0e1`, `3811d7c` and `c40a1f7` are ancestors of the merge, and every file the stimuli branch added or changed is byte-identical to its version there, apart from the approved move below.
2. Move `ORDER_TO_ROLE_HANDOFF.md` from the repository root to `docs/order_to_role/state_2026-11-16/HANDOFF.md`, so the shared root is unchanged.
3. Shared instrumentation: scoring and patching build chat prompts through one function, so a system prompt or assistant prefix is handled identically in both. Token ids are byte-identical to the previous code for every setting used by a completed run. This adds `tests/test_scoring_patching_consistency.py`.
4. CI in two jobs: *lightweight* (pytest and NumPy, Python 3.10 and 3.12) and *scientific CPU validation* (CPU PyTorch, Transformers, Tokenizers and Accelerate on tiny random models, with the hub offline). Passed, skipped and failed counts are reported separately.

**Tests.** Lightweight: 147 passed, 61 skipped, 0 failed on Python 3.10 and 3.12. Scientific CPU: 214 passed, 3 skipped (spaCy parser, not installed by design), 0 failed. `main`'s own test files also pass against the merged code, except one assertion on the passive agent question's wording. That change is deliberate and versioned (generator v1.3.0 replaced "Who was someone V-ed by?" with "By whom was someone V-ed?" as the default passive agent question), and the merged test file was updated with it.

### PR C · Scientific supplementary materials (`paper2/supplementary`, stacked on PR A)

Only files under `papers/order-to-role/supplementary/` change (plus one navigation row in the Paper II README). It contains seven pages, nine figures, fifteen tables and `analysis/build_supplement.py`, which reads the same pinned inputs (133 result files) and the frozen scorer's test0 summaries from PR A. About 1.3 MB, no raw outputs.

### Shared-code dependencies for the coordinating review

| Item | Why it matters | Assessment |
|:--|:--|:--|
| `.github/workflows/ci.yml` | 15 merged OTR test modules use `pytest` and one needs `numpy`; the current dependency-free `unittest` step cannot import them | Approved by the coordinating review (pytest and NumPy; lightweight and scientific jobs kept separate) |
| `src/relational_lab/instrumentation/hf_scoring.py`, `patching.py` (+ new `chat.py`) | Shared package | Approved; no conflicting Paper I changes. Additive, backward-compatible defaults; on `main` only OTR code and their own tests import them; Paper I is unaffected |
| `ORDER_TO_ROLE_HANDOFF.md` at the root | Would change the shared entrance | Moved into `docs/order_to_role/` |
| `STUDIES.md` (shared guide) | It says test0 scoring "is not asserted" and calls FREEZE.md §6 a dry run only | Not edited by Paper II. Suggested replacement sentence: *"English test0 was scored with the frozen scorer: 37 of 38 checks pass (phi-4 C5 fails a control); BX 15/15; Spanish 29/30; Galician 30/30 (evidence ledger: `papers/order-to-role/EVIDENCE.md`)."* |
| `papers/MAP.md`, `docs/INTEGRATION_PLAN.md` | Mention OTR code as branch-only | Update after PR B merges (coordinating chat) |

## 3. Results stewardship

- **Do not merge `order-to-role/results-dev` into `main`.** It adds 206 MB of raw outputs on top of the 21 November snapshot that PR #12 already imported.
- **`main` carries the curated layer:** `papers/order-to-role/results/`, regenerated by `build_results.py`, plus the manifest of source commits and input checksums.
- **Full archive.** Publish `results/` at `25c6914e` as a versioned data release (a GitHub release asset or Zenodo) with a checksum manifest, and link it from [Reproduce](REPRODUCE.md). Until then, the branch is the archive. Do not rewrite or delete it.
- **No model weights** are in either branch. The new results-branch files are result JSON (200), Slurm logs (165), PNG plots (87), two Markdown attempt logs, and the WIKI-ROLES item files (`wiki_roles_en.jsonl`, `wiki_roles_v2_en.jsonl`, 1.7 and 3.2 MB) with their check sheets. The WIKI-ROLES files contain Wikipedia sentences, which are CC BY-SA. A data release must carry that licence and attribution.

## 4. Working from `main` afterwards

- **One branch per study, from current `main`** (for example `otr/big-models-freeze`, `otr/question-battery`), merged back with a pull request that states the source SHA, tests run and evidence status.
- **Raw outputs stay out of git.** Each study adds its derived tables to `papers/order-to-role/results/` through `build_results.py`, and its raw outputs to the data release.
- **Frozen files never change.** That covers FREEZE.md, FREEZE_XL.md, the freeze input checksums and the scorer's frozen entries. A new freeze is a new addendum file committed before its test run.
- **Next freezes due:** the 70B addendum, and any test1 use (which requires its own pre-registration).

## 5. Status

| Step | Status |
|:--|:--|
| PR A (`paper2/research-section`) | Ready for review, rebased on `b170ac8` |
| PR B (`paper2/integrate-otr-code`) | Ready for review on `b170ac8`. CI split and instrumentation check done as the coordinating review asked |
| PR C (`paper2/supplementary`) | Ready for review, stacked on PR A |
| Shared navigation, repository map, integration plan, branch retirement | Coordinating chat |
| Data release of `results-dev` | Proposed |
| Switch these pages' code links from the pinned snapshot to paths on `main` | After PR B |
