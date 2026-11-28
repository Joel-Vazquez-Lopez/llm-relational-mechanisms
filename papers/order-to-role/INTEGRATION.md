[← Order-to-Role](README.md) · [Reproduce](REPRODUCE.md) · [Shared integration plan](../../docs/INTEGRATION_PLAN.md)

# Integration of Paper II into `main`

**Current state (28 November 2026):** Paper II code, raw tracked outputs, curated analyses, supplementary materials and OTR-12 knockout summaries are on `main` (PRs #19, #20, #21, #25 and #26). The original branch audit below was conducted against `main` at `b170ac8`; its counts of files *then missing* are a historical snapshot, not the current status. `order-to-role/results-dev` remains for pinned-commit provenance and pending external archival/licence decisions. This page documents the implementation of **Phase C** and the Paper II part of **Phase D** of the [shared integration plan](../../docs/INTEGRATION_PLAN.md). Paper I is outside its scope.

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
| `papers/MAP.md`, `docs/INTEGRATION_PLAN.md` | Previously described OTR code and results as branch-only | **Done in this documentation PR**: updated to canonical `main` paths and completed-phase status |

## 3. Results stewardship

*Revised 27 November, after the coordinating review asked for the complete record to be consolidated.*

- **The branch is not merged; its files are imported.** The results-consolidation PR checks out, byte for byte, the 454 results-dev files that are the authors' own outputs, at their original `results/` paths. They are grouped into commits for the test0 record, the development record and the 70B runs. results-dev's 13 commits are not merged, so `main`'s history gets no duplicate result commits. Everything on results-dev up to its branch point (`c3e07b3`) is already in `main`'s history.
- **Four files stay out of Git**: the WIKI-ROLES item sets and check sheets. They contain Wikipedia sentences (CC BY-SA 4.0), and the repository has no licence under which to redistribute them yet. One superseded earlier version of `wiki_roles_en.jsonl`, which exists only in results-dev history, is also kept out.
- **The archive** holds all 458 files plus that superseded version. It comes with an inventory (one row per file: source commit, experiment, role, evidence status, verification need, licence, destination), SHA-256 checksums and a deterministic builder. Everything is described in [`archive/README.md`](archive/README.md). Its builder was checked against an independent checkout: all 459 checksums match, and two builds are byte-identical.
- **`main` is now self-sufficient.** Run on `main`'s `results/`, both build scripts reproduce the committed tables, figures, confirmatory output and manifests byte for byte.
- **No model weights**, tokenizers or test sets 1–3 are in any of these files.

## 4. Working from `main` afterwards

- **One branch per study, from current `main`** (for example `otr/big-models-freeze`, `otr/question-battery`), merged back with a pull request that states the source SHA, tests run and evidence status.
- **Raw outputs go where this consolidation put them.** Own outputs go to `results/` (force-added, because the folder is gitignored), with a row in the archive inventory. Third-party text goes to the archive only. Derived tables go to `papers/order-to-role/results/` or `supplementary/` through the build scripts.
- **Frozen files never change.** That covers FREEZE.md, FREEZE_XL.md, the freeze input checksums and the scorer's frozen entries. A new freeze is a new addendum file committed before its test run.
- **Next freezes due:** the 70B addendum, and any test1 use (which requires its own pre-registration).

## 5. Status

| Step | Status |
|:--|:--|
| PR #19 code integration (merge commit) | Merged as `8c095fb` (head `0b323da`) |
| PR #20 research section | Merged as `e1b503a` (head `c8819a7`) |
| PR #21 supplementary materials | Merged as `dde717d` (head `49e2b91`) |
| PR #25 raw results consolidation (`paper2/results-consolidation`) | Merged as `08eae12` (head `b99340f`) |
| Archive deposit (Zenodo or a GitHub release) | Package built and verified locally; deposit awaits the licence decision |
| PR #26 OTR-12 knockout/steering supplement | Merged as `f8fda8a` (head `aacedd5`) |
| Shared navigation, repository map and integration plan | Updated in this documentation PR; branch retirement remains separate owner decision |
