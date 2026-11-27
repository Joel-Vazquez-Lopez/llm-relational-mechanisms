[← Paper II](../README.md) · [Reproduce](../REPRODUCE.md) · [Completeness](COMPLETENESS.md)

# Order-to-Role results inventory and archive

This folder accounts for every Order-to-Role result asset that existed only on branch `order-to-role/results-dev` (at `25c6914`): 458 files, 206.4 MB, all under `results/`. It records where each file goes, how it is checksummed, under what terms it may be redistributed, and what must happen before that branch can be retired.

| File | What it is |
|:--|:--|
| [`inventory.csv`](inventory.csv) | One row per file: path, source commit, bytes, git blob, SHA-256, content type, experiment, model, split, material, role in the record, evidence status, identical copy on `main`, verification need, frozen-evaluation flag, third-party flag, redistribution terms, destination |
| [`history.csv`](history.csv) | File versions that were superseded on results-dev and are reachable from no other branch (one) |
| [`SHA256SUMS`](SHA256SUMS) | Checksums of all 458 files and the superseded version |
| [`inventory_summary.json`](inventory_summary.json) | Counts by type, status, verification need and destination |
| [`ARCHIVE_README.md`](ARCHIVE_README.md) | The README packed into the archive: provenance, licences, retrieval |
| [`build_inventory.py`](build_inventory.py) | Regenerates the inventory from git objects (read-only) |
| [`make_archive.py`](make_archive.py) | Builds the deterministic archive and verifies it against `SHA256SUMS` |
| [`COMPLETENESS.md`](COMPLETENESS.md) | Study-by-study check of the OTR-01–15 record after consolidation |

## Inventory at a glance

| Group | Files | Needed for verification | Destination |
|:--|--:|:--|:--|
| test0 result JSONs read by the frozen scorer (88 jobs) | 88 | Required | Git, `results/` |
| test0 attempt logs (canonical: `test0_attempts_pelle.md`) | 2 | Required | Git |
| Slurm logs of the 88 test0 jobs | 88 | Supporting | Git |
| Development files the frozen layers and heads were read from | 20 | Required | Git |
| Development outputs read by the Paper II build scripts | 50 | Required | Git |
| Development checks not read by the build scripts | 36 | Supporting | Git |
| 70B-class exploratory runs | 4 | Required (3) / supporting (1) | Git |
| Other run logs (development, downloads, WIKI extraction, 70B smoke tests) | 77 | Supporting / not needed | Git |
| Runner plots (PNG; regenerable from their JSON) | 87 | Not needed | Git |
| WIKI-ROLES extraction summaries (counts only) | 2 | Supporting | Git |
| **WIKI-ROLES item sets and check sheets (Wikipedia text)** | **4** | Required (item sets) | **Archive only** |
| Superseded partial WIKI-ROLES item set (`history/9cca09e/`) | 1 | Not needed | Archive only |

Three content checks:
- Four plots are byte-identical to files already on `main`: development plots that the 26 November control runs reproduced exactly.
- Six Slurm logs are identical one-line GPU headers.
- No file is a test set 1–3 output, and none contains model weights.

## How the decisions were made

- **Git, at the original path, for the authors' own outputs.** This follows `main`'s existing convention: `main` already tracks 983 files in `results/`, including JSON, PNG plots and Slurm logs, force-added by earlier PRs (among them #9 and #12). Every file is under 7 MB, well within Git limits. JSON and logs are text, which Git compresses and diffs. `results/` is gitignored, so files are force-added, as before.
- **Test0 outputs are included.** Test set 0 was opened once, on 26 November, by the frozen protocol. Its item-level outputs are needed to re-score the confirmatory claims independently, and its items are reproducible from the public generator anyway. Including them opens nothing new. Test sets 1–3 stay closed.
- **Plots and logs are included, although regenerable or noisy.** They cost little and record what the runner saw (GPU, timing, warnings, failed attempts), and they match how earlier results were kept. They could move to the archive if the coordinating review prefers a leaner tree.
- **Wikipedia text is held back.** The WIKI-ROLES item sets and check sheets contain English Wikipedia sentences (CC BY-SA 4.0). They carry the needed attribution (article title and id, dump 20231101.en). Even so, the repository has no licence, and nobody has yet decided how CC BY-SA material should sit beside it, so these files go to the archive only. The WIKI result JSONs contain no text and are in Git.

## Verification done

- All 459 checksums in `SHA256SUMS` match an independent checkout of `25c6914`.
- Two builds of the archive are byte-identical. The archive is `order-to-role-results-dev-25c6914.tar.gz`, about 20 MB; its checksum is in the `.sha256` file produced next to it.
- Run on `main`'s `results/` after this PR, `build_results.py` and `build_supplement.py` reproduce every committed table, figure, confirmatory summary and manifest byte for byte. The frozen scorer reproduces test0 English 37/38, BX 15/15, Spanish 29/30 and Galician 30/30.
- results-dev changed only `results/` after its branch point, and everything up to that point is in `main`'s history. With this PR and the archive, every blob ever committed to results-dev is preserved.

## Open blockers (owner decisions)

1. **No repository licence.** The repository is public but declares no licence, so reuse terms for all research outputs, code included, are undefined. Choose licences for code and for data (for example MIT or Apache-2.0 for code, CC BY 4.0 for results) before depositing the archive with a DOI.
2. **Wikipedia text (CC BY-SA 4.0).** Decide whether the four WIKI-ROLES files may enter Git with a licence notice, or stay archive-only. Either way, a public archive must state CC BY-SA 4.0 and keep the per-sentence attribution.
3. **UD sentences already on `main`.** `src/relational_lab/experiments/ud_en_natural.json` (from PR #19) contains 89 EWT sentences (CC BY-SA 4.0) and 99 GUM sentences (CC BY-NC-SA 4.0, non-commercial). This PR adds an attribution and licence notice (`THIRD_PARTY_DATA.md`). The owner should confirm that a non-commercial subset is acceptable in the repository.
4. **Deposit.** Upload the archive (Zenodo for a DOI, or a GitHub release asset) once 1 and 2 are settled, then record its URL and checksum in `REPRODUCE.md`.

## Branch-retirement prerequisites (for the coordinating review)

`order-to-role/results-dev` can be retired only when all of these hold:

1. This PR is merged, so the 454 own-output files are on `main` at their original paths.
2. The archive is deposited at a persistent location, and an independent `make_archive.py verify` on the deposited file reports 459 members, 0 failures, 0 missing.
3. The archive's URL and checksum are recorded in `REPRODUCE.md`, and the deposit's licence statement is correct (blockers 1 and 2).
4. No page relies on a link to `25c6914`. Commit `25c6914` is not in `main`'s history, so deleting the branch would make it unreachable. The Paper II pages cite it only as a label: there are no blob or tree links to it, and the rebuild commands now use `main`'s `results/`.
5. A final `git diff` confirms that nothing else in the branch is absent from `main`, the archive, or `main`'s history (verified for `25c6914` on 27 November).

`order-to-role/stimuli` is already fully contained in `main`'s history (PR #19 was a merge commit). Its retirement needs only the coordinating review's sign-off. Note that pinned links to `9c43792` keep resolving, because that commit is an ancestor of `main`.
