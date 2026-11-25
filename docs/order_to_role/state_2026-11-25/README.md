# Order-to-role: state of the work, 25 Nov 2026

Handoff snapshot for anyone (or any agent) joining the paper. Everything here is dev split and
exploratory. **No test split has been opened.**

## The paper in one paragraph

Working title: *Thematic roles live on the names: how language models represent who did what*.
In five open model families (Gemma 4 31B-it, Qwen3.8-27B, phi-4, OLMo-3.1-32B-Instruct,
Granite-4.2-30B), the hidden states of the names carry their thematic role in the middle layers,
more strongly than word order or grammatical function (psych verbs show it is thematic, not
"subject"). A small set of attention heads reads the role out at a fixed depth, and the same heads
repair answers on new constructions, Spanish, Galician, UD sentences and Wikipedia text. A minimal
pair of questions ("Who was the chasing done by?" vs "Who was someone chased by?") shows that
retrieval can fail while storage stays intact; the failure enters at the question, mainly at "by".
Target: ACL Rolling Review, deadline 4 Jan 2027 (8 pages). Full draft due 14 Dec.

## Files in this folder

| File | What it is |
| --- | --- |
| `paper_draft.md` | First full draft (24 Nov), framed around internal states. `[test0]`, `[CI]`, `Q:` mark gaps |
| `results_log.md` | Running log of dev results, 20–24 Nov, with all numbers used in the draft |
| `FREEZE_draft.md` | Draft pre-registration for test set 0: claims C1–C8, pass rules, frozen layers and heads, commands. **Draft, not frozen** |
| `result_index.md` | Which result file (branch `order-to-role/results-dev`) backs which section and claim |
| `TODO.md` | Task list to submission, with owners and dates |

## Where things are

- Code: branch `order-to-role/stimuli` (patches up to 0067: test-split plumbing, freeze fixes, question battery).
- Results: branch `order-to-role/results-dev`, folder `results/` (JSON + PNG, about 500 files).
- Older review package: `docs/order_to_role/review_2026-11/` (17 Nov; framing and several numbers are outdated).
- WIKI-ROLES records: Pelle `/proj/uppmax2026-1-225/wiki` (not in git).

## Rules (do not break)

- Never run anything with `--split test0..test3`, and never read held-out items, before
  `docs/order_to_role/FREEZE.md` is committed. The freeze is Joel's commit, not anyone else's.
- Do not edit `FREEZE_draft.md` claims, pass rules, layers or heads; propose changes in writing.
- Never modify frozen historical scripts (`scripts/ml0*`, `scripts/ml100_*`).
- Swedish results are not reported until a native speaker has checked the materials.
- Keep exploratory and confirmatory results apart; keep negative results.
- ARR is double-blind: nothing written for the paper may name the authors or link this repository.

## Who does what (until the freeze, about 29 Nov)

- Claude (chat): freeze scorer and dev dry-run, confidence intervals, figures, freeze numbers.
- Joel: Pelle runs and pushes, Spanish/Galician judgements, final read and freeze commit.
- Other agents: see the task list given to them; avoid the items above.
