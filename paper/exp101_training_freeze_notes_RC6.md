# EXP-101 RC6 — bounded training authorization and execution record

13 November 2026; release-preparation record **before** experimental training.

## Authorization and immutable scientific scope

The study author explicitly authorized the original bounded EXP-101 comparison: eight paired seeds (`907,911,919,929,937,941,947,953`), both `familiar` and `target` exposure arms, exactly 12,800 AdamW steps per model, 56 common and 8 dose examples per step, the frozen training-only position-ID offset cycle `t % 9`, and nine familiar development competence cells. No adaptive checkpoints, new model families, hyperparameter changes, protected data, or post-hoc tuning. The independent grammar reviewer was waived **before** study training, recorded in `paper/exp101_author_waiver_RC5.md`; it is not described as independent validation.

## Grammatical/semantic review

The existing 16 template configurations were reviewed for the **task-specific** unambiguous agent and patient labels, not full-sentence semantic equivalence. The bare, L-only, I-only and L+I active/passive forms preserve core argument labels under transformations. The fronted *By X's gate* locative and the postposed *beside X's gate* locative are not passive agents; the passive *by Aven* phrase is the agent. The *near Dovan* adjunct denotes a non-core event/person location and may vary in attachment/scope without changing the designated two argument roles. Reordering events and changing voice can affect discourse chronology, focus and pragmatics; the study **does not** claim complete utterance-meaning invariance. This is a source/label consistency review, not independent native-speaker acceptability sign-off. No concrete gold-label contradiction was found in the reviewed structural templates.

## Mac MPS warning policy

Author-reported RC4 **two independent-process disposable fixtures** passed, with same-origin checkpoint max parameter differences `0` and `4.656612873077393e-10` and exact checkpoint round trips. Each logged 21 warnings, specifically including `embedding_dense_backward_mps` and `index_put_with_accumulate_mps`. These are acknowledged as nondeterministic MPS kernels under `torch.use_deterministic_algorithms(True, warn_only=True)`; this is *not* bitwise deterministic execution. The actual Mac raw JSON and stdout logs are locally verified and SHA-256 archived by `scripts/exp101_seal_local_release.py`; they have not been fetched by this chat. The short-run fixture does not establish bitwise 12,800-step replay.

## Final execution procedure

- The final Git commit, all source SHA-256s, config, data metadata and scoring source are bound by `reports/exp101/release_manifest_candidate.json` and the local sealed `frozen_release.json`.
- The sealed file authorizes **training only** and remains absent from the checked-in repository. `exp101_seal_local_release.py` verifies the source release and the two original Mac fixture raw JSON and log files. The author has already explicitly authorized training; the required local seal records that authorization rather than requesting new permission.
- `scripts/exp101_run_authorized.py --phase train` prints the source commit and configuration fingerprint **before** launching a model. It then runs all 16 specified seed/arm models serially, printing a progress line at each 800-step checkpoint. Saved checkpoints resume without altering the budget. No protected-test APIs are invoked.
- After **all 16** training runs finish, `development_checkpoint_scores` reads only the nine *familiar* development cells. `competence_gate` requires per-cell single-role accuracy >=95%, both-role correctness >=90%, >=96 events per cell **in both arms** for >=6/8 joint seed pairs. Every model/cell and the gate outcome are written to separate JSON files. Inadequate competence means STOP with no protected access; meeting the gate still requires **separate later authorization** for protected evaluation.

## Execution limitation

This chat's tools cannot execute Python processes on the author's Mac, and the 16 experimental runs have not been started here. Their future progress and model-measured development results must come from the pinned local Mac runner's logs and generated reports; no result is inferred in advance.
