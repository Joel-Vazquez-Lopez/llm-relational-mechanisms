# CORE-DIVERSITY-v1 — matched prospective training protocol

**Status:** New study implementation, NOT RUN, NOT YET FROZEN. Do not train any new model until the complete source manifest and study manifest have been generated, audited, and committed with the runner code.

## Goal and four controlled arms

Test if exposing scratch-trained Transformers to multiple balanced semantic role assignments causes improved generalization to entirely untrained assignments. Factor: one fixed core per participant quartet versus eight counterbalanced semantic-role cores. Cross with two voice exposures: same_voice (AA+PP), mixed_voice (AP+PA). Four arms: same_voice_fixed, same_voice_diverse, mixed_voice_fixed, mixed_voice_diverse.

Use only 300 OLD TRAIN quartet identifiers, freshly partitioned into 240 study TRAIN quartets and 60 study HELDOUT quartets; original 100 development and 95 reserved identifier sets are not used. Every quartet contains four familiar individual names. The withheld 60 are therefore unseen for THESE fresh randomly initialized models, not necessarily unseen for prior project models. All sources use bare/bare and do not use protected or within-event L+I.

For each train quartet, use the 8 dihedral role permutations (4 rotations plus 4 reversed rotations) as possible exposure for the core-diverse treatment. Every name appears twice in each of four semantic role positions across those eight assignments. The other 16 of the 24 role permutations are held out from BOTH treatments. On the 60 name-quartet holdouts, all 24 role assignments remain unseen during training. These splits, source texts, gold labels, and train/evaluation disjointness are verified in the source-only manifest.

All 4 arms see the SAME 8 randomly chosen quartets each of 6,400 optimizer steps, SAME two-voice pair per regime, SAME 64 supervised role-answer labels per update, SAME token multiset and SAME source lengths at each paired fixed/diverse comparison, SAME initial weights within seed, SAME optimizer hyperparameters (AdamW lr=0.002, weight_decay=0.01), SAME architecture (three causal Transformer blocks, width 96, 4 heads). The diverse arm cycles its 8 assignments within each quartet evenly; role-position distribution is intentionally different between treatments.

There are exactly 16 NEW independent seeds, no numerical repetitions and no seed/checkpoint filtering. The 4 arms use identical initialization within seed and each gets 6,400 fixed optimizer steps; no evaluation-based checkpoint selection, training-duration tuning or early stopping.

## Frozen primary, gated secondary and interpretation

**Primary endpoint:** on the 240 training quartets, strict four-question joint correctness (C_query) over all 16 NEVER-TRAINED role permutations, scored on each regime's familiar voice pair. Each seed contributes the mean of two paired differences (diverse minus fixed, for same_voice and mixed_voice). Student-t 95% CI across 16 seed values, df=15.

Success requires ALL of: the 95% CI lower bound is strictly positive, mean effect is at least +0.10 (10 percentage points), AND >=12/16 seeds have positive paired differences.

**Gated secondary:** only if primary succeeds, assess the same difference on ALL 24 permutations of 60 withheld four-name quartets, familiar voice pair for each regime. Pass if lower 95% seed-level t CI > 0. Secondary is always computed in complete summary but not interpreted confirmatorily if the primary fails.

**Exploratory only:** original training-core strict joint accuracy, individual-question and full-vocabulary accuracy, all per-arm seed distributions and paired-error breakdowns. Each seed is an independent replication unit. No competence filtering or selection of successful seeds.

This establishes, at most, a within-grammar causal impact of training-assignment diversity on heldout semantic-role combinations. This study does not prove a unique role-binding circuit, natural-language transfer, general pretrained-LM behavior, or out-of-grammar robustness. A genuinely different grammar remains a separate replication target.

## Mandatory freeze

Run source-only preflight prepare, run training-manifest prepare, and a random-weights smoke. Commit BOTH manifests, scripts and this protocol before loading any prospective seed/model. The run verifies committed byte-for-byte frozen files. Do not disclose interim test outcomes or look at post-hoc per-seed behavior before all 16 training seeds are complete. All 16 reports are retained; checkpoint tensors stay local.
