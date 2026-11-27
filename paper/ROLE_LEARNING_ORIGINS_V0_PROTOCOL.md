# ROLE-LEARNING-ORIGINS-v0

Prospective crossed-seed study, following ROLE-MECHANISM-REPLICATION-v0 and ROLE-REPAIR-SPECIFICITY-v1. The goal is to understand why event identification can emerge without participant-role binding. No previous checkpoints are used for training. No architectural, objective or hyperparameter change is introduced.

## Design

Eight fresh initialisation seeds (38101–38108) crossed with eight fresh training-stream seeds (39101–39108), 64 models, 51,200 uninterrupted AdamW updates each. The original roles_8 grammar, frozen partitions, vocabulary, batch construction, architecture and optimizer settings are reused. Each row shares exact initial weights; each column shares exact scheduled batches. The order factor changes the realised stream, including quartet sampling and per-quartet surface-cell permutation; finite-run exposure counts may differ. It is not a pure permutation of an identical finite multiset. Torch training RNG is reset to a common seed independent of both factors. Hardware nondeterminism is not ruled out. No adaptive stopping or seed selection.

## Outcomes

Primary: descriptive balanced two-way sums of squares on final strict four-question C_query, averaged equally across both held-out rich populations and both voices, using their full sets. Report init, stream and residual interaction fractions of total SS; a single run per cell cannot distinguish interaction from run noise. Do not treat cells as IID or interpret these fractions as population variance components. No significance gate or superiority claim is specified. A successful model reaches C_query >= .95 in all four held-out groups. Report final matrix and success matrix, not merely averages.

Secondary: fixed 128 source quartets in each of three populations (train-seen rich, unseen roles rich, unseen quartets rich), both voices, at step 0, every 200 steps through 6400, every 400 through 12800, every 800 through 51200. The same sample and index order are used across all runs. Final full evaluation includes train-seen. Existing metrics distinguish four-candidate joint correctness, full-vocabulary correctness, event mass and accuracy, conditional role probability and role accuracy, both-pair duplicate rate, and event/role NLL. Conditional role accuracy is gold-event-restricted comparison, not necessarily the answer the model would generate. Training loss is averaged over each checkpoint interval. Benchmark reused from prior studies, not independent linguistic validation.

At 0, 800, 1600, 3200, 6400, 12800, 25600, 51200, compute separate event and role NLL gradients on the first 16 fixed train-seen quartets per voice in eval mode, recording norms and cosine by parameter group and globally. These diagnostics do not update the optimizer. Exploratory only, not evidence of gradient conflict causing collapse. No predictor fitting or intervention in this study; a later independent cohort is required to validate predictors.

## Engineering pilot and execution

Two unrelated initialisation seeds crossed with two unrelated stream seeds, 400 updates per run; pilot outputs never enter scientific summary. Run first to 200, resume to 400, verify all 12 pilot reports, check runtime/disk and hashes. Do not revise main seeds, outcomes or thresholds based on pilot outcomes. Any engineering correction before main training must regenerate and recommit manifest; after main training starts, preserve protocol and create a versioned amendment if necessary.

Rolling latest state is atomically replaced at each metric checkpoint, containing model, optimizer, RNG, schedule digest and report. Missing latest report/archive is restored from saved state; earlier missing reports are an error. Archive eight full snapshots per scientific run. Do not delete or commit model weights. Keep branch fixed while running; other agents must use separate worktrees. Final report completeness and schedule/initial hashes checked in summary. The 64-model workload is 3,276,800 updates plus evaluation and gradient diagnostics.

## Interpretive boundaries

This study can distinguish sensitivity to sampled initialisations versus sampled training streams, not identify a unique causal circuit. The interaction/residual is descriptive. Partial learning and collapsed learning remain distinct. Predictions, prevention and grammar/architecture replication are separate prospective studies. An engineering pilot is not a confirmation of the phenomenon.
