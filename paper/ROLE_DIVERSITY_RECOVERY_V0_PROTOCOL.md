# ROLE-DIVERSITY-RECOVERY-v0

Exploratory, outcome-selected diagnostic on four previously evaluated seeds
(16101, 16127, 16143, 16167), all four original diversity arms. No independent
replication or reopening of the parent confirmatory gate.

Start at sealed 6400-update weights; reset AdamW for every arm using the
parent hyperparameters. Continue the deterministic quartet/cell/role schedule
with the original prefix verified exactly against the parent manifest. No
change to model architecture, vocabulary, role sets or evaluation populations.
The 6400 baseline is rescored and checked against the original report.
Milestones: 6400, 12800, 25600, 51200. Evaluate every arm at every milestone,
report every seed, and do not select a best checkpoint. Report mean training
loss in 800-update windows, native accuracy and pair taxonomy for all parent
populations, and L2 and relative L2 parameter displacement from original weights.
Save model, optimiser, CPU and MPS RNG state and schedule prefix at milestones.
Resume only from sealed milestones; schedule replay reconstructs exposure state.
No parent checkpoint or report may be overwritten.

Descriptive recovery classification using strict seen-role correctness:
<10% persistent collapse, 10–90% partial recovery, >90% strong recovery.
Inspect unseen performance separately. Recovery cannot distinguish additional
exposure from the optimiser reset. Persistent collapse does not establish
incapacity or a mechanistic explanation. No confirmatory hypothesis tests.
