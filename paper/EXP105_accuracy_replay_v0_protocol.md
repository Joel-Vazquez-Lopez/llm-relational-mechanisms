# EXP-105: accuracy-level MPS reproducibility check (exploratory)

**Question.** With fixed initialization and the same ordered source–question updates, do two *separate Python processes* trained on MPS reach meaningfully different held-out familiar-development accuracies?

**Motivation.** The EXP-105 short MPS replay shows nonidentical model hashes after the first update despite identical initialized weights and minibatch indices. The CPU 90-step replay was bitwise identical. Parameter-hash differences alone cannot establish whether numerical variability causes large differences in relational accuracy.

**Protocol.** Use EXP-105's existing development long-pilot model, cross-entropy loss, AdamW, exact v3 schedule, 216 familiar-development AA/PA/AP sentences, and existing exploratory seed **10801**. Run two complete 3,200-update paired-arm trainings in *separate Python processes* with the same evaluation milestones (800, 1200, 1600, 2400, 3200). Use fresh output filenames containing the repeat index; do not overwrite or reuse historical checkpoints. Capture protocol and schedule digests, initial and final model hashes, and all development metrics. This is NOT a new pair of independent random seeds.

**Interpretation.** Comparing the two repeated runs quantifies one seed's MPS repeatability. If accuracy diverges materially, increase repetitions and/or include deterministic CPU references before any causal attribution. If accuracy matches, the run is still non-bitwise reproducible and does not prove that MPS variation could not have affected prior runs. Compare differences between repeats with the **within-repeat** PA/AP intervention differences; do not use a two-repeat result as a statistical significance test. Record platform, PyTorch version, and hardware separately.

**Scope restriction.** No L+I evaluation, no reserved/protected examples, no 107xx confirmatory seeds, and no changes to previously saved results. This tool is for development debugging only.

**Mac commands** (from the repository root, after downloading/unzipping the bundle):

```bash
python3 scripts/exp105_accuracy_replay_v0.py --phase smoke --device mps

python3 -u scripts/exp105_accuracy_replay_v0.py --phase run --device mps --steps 3200 --repeat 1 \
  | tee reports/exp105_replay_mps_rep1.log

python3 -u scripts/exp105_accuracy_replay_v0.py --phase run --device mps --steps 3200 --repeat 2 \
  | tee reports/exp105_replay_mps_rep2.log

python3 scripts/exp105_accuracy_replay_v0.py --phase compare --device mps --steps 3200
```

Results are saved in `reports/exp105/accuracy_replay_v0/` and do **not** replace anything in `results/exp105/`.
