[← Paper I](README.md) · [Experiments](EXPERIMENTS.md)

# Reproducibility guide — Relational Learning in Transformers

**Start with the exact experiment protocol.** The experimental programme contains protected evaluation gates, script hashes and archived paths. This navigation refactor does not change them.

## 1. Lightweight repository checks (`main`)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m relational_lab.analysis examples/toy_counterfactual.json
```

These checks are not the full ML/EXP research suite and do not replicate published model outputs.

## 2. Choose a specific experiment

Locate it via the [experiment index](EXPERIMENTS.md) and read its frozen protocol and manifest **before** executing inference. Many scripts require additional numerical / model dependencies beyond `[dev]`, and the original checkpoint revisions and runtime may be essential.

For example, [ML-080's script](../../scripts/ml080_learning_curve.py) documents distinct planning, preflight, training and evaluation stages. Do not treat those stages as interchangeable or rerun protected phases without the original approvals and gates.

## 3. Reproduce reported statistics

Use the experiment's preserved output manifest, hashes, independent seed units, exclusion rules and analysis code. Some full checkpoints are local or gitignored; a JSON summary alone is not sufficient to reconstruct a training run.

The [complete experiment map](EXPERIMENTS.md) includes EXP/PHASE2, position controls and the 15 current studies, with pinned source snapshots and available protocols, summaries and implementations. A `main` checkout alone does not contain all these assets.

## 4. Before interpreting results

Distinguish native output from restricted-candidate scoring; distinguish within-grammar seed replication from novel linguistic-family transfer; and separate descriptive layer maps from causal tests. Preserve failures and deviations. See the [evidence policy](../../docs/EVIDENCE_POLICY.md) and [source provenance](../../docs/PROVENANCE.md).

## 5. Ongoing study and operational safeguards

ROLE-LEARNING-ORIGINS-v0 is running locally as of the 27 November handover. Do not switch branches, pull, rebase, reset, clean, delete checkpoints, restart or otherwise modify the active scratch worktree or training jobs for documentation work.

- [Branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/role-learning-origins-v0) · [Pinned snapshot](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702)
- [Protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_LEARNING_ORIGINS_V0_PROTOCOL.md) · [Study manifest](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/reports/role_learning_origins_v0/study_manifest.json) · [Implementation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/role_learning_origins_v0.py) · [Run wrapper](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/run_role_learning_origins_v0.sh)

The 8 × 8 crossed design separates initialisation and independent training-stream seeds, with 64 models and 51,200 updates each. Preserve developmental measurements, loss decomposition, gradient diagnostics, snapshots and resumable checkpoints. No scientific result summary is available yet.

Documentation work belongs on the PR branch through remote operations or an isolated worktree. Any future scientific integration requires separate authorisation. Before reproducing an archived study, verify its exact source SHA, manifests, original or fresh optimiser state, device, split gates and evaluation definitions. An expanded component search cannot substitute for the original prespecified replication criterion.

## 6. Archival gaps

The earlier named ROLE-CIRCUIT-DIAGNOSTICS-v0 branch head archives recovery trajectories, but the dedicated assets are available at the pinned cumulative snapshot 9c7146f8: [protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_CIRCUIT_DIAGNOSTICS_V0_PROTOCOL.md), [implementation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/role_circuit_diagnostics_v0.py), [runner](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/run_role_circuit_diagnostics_v0.sh), [summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/reports/role_circuit_diagnostics_v0/analysis_summary.json). They are also included in draft [PR #14](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/pull/14). Preserve this provenance distinction and use the pinned source until integration. Recomputing diagnostics from checkpoints requires the original local weights and source hashes; archived summaries alone support read-only inspection. ROLE-DIVERSITY-RECOVERY-v0 has per-checkpoint records and a manifest rather than a located aggregate summary. The ongoing origins study intentionally has no scientific result summary yet.
