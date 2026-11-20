[← Learning Relations](README.md) · [Experiments](EXPERIMENTS.md)

# Reproducibility guide — Learning Relations

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

Newer branch-only work is linked from [EXP/PHASE2 entries](EXPERIMENTS.md#experimental-branch-not-yet-integrated), not available as part of a `main` checkout.

## 4. Before interpreting results

Distinguish native output from restricted-candidate scoring; distinguish within-grammar seed replication from novel linguistic-family transfer; and separate descriptive layer maps from causal tests. Preserve failures and deviations. See the [evidence policy](../../docs/EVIDENCE_POLICY.md) and [source provenance](../../docs/PROVENANCE.md).
