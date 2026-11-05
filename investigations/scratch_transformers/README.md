# Investigation A — Scratch-trained Transformers

**Question:** How do relational capabilities emerge during training, and can models with similar behavior rely on different causal computations?

## Research phases

| Phase | Experiments | Research status |
|---|---|---|
| Initial scratch pilot and controls | ML-037–047 | Controlled synthetic task; several prospective hypotheses failed |
| Controlled-exposure feasibility | ML-056 | Separate pilot documented in [ML-056 report](../../reports/ml056/README.md); its runner is not among the tracked root `scripts/ml056_*.py` paths |
| Developmental checkpoints and interventions | ML-057–068 | Time alignment, token and edge ablations, Q/K/V restoration, role interchange and factorial causal sensitivity |
| New grammar and failure analysis | ML-069–071 | Prospective compositional transfer failed; structured role-response failures identified post hoc |
| Paired controlled exposure | ML-072 | **Active frozen experiment**; control, sparse and broad exposure arms |

## Navigation

- [Scratch experiment scripts](EXPERIMENT_INDEX.md)
- [Current findings and limitations](CURRENT_FINDINGS.md)
- [Frozen ML-072 protocol](../../reports/ml072/frozen_protocol.json)
- [Frozen ML-065 predictions](../../reports/ml065/frozen_predictions.json)

## Running ML-072 safely

From the repository root, preserving the frozen script location:

```bash
python3 scripts/ml072_compositional_exposure.py --phase plan
python3 scripts/ml072_compositional_exposure.py --phase train
# Run only at the intended evaluation stage:
python3 scripts/ml072_compositional_exposure.py --phase evaluate
```

All evaluation results and checkpoints should be preserved. Do not change the script after freezing its self-hash. Mechanistic results on the original synthetic grammar are **not** evidence of a universal semantic-role algorithm.
