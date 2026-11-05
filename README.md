# Model Linguistics — Relational Computation in Transformers

**Human-led computational research into how Transformers learn, represent, route and use relationships expressed in language.**

The project has **two separate investigations** sharing a common evidence policy. Neither is evidence for the other without an explicit bridging experiment.

| Investigation | Scientific objective | Research section |
|---|---|---|
| **Scratch-trained Transformers** | Development from random initialization; causal routing; compositional generalization | [Overview](investigations/scratch_transformers/README.md) · [Experiment scripts](investigations/scratch_transformers/EXPERIMENT_INDEX.md) · [Current findings](investigations/scratch_transformers/CURRENT_FINDINGS.md) |
| **Pretrained LLMs** | Relational computation in existing Qwen/Llama models, using lexical/position controls and causal interventions | [Overview](investigations/pretrained_llms/README.md) · [Experiment scripts](investigations/pretrained_llms/EXPERIMENT_INDEX.md) |
| **Shared research infrastructure** | Provenance, tests, instrumentation and reproducibility standards | [Shared resources](investigations/shared/README.md) |

## Quick start

Python 3.10+; research scripts may have additional model-specific dependencies. The lightweight repository test does not download a model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m relational_lab.analysis examples/toy_counterfactual.json
```

## Structure and compatibility

- `investigations/` — distinct research-track overviews, experiment indices, and evidence checkpoints.
- `scripts/` — **canonical executable code**, intentionally left at its existing paths.
- `src/relational_lab/` — shared analyses and instrumentation.
- `reports/` — archived research reports and frozen protocol files.
- `results/` — older tracked artifacts and newer local/gitignored experiment outputs; do not assume local runs have been committed.
- `docs/` — charter, evidence policy, provenance and workflows.
- `tests/`, `examples/`, `configs/`, `data/`, `notebooks/` — supporting material.

**Safety for active ML-072:** we have not moved or modified its Python script or frozen protocol. The ML-072 protocol records a hash of the script. A future physical code migration would require a new explicitly versioned protocol; do not relocate the active implementation mid-experiment.

## Scientific standards

Preserve failures and all seeds. Distinguish exploratory findings, controlled evaluations and independent replications; constrained answer accuracy is not full-vocabulary correctness. An activation intervention identifies causal sensitivity under that manipulation, not necessarily a unique internal algorithm. See [evidence policy](docs/EVIDENCE_POLICY.md), [scientific charter](docs/RESEARCH_CHARTER.md), and [provenance](docs/PROVENANCE.md).

[relational-transfer](https://github.com/Joel-Vazquez-Lopez/relational-transfer) remains a separate historical coordination and evidence archive. This repository is private; no open-source licence has been selected. Do not redistribute model weights or restricted datasets.
