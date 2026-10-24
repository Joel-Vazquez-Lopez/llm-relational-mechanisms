# LLM Relational Mechanisms

**Human-led computational research on how pretrained language models represent, select and use linguistic relationships.**

This is the official, researcher-written research codebase. The scientific objective is to identify reproducible, causally discriminating mechanisms in real pretrained LLMs, without assuming that an output contrast, probing result, or activation patch proves a unique computation.

**Status:** infrastructure / exploration. No novel LLM mechanism is established by this repository.

## Start here

Requires Python 3.10+; initial example has no model download or mandatory third-party dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m relational_lab.analysis examples/toy_counterfactual.json
```

The included counterfactual example is **synthetic test data**, not evidence about a real language model.

## Structure

- `src/relational_lab/analysis/` — read-only diagnostics
- `src/relational_lab/instrumentation/` — future model adapters and hooks
- `src/relational_lab/experiments/` — prespecified investigations
- `tests/` and `examples/` — verifiable unit tests and synthetic examples
- `configs/`, `notebooks/`, `data/`, `results/`, `reports/` — experiment organization
- `docs/` — charter, provenance, evidence policy and experiment templates
- `.github/workflows/ci.yml` — automated tests

## Scientific standards

Preserve raw evidence, failed gates and independent units. Before new model inference, write hypotheses, plausible copying/position/output baselines, a discriminating prediction, dataset and checkpoint revisions, budget, test integrity checks and stopping criteria. Interpret descriptive, predictive and causal evidence separately. Never treat a corrected numerical bug as confirmation of a scientific hypothesis.

## Research provenance

[relational-transfer](https://github.com/Joel-Vazquez-Lopez/relational-transfer) is the separate historical evidence and coordination repository. Its results have varied validation status. Nothing from it is automatically imported into this codebase; review provenance, permissions and licensing before copying artifacts.

## Workflow

- **Scientific Research Lab:** formulate hypotheses and interpret evidence.
- **Hands-On Coding Lab:** researcher implements and runs code here.
- **Codex HQ:** bounded implementation reviews and engineering coordination.

Private by default. No open-source license selected; do not redistribute restricted datasets or weights.
