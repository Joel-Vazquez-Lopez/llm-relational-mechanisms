<div align="center">
  <img src="assets/model-linguistics-banner.svg" alt="Model Linguistics — experience, computation and generalization" width="100%">
  <h3>How do Transformers learn and use linguistic relations?</h3>
  <p>Two complementary studies of relational generalization, grammatical roles and the computations behind model answers.</p>
  <p><a href="STUDIES.md">Research guide</a> · <a href="papers/MAP.md">Repository map</a> · <a href="docs/EVIDENCE_POLICY.md">Evidence standards</a></p>
</div>

---

## Research studies

| | **01 · Learning Relations** | **02 · Order-to-Role** |
|:--|:--|:--|
| **Research question** | How does controlled grammatical exposure affect generalization, joint-role errors and developing computations? | How do pretrained LLMs use order, grammatical structure, question wording and response conditions to select participants? |
| **Models** | Small Transformers trained from scratch | Pretrained language models |
| **Approach** | Paired training conditions, synthetic recombination, behavioral error analysis and causal interventions | Linguistic counterfactuals, matched response conditions, layerwise readouts and activation interventions |
| **Scope** | Evidence within specified synthetic grammars; external transfer remains a separate question | Development-stage behavioral and mechanistic findings; some effects depend on scoring or unusual questions |
| **Explore** | **[Study page](papers/learning-relations/README.md)** | **[Study page](papers/order-to-role/README.md)** |
| **Inspect** | [Evidence](papers/learning-relations/EVIDENCE.md) · [Experiments](papers/learning-relations/EXPERIMENTS.md) | [Evidence](papers/order-to-role/EVIDENCE.md) · [Experiments](papers/order-to-role/EXPERIMENTS.md) |
| **Reproduce** | [Guide](papers/learning-relations/REPRODUCE.md) | [Guide](papers/order-to-role/REPRODUCE.md) |

The studies have separate research questions and evidence records. A behavioral result in one is **not** a replication or mechanistic explanation of a result in the other.

## Start reading

- **[Research guide](STUDIES.md)** — the two paper directions, their status and how they connect.
- **[Learning Relations manuscript](paper/manuscript.md)** — the existing working draft; preserved in its canonical location.
- **[Order-to-Role layer investigation](papers/order-to-role/LAYERS.md)** — the distinction between accessible information, answer preferences and causal influence.
- **[Repository map](papers/MAP.md)** — ownership, branches and where code, protocols and results currently live.

## Repository status

The navigation in `papers/` lives on `main`. Research material is still being integrated **without rewriting frozen experiments**:

- The scratch-trained [EXP-109–116 and PHASE2 reports](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/exp102-clean-design) currently remain on `experiments/exp102-clean-design`.
- The active [Order-to-Role implementations](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) are on `order-to-role/stimuli`; the [development results archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results) is separate.
- Earlier experimental records, shared analysis code and the working scratch manuscript already exist on `main`.

These are navigation links, **not** claims that unmerged code is executable from `main` or that exploratory findings have passed independent confirmation. See the [integration plan](docs/INTEGRATION_PLAN.md).

## Lightweight local checks

The `main` checkout provides dependency-light tests that do not fetch model weights. The individual paper guides describe their additional requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m relational_lab.analysis examples/toy_counterfactual.json
```

## Research and reproducibility standards

We report failed hypotheses, counterexamples, frozen protocols and uncertainty alongside positive results. Candidate-restricted scoring is not equivalent to unrestricted generation; decodability is not evidence of causal use; and an activation intervention does not by itself establish a unique native mechanism.

Read the [evidence policy](docs/EVIDENCE_POLICY.md), [provenance guidance](docs/PROVENANCE.md) and [experiment reports](reports/README.md). Some raw results and model checkpoints are deliberately stored outside `main`. No open-source license has yet been selected; repository visibility alone does not grant reuse rights.
