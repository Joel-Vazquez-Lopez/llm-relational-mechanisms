[← Research atlas](../STUDIES.md)

# Repository map

A navigation layer over canonical files: no source files are relocated or duplicated.

| Location | Purpose / ownership |
|:--|:--|
| `STUDIES.md` | Front door to the two paper directions |
| `papers/learning-relations/` | Scratch-paper navigation and proposed development plan |
| `papers/order-to-role/` | LLM-paper navigation and layer-study plan |
| `paper/manuscript.md` | Existing scratch working manuscript, preserved |
| `paper/` | Historical protocols and manuscript material; retain exact paths |
| `investigations/scratch_transformers/` | Scratch evidence and canonical experiment index |
| `investigations/pretrained_llms/` | Earlier pretrained ML studies, background to the LLM direction |
| `scripts/ml*.py`, `scripts/exp*.py`, `scripts/paper_*` | Existing research scripts; consult indices for assignment |
| `scripts/otr*`, `scripts/slurm/otr*` | Active OTR scripts and cluster entry points, currently on stimuli branch |
| `src/relational_lab/experiments/order_role_*.py` | OTR experiment implementations |
| `src/relational_lab/experiments/stranded_p_stimuli.py` | Stranded-preposition stimuli |
| `src/relational_lab/instrumentation/` | Shared instrumentation, also used by OTR |
| `docs/order_to_role/review_2026-11/` | OTR review package, figures and tables |
| `reports/` | Archived earlier evidence, protocols and audits |
| `results/` on `order-to-role/results-dev` | OTR raw data store; archive at freeze |

## Tests

Earlier lightweight checks: `tests/test_counterfactual.py`, `tests/test_cue_interactions.py`.
Historical EXP offline checks: `scripts/test_exp101*.py`.

OTR: `tests/test_order_role_*`, `tests/test_stranded_p_stimuli.py`,
`tests/test_generate_check.py`, `tests/test_logit_lens.py`, `tests/test_transplant.py`.
Shared instrumentation coverage includes `tests/test_chat_template_patching.py`,
`tests/test_hf_scoring.py` and `tests/test_patching.py`. The OTR branch may add further tests.

Full suite from a checkout containing the code and its dependencies:

```sh
PYTHONPATH=src python -m pytest -q tests
```

## Preservation and data workflow

Existing `main` files, including frozen ML/EXP scripts, protocols and historical
tracked results, are unchanged by this navigation PR. The shared package serves several
studies. New raw OTR results belong only on `order-to-role/results-dev`; generated review
figures/tables under `docs/order_to_role/` are explicitly permitted.

After the separate OTR integration, use `main` or short-lived code branches. Retain the
data branch until a release or Zenodo archive is made at freeze. Documentation links to
the stimuli branch are intentional while integration remains pending.

## Ownership and open decisions

- Earlier pretrained ML studies are related background, not automatic replication of OTR.
- Exact manuscript allocation of each historical ML/EXP study is not inferred from its prefix.
- `scripts/show_chat_prompt.py` and `scripts/slurm/setup_pelle.sh` are reusable helpers;
  exclusive study ownership is unspecified.
- Final OTR title, confirmatory protocol and publication venue remain open.
- New branch-only EXP work is not silently added to the frozen manuscript or this PR.

**Reviewed navigation snapshot:** 18 November 2026. Research plans are proposals,
not completed experiments or authorizations to access protected evaluations.
