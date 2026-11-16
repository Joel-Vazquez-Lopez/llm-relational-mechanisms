# Research studies

This index preserves canonical paths. Shared package code in `src/relational_lab/`
serves several studies; a shared utility is not evidence of a shared scientific claim.

## Earlier relational-mechanisms programme — ML-0xx / EXP-101–106

- **Question:** How do training experience, grammatical construction and internal
  computations affect relational answering and generalisation?
- **Status:** frozen for integration: existing files, paths and protocol hashes must
  remain unchanged. This describes preservation, not confirmation of every result.
- **Paper:** *Targeted Experience, Relational Generalization, and Role-Assignment
  Errors in Transformers*, working draft `paper/manuscript.md`; manuscript deadline
  15 December 2026. The programme also includes historical pretrained-LLM work;
  its exact allocation to a separate manuscript is not established by this index.
- **Folders:** `investigations/scratch_transformers/`,
  `investigations/pretrained_llms/`, `investigations/shared/`, `paper/`, `reports/`.
- **Scripts:** existing `scripts/ml0*.py`, `scripts/ml100*.py`,
  `scripts/exp101*` through `scripts/exp106*`, and `scripts/paper_*`.
- **Tests:** `tests/test_counterfactual.py`, `tests/test_cue_interactions.py`,
  plus historical offline checks in `scripts/test_exp101*.py`.
- **Docs:** `docs/RESEARCH_CHARTER.md`, `docs/EVIDENCE_POLICY.md`,
  `docs/PROVENANCE.md`, and the investigation-specific experiment indices.

## Order-to-Role — OTR

- **Question:** How do word order, semantic roles, question wording and answer
  format affect participant selection and its layerwise computation in LLMs?
- **Status:** active; exploratory development work, separate from frozen studies.
- **Paper:** Order-to-Role pretrained-LLM study; final title and framing remain open.
  The review brief targets an early-2027 ACL Rolling Review submission.
- **Package:** `src/relational_lab/experiments/order_role_*.py`,
  `src/relational_lab/experiments/stranded_p_stimuli.py`, and shared helpers under
  `src/relational_lab/instrumentation/`.
- **Scripts:** `scripts/otr*`, `scripts/slurm/otr*`.
- **Tests:** `tests/test_order_role_*`, `tests/test_stranded_p_stimuli.py`,
  `tests/test_generate_check.py`, `tests/test_logit_lens.py`,
  `tests/test_transplant.py`; shared instrumentation tests include
  `tests/test_chat_template_patching.py`, `tests/test_hf_scoring.py`, and
  `tests/test_patching.py`.
- **Docs:** `docs/order_to_role/`, including the review package at
  `docs/order_to_role/review_2026-11/`.
- **Results:** `order-to-role/results-dev` is the OTR data store. New result files
  must not be committed to `main`; `results/` remains ignored, subject to the
  existing README exception. Historical results already tracked on `main` remain
  untouched. Archive OTR data in a release or Zenodo at study freeze.
- **Workflow:** merge the cleaned `order-to-role/stimuli` code PR first. After that
  merge, develop on `main` or short-lived branches; retain `results-dev` for data.

## Shared resources and unresolved ownership

`scripts/show_chat_prompt.py` and `scripts/slurm/setup_pelle.sh` arrive with OTR
but may be reusable infrastructure; this index does not assign them exclusively.
Historical reports and general package utilities may serve multiple studies.
The precise manuscript assignment of every earlier ML/EXP experiment remains an
author decision; consult the existing indices rather than inferring it from a prefix.

OTR paths above become available on `main` after the separate code integration PR.
