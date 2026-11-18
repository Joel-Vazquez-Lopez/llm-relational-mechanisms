![Model Linguistics — How relations are learned. How answers are formed.](papers/assets/research-atlas.svg)

<p align="center">
<a href="#01--learning-relations">Learning relations</a> &nbsp; / &nbsp;
<a href="#02--order-to-role">Order-to-Role</a> &nbsp; / &nbsp;
<a href="#materials--reproducibility">Materials &amp; reproducibility</a>
</p>

Language understanding involves identifying an event, distinguishing its participants,
and selecting the person a question asks about. This repository investigates how
Transformers acquire and use those abilities through two complementary studies.

One controls **the experience a model learns from**. The other examines **the computations
that lead a pretrained model to an answer**.

---

## 01 — Learning relations

### Targeted experience and relational generalisation

Small Transformers trained from scratch make it possible to compare controlled learning
histories. This study examines how grammatical exposure changes generalisation—and why
improved answer accuracy can coexist with persistent errors across complementary role questions.

**Focus:** controlled exposure · joint-role errors · development across training and layers

The existing manuscript brings together repeated behavioral effects within a synthetic
task family, contrasting error patterns, and tests of candidate computational explanations.
Its scripts and frozen protocols retain their original paths and hashes.

**[Read the working manuscript →](paper/manuscript.md)**

[Study overview](papers/learning-relations/README.md) ·
[Evidence and limitations](investigations/scratch_transformers/CURRENT_FINDINGS.md) ·
[Experiment index](investigations/scratch_transformers/EXPERIMENT_INDEX.md)

<sub>Working manuscript · ML / EXP programme · Existing experimental assets preserved</sub>

---

## 02 — Order-to-Role

### Grammatical information and answer selection in LLMs

Pretrained language models can respond differently to the same event when the question
or requested answer format changes. Order-to-Role investigates these differences by
separating word order, grammatical function and semantic role.

**Focus:** linguistic contrasts · response conditions · layerwise representations and interventions

The layer study connects the behavioral observations to a mechanistic question: where
is relevant information available, and which computations influence the participant
selected as the answer? Decoding, logit-lens analysis and causal interventions provide
different kinds of evidence; their conclusions are kept separate.

**[Read the study overview →](papers/order-to-role/README.md)**

[Review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) ·
[Layerwise analysis](papers/order-to-role/LAYERS.md) ·
[Review questions](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/REVIEW_BRIEF.md)

<sub>Active exploratory study · OTR programme · Final manuscript framing remains open</sub>

---

## Materials & reproducibility

| Material | Location |
|:--|:--|
| Code, tests and study ownership | [Repository map](papers/MAP.md) |
| OTR raw results | [Development data branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results) |
| Evidence and interpretation standards | [Evidence policy](docs/EVIDENCE_POLICY.md) |
| Research provenance | [Provenance guide](docs/PROVENANCE.md) |

The study pages distinguish archived findings, exploratory observations and proposed
experiments. Shared infrastructure does not imply that findings transfer between studies.

<details>
<summary><strong>Repository and archival notes</strong></summary>

Existing research files remain in their canonical locations. Shared code in
`src/relational_lab/` serves several studies. OTR code links currently point to
`order-to-role/stimuli` while its separate integration is pending; after integration,
new code belongs on `main` or short-lived branches.

Raw OTR results remain on `order-to-role/results-dev` until release or Zenodo archival
at freeze. The 99 historical results files already tracked on `main` remain untouched.
Review figures and tables under `docs/order_to_role/` are permitted documentation artifacts.

[Ownership and open decisions](papers/MAP.md#ownership-and-open-decisions)

</details>

<sub>Research guide · Updated 18 November 2026</sub>
