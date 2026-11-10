# Research state — 2026-11-09 (through ML-087)

**Status:** Retrospective evidence ledger and planning note, not a preregistration, new experiment, or causal claim.

## Central question

How does limited exposure to a previously withheld grammatical construction change a small Transformer's ability to answer *who did what to whom* across two events? Which internal computations explain that exposure-dependent generalization?

### Experimental scope and design

- Models: scratch-trained, three-layer, width-96, four-head causal Transformers; two events, four distinct participants, active/passive clauses, questions targeting event (first/second) and role (agent/patient).
- Matched arms: repeated exposure to **64 distinct questions**, either familiar-position passive or target-position passive, with the remaining training design matched. The 64 questions recur during training; they are **not** merely 64 total training examples.
- Withheld construction: passive target event in second position. Protected evaluation uses participant pairs and verbs excluded from the specific exposure examples, while the same vocabulary and overall grammar generator remain in use.
- ML-080, ML-084, ML-085 and ML-086 are fresh paired-seed cohorts, not external task-family or architecture replications.
- Throughout, a model's performance on familiar calibration grammar must not be conflated with its ability to generalize to withheld grammar.

## Behavioral evidence — the most robust finding

| Experiment | Familiar exposure: protected target choice accuracy | Target exposure: protected target choice accuracy | Mean paired benefit | Positive pairs |
|---|---:|---:|---:|---:|
| ML-080 | 15.54% | 57.85% | **+42.32 pp** | 8/8 |
| ML-084 | 8.6% | 56.1% | **+47.54 pp** | 8/8 |
| ML-085 | 9.3% | 58.5% | **+49.16 pp** | 8/8 |
| ML-086 | 12.44% | 51.91% | **+39.47 pp** | 8/8 |

**Total: 32/32 positive paired-seed effects across four fresh cohorts.** This is strong reproducibility *within* the fixed synthetic grammar family, training algorithm and model architecture. These effects are not 32 independent task-family replications.

Reported seed-paired bootstrap intervals for the mean benefit:
- ML-084: +36.84 to +59.03 pp.
- ML-085: +36.15 to +61.09 pp.
- ML-086: +26.38 to +55.68 pp.

The target-exposure arm does **not** universally solve the protected construction: mean choice accuracies remain approximately 52–59% in the later cohorts, with substantial seed variation.

Sources: [ML-080](ml080/protected_evaluation.json), [ML-084](ml084/protected_evaluation.json), [ML-085](ml085/protected_causal_evaluation.json), [ML-086](ml086/protected_evaluation.json).

## Mechanistic evidence — what the experiments actually establish

### ML-081 / ML-082: calibration interventions

Layerwise donor-directed attention/MLP replacements can redirect role or event answers on familiar calibration questions more effectively than norm-matched noise in several settings. This demonstrates *causal answer-relevant information transfer under intervention*, but not native necessity, unique encoding, or mediation of the exposure advantage. Results vary among seeds.

Sources: [ML-081](ml081/discovery.json), [ML-082](ml082/directional_specificity.json).

### ML-083: additive-composition hypothesis failed

Combining independent event and role differences generally failed to approximate the corresponding joint answer intervention at deeper sites; the preregistered L2-attention composition-minus-shuffled contrast was negative. The factorial interaction `gamma = h11 - h10 - h01 + h00` was large in many trained models, but that alone is not evidence of necessity or grammatical specificity. Preserve the failed composition test and all heterogeneous seeds.

Source: [ML-083](ml083/compositional_patching.json).

### ML-084: prospective geometry predictor failed

- Behavioral replication succeeded, but the frozen L2 relative-interaction ranking **did not predict** the size of the protected exposure benefit: Spearman rho **-0.0476** across eight paired seeds.
- A simpler calibration-margin ranking had rho **+0.4524**, an exploratory comparator result with only eight seeds; it is not validated as a reliable predictor.
- The observation that training alters interaction geometry does not identify the representation that causes generalization.

Sources: [frozen predictions](ml084/calibration_predictions.json), [protected result](ml084/protected_evaluation.json).

### ML-085: donor-directed causal transfer on withheld grammar succeeded, but is limited

- Target-arm primary role-donor hit rate minus structured event-control hit rate: **+39.68 pp**, bootstrap interval **+18.91 to +61.62 pp**, **7/8** positive seeds; the frozen rule passed.
- Across target-arm seeds: role-donor hit **49.9%**, event control **10.2%**, shuffled-role **22.9%**, random perturbation **22.9%**. This supports directional causal information transfer, although not all seeds beat the stronger controls.
- Donor-hit is *not the same as a native prediction switching*: a model might already predict the donor's answer before patching.
- The metric named `lexical_flip_all` actually computes **post-intervention error versus gold**, not a prediction-change rate. Its roughly 42.4% mean in the target arm is near baseline error and cannot be described as a 42.4% lexical flip rate.
- The test does not establish that this layer is normally necessary or that it mediates exposure-related improvement.

Source: [ML-085 protected result](ml085/protected_causal_evaluation.json).

### ML-086: subspace ablation suggests task-relevant directions, not unique role necessity

- Prospectively frozen primary difference-in-differences (role erasure vs norm-matched random, target arm minus familiar arm): **+5.27 pp**, **6/8** positive seeds. It *narrowly met* the threshold (>5 pp and >=6 positive), but its bootstrap interval **[-1.50, +13.05] pp includes zero**. This is suggestive and statistically uncertain.
- Target-exposure models, withheld grammar probe: native **52.08%**, role erasure **45.44%**, event erasure **44.69%**, random erasure **49.77%**.
- On familiar grammar in target-exposure models: native **100%**, role erasure **89.94%**, event erasure **89.36%**, random erasure **99.89%**.
- Role and event erasures have similar impact and also hurt mastered familiar grammar, undermining claims of a uniquely role-specific or unfamiliar-grammar-specific mechanism.
- Post-hoc role/event rank-8 subspace overlap: mean **0.839** in familiar-exposure and **0.740** in target-exposure models (independent random 8D-in-96D expectation **0.083**).
- Fitting factorial main effects reduces mean overlap to **0.690** / **0.633**, but substantial overlap remains. Interaction signal is present in both original difference matrices and may contribute to their shared directions.
- These are representational measurements; the interaction's squared-norm energy is *not* explained behavioral variance. Seed heterogeneity includes nearly absent interaction-energy fractions in some cases.

Sources: [frozen result](ml086/protected_evaluation.json), [post-hoc audit](ml086/posthoc_audit.json), [interpretation](ml086/interpretation.md). The post-hoc analyses are not independent confirmation.

### ML-087: interaction-only residual ablation did not meet its rule

- Reused the 16 ML-086 models; fitted directions on 64 familiar calibration events and tested 64 disjoint familiar calibration events; **no fresh protected evaluation** and no new model cohort.
- Primary target-arm random-minus-interaction-unique accuracy: **+0.39 pp**, bootstrap interval **[0, +1.17] pp**, **1/8** positive seeds; **frozen rule not met**.
- Native accuracy was **100%**, interaction-unique erasure **99.61%**, random control **100%** on the evaluated calibration questions. The floor for damage / ceiling in accuracy limits interpretability.
- The small effect does not prove factorial interactions are unimportant on harder held-out grammar. The associated directions also need not be uniquely semantic.
- A documented duplicate-initialization bug was fixed through a separate **pre-evaluation orchestration amendment**, without changing the original frozen protocol or hypothesis.

Sources: [ML-087 result](ml087/calibration_evaluation.json), [protocol](ml087/frozen_protocol.json), [amendment](ml087/AMENDMENT-001.md).

## Current conclusion — evidence grades

**Established within the synthetic framework:** target-position exposure robustly improves accuracy on held-out participant-pair / exposure-verb grammatical combinations.

**Supported but limited:** attention and other activation interventions can transfer answer-relevant information; certain learned subspaces affect predictions beyond random perturbations.

**Not established:** an identifiable single event-role binding circuit; geometry as a validated predictor of generalization; unique role-vs-event subspaces; necessity of interaction-only directions; a causal mediation pathway from exposure to successful generalization; transfer across model sizes, architectures or natural language.

## Leading competing explanations

1. **Representation-access / routing hypothesis:** familiar-arm models may possess usable event/role information but fail to retrieve or route it correctly when passive syntax appears in the second event; exposure changes conditional information access.
2. **Representation-content hypothesis:** target exposure changes the semantic features available in the network, rather than only their access.
3. **Multiple-mechanisms hypothesis:** different seeds solve the task via different circuits and can yield similar behavioral outcomes.

Current results do not distinguish these explanations.

## Next decisive experiment (proposal only; NOT a frozen protocol)

Design a **fresh-model, held-out, prospective dissociation** between information *availability* and *native accessibility*:

- Predeclare matched familiar/target questions and measure role/event information decodability at specified sites on calibration data, with controls for trivial answer-token leakage and changed surface cues.
- On separate held-out target constructions, test whether a prelearned routing/access intervention can selectively recover correct answers **without importing a donor answer**. Compare against matched generic disruption and representation-content manipulations.
- Evaluate native correctness, four-choice and full-vocabulary predictions, counterfactual event/role specificity, confidence margins, and intervention effect sizes by seed.
- Define a single primary causal contrast, sign and plausible minimum effect; report seed-paired uncertainty, gates, failure outcomes, and generalization across at least one genuinely changed grammar/task setting.
- Do not reuse opened protected examples as confirmatory data. Freeze source hashes, all selection rules, calibration procedures, metrics and planned comparisons before new model evaluation.

This is a research direction to refine, **not** an instruction to start ML-088 or to tune existing protected results.

## Reproducibility / record-keeping rules

- Keep all frozen protocols, original protected reports, and negative findings intact.
- Separate preregistered tests from exploratory analyses, retrospective interpretation and new hypotheses.
- Treat paired **seeds** as the relevant replication units within a cohort; avoid pseudoreplication from large counts of correlated questions.
- Avoid equating answer-donor injection with native causal necessity; avoid equating large geometric interactions with mediation.
- Continue one experiment at a time: plan → preflight → freeze/commit → train/gate → calibration freeze if needed → one-time protected evaluation → audit and document.
