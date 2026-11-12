# Scratch-trained Transformers — evidence checkpoint

**Evidence checkpoint:** controlled behavioral effects, bounded intervention results and failed mechanistic predictions. This document separates findings from interpretations; it does not claim a general learned role algorithm.

## Recent evidence (ML-072–087)

- **Exposure and behavior:** ML-072 and ML-080 compared matched training regimes; target exposure improved held-out recombination in the tested synthetic grammars. ML-084 replicated the ML-080 behavioral contrast on fresh seeds, names and pair splits. Familiar vocabulary and the grammar template remained shared within each study. [ML-080 results](../../reports/ml080/protected_evaluation.json) · [ML-084 results](../../reports/ml084/protected_evaluation.json)
- **A failed predictive account:** ML-084's frozen layer-geometry score did not predict the ordering of protected behavioral benefits. Its eight-seed rank correlation was approximately -0.048; the calibration-margin baseline was approximately +0.452. [Results](../../reports/ml084/protected_evaluation.json)
- **Protected intervention:** ML-085's target-exposure role-donor patch exceeded the norm-matched event-donor control by 0.397 on average; the frozen primary rule passed in 7/8 positive seeds. The outcome does not establish native necessity or mediation of the training effect. [Protocol](../../reports/ml085/frozen_protocol.json) · [Results](../../reports/ml085/protected_causal_evaluation.json)
- **Subspace overlap and a negative test:** ML-086's frozen erasure rule narrowly passed, while its bootstrap interval included zero. Role- and event-derived subspaces overlapped and produced similar erasure effects. ML-087's exploratory residualized interaction-ablation test did not meet its primary criterion. [ML-086 interpretation](../../reports/ml086/interpretation.md) · [ML-087 results](../../reports/ml087/calibration_evaluation.json)

The richer ML-075 benchmark failed its frozen competence gate. These results do not establish a unique grammatical circuit, explain generalization by causal mediation, or validate the synthetic findings in pretrained LLMs. [ML-075 gate](../../reports/ml075/competence_gate.json)

## Developmental investigation (ML-057–059)

Two initial seeds showed highly correlated example-level behavioral difficulty, but matching checkpoints by learning stage did **not** consistently improve agreement over comparing the same training step. The simple common-transition-at-different-speeds hypothesis was not supported.

## Causal routing (ML-060–065)

Token-wise ablations and layer-2 source-edge interventions suggested heterogeneous sensitivities: seed 7 depended more on layer-1 answer-position updates, while seed 17 depended more on earlier question positions. Q/K/V restoration effects differed. Fresh-seed tests had heterogeneous results; the frozen historical transfer classification matched 5/6 models, but a trivial always-Q classifier would also have matched 5/6, so this was not strong proof of discrete strategy classes.

## Counterfactual role transfer (ML-066–068)

Opposite-role layer-1 answer-position attention interchanges changed the selected participant in multiple independently trained models, whereas same-role, different-verb controls seldom flipped answers. A layer-2 factorial intervention produced a notable **operational double dissociation**: at step 300, seed 29 flipped 100% of eligible examples with attention-output-only transfer and 0% with input-state-only transfer; seed 41 flipped 15.1% and 84.9%, respectively. These contrasts remained directionally stable for each of the four voice/role combinations and all seven examined participant pairs.

**Limitations:** those seven name pairs were historically examined in ML-037; repeated analysis is not independent validation. Hybrid neural-state interventions measure effects under the manipulation and do not establish naturally distinct semantic algorithms. Seed 17 was largely unaffected by the tested answer-position interchange.

## Prospective compositional test (ML-069)

On a new synthetic grammar, eight models scored between 94.8% and 100% on unseen name pairs with familiar constructions but scored roughly chance on the unseen **passive + alternate-question** combination (one seed fell to approximately 25.5%). Frozen calibration pathway classifications transferred in **0/6 informative cases**. This is a failure of the planned transfer criterion, not evidence that the representations disappeared. The test changed multiple surface details; it is not natural-language validation.

## Error taxonomy and role signal (ML-070–071)

Most double-OOD failures selected the *same* participant for agent and patient questions. Seed 101 often reversed roles instead. Some models retained a correctly directed logit-level role contrast despite identical final choices, suggesting response bias and role sensitivity need to be measured separately. These were post-hoc diagnostics on the already opened ML-069 test set.

## Controlled exposure experiment (ML-072)

Three paired training arms—control, sparse, broad—tested whether limited or diverse exposure repairs the unseen grammatical combination. The [frozen protocol](../../reports/ml072/frozen_protocol.json) and [completed evaluation](../../reports/ml072/evaluation.json) are available. Average both-role accuracy on the target held-out combinations was 34.3% for control, 94.9% for sparse and 98.6% for broad exposure; broad varies both pair and verb diversity.

## Research priorities

1. Test whether a computational account predicts independent held-out effects beyond behavioral and positional baselines.
2. Separate behavioral competence from the interpretation of activation interventions.
3. Evaluate controlled lexical, positional and grammatical alternatives, then test any proposed bridge to pretrained models separately.
4. Audit prior art before claiming novelty or publication readiness.

Raw model checkpoints and recent result JSON files live locally (typically under gitignored `results/`) and are **not** assumed to exist in this GitHub branch. Use exact manifests and hashes for reproducibility.

## Final archived cross-cohort audit (12 November 2026)

The controlled target-vs-familiar exposure benefit has repeated across six eight-pair cohorts (ML-080, ML-084, ML-085, ML-086, ML-088, ML-099), all within the same three-layer synthetic grammar family. **48/48** paired seeds improve held-out target-choice accuracy, which should **not** be misdescribed as 48 independent structural or linguistic tasks.

A completed read-only exploratory cross-cohort audit of five protected summary reports with paired-role metrics (excluding ML-088, whose protected archive lacks these fields) found **40/40** positive paired improvements in both-role correctness, **33/40** decreases in *per-question* wrong-role predictions, and **26/40** increases in native *same-participant* responses. Average same-participant rate increased in each of the five cohorts despite improved joint correctness. The per-question wrong-role metric is **not** an event-level agent/patient reversal; no claim of independent five-cohort replication of the ML-088 oracle-restricted reversal effect is justified. [Final audit](../../reports/paper_cross_cohort/archived_error_taxonomy.md) · [Analysis script](../../scripts/paper_cross_cohort_audit.py).

Separately, a calibration-only exploratory audit of eight ML-088 pairs, with the candidate set restricted to the correct event's two participants, found that target exposure increased correct assignments (7.9%→39.8%) and decreased complete role reversals (39.5%→9.9%), but did not reliably decrease collapsed same-participant assignments (52.6%→50.3%). Because the candidate set is oracle-restricted and the data are previously examined calibration examples, do not pool these rates with the native protected cross-cohort results. The local script/summary must be committed with full provenance before final publication.

The final selected role-Q circuit prediction (ML-099) failed its frozen criterion, and ML-100's bounded exploratory factorial yielded no passing candidate. Do not claim a unique necessary mediating circuit. The separate public-development-only SLOG eligibility audit found 812 eligible multi-predicate role-pair sentences; it did **not** evaluate pretrained models or test SLOG structural generalization. That extension is deferred. **Experimental stop:** consolidate manuscript and evidence rather than running ML-101 or tuning models on more explored splits.
