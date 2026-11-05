# Scratch-trained Transformers — evidence checkpoint

**Status as of 5 November 2026:** exploratory mechanistic results and one prospective replication failure. This document separates findings from interpretations; it does not claim a general learned role algorithm.

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

Three paired training arms—control, sparse, broad—test whether limited or diverse exposure repairs the unseen grammatical combination. The protocol is [frozen](../../reports/ml072/frozen_protocol.json). At the time of this documentation update, the researcher reported 18/18 models at 100% familiar-task training accuracy, with calibration accuracy ranging from 92.6% to 100%; the protected ML-072 outcomes have **not** been incorporated into this record.

## Research priorities

1. Preserve ML-072's frozen protocol; evaluate and report **all** seeds and arms without modifying thresholds.
2. Separate behavioral competence from the interpretation of activation interventions.
3. Design a new external validation set, with controlled lexical, positional, and grammatical confounds.
4. Audit prior art before claiming novelty or publication readiness.

Raw model checkpoints and recent result JSON files live locally (typically under gitignored `results/`) and are **not** assumed to exist in this GitHub branch. Use exact manifests and hashes for reproducibility.
