# ROLE-ANCHOR-PROBE-v0 — independent 12-seed result audit

**Status: COMPLETE, EXPLORATORY, READ-ONLY.** This test was designed after observing failures in TRAIN-SURFACE-FACTORIAL-v0 and TRAIN-SURFACE-CROSS-EVAL-v0. It is not a new independent replication or mechanistic circuit proof. These are already-trained 48 checkpoints, grouped as 12 independent training seeds × four related arms.

## Provenance and reconciliation

- All **12 original frozen seed reports** and \`analysis_summary.json\` were retrieved from \`analysis/role-anchor-probe-v0/reports/train_role_anchor_probe_v0/\`.
- All 12 list the parent protocol digest matching \`reports/train_surface_factorial_v0/study_manifest.json\`. Original diagnostic committed head: \`68bfb1fb501dc36db298d60bf014fd15c84be2c8\`. All agree on one frozen cross-surface manifest hash \`d9b99fe8751585bea984298b7e2dedc80b3b8d302d530c285f0d1acfcc95ed06\`.
- The 12 reports show 12 distinct checkpoint-file SHA-256s, no new training and no original DEV/reserved/protected/L+I access. All arms, both voices, and all four test surfaces are present.
- Independent direct averaging over 12 reports exactly reproduces **every** saved \`analysis_summary.json\` metric for all four arms, surfaces and eight tracked statistics.
- Evaluation population is **24 TRAIN-derived familiar quartets × 16 untrained role permutations × AP/PA voices**; each seed/arm/surface scores 768 semantic-voice items. The same original role tuple is NEVER fully correct by design on these role-recombined items. The always-original-tuple rule gets **0% strict C_query**, **25% single-answer accuracy** on average, as a diagnostic baseline.

## Main descriptive result: original-role anchoring in failed combined-condition models

The primary question: are the models trained with **both prefixes and named adjuncts** and unable to recombine roles actually *predicting the original four-name event-role tuple*? The answer is **yes, strongly for the eight previously identified failing seeds**, with important variation.

| Combined training seed | Strict 4-answer accuracy vs novel roles (%) | Exact prediction of original 4-role tuple (%) | Original-role slot match (%) |
|---|---:|---:|---:|
| 14901 | 0.00 | 100.00 | 100.00 |
| 14903 | 100.00 | 0.00 | 25.00 |
| 14907 | 0.00 | 98.96 | 99.74 |
| 14909 | 0.00 | 100.00 | 100.00 |
| 14913 | 100.00 | 0.00 | 25.00 |
| 14919 | 0.26 | 59.77 | 86.10 |
| 14921 | 100.00 | 0.00 | 25.00 |
| 14927 | 0.00 | 100.00 | 100.00 |
| 14931 | 22.14 | 8.72 | 49.67 |
| 14933 | 0.00 | 76.95 | 92.32 |
| 14939 | 0.26 | 68.36 | 89.68 |
| 14943 | 0.00 | 97.92 | 99.48 |

**Values:** mean across AP and PA on **bare-surface** sources; same pattern appears under prefix, adjunct and combined test surfaces.

- Eight failing seeds (heldout joint accuracy below 1%): **mean 87.74% exact original-role tuple matching**, and **95.91% single-answer slot anchoring**. They include five seeds with roughly ≥97% exact tuple anchoring and several partially anchored seeds.
- Three generalizing seeds (14903, 14913, 14921): **100% true role recombination, 0% exact original-role tuple matching**, on bare-surface sampled examples.
- One intermediate seed (14931): **22.14% true all-four accuracy, 8.72% exact old-core tuple matching**.
- Combined-arm all-12 mean: **26.89% true joint accuracy**, **59.22% exact original role-tuple matching**, **74.33% per-answer original role-slot matching**, and **82.42% original-event-pair membership**.
- Highly anchored seed 14901 predicts the exact same original tuple on all 16 unseen role permutations of every sampled quartet (1 unique tuple/quartet); successful seed 14903 gives 16 distinct correct tuples/quartet. For the eight failing seeds, there is a pronounced rigidity in predictions even across changed sentence surfaces.

## Context from all four ORIGINAL training arms

On **bare test surfaces**, independent all-seed averages:

| Training arm | Novel-role strict C_query (%) | Exact original-role tuple prediction (%) | Mean distinct prediction tuples per 16 new roles |
|---|---:|---:|---:|
| Bare | 99.09 | 0.00 | 16.00 |
| Prefix only | 84.13 | 2.12 | 15.13 |
| Adjunct only | 96.39 | 0.00 | 16.00 |
| **Prefix + adjunct** | **26.89** | **59.22** | **6.20** |

These are DESCRIPTIVE post-hoc contrasts on models already used to formulate the hypothesis, not separately pre-registered tests.

## Inference and limits

**Supported behavioral inference:** for many seeds, especially those training on prefixes + name-bearing adjuncts, the model retains a **quartet-specific original role assignment** and outputs it on unseen role permutations instead of responding to the novel source semantics. This directly explains why decorating the test source with familiar training surfaces did not rescue those models: the output mapping is largely insensitive to the changed input roles.

**Not established:** an explicit lookup table implemented in weights, a uniquely identified attention head/role-binding circuit, an inevitable role-anchoring attractor, proof that models memorized training examples rather than another shortcut with identical outputs, an independent confirmatory interaction, or a necessity for every role permutation to be seen during training. Three successful combined-condition seeds show that the original single-role training exposure CAN yield correct generalization in this model/task family.

**Next strong causal training question:** holding the rich surface grammar and computational budget fixed, does exposing a quartet to 2, 4 or 8 distinct role permutations (as opposed to 1) reduce the incidence of original-role anchoring and improve transfer to the common 16 truly heldout permutations? The already source-checked proposed design is [ROLE-DIVERSITY-COMPLEX-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/design/role-diversity-complex-v0/paper/ROLE_DIVERSITY_COMPLEX_V0_DESIGN.md). This should be a **fresh-seed prospective** experiment with role-assignment and surface-cell schedules frozen before first training; repeated exposure to a second assignment is *not* a one-shot example.

## Research priority

Focus on the matched-diversity training intervention before another exploratory circuit search. Continue to track the scratch-trained causal-learning strand as separate from the main pretrained-LLM work; shared architectural terms do not establish a shared mechanism.
