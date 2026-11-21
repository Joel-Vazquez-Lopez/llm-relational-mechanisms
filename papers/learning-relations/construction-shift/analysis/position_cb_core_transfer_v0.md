# CORE-TRANSFER-v0: independently audited, exploratory checkpoint study

**Status:** All 16 original read-only core-transfer JSON reports present and independently reaggregated on the analysis branch. Eight independent seeds, two MPS numerical repeats per seed, two fixed arms = 32 original checkpoints. No retraining, new original development scoring, or protected/reserved/L+I access.

**Source:** [frozen manifest + all 16 original reports + summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/analysis/position-cb-core-transfer-v0/reports/position_cb_core_transfer_v0). Source SHA-256: `c935597e6ba0cb4a706a77b0d44687fea570526ae64a5f0d42362c1eba663d1e`. Study protocol digest: `0f9c867b5afb37c9bc73737a8b059fbf269d545731355bd222640e90d5c2f84f`. All 16 reports match the original source and protocol hashes; raw paired-seed estimates and 95% CIs exactly reproduce the archived summary.

## Core transfer, familiar grammar only

All manipulated examples use the same four names and both original verbs as an existing training core, identical familiar voice exposure (AA+PP in same_voice, AP+PA in mixed_voice), identical modifier category, and equal surface-token length. New role/event arrangements are disjoint from the original training-surface pool. The diagnostic sampled **144 distinct original training four-name sets** × 12 role configurations × four voices, comprising **6,912 independently parsed source presentations**. The three changed-core tiers each average multiple variants within a quartet.

| Condition | Same-voice strict joint accuracy | Mixed-voice strict joint accuracy |
|---|---:|---:|
| Original training core | 97.63% | 91.93% |
| Flip one/both within-event agent/patient roles | 34.98% | 37.36% |
| Swap event order, optionally reverse roles | 32.11% | 37.32% |
| Repartition participants into different event pairs | 33.47% | 37.35% |
| Existing independent-name-quartet familiar development (previously archived) | 35.44% | 38.54% |

Across both arms, the *original minus changed-core* seed-paired effects are:

| Exploratory comparison | Mean difference | 95% seed-level Student-t CI | Positive seeds |
|---|---:|---:|---:|
| Original − within-event role flips | **+58.62 pp** | **[+32.95,+84.28] pp** | 7/8 |
| Original − event order swap | **+60.07 pp** | **[+33.99,+86.14] pp** | 8/8 |
| Original − participant repartition | **+59.37 pp** | **[+33.62,+85.11] pp** | 8/8 |

All confidence intervals are *exploratory, unadjusted* and do not represent new prespecified confirmatory tests. Not all eight seeds fail: seed 12703 is essentially fully transferable in both arms, while others are often near zero on modified cores.

## The error-type distinction

Pair-level categories were evaluated with the original exhaustive mutually exclusive correctness/reversal/collapse/wrong-event taxonomy. Mean over the first and second events:

| Changed-core intervention | Both-role correct: same/mixed | Complete reversal: same/mixed | Wrong-event pair: same/mixed |
|---|---:|---:|---:|
| Within-event role flip | 57.57% / 56.28% | **34.76% / 37.49%** | 1.60% / 4.20% |
| Event swap | 39.17% / 38.18% | 2.72% / 0.69% | **52.81% / 59.55%** |
| Repartition | 41.91% / 38.75% | 3.34% / 1.27% | **49.00% / 57.92%** |

The same broad strict-joint accuracy failure masks **qualitatively distinct answer errors**. Role swaps increase paired agent–patient reversals, whereas event-position and participant-pair manipulations tend to produce names associated with the other event. This is a richer, falsifiable behavioral signature than aggregate `C_query`, but no head/circuit is identified and the explanations remain post hoc.

## Strong lexical-bag negative control

**Sixteen anchors used neither gate nor near distractors (bare/bare).** For every anchor and voice, all twelve core permutations used the exact same token multiset, not merely the same length. All 16 original reports include this subset; independently reaggregated across both training arms, the strict joint accuracy gaps persist:

| Bare/bare original minus | Gap | Exploratory 95% CI |
|---|---:|---:|
| Role flips | **+58.72 pp** | **[+33.12,+84.33] pp** |
| Event order swaps | **+59.72 pp** | **[+33.90,+85.54] pp** |
| Participant repartition | **+59.50 pp** | **[+33.64,+85.36] pp** |

This rules out changed *bag-of-words identity/frequency* as a complete explanation for the bare/bare behavioral collapse. It **does not** rule out learned absolute/relative token positions, position-specific participant-name associations, order statistics, or memorized source cores. The bare/bare subset is small and was analyzed after results motivated this diagnostic.

## Scientific decision

The prior unused-training-variant audit only held out nearly identical re-renderings of already encountered event cores; it scored around 95% jointly correct. This completed core-transfer audit holds out the actual **participant role/event assignment** within the same familiar name quartet, and produces familiar-grammar accuracy reductions that closely match the old development set. Thus, *new four-name-set identity is not required to reproduce the failure*.

**Strongest supported claim:** original scratch models often fail to recombine underlying participant-role/event relationships within previously seen four-name sets despite extensive training-variant coverage and, in an exact token-bag-controlled subset, without changing token multiset. The type of changed relationship predicts the dominant error class.

**Not supported:** a uniquely localized role-binding circuit, literal memorization of all input strings, causal necessity of one attention head, independent grammatical-family generalization, or causally identified learning mechanism.

**Next prospective study:** [CORE-DIVERSITY-v1](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/design/position-cb-core-diversity-v1), comparing fixed-core to balanced multiple-core training with fresh seeds, a jointly heldout role-permutation primary endpoint and an unseen-quartet gated secondary endpoint. Results are not yet available. Do not tune this new experiment after viewing its test outcomes.
