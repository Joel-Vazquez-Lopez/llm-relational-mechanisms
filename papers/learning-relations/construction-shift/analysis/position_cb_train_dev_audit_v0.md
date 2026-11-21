# POSITION-CB-v0 train-pool versus development checkpoint audit

**Status:** completed read-only post-hoc study. Source: [all 16 audit records plus summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/analysis/position-cb-train-dev-audit/reports/position_cb_train_dev_audit_v0). Existing 32 model checkpoints were used only for inference; no optimization or new protected/reserved/L+I evaluation.

## Numerical reconciliation

The 16 individual audit records were fetched and checked against the frozen original study protocol digest `38585ea4493efe715bcb019c30d6b1aaaaf68fb87de0c6deb3a07d1bb171c2d7` and evaluation SHA `b8fc2d585bd573a8c24dc4018d41c07e0d75e0dcc28e6f71553b71d35e794de9`; all align. All eight seeds and two numerical repeats are present, and the direct raw-record reaggregation reproduces the committed summary. 144 selected source-pool indices and 144 never-selected indices per seed were compared in both training arms.

**Metric:** strict `C_query`: all four event/role questions answered correctly, no oracle correction. Each cell is the mean across eight independent training seeds, after averaging the two MPS repetitions per seed.

| Training arm | Actually selected train variants | Never-selected train-pool variants | Familiar development | Unused pool − dev |
|---|---:|---:|---:|---:|
| Same-voice AA+PP | 97.81% | 97.59% | 35.44% | **+62.15 pp** |
| Mixed-voice AP+PA | 91.95% | 91.73% | 38.54% | **+53.19 pp** |
| Pooled paired arms | **94.88%** | **94.66%** | **36.99%** | **+57.67 pp** |

The pooled paired unused-pool−development difference has an **exploratory, unadjusted seed-level t 95% interval [+32.52,+82.82] pp**. Per-question four-candidate accuracy likewise averages **97.63%** on selected training variants, **97.55%** on never-selected training-pool variants, and **58.29%** on familiar development (unused-minus-development **+39.26 pp**).

The observed loss on *unselected variants within the same pool* is only **0.22 pp** for each arm. This is much smaller than either arm's train-pool→development gap, though a near-zero difference does **not** establish formal equivalence.

## Key interpretive boundary

The unselected examples are **never-selected pool indices**, not independently held-out four-name sets or unseen event cores. The old training pool has 13,500 variants across 300 train-name sets. Each training schedule selects approximately 13,185–13,236 distinct variant indices and leaves only 264–315 unused, indicating very dense coverage. Each source core occurs with numerous modifier/frame variants, so an unused index may be a near-identical re-rendering of an already learned underlying event. The development data instead use disjoint **sets of four participant names** and fresh core configurations, but the individual names remain from the same name vocabulary.

Consequently, this audit **localizes a failure to transfer outside the original training-pool name-set/core distribution**, but cannot identify which shift matters. Possible explanations include core-level memorization, dependence on specific participant quadruples, overfitting to training-core permutations, position/name interactions, and brittle learned rules that work only on the dense training set.

Several low-development-accuracy models nevertheless score near perfectly on these train-pool contrasts; the weak familiar development behavior is not explained by simple failure to fit observed source variants.

## Better next experiment

Before additional optimization or any claim of a new grammar, use existing checkpoints in a **frozen, read-only within-grammar source-core-disjoint diagnostic**, with three tiers:

1. Train-selected source cores, evaluated using familiar voices.
2. New event-core permutations from the **same 300 train four-name sets**, not selected or rendered during the original training schedule; exclude all exact old core orderings and any generated text overlapping the training pool.
3. Development sources from **the 100 disjoint four-name sets** already evaluated, keeping query format, grammatical voices and modifier distribution controlled.

The key differences—(1)→(2), (2)→(3)—can distinguish sensitivity to *new event-core assignments within familiar name combinations* from sensitivity to *entirely new four-name combinations*. Fix how many examples are sampled, source parsers, hashes and exclusions before checkpoint scoring. This is a **post-hoc mechanism diagnostic** on previously trained models, not a prospective neural replication.

If tier (2) already collapses, focus on participant-role core memorization or combinatorial recombination and redesign training examples. If (2) remains strong but (3) collapses, prioritize train/dev four-name-composition shift and design a training balance intervention. If both transfer, inspect the source distribution of the original dev and conditioning effects. No original seeds or evaluations may be discarded based on performance.

## Paper-level consequence

The original POSITION-CB-v0 primary (+14.41 pp CI barely positive) and gated secondary (+1.60 pp) are still nominally passed as prespecified. They **must now be contextualized** by an even larger train-pool→development generalization failure. Describing the nominal passes as definitive abstract relational binding would overstate the data. The useful contribution is a controlled demonstration that balanced marginal positions do not guarantee reliable **within-grammar construction transfer**, together with a transparent account of the nuisance train→dev shift and unstable training outcomes.

**Cautions:** inferential intervals here are exploratory t intervals, not bounded probability intervals; their untruncated limits may exceed 0–100%. The sampled training examples and the development examples do not form a randomized treatment comparison. No new grammatical family or protected evaluation was accessed.
