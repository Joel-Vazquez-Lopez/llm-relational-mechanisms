# CORE-DIVERSITY-v1: independently verified prospective intervention

**Status:** complete, 16 new independent seeds, four arms per seed, 64 newly trained models; frozen prospective endpoint and 10-percentage-point criterion **not passed**.

Data: [16 per-seed run reports, source manifest, study manifest, and immutable analysis summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/design/position-cb-core-diversity-v1/reports/core_diversity_v1). The 16 reports were fetched separately; every protocol digest, frozen schedule hash and no-protected-data declaration matched the committed manifest; direct paired-seed reaggregation exactly reproduced the released primary and secondary means.

## What was compared

Each seed trained identical-initialization 3-layer causal Transformers in four conditions: same_voice_fixed, same_voice_diverse, mixed_voice_fixed, mixed_voice_diverse. All arms used 6,400 optimizer steps, 240 training four-name quartets, *bare/bare* source forms, and controlled two-voice pairings. In fixed-core arms, each quartet always had its original agent/patient assignment; in diverse-core arms, each quartet cycled eight counterbalanced assignments. Primary test used the other **16 never-trained assignments per training quartet**. Gated secondary tested **all 24 assignments from 60 four-name quartets withheld from both treatments**.

### Absolute performance is nearly at ceiling

Strict all-four-question \`C_query\`, seed-level means:

| Training regime | Training original core | Unseen assignments on training quartets | Unseen quartet combinations |
|---|---:|---:|---:|
| Same-voice, fixed-core | 100.00% | **98.94%** | **99.28%** |
| Same-voice, diverse-core | 100.00% | **99.998%** | **100.00%** |
| Mixed-voice, fixed-core | 100.00% | **98.03%** | **98.49%** |
| Mixed-voice, diverse-core | 100.00% | **99.95%** | **99.96%** |

The intervention is not the only effective way for these small models to achieve excellent within-grammar role recombination: **the fixed-core baseline already generalizes at roughly 98–99%.** This nearly saturated baseline leaves little headroom for the prespecified **+10 pp** effect.

### Frozen inferential decisions

- **Primary:** averaged diverse-minus-fixed strict joint \`C_query\` on never-trained role assignments, paired within each seed and averaged across same/mixed voice: **+1.4884 percentage points**, 95% seed-level t CI **[+0.4976,+2.4793] pp**, 12/16 independent seed differences positive. Frozen success required simultaneously lower bound above 0, mean gain ≥10 pp, and ≥12/16 positive seeds. **Primary FAILS because +1.49 pp is below the frozen +10 pp threshold.** Do not call it a successful intervention by redefining the threshold.
- **Gated secondary:** same contrast on withheld four-name quartets: **+1.0948 pp**, exploratory CI **[+0.3578,+1.8319] pp**, 11/16 positive seeds. **NOT tested confirmatorily** because primary gate closed; do not promote its nominal CI.

All original-core training evaluations were at 100% strict joint accuracy for all 64 models. No competence exclusions, performance-based checkpoint selection or changes to the prescribed seeds/analysis were made.

### Reconciling with the older dramatic failures

The previous POSITION-CB-v0 models had >90% correctness on repeated original cores but only roughly **32–37%** on new role assignments, event swaps or participant repartition, including an exact **bare/bare word-bag control**. The new CORE-DIVERSITY-v1 fixed-core models achieve **98–99%** on even disjoint new roles and disjoint four-name quartets.

**These are different training cohorts with different training distributions and implementation conditions.** In particular, the earlier models were trained with modifier/adjunct variation and multiple sentence-prefix frames; CORE-DIVERSITY-v1 used only bare/bare sources. It also changed the set split (300 original train quartets vs 240), sampling schedule, source diversity, and training implementation. We cannot causally attribute the large across-cohort difference to any single factor. Earlier poor performance on bare/bare *test* items indicates that lexical distractors need not be present at evaluation time to observe the failure; it does not isolate the training-time factor.

**Proposed next causal question:** holding source-core selection, quartet splits, model implementation, training seeds, optimizer, questions, and evaluation fixed *within the same new study*, does introducing neutral sentence frames and named adjuncts during TRAINING change generalization to unseen semantic roles and withheld four-name combinations? This is a new, independent hypothesis generated after observing CORE-DIVERSITY-v1 and cannot be tested confirmatorily by reinterpreting this cohort.

### Paper-level claims

Supported:
1. Source/grammar/task conditions strongly affect apparent systematic generalization of small scratch-trained Transformers; the current dataset includes both severe role failures and near-perfect systematic transfer.
2. In the old cohort, core transformations lead to **qualitatively different** reversal vs wrong-event error profiles (post-hoc).
3. A frozen, 64-model intervention using extra core diversity did **not** clear its predeclared practically meaningful +10 pp threshold, even though the effect was descriptively small and positive.

Not supported:
- A unique agent/patient role-binding circuit.
- Role-core diversity as an **essential** ingredient for generalization.
- Proof that modifier or prefix **training** variation caused the old failure.
- Broad claims about pretrained LLMs or natural language.

**Recommended next step:** one prospective surface-factorial experiment, with independent seeds and the same code path, before considering a main-conference submission. A narrower, transparent single-grammar empirical workshop paper is a defensible lower-cost option.
