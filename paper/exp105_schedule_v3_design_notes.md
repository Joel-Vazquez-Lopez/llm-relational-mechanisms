# EXP-105 candidate training schedule — design note (v3)

**Status: unfrozen, development-only.** This is a schedule audit, not a model-training result and not permission to access reserved, protected or withheld-composition evaluation.

## Scientific question

Under equal training budget, does relocating a passive construction from the first event (**PA**) to the second event (**AP**) in the exposure dose improve generalization to new participant combinations and held-out **within-event L+I** compositions? This is not a fully independent grammar replication; the generator belongs to the EXP-102 lineage.

## Candidate training schedule

- **Eight fresh paired initializations:** 10701, 10703, 10709, 10713, 10721, 10727, 10729, 10733.
- **Proposed pilot budget:** 3,200 optimizer updates (not frozen yet).
- **Each update:** 14 common-base sources (7 AA, 7 PA) and 2 exposure-dose sources (PA in control / AP in targeted arm), with all 4 questions per source. This makes **64 supervision labels/update**, comprising 56 common-base and 8 exposure-dose labels; exactly 16 of each query type.
- Both arms sample exactly the same common-source IDs, dose semantic-core IDs, names, answers, modifiers, and temporal frames at every update. They are initialized identically per seed, in a *future* trainer.
- The passive **count** is matched at **9 passive events per update per arm**; event position in the dose is intentionally different.
- In each complete 45-step block, 9 modifier cells × 5 frame types are evenly represented; dose samples are also balanced by core layout parity. Sampling is without duplicate surface sources inside one update, but with replacement across updates.
- For modifier cells with intervention-sensitive punctuation lengths, paired dose cells cancel the **token multiset per update**. Source-length distributions match over **each full 45-update block**, but not necessarily each individual update. This is a documented limitation, not a full input-position or contextual control.
- The schedule implementation uses no model weights, gradients, protected examples, or reserved name-set renderings. The script audits eight complete deterministic schedules and records SHA256 digests.

## What would still need to be frozen before confirmatory training

1. Model/vocabulary (including all temporal frame words), maximum sequence length, optimizer, learning rate, weight decay and device policy.
2. Data encoding, batch order and a joint cryptographic fingerprint of every planned update; the **training runner must reuse the audited scheduling function**, not separately reimplement the randomization.
3. Development-only competence gate and fixed evaluation time(s); decide in advance what a failed gate means, without inspecting protected syntax.
4. Primary held-out endpoint, uncertainty method, multiple-comparison policy, negative controls, and a separate genuinely independent grammar replication.
5. Verification that the intended AP phenomenon is not a surface-position-only heuristic, with a predeclared set of counterfactual tests.

**Immediate next step after v3 passes on the user's Mac:** inspect the audit and build a disposable, development-only smoke-training runner that consumes this exact candidate schedule. Do not open L+I reserved tests.
