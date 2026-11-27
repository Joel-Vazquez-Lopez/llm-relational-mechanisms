# TRAIN-SURFACE-CROSS-EVAL-v0: read-only evaluation of the frozen factorial cohort

**Status: EXPLORATORY FOLLOW-UP — not an independent new training experiment and not a new confirmatory hypothesis test.** The question and the diagnostic subset were chosen AFTER examining the full 12-seed TRAIN-SURFACE-FACTORIAL-v0 bare-surface results. The scorer never updates weights and cannot access original DEV, reserved, protected or within-event L+I examples.

## Scientific decision

The original prospectively trained four-arm models were evaluated on **bare/no-prefix** source surfaces; the combined prefix+adjunct treatment demonstrated extremely heterogeneous unseen-role transfer (near zero in eight seeds; 100% in three). We must distinguish:

1. **Mismatched-surface explanation:** newly recombined roles are solved much more reliably if evaluated with familiar decorated frames/adjuncts. If so, the original bare-only test may have disproportionately exposed a surface-format mismatch.
2. **Transfer/relational failure:** the same unseen role rearrangements remain incorrect even when expressed using familiar rich surface variants. This suggests failure to generalize the semantic/event structure under the studied training distribution, but does not yet isolate a computational circuit.
3. **Both may contribute:** original role cores may be learned on all source variants while role recombination remains surface-sensitive. No single metric decides the neural mechanism.

## Frozen source construction before model inference

Reuse the original protocol-digest-verified 240 study TRAIN quartets and 60 withheld **TRAIN-derived** quartets. With selection seed 157021, sample exactly **24** TRAIN quartets and **12** of the previously withheld TRAIN-derived quartets. Do not render the original 100 DEV or 95 reserved quartets. Old three-layer model weights must come from the 12 previously sealed seed checkpoints of the 48-model train-surface factorial, loaded read-only. Never retrain a missing checkpoint.

- **Familiar-core control:** all 45 source variants (5 prefixes ×9 allowed modifier pairs) for the original fixed semantic core, two familiar voices (AP and PA), on the 24 sampled TRAIN quartets. These are **eligible source variants** under the old training distribution; some may never have been sampled in a specific stochastic run. The experiment did not record every training-schedule example explicitly.
- **Recombined-role diagnostic:** all 16 originally **never-trained** semantic role permutations for each selected TRAIN quartet; two familiar voices. Present each semantic/voice item in FOUR test-surface bundles: bare (fi=0,mi=0); prefix-only (fi=1..4,mi=0); adjunct-only (fi=0,mi=1..8); combined (fi=1..4,mi=1..8). For a given semantic item, one nonempty prefix \`fi=1+(index%4)\` and one nonempty modifier pair \`mi=1+((index//4)%8)\` is chosen in globally balanced, deterministic order. Thus every semantic target is paired across **identical source conditions except surface variation**, while all 4×8 nonempty joint cells are represented equally across the corpus.
- **New-quartet diagnostic:** all 24 role permutations for each of the 12 TRAIN-derived name quartets wholly withheld from the old 48-model cohort, paired into the same four test-surface bundles and two familiar voices.

Every source is independently parsed to the same exact four agent/patient labels across surfaces. Training and test texts are disjoint; role-only permutations are unseen in old training. All prompts fit 48 positions and use the original model vocabulary. The same semantic items and test surfaces are used for every model/seed.

**Compute:** this is a selected diagnostic subset, NOT a claim of full-population coverage. Each model evaluates **2,160 familiar-core source variants**, **3,072 recombined-role variants**, and **2,304 wholly withheld-quartet variants**, all with four role questions per source. Four arms ×12 independent seeds: **48 already-trained models**, no gradient steps. Keep original .pt weights unchanged and local.

## Endpoints and interpretation

- Strict four-answer joint correct (\`C_query\`), per-question correctness, full-vocabulary correctness, and mutually exclusive event-pair error classes on each test surface.
- **Same-item cross-surface transitions:** among role-recombined examples, fraction wrong on bare but correct on a decorated surface, fraction correct on bare but wrong decorated, and fraction with identical full native four-answer predictions. These are computed by paired item matching, not by subtracting unrelated marginals.
- Compare transfer under decorated surfaces with familiar-core accuracy on the matching source family, especially for the eight historically poor combined-treatment seeds.
- All four arms and all 12 seeds are evaluated and reported. Source selection is frozen before inspecting ANY new decorated test outcomes. Do not filter seeds for competence or choose an especially favorable prefix/adjunct template after seeing results.
- Descriptive seed-level intervals MAY be computed and labeled *exploratory*, but **no new primary p-value or independent confirmatory claim** is justified. A large bare→decorated rescue would be evidence for surface-dependent evaluation sensitivity; persistent failure despite high familiar-core scores would motivate a follow-up to isolate learned relational/positional mechanisms.
- The diagnostic cannot distinguish lexical distractor identity from absolute position and length, nor prove a unique attention head or semantic-role vector.

## Freeze and safety

First run source-only preflight and a random-weight smoke on ORIGINAL trained-role cores only, then commit script, protocol and the source-only manifest. Read-only scorer requires byte equality between those files and their HEAD blobs; it requires the unchanged original parent protocol and source/vocabulary hashes; and it refuses to overwrite an existing diagnostic seed JSON. Each loaded .pt must match the original seed and protocol digest, contain all four arms, and be at the final 6,400 updates. Summary will not aggregate until all 12 old seeds have sealed diagnostic JSON reports. Never open original DEV, reserved, protected or LI evaluation.
