# Relational mechanisms side strand — scientific decision tree (2026-11-23)

**Scope:** scratch-trained controlled Transformers, separate from the main pretrained-LLM programme. Do **not** optimize for a submission date. Old experiments are complete and frozen; do not rewrite their decisions.

## Established

- EXP-109: construction-dependent role errors with seed-level prospective test.
- EXP-114: early position and residual-update causal sensitivity, but no unique selective role circuit established.
- CORE-TRANSFER: large exploratory role/event reassignment failures on previously trained models; different error classes.
- CORE-DIVERSITY-v1: prospective diverse-core effect failed the +10pp threshold; simpler fixed-core training already transferred nearly perfectly.
- TRAIN-SURFACE-FACTORIAL-v0: prospectively passed the adjunct-associated averaged main effect (+29.86 pp on unseen role permutations, +29.37 pp gated withheld quartets); *exploratory* four-arm decomposition shows the combined prefix+adjunct condition has severe seed heterogeneity and predominantly wrong-event errors. No unique mechanism or confirmatory prefix×adjunct interaction established.

## Stage 1: Diagnose test-surface dependence (implemented)

Use the **existing** 12×4 sealed models with TRAIN-SURFACE-CROSS-EVAL-v0 on a frozen source-only subset: same semantic roles rendered as bare, prefix, adjunct and both; include familiar original-core test variants. Report per-item bare→decorated rescue/loss and paired prediction invariance, not just unmatched mean accuracies. Every one of the 12 seeds and all four arms must appear.

**Decision rules (scientific reasoning only, not a formal preregistration):**

- If combined-arm near-zero bare scores substantially recover on decorated versions of the *same heldout roles*, emphasize the **evaluation-surface / learned surface-conditional strategy** explanation. Do not call it proof of systematic role representations: they may still depend on fragile surface heuristics.
- If those heldout roles remain wrong on decorated variants but original familiar role cores are solved across all 45 training-eligible surface cells, emphasize **semantic recombination failure despite source-family competence**. Seek a causal mechanism for **event selection**, not a generic inference-format fix.
- If decorated familiar cores are ALSO poor, the old bare-only original-core scores overstated in-distribution learning; treat this as an optimization/capacity exposure problem before analyzing role circuits.
- If mixed patterns occur across seeds, stratify descriptively using *all* seeds and avoid fitting an inferential classifier to a researcher-selected threshold.

**No independent confirmatory claim can be obtained from this checkpoint-only diagnostic**, because the scientific question was motivated by the already-seen training cohort.

## Stage 2: Compare successful and failed representations without selection bias

After Stage 1 is sealed:
- Record learning curves from ALL 12 existing logs and final checkpoint norm/logit distributions, not only spectacular seeds.
- Probe whether *event identity*, *participant identity* and *agent/patient role* can be separately decoded across **unseen arrangements** from specified layer/token readouts. Use train-only source cores for probe fitting, disjoint role permutations and name quartets for probe evaluation, plus name/position-only controls. A high linear-probe score is information availability, not proof of use.
- Prefer **predefined token-span alignments** and surface-equivalent source pairs. A layer representation at token index 20 of a long prefixed sequence is not directly aligned with token index 20 of a bare sequence.
- Do not retroactively promote which layers/heads look compelling into confirmatory causal hypotheses.

## Stage 3: Test exactly one causal intervention

Choose a prediction **before** opening any additional intervention outcomes. Example: on seeds failing a role-recombined source, transplant the event-conditioned representation at a precisely aligned boundary from an otherwise semantically identical input with a different surface treatment. Compare a semantically matched donor versus a matched wrong-event or arbitrary-position control. Measure **wrong-event→correct** transitions, reversed-role changes, and effects on the other event; avoid mistaking general logit disruption for selective mechanism repair.

Use only original TRAIN-derived nonprotected sources, full seed reporting, immutable intervention design, independent controls and a separate analysis branch. If matched surface donor also fails, move to a training-distribution control rather than speculative head scans.

## Stage 4: New training only for a sharper falsifiable explanation

If needed, train fresh-seed interventions that separate:
(1) frequency of unmodified bare source exposure,
(2) neutral-prefix positional shifts and length,
(3) name-bearing versus nameless adjuncts, and
(4) relative/absolute position effects.
Equal source counts and optimizer updates alone do NOT make token length, training compute or bare dose identical. Freeze matching audits and stop rules first. If a single-family mechanism holds, repeat it in one genuinely different grammar before claiming generality. Preserve relation to the pretrained-LLM project as a hypothesis until matched behavioral and causal signatures are observed there.

## Keep the two strands complementary

Scratch model: control **learning distribution**, inspect mechanistic alternatives.
Pretrained LLMs: test **representational and behavioral transfer** in non-toy settings.
A shared error signature is not automatically a shared circuit. Do not pool seed-level uncertainty across these different populations.
