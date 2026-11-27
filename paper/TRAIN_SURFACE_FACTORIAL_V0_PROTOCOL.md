# TRAIN-SURFACE-FACTORIAL-v0: prospective four-arm training-distribution experiment

**IMPLEMENTED, NOT YET FROZEN OR RUN.** Freeze/commit source and study manifests before any new-seed inference. Use only old TRAIN quartets (the exact 240/60 split already used by CORE-DIVERSITY-v1). Never generate original DEV, reserved/protected or LI texts.

## Question and randomized training arms

Does training-time named adjunct variation reduce transfer to new semantic-role configurations, potentially by encouraging shortcuts tied to name positions? Test a 2×2 factorial on the same underlying participant-role assignments:

- **bare:** always original bare/bare source with no prefix.
- **frame_only:** one of five neutral prefixes (including empty), bare/bare modifier.
- **adjunct_only:** no prefix, one of nine paired bare/L/I modifiers (including bare/bare; no LI).
- **frame_adjunct:** one of five prefixes × nine modifier pairs.

Every seed and arm sees the SAME 8 selected four-name quartets at each of 6400 training updates, with the same AP+PA voice presentations and 64 supervised answers. Assigned frame and modifier cells cycle uniformly across the 45 strata. Exactly the same 4 gold role labels and 2 source voices per quartet/step, the same initial weights within each independent seed, same 3-layer width-96 four-head causal Transformer, 48 input positions, AdamW lr=0.002 wd=0.01, no early stopping. 12 fresh independent seeds ×4 arms =48 scratch-trained models. Full-vocabulary CE training. One final checkpoint per arm; retain all seeds, no competence filtering.

**Matching limitations:** presence/absence of neutral frames and adjuncts necessarily changes token sequence lengths, positions, number of named mentions and training compute per update. These are intended factors and potential alternative explanations, not matched controls; the source-only audit records the distributions. The hypothesis was chosen after observing CORE-DIVERSITY-v1 and is prospectively new, not retroactively registered.

## Frozen test hierarchy

All model arms evaluated on exactly the SAME bare/bare no-frame surfaces with AP and PA voice options. Within all 240 training quartets, the 16 heldout semantic-role assignments per quartet are never used by any treatment. An additional 60 fully withheld four-name quartets each has 24 role assignments. Four named role answers are scored jointly, strictly.

- **Primary:** per seed adjunct-associated degradation on the 16 heldout role permutations, computed as 0.5×[(bare − adjunct_only) + (frame_only − frame_adjunct)] in strict four-query joint correct (\`C_query\`), averaging the two familiar voice presentations. Across 12 independent seeds, primary success requires ALL: two-sided t 95% CI lower bound >0, mean effect ≥+10 percentage points, ≥9/12 positive seed contrasts. These thresholds are frozen before scoring and cannot be lowered to rescue a null.
- **Gated secondary:** same adjunct contrast on all 24 role permutations for 60 withheld quartets. Only analyzed confirmatorily IF primary passes; secondary success requires two-sided t 95% CI lower >0.
- **Exploratory:** prefix-associated main contrast, factorial interaction, single-answer and full-vocab accuracy, original-core training competence, name-selection pair error types. These cannot rescue a failed primary.
- **Inferential guard:** summary refuses to compute primary before all 12 sealed seed reports with matching protocol and schedule hashes. Never select a preferred seed/checkpoint or tune after seeing test outcomes.

## Limits and reporting

A positive result could establish a causal *within-grammar training-distribution* effect of named adjunct exposure under this architecture/optimizer schedule. It does not identify a particular mechanism, distinguish all possible token-length or position mediators, or establish generalization to natural languages, pretrained LLMs or another grammatical family. A negative result is reported without modification, and provides an empirical constraint on the post-hoc hypothesis. All seed reports public; model tensor checkpoints stay local.
