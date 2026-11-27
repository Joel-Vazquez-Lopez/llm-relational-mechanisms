# ROLE-ANCHOR-PROBE-v0 — original role-mapping audit

**EXPLORATORY, READ-ONLY**: use the frozen 12-seed, four-arm TRAIN-SURFACE-FACTORIAL-v0 checkpoints, with the already-sealed TRAIN-SURFACE-CROSS-EVAL-v0 source sample. No training, no original DEV, reserved, protected or L+I.

## Main falsifiable hypothesis

Failed models may predict each familiar quartet's **original trained four-role answer tuple** even when a new sentence assigns those same four names to different roles. Distinguish this narrow hypothesis from other rigid strategies such as repeatedly predicting some *other* tuple or simply failing to select the correct event.

Evaluate the same 24 sampled TRAIN-derived quartets, each with all 16 never-trained role permutations, AP and PA familiar voices and four matched input surfaces: bare, prefix, adjunct and combined. Exact source and sample ordering are taken from frozen CROSS-EVAL-v0, with source-only manifest and original study protocol hashes verified.

For every semantic item, compare the four restricted-choice predictions to BOTH (a) its current true role assignment and (b) the original trained role answer tuple in the canonical first-agent, first-patient, second-agent, second-patient question order. The original tuple is never equal to the fully correct tuple on these 16 withheld assignments.

## Frozen descriptive metrics

- Exact match with all FOUR original trained role answers (\`matches_original_role_tuple_exact\`);
- Fraction of individual answers matching their original training-role slot (\`matches_original_role_slots\`);
- Predicted person belongs to the originally trained event pair (\`answers_in_original_event_pair\`);
- Correctness against the *new* roles (\`strict_cquery\`, \`accuracy4\`);
- Per quartet, the fraction of its 16 heldout permutations receiving the most frequent prediction tuple, and the number of distinct tuples;
- Number of quartets whose most frequent prediction is *precisely* the original trained tuple.

Report **every seed, every training arm, both voices and all four surfaces**. No seed selection, no new claim of independent replication, and no retroactively confirmatory test. Comparisons use descriptive seed-level t intervals only. A uniformly wrong but non-original tuple is evidence *against* exact original-role anchoring and cannot be relabeled as the same mechanism.

## Source and execution safeguards

The scorer uses the CROSS-EVAL-v0 frozen source manifest and parent vocabulary, requires original final-step checkpoints with correct seed and parent protocol digest, loads state_dicts with \`weights_only=True\`, and performs only \`@torch.inference_mode\` forward passes. A missing .pt triggers an error, never retraining. Refuse overwriting any seed result. Freeze scorer+protocol in a Git commit before opening checkpoints. Aggregate only after 12 reports. Never upload original weights.
