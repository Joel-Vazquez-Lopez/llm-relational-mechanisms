# TRAIN-SURFACE-FACTORIAL-v0: prospective 12-seed source and result audit

**Study:** 12 independent prospective seeds × 4 matched-initialization arms = 48 scratch-trained Transformers, final step 6,400, all reported. **Original frozen primary PASSED. Gated secondary PASSED.** All claims here are within one controlled synthetic two-event grammatical family; the secondary examines name-quartet combination generalization, **not an independent grammar**.

**Archive:** [source-only manifest, study manifest, all 12 per-seed raw JSON reports and summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/train-surface-factorial-v0/reports/train_surface_factorial_v0).

**Frozen study commit:** \`0dd417912377ee36ea6b7c9f349d9e6c460bb628\`. All 12 raw reports carry that same frozen commit and the same protocol digest \`e765a55d7f4bd5a8cb80a4a17a0bd34156ff7018a6e90968781b6722893c516f\`. Independently fetched and compared six frozen input files (preflight script, runner, launcher, protocol and both manifests): their content on the results branch exactly matches their content at the frozen commit. Every seed and seed-specific schedule hash matches the frozen study manifest. All reports declare no original development, reserved, protected or within-event L+I evaluation, and expose all four prespecified arms and populations. Independent recomputation from the 12 raw JSONs **exactly reproduces** the archived \`analysis_summary.json\` primary and secondary means/CIs.

## Design and crucial caveat

All arms use the SAME 240 TRAIN four-name quartets and fixed event-core assignments, selected at each of 6,400 optimization steps using the same schedule, two familiar AP+PA voices, 64 answer labels per update and paired seed initialization. A 2×2 experiment manipulated the TRAINING distribution:

- \`bare\`: no frame, no modifiers.
- \`frame_only\`: one of five neutral sentence prefixes; no modifiers.
- \`adjunct_only\`: no prefix; nine balanced pairs of bare/L/I modifiers, never within-event LI.
- \`frame_adjunct\`: crossed five prefixes with nine modifier pairs.

The shared evaluation on original training cores, **16 role permutations untrained in any arm** per familiar quartet, and **24 role permutations per each of 60 fully withheld quartets** always uses *bare/bare, no-prefix* examples. The original 100 DEV and 95 reserved quartet identifiers were not rendered.

Training arms differ, intentionally, in source length, frequency of name mentions, token positions **and relative exposure to the exact bare/no-prefix source family**. The 45-cell balanced schedule means fully unmodified source presentations are proportionally less frequent in the most complex treatment. The study thus identifies a **bundled training-distribution effect**, but not whether source length, relative exposure, distractor names, positional cues or their interaction is the learned causal mechanism. Similar ordinary in-distribution accuracy for complex treatment training sources was **not separately measured**: "train_original" refers specifically to bare/no-prefix examples with the trained original core.

## Frozen primary and gated secondary

The **preregistered seed-wise adjunct-associated degradation** is

\[
D_{\mathrm{adj}} = \tfrac{1}{2}\left[
(C_{\mathrm{bare}} - C_{\mathrm{adjunct}})
+ (C_{\mathrm{frame}} - C_{\mathrm{frame+adjunct}})
\right],
\]

where each \`C\` is strict native **four-query joint correct** \`C_query\`, averaged across familiar voices on the same heldout test examples. The 12 independent seeds, not the thousands of examples or 48 correlated models, are the inferential units. Two-sided Student-t 95% CI with df=11. The *conjunctive* primary-success gate required positive lower CI, mean ≥+10 percentage points and ≥9/12 positive seeds. The gated secondary required a positive CI lower limit on 60 unseen name quartets after the primary passed.

| Frozen comparison | Mean effect (pp) | Seed-level 95% t CI (pp) | Positive seeds | Decision |
|---|---:|---:|---:|---|
| **Primary:** adjunct-associated degradation on 16 untrained roles per training quartet | **+29.86** | **[+13.46,+46.26]** | **9/12** | **PASS** |
| **Gated secondary:** same contrast on 60 completely withheld quartets | **+29.37** | **[+13.56,+45.18]** | **9/12** | **PASS** |

All seeds, including three slightly negative effects and seed-specific severe collapses, are retained; no threshold was altered after outcomes. The primary CI is broad: important seed heterogeneity remains.

## Four-arm absolute accuracy

Strict \`C_query\` seed-level means:

| TRAINING condition | Original train core / bare test | 16 unseen role assignments | 60 heldout name quartets |
|---|---:|---:|---:|
| Bare | **100.00%** | **99.21%** | **99.43%** |
| Frames only | **100.00%** | **84.25%** | **85.05%** |
| Adjuncts only | **99.93%** | **97.01%** | **97.85%** |
| **Frames + adjuncts** | **99.86%** | **26.73%** | **27.90%** |

The sharply poorer heldout generalization is **not** accompanied by a comparable deficit on the tested familiar bare-core examples: even the most complex treatment has ~99.86% original-core accuracy. But training-original test scores do not demonstrate competence on every decorated training-source variant.

## Exploratory interaction: the dominant descriptive phenomenon

The frozen **primary** is the averaged main contrast across both prefix states. Inspecting the four arms reveals that its magnitude is **not** consistent with a large isolated adjunct-only loss:

- Bare − adjunct-only on unseen roles: **+2.20 pp**, exploratory 95% t CI **[−1.06,+5.46]** (7/12 positive). This alone offers little evidence for a large failure caused by adjuncts without prefixes.
- Frames-only − frames+adjuncts: **+57.52 pp**, exploratory CI **[+26.46,+88.58]** (9/12 positive).
- The **difference between those two adjunct penalties** is **+55.32 pp** (11/12 positive; exploratory unadjusted 95% CI **[+25.73,+84.91]**). Equivalently, the script's \`interaction\` convention (bare−adjunct_only)−(frame_only−frame_adjunct) is **−55.32 pp**.
- The seed-averaged prefix-associated main contrast is **+42.62 pp**, exploratory CI **[+22.27,+62.97]**; also driven by the combined condition.

These interactions and individual arm contrasts were **exploratory**, not components of the confirmatory hierarchical tests. The conclusion supported by the current data is: the bundled named-adjunct training factor robustly degrades generalization **averaged across prefix contexts**, with an unexpectedly strong apparent interaction between framing and adjunct exposure. A **fresh, independently preregistered interaction-focused study** would be needed for a separate confirmatory interaction claim.

## Paired error taxonomy

Exploratory native pair outcome proportions, event 1 and event 2 averaged equally, on the *unseen-role primary population*:

| TRAINING arm | Both roles correct | Complete agent/patient reversal | Same person twice | ≥1 wrong-event name |
|---|---:|---:|---:|---:|
| Bare | **99.60%** | **0.00%** | **0.30%** | **0.09%** |
| Frames only | **88.33%** | **2.19%** | **3.50%** | **5.97%** |
| Adjuncts only | **98.50%** | **0.00%** | **0.59%** | **0.91%** |
| **Frames + adjuncts** | **34.34%** | **5.54%** | **1.51%** | **58.61%** |

The combined condition is dominated by choosing participant names from the wrong event, not merely by swapping agent and patient roles within the correct event. These metrics are **event-pair fractions**, not strict all-four-query \`C_query\` and not the per-question wrong-event fraction from prior studies. That difference in denominators matters.

## Strong across-seed heterogeneity

For the combined frames+adjunct condition, strict joint accuracy on unseen roles is near **0% in eight of twelve seeds** (14901,14907,14909,14919,14927,14933,14939,14943), reaches **100% in three** (14903,14913,14921), and is **20.34% in one** (14931). Yet all twelve score at least **99.17%** on original bare-core training examples. This sharp bimodality is not to be hidden by the ~26.73% mean. The independent seeds also establish that a simple "always fails" claim would be false.

## Strongest claim, limits and follow-up

**Strongest supported claim:** in one controlled synthetic grammar and matched model family, manipulating the training distribution to include name-bearing adjuncts causes a sizable, reproducible degradation in unseen semantic-role and unseen-quartet generalization when averaged over two neutral-prefix regimes, even though evaluated original bare-core accuracy remains nearly perfect. A marked **exploratory interaction** shows that combining prefix frames and adjuncts, not adjuncts alone, is associated with the most dramatic brittle solutions.

**Not yet established:** a unique mechanistic shortcut/circuit, position-specific causation, reason for seed bimodality, degradation of performance on all rich training source forms, cross-grammar transfer of the phenomenon, pretrained LLM implications, or an independent confirmatory effect of the observed interaction.

**Most valuable next experiment (if a stronger paper is the aim):** one tightly focused fresh-seed interaction replication that fixes dose/length or systematically audits exposures; avoid another broad search over constructions or heads. If time/budget is limited, the now-confirmed bundled effect plus transparent exploratory interaction can already anchor a narrowly framed empirical workshop submission, supported by prior EXP-109 and EXP-114 prespecified findings.
