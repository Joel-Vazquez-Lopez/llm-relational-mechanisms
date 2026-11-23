# Scope and decision log

**Frozen source:** `experiments/exp102-clean-design` at `df82ab3ad9db5e13db80cf47caf8f2142afa3dc7`. **New research branch:** `paper/construction-sensitive-roles-20261121`. Do not merge the experimental branch into `main` without resolving its divergent history. Order-to-Role is entirely out of scope.

## Central publication claim

In a restricted two-event synthetic grammar, scratch-trained Transformers can show a large, replicated construction/position-dependent relational failure despite literal clause content preservation. A significant portion of the observable error is role reversal rather than wrong-event choice. Early-layer positional and residual-update interventions influence answer selection on fresh training seeds and within-family development sentences, **but available interventions do not isolate selective role correction**.

## Questions and dispositions

1. **RQ1: Construction sensitivity?** Supported by EXP-109, within family.
2. **RQ2: Error type?** Supported descriptively by EXP-110; not a prospectively confirmed new primary endpoint.
3. **RQ3: Causal layerwise sensitivity?** Supported by EXP-114's three frozen tests and reciprocal direction; controls limit interpretations.
4. **RQ4: Unique role-specific circuit?** NOT established. EXP-115/116 negative/ambiguous controls must be retained.
5. **RQ5: Out-of-grammar/general architecture?** PHASE2-02 tested unseen adjunct orders within the same grammar and failed its frozen primary exposure-transfer test. A truly different grammar or model family remains untested.

## New executed post-hoc positional audit

- Frozen baseline script + protocol committed before GitHub Actions run #44898500105.
- Exact EXP-109 development SHA-256 and 72 fixture checks PASS.
- PA-only raw-name prior: +44.4 pp PA−AP strict C_query gap; generous oracle core-slot prior: +100 pp; neural mean: +61.4 pp.
- Neither baseline generates AP same-person collapses; the raw prior can make wrong-event responses by selecting distractor mentions.
- Do not claim the slot heuristic uniquely explains the actual neural mechanism or that failing to mimic all errors rules out other heuristics.

## Completed position-counterbalanced co-occurrence study

- New frozen 8-seed / 2-MPS-repeat / 2-arm study: AA+PP vs AP+PA on the same source cores with balanced per-event voice marginals and per-pair aggregate token bags; source-level co-occurrence is manipulated. The original two-event grammar is unchanged.
- Primary familiar−unseen strict C_query +14.4097 pp, 95% t CI [+0.0041,+28.8154] pp, nominal pass, 7/8 seed signs positive.
- Gated reversal increase +1.6005 pp CI [+0.0139,+3.1871], nominal pass, 7/8 positive.
- Strong instability: some arms near perfect and others near zero familiarity, 3 seeds contribute ~90.3% of summed effects; the paired arm interaction is inconclusive. Preserve all seeds; do not turn this into a strong robustness claim.
- For figures, include all eight seed-level paired traces and label confidence limits with four decimals. See `analysis/position_cb_v0_audit.md`.

## Completed post-hoc training-pool transfer audit

The follow-up run found same-voice familiar accuracy of 97.81% on selected training variants and 97.59% on never-selected training-pool variants, versus 35.44% on familiar development. Corresponding mixed-voice values: 91.95%, 91.73% and 38.54%. The pooled across-arm unused-pool−development gap is +57.67 pp (exploratory t CI [+32.52,+82.82]). **Unused pool variants may share entire source cores with previously sampled variants**; development uses fresh four-name sets. Distinguishing core memorization from four-name combination shift is the next priority, not a new head/patch sweep. No change to the original frozen inferential decisions.

## Completed core-transfer permutation diagnostic

All 16 original JSON reports and the frozen manifest are archived and independently reaggregated. Role flips, event swaps and participant repartitioning with familiar training name quartets cause paired strict C_query drops of approximately 59–60 pp compared with unchanged cores, in both voice regimes. The role-flip failures are substantially complete role reversals; swaps and new pairings mostly produce wrong-event names. The same effects persist in the bare/bare exact-token-multiset subset. **This is new post-hoc diagnostic information, not an independently prospective replication.** It does not isolate a unique circuit or prove word-level memorization. Full details: `analysis/position_cb_core_transfer_v0.md`.

## CORE-DIVERSITY-v1: additional completed prospective boundary

A new 16-seed / 64-model training study compared fixed-core versus eight balanced core assignments under AA+PP and AP+PA exposure with bare/bare inputs. Both fixed-core arms reached 98–99% on unseen role permutations/new four-name quartets, and diverse-core arms nearly 100%. Diverse-minus-fixed primary +1.488 pp CI [+0.498,+2.479] pp; **FAILED** prespecified +10 pp meaningful-effect criterion. The gated new-quartet secondary was not opened. This requires a different next hypothesis: surface-complexity/position effects in the TRAINING distribution, *not* a blanket inability to recombine roles after fixed-core training. Confounded cross-cohort observations cannot establish such a cause. Audit: \`analysis/core_diversity_v1_audit.md\`. Planning: \`PUBLICATION_ROADMAP_2026_11.md\`.

## TRAIN-SURFACE-FACTORIAL-v0: successful independent prospective causal-training-distribution test

A frozen 2×2 training intervention with **12 fresh seeds / 48 scratch models** compared bare, neutral-prefix-only, name-bearing-adjunct-only and combined prefix+adjunct training, holding original event cores, labels, quartet sampling and architecture fixed. **Primary PASSED:** averaged adjunct-associated degradation **+29.86 pp** 95% t CI [+13.46,+46.26], 9/12 positive seeds, exceeding preregistered +10 pp criterion. **Gated secondary PASSED:** +29.37 pp CI [+13.56,+45.18] on 60 unseen four-name quartets. The four-arm combined-condition collapse (26.73% joint vs 99.21% bare) and 55.32 pp descriptive interaction are exploratory; the training regime also changes token lengths, name frequencies, input positions and dose of unmodified sources. Eight combined-arm seeds near zero and three at 100% make heterogeneity central. This is one-grammar evidence, not a proof of a unique circuit. [Independent audit](analysis/train_surface_factorial_v0_audit.md).

## Additional prospective evidence (PHASE2-01–03)

- PHASE2-01: direct exposure 20% primary failed; full-dose gated secondary descriptively positive but not formally reached.
- PHASE2-02: unseen adjunct-placement transfer primary failed; not an entirely new syntax family.
- PHASE2-03: named-distractor relocation primary failed; later positive p100 tertiary is exploratory because the gate was closed.
- These results should appear in the main Results as boundary tests and in Supplement S1. Do not launch another within-grammar adjunct sweep just to obtain a positive result.

## Do not do

- Do not rename these exploratory analyses as replications.
- Do not report 16 MPS runs as 16 independent seeds.
- Do not infer a unique head from cherry-picked exploratory EXP-115 maps.
- Do not interpret incorrect-event transfers as selective semantic-role repair.
- Do not open reserved/protected or L+I splits.
- Do not start EXP-117 without a frozen design and separate held-out surface family.
- Do not edit OTR or existing frozen EXP reports.

## Submission gates

- [x] Four principal numerical stories present in archived reports.
- [x] Frozen within-grammar replication available for behavioral and intervention effects.
- [x] Four original main figures generated from archived aggregate data; the newer POSITION-CB result is included as Figure 5 with individual seeds.
- [x] Claim-level evidence ledger and provenance.
- [ ] Finish and verify full manuscript against code (draft in this package).
- [ ] Confirm per-run source/checkpoint hashes; clean machine environment file; make all plot generation reproducible from reports.
- [ ] Audit labels, figure typography and thresholds by independent review.
- [x] Run initial simple positional-shortcut audit (raw mentions and privileged core-slot comparator), matched to the exact EXP-109 development hash. It explains a large construction gap in principle, but not the neural error mixture. Further capacity-matched/learned shortcuts remain untested.
- [ ] Complete literature and novelty review; select appropriate venue.
- [ ] Independent grammar study **only if** claiming transfer beyond this grammar.

## Conditional new study (separate future protocol, NOT preregistered)

PHASE2-01–03 already investigate coverage, new adjunct orders and distractors. If a further study is justified, freeze a **genuinely new grammar or model family** (not merely more seeds, names or near-tail placement); prespecify PA–AP semantic contrast and one intervention predicted to increase correct-role answers without increasing wrong-event errors; include position-only heuristics and negative controls; use fresh training seeds and dev/test partitions. Do not select best head, seed, checkpoint or grammar after seeing results. If the study fails, restrict final paper to this well-characterized grammar.
