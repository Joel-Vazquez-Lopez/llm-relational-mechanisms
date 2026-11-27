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
- [x] Four figures generated from archived aggregate data.
- [x] Claim-level evidence ledger and provenance.
- [ ] Finish and verify full manuscript against code (draft in this package).
- [ ] Confirm per-run source/checkpoint hashes; clean machine environment file; make all plot generation reproducible from reports.
- [ ] Audit labels, figure typography and thresholds by independent review.
- [x] Run initial simple positional-shortcut audit (raw mentions and privileged core-slot comparator), matched to the exact EXP-109 development hash. It explains a large construction gap in principle, but not the neural error mixture. Further capacity-matched/learned shortcuts remain untested.
- [ ] Complete literature and novelty review; select appropriate venue.
- [ ] Independent grammar study **only if** claiming transfer beyond this grammar.

## Conditional new study (separate future protocol, NOT preregistered)

PHASE2-01–03 already investigate coverage, new adjunct orders and distractors. If a further study is justified, freeze a **genuinely new grammar or model family** (not merely more seeds, names or near-tail placement); prespecify PA–AP semantic contrast and one intervention predicted to increase correct-role answers without increasing wrong-event errors; include position-only heuristics and negative controls; use fresh training seeds and dev/test partitions. Do not select best head, seed, checkpoint or grammar after seeing results. If the study fails, restrict final paper to this well-characterized grammar.
