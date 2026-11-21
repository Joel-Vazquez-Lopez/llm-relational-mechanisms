# Scope and decision log

**Frozen source:** `experiments/exp102-clean-design` at `df82ab3ad9db5e13db80cf47caf8f2142afa3dc7`. **New research branch:** `paper/construction-sensitive-roles-20261121`. Do not merge the experimental branch into `main` without resolving its divergent history. Order-to-Role is entirely out of scope.

## Central publication claim

In a restricted two-event synthetic grammar, scratch-trained Transformers can show a large, replicated construction/position-dependent relational failure despite literal clause content preservation. A significant portion of the observable error is role reversal rather than wrong-event choice. Early-layer positional and residual-update interventions influence answer selection on fresh training seeds and within-family development sentences, **but available interventions do not isolate selective role correction**.

## Questions and dispositions

1. **RQ1: Construction sensitivity?** Supported by EXP-109, within family.
2. **RQ2: Error type?** Supported descriptively by EXP-110; not a prospectively confirmed new primary endpoint.
3. **RQ3: Causal layerwise sensitivity?** Supported by EXP-114's three frozen tests and reciprocal direction; controls limit interpretations.
4. **RQ4: Unique role-specific circuit?** NOT established. EXP-115/116 negative/ambiguous controls must be retained.
5. **RQ5: Out-of-grammar/general architecture?** NOT tested. New grammar or model family needed for broader paper claim.

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
- [ ] Add simple lexical/absolute-position shortcut baselines or sharpen alternative explanations.
- [ ] Complete literature and novelty review; select appropriate venue.
- [ ] Independent grammar study **only if** claiming transfer beyond this grammar.

## Conditional new study (EXP-117 concept, NOT preregistered)

Freeze a genuinely new grammar (not merely more seeds or names); prespecify PA–AP semantic contrast and one intervention predicted to increase correct-role answers without increasing wrong-event errors; include position-only heuristics and negative controls; use fresh training seeds and dev/test partitions. Do not select best head, seed, checkpoint or grammar after seeing results. If the study fails, restrict final paper to this well-characterized grammar.
