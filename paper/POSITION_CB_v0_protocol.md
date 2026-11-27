# POSITION-CB-v0: matched voice co-occurrence under balanced role positions

**Status:** pre-execution protocol and prepared runner, not yet run. This is a prospective within-project experiment on the existing two-event grammar, **not a second independent grammar family** or external preregistration. The scientific protocol is frozen when the generated audit manifest is committed before any model inference.

## Hypothesis and outcome

We examine whether a scratch-trained Transformer generalizes from familiar *pairs of clause voices* to unseen pairs after balancing the role of the first main argument at **both** event positions. The manipulated variable is the two-event **within-source voice co-occurrence**.

At each optimizer update, draw eight unique training cores. Each core contributes two sentences in each training arm, with equal gold labels and identical aggregate token bags:

- `same_voice`: **AA + PP** (first and second event both active, then both passive).
- `mixed_voice`: **AP + PA** (the two events have opposite voices).

A means active, P passive. Because each underlying event appears in active and passive form once per core in either arm, the first main-argument position is agent 50% and patient 50% in **each event position** in both arms. Eight pairs = sixteen sources = 64 role-question labels per update. All 13,500 potential dose cores are checked for exact token-multiset and label equality before training.

**Important:** this does *not* make individual sentence lengths, absolute token positions or within-source voice conjunctions identical. Those are remaining treatment-associated differences; the exact matching claim concerns aggregated source-core token multisets, gold labels and marginal main-argument role/voice at event positions.

**Primary frozen directional test:** per seed and averaged across both arms, mean `C_query(familiar voice pairings) - C_query(unseen voice pairings)` on the four questions per development source, where C_query requires *all four correct*. For `same_voice`, familiar=(AA,PP), unseen=(AP,PA). For `mixed_voice`, familiar=(AP,PA), unseen=(AA,PP). Success: lower bound of two-sided 95% Student-t CI >0 across eight independent seeds, with two MPS repetitions averaged per seed.

**Gated secondary test:** difference `complete agent/patient reversal rate(unseen) - reversal rate(familiar)` averaged over both original event positions and both arms. Same CI rule; only tested inferentially if primary succeeds. All arm-specific, question-level, full-vocabulary, wrong-role, wrong-event and same-person outcomes are exploratory/descriptive, and no multiple comparison adjustment is claimed for them.

## Fixed training design

Three-layer, width-96, four-head pre-LN causal Transformer; identical initial weights across arms *within seed*. Same optimizer, vocabulary, 6400-step endpoint, and full-vocabulary supervision as EXP-109/114. New independent seeds `12701,12703,12707,12709,12713,12719,12721,12727`; two Apple MPS numerical repetitions each; two arms per repetition: **16 scientific training processes, 32 model states**. No competence filtering, checkpoint selection, stopping after favorable results or discretionary seed replacements. Schedule balances 45 modifier/frame strata, sampling 4 distinct cores per parity group per step. Each schedule is hashed before any training.

## New development evaluation

Use only 100 non-reserved development four-name sets and original bare/L/I modifier family. Draw 144 fresh semantic cores (16 per each of the nine modifier combinations), each with AA,PP,AP,PA, total 576 surface presentations. Exclude previously examined legacy/EXP-109/EXP-114/PHASE2-01/02/03 source surfaces and core orderings; reconstruct and check existing stored evaluation hashes. Verify gold labels with the independent parser and unchanged per-core token bags; ensure distinct role participants, vocabulary coverage and the 48-token cap. All 95 reserved identifiers and within-event L+I remain closed. Save only evaluation digests and aggregate metrics, no item-level source strings or predictions.

## Workflow and integrity gates

1. Run `prepare`. It validates all training-pair token bags/labels, checks new non-reserved evaluation metadata (without inference), produces all eight schedule SHA256 hashes, and writes `reports/position_cb_v0/manifest.json` once.
2. Run `smoke`. It performs two training updates on random models and scores **only previously explored** EXP-109 development examples. No fresh evaluation outcome is seen.
3. **Commit the runner, launcher, protocol, and generated manifest**. Before any real run the runner verifies each of these four exact files equals the committed version at Git HEAD.
4. Run all eight seeds × two MPS repetitions using the launcher. No evaluation accuracy printed or inspected until 16/16 runs. Each successful process saves one local-only `.pt` checkpoint and one sealed JSON report; partial report/checkpoint pairs block automatic retry.
5. `summary` requires 16/16 complete report/checkpoint pairs with matching digests, then computes the two prespecified CI gates, all descriptive error classes and outputs `analysis_summary.json`. Report positive, negative, and null effects equally.

## Interpretation and stopping

A primary-positive result would establish voice-co-occurrence dependence despite equal core-argument role/position marginals, *within this synthetic two-event grammar*. It would not establish a unique role-binding circuit, new-grammar transfer or absence of more sophisticated position-based heuristics. A primary-null result would constrain the scope of this shortcut hypothesis, not prove abstraction. The role-specific secondary remains gated. Do not change training arms, confidence intervals, evaluation partitions or seed count after seeing scientific results.
