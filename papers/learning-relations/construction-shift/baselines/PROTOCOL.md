# Positional-shortcut audit v0 — prepared before execution

**Status:** executable, read-only, post-hoc descriptive baseline study. No new neural training, inference or held-out confirmation. Existing frozen EXP-109/110 reports remain unchanged.

## Question

Can a frequency-learned *position-only* lookup reproduce the AP versus literal PA construction gap observed in the PA-only scratch-trained Transformer condition? Does it reproduce the **joint error taxonomy**, or only the marginal construction gap?

## Frozen comparison design

Use original EXP-105 non-reserved training name sets to compute a source-pool **expected exposure** prior for three training schedules: PA-only (7 AA + 7 PA shared + 2 PA dose), indirect PP+AA (same shared core + PP and AA dose), direct PA+AP (same shared core + PA and AP dose). Weight each source uniformly inside its stratum, with total expected counts per update as specified. This is *not* the realized 6,400-step seed-specific sampling trajectory.

Two fixed deterministic baselines:
1. **Raw mention index**: condition on event index, queried role, and number of name mentions within the clause. Predict the mention position most associated with the queried gold role in the training pool. Includes modifier distractors; no `was`, `by`, verbs, voice or semantic roles are used at prediction time.
2. **Core participant slot (optimistic oracle segmentation)**: first remove syntactically known `near NAME`, `By NAME's gate` and `beside NAME's gate` adjunct names to reveal precisely two main-argument name slots per clause. Learn the modal core slot per event and role from training. Again, no voice features determine which slot is agent, but identifying *core participants* uses hand-coded knowledge of modifier structure. **This is a generous upper-bound on slot-only behavior, not a deployable or fair capacity-matched neural baseline.**

Both baselines use the original visible semicolon event boundary and the question's first/second event index. Tie-breaking is always the lowest source-mention index; tied distributions must be disclosed.

Freeze all evaluations to EXP-109's original 144 AP development anchors and the exact five predefined transformations. Extract the original `exp109.evaluation()` and development-selection functions from the archived research scripts, verifying script SHA-256 values and the pre-existing frozen evaluation digest `5063bec0fceca42ce9a899ffeaa93c0e26a56b93734e80d5388e211e7c9e0b76`. Do not construct or score reserved, protected or within-event L+I samples. No source sentence strings or per-example predictions are written.

For each condition report native four-name question accuracy, four-query joint accuracy, per-question wrong-role/wrong-event rates, and mutually exclusive **second-event pair** classes: both correct, reversed, collapsed, wrong event. The two baselines cannot cause wrong-event predictions in the generous core-slot variant because its output is restricted to known core names of the selected event; that is a design advantage, not a mechanistic explanation of the neural results.

Compare descriptively with the unmodified EXP-109/110 archived PA-only neural model averages. Do **not** pool deterministic baseline outcomes with eight neural training seed repeats to form a new inferential test. Results must be reported even if they fully explain the construction gap or show nontrivial limitations.

## Guards and execution

The baseline must pass its built-in 72 source fixtures across all four voice pairs, nine modifier pairs and two layout patterns, plus original script SHA checks and the original EXP-109 evaluation SHA check. The exact analytical script and this protocol are committed on the separate manuscript branch before execution.

To run locally:
```bash
python papers/learning-relations/construction-shift/baselines/shortcut_baselines.py --emit-json
```
The script uses Python standard library only plus original repository pure-Python generator; no PyTorch installation is necessary. The CI runner is intentionally scoped to file changes under this baseline folder and prints only aggregate JSON results to its job log.

## Interpretation and stop rule

- A large PA−AP baseline gap shows a *possible shortcut account* for construction sensitivity; it does not demonstrate the trained model actually implements that specific rule.
- Matching reversal rates but not wrong-event/collapse rates establishes only partial explanatory coverage.
- The optimistic core segmentation is known in advance to overestimate what a blind positional heuristic could extract from raw text.
- A failed shortcut comparison does not itself prove abstract semantic roles. Alternate heuristics might exist.
- Never adjust tie-breaking, training weights, feature definitions, test examples or report thresholds after seeing results. Any extension gets a new version and is exploratory.
