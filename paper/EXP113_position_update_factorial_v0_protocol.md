# EXP-113 — Factorial separation of positional code and Block-1 update in clause-state patching

**Status:** fixed exploratory specification v0, written before any EXP-113 analysis of the Mac checkpoints. **Not** a preregistered mechanistic confirmation, new generalization benchmark, or new trained-model sample. All outcomes are descriptive, post-hoc to EXP-109/110/111/112. This does not replace or revise their scripts, results, or interpretations.

## Why this study

EXP-109 found a replicated AP versus literal PA construction gap in PA-only models. EXP-110 showed a high rate of reversed second-event role assignments. EXP-111 showed that second-event role logits frequently favor the wrong participant in AP. EXP-112 demonstrated that replacing *source-clause* states can rescue some AP answers and disrupt familiar PA answers, but an aligned-versus-reversed hidden-state control did not demonstrate token-specific alignment at Blocks 1–2. Moreover, the aligned **embedding-stage** EXP-112 intervention moved the *same lexical tokens* from PA into AP, so it transplanted their **absolute positional embeddings**, rather than contextual semantic representations.

**Question for EXP-113:** In the EXP-112 Block-1 passive-clause intervention, how much of the observed change can be produced by transplanting the explicit absolute positional code alone, by transplanting the Block-1 update alone, or by transplanting both? Does targeted position-code replacement at the input depend on which passive-clause tokens are changed?

## Immutable sources and unit of analysis

- All **48** existing EXP-109 final-step model states (8 independent seeds × 2 numerical MPS repetitions × 3 arms). No retraining or preferential checkpoint/model selection.
- Exactly the already-evaluated **144 AP anchors and paired literal PA clause swaps**. No new source distribution and no test-set opening. Both directions included: `AP <- PA_literal` and `PA_literal <- AP`.
- The exact frozen EXP-109 manifest, reported native outcomes, checkpoint hash, and relevant EXP-112 report must be verified for each run. The runner **refuses overwriting** any existing output.
- The gold labels are used to calculate margins and accuracy but **never** to construct an intervention or select an example.
- Only source tokens are altered. No query suffix, answer token, donor answer state, gradient update, participant label, or model weight is changed. Nothing from reserved/protected or L+I examples is generated or scored.
- Independent unit for descriptive intervals: **seed**; two numerical repeats averaged within each seed; 95% Student t intervals (7 df) are descriptive and not adjusted for exploratory multiplicity.

## The exact algebra

For a clause token *i*, write the answer-independent input representation as:

\[
B_i = E(t_i) + P(i).
\]

Let the residual representation after Transformer Block 1 be \(h_i^{(1)}\), and define the **total first-block update** as:

\[
U_i = h_i^{(1)} - B_i.
\]

For a matched recipient AP token *r* and donor PA token *d*, lexical identity \(t_r=t_d\) is asserted. Block-1 interventions are:

| Mode | Recipient Block-1 state before continuing Blocks 2–3 | Interpretation |
|---|---|---|
| Native | \(E(t_r)+P(r)+U_r\) | No change |
| `late_position_passive` | \(E(t_r)+P(d)+U_r\) | Swap **explicit positional-code component**, hold already-computed recipient update fixed |
| `late_update_passive` | \(E(t_r)+P(r)+U_d\) | Swap **learned first-block update**, retain recipient explicit positional code |
| `late_full_passive` | \(E(t_r)+P(d)+U_d = h_d^{(1)}\) | Reproduce EXP-112 full Block-1 passive patch |

These definitions create a 2×2 *intervention factorial at one depth*, rather than a comparison of incomparable checkpoint families. A change in the full patch minus the sum of individual changes is a **nonadditive outcome interaction**, not evidence for a unique representational interaction.

Crucial limitation: \(U_d\) was *computed using the donor's position and preceding context*, so `late_update` is **not** “pure, position-free semantic content.” It is merely the residual update after subtraction of the explicitly identifiable embedding component. Whole-clause state transfers can also create off-manifold combinations.

A separate `input_passive_all` mode replaces only \(P(r)\) with \(P(d)\) *before Block 1 is computed*, allowing all subsequent updates to respond. This exactly reproduces the **aligned EXP-112 embedding-stage patch**. It is not new independent replication. `input_active_all` is the matched-clause control.

## Additional position-code controls

At the **input only**, split the passive-clause lexical token positions into mutually exclusive sets:

- `input_passive_core_names`: the grammatical passive patient immediately before *was* and passive agent immediately after *by*. **Two** token positions per source.
- `input_passive_voice_cues`: *was*, the immediately following verb, and lowercase *by*. **Three** positions per source.
- `input_passive_other`: all remaining tokens, including positional/distractor modifiers and punctuation *within* the passive clause (but not the semicolon separating clauses).
- `input_passive_all`: replace all three groups jointly.
- `input_passive_reverse`: replace positional embeddings for all passive tokens, but reverse the **within-clause mapping of donor position embeddings**. The *original recipient lexical token embeddings remain unchanged*. This differs critically from EXP-112's `passive_misaligned` hidden-state swap, which also changes lexical token embeddings when donor/recipient tokens differ.
- `input_active_all`: matched active clause positions, negative/disruption control.

The groups are not matched in cardinality or salience; differences in effect **cannot** uniquely identify a causal syntactic category. Their joint effect need not equal the sum of the separate interventions because the model is nonlinear.

At **Block 1**, also test `late_position_active`, `late_update_active`, `late_full_active`, `late_position_passive_reverse`, and `late_update_passive_reverse` as controls. An EXP-112 `block3` source-only intervention is already known to have no effect once the answer representation has been computed; this study does not rerun that trivially null condition.

## Exact planned measurements

For **both directions**, all four query types, all 144 sources, and all 48 models, report changes from each unpatched recipient:

1. Native four-name candidate-restricted accuracy for each query type, four-query C_query, plus wrong-role and wrong-event rates.
2. **Correct-role logit margin**, `logit(gold participant) − logit(opposite-role participant of the same event)`. A positive margin is insufficient for native correctness when wrong-event names win.
3. AP **second-event** agent/patient changes and collateral AP **first-event** changes; reciprocal PA **first-event** changes.
4. A predeclared summary of seed-level means/intervals for each mode, and five *post-hoc exploratory contrasts*: `late_full − late_position − late_update`, `late_position − late_update`, `input_all − late_position`, `input_all − (names + cues + other)`, `input_aligned − input_reversed`.

**Reading rule:** Do not turn effects discovered in this repeatedly examined development set into independent confirmation. Even if a descriptive confidence interval excludes zero, the inference is to these models and already-explored data. Do not filter out low-competence numerical repeats.

## Strong integrity checks

- Recreate only EXP-109 familiar-development examples and check their frozen digest/vocab hash and all canonical AP↔PA lexical token alignments.
- Verify partition into passive core names / voice cues / other for every example and voice construction; all map positions are within the source prefix.
- Verify EXP-109 original native metrics for each checkpoint.
- **Reconcile exact aligned intervention outcomes** against the saved EXP-112 file in both directions: `input_passive_all` ↔ `embed__passive`, `input_active_all` ↔ `embed__active`, `late_full_passive` ↔ `block1__passive`, `late_full_active` ↔ `block1__active` (allow only tiny numerical tolerance).
- Do **not** compare the reversed-position-only condition against EXP-112's reversed hidden-state control: those perform different interventions.
- `--phase smoke` on a random model and earlier PA development texts only. `--phase summary` requires all 48 model files and refuses mixing script hashes.

## What would change our scientific interpretation?

- **Explicit-position-driven signal:** `late_position_passive` reproduces a substantial, selective part of the Block-1 effect, while `late_update_passive` is weak; similarly, `input_passive_all` rescues AP with relatively limited collateral damage. This would show causal sensitivity to *explicit positional-code differences*, **not** prove the full model is a position-only heuristic.
- **Learned-update-driven signal:** `late_update_passive` reproduces much more of the full effect than the explicit-position component. That would justify targeted examination of attention and MLP updates, but the update itself still encodes positional context.
- **Interaction-driven or neither:** both component patches are weak, opposite-signed, or noisy while the full patch is large. That would suggest nonlinear dependence on the combination or off-manifold confounds, and weaken a simple attribution.
- **Nonselective positional disruption:** active-clause, reversed-pos, or non-core-token substitutions yield comparable changes or increase wrong-event errors. That would undermine a token-/role-specific account even if AP accuracy improves.

A future independent mechanistic claim requires freezing a specific prediction and testing it in **new model seeds and new permitted, non-reserved development-family examples**, with a more precise causal control. The existing protected/L+I tests remain unopened.

## Local execution

From `~/Developer/exp102-clean-design` with the same main-repo virtualenv used for EXP-109:

```bash
source ~/Developer/llm-relational-mechanisms/.venv/bin/activate
unzip -n ~/Downloads/exp113_position_update_factorial_v0_bundle.zip -d .
python3 scripts/exp113_position_update_factorial_v0.py --phase prepare
python3 scripts/exp113_position_update_factorial_v0.py --phase smoke --device mps
# Ideally commit the EXP113 script/protocol BEFORE running actual checkpoints.
python3 -u scripts/exp113_position_update_factorial_v0.py --phase run --seed 10901 --repeat 1 --arm pa_only --device mps
bash scripts/run_exp113_all_v0.sh
# The launcher runs all remaining models and then --phase summary.
```

## Verification status before shipping

- Python syntax compilation: PASS.
- CPU smoke test on earlier PA development examples with random model: PASS.
- End-to-end CPU run using a *synthetic checkpoint and synthetic report* in an isolated fixture: PASS, including EXP-112 reconciliations, native consistency and fresh output. **No scientific EXP-113 outcomes have been computed on the user's Mac checkpoints in this preparation.**
