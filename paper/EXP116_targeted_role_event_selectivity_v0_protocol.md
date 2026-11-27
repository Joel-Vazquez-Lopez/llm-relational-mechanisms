# EXP-116 — Final exploratory test of targeted role rescue versus event confusion

**Status: Fixed exploratory protocol, version 0.** This is the planned final intervention experiment in the EXP-109–116 sequence, not an independent validation, externally preregistered study, or proof of a grammatical-role circuit. The EXP-114 trained checkpoints and 144 within-family development AP/PA examples have been inspected repeatedly. All diagnostic intervals are descriptive and unadjusted for multiple comparisons. **Never open the protected/reserved or L+I test gate.**

## 1. Motivation and falsifiable question

EXP-114 independently replicated a strong PA-only AP positional-input rescue (+29.23 pp), the Block-1 update-over-late-position advantage (+16.97 pp), and reciprocal PA deterioration (−34.27 pp). EXP-115 decomposed the Block-1 update and found that the attention and MLP residual contributions influence outputs, without localizing a reliable isolated head. Transferring the two core passive participant-name update states was close to transferring the whole passive clause. Yet those interventions also increased *wrong-event* decisions, and reversing source-token correspondence often preserved most of their effects.

**Question:** Is any preselected, *smaller* perturbation of the passive participant-name Block-1 updates capable of increasing AP second-event correctness **without** increasing wrong-event predictions? How do the passive agent and patient token states individually affect this trade-off? The study may return a clear negative outcome: role and event errors may remain coupled across all tested interventions.

## 2. Inputs, exclusions, and inference unit

- Use precisely the existing **48 EXP-114 final trained model states**: 8 seeds × 2 numerical MPS repetitions × 3 arms (`pa_only`, `indirect_pp_aa`, `direct_pa_ap`). No retraining, fine-tuning, weight updates, or checkpoint selection.
- Use exactly the EXP-114 **144 already-scored AP sources** and their literal PA counterparts from the same non-reserved development generator. No extra prompts, examples, names, modifiers, or out-of-grammar tests. The runner checks EXP-114 and EXP-115 report, manifest, evaluation, source and checkpoint hashes.
- Both `AP <- PA_literal` and `PA_literal <- AP` are evaluated for every model. Patch **source-clause states at the end of Block 1**. Leave all query and answer-token states untouched. Gold labels influence only metrics; they never determine which examples or tokens to patch.
- Independent numerical summary unit is **training seed**, with both numerical repetitions averaged within each seed; report n=8 seed-level values and descriptive Student-t 95% intervals, df=7. Preserve every model and baseline competence level.
- EXP-116 is **not** another confirmation of EXP-114. All interpretations are within the same small architecture, fixed synthetic grammar and previously explored source examples.

## 3. Exact perturbation

For a matched recipient source-token position `r` and donor position `d` (same frozen model, same underlying event), define the first-block residual update

`U(x,i) = [sum over four post-W_o attention-head contributions at i] + [post-attention MLP residual contribution at i]`.

At the end of Block 1, write

`h'_r = h_r + alpha * (U_d - U_r)`

for selected source tokens, then propagate the recipient model through Blocks 2–3 without altering the query or answer tokens. The donor update is computed using the donor source's original position and prior context; it does **not** isolate a position-independent grammatical-role variable. For `alpha=1`, this is exactly EXP-115's update-only patch on the corresponding selected tokens (numeric reconciliation required). For `alpha=0`, the original model is unchanged (identity guard).

### Frozen mode set (21 total)

| Token group | Alpha values | Interpretation |
|---|---|---|
| `passive_patient` | 0.25, 0.50, 0.75, 1.00 | The passive core patient token immediately before *was* |
| `passive_agent` | 0.25, 0.50, 0.75, 1.00 | The passive core agent token immediately after lowercase *by* |
| `passive_names` | 0.25, 0.50, 0.75, 1.00 | Both core participant-name tokens |
| `passive_all` | 0.50, 1.00 | Entire matched passive clause, comparison to EXP-115 |
| `active_names` | 0.50, 1.00 | Both *core* names in the matched active clause, non-target-event control |
| `passive_voice_cues` | 0.50, 1.00 | *was*, verb, and lowercase *by*, grammatical-cue control (3 tokens, not cardinality matched) |
| `passive_names_reversed` | 0.50, 1.00 | Deliberately swap the two donor core-name token positions, an off-manifold ordering control |
| `passive_names` | 0.00 | Identity/no-op control |

The deterministic name locator excludes incidental name mentions in gate or near-person modifiers. The patient/agent labels are derived from the grammatical template **before** any predictions, not from successful or failed responses. There is no adaptive dose choice, posterior model selection, or post-hoc deletion of conditions.

### Mandatory numerical and provenance guards

1. Reconstruct the original 144 EXP-114 sources and both recipient variants; verify known frozen hashes.
2. Check decomposed Block 1 against the original network and native predictions against the EXP-114 report.
3. For every model, both directions and all four questions, verify `passive_names_a100` agrees numerically with EXP-115 `passive_names_update_all`, and `passive_all_a100` with EXP-115 `passive_update_all`.
4. Confirm `passive_names_a000` reproduces the identity output. Verify transition-rate accounting exactly exhausts three answer categories.
5. Prepare and freeze the runner, protocol, launcher and immutable manifest in **one Git commit before scoring**. Each run rechecks the frozen file bytes. Refuse overwriting existing reports and changing previous EXP-114/115 inputs; generate the summary only when all 48 model reports exist.
6. Persist aggregated numerical outcomes and integrity metadata only. No source sentence strings, token-level activations, per-example predictions, reserved exemplars, or training checkpoints in Git.

## 4. Prespecified outcome reporting

For every model, mode, direction and four query types, compute native four-candidate correctness; same-event opposite-role errors; other-event errors; correct-role logit margin. For each query, classify the native and patched predictions as exactly one of:

- `C`: correct participant;
- `R`: the opposite participant **of the correct event** (wrong role);
- `E`: participant of **the other event** (wrong event).

For each of the six nontrivial transitions `R→C`, `R→E`, `E→C`, `C→E`, `C→R`, `E→R`, record rates as proportions of *all* queries in that query type. Same-class transitions are implicit and are checked via conservation. Avoid choosing or filtering an intervention based on gold error status; only the **analysis** examines the resulting transition categories. The AP second-agent and second-patient questions are weighted equally in the primary descriptive display; reciprocal PA first-agent/patient are also displayed. AP first-event collateral accuracy remains recorded. Record all three arms and all doses.

### Stringent descriptive selectivity screen — NOT a significance test

For each of the 12 predeclared single-agent, single-patient and two-name dose modes in the **PA-only AP** focus, label a mode *descriptively selective* only if all hold:

1. Lower seed-level 95% CI for accuracy improvement is strictly positive.
2. Upper seed-level 95% CI for wrong-event change is ≤0.
3. Lower 95% CI for `R→C` exceeds the upper 95% CI for `R→E`.

This is deliberately conservative and not a formal equivalence or multiple-comparison-adjusted test. **No qualifying mode is an acceptable outcome**: conclude that selective rescue was not demonstrated under the tested interventions. A qualifying mode is a *new exploratory candidate* requiring independent confirmation before any mechanistic claim. The original EXP-114 positive replication remains valid regardless.

Descriptive contrasts to inspect without post-hoc hypothesis promotion: alpha dose-response, patient versus agent, both names versus separate names, matched passive versus active names, reversed-name-state transfer, full passive versus two names, and transfer direction. Differences from zero in these highly correlated conditions are not independent discoveries.

## 5. Interpretive limits and endpoint

Changes from these residual interventions establish that transplanted source-state information influences downstream predictions, not that a specific neuron, head, MLP or token **represents the semantic role**. Mixed states are potentially off the model's natural activation manifold; token positions and previous context differ between PA and AP. Even accurate improvement accompanied by wrong-event increase is not a selective role repair. Reversed-token interventions remain deliberately off-manifold.

After EXP-116, **stop adding experiments based on the same repeatedly explored source set**. Freeze the complete EXP-109–116 results, report positive and negative controls, and prepare a scientific methods/results write-up. Any subsequent model/grammar generalization or mechanism confirmation is a *separately justified, independently frozen project* and should not be folded back into EXP-116.

## 6. Run procedure

From `~/Developer/exp102-clean-design`, using existing main-repo `.venv`:

```bash
source ~/Developer/llm-relational-mechanisms/.venv/bin/activate
unzip -n ~/Downloads/exp116_targeted_role_event_selectivity_v0_bundle.zip -d .
python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase prepare
python3 scripts/exp116_targeted_role_event_selectivity_v0.py --phase smoke --device mps
# Commit exact runner, launcher, protocol and forced-add manifest BEFORE scientific runs.
bash scripts/run_exp116_all_v0.sh
```

The shell launcher is resumable (skips immutable per-model JSON reports), writes a final summary only after all 48 checks, and never retrains or opens protected evaluation. On macOS it may be executed under `caffeinate -dimsu` with AC power and lid open. Runtime depends on Mac hardware and must not be assumed from previous experiments.

## Verification before shipment

- Python compilation: PASS.
- CPU smoke test on random model plus **previously examined legacy development examples only**: PASS for all 21 modes, exact numerics of full-dose patches against EXP-115, identity check and event/role transition conservation.
- Real EXP-114 checkpoint outputs: **not seen or used when constructing this protocol**.
