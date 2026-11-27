# EXP-112 — Causal clause-state interventions on AP↔PA role assignment

**Status:** v0 exploratory, post-hoc *causal intervention* on previously scored EXP-109 development examples. **Not preregistered confirmation of a mechanism, not a new generalization benchmark, not evidence of learning trajectories.** This protocol is authored before running EXP-112 on the Mac-trained checkpoints; ideally commit it with the script before any real EXP-112 scoring. No changes to EXP-109/110/111 frozen artifacts.

## Motivation and falsifiable candidate

The prospective EXP-109 study (8 independent seeds × 2 MPS repeats × 3 arms) found a large PA-only **literal PA minus AP** four-query accuracy gap. Subsequent exploratory EXP-110 and EXP-111 showed that, on AP, wrong-role responses are prominent and the PA-only final output margins favor the *opposite* agent/patient on the passive second event. EXP-111's early linear-name probes do not establish a useful hidden role representation and do not localize a causal circuit.

**Candidate hypothesis:** context-dependent source-clause representations contribute to construction-sensitive role assignment. If this is true in a way accessible to the proposed intervention, replacing AP's passive second-event source states with states for the **same lexical passive clause placed first in the PA counterpart** may shift AP second-event agent/patient logits toward correct roles; reciprocally, the reverse replacement in PA may degrade first-event role evidence. Specificity should exceed both active-clause patching and scrambled within-clause alignment.

**Not claimed:** any discovered neuron, head, abstract role variable, semantic-role mediation, or isolated effect of passive voice. An intervention may fail to produce the predicted effect even when a position-dependent role mechanism exists.

## Immutable scientific inputs

- Same **48 saved 6,400-step EXP-109 final models**; no retraining and no checkpoint selection.
- Reconstruct exactly **144 already evaluated EXP-109 AP anchors** and their **144 literal PA clause swaps**, with all four first/second × agent/patient questions per source.
- Verify the unchanged EXP-109 manifest, source report, checkpoint metadata and SHA256 provenance. Compare every unpatched run's native C_query, per-query accuracy and full-vocabulary accuracy against the existing EXP-109 report. Refuse discrepant runs.
- No additional data distribution, model-training intervention, new name set, new verb, L+I, reserved examples or protected evaluation is opened. The 144 examples are **not independent of prior analysis**; this is the same studied development set.
- No source sentences, per-example hidden states or predictions are written to reports. Only aggregated outcomes and source/code hashes are saved.

## Exact intervention and controls

Model: original three-block width-96 four-head causal Transformer. Capture source-token residual activations at embedding, after block 1, after block 2, and after block 3. For each pair, the AP passive *second* clause has exactly the same surface tokens as the PA passive *first* clause; the active clause also matches lexically. Map each clause token by **within-clause word/token identity**, *not* by absolute sentence position. Neither punctuation outside the clause nor any query/answer tokens are patched.

At a specified depth, replace recipient **source-clause states only** with donor clause states from the **same checkpoint** and matching underlying event, then continue running subsequent blocks to the original recipient answer token. Importantly, *source hidden states are causally upstream of the query*; audit that they are invariant to first/second and agent/patient query suffixes. This prevents trivial copying of a donor's answer-token representation. No cross-model swaps or coordinate-alignment assumptions.

**Directions (both mandatory):**

1. `AP <- PA_literal`: donor passive clause appears first; target passive clause appears second. Test AP first- and second-event questions; **second-event** role outcomes are the focus.
2. `PA_literal <- AP`: donor passive clause appears second; target passive clause appears first. Test PA first- and second-event questions; **first-event** role outcomes are the reciprocal focus.

**Patches at every depth, with all models and sentences included:**

| Patch | Content and purpose |
|---|---|
| `identity` | No modification: exact native-forward identity guard |
| `passive` | Semantically identical passive clause, same within-clause token order |
| `active` | Semantically identical active clause from swapped source; event/context control |
| `both` | Both matched clauses, potentially nonlinear, not attributable to either alone |
| `passive_misaligned` | Reverse the correspondence within the passive donor clause; same donor-state multiset, deliberately *not* lexical alignment. This is an imperfect perturbation control, not an in-distribution counterfactual |

Depths `embed`, `block1`, `block2`, `block3`. A **source-only patch after block3** must have zero effect because the recipient answer-token state is already computed; this tests implementation, not a theoretical absence of layer-3 involvement. An `identity` patch must have no effect at every depth.

At the embedding stage, replacing states also transfers **absolute position embeddings**. At later depths it transfers both position and prior context. A positive intervention therefore *cannot* by itself distinguish structural role representations from positional/lexical/contextual effects or distribution shift caused by moving hidden states across positions. In particular, the literal clause swap preserves token contents but **does not preserve token positions or question event indices**.

## Recorded measurements

All predictions are the native four-candidate predictions; no gold-event oracle is used for the decision. For AP and PA, all four query categories, each depth and patch type:

- Four-query source accuracy `C_query`; per-question native accuracy, wrong-role and wrong-event rates.
- Gold-role margin: `logit(correct participant) - logit(opposite participant in same event)`. A positive margin does not guarantee native correctness because a participant from the other event can win.
- Fraction of individual queries changed from incorrect→correct or correct→incorrect. No filtering to only AP-fail/PA-pass cases.
- Changes relative to the **unpatched recipient** for each exact condition, plus candidate contrasts combining the two relevant query types.
- Hashes of checkpoint, source report and analyzer; identical dependency checks as EXP-111.

**Post-hoc candidate contrasts, averaged within seed over numerical repeats, then across 8 seeds:** (i) AP second-role passive-patch margin change; (ii) passive minus active patch difference; (iii) aligned passive minus misaligned passive difference; (iv) AP first-role preservation; (v) reciprocal PA first-role change; (vi) native AP second-query accuracy change. Report per-seed values and a descriptive 95% t interval (df=7). No exclusion of incompetent runs or weak repeats.

## Candidate nomination rule (exploratory, NOT confirmatory significance)

A pattern worth independently testing would show, for the *same* pre-selected depth (`block1` or `block2`):

- AP second-agent **and** second-patient margins shift positively under passive patch; native accuracy tends to improve rather than simply moving logits.
- PA first-agent and first-patient margins shift negatively under the reciprocal passive patch.
- Passive alignment is more selective for the relevant event than the active patch and deliberately misaligned passive patch; first-event AP roles and wrong-event rates do not show indiscriminate disruption.
- Effects recur over independent training seeds, not just a few models; retain and inspect all 16 numerical runs per arm.

Given post-hoc layer/patch/arm exploration and prior exposure to the AP/PA evaluation, no t-interval or consistency count can turn this into independent confirmation. We do **not** retrospectively select the best layer as a validated circuit. If a promising prediction emerges, freeze it *before* using **new seeds and new, protected-by-design but explicitly authorized evaluation examples** from the non-reserved development family. Do not open the existing L+I/protected tests without their separate preregistration and gate.

## Interpretability limitations

1. Moving entire clause residual streams is a coarse causal intervention, not a selective circuit operation. It can introduce off-manifold hidden states, mismatched positional codes or altered lexical context.
2. A positive rescue does not prove the model has a generalized semantic-role representation. A negative result does not show that the network lacks one.
3. There is no new training trajectory: final EXP-109 checkpoints only.
4. `active` and `passive_misaligned` are informative but imperfect controls. More precise, low-level interventions would follow only if results support them.
5. AP/PA anchor examples have already been examined; this is an exploratory analysis of previously unsealed data.
6. Causal mediation by a learned representation would require independent and targeted interventions with appropriate counterfactuals, not merely improvement from these swaps.

## Commands

Use `~/Developer/exp102-clean-design` and activate `~/Developer/llm-relational-mechanisms/.venv/bin/activate` as needed.

```bash
unzip -n ~/Downloads/exp112_clause_state_patch_v0_bundle.zip -d .
python3 scripts/exp112_clause_state_causal_patch_v0.py --phase prepare
python3 scripts/exp112_clause_state_causal_patch_v0.py --phase smoke --device mps

# Commit script/protocol before looking at real checkpoint intervention results.
python3 -u scripts/exp112_clause_state_causal_patch_v0.py --phase run --seed 10901 --repeat 1 --arm pa_only --device mps

# Then use included safe sequential launcher (skips previously completed reports):
bash scripts/run_exp112_all_v0.sh
```

`--phase summary` refuses a partial aggregate, requires all 48 output reports, averages numerical repetitions within seed, and refuses mixing results from different script versions. The launcher will not overwrite any existing report. Do not force-add the large `.pt` checkpoints to Git.

## Test status as shipped

- Python syntax compilation PASS.
- CPU familiar-dev random-model smoke PASS: native readout consistency, source-only patching, identity and final-block no-effect guards.
- Exact EXP-109 evaluation digest/vocab verified in isolated container; lexical alignment verified for all 144 AP↔PA anchors.
- End-to-end CPU run with **synthetic random checkpoint and synthetic report in an isolated fixture** PASS, including native-score reconciliation, metadata, source-query invariance, reciprocal interventions, and report writing. These fixture artifacts **are not scientific data and are excluded from the bundle**.
- Real Mac-trained checkpoint interventions **not run here**; their results remain unknown.
