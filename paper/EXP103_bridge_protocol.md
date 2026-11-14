# EXP-103 — Connecting exposure-driven and supervision-policy-driven relational learning

**Status:** Hypothesis and audit protocol only; **not frozen**, **not confirmatory**, and **no new protected evaluation**.

## Research question

Do targeted exposure (ML-080–ML-099) and full-quartet question batching (EXP-102) improve relational generalization by changing the same kind of error, or do they correct distinct failures of event localization and role assignment?

### Empirical background

- The original ML cohorts varied the *placement/voice of a sparse targeted exposure* while retaining a familiar grammatical base, evaluated an unseen second-event passive construction, and logged `wrong_event_rate`, `wrong_role_rate`, `same_answer_rate`, `correct_both_rate` and individual candidate-restricted accuracy.
- EXP-102 compared *training update composition*, initially in a simple grammar and then in a richer distractor grammar. Its v2 reports log `correct_event_identity_*`, `role_copy_*`, `both_correct_*`, four-question `C_query`, and individual candidate-restricted accuracy.
- The v2 extension was selected **after** the 1,600-step developmental study; its 3,200-step contrast is exploratory.
- Both families use small scratch-trained Transformers with similar model scales, but not identical data, target distribution, or training regimes. Their reported effect sizes should not be subtracted from one another.

## Step 0: Read-only error-taxonomy bridge

Run `python3 scripts/exp103_bridge_audit.py --write`. This only reads **previously saved** ML-080/084/085/086/099 historical evaluation summaries and EXP-102 v2 familiar-development JSON files. It writes derived reports under `reports/exp103_bridge/` with original SHA-256 fingerprints; never reads new protected sets, loads models, or generates test examples.

For the older experiment, analyze **within-cohort matched seed differences** between target and familiar exposure on exactly the same held-out target condition.

For EXP-102, examine quartet vs balanced-unpaired at step 1,600 (original development endpoint) and 3,200 (later exploratory extension), with the same familiar-syntax development pool in each arm.

### Metrics bridge (not identical test distributions)

| Functional quantity | Historical ML reports | EXP-102 v2 reports | Caveat |
|---|---|---|---|
| Per-question accuracy | `choice_accuracy` | `accuracy4` | Both four-name restricted but distinct data regimes |
| Wrong-event answer | `wrong_event_rate` | `1 - mean(correct_event_identity_first, correct_event_identity_second)` | EXP-102 estimates across all four questions; ML targets one event |
| Wrong-role answer inside correct event | `wrong_role_rate` | `mean(correct_event_identity_first, correct_event_identity_second) - accuracy4` | Requires distinct event participant sets; verified in both grammars |
| Agent/patient identical-answer failure | `same_answer_rate` | `mean(role_copy_first, role_copy_second)` | ML one target event; EXP-102 mean across both events |
| Complete event roles correct | `correct_both_rate` | `mean(both_correct_first, both_correct_second)` | Two role questions for one event |
| All four questions correct | Not directly recorded in original ML historical protected summaries | `C_query` | **Never equate historical `correct_both_rate` with C_query** |

**Audit decision rule:** If old `same_answer_rate` does not decrease while `wrong_event_rate` falls, describe interventions as producing **different behavioral trajectories**, not as proof that they both solve the same role-copy bottleneck.

## Step 1: Define a controlled factorial replication on one grammar (future)

Proposed **2×2 design**, not yet implemented:

| Factor | Condition 0 | Condition 1 |
|---|---|---|
| Exposure content | Familiar-position passive *dose* | Target-position passive *dose* |
| Update policy | Complementary questions **distributed** over a four-update window | All four questions for each source **co-located** within one update |

Use one consistent corpus, vocabulary, architecture, optimizer, candidate decoder, source pool, initialization seed panel, frozen familiar-development set, and separate reserved set. Train four matched models per seed; predefine all budgets, gates and analysis before training.

### Important design constraints

1. **Avoid accidental target-cell leakage from full quartets.** In the original ML study, second-event passive questions were rare/withheld. If a sentence with a passive second event appears in *any* full-quartet base training batch, its second-event agent and patient become supervised. The new base grammar must exclude the target event/voice *at the queried event level*, not merely label a source sentence as a familiar example. Verify this across all questions.
2. **Match exposures over four optimizer steps, not just total labels.** Sample 64 source sentences and their four questions, yielding 256 source-question items per block. The quartet policy uses four consecutive batches of 16 sources × four questions. The distributed policy uses four balanced batches of 64 sources × one question each, rotating question types for each source via a Latin square. Each block sees the exact same 256 labeled items across arms, with 16 of each question type per update. Differences concern within-update co-occurrence / gradient variance, not differing long-run label exposures.
3. **Acknowledge unavoidable within-update source diversity.** Quartet updates use 16 distinct sources; distributed updates use 64. That is part of the policy being tested. The four-update multiset is identical, but optimizer nonlinearity means *when* examples enter still matters. Explicitly avoid claiming to isolate an intrinsic cross-example computation.
4. **Gradient mathematics sanity check.** A Transformer without cross-example interactions optimizes a sum of per-example losses. For an identical multiset in one update, regrouping/reshuffling examples alone does not change its expected gradient. We are testing **update scheduling and gradient structure**, not whether the Transformer directly observes other sentences in a minibatch.
5. **Both-event competence gate.** Freeze development requirements for first/second participant selection and agent/patient distinctness *before* any composition or reserved evaluation. Avoid selecting only successful seeds.
6. **Error decomposition is primary.** Log event miss, within-event wrong-role, same-answer rate, both-role correctness and all-four correctness for **every model** at all milestones. The 2×2 interaction in error-type changes is the bridge test; do not focus only on aggregate accuracy.
7. **Do not conflate familiarity and compositional transfer.** L+I has known absolute-position novelty and previously inspected design examples. Syntax-compositional generalization must use a separately frozen design with prevalidated labels and a position-matched control, only after competent familiar-grammar training.
8. **Historical ML-099 causal perturbations are not native mechanism identification.** Q-transplant can import answer information, and the existing causal investigations did not isolate a unique native circuit. Do not upgrade observational bridge findings to a circuit claim.

## Falsifiable predictions

- **Common-bottleneck prediction:** Both targeted exposure and quartet batching should lower within-event wrong-role and same-answer errors on the same evaluation regime, with smaller or secondary changes to wrong-event errors. A positive interaction could suggest they act on related learning dependencies.
- **Distinct-bottlenecks prediction:** Targeted exposure mainly changes wrong-event selection/structural transfer, while quartet batching primarily changes role-copy collapse and role differentiation. The 2×2 interaction would be small or asymmetric across the error categories.
- **Dose/scheduling artifact prediction:** After matching the complete multiset of source-question exposures over each four-update window, the apparent quartet advantage disappears or becomes markedly smaller. This would revise the original attribution without invalidating the empirical observation that the original sampling policies differed.

## Workflow and stopping decisions

1. Read-only audit of existing reports; archive error taxonomy and all source hashes.
2. Independently inspect the old and current metric implementations; use familiar development only for design.
3. Create a **new** deterministic generator for the factorial study, with a source-level exposure audit and independent label oracle. Freeze base/target grammar cells, seed panel, batch schedule and version hashes.
4. Train all four arms over paired seeds at fixed optimization budgets; no dropping slow learners and no retroactively changing milestones.
5. Analyze the prespecified factorial contrasts across seeds; summarize failures as accurately as successes.
6. Only if competence gates pass: evaluate a separately reserved compositional condition exactly once, under its preregistered unlock procedure.

**Publication framing under investigation:** “Event Selection and Role Assignment Are Distinct Learning Bottlenecks: Effects of Relational Exposure and Update Scheduling in Small Transformers.” This is a hypothesis, not yet a demonstrated mechanistic finding.
