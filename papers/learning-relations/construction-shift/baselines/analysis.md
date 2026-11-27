# Results of frozen EXP-109 positional-shortcut analysis

**Execution:** GitHub Actions [successful run #44898500105](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/actions/runs/44898500105) on 2026-11-21. **Code commit:** `1db6fe54ea478560ab2c6eecc966e9c2e728e2ab`. The pre-run protocol was committed in the same source change. **Evaluation digest matched the frozen EXP-109 SHA-256** (`5063bec0fceca42ce9a899ffeaa93c0e26a56b93734e80d5388e211e7c9e0b76`). All 72 fixture checks passed; 144 development anchors × six original source conditions were reconstructed. No trained model checkpoints, protected examples or L+I tests were opened. Output is a post-hoc *deterministic rule-based benchmark*.

## Main question: can positions alone create an AP/PA gap?

| Condition | EXP-109 PA-only trained Transformers (8-seed mean) | Raw name-mention positional lookup | Core-participant slot lookup (optimistic segmentation) |
|---|---:|---:|---:|
| AP four-query C_query | **6.34%** | **0.00%** | **0.00%** |
| Literal PA C_query | **67.71%** | **44.44%** | **100.00%** |
| Literal PA − AP | **+61.37 pp** | **+44.44 pp** | **+100.00 pp** |
| AP individual-question accuracy | **47.30%** | **0.00%** | **0.00%** |

A simple rule learned from the expected 7 AA + 7 PA + 2 PA exposure mixture can produce a very large construction gap despite ignoring voice markers and syntactic roles when predicting. **Therefore, the construction gap alone is not evidence of native abstract semantic-role binding.** The optimistic baseline receives clause-core segmentation by hand-crafted removal of the generated `near NAME` and gate adjuncts; it is not a deployable semantic parser, not capacity-matched to the neural model, and should not be called a fair neural baseline.

## Joint error profiles differ

| AP second-event pair outcome | Trained Transformer 8-seed mean | Raw name-position lookup | Core-slot lookup |
|---|---:|---:|---:|
| Both roles correct | 11.07% | 0.00% | 0.00% |
| Complete reversal | 41.19% | 66.67% | 100.00% |
| Same participant twice | 21.35% | 0.00% | 0.00% |
| At least one wrong-event answer | 26.39% | 33.33% | 0.00% |

The deterministic slot rules cannot generate same-person collapse under their fixed complementary selections. The raw-mention rule can select modifier distractors, but it does not reproduce the model's complete error mixture. This difference is **descriptive**, not proof of a neural role variable; other simple heuristics could produce collapses.

## Exposure-arm dependence not reproduced by this shortcut

| Arm | Original Transformer AP C_query (8-seed mean) | Raw positional rule | Core-slot positional rule |
|---|---:|---:|---:|
| PA only | 6.34% | 0.00% | 0.00% |
| Indirect PP+AA | 29.30% | 0.00% | 0.00% |
| Direct PA+AP | 35.81% | 0.00% | 0.00% |

The fixed learned slot priors do not capture the improved AP performance after indirect or direct exposure. However, the EXP-109 indirect-minus-baseline primary/secondary sampling uncertainty remains as reported in the original study (**+22.96 pp, 95% CI −2.11 to +48.03; inconclusive**). A deterministic baseline that fails is not sufficient to claim an abstract generalization mechanism.

## Design and caution

- **Raw** uses event index, queried role and total name mentions in each clause; it does not inspect `was`, `by`, grammatical voice, verb identity or gold labels at prediction.
- **Core slot** first removes adjunct names using a known grammar-specific rule. This is an optimistic source segmentation advantage, not an autonomous parser. It also cannot choose the wrong event by construction.
- Expected frequencies are estimated from a balanced *training source pool*, not from each individual 6,400-step stochastic model schedule. Thus no seed-level uncertainty is claimed for the rule baselines.
- All comparisons are on already examined EXP-109 development surfaces, so this audit is **not new prospective held-out validation**.
- The model is a neural learner with numerical variability and 8 independent seeds; the baseline is one deterministic rule per exposure arm. Neither should be used as independent replicates for a new significance test.
- A simple grammar-aware voice rule could trivially solve these sentence templates once explicitly supplied the correct parse rules; this would be a solvability control, not evidence that the neural model uses those rules.

**Decision:** Treat the large construction gap as explainable by simple positional priors in principle; focus subsequent studies on the neural model's *departure from these predictions*, especially role-collapse/event-confusion transitions and improvement under exposure arms. A new independent grammar or a controlled position-role decorrelation test is needed to claim broader relational generalization.

The complete deterministic `analysis_summary.json` is archived alongside the script and pre-run protocol. Only aggregate reports are stored; no per-sentence sources or predictions are published.
