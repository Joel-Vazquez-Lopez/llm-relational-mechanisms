# EXP-110 — Read-only EXP-109 error decomposition

**Status:** post-hoc analysis of the complete, frozen EXP-109 study, not a new independent confirmation. Analysed on 17 November 2026. Source: `experiments/exp102-clean-design` branch, all 16 committed JSON reports in `reports/exp109/replication_v0/`.

## What was actually established in EXP-109

Eight independently initialized training seeds, each run twice on Apple MPS, were analyzed by first averaging repeats within seed. The prespecified PA-only difference between literal PA and AP four-question accuracy is **+61.37 percentage points** (95% seed-level t interval **+41.33 to +81.41 pp**, positive in 8/8 seeds). The prespecified indirect-minus-baseline AP gain is **+22.96 pp** (95% interval **−2.11 to +48.03 pp**): suggestive, but **inconclusive**.

The literal clause swap preserves source tokens, clause-internal syntax, event participants, and event meanings; it changes first/second event position and question references. This is strong evidence of **construction/position-sensitive behavior in a small scratch-trained model within one synthetic grammar**, not proof of a language-general failure or a particular internal mechanism.

## New, explicitly exploratory error findings

All numbers below are means of seed means (8 seeds, 2 numerical repeats per seed); sample units are *seeds*, not questions or prompt instances. The reported 95% intervals are descriptive, unadjusted for multiplicity.

| PA-only model | AP | Literal PA | AP − PA |
|---|---:|---:|---:|
| All four answers correct | 6.34% | 67.71% | −61.37 pp |
| Individual question accuracy | 47.30% | 88.98% | −41.68 pp |
| Full-vocabulary question accuracy | 46.54% | 88.55% | −42.01 pp |
| Wrong-event per-question rate | 13.98% | 4.74% | **+9.23 pp** |
| Wrong-role per-question rate | 38.73% | 6.28% | **+32.44 pp** |
| Same-participant response rate | 19.86% | 9.01% | **+10.85 pp** |

The *wrong-role rate* increases more than the wrong-event rate when going from literal PA to original AP **in every one of the eight seed-level comparisons**. The descriptive interval for the AP-minus-PA wrong-role increase is **+25.79 to +39.10 pp**; for wrong-event increase **+2.51 to +15.96 pp**. These rates partition native four-name candidate question errors together with correct answers. Same-participant output is a different statistic and overlaps the other categories—do **not** add it to them.

### Second-event agent/patient answer pair on original AP

The following are mutually exclusive categories on each full second-event agent/patient *pair*, using native **four-name candidate** outputs (not an oracle that forces the correct event):

| Outcome | PA-only | Indirect PP+AA | Direct PA+AP |
|---|---:|---:|---:|
| Both roles correct | **11.1%** | 36.4% | 43.8% |
| Both roles reversed | **41.2%** | 17.1% | 10.8% |
| Same-participant collapse | **21.4%** | 27.2% | 27.8% |
| At least one name from wrong event | **26.4%** | 19.3% | 17.6% |

The PA-only baseline's most common second-event pair outcome is **role reversal** rather than failure to locate the event. That is a *behavioral result*. It **does not establish** that the network represents roles internally and swaps them; a positional heuristic could yield the same output pattern.

### Secondary exposure contrast and numerical variability

On AP, indirect exposure had **+19.35 pp** average individual-question accuracy relative to PA-only and **+19.27 pp** full-vocabulary accuracy; post-hoc descriptive 95% t intervals are approximately **+2.86 to +35.83 pp** and **+3.10 to +35.45 pp**, respectively. These are secondary, uncorrected and should not be promoted to a new confirmatory success after the predeclared all-four-answer endpoint was inconclusive.

The mean *absolute* difference between the two MPS repetitions of the same seed was **13.7 pp** for the primary construction-gap measure (maximum **44.4 pp**) and **14.8 pp** for the indirect-lift measure (maximum **41.7 pp**). These are within-seed numerical-repeat differences in *effects*; they illustrate the sensitivity of the training trajectory and do not represent independent-seed variation.

## What to test next (development only; not a circuit claim)

**Candidate H1 (voice × event-position role mapping):** The PA-only model applies a learned participant-to-role mapping insufficiently conditioned on *second-event passive voice*. It can locate the relevant participants but frequently outputs reversed roles.

Competing explanations to rule out:
- **Position/answer heuristic:** the network relies on token order, nearest names, or question-specific position rather than abstract grammatical roles.
- **Training coverage:** PA-only sees no second-position passive examples, making the AP failure expected even without a distinct mechanism.
- **Optimization variability:** models can take different trajectories on MPS despite the same initialization and training schedule.

**Recommended next stage:** select a very small set of *development-only, within-checkpoint* counterfactual experiments to probe role logit margins and event-vs-role access while holding clause tokens fixed. Evaluate event and role questions separately. Only after a prediction is fixed should we perform causal interventions (e.g., within-model activation patching) on independently matched pairs with control positions and placebo perturbations, then replicate on new seeds. Do not treat an exploratory layer or head found by searching all components as an independently confirmed mechanism.

## Data and reproducibility

- Read-only analysis source: the 16 `seed*_repeat*_mps_step6400.json` files on the experimental branch; original scripts and results should remain untouched.
- Run `scripts/exp110_exp109_readonly_error_audit_v0.py --write` from `exp102-clean-design` to recompute all derived metrics and store `reports/exp110/read_only_error_audit_v0/analysis.{md,json}`. The analyzer uses only the frozen aggregates, never opens `.pt` checkpoints or reconstructs sentence sources.
- The analyst should review and commit **only** new EXP-110 artifacts after verifying the numbers; no need to rerun EXP-109.
- Existing protected, reserved, L+I and candidate 107xx evaluations remain closed.
