# Evidence ledger — construction-sensitive roles

**Status:** working manuscript package, evidence frozen from the `experiments/exp102-clean-design` branch at commit `df82ab3ad9db5e13db80cf47caf8f2142afa3dc7`. Prepared 2026-11-21. This ledger does **not** introduce new evaluations.

| ID | Claim and quantitative estimate | Design/evidence status | Exact source | Boundary |
|---|---|---|---|---|
| C1 | PA-only literal PA minus AP four-query accuracy **+61.37 pp**; 95% seed-level t CI **[+41.33,+81.41] pp**; 8/8 seeds positive | EXP-109 prospective, 8 independent seeds × 2 MPS repeats | `reports/exp110/read_only_error_audit_v0/analysis.json` `locked_outcomes.primary_pa_only_gap`; EXP-109 frozen run reports | Same synthetic grammar, new sentence permutations; same lexical identity and clauses, reordered event position / reference; not naturally occurring language |
| C2 | Indirect PP+AA minus PA-only on AP: **+22.96 pp**, CI **[−2.11,+48.03] pp**, 6/8 positive | EXP-109 secondary **inconclusive** | EXP-110 `locked_outcomes.secondary_indirect_AP_lift` | Training alters second-event passive exposure; cannot attribute to abstract composition |
| C3 | PA-only AP second-event pairs: **11.07% correct, 41.19% reversed, 21.35% same-person, 26.39% wrong-event** | EXP-110 post-hoc descriptive, native four-name predictions | EXP-110 `descriptive_AP_second_event_pair_classes.pa_only` | Exclusive at pair level for this endpoint; do not confuse with question-level wrong-role |
| C4 | AP minus literal PA per-question wrong-role **+32.44 pp** and wrong-event **+9.23 pp** | EXP-110 exploratory descriptive; eight seed means | EXP-110 `descriptive_comparisons` | Different definitions from C3; no multiplicity adjustment |
| C5 | Passive-clause input position-code edit improves AP second-event native answer accuracy **+29.23 pp**; CI **[+18.49,+39.97]**; 8/8 seeds positive | EXP-114 first predeclared hierarchical test, fresh seeds and disjoint dev surfaces | `reports/exp114/independent_replication_v0/analysis_summary.json` `test_decisions.primary_AP_input_position_accuracy_rescue` | Intervention also changes upstream computations; reversed alignment comparable in explored controls |
| C6 | First-block learned-update transplant exceeds *late* explicit positional-code transplant on AP by **+16.97 pp**; CI **[+1.73,+32.21]**; 7/8 positive | EXP-114 gated secondary 1; passed | EXP-114 `test_decisions.secondary_1_AP_update_greater_than_late_position_accuracy` | Learned update itself depends on absolute position and context; not a position-independent semantic code |
| C7 | Reverse transfer into familiar PA reduces first-event native accuracy **−34.27 pp**, CI **[−47.80,−20.73]**; all eight negative | EXP-114 gated secondary 2; passed | EXP-114 `test_decisions.secondary_2_PA_reciprocal_update_disruption` | Within-grammar, within-architecture causal influence, not a unique circuit |
| C8 | Decomposing first-block head/MLP writes shows causal influence; none isolates a universally selective role circuit | EXP-115 exploratory using EXP-114's **same** checkpoints and dev examples | `reports/exp115/block1_head_mlp_causal_v0/analysis_summary.json` | Multiple related interventions; off-manifold effects and wrong-event increases |
| C9 | **0/12** candidate dose modes satisfied EXP-116's strict descriptive selectivity screen | EXP-116 final exploratory study on same checkpoints/examples | `reports/exp116/targeted_role_event_selectivity_v0/analysis_summary.json` `strict_descriptive_selectivity_screen` | Failure of a screen is not proof that no selective mechanism exists; screen unadjusted and descriptive |
| C10 | Different scratch-Transformer training cohorts show exposure benefits (48/48 paired seeds), but this is not 48 independent grammars | Historical ML-080/084/085/086/088/099 | `paper/manuscript.md`; ML archived JSON reports | Context only, not pooled into EXP-109 confidence intervals |

| C11 | PHASE2-01: p020−p000 AP accuracy: **+10.35 pp; CI [−10.27,+30.97]; failed primary** | New-seed prospective PHASE2 study; scope as labelled | `reports/phase2_01/coverage_curve_v0/analysis_summary.json` `tests` | 20% means eligible dose-pair steps, not all examples |
| C12 | PHASE2-01: p100−p000 AP accuracy: **+27.30 pp; CI [+13.04,+41.56]; gated secondary NOT reached** | New-seed prospective PHASE2 study; scope as labelled | `reports/phase2_01/coverage_curve_v0/analysis_summary.json` `tests` | Positive interval descriptive only after failed primary |
| C13 | PHASE2-02: p100−p000 shifted-AP accuracy: **+6.97 pp; CI [−13.10,+27.03]; failed primary** | New-seed prospective PHASE2 study; scope as labelled | `reports/phase2_02/structural_transfer_v0/analysis_summary.json` `tests` | Two novel adjunct placements, same grammar/vocab |
| C14 | PHASE2-03: p000 tail named-distractor specificity: **+6.46 pp; CI [−3.09,+16.00]; failed primary** | New-seed prospective PHASE2 study; scope as labelled | `reports/phase2_03/distractor_event_binding_v0/analysis_summary.json` `tests` | Tertiary p100 positive interval is gated/unreached |

| C15 | Post-hoc source-position shortcut: PA-only literal PA−AP strict C_query gap **+44.44 pp** for raw name-mention prior, **+100.00 pp** for oracle-segmented core-slot prior; both fail to reproduce same-person collapse; core-slot oracle 0% wrong-event by design | Read-only deterministic baselines run on frozen EXP-109 dev surfaces; no new neural inference | `papers/learning-relations/construction-shift/baselines/analysis_summary.json`, successful GitHub Actions run #44898500105 | Core-slot extractor uses privileged grammar-specific modifier stripping; expected pool weights not exact per-seed optimization histories; post-hoc, not new confirmation |

| C16 | POSITION-CB-v0 frozen across-arm familiar−unseen `C_query` **+14.4097 pp**, CI **[+0.0041,+28.8154] pp**, 7/8 seeds positive | Prospective *within-grammar* new-seed study; 8 seeds × 2 MPS repeats × 2 arms; passed primary near threshold | `experiments/position-counterbalanced-roles-v0/reports/position_cb_v0/analysis_summary.json` and 16 raw reports | Same original grammar; balanced voice-position marginals and token bags **not** all absolute positions; severe heterogeneous competence (3 seeds ~90% of sum of effects) |
| C17 | POSITION-CB-v0 gated unseen−familiar complete role reversal **+1.6005 pp**, CI **[+0.0139,+3.1871]**, 7/8 seeds positive | Gated secondary passed *barely*; same seed cohort as primary | Same frozen summary `tests.gated_secondary_reversal_increase` | Exploratory paired-error categories also show collapsed and wrong-event changes; not a unique role circuit |
| C18 | Same-voice familiar/unseen 35.44%/28.93%, mixed-voice 38.54%/16.23%. Gap interaction **+15.80 pp** paired exploratory CI **[−10.01,+41.61]** | Descriptive interaction; not prespecified test | 16 raw reports; `analysis/position_cb_v0_audit.md` | No confirmed between-arm interaction; different unseen voice sets; extreme per-seed variance |

## Required reporting rules

- Seed is the replication unit. Average the two MPS numerical repeats within each seed before statistics.
- POSITION-CB-v0 is a subsequent, separately frozen within-grammar study with two **nominally passed but extremely borderline** hierarchical tests. These are not cross-grammar replication or circuit identification. All confidence intervals are computed from 8 seed means, not 16 process repeats or 32 model states.
- Label EXP-110, EXP-111–113 and EXP-115–116 **exploratory**. PHASE2-01–03 were prospective *within-project* follow-ups with independent seeds, but their frozen primary tests failed; gated secondary/tertiary positive intervals must never be promoted as successful tests. EXP-114 is a prospective within-grammar replication of effects selected on earlier development results, not independent grammatical-family validation.
- Distinguish `C_query` (strict all-four questions correct), single-question native four-candidate accuracy, full-vocabulary accuracy, and correct-event-oracle restricted outcomes.
- `wrong_role_rate` is *per-question* selecting the other member of the correct event. A **complete pair reversal** is an event-level paired outcome. Do not conflate these.
- A positive patching effect is a causal response to an imposed perturbation; not evidence of necessity, uniquely decoded native pathway, semantic-role representation or context-independent mediation.
- No protected/reserved or within-event L+I evaluation was accessed for this package.
- Confirm all original reports and script hashes before submission. Checkpoint weights are intentionally not in Git; production replication requires the local EXP-114 `.pt` files.
- Figures 2–4 are derived from archived aggregate values (via `figure_data.json`); no experiment was rerun and no plots are new confirmatory tests.

## Closest literature to compare explicitly

- Kim & Linzen (2020), COGS, https://aclanthology.org/2020.emnlp-main.731/
- Li et al. (2023), SLOG, https://aclanthology.org/2023.emnlp-main.194/
- *Learning Syntax Without Planting Trees* (TACL), https://doi.org/10.1162/tacl_a_00733
- Tang, Lake & Jazayeri (2026), *Circuit explained: How does a transformer perform compositional generalization*, https://doi.org/10.1371/journal.pone.0340088

These are background comparators; an exhaustive novelty review remains outstanding.
