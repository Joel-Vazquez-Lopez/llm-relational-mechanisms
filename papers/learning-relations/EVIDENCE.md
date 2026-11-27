[← Paper I](README.md) · [Experiment index](EXPERIMENTS.md)

# Evidence and limitations

This page is a **reading aid**, not a replacement for the original protocols or the [working manuscript](../../paper/manuscript.md). Read each result at its recorded evaluation status.

| Question / observation | Evidence to inspect | What it does **not** show |
|:--|:--|:--|
| Does target exposure help within the studied synthetic grammar? | [ML-080](../../reports/ml080/protected_evaluation.json), [ML-084](../../reports/ml084/protected_evaluation.json), [cross-cohort summary](../../reports/paper_cross_cohort/archived_error_taxonomy.md) | General transfer to ordinary language |
| Do joint-role errors behave like individual answer accuracy? | [Manuscript](../../paper/manuscript.md), [cross-cohort audit](../../reports/paper_cross_cohort/archived_error_taxonomy.md) | That every repeated or duplicate answer reflects a single neural failure mode |
| Are role-related states sensitive to intervention? | [ML-085](../../reports/ml085/protected_causal_evaluation.json), [ML-086 interpretation](../../reports/ml086/interpretation.md) | Necessity, mediation of exposure benefits or a unique native role circuit |
| Do proposed mechanistic interactions replicate? | [ML-087](../../reports/ml087/calibration_evaluation.json), [ML-099](../../reports/ml099_role_q/protected_evaluation.json), [ML-100](../../reports/ml100_source_access/exploratory_summary.json) | That a positive exploratory contrast is independently validated; several tested predictions failed |
| Does a structural external benchmark validate the task? | [Manuscript's SLOG discussion](../../paper/manuscript.md) | Protected SLOG structural-generalization performance; the pilot was development-only |

## Earlier bridge work outside `main`

The experimental branch has newer work that should be reviewed against its own protocols before paper incorporation:

- [EXP-109 prospective within-grammar replication](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/df82ab3ad9db5e13db80cf47caf8f2142afa3dc7/paper/EXP109_results_summary_v0.md) — a construction/position contrast, not general linguistic transfer.
- [EXP-110 read-only error decomposition](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/df82ab3ad9db5e13db80cf47caf8f2142afa3dc7/paper/EXP110_readonly_error_analysis_v0.md) — post-hoc analysis of EXP-109, not an independent replication.
- [PHASE2-03 fixed protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/df82ab3ad9db5e13db80cf47caf8f2142afa3dc7/paper/PHASE2_03_distractor_event_binding_v0_protocol.md) — a specified within-project distractor sensitivity test; inspect the archived results and analysis before interpreting it.

These are explicitly **branch-only references**. Their presence here does not mean that all materials have been merged or that the canonical manuscript has adopted their conclusions.

## Interpretive boundaries

- Synthetic training supports control, not ecological validity by default.
- Independent training seeds do not create independent linguistic task families.
- Restricted-candidate accuracy must be distinguished from native full-vocabulary responses.
- Layerwise decodability and intervention sensitivity are narrower than a demonstrated native pathway.
- Failed gates and inconclusive replications remain visible.

Original evaluation details and limitations: [scratch checkpoint](../../investigations/scratch_transformers/CURRENT_FINDINGS.md) · [evidence policy](../../docs/EVIDENCE_POLICY.md).

## Current programme — claim ledger, 27 November 2026

The following outcomes follow the researcher-provided handover, not a new statistical audit. Consult the original sources in the [experiment map](EXPERIMENTS.md).

| Observation | Status and interpretation |
|:--|:--|
| **TRAIN-SURFACE-FACTORIAL-v0** | Completed 12-seed factorial: unseen-role strict four-query joint C_query on the same bare test surfaces: bare 99.21%, prefixes 84.25%, adjuncts 97.01%, combined 26.73%. Adjunct-associated degradation +29.86 pp (95% t CI [13.46, 46.26], 9/12 positive). Original-mapping acquisition remained high; transfer deteriorated. |
| **TRAIN-SURFACE-CROSS-EVAL-v0** | Exploratory cross-evaluation: changing surface decoration generally did not rescue unseen assignments; generalisation varied across seeds. |
| **ROLE-DIVERSITY-COMPLEX-v0** | Completed prespecified factorial: 16 seeds, 1/2/4/8 assignments per quartet. Primary hypothesis not supported at 6,400 updates; 4/8 assignments caused acquisition difficulties. |
| **ROLE-DIVERSITY-RECOVERY-v0** | Selected-cohort exploratory continuation: four seeds to 51,200 updates, fresh optimiser. Some recovered near-perfect competence; others remained collapsed. Not a population recovery rate. |
| **ROLE-CIRCUIT-DIAGNOSTICS-v0** | Exploratory probability/loss decomposition: near-perfect event selection with near-equal participant probabilities in persistent collapse. The earlier named branch lacks the dedicated assets; they are preserved in the cumulative snapshot ([protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_CIRCUIT_DIAGNOSTICS_V0_PROTOCOL.md), [implementation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/role_circuit_diagnostics_v0.py), [runner](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/scripts/run_role_circuit_diagnostics_v0.sh), [summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/reports/role_circuit_diagnostics_v0/analysis_summary.json)) and proposed for integration in [PR #14](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/pull/14). |
| **ROLE-REPRESENTATION-LOCALISATION-v0** | Exploratory layerwise event, role and candidate profiles; decodability alone did not reliably predict competence. |
| **ROLE-CAUSAL-ROUTING-v0** | Selected successful models: attention-head, MLP and residual interventions found selective event/role effects; locations varied by seed. |
| **ROLE-CAUSAL-PATHWAYS-v0** | Joint event/role interventions, Q/K/V and positional decomposition, downstream restoration. Compositional causal control in selected models, not a universal native circuit. |
| **ROLE-BINDING-CAUSAL-VALIDATION-v0** | Two selected successful models: cross-source, cross-voice and participant-identity interventions showed differing transfer, especially event selection. |
| **ROLE-BINDING-LOCALISATION-v0** | Exploratory entity/non-entity token and distributed head effects; different computational organisations in individual successful models. |
| **ROLE-MECHANISM-REPLICATION-v0** | Independent seeded replication (16 models, 51,200 updates): 9 successful, 3 intermediate, 4 fully collapsed. Near-perfect event selection but chance-like conditional role discrimination in collapsed models. Prespecified mechanism-category replication failed. |
| **ROLE-MECHANISM-DISCOVERY-v1** | Exploratory expanded search over distributed heads and MLPs: recovered both positive controls, improved interventions in several new models, identified one additional source-dependent mechanism. Not retroactive confirmation. |
| **ROLE-MECHANISM-REPAIR-v0** | Preliminary selected-cohort repair: substantial targeted recovery in 1/4 collapsed models; earlier observed recovery under role weighting in that model. Full-model and parameter-matched random controls. |
| **ROLE-REPAIR-SPECIFICITY-v1** | Completed prespecified follow-up: 8 schedules from one checkpoint, 8 arms, 384 reports. Primary acceleration contrast at 1,600 updates = 0. At 6,400, targeted CE and weighted each 2/8; random and layer-matched controls 0/8. Approx. +25 pp, paired sign-flip p=.5; specificity inconclusive. |
| **ROLE-LEARNING-ORIGINS-v0** | ONGOING locally as of 27 Nov; engineering pilot reported passed; no scientific results. Fully crossed 8 initialisation × 8 independent training-stream seeds, 64 models, 51,200 updates/model; dense development, event/conditional-role loss, gradient magnitudes/alignment, preserved snapshots and resumable checkpoints. |

## What the dissociation establishes

Correct-answer probability can be decomposed, under the task's defined participant sets, into probability mass on the correct event times conditional probability of its correct participant. The corresponding negative log-probabilities add when defined. Near-perfect event selection with chance-like conditional role discrimination identifies a functional dissociation, not a unique neural circuit.

Distinguish initialisation seeds, training streams, matched arms, repeated continuation schedules, checkpoints and individual items. In particular, 384 checkpoint reports are not 384 independent replications. The eight repair schedules share a single checkpoint. A fresh optimiser in a continuation limits interpretations about the original training trajectory.

The failed prespecified mechanistic replication remains failed regardless of subsequent expanded searches. The observed late repair contrast is inconclusive, and accelerated recovery did not replicate. The 64-model origins study has no scientific outcomes yet. Scope remains the studied synthetic grammar and architecture; generalisation beyond them is open.

The existing manuscript remains unchanged. No direct mechanistic equivalence to Order-to-Role is established.
