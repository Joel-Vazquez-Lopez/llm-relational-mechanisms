# Targeted Experience, Relational Generalization, and Role-Assignment Errors in Transformers

**Working manuscript — 12 November 2026.** Target first complete draft: 15 December 2026. This file is the repository's editable working draft; it does **not** replace the separate manuscript package on the local Mac. Numeric results in Sections 3–4 are backed by the archived reports cited below. The proposed SLOG analysis has **not** produced model results.

## Working abstract

Generalizing a linguistic relation requires selecting the relevant event and distinguishing its participants' semantic roles. We study how targeted grammatical experience changes relational question answering in small, scratch-trained causal Transformers. Across six paired cohorts using a shared synthetic two-event task family, the targeted-training arm improves held-out four-choice accuracy in all 48 seed pairs. Paired questions expose additional structure: across five cohorts with archived joint-response metrics, both-role correctness improves in all 40 matched seed pairs, but the same-participant answer rate does not reliably decline and increases on average in each cohort. In a separate calibration-only follow-up on an eight-seed cohort, correct-event-restricted scoring reveals a shift from reversed to correct assignments, but no reliable reduction in collapsed assignments. A different controlled-exposure experiment reveals that such collapses can largely disappear under another training design, precluding a universal failure claim. Layerwise analyses demonstrate intervention sensitivity but do not establish a replicated, unique mechanism: the prospectively tested layer-2 role-query interaction fails its success rule, and a bounded source/access intervention screen yields no passing candidate. We delineate what these experiments establish, and design a separate external linguistic validation for pretrained models.

## 1. Introduction [DRAFT TO COMPLETE]

Relational language understanding requires coordinating event reference and semantic-role selection. Marginal question-answering accuracy can hide qualitatively different response patterns: selecting the wrong event, reversing agent and patient, or answering two complementary queries with the same participant. Controlled training contrasts help isolate the effect of exposure, while layerwise interventions can test candidate explanations; neither alone establishes generalization to naturally occurring language or large pretrained models.

**Research questions:** (RQ1) Does repeated experience with a particular grammatical configuration reliably improve out-of-exposure relational answering? (RQ2) How does the distribution of joint-role errors change? (RQ3) Which aspects of the observed improvement are captured by independently tested layerwise explanations? (RQ4, prospective) Does a comparable role-error taxonomy reveal informative variation in external, structurally diverse sentences and pretrained language models?

## 2. Methods [DRAFT TO COMPLETE]

### 2.1 Controlled synthetic learning
The principal experiments use three-layer, pre-layer-normalization causal Transformers (width 96; four attention heads) trained from scratch on a two-event, four-participant synthetic grammar. Complementary questions select the first/second event and agent/patient role. Paired training arms differ in repeated exposure to 64 distinct first-position-passive versus second-position-passive training questions while other design choices are matched within each cohort. Models run for 12,800 training steps. Held-out combinations are evaluated under both four-name forced choice and unrestricted vocabulary scoring. Cohorts vary training seeds, names, and held-out participant-pair partitions, but share the basic architecture and grammar family; thus 48 paired seed effects are not 48 independent linguistic tasks.

### 2.2 Joint-role outcomes
For each event, predictions for the agent and patient questions are paired. Outcomes distinguish (i) both correct, (ii) exactly one correct, (iii) neither correct, and (iv) selecting the same name for both roles. When candidates are restricted to the **two gold participants of the correct event**, outcomes form a mutually exclusive and exhaustive partition: *correct assignment*, *reversal*, or *collapse* (same participant for both roles). Oracle restriction changes the task by removing event and output competition; do not equate the restricted and native prediction metrics. Statistics and intervals are calculated across matched **model seeds**, not treating multiple prompts or examples from the same trained model as independent replicates.

### 2.3 Limits on interpretation
Exploratory post-hoc analyses of already examined calibration examples are not treated as protected confirmatory tests. The product of marginal agent and patient accuracy is a descriptive independence reference, not a robust null model in the presence of correlated sentence difficulty. Activation patching establishes sensitivity to an intervention, not mediation, causal necessity, or a unique native circuit without appropriately controlled independent evidence.

## 3. Results

### 3.1 Targeted exposure consistently improves held-out relational answering

Six separately trained paired cohorts (ML-080, ML-084, ML-085, ML-086, ML-088, ML-099; eight seed pairs each) show positive target-minus-familiar four-choice differences in 48 of 48 pairs on held-out target-construction evaluations. Across those cohorts, reported mean gains are approximately +35.6 to +49.2 percentage points, depending on cohort. Performance on unrestricted vocabulary predictions also improves. This is a repeated result **within one synthetic task family**, not an external-benchmark or pretrained-LLM validation.

Source reports: `reports/ml080/protected_evaluation.json`, `reports/ml084/protected_evaluation.json`, `reports/ml085/protected_causal_evaluation.json`, `reports/ml086/protected_evaluation.json`, `reports/ml088/protected_evaluation.json`, `reports/ml099_role_q/protected_evaluation.json`.

### 3.2 Cross-cohort reanalysis: consistent joint improvement, persistent native duplicate answers

We performed a final, read-only exploratory reanalysis of five previously archived protected evaluation reports (ML-080, ML-084, ML-085, ML-086, ML-099; eight matched seed pairs per cohort). Under native four-name candidate scoring on the passive second-event target construction, individual-answer and both-role correctness improved in **40/40 paired seeds**. Mean improvements in the rate of answering *both* complementary questions correctly were **+26.6**, **+20.8**, **+28.2**, **+20.1**, and **+34.8 percentage points**, respectively. The native per-question rate of choosing the opposite-role participant from the correct event decreased by **23.6**, **16.2**, **11.6**, **23.0**, and **19.8 points**, respectively (**33/40** seed pairs showing a decrease). Per-question wrong-event choices decreased in **34/40** seed pairs.

In contrast, native **same-participant answers** to agent and patient questions did not reliably disappear. Their cohort-mean target-minus-familiar differences were **+2.9**, **+15.1**, **+15.5**, **+6.9**, and **+1.0 percentage points**; increases occurred in **26/40** matched seed pairs. This contrast shows why improving marginal accuracy or even both-role correctness need not eradicate a particular joint response pattern. One possible contributor is the redistribution of predictions from the distractor event toward candidates from the target event; the archived aggregates cannot distinguish that explanation from other sources of correlation.

Critically, a **wrong-role prediction** here refers to a *single question* selecting the other participant of the correct event. It is **not** synonymous with swapping agent and patient on the two-question pair. Accordingly, the oracle-restricted reversal effect in Section 3.3 is not replicated directly by this cross-cohort reanalysis. ML-088's protected report did not retain paired-role statistics and is therefore excluded from this five-cohort comparison. All five cohorts use the same synthetic task family; the 40 seed pairs are not independent linguistic generalization tasks.

Reproducible audit: `scripts/paper_cross_cohort_audit.py`, `reports/paper_cross_cohort/archived_error_taxonomy.md`, and the five archived source reports. This analysis reused existing protected **summary statistics** only; no models were rerun or protected examples regenerated.

### 3.3 Targeted exposure changes the structure of role-assignment errors (exploratory calibration analysis)

We analyzed previously examined calibration sentences with 16 ML-088 checkpoints (eight paired seeds) and complementary agent/patient questions about the passive second event. Under correct-event two-name restriction, correct agent–patient pair assignments rose from **7.9%** (familiar exposure) to **39.8%** (target exposure), a mean paired difference of **+31.9 percentage points** (seed-bootstrap 95% interval **[+8.7, +58.6]**; seven of eight positive). Reversed assignments declined from **39.5%** to **9.9%**, a paired difference of **−29.6 points** (95% interval **[−45.6, −14.1]**; seven of eight seeds show a reduction). Collapsed assignments remained frequent: **52.6%** versus **50.3%**, a paired difference of **−2.3 points** (95% interval **[−14.7, +9.8]**). These three outcomes sum to 100% in each condition; their changes must not be treated as independent statistical discoveries.

On unrestricted native four-name predictions, individual second-event-passive answer accuracy increased from **14.0%** to **50.5%**, while accuracy for *both* answers rose from **0%** to **24.9%**. Both queries chose members of the correct event in **39.6%** versus **47.4%** of cases (paired +7.8 points, interval spans zero). The native same-participant rate was **25.3%** versus **22.9%**, with no reliable change. Inferences about role representation cannot be directly transferred from the oracle-restricted task to native decoding.

The paired calibration data therefore support a *redistribution of assignment outcomes*: improved role directionality, but a persistent collapse mode in some seeds. Since these are previously examined calibration examples, this is an **exploratory behavioral analysis**, not new independent confirmation. Reference: local analysis `reports/paper_role_consistency/summary.json` and per-sentence `calibration_observations.json` (these local files must be archived with provenance before final publication).

### 3.4 Collapsed assignments are not universally persistent in a different synthetic regime (ML-072)

The previously completed ML-072 study used a different synthetic grammar/training design, three exposure conditions (control, sparse, broad), and six paired seeds. On combinations simultaneously unseen in participant pairs and verbs, two-candidate correct-event scoring yielded **34.3%** correct assignments, **11.1%** reversals, and **54.6%** collapses in the control arm. Corresponding results were **94.9% / 0% / 5.1%** with sparse target exposure and **98.6% / 0% / 1.4%** with broad exposure. This independent experimental setting demonstrates that the collapse mode can become rare under another training design. Because training contrasts, grammar, evaluation pools, and checkpoints differ, the between-study discrepancy does **not** isolate which design factor causes it. The ML-075 richer-grammar gate failure remains part of the evidence record and is not overridden by ML-072.

Source: `reports/ml072/evaluation.json`, `reports/ml072/frozen_protocol.json`; see `reports/ml075/competence_gate.json`.

### 3.5 Layerwise interventions constrain, but do not uniquely explain, the learning effect

ML-085 role-donor interventions have a substantial control-relative effect in the protected cohort, but this does not establish mediation or necessity; ML-086's erasure result has a confidence interval spanning zero and overlapping role/event subspaces. A selected exploratory L2 question-path effect (ML-098) failed its independent prespecified replication in ML-099: the primary role-Q exposure-by-voice contrast was positive for **5/8** pairs, mean **+7.72** logit units, with 95% seed-bootstrap interval **[−3.29, +19.87]**, not meeting its frozen criterion. In ML-100, none of six bounded source/query/joint candidate screens across layers 2 and 3 passed the compound selection rule. Negative specificity results exclude the tested proposals under their tested conditions, **not all** possible layerwise mechanisms.

References: `reports/ml085/protected_causal_evaluation.json`, `reports/ml086/protected_evaluation.json`, `reports/ml099_role_q/protected_evaluation.json`, `reports/ml100_source_access/exploratory_summary.json`.

## 4. Prospective external-validation study [PROTOCOL DEVELOPMENT — NO RESULTS]

We intend to inspect the **public SLOG development split only** to determine whether a subset of semantic representations supports an unambiguous pair of `agent` and `theme` arguments for a single identifiable predicate. SLOG's annotated *theme* is **not literally identical** to the synthetic task's *patient*: terminology and semantic scope must be stated explicitly. The SLOG structural generalization test archive must remain unopened until eligibility, extraction, prompts, checkpoint IDs, scoring, and exclusions are frozen. The public development-split extraction preflight has **passed**. From 4,046 development sentences (SHA256 `177128c3f64be56732a102db119139bbec33a771c485a525ea0f9f4e45926a5c`), 812 contain an eligible agent–theme role pair within a multi-predicate sentence; 462 have two eligible events and only 53 have two events with four distinct participant heads. The dominant eligible category is object-modifying relative clauses (613); in-distribution sentences contribute 149, center embedding at depth 2 contributes 29 and at depth 4 contributes 14, complement-clause depth 4 contributes six, and prepositional-phrase depth 4 contributes one. None of the public `wh_Q_simple_trans` rows passes these conservative multi-event conditions. These are eligibility counts, **not** scored performance. A deterministic public-dev label audit has now selected 50 unique sentences for QA projection review; the script's indexing self-test passed, the printed 14 sample projections were visually plausible, and the local Mac reports MPS availability, but full manual validation remains outstanding. The publicly released dev file includes `in_distribution` plus six named structural categories; the categories are recorded as metadata, while any model/prompt calibration should use `in_distribution` only. A subset that collapses to trivial single-clause sentences would not validate our two-event binding claim; report such failure honestly and consider a different benchmark before testing models.

A future pretrained-model evaluation must specify fixed model revisions, whether the output is unconstrained generation or candidates, role-specific baselines, event-conditional scoring, in-domain controls, structural category stratification, and a stopping rule. Crucially, a pretrained-model evaluation would test **error-taxonomy transfer**, not replicate the **causal effect of training exposure**, unless a controlled exposure intervention is separately designed.

Official source: https://github.com/bingzhilee/SLOG ; benchmark paper: Li et al., EMNLP 2023, https://aclanthology.org/2023.emnlp-main.194/ .

## 5. Discussion and limitations [TO WRITE]

- Cohort replication is within a narrow synthetic family.
- Seed variability is large; a few seeds contribute outsized individual effects, though the qualitative reversal reduction spans seven pairs.
- Oracle-restricted decisions do not measure native LLM performance.
- An absence of success for selected intervention hypotheses cannot establish absence of role computations.
- Generalization to realistic structural phenomena and pretrained LLMs remains **unverified** pending a genuinely held-out external study.

## 6. Figure plan

1. **Main Fig. 1:** Six-cohort held-out exposure benefit (48 paired seed effects) and full-vocabulary comparison.
2. **Main Fig. 2:** Paired-role error taxonomy: native versus oracle-restricted answers for ML-088 calibration and each seed; ML-072 as a clearly marked, non-controlled cross-study comparison.
3. **Main Fig. 3 or supplement:** ML-099 prespecified interaction failure and heterogeneous seed effects.
4. **Supplement:** ML-100 control-relative specificity screening and null outcomes.
5. **External benchmark figure:** **Reserved; do not create results panel before evaluation.**

## References [TO FORMAT]

- Kim & Linzen (2020), COGS, EMNLP, https://aclanthology.org/2020.emnlp-main.731/
- Wu et al. (2023), ReCOGS, TACL, https://aclanthology.org/2023.tacl-1.96/
- Li et al. (2023), SLOG, EMNLP, https://aclanthology.org/2023.emnlp-main.194/

## Evidence and analysis status

Archived experiment reports: repo-accessible. ML-088 paired-role local observations and summary: user has generated files on their Mac; **not yet verified committed in this branch**. SLOG external dataset/model results: **not run**. Do not imply any protected SLOG scores have been seen.
