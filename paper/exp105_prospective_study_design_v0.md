# EXP-105 — Prospective targeted-exposure × structural-composition transfer

**Status:** DESIGN DRAFT v0 — for joint review. **Not frozen, not preregistered, not a permission to train or open protected tests.**

## 1. Why this study exists

ML-080–ML-100 repeatedly show targeted-exposure benefits in one synthetic relational task family. EXP-103 extends the behavioral pattern to a different small synthetic grammar under familiar modifiers. EXP-102–104 show that batching/scheduling effects can be large early but are not consistently superior at a fixed, source-matched endpoint. The decisive missing evidence is whether **targeted exposure to one grammatical configuration transfers to a distinct, unseen composition of independently learned grammatical elements**.

**Scientific question:** Does targeted experience with a difficult event-position/voice combination increase systematic agent/patient generalization to that combination embedded in an unseen modifier composition, over and above equal-size alternative grammatical experience?

**Primary causal contrast (proposed):** matched `target-exposure` versus `familiar-exposure` training, using the *same fixed training schedule in both arms*. Batch scheduling is not manipulated in this study.

## 2. Hypotheses and falsifiers

- **H1 — transfer:** On a newly generated, held-out four-name-set test with second-event passive and same-event L+I modifiers, the target arm has higher mean `C_query` than the matched familiar arm. This must be evaluated at the endpoint frozen **before** protected evaluation.
- **H2 — specificity (secondary):** Improvement on this target construction exceeds any change in matched familiar-construction performance; analyze the arm × construction interaction at seed level.
- **H3 — error decomposition (secondary):** Report separately changes in wrong-event selection, wrong-role selection, same-answer copying, and exact two-query role accuracy. Do not predict that every error mode will improve.
- **Falsifiers/alternative explanations:** no transfer to L+I despite gains on familiar modifiers; failure of both arms to learn familiar grammar; gains disappearing under lexically matched training-dose controls; gains explainable by answer-position or participant-frequency shortcuts; apparent effect driven by one or two seeds.

## 3. Synthetic grammar and split logic

Reuse the **concept** of the EXP-102 two-event generator, but create a new independently partitioned experimental dataset. Each source sentence contains two distinguishable events with four distinct agent/patient participants and four questions (`first/second × agent/patient`). Active and passive voice are available; modifiers are `bare`, `L` (locative), and `I` (intervening mention). The **within-event L+I combination** is absent from any training source; it appears only in deliberately defined evaluation strata. The training generator must still expose `L` and `I` individually, including both orders *across different events*.

**Construction separation:**

| Evaluation stratum | What it probes | Status |
|---|---|---|
| AA/PA with familiar modifiers | Base grammatical competence | Development gate; both arms |
| AP with familiar modifiers | Transfer to the targeted position × voice combination | Development diagnostic / prespecified secondary independent test |
| AA/PA with same-event L+I | Modifier compositionality without target AP | Independent test stratum |
| **AP with same-event L+I** | **Joint position/voice × modifier transfer** | **Proposed primary protected stratum** |

The exact set of voice cells eligible for base training must be fixed after generator design. In particular, the *familiar* arm must not accidentally see the target AP configuration through ordinary base examples. Both arms must see the individual lexical tokens and role labels; only the intentionally targeted grammatical relationship differs. An equal-size replacement dose should match agent/patient frequencies, verb and participant marginals, and number of training labels. Token-sequence lengths and target-token position histograms need explicit diagnostics, with residual positional differences disclosed.

**Fresh split:** use new four-name-set partitions (and ideally a distinct set of participant names) rather than reopening the prior EXP-102 reserved partition. Keep training, development, and protected four-name combinations disjoint; do not generate protected example text until authorized. Also include a distinct lexical-recombination axis if the preflight can guarantee balanced tokens and enough examples.

## 4. The two intervention arms

1. `familiar`: a fixed number of extra familiar-configuration examples, matched in grammar components, labels, lexical items, and size.
2. `target`: the same number of extra examples exposing second-event passive role questions, without any same-event L+I training examples.

Both use the same source pool for common training examples, identical model initialization within seed, optimization hyperparameters, query balance, update schedule, and number of total labels. **Do not add a quartet-versus-distributed factor:** EXP-103/104 already show scheduling adds variability. Select one balanced policy using an independent development feasibility pilot and freeze it. Optionally include a third *dose-matched diverse familiar* control if doing so isolates a meaningful coverage alternative; do not introduce extra arms without a distinct falsifiable purpose.

## 5. Primary endpoint and analysis

- **Primary endpoint:** `C_query` = fraction of held-out AP+L+I source sentences where **all four** first/second × agent/patient answers are correct under a predeclared four-candidate evaluation. Report full-vocabulary accuracy alongside it; do not conflate constrained output with generation.
- **Independent replication unit:** paired initialization seed. Source sentences and individual questions are not independent training replications.
- **Primary estimand:** mean seed-paired `target − familiar` difference in protected AP+L+I `C_query` at a predeclared final checkpoint.
- **Analysis:** all seeds included; seed-level differences and confidence intervals, exact paired sign-flip statistic when appropriate; separate exploratory family for other strata and error metrics. Avoid protected checkpoint selection, threshold tuning, and post hoc seed exclusion.
- **Seed count:** determine during development using variability from entirely separate pilot seeds, then freeze a fresh confirmatory seed panel. A **minimum target of 16 fresh paired seeds** is a planning assumption, **not yet the registered sample size**.
- **No inference from old protected results as if they were newly held out.**

## 6. Mandatory audits before training

1. Independent parser validates every development source and counterfactual role/voice transformation.
2. Lexical, label, participant, verb, question-type, voice, and modifier frequencies audited across arms.
3. Four-name-set partitions disjoint; full training prompt overlap versus development and future protected selection rules audited without generating protected text.
4. All evaluation vocabulary in training; verify positional coverage (not only global maximum length), particularly for agent/patient mentions under L+I.
5. Shortcut baselines (event-only, surface-order, role-blind, name-frequency, position-based) fail full `C_query`; parser oracle passes.
6. A disposable smoke fit on a tiny *development-only* example pool establishes implementation correctness, not generalization.
7. Code, split metadata, seed schedule, optimizer, training horizon, score definitions, competence gate and one-time protected-access mechanism are frozen by hashes **after** development and **before** confirmatory training.

## 7. Stop/go rule and protected safety

- First perform **development-only** training and evaluate familiar grammar competence without opening any protected AP+L+I examples. Decide feasibility, endpoint, and cohort-level gate in advance of the confirmatory run.
- Run fresh confirmatory training once the protocol is frozen; inspect **only authorized development metrics** for the predeclared competence gate.
- **If the cohort gate fails**, report the failure and stop; do not change the gate, select the best seeds, lengthen training post hoc, or open protected tests.
- **If the gate passes**, a separate explicit decision and a single-use protected evaluator may be created/reviewed. Passing the gate is not itself permission to open the test.
- Existing EXP-101/102/103/104 reserved and protected partitions stay untouched.

## 8. Robustness after primary study

If the primary test supports transfer, replicate the principal result on a second grammar or architecture and test a prespecified exposure dose-response. Only then consider pretrained LLM/LoRA transfer. The prior Qwen/SLOG development pilot had severe option-order sensitivity, so prompt measurement must be fixed before drawing model-level role-binding conclusions.

## 9. What we will do together next

**Milestone A (now):** inspect/fix the generator design and agree on the exact `base` vs `dose` example pools; resolve whether target AP is absent from both shared base training and the familiar arm. This is the most important leakage check.

**Milestone B:** implement `scripts/exp105_generator_preflight.py`, with no model training and no protected example generation; produce an auditable balance and position report.

**Milestone C:** run independent development feasibility pilots, then freeze a final protocol and fresh seed panel before confirmatory training.

**Milestone D:** guarded evaluation, full paired analysis, counterexample analysis, and manuscript figures.

**This document is a plan, not evidence of experimental success or a frozen protocol.**
