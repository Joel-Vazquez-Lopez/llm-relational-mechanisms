# From EXP-102 to a defensible relational-learning study

**Working design memo — 13 November 2026**  
**Status:** proposed next study, NOT frozen, NOT an authorization to run a new protected test.  
**Sources:** existing EXP-102 exploratory runs, the EXP-101 research record, and public COGS/SLOG literature.

## 1. What has actually been established

The eight previously unused EXP-102 seeds reported on the existing *four-name-set-OOD* development split showed:

- `C_query` at step 1600: `quartet` 1.000 in 8/8 seeds; `balanced_unpaired` 1.000, 1.000, 0.000, 0.375, 0.000, 0.988, 1.000, 0.762 (reported to three decimals).
- Mean paired difference approximately +0.3594 (35.94 percentage points); wins/ties/losses 5/3/0. On rounded values, a two-sided sign-flip test is p=0.0625; do not call this a definitive confirmatory finding.
- Source membership is distinguished from semantic-role identity: the weaker runs frequently identify the correct event while copying the same answer for complementary roles.
- Quartet superiority is about the **training policy** as a whole. Full quartets sample approximately 16 source sentences/update; balanced-unpaired draws up to 64 source sentences/update. They differ in unique source diversity, co-occurrence structure, and the resulting stochastic gradients. The design does not isolate a single cognitive mechanism.
- The OOD split withholds four-name combinations, **not unfamiliar syntax**. Four-name forced-choice output also simplifies the task relative to natural-language generation.
- This remains exploratory: the task family and development pool had been examined many times before the fresh seed panel. It is not a new independent linguistic-generalization benchmark.

## 2. Research question for the next study

**Does structured, complementary-question sampling improve a Transformer's reliability and sample efficiency in learning event-conditioned semantic roles, and does that advantage transfer to held-out syntactic compositions?**

Important alternative explanations: (A) source repetition / low source diversity; (B) different gradient variance; (C) balanced query frequencies; (D) memorized surface strategies; (E) positional-embedding exposure artifacts; (F) flawed label projection or evaluation asymmetry.

## 3. Two components, in this order

### Study A — close out EXP-102 rigorously (no new training)

1. Run `exp102_evidence_audit.py` against all 32 fixed reports. Verify protocol hash, source hashes, seed IDs, steps, and full precision at each recorded milestone.
2. Report the preselected step-1600 paired difference, descriptive paired-seed bootstrap interval, exact two-sided sign-flip statistic, 5 wins/3 ties, and individual seed results. Label interim learning-curve results **secondary**.
3. Preserve both successes and failures, including the failing EXP-101 gate. Do not change the earlier endpoint based on the trajectory.
4. Make an evidence map with: observation → supporting report → competing explanations → experiment that would discriminate them.

### Study B — separate prospective compositional-transfer experiment (NEW protocol needed)

**Conceptual generator:** two events with four distinct core participants, active/passive alternation, and independently parameterized *L* (locative) and *I* (intervening) distractor modifications. Use all four questions about the same source: first/second event × agent/patient. Every name should be a plausible distractor in the rendered surface, and the gold must be recoverable by an independent text parser.

**Training/development:** ensure both arms learn the base grammar; expose each individual modification (*L*, *I*) in training, but not their joint composition (*L+I*). Hold out participant combinations and recombined verbs as separate factors. Crucially, all tokens used at test must already have been seen in training, and test token positions must be represented during training—or be explicitly identified as a separate positional-generalization variable. This addresses problems encountered in EXP-101.

**Protected evaluation:** freeze a new set of *L+I* sentences, independent of development. Evaluate `C_query` (all four answers correct on the *same source*) as the primary semantic-role endpoint, alongside marginal accuracy, both-role correctness per event, role-copy rates, and counterfactual consistency. Never select seeds or stopping points using protected performance.

**Preflight, before training:** independently parse every rendered sentence and question; verify all active↔passive and role-reversal transformations preserve or change the exact gold as intended; measure source/role/voice/position balance; confirm anti-shortcut baselines fail `C_query` and that a parser oracle succeeds; test shapes and positional coverage; run a tiny competence smoke test on a separate *development-only* generator.

**Initial training contrast:** compare quartet and balanced-unpaired batching with the **same** model, data, optimizer, loss, steps, evaluation, and initial weights across each seed. Describe the manipulation honestly as a **batch-sampling policy** contrast; the unique-source-count difference remains a possible mediator/confound. A distinct factor-isolation study would be needed to claim a uniquely causal effect of co-occurrence.

**Proposed decision rule to freeze before protected data:** an adequate development competence rule must be chosen using independent development-only pilots, with no access to the protected *L+I* cases. Do not reuse EXP-101's unsuccessful 95%/90% nine-cell gate unquestioningly. Explicitly define what happens when competence fails (report gate failure; do not optimize to protected performance).

**Statistics:** seed is the independent replication unit; use matched-seed differences with uncertainty, full distributions and multiple-comparison discipline. Report per-step trajectories as prespecified secondary analyses if desired. An eight-seed panel is useful but does not automatically provide definitive power, especially with ties and strong between-seed heterogeneity.

## 4. Pretrained-LLM connection: separate validation, not assumed equivalence

The previous Qwen2.5-1.5B-Instruct/SLOG development pilot yielded only 3/16 sentences with both role answers invariant to A/B option order. Therefore its apparent same-person responses **cannot be equated** with the synthetic role-copy mode.

1. **Measurement calibration first:** choose development-only two-event sentences with independently checked agent/theme labels. Score both role questions for the same verb/event. Shuffle candidate order, randomize participant names, balance first/second mentions, and test multiple prompt templates. Track exact name predictions, invalid outputs, option order invariance, event selection, role copying, and complete four-query correctness.
2. **Run pretrained baseline without fine-tuning.** The aim is to characterize whether a stable, prompt-robust behavioral error exists. Do not claim the synthetic and pretrained models share an internal circuit.
3. **Only after measurement is stable**, consider a small, matched LoRA/SFT experiment with quartet versus balanced-unpaired batching, equal optimizer updates and token budgets, a frozen development gate, fresh held-out grammatical compositions, and independent random seeds. Batch grouping at training is **not** equivalent to showing four questions in a single inference prompt.
4. Keep this LLM work as a distinct study or external-validation section unless it achieves a genuinely robust protocol. The old SLOG protected split remains closed under the previous stop decision.

## 5. Required evidence before an ambitious paper claim

- Reproduction of the batching effect on a fresh, prevalidated structural generalization task.
- Robustness against lexical/positional/distractor heuristics and option-order prompt artifacts.
- Explicit accounting of source-diversity / gradient-noise alternatives.
- Honest separation of behavior (observed), causal effect of the whole batching policy (experimentally manipulable), and internal mechanism (not yet established).
- Literature positioning against COGS (Kim & Linzen, EMNLP 2020), SLOG (Li et al., EMNLP 2023), and work using paired examples as supervision (Gupta et al., EMNLP 2021). The last employs an additional latent-consistency objective; that is **not** the same training intervention as ordinary supervised minibatch grouping. Novelty still needs a broader literature audit.

**Suggested title, provisional:** *When Does a Transformer Learn Who Did What to Whom? Batch Structure, Role Collapse, and Compositional Transfer.*

## 6. Immediate next action

Run the read-only evidence audit from the repository root. It reads existing JSON only. Review its output and the source-confound caveat **before** writing a new generator or launching another multi-seed training study.

Sources:
- COGS: https://aclanthology.org/2020.emnlp-main.731/
- SLOG: https://aclanthology.org/2023.emnlp-main.194/
- Paired Examples as Indirect Supervision: https://aclanthology.org/2021.emnlp-main.466/
