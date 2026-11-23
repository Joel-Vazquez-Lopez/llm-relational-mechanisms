# Publication decision after the prospectively successful training-surface factorial

**Updated:** 2026-11-23 after complete TRAIN-SURFACE-FACTORIAL-v0 results. **Status:** publication planning; not a frozen protocol for any future study. New model performance must not be inspected or used to choose a preferred condition until a fresh protocol is frozen. Existing protected/reserved/L+I gates remain CLOSED. Separate from Order-to-Role.

## New result and publication decision

TRAIN-SURFACE-FACTORIAL-v0 is **COMPLETE**, not a proposed experiment. The 12 independent new seeds and 48 trained models passed the prospective adjunct-associated main-effect primary (+29.86 pp, CI [+13.46,+46.26]) and its gated unseen-quartet secondary (+29.37 pp, CI [+13.56,+45.18]). Absolute heldout role accuracies were 99.21% bare, 84.25% prefixes only, 97.01% adjuncts only, and 26.73% combined. Thus the strongest **exploratory** pattern is a prefix×adjunct interaction of ~+55.32 pp difference-in-penalties, with severe seed bimodality and wrong-event errors in the combined arm. Do **not** promote this post-primary interaction to independent confirmation. The manipulation is a bundled training distribution: token lengths, positional statistics, name mentions and dose of unmodified sources differ.

**Revised recommendation:** immediately prepare a bounded empirical submission with existing **confirmed primary+secondary** and earlier replicated position-dependent interventions, keeping one-grammar scope and identifying all exploratory status. A stronger ACL/EMNLP-style paper would benefit from one *new, separately frozen* direct interaction replication with more carefully controlled exposure frequency, plus a second genuinely distinct grammar before broad generality claims. Do not rerun the same 48-model test or tune its conditions after the result.

## New synthesis

- EXP-109 prespecified: same-clause literal PA vs AP gap **+61.37 pp**, seed-level t CI [+41.33,+81.41] pp. This establishes severe construction sensitivity in one synthetic training regime; it does **not** by itself establish role abstraction.
- EXP-114: position-code and early learned-update interventions causally influence predictions, but the repair was not narrowly selective for roles; see EXP-116 (0/12 selected candidates passed a descriptive selectivity screen).
- POSITION-CB-v0: marginal voice/position balance was not sufficient for robust novel-voice co-occurrence generalization; nominal +14.41 pp primary was extremely borderline and heavily seed-dependent.
- Post-hoc CORE-TRANSFER-v0 (32 old model states): original cores 92–98% joint correct, new within-quartet semantic role/event arrangements ~32–37%; exact bare/bare token-bag controls preserve the ~59–60-pp drop; role flips lead primarily to reversals, event swaps to wrong-event answers.
- Prospective CORE-DIVERSITY-v1 (16 fresh seeds, 64 new models): fixed-core *and* eight-core-diverse arms reach **98–100%** strict joint accuracy on unseen role assignments and unseen four-name quartets using **bare/bare** inputs. Frozen primary diverse-minus-fixed = +1.49 pp CI [+0.50,+2.48], but **FAILED** its ≥10-pp success criterion. Gated secondary remains descriptive only.

**Crucial new fact:** *The fixed-core bare/bare treatment already generalizes.* Therefore, training only one role assignment per quartet cannot, by itself, explain the old performance collapse. The large between-cohort difference has several confounds (training modifiers, neutral prefix frames, split cardinality, schedule, implementation), so we CANNOT assert "surface noise caused failure" yet.

## Viable submission routes

**A. Short empirical / workshop paper now:** Theme: *How controlled recombinations expose different failure types, and how a high-complexity training task can appear compositionality-limited while a simpler cohort reaches ceiling.* Use the original preregistered behavioral and causal intervention tests, exact-token controls, reverse/wrong-event error decomposition, and the **negative** prospective core-diversity experiment. Label post-hoc diagnostics unmistakably. This is publishable in principle as a sharply delimited empirical case study; novelty depends on reviewer judgement, and no venue acceptance is guaranteed. Prioritize coherent paper and release of code + manifests + all seeds, not another unconstrained exploratory sweep.

**B. Stronger full NLP conference / Findings attempt:** Execute ONE fresh, fully controlled *training surface-factorial* (below), then a genuinely distinct grammatical family if claiming generality across grammars. This is stronger but demands new evidence; prior work already investigates data complexity, shortcut strategies and compositionality.

## Historical prospective design: TRAIN-SURFACE-FACTORIAL-v0 (now completed)

### Hypothesis

Under a fixed source-core schedule and identical semantic labels, adding irrelevant name-bearing adjuncts and/or variable neutral sentence prefixes during TRAINING may steer model learning toward less-transferable input-position/name associations. The effect might be an interaction between surface complexity and which role configuration is repeated, rather than an inherent inability of the architecture to perform role composition.

### Factorial design

**Hold constant within the new experiment:** same preflight generator, 240 TRAIN + 60 withheld four-name quartets from the old TRAIN identifier partition; same original core assignments, seed grid, model architecture and code path, 6,400 updates, two-voice exposure regime, label vocabulary, query definitions, training examples per step, and frozen evaluation source/answer sets.

**Factors:**
- Variable neutral **prefix framing** during training: absent versus the five old training prefix alternatives.
- **Named adjunct/distractor** use during training: absent versus balanced old \`bare/L/I\` modifier combinations without protected \`LI\`.

Four training arms: bare, frame-only, adjunct-only, frame+adjunct. No core-diverse treatment in the first factorial: all models use the SAME fixed underlying event core per training quartet, because the purpose is to isolate conditions under which previously competent fixed-core models break down.

**Compute-conscious preference:** choose ONE original voice-pairing regime before freezing and avoid comparing more than the four necessary training conditions. Candidate: mixed_voice AP+PA to retain both active/passive voices at each event position, but it should not be selected on the basis of favorable post-hoc effects. Use a fresh planned grid, e.g. 12 independent seeds ×4 arms, after a *source-only* generator preflight and training-only feasibility pilot. Do not reuse 139xx CORE-DIVERSITY seeds as independent confirmations.

**Matching claims should be precise:** four-arm total update count, supervised labels, train quartet selection and semantic assignment are equal. Vocabulary can be shared explicitly; sentence token counts, absolute positions, and adjunct-name mentions cannot all be matched when testing their presence/absence. Those are the intended manipulated factors; list the resulting distributional differences rather than falsely claiming perfect matching.

**Evaluation:** freeze all inputs and source hashes before any model inference. Primary measure strict four-question joint \`C_query\` on never-trained participant-role permutations from the same 240 quartets, in each arm's familiar voice construction. Hold-out transfer to the 60 totally unseen quartets is a gated second measure. Include event wrong-role/wrong-event pair classes, train-core proficiency and a position-only heuristic baseline, with no seed exclusions. Use the SAME test inputs for all four training arms (ideally a prespecified bare/bare common evaluation plus a richer surface-form robustness set as distinct endpoints), so no difference is due merely to different test samples.

**Primary contrast and thresholds to decide and freeze BEFORE training:** factorial main effect of named adjuncts averaged over prefix states, on familiar-voice generalization under fixed core; positive "degradation" means bare-trained outperforms adjunct-trained on the shared heldout role task. Freeze a meaningful absolute drop threshold, expected seed variance/power and a multiple-comparison/hierarchical testing plan. The prefix main effect and interaction can be prespecified secondary/exploratory. Report every seed, even if some models fail to fit original cores.

**Blind pilot and stop rule:** generator/oracle/parser/overlap validation first; a short *train-only* feasibility pilot may check source tensors, gradients, loss and in-distribution fit on disjoint pilot quartet IDs; it must NEVER inspect intended OOD outcomes or be recycled as confirmatory data. If full training shows no difference, report the null; do not tweak distractor frequency until it produces the desired failure.

### Reviewer-safe interpretation

- If added surface complexity robustly reduces heldout role correctness, without a corresponding loss on trained cores, we have a prospective causal argument for **training-distribution-induced shortcut sensitivity**.
- If the effect depends on the interaction between prefixes and name adjuncts, that is a sharper claim but must be tested with appropriately controlled multiplicity.
- If there is no strong surface effect, the old/new discrepancy remains unresolved. Publish a limited exploratory case study; do not invent a causal explanation.
- Even a strong one-grammar factorial does not prove a specific attention circuit or general pretrained LLM deficiency. Independent grammar replication and mechanistically specific intervention would strengthen a broad full-conference case.

## Relevant prior work (must distinguish our contribution)

- Kim & Linzen (EMNLP 2020), **COGS**, tests novel combinations of familiar words/structures: https://aclanthology.org/2020.emnlp-main.731/
- Li et al. (EMNLP 2023), **SLOG**, extends structural semantic parsing generalization: https://aclanthology.org/2023.emnlp-main.194/
- **Data Factors for Better Compositional Generalization** (EMNLP 2023), examines effects of scale, pattern complexity and example difficulty: https://aclanthology.org/2023.emnlp-main.898/
- **Analyzing the Inner Workings of Transformers in Compositional Generalization** (NAACL 2025), examines syntactic sensitivity and non-compositional algorithms: https://aclanthology.org/2025.naacl-long.432/
- Kawata et al., **From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers**, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/6499b639e8a4b5c9a780d9b88c09722f-Abstract-Conference.html
- Tang, Lake & Jazayeri (PLOS ONE 2026), **Circuit explained: How does a transformer perform compositional generalization**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340088

**Positioning:** novelty cannot be "Transformers take shortcuts" or "data diversity helps." The distinctive target, if established, is **how controlled nuisance surface elements interact with semantic role/event selection in a two-event answer task**, alongside pre-specified role-reversal versus wrong-event errors and independent controlled positional interventions.

## Manuscript structure after results are finalized

1. Motivation: semantically equivalent sentence transformations can yield conflicting role predictions.
2. Controlled two-event grammar, competing positional shortcuts, model families and gold parser.
3. Original prespecified construction gap and robust seed-level variability.
4. Error composition and simple positional-baseline challenge.
5. Existing causal position/update interventions, including failure of specific selective repair.
6. Core-transfer controlled transformations and token-bag-matched negative control (post-hoc).
7. Prospective CORE-DIVERSITY-v1 **failed primary**, with near-ceiling fixed-core discovery.
8. If executed: new prospective training-surface factorial and its specified endpoint.
9. Limitations: all studies are small scratch Transformers in one grammar; extensive researcher degrees of freedom in older exploratory studies; no uniquely identified role circuit.
10. Conclusion: bounded behavioral claim, not an architecture-universal impossibility theorem.

## Immediate execution order — revised after completed factorial

1. **DONE:** archive independently verified 12 full reports and frozen six-file experiment inputs.
2. **DONE:** update manuscript, evidence ledger, README and audit to report successful prespecified primary and gated secondary, plus honest exploratory interaction and seed heterogeneity.
3. **NEXT:** perform a final independent numerical/figure audit against the manuscript and identify a target publication venue. Finalize a narrow, accurate paper first.
4. **OPTIONAL STRONGER PAPER:** separately freeze one **new** interaction-focused study testing frequency/length/position controls; then test a distinct grammar for generality. Do not label the already-seen interaction confirmatory.
