# What could materially strengthen the construction-sensitive roles paper?

**Authoring date:** 2026-11-21. **Status: RESEARCH DESIGN NOTE — NOT A FROZEN PROTOCOL OR EXPERIMENT.** No study described here has been run. Do not interpret recommendations as evidence. This note does not modify existing experimental plans and does not authorize access to protected/reserved or L+I evaluations.

## Current evidence and risk

- EXP-109: a large prospectively evaluated literal clause-swap construction gap; same two-event grammar and four-name candidate task.
- EXP-110: post-hoc native role reversals predominate over wrong-event errors in the difficult AP construction.
- EXP-114: new-seed/disjoint-dev-surface intervention replication of input positional edit, first-block update versus late explicit position and reciprocal transfer, all three frozen tests passed.
- EXP-116: 0/12 candidate partial-state dose conditions passed a descriptive selective-repair screen.
- PHASE2-01: its 20%-direct primary failed; full-dose advantage was descriptively positive but gated.
- PHASE2-02: the primary full-dose advantage on unseen adjunct orders failed.
- PHASE2-03: its named-distractor specificity primary failed.
- Closest relevant comparators: Kawata et al. (NeurIPS 2025) on shortcut-vs-induction algorithm selection; Tang et al. (PLOS ONE 2026) on a synthetic compositional-generalization circuit; the NAACL 2025 paper *Analyzing the Inner Workings of Transformers in Compositional Generalization* on noncompositional subnetwork algorithms.

**Main reviewer objection:** PA-only training never exposed the model to AP. A large AP–PA gap may be produced by the simple rule “when the target is the second event, take the first mentioned target-event name as agent and the other as patient,” without any abstract binding mechanism. Successful cross-position patches may alter context or event selection rather than repair semantic-role assignment.

## Gate A: inexpensive anti-shortcut tests before new neural training

Construct a fully deterministic, read-only baseline/evaluation analysis on already public non-reserved development source families (without touching protected/reserved/L+I). **Do not** re-score historical checkpoints unless the local owner approves because the saved public aggregates should not be presented as new confirmatory evidence.

1. **Slot-frequency shortcut:** estimate the empirical mapping from (queried event index, queried role) to (first/second core participant position within that event) using only the original training sources and labels. Do not read voice, `was`, `by`, verb identity, or target gold labels at test time. The baseline ignores syntax beyond event boundaries. Report performance under original AP and literal PA and the paired error taxonomy.
2. **Shallow voice-aware rule:** a hand-specified transparent control that identifies passive voice and the `by` phrase (no learned neural representation). It is a *task solvability upper bound*, not a learned-model control and not novel evidence for compositionality.
3. **Counterfactual consistency audit:** apply pre-existing public source transformations (literal clause swap, target role/name swap where available, event distractor relocation). Report exactly which transformed input elements change while gold semantic roles are held fixed or predictably permuted. Quantify whether each baseline reproduces the *qualitative* model failure mode; do not turn descriptive similarity into a hypothesis test.

**Decision A:** If a trivial slot-frequency baseline reproduces most of the AP error-class profile, explicitly foreground that as a partial explanatory account. If it cannot, identify the specific residual phenomena requiring neural explanation. Either result improves scientific clarity; do not discard a null or embarrassing baseline.

## Gate B: a single new grammatical-family replication, only if a wider paper is justified

Design a **second synthetic grammar family** that preserves two-event/agent/patient semantics and complementary questions but has different clause templates, dependency placement and surface position statistics, not just new names or another adjunct permutation. Example family to prototype: paired subordinate constructions with variable positions of the passive agent phrase and clause order, independently parseable into exactly two disjoint agent/patient pairs. The exact linguistic templates require a separate grammar and parsing validation. A mere `;`→`and` replacement does *not* qualify.

Use **within-family**, independently generated training and evaluation data: the new models should learn the new grammar's familiar constructions before encountering its held-out composition. This tests phenomenon replication *across grammars*, not zero-shot transfer of weights trained on the old grammar. Crucially, balance lexical identities, questions, length/position distributions and event-voice marginals **as far as feasible** across exposure arms. Any unmatched factor must be displayed in a preflight table and considered a competing explanation.

Proposed minimum design, subject to preflight feasibility and seed-based power simulation from archived variation:
- Two training distributions: (i) a position-correlated, AP-omitting analogue of PA-only; (ii) a coverage-controlled distribution with direct exposure or decorrelated positions, maintaining equal supervised examples and vocabulary. We should choose the *one* comparison whose token/position matching can actually be enforced before freezing.
- Eight fresh training seeds and two numerical repeats per seed (32 model states for two arms), fixed update count and hyperparameters, no checkpoint/seed selection or optional stopping. Eight seeds are a continuity reference, *not* a guaranteed adequate power calculation.
- Fresh, non-reserved task-specific test partition; no overlap in source sentences, participant orderings or evaluation surfaces. Independent semantic oracle and text parser must agree; freeze generator/protocol/manifest/script hashes before scoring.
- **Primary**: prespecified role-pair correctness on an unseen construction, comparing training arms across independent seeds. Decide the exact estimator and minimum scientifically meaningful effect *before* examining results. Report native four-name and full-vocabulary metrics separately.
- **Required diagnostic**: mutually exclusive second-event pair categories (both correct, reversed, collapsed, wrong event) and explicit comparison with the slot-frequency shortcut from Gate A. These may be exploratory unless an additional ordered hypothesis is frozen.
- **Structural control**: paired surface transformations that preserve gold roles and token multiset when possible, plus a model-independent parser/oracle validation.
- **Negative/control analyses**: competence on familiar constructions, position-only predictors, familiar versus held-out construction, and actual token/position marginal audits. Report failed seeds.
- **Mechanistic interventions**: defer until this grammar-level behavior is demonstrated. If attempted later, preregister exactly one causally specific patch contrast and one matched control on a further independent cohort; record both role and wrong-event transition rates. No head search or parameter sweeps.

**Decision B:** If the new grammar repeats a construction-specific role failure beyond the slot heuristic, the paper may claim a more general *class* of failures across two controlled grammars. If not, retain a single-grammar methods/negative-control case study and target a focused workshop rather than rerunning with increasingly favorable constructions.

## Stop rules and sequencing

- Finish Gate A before implementing new grammar or loading old checkpoints.
- A new grammar pilot may validate syntax, token lengths and source matching only; do not evaluate trained models until the complete protocol and stopping rules are frozen.
- One frozen scientific run only. All prespecified seeds, repeats and examples are reported. No tuning grammar templates to rescue a failed hypothesis after test results.
- Never open the protected/reserved/L+I gates or mix the old archived held-out tests into new confirmatory evidence.
- Keep the new work on its own research branch, not inside the frozen EXP109–116/PHASE2 sequence.
- Freeze manuscript version and venue framing *after* the anti-shortcut results, not before.

## Reviewer's decision threshold

This is a meaningful step toward a full conference-style paper only if it yields at least one of: (1) a prospectively replicated, distinguishable behavioral phenomenon across a second grammar; or (2) a controlled test that rules out a plausible simple shortcut while retaining an interpretable causal prediction. Otherwise the present evidence best supports a narrower empirical workshop paper. A clean negative finding and a strong artifact may still be worth publishing.
