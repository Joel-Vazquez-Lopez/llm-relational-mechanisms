# EXP-101 R2 — Counterfactual role generalization under balanced adjunct interference

**Status: revised design and generator-only dry-run; NOT frozen for training or held-out evaluation.** Revision 13 November 2026. Supersedes EXP-101 R1 (the single adjunct-bearing target clause was shortcut-solvable). No model loaded or trained; no past protected evaluations opened or new protected sentences generated.

## 1. Prior overlap and unresolved alternative

The source/protocol audit covered the ML-072–100 series. ML-072–074 studied exposure and matched intervention controls in another synthetic setting. ML-075 attempted richer grammar, but sparse/broad arms failed the competence gate. ML-078–079 tested first/second **query-cue** changes with an unchanged sentence. ML-080–086, ML-088 and ML-099 used fresh matched exposure cohorts, but preserved the basic short-clause grammar. ML-091 already tested four role/event queries and active/passive alternations with names and roles held constant on previously explored ML-088 calibration checkpoints. ML-092–100 tested activation/source/query mechanisms, with a failed ML-099 prospective mechanism prediction and no ML-100 passing candidate. **Ordinary voice invariance and event-query alternations are therefore not new claims.**

**Unresolved explanation:** target-exposure improvements may reflect voice- and position-conditioned *surface extraction heuristics* rather than correct role assignment when known locative fragments are combined. R1 contained a fatal instance: “choose the only adjunct-bearing clause, keep global singleton names, and order them by voice” solves all eight illustrative transformations **8/8 despite ignoring the event query**. R2 explicitly attacks that baseline.

## 2. Balanced two-event generator and reference conditions

The primary held-out composition is **L+I**, where L is a non-agent locative phrase introduced by *By X's gate* and I is an intervening *near Y* phrase. The same role-relevant participants, agent A and patient P, appear once as core arguments; distractor event participants C,D appear as nonargument locative references, and vice versa. **Both clauses carry the same adjunct condition**, so “pick the adjunct-bearing clause” cannot identify the target event. Two independent referent-swap bits determine which opposite-event person fills the L versus I slot; the target and distractor L prefix/suffix placements are independently balanced. Target event position, voice, role direction, distractor voice, **distractor role direction**, target/distractor event identity and target/distractor verb identities are counterbalanced. The dry-run also swaps the two event-participant pairs across target/distractor status, so no name is permanently a queried-event person.

A fully composed example:

> By Cira's gate, Aven, near Dovan, helped Bero ; By Aven's gate, Cira, near Bero, visited Dovan.

Ask for the **first event agent**: Aven; **first event patient**: Bero. A voice-preserving alternative changes only the queried clause to *By Cira's gate, Bero was helped near Dovan by Aven*. The role reversal changes who helps whom. The event-order transformation physically exchanges the two complete clauses and updates *first↔second* in the question.

Second locative placement: *Aven, near Dovan, helped Bero by Cira's gate*; the passive analogue uses *Bero was helped near Dovan by Aven by Cira's gate*. This repeated *by* is an intentional distractor, but must undergo human grammatical review before freeze. The I phrase is kept in the intervening position to challenge nearest-name heuristics; moving it to a benign prefix made that heuristic solve an entire packet and was **removed before training**.

**Exactly four conditions, in the same future cohort:** BARE (both clauses bare), L-only (both clauses contain L), I-only (both clauses contain I), and L+I (both clauses composed). All other event semantics and label mappings remain matched. BARE/L/I are **same-cohort, same-held-out-combination second-passive reference conditions**, not new training arms or an opportunity to retune the protocol. BARE/L/I patterns occur in training/development; L+I is withheld from all training and development inputs and any prompt tuning. Under L+I every name occurs **exactly twice** in the full sentence: once as a core argument, once as an off-event locative reference; no globally singleton name exists. Under L or I, within-sentence frequencies are (1,1,2,2), but the repeated person in each event is balanced by two independent referent-swap bits across the fixed packet universe; BARE is (1,1,1,1). No event is uniquely adjunct-bearing in any condition. Counts for each identity, role, voice, location-owner and event position must be equal over enumerated generator blocks, not merely approximately equal in a small sample.

**Transformations, each with a distinct function:**
- **Voice change (meaning preserving):** active/passive rewrite of the same target event, with agent/patient unchanged. This is not a novelty claim relative to ML-091; it is required to establish correctness on the withheld adjunct conjunction.
- **Role reversal (meaning changing):** swap A/P semantic roles with the same people, verb, other event, locatives and query; the two gold answers must exchange.
- **Event order (meaning preserving):** exchange entire clauses and update first/second query, holding the *queried semantic event* and roles fixed. Unlike ML-078, the clauses actually move; unlike R1, adjunct and mention-frequency cues no longer identify the target event.

For each anchor, cross 2 voices × 2 role directions × 2 target-event positions, and ask both roles: **16 answers per condition per anchor**. Distractor voice, locative-owner assignment and two independent L placements are balanced across anchors and kept fixed within transformation packets.

## 3. Static shortcut checks and independent text validation

Two executable, stdlib-only files accompany R2:
- `scripts/exp101_generator_preflight.py`: **development-only** synthetic packets, controlled counterfactual transformations, explicit heuristic baselines, mention/role/position balance assertions. `--phase preflight` is the **only** allowed phase; no model imports, training, test access or checkpoint loading.
- `scripts/exp101_render_validator.py`: separate regex-based **rendered-text parser**, importing no generator metadata. It recovers both event predicates, agents, patients, voices and locative owners from the realized source text; a separate query-string parser recovers the **actually rendered first/second event and agent/patient question**. Both are cross-checked against gold labels. This is a structural positive control, not a negative baseline. A human acceptability/meaning audit is **still mandatory** before freeze.

**Development dry-run actually executed locally, no models:** 48 development-only lexical/verb/target-status anchors (four quartets × six verbs × two target/distractor pair orientations) × 2 target role directions × 2 target voices × 2 distractor voices × 2 distractor role directions × 2 target reference assignments × 2 distractor reference assignments × 2 target L placements × 2 distractor L placements × 2 target-event positions = **24,576 packets per condition**, 98,304 total. The independent sentence-and-question parsers and gold semantic invariants pass; every L+I packet has both clauses adjunct-bearing and two appearances of each of four people. Each pseudoname has equal enumerated frequencies as a target/distractor agent and patient. Reference cells meet the declared (1,1,1,1) or (1,1,2,2) mention-count patterns. The original attack is reproduced at **8/8** when only the target clause bears adjuncts, then fails the redesigned test.

| Non-oracle baseline on R2 L+I | Exact agent+patient / 24,576 individual packets | Voice-paired correct / 6,144 packets with target event SECOND |
|---|---:|---:|
| **Adjunct-clause + global singletons + voice order, ignores event cue** | 12,288 | **0** |
| Fixed global alphabetical names | 6,144 | 1,536 |
| First/last name in queried clause | 0 | 0 |
| Nearest name left/right of verb, voice-conditioned | 0 | 0 |
| First-*by* heuristic | 6,144 | 0 |
| Last-*by* heuristic | 6,144 | 0 |
| Event-blind first-clause parser | 12,288 | 0 |
| Independent structural oracle | 24,576 | 6,144 |

These are **deterministic generator-only checks**, not observed Transformer outcomes. The strongest listed shortcut still achieves 25% on the primary task, so absolute model correctness and its margin over these baselines must be checked. The script tests that none of these baselines solves a whole eight-cell packet or the full target-second voice-paired diagnostic. If another concrete shortcut achieves full correctness, **stop and redesign before training**. A sufficiently sophisticated symbolic parser may succeed legitimately; this design rejects specified shallow heuristics, not every imaginable algorithm.

The generator-only dry-run does **not** yet certify the unimplemented production split/seal and training sampler. Formal freeze requires independent rendered-text-to-gold checks on the full *training/development generator* (without generating protected examples), component-leakage checks, duplicate detection, output-vocabulary tests and human review of all syntax templates. Protected cases will only be generated on a separately authorized one-time unlock after training gates.

## 4. One primary contrast; consistency versus correctness

For a model and event, let `f(x,agent)` and `f(x,patient)` be **selected person identities**, from the four event-participant names (not A/B option labels), and let `g` be gold identities.

- **Consistency S_voice / S_event (descriptive):** both answers are unchanged under a meaning-preserving transformation.
- **Consistency S_reverse (descriptive):** the two predicted identities exchange under role reversal. Constant same-person responses can pass this test; do **not** call that correct generalization.
- **Paired correctness C_T:** the two complementary answers must be **gold correct on BOTH sides** of transformation T (four correct answers). A consistently wrong response scores zero.
- Always separately report initial/final both-role correctness, per-question accuracy, correct-event wrong-role, wrong-event, duplicate-answer rate, and truly reversed *pairs*. A per-question wrong-role answer is not automatically a paired reversal.

**One primary outcome:** `C_voice` on withheld L+I with the queried event physically **second**, averaged over BOTH role directions, all frozen anchors and layout/referent assignments for each model. **One primary exposure contrast:** target-exposure minus familiar-exposure, paired by initialization seed. C_reverse and C_event and all S metrics are secondary/descriptive and cannot replace the primary.

Compute the *identical C_voice target-second outcome* on **BARE, L-only and I-only** held-out reference conditions using the same model checkpoints, seed pairs, event combinations and candidate pool. These three **predeclared diagnostic reference contrasts** establish whether the cohort reproduced the starting effect and its single-component generalization. They are not extra training contrasts or selectable primary tests.

## 5. Fixed training, split, competence and inference budgets

**Architecture:** only the ML-080-style 3-block/width-96/four-head causal Transformer with newly tokenized L and I phrases. **One training contrast:** target versus familiar supervision, no architecture sweep or added conditions. Both arms see identical balanced 56-example shared base batches plus 8 extra samples from the same 64-dose pool per update; the 64 dose items comprise 32 L-only and 32 I-only questions, matched for all identities, roles, verbs, adjuncts and syntax except whether the **passive queried event is first (familiar) or second (target)**. Clause order/query cue necessarily change together; do not attribute a benefit to voice alone.

**Split:** 16 fixed novel names; 6 fixed transitive verbs; 15 one-factorized disjoint perfect-matchings (9 train, 3 development, 3 sealed protected), no unordered participant pair shared across splits. Held-out uses new pairs of familiar names/verbs **and** globally withheld L+I combination. Freeze source hashes, RNG streams, all matching identifiers/selection hashes, and the mapping of target/distractor locatives before training; the protected **rendered sentences and outputs must not be constructed, read or inspected now**.

**Seeds and optimizer:** exactly 8 new paired initialization seeds `(907, 911, 919, 929, 937, 941, 947, 953)`; same initial weights and shared minibatch order within pair; exactly **12,800 steps/model**, batch 56+8, 16 models = **204,800 optimizer steps**. Fixed hyperparameters and learning-rate schedule must be copied into the fully implemented training protocol and frozen, not chosen after a model run. Optional 3,200-step checkpoint is for observation only; never select a best checkpoint.

**Cell-level development competence gate before protected access:** use only **BARE, L, I** and three familiar train-grammar query/voice cells `(first,active)`, `(first,passive)`, `(second,active)` = **nine separately reported development cells** per seed and arm. Require ≥96 distinct events/cell, native four-choice single-question accuracy ≥.95 **and** both-role correctness ≥.90 **in every cell for both members of each seed pair**. At least **6/8 jointly competent seed pairs** must pass; 6/8 per arm independently is insufficient if they are not the same six pairs. No seed is replaced or discarded. If fewer than six jointly pass: **STOP, do not unlock held-out**. If the gate passes, **all eight pairs remain in the intention-to-treat primary estimate**, including any incompetent pairs; the jointly competent subset is a labeled sensitivity analysis only. Target-second-passive development observations must **not** be used to retune prompts or drive the gate.

**Single future protected evaluation:** upon explicit authorization only after a complete code/data freeze and successful gate, 24 sealed participant pairs × 6 target verbs × 2 predetermined distractor-map choices = **288 anchors**. Four conditions × 8 counterfactual cells × 2 role queries = **4,608 prompts/condition/model**, 18,432/model, **294,912 total across 16 models**. Balance layout, references, event direction and distractor voice through a frozen deterministic assignment across anchors. One frozen four-choice decoding rule; unrestricted vocabulary and any logits are secondary; no post-hoc exclusion or resampling of protected cases.

**Uncertainty:** 8 paired-seed differences per condition. For the primary and each reference, report mean and eight effects with fixed 20,000-replicate seed-pair bootstrap 95% intervals (bootstrap RNG seed 101101). For reference comparisons use synchronized paired resampling of the *same* eight seed differences. Never treat prompts as independent replicates.

## 6. Predeclared decisions and paper contribution

**Strong positive transfer** only if ALL hold: (i) primary L+I exposure effect ≥+0.15 absolute, (ii) paired bootstrap 95% lower bound >0, (iii) ≥6/8 paired seeds positive, (iv) target-arm primary C_voice ≥.65 and ≥.15 above the strongest audited *non-oracle* baseline on the same endpoint. This is a demanding success definition; failing it **does not itself demonstrate nontransfer**.

**Evidence for bounded nontransfer to conjunction** only if the jointly competent gate passed and **BARE, L-only and I-only each reproduced a meaningful reference effect** (each target−familiar ≥+.15 with seed-bootstrap lower bound >0), but the L+I effect's bootstrap **upper bound is <+.10** and the paired contrast `(L+I effect) − mean(L-only,I-only effects)` has a bootstrap upper bound <0. That would show a specific failure to extend *despite* same-cohort reference successes. It would not show a universal inability to learn conjunctions.

**Inconclusive / partial:** gate failure, independent semantic validation failure, baseline solving the full diagnostic, too-wide interval, borderline positive effect below the compound strong threshold, or lack of reference replication. Report signs, estimates, uncertainty and each failed criterion. An L+I effect with interval excluding zero but below +.15 is **positive but not strong**, not automatically nontransfer. If references fail, say the experiment did not reproduce the starting exposure advantage in this cohort; do not attribute L+I failure specifically to compositionality.

**Distinct from ML-081–100 intervention search:** EXP-101 is a prospective **behavioral counterfactual input/output generalization** test under independent training seeds; it does not ablate/patch neurons, localize a layer, or test a unique circuit. Positive evidence would strengthen the scope of the exposure effect against these explicit shortcuts, without proving a particular internal mechanism. Negative/inconclusive outcomes would delimit the paper's claim without hidden retraining.

**Stop rule:** no experimental model training, protected-case construction, or protected evaluation in the current protocol revision. Before training: implement/freeze complete generator, split metadata, optimizer and source hashes; repeat all generator-only checks; conduct independent semantic/human acceptance audit; validate that shortcuts do not solve full design. After authorization: train exactly 16 models to the fixed budget, gate once, evaluate protected cases at most once if eligible, report all outcomes, and **stop**. No circuit search or pretrained LLM study is added.