# EXP-101 — Counterfactual role generalization under compositional adjunct interference

**Protocol state: DESIGN LOCK CANDIDATE — NOT YET FROZEN FOR TRAINING OR TEST.**
Authored 13 November 2026 after a source-code/protocol audit of the ML-072–100
series. **No pretrained models, scratch models, checkpoints, protected evaluation
records, or protected test examples were opened for EXP-101. No training was
started.** The previous manuscript's experimental stop remains in force unless
this new design passes a *generator-only* validation, is frozen with code/data
hashes, and is explicitly authorized to train.

## 1. Question, contrast, and unresolved alternative

**Question:** Does supervised exposure to answering questions about the
*second-passive event* help a scratch-trained Transformer generalize semantic
agent/patient assignments when two individually familiar, role-irrelevant
locative constructions are **composed in a previously unseen arrangement**,
including matched voice and role-reversal counterfactuals?

**One primary contrast:** paired target-exposure minus familiar-exposure
performance. No additional arms, architectures, or circuit interventions.

**Specific alternative:** the substantial ML-080–099 target-exposure advantage
could be achieved partly by learning a finite collection of **local
position-and-cue templates** tailored to the original three-token active
`[AGENT VERB PATIENT]` and five-token passive
`[PATIENT was VERB by AGENT]` clauses, rather than a role rule that is robust
to intervening, role-irrelevant participant mentions. A model might infer
"agent = nearest name left of the active verb" or "agent = first name following
by" without tracking the subject/object relation compositionally.

The original ML-080 generator balanced participant directions, target/distractor
event positions and voices, and held out participant pairs/verbs, but kept
these simple **clause-internal** realizations. The observed within-family
replication and joint-role improvements do not exclude those richer
voice-conditioned surface heuristics. In the matched exposure comparison,
target passive clauses are **not always unseen as input**: they may occur
as unqueried distractors. The causal manipulation is targeted **supervision
for the clause/query configuration**, not exposure to a wholly invisible
surface string.

### Prior-work/overlap audit (source files and calibration-only documentation)

| Existing study | Already tested | What remains open |
|---|---|---|
| ML-072–074: `ml072_compositional_exposure.py`, `ml073_causal_role_balance.py`, `ml074_matched_causal_controls.py` | Alternative exposure doses, role-signal and matched intervention controls in another limited synthetic grammar | No held-out composition of *two irrelevant adjunct constructions* with truth-preserving/altering paired counterfactuals |
| ML-075: `ml075_grammar_benchmark.py` | Richer grammar attempt | Sparse and broad arms failed the prespecified competence gate; no protected conclusion |
| ML-078–079: `ml078_event_cue_counterfactual.py`, `ml079_event_logit_diagnostics.py` | Changing **first/second query cue** for an unchanged sentence and examining response/logit shifts on familiar calibration grammar | Does not physically move the same queried event while holding its meaning and role labels fixed in a composed grammar |
| ML-080; ML-084–086; ML-088; ML-099 | Fresh seed/lexical-pair cohorts, targeted versus familiar exposure, archived accuracy and some role-error summaries | Same short clause templates; rebalanced seeds do not break clause-local positional heuristics |
| ML-081–083; ML-085–087; ML-089–090; ML-092–098; ML-100 | Layerwise geometry, activation patching, role-cue, source/query and access interventions | Interventions test *internal sensitivity/specificity*, not counterfactual **input-output generalization** to a new surface compositional arrangement |
| **ML-091**: `ml091_counterfactual_binding.py` | SAME sentence × four first/second-agent/patient questions; second-clause **active/passive** contrast with names/verbs/role truths matched, first clause fixed; discovery-only, on old ML-088 calibration cohort | This proposal must not claim ordinary active/passive invariance is novel. New evidence can only arise from **unseen adjunct composition**, independent newly trained pairs, crossing with **role reversal**, plus physically relocating the event while updating the query |
| ML-099/100 | Prospective L2 role-Q prediction failed its prespecified criterion; bounded exploratory source/access search found no passing candidate | No support for a unique necessary circuit; EXP-101 will not reopen that search |

The completed exploratory ML-091 calibration did show a voice-conditioned
event–role logit interaction, but it reused earlier model checkpoints and
simple sentences. Neither that finding nor the synthetic cross-cohort accuracy
results demonstrate robustness when **a non-agent `by` phrase and an
intervening non-participant name are simultaneously present**.

## 2. Syntax and transformations (single bounded diagnostic)

All sentences contain **two distinct event predicates**, four event participants
A,P,C,D, and a question specifying first/second event and agent/patient role.
The target event uses A/P; the other event uses C/D, with C and D also appearing
as *non-argument locative mentions* in the target clause. No coreference:
the locative person is **not** the agent/patient of the target verb.
Names and verbs are distinct within each event packet. Keep the four
actual event participants as the **same four answer candidates** for all
transformations; unrestricted full-vocabulary prediction is secondary.

Examples (`helped` is target verb; `visited` is distractor verb):

**Component L (locative `by`, not the passive agent):**
- Active: `By Cira's gate, Aven helped Bero`
- Passive: `By Cira's gate, Bero was helped by Aven`

**Component I (intervening irrelevant name):**
- Active: `Aven, near Dovan, helped Bero`
- Passive: `Bero was helped near Dovan by Aven`

**Primary held-out L+I conjunction, never in training or calibration:**
- Active: `By Cira's gate, Aven, near Dovan, helped Bero`
- Passive: `By Cira's gate, Bero was helped near Dovan by Aven`

Two-event realized source (illustrative, with a `;` clause boundary):
`Cira visited Dovan; By Cira's gate, Bero was helped near Dovan by Aven.`
Ask `In the second event, who was the agent?` (gold `Aven`) and
`... patient?` (gold `Bero`). The other event can also come second.
Use event-specific query metadata as in ML-080. These forms must pass
human grammatical/semantic checks before any training is authorized;
ungrammatical realizations are a **pretraining stop**, not post-hoc exclusions.

Each diagnostic **anchor** is crossed in an eight-cell
`voice ∈ {active,passive}` × `role-direction ∈ {A→P,P→A}` ×
`physical target-event position ∈ {first,second}` factorial, with
two complementary role queries for every cell (16 predictions per anchor).

- **T_voice — meaning preserving:** active ↔ passive; semantic agent/patient
  *unchanged*, locative identity and target event unchanged. Rejects
  voice-specific overfitting and tests invariant role selection.
- **T_reverse — meaning changing:** swap which of A/P is agent, preserving verb,
  named participants, locative mentions, voice, other event and target-event
  position. Gold agent/patient **exchange**. Tests appropriate equivariance,
  rather than unchanged answers to changed meaning.
- **T_event — meaning preserving:** physically swap the two entire event
  clauses and **update** the query first↔second to keep the same semantic target.
  Roles remain unchanged. Distinct from ML-078/091's question-cue-only
  manipulation; tests event identity under position relocation.
- The *adjunct conjunction* L+I is itself a meaning-preserving addition
  relative to bare clauses, with two non-argument people. It is the central
  withheld compositional challenge.

Do not introduce extra sentence templates or perturbations after seeing
development model performance. If the static baseline-vs-oracle audit finds
a shortcut that passes all eight cells, revise and re-freeze the generator
**before training**, not through protected-score-driven adaptation.

## 3. Metrics: consistency is NOT correctness

Define (f_m(x,r)) as a trained model's selected **person identity**
under its native four-name forced-choice logits (not an A/B token label);
(g(x,r)) as the gold identity; (r∈\{agent,patient\}).
A *paired-role correct event* requires **both** role answers to be right.

For a transformation `T`, report separately:

- **Transformation consistency** `S_T` (descriptive): for voice/event-order,
  both selected identities remain unchanged; for reversal,
  `f(Tx,agent)=f(x,patient)` AND
  `f(Tx,patient)=f(x,agent)`. This measures response equivariance **only**.
- **Transformation-paired correctness** `C_T`: **all four** answers
  (agent and patient for both `x` and `Tx`) equal their appropriate gold
  identities. A perfectly consistent but always-wrong, fixed-name or
  same-name predictor obtains **zero** on such cases.
- Also report each side's `both_correct` fraction,
  individual accuracy, wrong-event/role per-question errors, and
  `collapse` (same-person pair). Role reversals count as such **only**
  when both gold event participants are swapped in the paired responses;
  wrong-role per question must not be labeled full reversal.
- For reversal consistency, stratify by collapsed/noncollapsed initial
  responses. A constant-name predictor can trivially satisfy `S_reverse`
  when it outputs that same name for both questions, but **fails**
  `C_reverse`.

**Exactly one primary behavioral estimand:** the cohort-mean paired
target-minus-familiar difference in **voice-transformation-paired correctness
on BOTH role directions, at target event position SECOND, under L+I**.
For each held-out anchor and each role direction `d`, set
`PC_voice(anchor,d)=1` only if agent and patient are both correct
in the ACTIVE and PASSIVE renderings of that semantic event
(**four gold-correct responses**). Average `PC_voice` over the two
role directions and all anchors to obtain one score per model. Its
paired target−familiar difference, averaged over eight training seeds, is
the **sole primary exposure contrast**.

Predeclared *secondary, non-gating* outcomes: `C_reverse` separately
by voice; `C_event` separately by voice and role direction;
`S_voice, S_reverse, S_event` and their correctness-conditioned
versions; fully correct **eight-cell packet** rate; error taxonomy,
four-choice versus unrestricted outputs and bare/L-only/I-only
development generalization. Do **not** turn whichever of these looks best
into a new primary endpoint.

## 4. Shortcut baselines and required static preflight

Implement and run on **the complete enumerated diagnostic design,
without training**:

1. **Fixed lexical ranking:** return the first two names in a
   deterministic alphabetic order as agent/patient, ignoring clause/voice.
2. **Clause-span head/tail:** assign agent to first recognized name and
   patient to last recognized name in the *queried clause*.
3. **Nearest-name-to-verb:** active agent = closest name left of target verb,
   patient = closest right; passive agent = closest right, patient = closest
   left.
4. **Voice-aware first-`by` heuristic:** active uses nearest-name rule;
   passive agent = the first name following ANY `by` (including
   non-agent locative `By Cira's gate`), patient = closest name left of verb.
5. **Voice-aware last-`by` heuristic:** as (4), but passive agent =
   last name following a `by`; this **succeeds on the passive half**
   of the toy packet but fails the active half, which motivates including
   voice-paired correctness as the primary rather than passive-only scores.
6. **Ignore the event cue:** always answer about a fixed physical clause;
   unlike (2–5), this must fail when the target event is physically moved
   while the query is updated.
7. **Structural-oracle positive control:** using the generator's parsed
   agent/patient indices, recover both roles perfectly under all cells.
   This proves the labels are coherent; it is **not a shortcut model**.

**Checked minimal toy packet** with (Aven,Bero,Cira,Dovan; helped/visited),
L+I and all eight factorial cells (see accompanying dry-run logic; no
model run):

| Deterministic diagnostic | Exact two-role cells /8 | Individual /16 | Voice equivariance /4 | Reversal equivariance /4 | Event-order equivariance /4 |
|---|---:|---:|---:|---:|---:|
| Fixed lexical | 4 | 8 | 4 | 0 | 4 |
| First/last in queried clause | 0 | 4 | 0 | 0 | 4 |
| Voice-aware nearest to verb | 0 | 8 | 4 | 0 | 4 |
| First `by` on passive | 0 | 8 | 0 | 0 | 4 |
| Last `by` on passive | 4 | 12 | 0 | 2 | 4 |
| Structural oracle | 8 | 16 | 4 | 4 | 4 |

Crucial: **every tested shallow shortcut fails complete eight-cell
correctness**, but some achieve **perfect event-order consistency**, and
the last-`by` heuristic gets all passive second-event answers correct.
Thus passive-only accuracy or consistency alone is **inadequate**.
Implement these baselines with explicit tokenization and ensure on the
full generator that none obtains all-cells-perfect correctness or a
primary score high enough to trivialize the criterion. If any proposed
surface shortcut passes, do **not** train: revise the grammar and redo
all baseline checks. The structural oracle *must* pass all cells.
A sufficiently competent finite-state grammar parser could still
solve a finite artificial grammar; this protocol can reject the
specified shortcuts, **not all possible hand-coded parsers or
prove a specific learned circuit**.

## 5. Splits, randomized contrasts, budget, and access control

- **Model:** one model family only, the ML-080 three-block causal
  Transformer (width=96, four attention heads), with expanded vocabulary
  and embeddings for the new locative tokens. No model-size or
  architecture sweep.
- **Vocabulary:** 16 novel pseudonames, six transitive past-tense
  verbs (`helped,praised,thanked,followed,noticed,visited`);
  distinct target/distractor verbs. Every event packet uses four
  distinct event people. The two locative mentions reference the two
  *distractor-event* people and must be counterbalanced. Semantic
  agent/patient identity must not be correlated with pseudoname
  identity or frequency, modifier assignment, grammatical voice,
  frame or event position.
- **Pair partitions:** seeded one-factorization of the complete graph
  on 16 names into 15 perfect matchings. Freeze nine matchings
  (72 unordered pairs) for training, three (24 pairs) for development,
  and the last three (24 pairs) **sealed for protected evaluation**.
  No participant-pair identity crosses splits; people and verbs can
  recur. Freeze split RNG seed, ordered pair IDs and hashes before
  training. Never regenerate a different split after outcomes.
- **Training generator:** bare, L-only and I-only constructions, with
  `target_pos×voice` familiar cells
  `(first,active),(first,passive),(second,active)`.
  The **L+I conjunction is withheld globally** from train and
  development, across all positions, voices, frames and participants.
  Balanced role directions, participants, locative referents,
  frames, queries and target/distractor lexemes; automatically assert
  frequency and answer-label balance. The agent identity is the
  *semantic participant* of the queried verb, not the nearest named
  word or location owner.
- **Single exposure intervention:** in each training update, 56 shared
  base examples + 8 dose examples. Exactly 64 dose **question items**
  per arm, replayed on the same schedule. Dose items split 32 L-only
  and 32 I-only, agent/patient and role direction balanced. Both
  arms use the **same source people, verbs, locative names, sentences
  modulo event-order movement, and target role meanings**. Familiar
  arm labels/query a passive **first** event; target arm labels/query
  the corresponding passive **second** event. Clause order and ordinal
  query necessarily covary; do **not** attribute any resulting effect
  to voice alone or to a unique subcomponent of the intervention.
- **Fresh independent runs:** 8 preselected paired initialization seeds,
  `(907,911,919,929,937,941,947,953)`, not used in ML-080–100.
  Match initial weights and base-batch schedules within each
  target/familiar seed pair; freeze RNG streams separately for grammar
  sampling, data shuffling, model initialization, and dose.
- **Fixed training budget:** 12,800 update steps per model;
  **16 models × 12,800 = 204,800 optimizer steps total**,
  `batch=56 shared+8 exposure`, no adaptive extension or
  architecture change after training begins. An intermediate 3,200
  checkpoint may be logged for learning curves but cannot change
  training parameters or model selection.
- **Development competence gate:** use ONLY the 24 development
  participant pairs under bare, L-only and I-only **familiar cells**
  (not any protected L+I example); evaluate both complementary role
  questions under the native four-choice task. Require
  **≥6/8 seeds in EACH arm** to exceed
  `single-question_accuracy≥0.95` and
  `both-role_correctness≥0.90` at final checkpoint; report all
  arm/seed failures, with no dropout or cherry-picking. Calibration
  metric computation and representative dry-runs must pass synthetic
  oracle and negative-control tests.
- **Protected set is not generated or accessed now.** If and only if
  both-arm gate succeeds, require one explicit release decision,
  an immutable code/protocol/split-hash manifest and a one-time
  protected unlock. On that single unlock, construct **exactly
  288 base anchors**: each of 24 protected target pairs × six target
  verbs × two fixed frames. The distractor pairing/verb and two
  locative assignments are chosen by a frozen deterministic,
  frequency-balanced mapping. For each anchor, eight factorial
  transformations × two complementary queries = **4,608 prompts
  per model**; 16 models = **73,728 protected prompts**. Score all
  models exactly once; abort if cases overlap train/development
  or if any generator semantic invariant fails, *before inference*.
- **Inference and uncertainty:** one native four-name decoding
  rule frozen before training; report full-vocab results separately.
  For each seed, average 288 anchors and both role directions to
  obtain its primary score in each arm. Compute eight matched
  seed-level `target−familiar` effects, their mean, **20,000
  paired-seed percentile bootstrap** replicates with fixed seed
  `101101`, and all eight signed differences. Optionally provide
  the exact 256-sign-flip randomization reference as descriptive
  supplementary inference; do **not** treat 73,728 prompts as
  independent replicates.
- **Frozen success rule for *strong* transfer** (if protocol ultimately
  ratified): mean primary benefit **≥+0.15 absolute**, paired-seed
  bootstrap 95% lower bound **>0**, **≥6/8** seeds positive, and
  target-arm primary score **≥0.65** and **≥0.15 above the strongest
  audited non-oracle surface shortcut** under the identical primary
  metric. All four components are conjunctive. Failure to meet
  the rule is a reported negative/inconclusive outcome, not grounds
  for tuning on protected cases.

## 6. Stopping, interpretation, and relation to intervention studies

**Before training:** freeze the *complete implemented* generator, exact
tokenizer, seed list, split identifiers and pair hashes, baselines, training
scheduler, calibration gate, outcome code, protected query manifest HASH
(not protected text), model architecture and intervention code. Run a
static, nonlearned preflight: role truths, transformation semantics,
unique answer candidates, role/position/lexical balance, no L+I
leakage, no cross-split pair overlap, structural oracle perfect,
all listed shortcuts fail the complete test and meet the primary
baseline bound. Any defect -> fix in a new revision **before** freeze.

**After training starts:** do not add seeds, extend updates, change
the grammar, pick best checkpoints, revise metrics, retune
selection criteria or adapt to protected results. If development
competence gate fails -> **STOP without protected access**.
If protected evaluation is unlocked, open once, record every
predeclared analysis, and stop regardless of sign. No further circuit
search and no separate LLM evaluation in EXP-101.

- **Positive:** stronger *input/output* evidence that targeted
  query/exposure supervision supports correct counterfactual
  relational assignments in an unseen composition of two familiar
  locative constructions, beyond the specified lexical/position
  shortcuts. Does **not** prove general natural-language understanding,
  that all shortcuts are excluded, or a particular internal mediator.
- **Negative with passing gate:** a bounded nontransfer result
  showing the old exposure advantage is fragile to this specific
  compositional interference; report any difference between
  correctness and superficial equivariance as a scientific result.
- **Inconclusive:** competence gate failure, implementation/oracle
  failure, or noisy/intermediate effects failing the compound
  criterion; document openly, do not reclassify it as a success
  based on descriptive metrics.

**Difference from ML-081–100:** we manipulate the *sentences and
their truth conditions* and measure behavioral generalization across
new independently trained model pairs. We do **not** transplant
activations, ablate a head, compare layerwise path-specific
interventions, or claim mediation/necessity. EXP-101 is a
**counterfactual generalization experiment**, not another
mechanistic circuit search.

## 7. Outstanding mandatory freeze artifacts — no work authorized yet

1. Source and unit tests for a new bounded generator and all seven
   baselines; include a parse oracle, exact clause boundary detection,
   case tokenization and a reproducible full-design *metadata*
   manifest without emitting protected example text.
2. A full dataset/static preflight under the generator: audit
   slot/locative/verb/name-role balance and verify the "by" locative
   acts as a distractor without introducing a second target agent.
3. Human review of the finite grammar for naturalness,
   voice-preserving equivalence, true role-reversal and stable
   event identity after swapping order. If ambiguous, **redesign
   before freeze**.
4. Commit code/protocol/test manifests with pinned SHAs; freeze
   exact files and authorize training *only in a later turn*.

**No EXP-101 model has been trained, evaluated, or compared as a result
of this design document.** The toy shortcut table is a static design
check, not data from any Transformer.
