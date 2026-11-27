# PHASE2-02 — Training-coverage transfer to unseen passive-adjunct word order

**Status:** prospectively frozen, new independent training seeds and new paired non-reserved development examples; study **designed after inspecting PHASE2-01**. Not externally preregistered. A narrow structural-order shift within the *same* two-event grammar and familiar vocabulary; **not** transfer to a wholly different grammar, novel lexical items, or natural language. The protected/reserved 95 name-set evaluation and within-event `L+I` tests stay closed.

## Question

PHASE2-01 found a descriptive +27.30 pp AP second-event advantage for 100% direct dose-pair exposure relative to matched indirect-only evidence, with a positive paired confidence interval but a failed *earlier gated* 20% primary. Its role errors and wrong-event errors both decreased at full exposure. **New hypothesis**: does this behavior transfer to two passive-clause adjunct word orders that never occurred during training, holding tokens, questions, labels and length fixed for each evaluation pair?

The alternatives are (i) direct exposure supports processing across at least some changed adjunct positions, versus (ii) it mainly improves the familiar arrangement or general training competence. Even a successful new test is not proof of abstract role binding; the syntax perturbations and model family are deliberately narrow.

## Training

- Exact existing 3-layer width-96 four-head scratch Transformer, optimizer, vocabulary, supervision, common training pool and 6,400-step schedule algorithm from PHASE2-01 / EXP-109. Eight **new** independent seeds `11801, 11803, 11807, 11809, 11813, 11819, 11821, 11827`; two MPS numerical repetitions each.
- Three preselected conditions, all sharing initial weights within each seed: `p000` has **zero** direct `PA+AP` pair-steps; `p020` has 1,280/6,400 (20%); `p100` has 6,400/6,400 (100%). At each step there are 14 identical common sources and two dose sources, 16 sources and 64 gold answers total. Direct replaces same-core `PP+AA` by `PA+AP`, with identical per-step token multisets, label identities, passive voice counts and position marginals. Sentence-level voice combinations/length patterns differ between conditions; this is not a pure role-variable manipulation.
- `p020` repeats the PHASE2-01 primary contrast as a fixed **descriptive** intermediate condition; `p100` is the single new primary contrast, selected after the prior evidence and now independently tested. No `p005` arm selected post hoc on the apparent PHASE2-01 peak.
- Fresh stratified 20% schedule selection deterministic per seed and independent of model performance. Exactly 1,280 mixed steps, approximately equal representation of 45 modifier × frame strata. No training on the new held-out word orders. No checkpoint selection or early stopping.
- Models: 8 seeds × 2 numerical repeats × 3 arms = **48** model states (16 training processes), all at 6,400 fixed steps. Two numerical repeats are **not** independent seed units.

## Frozen structural evaluation

Use only the 100 *development* four-name sets, keeping the reserved 95 identifier sets closed. Exclude familiar old development source surfaces/core permutations, EXP-109, EXP-114 and PHASE2-01, including literal reversed-source orders. Fixed evaluation sampling RNG `1182026`; exactly **144 new core/modifier anchors**, balanced:

- **72 `near_tail`**: 24 per modifier pair `(bare,I), (L,I), (I,I)`. For the *second original event when it is passive*, move `near PERSON` to after the `by AGENT` phrase. Example schematic: `PATIENT was greeted near NAME by AGENT` → `PATIENT was greeted by AGENT near NAME`. Same exact token multiset/count and same gold answers.
- **72 `gate_infix`**: 24 per `(bare,L), (L,L), (I,L)`. Fix the second event's gate-layout parity to the familiar *postposed* version, then move `beside NAME's gate` from after the agent to just after the passive subject: `PATIENT was greeted by AGENT beside NAME's gate` → `PATIENT beside NAME's gate was greeted by AGENT`. Same exact token multiset/count and gold answers. These strings are controlled synthetic English-like constructions; no claim they preserve every pragmatic nuance of an adjunct.

For each anchor, construct **canonical vs shifted** sentence presentations under `AP`, `PA_literal` (clause-swapped AP), `AA` and `PP` with **the same four questions**. `AA` is an exact identity control: no passive clause is shifted and canonical/shifted token strings must be identical. `PA_literal` shifts the *same underlying passive event* now moved to the first event position. `PP` shifts only the original second event. The novel order changes passive *adjunct attachment/placement*, **not** the two-event composition of the main verbal predicates.

All canonical examples must pass the existing independent source-text parser. Shifted examples must be constructed from the **same event objects**, recover canonical text via the deterministic inverse transformation, and retain gold role labels. For each matched canonical/shifted pair assert equality of complete token bags, token lengths, question/readout shape, vocabulary coverage and the model's 48-token cap. Identity AA pairs count as a control, not 144 additional unique novel surfaces. There are **144 anchors × 2 versions × 4 conditions = 1,152 presentations, of which 1,008 texts are distinct**. No old evaluated surfaces or previously evaluated core orders may overlap. Store only hashes and aggregate statistics, not actual source texts or protected examples.

## Predictions and inference

**Primary directional test:** paired `p100 - p000` **shifted AP second-agent/patient accuracy** averaged equally over the 72 near-tail and 72 gate-infix anchors. Success = lower bound of two-sided 95% seed-level Student t CI >0, n=8 seeds with both MPS repetitions averaged first.

**Gated secondary:** same `p100 - p000` contrast on **canonical AP** for the exact same anchors, interpreted as a second passed test only if the primary passes. If primary fails, report secondary descriptively only. The primary was deliberately informed by the earlier p100 contrast; this is independent testing on fresh seeds/examples, not an untouched hypothesis-selection process.

**Always report descriptive, unadjusted**: both families individually, p020 vs p000, OOD-canonical gaps and cross-arm difference-in-gaps, wrong-role and wrong-event rates, strict four-question `C_query`, matched PA_LITERAL retention, AA/PP controls and all eight seed effects (n=8). Important: a positive primary alone only means p100 performs better than p000 on shifted inputs; **it does not establish robust OOD transfer** unless performance gaps between shifted and canonical forms are separately interpreted. No exploratory contrast should be silently reclassified as confirmatory.

## Audits, execution, and stop rule

1. `--phase prepare` regenerates all non-reserved exclusions and the exact held-out paired evaluation, including token-bag and role-oracle checks. Audits 8 new deterministic training schedules with equal batch token bags and locked dose counts. Writes one immutable manifest; **never scores any models**.
2. `--phase smoke --device mps` uses two training updates on random models and **old, already examined development sentences only**, not the newly held-out structures.
3. Commit runner, launcher, protocol and manifest **together before the first scientific model run**. Every run checks exact frozen committed bytes and protocol/schedule/eval hashes. Reject overwrite of an existing report or checkpoint; never silently resume a partially saved pair.
4. Run all 16 seed × repetition processes; each trains exactly three arms. Save local `.pt` checkpoints and Git-trackable per-run JSON reports, **do not commit `.pt`**. No baseline competence filtering, no optional stopping, no cherry-picking of seeds, modes or doses. Inspect individual outcomes only after 16/16.
5. Summary verifies the exact reports and hashes and produces mean, paired 95% CIs and prespecified gate decisions. Stop afterward and discuss conclusions whether positive or negative.

This study only probes generalization to **novel adjunct placements using familiar words within the same restricted grammar**. If we want truly new predicate types, clause counts, nested dependencies, natural language or a new architecture, those require separate scientifically justified experiments; none are claimed here.
