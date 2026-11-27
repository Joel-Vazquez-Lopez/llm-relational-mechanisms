# PHASE2-03 — Distractor relocation and event-identity errors

**Status:** Fixed, prospective *within-project* v0 protocol. Developed after examining PHASE2-02; not externally preregistered. Training and evaluation code, this protocol, launcher and manifest **must be committed together before training**. This is an independent new-seed/new-development-surface test of the *newly selected hypothesis*, not an untouched confirmation of prior scientific hypothesis selection. The reserved 95 name-set evaluation and within-event L+I test remain CLOSED.

## Motivation and falsifiable question

PHASE2-02 found that moving an existing `near PERSON` adjunct from before the passive `by` phrase to after the agent increased wrong-event predictions by roughly 15 percentage points across the three tested training arms, even though overall AP accuracy changed little. In the synthetic generator, the `near` PERSON in original Event 2 is deliberately a participant of Event 1. Therefore the increase could be caused by a **named distractor becoming a more attractive answer**, by more generic event-index confusion, or by unrelated position effects. Prior aggregates did not distinguish which wrong-event participant was selected.

**Primary question:** Does moving `near PERSON` to passive-clause tail **increase selection of the explicitly named distractor more than selection of the other participant from the wrong event**? The paired within-model contrast on AP second-agent/patient questions is:

`S = (tail - canonical) P(answer = near PERSON) - (tail - canonical) P(answer = other Event 1 participant)`.

Test in the `p000` arm, averaging the two possible Event 1 referents, both question roles, 144 fixed anchors and two numerical repeats within each of eight seeds. Primary success criterion: the lower bound of a two-sided 95% paired Student-t (df=7) seed-level CI for `S` is **strictly positive**. A positive result supports *distractor-specific response sensitivity under relocation*, **not** proof of a specialized event-binding circuit.

## Fixed evaluation contrasts

- Exactly **144 independently selected source core/modifier anchors** from existing non-reserved development name sets: 72 each of modifier cells `(bare, I)` and `(L, I)`. The original second event always has the same `I` modifier and the passive target for AP and PP. The first event is never `I`, avoiding a second unrelated `near` phrase. Only original two-event grammar, original verbs/names and existing 48-token cap are used.
- For *every* anchor, generate two paired referent variants: `near E1_AGENT` and `near E1_PATIENT`, with all four gold role answers unchanged. Both referents belong to the *other* event of the targeted AP/PP question (or the second event for clause-swapped PA_literal). This manipulation changes one repeated name token **across** referent variants; it is *not* a same-token-bag contrast across those variants. Averaging the two variants balances which first-event role gets mentioned.
- For each referent variant generate three passive target-clause variants, all with **exactly identical token multisets, token count, gold answers, question strings and candidate lists**:

  1. **canonical**: `PATIENT was greeted near DISTRACTOR by AGENT`
  2. **subject_infix**: `PATIENT near DISTRACTOR was greeted by AGENT`
  3. **tail**: `PATIENT was greeted by AGENT near DISTRACTOR`

- Four directions per combination: AP (passive target second), literal PA swapped AP source (same passive target first), PP (second passive target), AA (identity control; no passive target changed, all three source strings identical). Target questions are both roles of the *original second event*: AP/PP/AA second-agent/patient, PA_literal first-agent/patient. All four questions are scored for strict C_query.
- Each of the 144 anchor × 2 referent combinations has 3 × 3 + 1 = **10 unique texts** because the AA identity is shared across all placements; 2,880 unique source texts, 3,456 source presentations including repeated AA. No prior development surfaces or core role-orderings from legacy/EXP-109/EXP-114/PHASE2-01/PHASE2-02 are reused. Generate sources exclusively in the existing non-reserved dev partition. The frozen hash covers every source text and gold label but the reports will contain no text strings.
- Original source parser validates **canonical** sentence labels. Shifted surfaces are deterministic re-orderings of that same source and must recover their canonical token bags; check independent target roles/oracle provenance and lengths, not the original parser on a newly shifted grammar it cannot parse.

## Training design

- Same scratch 3-block width-96 four-head causal Transformer, optimizer, vocabulary, 6,400-step schedule, 14 common sources + 2 controlled dose sources per training step as PHASE2-01/02. The two arms start with identical weights within a seed and receive exactly the same number of supervised examples, questions and *aggregate per-step training token bags*:
  - `p000`: indirect `PP + AA` pair throughout; 0 direct AP examples in dose positions.
  - `p100`: direct `PA + AP` pair throughout; one AP source per 16-source training batch.
- Eight **new** seeds: `11901,11903,11907,11909,11913,11919,11921,11927`, two MPS numerical repeats per seed, **16 processes / 32 trained models**. No checkpoint or seed selection; fixed 6,400 updates; report all failures. Training grammar contains neither the `subject_infix` nor the `tail` arrangement. Source-pair assignment across arms is matched, with one passive first and one passive second event in both arm dose pairs. The conditions differ in sentence-level voice co-occurrence/length distributions and thus do **not** isolate an abstract role variable.

## Fixed-order tests, descriptive analyses and interpretation

1. **Primary:** `p000 AP S(tail,canonical)` defined above >0, lower n=8 seed-level 95% CI >0.
2. **Secondary, gated on primary success:** `p000 AP S(tail,subject_infix)` >0, same criterion. This is a matched *different novel relocation* control: a generic response to any unseen `near` placement need not produce a tail-specific effect. Failing it limits claims of tail specificity.
3. **Tertiary, gated on prior two successes:** `p100 AP S(tail,canonical)` >0, same criterion. This tests persistence with direct AP exposure, not independence from training distribution.

Report all three estimates regardless of gate, with `passes=null` for a test not formally reached. The ordered gates avoid treating all three as independently validated confirmatory results. Intervals for **all other metrics** are descriptive and not multiplicity-adjusted.

Descriptives required for every referent, placement, arm and direction: candidate-restricted accuracy, wrong-role errors, total wrong-event errors, selection of the **explicit near distractor** versus the *other wrong-event participant*, and strict 4-query C_query. Include a paired name-swap response audit: holding roles/structure fixed, does changing `near E1_AGENT` to `near E1_PATIENT` change which wrong-event participant is selected? Report `near_follows_swap`, `reverse_follows_swap` and any prediction change. Do **not** select only mistaken baseline responses for evaluation or drop low-accuracy seeds; classifications are post hoc **measurements** on all questions, not intervention filters. The `AA` identity check must exactly reproduce scores for each position.

Possible outcomes and boundaries:
- Primary positive: evidence of a named-distractor-specific association induced by word position under this synthetic grammar, **not** definitive neural event-identity binding.
- Primary null: could mean both relocations behave similarly, both error types change similarly, or high seed variance; cannot conclude absence of mechanism from a non-significant result.
- Similar canonical, subject-infix and tail behavior with directional name-swap tracking would indicate more general sensitivity to mention identity than tail-only recency.
- A correct-versus-wrong-event tradeoff can occur without clear overall accuracy loss. The study remains separate from natural language and fully new grammar tests.

## Safeguards and workflow

1. `--phase prepare`: reconstruct and verify existing EXP-109, EXP-114, PHASE2-01/02 non-reserved audit manifests; build 144 fresh anchors, audit fixed evaluation and all eight matched schedules; create immutable PHASE2-03 manifest. **Do not score the new evaluation**.
2. `--phase smoke --device mps`: two training steps on randomly initialized models and *previously examined EXP-109 development* examples only; not the new PHASE2-03 anchors.
3. Commit **runner, launcher, protocol and manifest in one commit** before the first `--phase run`. Code rechecks frozen file bytes, manifest hashes and commit ancestry before training. Never edit code, the protocol or manifest midway through runs.
4. Run the 16 seed×repeat processes. Save one immutable JSON with both models and a **local-only** `.pt` checkpoint. Inspect no individual result until all runs done. The launcher skips only existing completed JSON files. Do not commit `.pt` or protected evaluation materials.
5. `--phase summary`: require 16/16 valid reports with all 2 arms×2 referents×3 placements×4 directions, verify provenance and identity controls, aggregate numerical repeats within each seed, compute fixed-order tests and comprehensive descriptive tables. If incomplete, write no aggregate.

## Commands

```bash
cd ~/Developer/exp102-clean-design
source ~/Developer/llm-relational-mechanisms/.venv/bin/activate
unzip -n ~/Downloads/phase2_03_distractor_event_binding_v0_bundle.zip -d .
python3 scripts/phase2_03_distractor_event_binding_v0.py --phase prepare
python3 scripts/phase2_03_distractor_event_binding_v0.py --phase smoke --device mps
# Stage/inspect/commit/push the 4 frozen files before the scientific run.
caffeinate -dimsu bash scripts/run_phase2_03_all_v0.sh
```

**Verification before delivery:** Python compilation, CPU random-weight smoke, immutable audit/aggregation and synthetic-fixture checks are development safeguards only. Real MPS trained outcome and protected/L+I results remain uninspected until after the prospective freeze.
