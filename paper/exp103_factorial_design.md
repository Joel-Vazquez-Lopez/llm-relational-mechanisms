# EXP-103 — Matched-update factorial development study

**Design status:** prospective *for the next runs*, but **exploratory development**, because both prior interventions and the proposed mechanism were discovered during iterative research. This protocol must not be labeled preregistered confirmation. No reserved or L+I examples are created or evaluated.

## Question

Do (A) targeted exposure to a previously unsupervised second-event passive construction and (B) organizing complementary role questions together within one optimization update produce distinguishable improvements in event selection and role assignment?

## Four conditions

| Exposure factor | Quartet policy | Distributed policy |
|---|---|---|
| Familiar-position passive dose | `familiar/quartet` | `familiar/distributed` |
| Target-position passive dose | `target/quartet` | `target/distributed` |

### What is held constant

- Existing EXP-102 v2 four-name-set split fingerprint: `2f9ba0c6f44c67094f9fa2786853ade1c8c662fd0afd74295af98384b17dd287`.
- All individual names and verbs are familiar. Base training samples use nine familiar modifier pairs only (`bare`, `L`, `I`) and voice pairs `AA` (first active, second active) and `PA` (first passive, second active). No base training queries target a **second passive** event.
- Common base: **1,152** stratified training sentences. Exposure source pool: **72** matched first-passive/second-passive reordered pairs, chosen from the training name sets but distinct from common base texts. The familiar dose uses first-passive/second-active sources; the target dose reverses those two semantic events so the passive event occurs second. Both dose variants provide all four role labels.
- Familiar development: **216** sentences on disjoint four-name sets, **72 each** in `AA`, `PA`, and `AP` (first active, second passive). All three have familiar modifiers. `AP` is the primary developmental target.
- Model: EXP-102 v2 scratch causal Transformer, 3 layers × width 96 × 4 heads; AdamW, lr=0.002, wd=0.01; cross-entropy over full vocabulary, evaluation restricted to the four participant candidates.
- Fixed seeds: `10501, 10503, 10509, 10513, 10521, 10527, 10529, 10533`; all four conditions start with identical parameters **within a seed**.
- Fixed budget: **3,200 updates**, **64 supervised labels per update**; checkpoints evaluated only at steps `800,1600,2400,3200`, final-step selection only.

## The source-matched update control

Each four-update window samples **64 distinct** source IDs: 56 from the common base and 8 from the appropriate dose. The two batching policies see **precisely the same 256 `(source ID, question ID)` training items**, exactly once each, over that window.

- **Quartet:** In each update, 16 distinct sources (14 base, 2 dose) contribute all four event/role queries.
- **Distributed:** In each update, all 64 distinct sources contribute one query each. A Latin-square rotation ensures every source receives all four distinct queries over the four updates. Question counts per update are exactly 16 per type, with 14 base and 2 dose labels per question type.
- Every update has 56 base and 8 dose labels. The 4-update multiset matching is exact **within an exposure factor**. Exposure placement necessarily changes source text between `familiar` and `target`.

This removes the earlier mismatch in the **total source×question exposure distribution over a four-step window**, but it does **not** match within-step source diversity or AdamW's parameter trajectory. Any effect is an *optimization-policy* effect, not evidence of direct cross-example communication inside a Transformer minibatch.

## Primary descriptive contrast (development only)

At the fixed 3,200-update endpoint, for each initialization seed, let `Y(e,p)` be exact `C_query` (all four answers correct) on the held-out-name-set **AP** familiar-modifier development set. Estimate the within-seed 2×2 interaction:

`[Y(target, quartet) - Y(target, distributed)] - [Y(familiar, quartet) - Y(familiar, distributed)]`

Report all eight paired interactions, their mean, seed-level bootstrap interval, and directional counts. No exclusion of slow learners, no best-checkpoint selection. No independent or confirmatory p-value claims from this exploratory design.

## Prespecified secondary outcomes

For AP development and AA/PA familiar development separately: four-choice question accuracy, wrong-event rate, within-correct-event wrong-role rate, identical agent/patient answer rate, per-event both-role correctness, all-four `C_query`, and learning trajectories. All measures are derived from exactly the same saved logits/predictions and validated identities.

## Competence gate for any *future* reserved evaluation

This developmental runner cannot generate or open a reserved set. Before a **separate** reserved evaluation is designed, require **at least six of eight seeds within each of the four arms** to attain, on both AA and PA familiar development cells, per-question accuracy ≥0.95 and mean per-event both-role correctness ≥0.90. An unmet gate means publish descriptive development findings, not a protected generalization claim. Do not retrospectively change this gate.

The earlier L+I composition designs have already been inspected and include novel absolute token positions. They are **not a protected test set**; later structural-generalization testing requires a new fixed protocol and position-matched control.

## Limits and distinctions from historical ML-080

The exposure factor is **inspired by**, not an exact reproduction of, ML-080: here each dose sentence supplies four complementary role labels, and common base examples never have a second-passive event at all. Earlier ML cohorts varied which event/voice was *queried* and could include passive distractor clauses. This new stricter design makes the 2×2 interaction interpretable on one common synthetic grammar, but it does not directly prove the same historical effect or neural mechanism.

## Practical commands

From `exp102-clean-design`:

```bash
cp ~/Downloads/exp103_factorial_bridge.py scripts/
cp ~/Downloads/exp103_factorial_design.md paper/
python3 scripts/exp103_factorial_bridge.py --phase prepare
python3 scripts/exp103_factorial_bridge.py --phase smoke --device mps
python3 -u scripts/exp103_factorial_bridge.py --phase run --seeds 10501 --device mps
python3 scripts/exp103_factorial_bridge.py --phase summary
```

Run the eight seeds only with the **locked**, unchanged files. Preserve reports in `reports/exp103/factorial_dev/`, and back up `.pt` checkpoints separately (`results/exp103/factorial_dev/`). Do not modify the old ML or EXP-102 evidence.
