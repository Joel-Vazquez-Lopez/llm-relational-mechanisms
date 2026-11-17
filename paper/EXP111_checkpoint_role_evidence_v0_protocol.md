# EXP-111 — Read-only role evidence across layers in EXP-109 checkpoints

**Status:** post-hoc exploratory mechanistic *diagnostic*; NOT a causal-mechanism result, independent replication, or a new confirmatory structural-generalization evaluation.

**Inputs:** the **16 existing** EXP-109 6,400-step checkpoint bundles (8 training seeds × 2 MPS numerical repetitions), each containing the `pa_only`, `indirect_pp_aa`, and `direct_pa_ap` state dictionaries. No Transformer retraining, checkpoint selection, parameter changes, or new exposure interventions. The final output contains one JSON report per model (48 total), and a separately aggregated summary. The source EXP-109 reports and manifest must pass provenance checks.

## Scientific motivation

Frozen EXP-109 primary outcome: `PA_literal − AP` all-four-question accuracy in PA-only models, +61.4 percentage points (95% t interval +41.3 to +81.4 pp; 8/8 seeds positive). The *subsequent exploratory* EXP-110 audit shows that AP has especially many wrong-role responses, including reversed agent/patient assignments, relative to literal PA. We ask whether the native model's role direction evidence differs systematically between AP and its token-identical literal PA reorder, and at what layer its prediction-facing signal becomes accessible. **This analysis does not establish an internal mechanism.**

## Data policy and unit of analysis

- Probe fitting data: the original EXP-106 familiar development examples for **AA and PA only**: 144 source sentences × four questions. This adds a supervised readout but never trains the Transformer. No AP examples and no EXP-109 evaluation labels are used to fit probes. The earlier development sentences have already been examined in exploratory research, so the fitted probes do not create an untouched development set.
- Probe evaluation and native logit analysis: **144 previously scored EXP-109 AP sources and their 144 paired literal PA swaps**, four queries each, drawn by the unchanged frozen EXP-109 generator and verified by its evaluation digest. These data are already analyzed in EXP-109/110. This is *post-hoc* and cannot be counted as fresh confirmation.
- No source text, per-example predictions, or new model checkpoints are written. No within-event L+I examples, reserved/protected examples, or independent protected-test access.
- Average the two numerical repetitions **within each seed** before computing any across-seed mean or descriptive t interval; do not count sentences or 48 correlated models as independent scientific replicates. No significance claims from exploratory multiple comparisons.

## Models and measurements

Uses the exact `pilot.Model` (three pre-LN causal Transformer blocks, width 96, four heads, 48-token maximum, full-vocabulary output) and reconstructs its forward pass, verifying that the final residual's logits match the native model forward. All operations are within each loaded checkpoint; no cross-model activation alignment or substitution.

For both `AP` and `PA_literal`, at the answer token of every first/second agent/patient question:

1. **Native final-layer role margin:** `logit(correct participant) − logit(other participant of the correct event)`. This is a *descriptive logit difference*, not an oracle-restricted prediction; an incorrect-event participant can still have higher output logit. Report the mean, positive-margin fraction, and per-question category (first-agent, first-patient, second-agent, second-patient).
2. **Native event margin:** `logit(correct participant) − max(logits(other event's two participants))`. Compare wrong-event and wrong-role error frequencies under the unmodified four-name candidate decision. Native C_query and unrestricted-vocabulary accuracy must reproduce the archived EXP-109 report within one numerical count, or the analyzer refuses to save its result.
3. **Residual-layer logit lens:** Capture answer-token hidden states at `embed` and after each of blocks 1/2/3. Apply the **same trained final norm and output head** to each intermediate residual. This is an *uncalibrated* logit lens: intermediate states were not trained to use the final readout directly. Intermediate margin signs are suggestive visualizations, not evidence that a circuit is necessary or a role is represented in a specific neuron.
4. **Supplemental fixed linear probe:** A ridge-regularized, full-name decoder fitted separately at each layer to earlier **AA and PA** sentences, using the same supervision for all arms. Fixed L2 λ=10, centered one-hot targets, standardized features, no hyperparameter tuning on EXP-109. Probe scoring uses only the four names in the already evaluated sentence. A deterministic shuffled-training-label control is also fitted. A trained probe can learn a *new decision rule*, so probe success does **not** prove that the native output computation uses the decoded information.

The primary useful descriptive comparison is whether AP errors have negative native role margins for the *second event* while literal PA margins are positive, especially at the final model layer, and whether earlier residuals display a related pattern. A second question is whether the fixed readout probe succeeds on AP when the native head fails; alternative explanations (a newly learned probe decoder, lexical biases, coarse syntax, and sentence position) remain.

## Audit and falsification safeguards

- No conditioning on success, competence gate, favorable seeds, or best checkpoints.
- Output is post-hoc descriptive even if some intervals do not cross zero; multiple layer, query and probe comparisons are exploratory.
- No new learning curves: EXP-109 saved **only final-step checkpoints**, so this script cannot make claims about how representations changed *over training time*. Testing developmental mechanisms requires a new, separately predeclared checkpoint-saving design.
- No causal attribution from a logit lens or decodability. Future within-model interventions must include matched controls, preservation of overall prediction ability, explicit counterfactual predictions and independent validation on new models/evaluation instances.
- `prepare` validates source reports/checkpoints without scoring evaluation sentences; `smoke` uses only the **old** dev examples and an untrained model. `run` reconstructs the EXP-109 already-used evaluations and loads a saved checkpoint read-only. `summary` requires **all 48** run reports before giving an aggregate.
- Original EXP-109 JSON, frozen manifest, 16 `.pt` checkpoints and all protected/L+I materials remain unchanged.

## Running locally

From the experimental worktree `~/Developer/exp102-clean-design` using the already active Python environment:

```bash
unzip -n ~/Downloads/exp111_checkpoint_role_evidence_v0_bundle.zip -d .
python3 scripts/exp111_checkpoint_role_evidence_v0.py --phase prepare
python3 scripts/exp111_checkpoint_role_evidence_v0.py --phase smoke --device mps

# First diagnostic against one saved model:
python3 -u scripts/exp111_checkpoint_role_evidence_v0.py --phase run \
  --seed 10901 --repeat 1 --arm pa_only --device mps
```

Then to evaluate every model (skip any outputs already completed without modifying them):

```bash
set -o pipefail
for seed in 10901 10903 10907 10909 10913 10919 10921 10927; do
  for repeat in 1 2; do
    for arm in pa_only indirect_pp_aa direct_pa_ap; do
      result="reports/exp111/checkpoint_role_evidence_v0/seed${seed}_repeat${repeat}_${arm}.json"
      if [ -f "$result" ]; then echo "SKIP $result"; continue; fi
      python3 -u scripts/exp111_checkpoint_role_evidence_v0.py --phase run \
        --seed "$seed" --repeat "$repeat" --arm "$arm" --device mps || exit 1
    done
  done
done
python3 scripts/exp111_checkpoint_role_evidence_v0.py --phase summary
```

Do not interpret a single strong or weak model until the aggregate is available. Store source scripts and derived results in Git if approved, but do not commit the raw `.pt` files.

## Implemented/tested locally

- Syntax compilation PASS.
- CPU `smoke` PASS: exact native forward consistency, ridge and logit diagnostics on old-dev sentences only.
- Frozen evaluation generator and vocabulary reproduce the bundled EXP-109 manifest digests.
- End-to-end `run` PASS in an isolated **random-weight synthetic checkpoint fixture**, using synthetic report metadata; it verified loading, readout, metric matching and report writes. **No actual Mac-trained EXP-109 checkpoint was available for testing here**.
- Aggregate `summary` PASS on a synthetic 48-report fixture, not a scientific result.
