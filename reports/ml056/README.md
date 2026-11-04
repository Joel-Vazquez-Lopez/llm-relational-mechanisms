# ML-056: Controlled exposure and relational generalization — feasibility pilot

**Status: development / calibration-only feasibility experiment.** This is part of the larger [LLM Relational Mechanisms](../../README.md) programme; no pretrained-model claim or neural causal mechanism has been established.

## Research question

Do controlled differences in training histories change what a small causal Transformer learns about relational operations, and is that difference associated with generalization beyond memorized operator combinations? This pilot verifies training competence and calibration generalization before any protected evaluation or causal-neural interpretation.

## Inputs and design

- 10 identities, 2 actions, three visible participants (actor, patient, distractor).
- 20 degree-balanced role-bearing training edges per graph; 21 common held-out actor–patient pairs across three matched feasible graphs (ring, lower, higher).
- Exposure arms A and B each use 12 three-symbol operator strings, with identical operator-by-position frequencies, operation counts, and final role-position permutation distributions, but different co-occurrences.
- **Protected operator strings:** `IRS`, `ISR`, `RSI`, `SRI`. Do not evaluate them while developing the model or selecting a training recipe.
- **Calibration operator strings:** `SIS`, `RIR`. They are distinct from both training and protected operator strings.
- The generator exposes each name in multiple distractor contexts; held-out actor–patient pairs are **not** held out from all co-occurrence.
- A compositional position-based algorithm can solve the task. Accuracy alone is not evidence of a semantic-role mechanism.

## Run from the repository root

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/ml056/requirements.txt
python scripts/ml056/ml056_run.py --phase preflight
python -m unittest discover -s scripts/ml056 -p 'test_ml056_offline.py' -v
python scripts/ml056/ml056_run.py --phase batch --graphs ring --arms A B --seeds 101 --steps 1500 --batch-size 256
python scripts/ml056/ml056_run.py --phase summarize
```

When the paired ring runs are reviewed, replicate with seeds 103, 107 and 109 and then other graphs using the same locked budget. Three graphs x two arms x four seeds amounts to 24 trained models; it is not a general graph law.

## Results, provenance and safeguards

- Runner emits `scripts/ml056/runs/<run_id>/checkpoint.pt` and `metrics.json`, and writes `scripts/ml056/runs/calibration_summary.json` when summarizing. The local `runs/` folder is gitignored.
- Keep run directories and checkpoints for later mechanistic evaluation. Only promote reviewed findings into the central research log, linking exact hashes and source revisions; no cherry-picking seeds.
- The pilot preserves architecture, initialization and sampling seed for matched A/B comparisons, and uses a fixed training budget without adaptive early stopping.
- The single attached reference smoke result is **one exploratory CPU run**, not evidence of an exposure-arm effect; it is included for debugging only.
- Neural bottleneck interventions have **not** been validated by this training runner. The hand-coded identity-versus-pointer programs in `ml056_offline_pretraining.py` are oracle instrumentation checks, not discovered Transformer circuits.
- Do not interpret training-set fit, calibration failure or small seed differences as identifying the learned neural algorithm.

## Required checkpoint before a protected evaluation

1. Demonstrate that models learn the supervised task robustly without looking at protected operator examples.
2. Audit the operator/label distribution, competence, complete name and pair coverage, and calibration behavior across all intended seeds.
3. Specify a physically realizable intervention, sites, transport/remapping, controls and acceptance/failure criteria *before* protected neural testing.
4. Freeze architecture, training recipe, data hashes, analysis plan and model-selection criteria in a committed manifest. Only then evaluate the protected split once.
