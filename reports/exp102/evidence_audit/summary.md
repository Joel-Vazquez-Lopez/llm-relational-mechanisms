# EXP-102 — eight-seed evidence audit

**Status:** exploratory, existing saved reports only; no new protected evidence.

## Primary comparison

Quartet minus balanced-unpaired, on four-name-set-OOD `C_query` at step 1600:

- Mean paired difference: **+35.94 percentage points**
- Seed-bootstrap 95% percentile interval: **[+8.11, +65.62] pp**
- Wins/ties/losses: **5/3/0**
- Exploratory exact two-sided sign-flip p: **0.0625**

This p-value should not be used as a preregistered confirmatory significance claim.

## Eight-seed learning curves

| Updates | Quartet | Balanced unpaired | Role pair | Event pair |
|---:|---:|---:|---:|---:|
| 400 | 32.5% | 0.0% | 0.9% | 0.5% |
| 800 | 76.6% | 14.7% | 26.2% | 6.6% |
| 1200 | 99.7% | 44.4% | 60.1% | 40.9% |
| 1600 | 100.0% | 64.1% | 83.3% | 49.7% |

## Seed-level final OOD C_query

| Seed | Quartet | Balanced unpaired | Role pair | Event pair |
|---|---:|---:|---:|---:|
| 10301 | 100.0% | 100.0% | 94.1% | 93.8% |
| 10303 | 100.0% | 100.0% | 100.0% | 100.0% |
| 10309 | 100.0% | 0.0% | 99.6% | 0.0% |
| 10313 | 100.0% | 37.5% | 72.7% | 6.2% |
| 10321 | 100.0% | 0.0% | 0.0% | 0.0% |
| 10331 | 100.0% | 98.8% | 100.0% | 0.0% |
| 10333 | 100.0% | 100.0% | 100.0% | 97.7% |
| 10337 | 100.0% | 76.2% | 100.0% | 100.0% |

## Interpretation safeguards

- The synthetic task/ID/OOD data family was repeatedly explored before this seed panel.
- Different methods change unique source count and within-update co-occurrence; not a uniquely identified mechanism.
- The statistical replication unit is initialization seed (n=8), not 256 sentences per seed.
- Name-set-OOD tests recombination of familiar vocabulary/grammar, NOT unseen syntax.
- Performance uses four-name candidate-restricted decoding, not natural-language generation.
- Learning-curve milestone comparisons are secondary/descriptive; primary step was 1600.

**Next appropriate test:** a separately frozen, syntax-compositional
generalization evaluation with prevalidated labeling and anti-shortcut controls.
