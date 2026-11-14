# EXP-105 positional-coverage control — design preflight v1

**Unfrozen design audit only. No training, reserved examples, or protected evaluation.**

Time-frame token offsets: 0, 4, 6, 8, 10.
Development L+I sources: 1800; longest full input: 36 tokens.
Distinct development full-prompt (token, absolute-position) pairs: 480.

| Arm | Original training sources | Framed training sources | Missing token/positions before | Missing after | Max train tokens |
|---|---:|---:|---:|---:|---:|
| familiar | 13500 | 67500 | 109 | 0 | 42 |
| target | 13500 | 67500 | 96 | 0 | 42 |

**PASS:** All development L+I full-prompt exact token/position pairs and lengths are covered separately by each training arm; no within-event L+I examples enter training.

## Remaining interpretation limits
- Absolute position coverage is necessary, not sufficient, for systematic structural generalization.
- Training on longer neutral time frames changes the training distribution and should be included in both arms in any future runner.
- Exposure AA versus AP still changes passive syntax, surface length and event-specific voice coverage; these are the intended intervention plus inseparable possible alternatives.
- No conditional alignment check proves that the familiar arm has seen every structural token in the same syntactic context.
- This is the existing EXP-102-style grammar family and not an independent grammar replication.
- The dev L+I examples were generated for design audit only. Reserved name-set identifiers were not rendered.
- Preflight passed, but a new training protocol must still be designed, piloted and frozen before confirmatory training.

**Next:** independently review the new training distribution and exposure-matching plan before using it in a model runner.
