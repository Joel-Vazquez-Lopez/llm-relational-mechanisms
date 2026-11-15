# EXP-105 exposure matching — development-only preflight v2

**Status: unfrozen design audit. No training, reserved examples, or protected evaluation.**

**Proposed primary contrast: PA (first-event passive) vs AP (second-event passive).**
The earlier AA-vs-AP contrast additionally changed the number of passive events.

Shared framed base sources: 54000; 
per-arm framed dose sources: 13500; 
development L+I sources: 1800.

| Training arm | Shared base | Exposure dose | Total training sources | Passive event occurrences | Missing L+I token-positions | Max input tokens |
|---|---:|---:|---:|---:|---:|---:|
| AA | 54000 | 13500 | 67500 | 27000 | 0 | 42 |
| PA | 54000 | 13500 | 67500 | 40500 | 0 | 42 |
| AP | 54000 | 13500 | 67500 | 40500 | 0 | 42 |

**PASS:** PA and AP exposure doses have identical paired cores, gold answers, modifier frequencies,
question counts, verb frequencies, source token counts and full-input-length histograms.
Both retain zero missing development L+I token/absolute-position pairs.

Individual paired sources of equal length: 1500/2700 (modifiers change comma placement).
Individual paired sources with identical token multisets: 1500/2700.

**Still different by design:** PA vs AP source token-position histograms differ at 544 token/position keys
(L1 count difference 126030).

## Limits and next action
- PA vs AP matches passive counts, aggregate input-length distribution, aggregate lexical counts and gold labels. Individual I-modifier pairs can differ in punctuation and length.
- Token-by-position occurrence frequencies still differ; full support coverage is not contextual or causal equality.
- A training runner must predeclare batch source/voice/modifier/position scheduling, not infer matching from aggregate pools alone.
- Only dev name-set compositions were generated. Existing reserved identifiers were never rendered.
- The design uses an existing synthetic grammar family; independent grammar replication remains a separate goal.
- This is not a frozen confirmatory protocol and gives no permission to train or open protected evaluations.

**Next:** write and review an explicit source/label sampling schedule for a disposable
development-only training smoke test. Do not freeze or start confirmatory training yet.
