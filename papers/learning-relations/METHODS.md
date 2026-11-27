[← Paper I](README.md) · [Results](RESULTS.md) · [Methods](METHODS.md) · [Mechanisms](MECHANISMS.md) · [Experiments](EXPERIMENTS.md)

# Methods

## Controlled relational task

A source describes two events with four distinct participants. Four questions target first/second event × agent/patient. AP and PA voices vary the mapping between surface order and semantic role. Neutral prefixes and name-bearing adjuncts vary surface context. The target is the relevant participant, not a grammatical-position label. The rich task uses five frame choices and nine modifier-pair cells. Training uses 240 quartets; 60 withheld quartets and 16 unseen mappings per familiar quartet provide two distinct transfer populations. All are derived from the original TRAIN identifiers, not the original protected split.

## Model and training

Three causal blocks, width 96, four heads, 48 token positions. AdamW lr .002, weight decay .01. Eight quartets per update, two voices, four queries per source (64 labels). Role diversity arms share initial weights and quartet/surface schedule within seed; allowable role assignments differ. The original budget was 6,400 updates; selected recovery continues to 51,200 using fresh optimiser state. Independent replication trains uninterrupted to 51,200. [Diversity protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_DIVERSITY_COMPLEX_V0_PROTOCOL.md) · [Replication protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_MECHANISM_REPLICATION_V0_PROTOCOL.md).

## Metrics

- **C_query:** strict four-query joint correctness under four-candidate readout. All four must be correct. Separate from individual-answer accuracy (accuracy4), full-vocabulary argmax, full-vocabulary joint correctness or generated text.
- **Event mass:** P(gold participant)+P(other participant of gold event), on full vocabulary.
- **Conditional role probability:** P(gold participant)/event mass; not a binary accuracy.
- **Conditional role accuracy:** gold participant versus event-mate logit comparison. Circuit diagnostics gives half credit for ties; later repair/origins implementations use strict greater-than. Consult the original protocol rather than pooling these definitions.
- **Both-pair collapse:** same candidate predicted for both role queries in each event.

For a gold participant a and event mate b, token cross-entropy decomposes exactly as −log(P(a)+P(b)) − log(P(a)/(P(a)+P(b))). The second term approaches log 2 under equal role probability. A high event mass alone does not imply correct binding.

## Controls and inference

Seed is the independent unit in paired training contrasts, not a question or token. Selected recovery/repair cohorts and diagnostic samples are exploratory. Candidate-restricted accuracy and full-vocabulary intervention scores are kept distinct. Probes use disjoint fit/test sources, surface/voice transfer and shuffled labels; they do not establish causal use. Causal swaps use within-model donors, matched queries and self-patch/reconstruction checks. See [Mechanisms](MECHANISMS.md).

## Crossed origins study

Eight initialisation seeds × eight training-stream seeds; each row shares initial parameters, each column shares exact scheduled batches. Finite-run exposure counts may differ across streams. Final primary outcome is equally averaged C_query across four rich held-out groups, with descriptive two-way sums of squares. One run per cell cannot distinguish interaction from execution noise. Dense fixed-sample trajectories and separate event/role gradient diagnostics are exploratory; no predictor or prevention intervention is tested yet. [Protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_LEARNING_ORIGINS_V0_PROTOCOL.md).
