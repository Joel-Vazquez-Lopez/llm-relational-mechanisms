# Model Linguistics — ML-032 through ML-036
Date: 2026-10-30
Status: Exploratory mechanistic research checkpoint

## Central research question

Do Transformer models organize relational information through
coordinated, context-dependent computations, and can we identify
and predict the causal dependencies between those computations?

This remains a hypothesis, not an established mechanism.

## ML-032 — Question paraphrase transfer

Model: Qwen 7B.
Behavioral competence: 44/64, below frozen 56/64 gate.
Passive-agent accuracy:
- Original question wording: 0/8
- Paraphrased question wording: 5/8

Gate: FAIL. No causal patches performed.

Interpretation: relational answer behavior is sensitive to wording,
construction and lexical context. This does not identify the cause.

## ML-033 — Causal state compatibility

Qwen 7B, layer 20, 16 fixed recipient conditions.

Voice x question contrast:
- True donor alpha=.25: +6.666, positive 16/16
- True donor alpha=.50: +13.914, positive 16/16
- True donor alpha=1.00: +28.039, positive 16/16
- Signflip alpha=.25: -0.200
- Signflip alpha=.50: -0.369

Donor-minus-control at alpha=.50:
+14.283, positive 16/16.

Frozen primary criterion: PASS.

Interpretation: direction-specific causal sensitivity beyond the
tested sign-flipped controls. These controls do not establish that
the donor perturbations are on-manifold.

## ML-034 — Interaction decomposition

Qwen 7B, layer 20.

Alpha=.50:
- Total interaction: +13.8145
- State-nonadditivity contribution: +14.6907
- Downstream-nonlinearity contribution: -0.8762
- Absolute downstream/total ratio: 0.1481

Alpha=1.00:
- Total: +28.3174
- State contribution: +24.9774
- Downstream contribution: +3.3400
- Absolute downstream/total ratio: 0.1642

Frozen primary prediction: PASS.

The decomposition is exact algebraically but depends on the
chosen synthetic additive state. It does not establish where
or when the underlying computation occurs.

## ML-035 — Interaction-vector causal trace

Layers 3, 7, 13, 20.
Perturbations normalized to 20% of recipient state norm.

Layer | Delta/state norm | Excess over controls
3     | 0.004            | -0.106
7     | 0.006            | -0.032
13    | 0.017            | +0.247
20    | 0.392            | +11.553

Layer-20 frozen gate: PASS, positive excess 16/16.

Early vectors were amplified relative to their natural magnitudes.
Layer comparisons do not identify where the interaction originated.

## ML-036 — Interaction trajectory

Qwen 7B has 28 transformer blocks; outputs 0..20 examined.

Layer | Projection onto L20 | Increment
16    | 0.0046              | +0.0019
17    | 0.0219              | +0.0173
18    | 0.1412              | +0.1192
19    | 0.3448              | +0.2036
20    | 1.0000              | +0.6552

Increments for outputs 17..20 positive in all 16 cases.

This is descriptive, not causal localization. The L20 reference
is selected by prior exploration, and projection at L20 equals
one by definition.

## Main finding

A task-conditioned voice x question interaction in Qwen residual
states is causally consequential under tested interventions.

Its relative magnitude and alignment with its L20 form increase
markedly over block outputs 17..20.

These experiments do NOT establish:
- An abstract grammatical representation
- A unique attention-head or MLP circuit
- Generalization to unseen lexical or structural conditions
- The necessity of this interaction for relational competence
- A shared mechanism across model architectures

The 16 analyzed cases share underlying factorial states and
should not be considered 16 independent replications.

## Next scientific priorities

1. Independently audit intervention implementation and conclusions.
2. Examine outputs L21..L27 using alternative reference layers.
3. Causally separate attention and MLP contributions at candidate blocks.
4. Test prospective predictions on genuinely new relationships.
5. Compare with Llama using functionally aligned intervention sites.
6. Train a standard small Transformer from scratch on controlled
   relational tasks, retaining independent seeds and checkpoints.
7. Extend functional tests to encoder-only or other Transformers.

Keep negative results, raw data, design hashes and failed predictions.
Do not promote exploratory effects into confirmed mechanisms.
