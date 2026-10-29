# Model Linguistics — ML-029 to ML-031 Research Update

Date: 2026-10-29
Status: Exploratory causal experiments completed.

## Research question

How do transformer language models coordinate event meaning,
grammatical construction, question identity, and answer selection?

## ML-029: Event vs answer disentanglement

Design: event reversal x question reversal, with opposite-voice donors.
Layer 20, readout position, 128 interventions per model.

Llama 3B:
- Answer-mismatch interaction: -1.8452 (32/32 negative)
- Event contrast: +0.0205
- Question contrast: +0.6299

Qwen 7B:
- Answer-mismatch interaction: -4.1626 (24/32 negative)
- Event contrast: -0.1494
- Question contrast: +10.1300

Interpretation:
Late-state effects are associated with answer identity and question
context. An answer-only interpretation is insufficient for Qwen.

## ML-030: Earlier readout intervention

Llama 3B, layer 13:
- Answer-mismatch interaction: -0.1252 (25/32 negative)
- Event contrast: -0.0349
- Question contrast: +0.1960

The answer-associated causal effect is substantially weaker at
layer 13 than layer 20 under this intervention.

This does not establish when answer information is first computed.

## ML-031: Donor-voice factorial

Qwen 7B, layer 20, readout.
Factors: donor voice x event reversal x question reversal.

Same-voice donors:
- Event contrast: -0.082
- Question contrast: -17.783 (32/32 negative)
- Answer-mismatch interaction: -4.312

Opposite-voice donors:
- Event contrast: -0.149
- Question contrast: +10.130
- Answer-mismatch interaction: -4.163

Interactions (opposite minus same):
- Voice x event: -0.067
- Voice x question: +27.913 (32/32 positive)
- Voice x answer-mismatch: +0.150

The voice-by-question interaction remains strong even when
considering reversed-event donors separately.

## Scientific interpretation

The central empirical observation is a large voice-by-question
interaction in Qwen's late readout-state intervention effects.

This is evidence of contextual and construction-dependent causal
sensitivity. It is NOT proof of a coordinated grammar circuit,
an abstract relational representation, or a universal mechanism.

Limitations:
- Small, repeatedly examined stimulus set.
- Only two name pairs and two verbs.
- Event reversal also changes name order.
- Question reversal changes textual context.
- Answer mismatch is an event x question interaction.
- Same/same/same is a zero-effect self-control.
- Cross-construction residual patches may be off-manifold.
- Layer 20 is not necessarily homologous across models.
- No new held-out replication has been performed.

## Next research direction

Independently audit the implementation and causal interpretation.
Develop fresh held-out tests with strong surface, positional,
answer-identity, construction, and activation controls.

Find falsifiable evidence for or against coordinated relational
computation rather than assuming that the hypothesis is true.

Preserve all negative findings.
