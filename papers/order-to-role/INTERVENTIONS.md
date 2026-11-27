[← Order-to-Role](README.md) · [Results](RESULTS.md) · [Layers](LAYERS.md)

# Answers and causal interventions

How the study measures an answer, and how it changes answers from inside the model. The first section explains why measurement mattered so much in this study. The rest describes each intervention with its controls and results.

## 1. Measuring the answer

**Forced choice versus generation.** Early runs compared the log-probabilities of the two names as the first answer token. Without an instruction, chat models usually restate the sentence ("In the sentence 'Lisa followed Laura,' the person who was followed is Laura."). Forced choice then scores which name begins the restatement, usually the sentence's first name. On development items this produced an apparent human-like subject–object asymmetry that free generation removes:

| Model | Forced choice (2,624 items) | Generation, "Reply with one name only." (256 items) |
|:--|--:|--:|
| Qwen2.5-14B-Instruct | .71 | .96 |
| OLMo-2-32B-Instruct | .78 | .89 |
| Glimmer-30B | .74 | .81 |
| Qwen2.5-7B-Instruct | .64 | .74 |
| Mistral-Small-3.2-24B | .84 | .88 |
| OLMo-2-13B-Instruct | .88 | .81 |

*Development data, different item samples for the two columns; [review package, table t1](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/tables/t1_forced_choice_vs_generation.csv).*

The artefact is not a uniform inflation of errors (OLMo-2-13B moves the other way). Since 16 November, behavioural claims use greedy generation with a bare-name instruction and a rule-based extractor ([`otr_generate_check.py`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr_generate_check.py)). The extractor credits the gold name in a bare answer, a frame or a full sentence, and counts ambiguous or empty replies as wrong. Mechanistic measures use the first-token logit margin, but only inside the bare-name prompt, where the first token is the name.

**Prompt format (OTR-15).** Whether the generation effect survives other prompt layouts is tested in OTR-15. These results are exploratory and not part of the frozen claims.

## 2. Answer-position repair (OTR-09)

**What it does.** At one layer, the residual stream at the answer position is copied from the working run ("Who was the V-ing done by?") into the failing run ("Who was someone V-ed by?").

**What is measured.** The repair is the share of the logit-margin gap that comes back: (patched − failing) / (working − failing), per item. Only items with a working–failing gap above 1 logit are included.

**Result.** Repair is near zero in early layers, rises at a model-specific handover layer and approaches 1 near the output ([Layers §3](LAYERS.md#3-stage-three-the-handover-to-the-answer-position-otr-09-otr-14)). The handover layer is confirmed on held-out names [test0, C4].

## 3. Head-level repair and its controls (OTR-09)

**What it does.** The same patch, applied to the outputs of single attention heads. Heads are ranked on part of the development families; the top 20 are then evaluated on held-out families and on new materials.

| Control | What it rules out | Result |
|:--|:--|:--|
| 20 random heads from the same layers (5 draws) | That any heads at that depth would do | ≤ .11 on development; ≤ .07 on test0 |
| Head outputs from another item's failing run (`other_failing`) | That the effect is generic | ≤ .10 in four models on test0; **.26 in phi-4 (C5 fail)** |
| A random push of the same size (`random_nudge`) | That any perturbation repairs | ≈ 0 in all models on test0 |
| Another item's working-run output (C5b, Spanish/Galician) | That the repair is not item-specific | Own output repairs more in all ten model–language cells [test0] |
| Heads ranked afresh on each new set | That the frozen heads are a fit to the templates | 13–19 of the top 20 overlap; frozen heads repair about as well |

The phi-4 result shows why the content control was revised. Replacing the heads' outputs with any other item's output removes the heads' own wrong-way push, and that alone repairs about a quarter of the gap. The English failure stands, and the addendum scores C5b instead ([evidence ledger](EVIDENCE.md#phi-4-c5)).

## 4. Knockout and steering (OTR-12)

OTR-12 zeroes the selected heads in the working run, to ask whether they are needed rather than only sufficient. Separately, it adds an averaged "role" direction to test whether one direction can steer the answer. These runs exist for all five models (development, 19 November). They are not part of the frozen claims, and this section does not summarise them numerically yet.

## 5. Question-token interventions (OTR-13, OTR-14)

Single question tokens are swapped between the two questions, and attention from the answer position to the question or to the names is blocked layer by layer. The state at "by" carries most of the failure in every model [test0, C6, C8]; details in [Layers §2](LAYERS.md#2-stage-two-the-question-and-the-word-by-otr-13-otr-14).

## 6. The answer process: does the model know? (OTR-07)

On items whose bare answer names the patient, the model is asked to continue "Someone was V-ed by". The prediction at the by-slot names the agent on 100% of Gemma's wrong items, 82% of Qwen3.8's and 84% of Granite's [test0, C7]. On development data the share is .59 for OLMo and .29 for phi-4, so no claim was made for them.

An earlier version used a prefix that named the patient ("Mira was praised by"). It overstated the effect, because the patient's name in the prefix could point the model to the other name. Both versions are kept in the results; only the neutral one is claimed.

## 7. Behaviour through an API

A runner for hosted frontier models (generation only, no internals) exists for a possible appendix ([`otr_generate_api.py`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr_generate_api.py), with `api_behaviour.py` and their tests). No API result file is in the repository or in the results archive, so no API behaviour is part of the evidence. Any such results would be behavioural only and outside the frozen claims.
