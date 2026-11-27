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

Patching shows that the heads are *sufficient* to repair the answer. OTR-12 asks whether they are also *needed*. At the answer position, it mean-ablates the top-k OTR-09 heads, replacing each head's output with its mean over other items. It does this in the working question and in an ordinary agent question ("Who V-ed someone?"), and compares the result with ablating k random heads from the same layers. The heads were ranked on other development families, and evaluation uses 51 held-out items per model. Separately, OTR-12 adds one averaged "working − failing" direction to the residual stream. **Development only, exploratory.** These runs are not frozen claims; full tables are in [supplement S3 §5](supplementary/S3-layerwise-and-heads.md#5-are-the-heads-needed-knockout-otr-12).

| Model | Agent margin kept after ablating the top-20 heads | … after 20 random heads | Answers still naming the agent: none ablated / top-20 ablated |
|:--|--:|--:|:--|
| phi-4 | 18% | 100% | 90% / 65% |
| Gemma | 38% | 100% | 100% / 92% |
| Qwen3.8 | 40% | 100% | 100% / 86% |
| Granite | 49% | 100% | 100% / 98% |
| OLMo | 84% | 100% | 100% / 100% |

*Working question, k = 20. The ordinary agent question gives the same picture: 21%, 37%, 43%, 49% and 82% of the margin kept.*

- **The heads are needed for much of the agent margin.** Ablating them removes 51–82% of the margin in four models; ablating random heads from the same layers removes none. The effect grows with k (supplement figure).
- **They matter for ordinary agent questions too.** These heads are part of how the agent is retrieved in general, not only of the failing/working contrast.
- **They are not universally necessary for a correct answer.** Most answers survive the ablation: 65% (phi-4) to 100% (OLMo) still name the agent, although removing these heads does change some answers. The heads are an important part of the read-out, but the evidence does not establish an exclusive pathway.
- **OLMo is the exception, and it fits the scan window.** Only 16% of its margin goes, but its head scan began at layer 35, after its handover at layer 26, so its top heads are not the ones at the handover ([S3 §3](supplementary/S3-layerwise-and-heads.md#3-how-many-heads-and-where-they-are)).
- **One averaged direction does not act as a "fetch the agent" switch.** Adding the averaged direction at the handover layer partly repairs the failing question: .13–.37 of the margin at strength 1, up to .77 at strength 2. But at strength 1 it barely turns a patient question into an agent question (≤ .20 of answers, except phi-4 at .49), and subtracting it from the working question rarely breaks it (≥ 92% still correct at strength 1). A direction fitted on actives transfers inconsistently to the other constructions. Random directions of the same norm do nothing, except in Gemma, where they push the margin the wrong way (−.75).

**Limits.** Development data only. 51 items per model, one random draw per k, and no intervals, because the runs did not store per-item records. The ablation acts at the answer position only. The knockout therefore establishes partial necessity, not necessity of a complete circuit.

## 5. Question-token interventions (OTR-13, OTR-14)

Single question tokens are swapped between the two questions, and attention from the answer position to the question or to the names is blocked layer by layer. The state at "by" carries most of the failure in every model [test0, C6, C8]; details in [Layers §2](LAYERS.md#2-stage-two-the-question-and-the-word-by-otr-13-otr-14).

## 6. The answer process: does the model know? (OTR-07)

On items whose bare answer names the patient, the model is asked to continue "Someone was V-ed by". The prediction at the by-slot names the agent on 100% of Gemma's wrong items, 82% of Qwen3.8's and 84% of Granite's [test0, C7]. On development data the share is .59 for OLMo and .29 for phi-4, so no claim was made for them.

An earlier version used a prefix that named the patient ("Mira was praised by"). It overstated the effect, because the patient's name in the prefix could point the model to the other name. Both versions are kept in the results; only the neutral one is claimed.

## 7. Behaviour through an API

A runner for hosted frontier models (generation only, no internals) exists for a possible appendix ([`otr_generate_api.py`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr_generate_api.py), with `api_behaviour.py` and their tests). No API result file is in the repository or in the results archive, so no API behaviour is part of the evidence. Any such results would be behavioural only and outside the frozen claims.
