[← Model Linguistics](../../README.md) · [Research history](../../investigations/README.md) · [Paper I](../learning-relations/README.md)

# Paper II · Order-to-Role

### How pretrained language models store who did what to whom, and how a question reads it out

*Working title: "Thematic roles live on the names: how language models represent who did what" (provisional).*

**Status, 27 November 2026.** Development work is complete for five open models. Eight claims were frozen on 26 November and tested once on held-out names: **37 of 38 checks pass** in English, **15 of 15** on a broader English replication, **29 of 30** in Spanish and **30 of 30** in Galician ([scorecards](EVIDENCE.md#1-confirmatory-results-test-set-0)). A 70B-class extension is in exploratory development. Manuscript in preparation.

---

## The question

*Leon praised Mira* and *Mira was praised by Leon* describe the same event. In the first, Leon is the first name, the grammatical subject and the agent. In the second, he is still the agent, but he is neither first nor the subject. Linguistic theory keeps three properties apart that usually travel together: **word order**, **grammatical function** and **thematic role**.

When a language model answers "who did it?", which of the three does it actually use? Where in the network is that information, and how does a question get to it?

| Construction | Example | First name | Subject | Agent |
|:--|:--|:--|:--|:--|
| Active | Leon praised Mira. | Leon | Leon | **Leon** |
| Passive | Mira was praised by Leon. | Mira | Mira | **Leon** |
| Object cleft | It was Mira that Leon praised. | Mira | Leon | **Leon** |
| By-cleft | It was by Leon that Mira was praised. | Leon | Mira | **Leon** |

Across these four constructions the three properties are exactly uncorrelated, so their effects can be separated.

## What we found

1. **The names carry their thematic role.** Moving the hidden state of a single name from one sentence into another shifts the answer according to that name's role. Grammatical subjecthood adds little: on the core set its coefficient is near zero or negative. The role signal peaks in the middle of the network (28–58% of depth) in all five models, and is the largest of the three effects there. Word order or focus also carries a smaller signal in three models. → [Results §2](RESULTS.md#2-the-names-carry-their-thematic-role)
2. **A question reads the role out late, through a small set of attention heads.** At one model-specific depth the question's contribution is handed to the answer position. From there, 20 attention heads are enough to repair most of an error, against about 0–11% for 20 random heads from the same layers. → [Results §3–4](RESULTS.md#3-from-storage-to-answer-the-layer-pipeline)
3. **Storage and retrieval come apart.** One wording, *"Who was someone praised by?"*, makes models name the patient although the sentence, and therefore every hidden state of the names, is identical to a working question. The failure enters at the word "by", and three models still know the agent when asked differently. → [Results §5](RESULTS.md#5-stored-correctly-retrieved-wrongly)
4. **The same heads work beyond the templates.** Heads chosen on English template sentences repair answers on new English verbs, Spanish, Galician, Universal Dependencies sentences and Wikipedia text, with real or neutral names. → [Results §6](RESULTS.md#6-beyond-the-templates-new-words-languages-and-natural-text)
5. **The claims hold on held-out data.** The claims, layers, heads and pass rules were committed publicly before test set 0 was opened. → [Results §7](RESULTS.md#7-held-out-confirmation)

<p align="center"><img src="results/figures/fig3_question_to_answer.svg" alt="Five panels, one per model: swapping the whole question repairs the answer in early and middle layers and stops working at a model-specific depth, where patching the answer position starts to repair it; the 'by' token's effect peaks before that handover." width="100%"></p>

*In every model, the question matters until a fixed depth, after which the answer position carries the decision (development data; [details](LAYERS.md)).*

## Results at a glance

| Model | Failing / working question, dev | Role coefficient at role layer, dev [95% CI] | Handover layer | 20 frozen heads vs 20 random, held-out dev | Test set 0: English · BX · Spanish · Galician |
|:--|:--|:--|:--|:--|:--|
| Gemma 4 31B | 69% / 100% | 1.08 [0.92, 1.21] | 40 of 60 | .82 vs .05 | 8/8 · 3/3 · 5/6 · 6/6 |
| Qwen3.8 27B | 52% / 97% | 0.58 [0.54, 0.62] | 54 of 64 | .36 vs .11 | 8/8 · 3/3 · 6/6 · 6/6 |
| phi-4 14B | 11% / 89% | 0.59 [0.53, 0.65] | 22 of 40 | .75 vs .05 | 6/7 · 3/3 · 6/6 · 6/6 |
| OLMo 3.1 32B | 68% / 100% | 0.53 [0.46, 0.58] | 26 of 64 | .42 vs .00 | 7/7 · 3/3 · 6/6 · 6/6 |
| Granite 4.2 30B | 80% / 99% | 0.75 [0.64, 0.84] | 52 of 64 | .50 vs .01 | 8/8 · 3/3 · 6/6 · 6/6 |

Failing question: *"Who was someone V-ed by?"*; working question: *"Who was the V-ing done by?"* (generation, "Reply with one name only."). Test-set counts are claims passed out of claims made (C7 is not claimed for phi-4 or OLMo). The two failures are phi-4 C5 (its control heads also repair, .26 against a .10 limit) and Gemma Spanish C3 (CI [−0.01, 0.23]); both are kept and explained in the [evidence ledger](EVIDENCE.md#notes-on-the-confirmatory-results). Sources: [`behaviour_dev.csv`](results/tables/behaviour_dev.csv), [`role_code_dev.csv`](results/tables/role_code_dev.csv), [`layers_and_heads_dev.csv`](results/tables/layers_and_heads_dev.csv), [`confirmatory_test0.csv`](results/tables/confirmatory_test0.csv).

---

## Read the paper's evidence

| | Page | What it covers |
|:--|:--|:--|
| 1 | **[Results](RESULTS.md)** | The findings in order, with figures and tables, development and confirmatory results kept apart |
| 2 | [Methods](METHODS.md) | Linguistic design, stimulus sets, models, prompting, scoring, interventions, splits and freeze |
| 3 | [Layers and mechanism](LAYERS.md) | Role code across depth, question-to-answer handover, the "by" token, cross-model layer order |
| 4 | [Answers and interventions](INTERVENTIONS.md) | Answer generation, the scoring artefact, head repair, knockout, steering and controls |
| 5 | [Generality](TRANSFER.md) | New constructions, Spanish, Galician, natural sentences, real vs neutral names, 70B models |
| 6 | [Evidence ledger](EVIDENCE.md) | Confirmatory scorecards; exploratory, failed, inconclusive and ongoing work; corrections |
| 7 | [How it developed](DEVELOPMENT.md) | From the first behavioural test to the frozen evaluation, including dead ends |
| 8 | [Experiment map](EXPERIMENTS.md) | OTR-01 to OTR-15 and the stimulus generators |
| 9 | [Manuscript map](PAPER_MAP.md) | Section and figure plan of the working manuscript |
| 10 | [Reproduce](REPRODUCE.md) | Rebuilding every table and figure, and re-scoring the frozen evaluation |
| 11 | [Integration](INTEGRATION.md) | Where the code and results live now, and the plan to unify them on `main` |

## Models and scope

Five instruction-tuned open models of 14–32B parameters passed a behavioural inclusion gate fixed on development data: **Gemma 4 31B-it, Qwen3.8-27B, phi-4, OLMo-3.1-32B-Instruct and Granite-4.2-30B**. Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct are an exploratory extension. Earlier pretrained-model experiments in the [historical record](../../investigations/pretrained_llms/README.md) motivated this study, but they are not part of its evidence.

## Where things live

The numbers on these pages are rebuilt from archived result files by [`analysis/build_results.py`](analysis/build_results.py). Source commits and the checksum of every input file are recorded in [`results/MANIFEST.json`](results/MANIFEST.json). The OTR code comes from [`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) and is being merged into `main` with its history intact (see the [integration plan](INTEGRATION.md)). The raw outputs stay on the provenance branch [`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev); readers do not need them, because every number above comes from the curated tables in [`results/`](results/).

*Paper II section revision: 27 November 2026. Code snapshot `9c43792`, results snapshot `25c6914`.*
