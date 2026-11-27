# From Order to Role: handoff README (state on 16 Nov 2026)

**This is the file a new chat reads first.** It replaces the earlier context package from 14 Nov 2026. It records what the study is, what was found, what is new, where everything lives, how to run things, the rules, and what to do next. All numbers are from the **dev split** and are exploratory. **No test set has been opened.**

Owner: Joel Vázquez López. Working title: *From Order to Role: Do Language Models Know Who Did What to Whom?* Target: ACL/EMNLP main venue.

Joel's working style: he gives a broad mandate ("get me the paper"). He wants plain explanations, exact copy-paste commands without `#` comments (his Mac uses zsh), and honesty about what is and isn't shown.

---

## 0. Resume here (first actions for a new chat)

1. **Patches 0021–0025** (generation check, this README, figures, layer section, better answer extraction) are in `order-role-patches/` on the Mac. Apply and push them in order (§7). Check `git log --oneline -6` first: some may already be applied.
2. **Generation check, round 1 (16 Nov; 12 tokens; `results/otrgen_*_20261116T09*.json`).** It validated forced choice for two models and raised a **validity question** for the others (§2c):
   - **Gemma and Qwen3.8 answer with a bare name every time.** Generation accuracy (1.00 and 0.87) matches forced choice, so their scores are valid.
   - **Qwen2.5-14B-Instruct never answers with a bare name.** It writes "In the sentence "…," the person who was…", so 12 tokens was too short to judge. The main storyline rests on Qwen2.5-Instruct's forced-choice failure, so **round 2 must check that the failure also shows up in what the model generates**.
   - **Glimmer** is verbose. Where an answer could be extracted, it got "Who did someone V?" right (51/51), unlike its forced-choice 0.18. Its forced-choice failure is probably a scoring artefact.
   - **Mistral** mostly restates the question. Undecided.
3. **Run round 2** (commands in §7): 40 tokens, both with and without the instruction "Reply with one name only.", on Qwen2.5-14B-I, Qwen2.5-7B-I, OLMo-13B-I, OLMo-32B-I, Glimmer and Mistral.
4. **Do not open any test set yet.** First add a `--test-set` flag to OTR-02/03/04/05 and the generation check (§9).
   - **Only OTR-01 has this flag now.** In the other scripts, `--split test` would load **all four test sets at once**.
   - Then write the freeze document (§9).

---

## 1. The study

When an LLM answers "who did what to whom", does it track one of these, and where in the network is the answer decided?

- **Linear order:** first vs second name.
- **Grammatical function:** subject vs object.
- **Thematic role:** agent vs patient.

The constructions pull the three apart:

| Construction | Example (Leon = agent) | First | Subject | Agent |
|---|---|---|---|---|
| active | Leon praised Mira. | Leon | Leon | Leon |
| passive | Mira was praised by Leon. | Mira | Mira | Leon |
| object_cleft | It was Mira that Leon praised. | Mira | Leon | Leon |
| by_cleft | It was by Leon that Mira was praised. | Leon | Mira | Leon |
| subject_cleft (control) | It was Leon who praised Mira. | Leon | Leon | Leon |
| relative_active (held out) | The one who praised Mira was Leon. | Mira | Leon | Leon |
| relative_passive (held out) | The one who was praised by Leon was Mira. | Leon | Mira | Leon |

- Across the four main constructions the three labels are exactly uncorrelated.
- **Questions never name a participant**, so the model cannot answer by elimination. The four question types:
  - **Who V-ed someone?** asks for the agent; active question.
  - **Who did someone V?** asks for the patient; active question with an object wh-phrase.
  - **By whom was someone V-ed?** asks for the agent; passive question. In v1.3.0 this replaced the stranded "Who was someone V-ed by?", which even Gemma misreads (0.36).
  - **Who was V-ed by someone?** asks for the patient; passive question.
- **Prompt:** `Sentence: {s}\nQuestion: {q}\nAnswer:`. Instruct models get it inside their chat template.
- **Scoring:** we compare the log-probability of the correct name against the other name (candidate-restricted forced choice).
- **Stimuli:** generator v1.3.0. 24 names × 16 verbs and 276 name pairs give 30,912 items. The split is by name-pair family: 15% dev (4,592 items) and four test sets.

## 2. What we found (dev split, exploratory)

### Novelty in one list (what the paper can claim if the test set confirms it)

1. **The failure.** LLMs, including 32B models, fail object wh-questions about simple transitive sentences and answer with the subject. Reformulating the question as a cloze fixes it, so the knowledge is present but retrieval fails. This is a parallel to the human subject/object wh-asymmetry.
2. **The cause.** The failure sits in a **question-side state** at "?" or ":". The **same state copied from a different sentence** repairs 21–50% of natural errors, against 1–7% for controls, in 8 models from 3 families.
3. **The design.** Order, grammatical function and thematic role are fully decorrelated. Questions never name a participant, and transplants use foreign names.
4. **Role storage.** A strong role code on names is neither necessary nor sufficient. Qwen2.5 has the biggest code but fails; OLMo succeeds with a tiny one.
5. **The generation shift.** The newer, fixed generation (Qwen3.8) reads the question earlier, at "?", like Gemma. This is a hypothesis.

What is not new: question-time retrieval at ":" itself. See the framing constraint below.

### Headline storyline (candidate paper)

1. **Object wh-questions make models answer with the subject.**
   - **Who fails.** Every Qwen2.5-Instruct model from 0.5B to 32B fails "Who did someone V?", as do OLMo-2 1B and 7B.
   - **What the errors look like.** The model picks the **grammatical subject of the sentence**.
   - **Not just wording.** The failure survives every rewording ("Whom…", "Which person…", cleft).
   - **Cloze removes it.** Asking the same thing as a cloze ("The one someone praised was") gives near-perfect answers, e.g. Qwen2.5-14B on actives goes from 0.39 to 0.98.
   - **So the model knows the roles.** It fails to *retrieve* them when the question is an object wh-question. This parallels the human subject/object wh-asymmetry seen in children, L2 learners and aphasia, and Bader & Meng's retrieval account of misinterpretation.
2. **A transferable question-side state causes it, and patching that state fixes it.**
   - **Where the state is.** OTR-04 finds the step at the "?" or at the ":" of "Answer:".
   - **The repair.** Copying it from a *different sentence with disjoint names* (repair_x) fixes 21–50% of natural errors in every Instruct model. The control (period position) fixes only 1–7%.
   - **Selection check.** A half-split check confirms this is not a cherry-picked position or layer.
3. **The order of processing is the same in every Instruct model.** Role information peaks on the name tokens first, or at the same layer. Then the question state is strongest. Then the answer forms at the last position. Details and the exceptions are in "Layer investigation" below. OTR-02 found no answer information at the verb, *was*, *by* or the period in any model.
4. **Generation shift (new, 16 Nov).**
   - **Where the question is read.** Qwen2.5 reads it late, at ":" (67–73% depth). Qwen3.8 reads it at "?" (54%), like Gemma (57%). These two are the only models without the object-question error.
   - **Status: hypothesis.** OLMo-13B reads at ":" but early (45%) and is fairly accurate (0.89), so the real variable may be *how early* the question is bound rather than *which token* carries it. More models are needed.
5. **Different families store roles differently.**
   - **Qwen2.5:** the portable role code on name tokens grows from 2% to 11% of identity with scale, yet accuracy stays flat.
   - **OLMo:** succeeds with about 2%.
   - **Interpretation.** A role code on the names is neither necessary nor sufficient. Hypothesis: OLMo stores role relationally.
   - **OLMo-32B (OTR-02):** leans on the second name, 0.80 vs 0.22.

**What is NOT new (framing constraint).** Question-time retrieval with readout at the final ":" is known. Relevant work:
- "Lookback" mechanisms (Prakash et al., ICLR 2026).
- A 2026 entity-tracking paper.
- Rebinding circuits (2026).

Present our ":"/"?" state as **an instance of that known retrieval mechanism, which object/oblique wh-questions break**. Do not claim to have discovered question-time retrieval. Our novelty:
- the linguistic failure itself, with the decorrelated order/function/role design;
- causal repair of *natural errors* using cross-sentence states;
- replication across families and scales;
- the Qwen2.5→3.8 shift;
- the human parallel.

### Gate (OTR-01, ≥0.90 in every construction × role cell)

Only **Gemma 4 31B-it passes (0.98)**, out of 16 models.

| Model | Overall | Who V-ed someone? | Who did someone V? | By whom was someone V-ed? | Who was V-ed by someone? | "Who did someone V?": active / passive / obj-cleft / by-cleft |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.50 | 0.38 | 0.67 | 0.32 | 0.64 | 0.00 / 1.00 / 0.66 / 1.00 |
| Qwen2.5-1.5B-Instruct | 0.75 | 0.82 | 0.52 | 0.67 | 0.60 | 0.48 / 0.53 / 0.07 / 0.99 |
| Qwen2.5-3B-Instruct | 0.59 | 0.61 | 0.51 | 0.55 | 0.54 | 0.00 / 0.96 / 0.09 / 1.00 |
| Qwen2.5-7B-Instruct | 0.70 | 0.88 | 0.44 | 0.50 | 0.74 | 0.09 / 0.55 / 0.17 / 0.94 |
| Qwen2.5-14B-Instruct | 0.76 | 0.95 | 0.57 | 0.44 | 0.90 | 0.39 / 0.63 / 0.29 / 0.98 |
| Qwen2.5-32B-Instruct | 0.79 | 0.91 | 0.40 | 0.78 | 0.90 | 0.18 / 0.39 / 0.02 / 0.99 |
| Qwen3.8-27B (no thinking) | 0.90 | 0.85 | 0.95 | 0.74 | 0.98 | 1.00 / 0.99 / 0.90 / 0.90 |
| OLMo-2-1B-Instruct | 0.57 | 0.69 | 0.48 | 0.65 | 0.47 | 0.09 / 0.72 / 0.78 / 0.32 |
| OLMo-2-7B-Instruct | 0.65 | 0.75 | 0.56 | 0.65 | 0.59 | 0.03 / 0.91 / 0.76 / 0.53 |
| OLMo-2-13B-Instruct | 0.89 | 0.87 | 0.86 | 0.80 | 0.99 | 0.67 / 0.91 / 0.96 / 0.91 |
| OLMo-2-32B-Instruct | 0.85 | 0.89 | 0.68 | 0.70 | 0.84 | 0.82 / 0.57 / 0.36 / 0.98 |
| Muse Glimmer 30B * | 0.76 | 0.83 | 0.78 | 0.45 | 0.91 | 0.18 / 1.00 / 0.93 / 0.99 |
| Mistral Small 3.2 24B * | 0.88 | 0.97 | 0.62 | 0.85 | 0.90 | 0.94 / 0.18 / 0.39 / 0.99 |
| Qwen2.5-7B base | 0.69 | 0.56 | 0.66 | 0.53 | 0.88 | 0.34 / 0.74 / 0.93 / 0.63 |
| Qwen2.5-1.5B base | 0.59 | 0.55 | 0.56 | 0.57 | 0.55 | 0.50 / 0.70 / 0.23 / 0.82 |
| **Gemma 4 31B-it** | **0.98** | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |

\* The forced-choice scores for these two may not fit how they answer: a name was the top-1 first token only 48% (Glimmer) and 0% (Mistral) of the time.
- As scored, Mistral answers "Who did someone V?" with the last-mentioned name.
- Glimmer fails actives even in the **cloze** version (0.03, see below), which suggests a scoring problem rather than syntax.
- Validate both with the generation check before interpreting either.

### Robustness (OTR-05, accuracy per wording variant)

| Model | Who did someone V? (orig) | Whom | Which person | Cleft | **Cloze (active)** | Patient Q, passive voice | By whom | Who was … by (stranded) | Agent cloze |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-14B-I | 0.57 | 0.64 | 0.59 | 0.81 | **0.99** | 0.90 | 0.44 | 0.36 | 0.88 |
| Qwen2.5-7B-I | 0.44 | 0.51 | 0.38 | 0.55 | 0.78 | 0.73 | 0.50 | 0.43 | 0.68 |
| Qwen2.5-7B base | 0.65 | 0.76 | 0.58 | 0.53 | 0.88 | 0.87 | 0.53 | 0.45 | 0.64 |
| Qwen3.8-27B | 0.95 | 0.94 | 0.94 | 0.93 | 0.91 | 0.98 | 0.74 | 0.48 | 0.84 |
| OLMo-2-32B-I | 0.68 | 0.68 | 0.65 | 0.86 | 0.88 | 0.84 | 0.70 | 0.40 | 0.97 |
| OLMo-2-13B-I | 0.86 | 0.92 | 0.88 | 0.91 | 1.00 | 0.99 | 0.80 | 0.49 | 0.86 |
| Gemma 4 31B-it | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.36 | 1.00 |
| Glimmer 30B * | 0.78 | 0.78 | 0.86 | 0.79 | 0.74 | 0.91 | 0.45 | 0.29 | 0.79 |
| Mistral 3.2 24B * | 0.62 | 0.88 | 0.80 | 0.95 | 0.88 | 0.90 | 0.85 | 0.23 | 0.92 |

"Who did someone V?" on active sentences only, original wording vs the active cloze:

| Model | Original | Cloze |
|---|---|---|
| Qwen-14B-I | 0.39 | 0.98 |
| Qwen-7B-I | 0.09 | 0.67 |
| OLMo-32B-I | 0.82 | 0.99 |
| OLMo-13B-I | 0.66 | 1.00 |
| Qwen-7B base | 0.33 | 0.99 |
| Glimmer | 0.18 | 0.03 |

The stranded "Who was someone V-ed by?" fails in every model, Gemma included.

### Internals (OTR-03 role code; OTR-04 question side)

| Model | Role peak @ layer (depth) | % of identity | Question-side best (flip) | repair_x vs control_x |
|---|---|---|---|---|
| Qwen2.5-7B base | 0.06 @ L21/28 | 0.6% | "?" 0.13 @ L15 | 11% vs 0% |
| Qwen2.5-1.5B-I | 0.25 @ L15/28 | 2.1% | "?" 0.40 @ L15 | 25% vs 1% |
| Qwen2.5-7B-I | 0.65 @ L15/28 | 4.2% | ":" 0.38 @ L20 (71%) | 45% vs 4% |
| Qwen2.5-14B-I | 2.07 @ L22/48 | 9.9% | ":" 0.37 @ L32 (67%) | 44% vs 3% |
| Qwen2.5-32B-I | 2.32 @ L38/64 | 11.2% | ":" 0.54 @ L47 (73%) | 49% vs 3% |
| Qwen3.8-27B | 0.58 [0.54, 0.62] @ L29/64 (45%) | 4.9% | "?" 0.47 @ L35 (54%) | 50% vs 3% ("?" L39) |
| OLMo-2-7B-I | 0.10 @ L8/32 | 0.8% | "?" 0.40 @ L14 (44%) | 21% vs 1% |
| OLMo-2-13B-I | 0.25 @ L13/40 | 2.0% | ":" 0.30 @ L18 (45%) | 36% vs 7% |
| OLMo-2-32B-I | 0.18 @ L33/64 | 1.6% | ":" 0.57 @ L44 (69%) | 46% vs 7% |
| Gemma 4 31B-it | 1.36 / 0.75 @ L28/60 | 6.2% / 3.5% | "?" 0.54 @ L34 (57%) | too few errors |

Notes:
- **Qwen3.8 OTR-03 details:**
  - Order −0.19, function ≤0.09, source-slot nuisance −0.19, identity 11.8, validity check 0.
  - Role code by question type: Who V-ed 0.46 / Who did someone V? 0.65 / By whom 0.32 / Who was V-ed 0.88. The weakest, "By whom", is also its weakest question behaviourally.
  - Same-sentence repair at "?" L39: 60% of 107 errors.
- **Half-split check:** choose position and layer on half the name families, test on the other half, over 20 splits. Held-out repair is 0.09–0.49 vs control 0.01–0.07. Shrinkage ≤0.08 and the picks are stable.
- **OTR-02 (Qwen 1.5B/7B base, Qwen-14B-I, OLMo-13B/32B-I, Gemma):** no answer information at verb, aux, *by* or period.
- **Figures:** see §2b.

### 2c. Generation check: does forced choice fit each model? (round 1, 16 Nov)

Each model greedy-generated 12 tokens for 256 balanced dev prompts (16 per construction × role × question-voice cell). Answers are classified by `extract_answer`, which ignores quoted text and copies of the sentence or question:
- **direct**: a bare name;
- **inferred**: exactly one name left;
- **ambiguous**: both names left;
- **none**: no name.

| Model | Direct | Inferred | Ambiguous / none | Generation accuracy | Forced choice (OTR-01) | "Who did someone V?" among decided answers |
|---|---|---|---|---|---|---|
| Gemma 4 31B-it | 100% | – | – | 1.00 | 0.98 | 64/64 |
| Qwen3.8-27B (no thinking) | 100% | – | – | 0.87 | 0.90 | active 16/16, passive 15/16, obj-cleft 16/16, by-cleft 13/16 |
| Qwen2.5-14B-Instruct | 0% | 35% | 39% / 27% | 0.27 (truncated) | 0.76 | too few decided (16) |
| Muse Glimmer 30B | 3% | 69% | 14% / 14% | 0.70 (truncated) | 0.76 | 51/51 correct |
| Mistral Small 3.2 24B | 0% | 39% | 14% / 47% | 0.38 (truncated) | 0.88 | 12 decided |

What this means:
- Forced choice is **validated for Gemma and Qwen3.8**.
- For **Qwen2.5-Instruct** (and probably OLMo-Instruct, not yet checked), forced choice scores a continuation the model would not produce on its own. Two things support the failure being real:
  - the base model Qwen2.5-7B also fails ("Who did someone V?" on actives 0.33), and for a base model a name after "Answer:" is the natural continuation;
  - the cloze version is near-perfect under the same scoring.

  But **reviewers will ask**, so round 2 must show the failure in generated answers. If it does not, the paper's behavioural claim moves to base models plus instructed generation.
- **Glimmer:** its forced-choice failure on object questions looks like a scoring artefact. Its generated answers are right.

### 2a. Layer investigation (OTR-02/03/04 by depth; `layer_summary.csv`)

Depth is layer / n_layers. The role peak comes from OTR-03. The question peak is the larger of the OTR-04 flip effects at "?" and at ":".

| Model | Role peak depth (% of identity) | "?" peak (depth) | ":" peak (depth) | Strongest question state |
|---|---|---|---|---|
| Qwen2.5-7B base | 75% (0.6%) | 0.13 (54%) | final token, not usable | "?" 54% |
| Qwen2.5-1.5B-I | 54% (2.2%) | 0.40 (54%) | 0.12 (64%) | "?" 54% |
| Qwen2.5-7B-I | 54% (4.2%) | 0.23 (54%) | 0.38 (71%) | ":" 71% |
| Qwen2.5-14B-I | 46% (10.1%) | 0.22 (54%) | 0.37 (67%) | ":" 67% |
| Qwen2.5-32B-I | 59% (11.3%) | 0.12 (63%) | 0.54 (73%) | ":" 73% |
| OLMo-2-7B-I | 25% (0.8%) | 0.40 (44%) | 0.24 (59%) | "?" 44% |
| OLMo-2-13B-I | 33% (2.0%) | 0.25 (38%) | 0.30 (45%) | ":" 45% |
| OLMo-2-32B-I | 52% (1.6%) | 0.14 (52%) | 0.57 (69%) | ":" 69% |
| Qwen3.8-27B | 45% (5.1%) | 0.47 (55%) | 0.21 (69%) | "?" 55% |
| Gemma 4 31B-it | 47% (5.9%) | 0.54 (57%) | 0.14 (65%) | "?" 57% |

What the layer curves show (dev, exploratory):
- **The sequence.** In all 9 Instruct models, the role code on names peaks at or before the strongest question state.
  - It ties with the "?" peak in Qwen2.5-1.5B-I and 7B-I, and in OLMo-32B.
  - It is clearly earlier in the others.
  - **Exception:** Qwen2.5-7B base. Its role code is tiny (0.6%) and peaks late (75%), after its weak "?" state (54%). Don't claim the sequence for base models.
- **Two stages in the question.** Where both tokens carry the question, "?" rises first and ":" takes over later: the state moves from the question mark to the answer cue. Models differ in which stage is strongest:
  - Qwen2.5-7B-I and larger, and OLMo-13B/32B: ":" dominates, late.
  - Qwen3.8, Gemma, OLMo-7B and Qwen2.5-1.5B-I: "?" dominates.
- **Control.** The sentence period stays at ≈0 at every layer in every model (max 0.003). The question-side effect is not generic patching.
- **Gemma and Qwen3.8 look alike.**
  - Their role code appears at 40–47% depth, about 5–6% of identity.
  - Their "?" peak is sharp at 55–57%.
  - Their ":" effect is small.
  - Qwen2.5 has a larger role code but reads the question late at ":".
- **Related figures.**
  - The OTR-02 heat maps (per-run PNGs) back the "no answer information at verb / *was* / *by* / period" claim.
  - OLMo-32B's reliance on the second name is visible in its OTR-02 map.

### 2b. Figures

**Cross-model summaries** are in `docs/order_to_role/figures/` and are regenerated by `python make_cross_model_figures.py`. The numbers are hard-coded from the tables above; update them when new models arrive.

- **`cross_model_otr.png`**
  - Panel A: role code on names vs accuracy. A strong role code doesn't predict success.
  - Panel B: cross-sentence repair vs control, for each model.
- **`layer_role_code.png`**: OTR-03 role code (% of identity, 95% CI) by relative depth. Three panels: Qwen2.5 sizes, OLMo-2 sizes, and Qwen3.8 + Gemma. Made by `python make_layer_figures.py <results-dir>`, which reads the JSONs directly and also writes `layer_summary.csv`.
- **`layer_question_side.png`**: OTR-04 flip by relative depth, one panel per model, with "?" (blue), ":" (orange), the period control (gray) and the OTR-03 role peak (dotted). This is the main layer figure.
- **`question_binding_depth.png`**: the depth where the question-side state is strongest vs accuracy on "Who did someone V?". Filled markers read the question at "?"; hollow markers read it at ":". This is the Qwen2.5→3.8 / Gemma hypothesis figure.

**Per-run figures** are on branch `order-to-role/results-dev`, next to each JSON:
- `results/otr02_*.png`: position × layer heat maps. 6 models.
- `results/otr03_*.png`: order / function / role code by layer. 10 models; Qwen2.5-7B and Gemma have extra earlier runs.
- `results/otr04_*.png`: question-side flip and repair curves by layer. 10 models.

OTR-01 and OTR-05 have JSON only. The tables in §2 are made from those files.

**Not yet made (paper figures):**
- OTR-05 bar chart: original wording vs cloze, per model.
- An example-sentence schematic of the design.

### Confounds already caught (do not reintroduce)

1. A single foreign pair per base tied "source is agent" to one name. Fixed by counterbalancing both assignments.
2. A pooled regression let "this slot holds the answer" (a big identity effect) look like role. Fixed by estimating within (base sentence, slot), plus a `source_slot` nuisance term.
3. A flip with a tie at 0 counted as a flip. `_crossed` now requires a strict crossing.
4. The stranded passive question was a stimulus artefact. Fixed in v1.3.0.
5. Forced choice can misfit chat models. Check the top-1 name rate, and use the generation check.

## 3. Literature check (15 Nov; quick, about 7 searches; redo properly for the paper)

- **Not scooped.** Closest: Denning, Guo, Snefjella & Blank 2025, "Do LLMs know who did what to whom?". It uses probing/RSA, with no patching and no QA.
- **Filler–gap interpretability:**
  - Boguraev et al., EMNLP 2025: DAS on Pythia, gap prediction, object extraction only. No thematic roles, no subject/object asymmetry, no QA.
  - A 2026 follow-up: a shared filler–gap mechanism in early–middle layers.
- **Retrieval:** lookback (Prakash et al., ICLR 2026; readout at ":"), entity tracking 2026, rebinding 2026. See the framing constraint above.
- **Behavioural:**
  - Emura & Sugawara, ACL 2026: plausibility vs grammar under dual-task load.
  - A 12-language yes/no comprehension benchmark (arXiv 2602.20065; 40 items, adapted by native speakers; covers Russian, Turkish, Chinese and Japanese). Possible outside check.
- **Human:** subject–object wh asymmetry (children, L2, aphasia); Ferreira 2003 (misinterpretation of noncanonical sentences); Bader & Meng (misinterpretations as retrieval difficulties).
- **Datasets:** none fits our design.
  - MultiBLiMP covers agreement only.
  - Universal PropBank 2.0, CoNLL-09 and X-SRL are natural, uncontrolled SRL text.
  - So we build our own templates, including for the multilingual extension.

## 4. Code map (repo `Joel-Vazquez-Lopez/llm-relational-mechanisms`, branch `order-to-role/stimuli`)

| File | Role |
|---|---|
| `src/relational_lab/experiments/order_role_stimuli.py` | Generator v1.3.0. Items, spans, family split (`split_families`, `dev_fraction=0.15`, `n_test_sets=4`), `role_swapped`. |
| `src/relational_lab/experiments/order_role_behavior.py` | OTR-01 summaries: cells, bootstrap CIs, gate. |
| `src/relational_lab/experiments/order_role_patching.py` | OTR-02 sweep and aggregation. Has a `chat_template` parameter. |
| `src/relational_lab/experiments/order_role_transplant.py` | OTR-03 counterbalanced foreign-name transplants. `fit_profile` estimates within (base, slot) with `NUISANCE=("source_slot",)`; also `left_context_null`. |
| `src/relational_lab/experiments/order_role_question.py` | OTR-04. `MODES=("flip","repair")`, `CROSS_MODES=("repair_x","control_x","flip_x")`; `foreign_names`, `partner`, `shared_tail`, `sweep`, `aggregate`. |
| `src/relational_lab/experiments/order_role_robustness.py` | OTR-05. `ROBUSTNESS_VERSION="1.0.0"`; `VARIANTS` (A_orig_active, A_which, A_cleft, A_orig_passive, A_who_by, A_cloze, P_orig_active, P_whom, P_which, P_cleft, P_orig_passive, P_cloze_active, P_cloze_passive); `render`, `make_items`, `summarize`. |
| `src/relational_lab/instrumentation/hf_scoring.py` | `HFScorer`, `load_tokenizer` (Mistral regex fix), `load_lm` (multimodal fallbacks), `check_names_for_patching`. |
| `src/relational_lab/instrumentation/patching.py` | Hooks; `chat_text` (template plus assistant prefix, with offset), `encode_batch(..., chat_template)`, `answer_token(..., leading_space)`. |
| `src/relational_lab/instrumentation/chat.py` | `configure(no_thinking, template_kwargs, assistant_prefix)`; global template kwargs and assistant prefix, recorded in result JSON. |
| `scripts/otr01_behavior_gate.py` … `otr05_robustness.py` | Runners. Shared flags: `--chat-template --no-thinking --template-kwarg K=V --assistant-prefix S`. Each writes `results/otr0X_<org>__<model>_<split>_<stamp>.json` with hashes and versions. Only OTR-01 has `--test-set`. |
| `scripts/otr_generate_check.py` (patches 0021, 0025) | Greedy generation (default 40 tokens) on a balanced dev sample (`--n 256`). `extract_answer` classifies each answer as direct / inferred / ambiguous / none; `--instruction` adds a line before "Answer:". Reports answer kinds, generation accuracy, direct-only accuracy, per-cell accuracy and examples. Writes `results/otrgen_<model>_dev_<stamp>.json`. |
| `scripts/show_chat_prompt.py` | Prints the exact templated prompt for a model, to verify chat settings. |
| `scripts/slurm/` | `setup_pelle.sh`, `otr01–05_pelle.sbatch`, `otrgen_pelle.sbatch`, `otr_download.sbatch`. GPU jobs set `HF_HUB_OFFLINE=1`, so download first. |
| `tests/` | 96 tests pass: exact patching identities, planted-code recovery, decorrelation, counterbalancing, cross modes, robustness variants. |

**Chat settings per model** (verified with `show_chat_prompt.py`):

| Model | Flags |
|---|---|
| Instruct models (Qwen2.5, OLMo-2, Gemma) | `--chat-template` |
| Base models | no flag |
| `Qwen/Qwen3.8-27B` | `--chat-template --no-thinking` (gives `<think>\n\n</think>\n\n`) |
| `meta-models/Muse-Glimmer-30B` | `--chat-template --template-kwarg reasoning_strength=low --assistant-prefix ' to=user<|message|>'` (gives `<|start|>assistant to=user<|message|>Leon<|eot|>`) |
| `unsloth/Mistral-Small-3.2-24B-Instruct-2506` | `--chat-template` (the official mistralai repo has no HF chat template) |

**Patches:**
- 0001–0020: pushed and on Pelle.
- 0021: generation check, written to the Mac, not applied.
- 0022: this README.
- 0023: novelty list, figures section and `docs/order_to_role/figures/`.
- 0024: layer investigation section, layer figures and `make_layer_figures.py`.
- 0025: `extract_answer` (direct / inferred / ambiguous / none), `--instruction`, default 40 tokens, tests, round-1 findings.
- A new chat whose sessions *can* push (check whether the Claude GitHub App is linked now) can push directly instead.

**Results:** branch `order-to-role/results-dev`. Files are added with `git add -f`, because `results/` is gitignored. Existing OTR results:
- OTR-01: 21 files
- OTR-02: 6
- OTR-03: 13
- OTR-04: 12
- OTR-05: 9

## 5. Compute (UPPMAX Pelle)

- **Project:** always `-A uppmax2026-1-225` (course 5LN714, 4,000 GiB). **Never** use `uppmax2026-1-123`: it is nearly full and belongs to a different project.
- **Paths:**
  - Code: `~/private/llm_relation`, on branch `order-to-role/stimuli`.
  - Results worktree: `~/private/results-wt`, on branch `order-to-role/results-dev`.
  - Venv: `/proj/uppmax2026-1-225/venvs/order-role`.
  - HF cache: `/proj/uppmax2026-1-225/hf_cache`.
- **Session start:**
  ```
  cd ~/private/llm_relation && source /proj/uppmax2026-1-225/venvs/order-role/bin/activate && export HF_HOME=/proj/uppmax2026-1-225/hf_cache
  ```
- **GPUs:**
  - L40s (48 GB) for models up to about 14B: the default in the sbatch files.
  - H100 (94 GB, 4 total) for models around 24–32B: add `--gpus=h100:1`.
- **Downloads run as batch jobs**, not on the login node; compute nodes have internet:
  ```
  DL=$(sbatch --parsable -A uppmax2026-1-225 scripts/slurm/otr_download.sbatch <model>); DL=${DL%%;*}
  sbatch -A uppmax2026-1-225 --dependency=afterany:$DL --gpus=h100:1 scripts/slurm/otr01_pelle.sbatch <model> --chat-template
  ```
- **Storage:** about 685 GiB used. One big model at a time. Check with `uquota`, which can lag after deletions. Delete only with `huggingface_hub.scan_cache_dir().delete_revisions(...)`, never `rm -rf`: the cache uses a shared blob store.
- **Cached models:**
  - gemma-4-31B-it
  - OLMo-2-0325-32B-Instruct
  - OLMo-2-1124-7B/13B-Instruct
  - Qwen2.5-1.5B/7B/14B-Instruct
  - Qwen2.5-7B
  - Qwen3.8-27B
  - Muse-Glimmer-30B
  - unsloth Mistral-Small-3.2

  Some small ladder models may already be deleted.
- **Other users' folders:** `hgraw`, `yanit` and `rata2388` in the project belong to other users. Never touch them.

## 6. Rules

- Follow the evidence policy in `EVIDENCE_POLICY.md`:
  - Freeze the protocol (claims, frozen positions and layers per model, metrics, code hashes) before any confirmatory run.
  - Open each test set **once**.
  - Keep negative results.
  - Keep candidate-restricted, full-vocabulary and generated answers separate.
  - Label results as exploratory or confirmatory.
- **Never modify frozen historical files**: `scripts/ml100_source_access_factorial.py` and the other ML-0xx frozen scripts.
- **Never `git checkout` in `~/private/llm_relation` while jobs are running.** Push results through `~/private/results-wt`.
- **Berget AI:** Joel has an account. **Never read his key file.** The service has no logprobs, so it supports generation only.
- **Naming:** the study prefix is OTR-. The local EXP-101/EXP-102 are separate studies.
- **Joel's other deadline:** the manuscript *Targeted Experience, Relational Generalization, and Role-Assignment Errors in Transformers* is due **15 Dec 2026**. Keep it on track.

## 7. Workflow and exact commands

**Mac** (clone `~/Research/llm-relational-mechanisms`, branch `order-to-role/stimuli`). Apply each patch once. If `git am` complains, run `git am --abort` and check `git log --oneline -3`.
```
cd ~/Research/llm-relational-mechanisms
git am order-role-patches/0021-Generation-check-for-chat-models-scripts-otr_generat.patch
git am order-role-patches/0022-Add-order-to-role-handoff-README.patch
git am order-role-patches/0023-Handoff-novelty-list-and-cross-model-figures.patch
git am order-role-patches/0024-Handoff-layer-investigation-section-and-layer-figure.patch
git am order-role-patches/0025-Generation-check-answer-extraction-instruction-flag.patch
git push origin order-to-role/stimuli
```

**Pelle: pull and run the generation check**
```
cd ~/private/llm_relation
git pull
source /proj/uppmax2026-1-225/venvs/order-role/bin/activate
export HF_HOME=/proj/uppmax2026-1-225/hf_cache
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch unsloth/Mistral-Small-3.2-24B-Instruct-2506 --chat-template
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch meta-models/Muse-Glimmer-30B --chat-template --template-kwarg reasoning_strength=low --assistant-prefix ' to=user<|message|>'
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch Qwen/Qwen3.8-27B --chat-template --no-thinking
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch google/gemma-4-31B-it --chat-template
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch Qwen/Qwen2.5-14B-Instruct --chat-template
```

**Pelle: generation check, round 2** (after `git pull`; Glimmer's flags are as in the chat-settings table):
```
for M in Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-13B-Instruct allenai/OLMo-2-0325-32B-Instruct unsloth/Mistral-Small-3.2-24B-Instruct-2506; do sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch $M --chat-template; sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch $M --chat-template --instruction "Reply with one name only."; done
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch meta-models/Muse-Glimmer-30B --chat-template --template-kwarg reasoning_strength=low --assistant-prefix ' to=user<|message|>'
sbatch -A uppmax2026-1-225 --gpus=h100:1 scripts/slurm/otrgen_pelle.sbatch meta-models/Muse-Glimmer-30B --chat-template --template-kwarg reasoning_strength=low --assistant-prefix ' to=user<|message|>' --instruction "Reply with one name only."
```

**Pelle: send results up** (the worktree avoids checkouts in the main clone):
```
cp ~/private/llm_relation/results/otrgen_*.json ~/private/results-wt/results/
cd ~/private/results-wt
git add -f results/otrgen_*.json
git commit -m "Generation check round 2"
git push origin order-to-role/results-dev
```

**Claude side:**
- Clone or pull `order-to-role/results-dev` to analyse.
- Write patches through the desktop bridge (`device_commit_files`) into `~/Research/llm-relational-mechanisms/order-role-patches/`, staged from `/mnt/user-data/outputs/`.
- Use the patch's real filename: `format-patch` truncates long subjects.

## 8. Open questions and hypotheses

- **Question binding.** Is early binding the real variable (Qwen3.8/Gemma at "?" around 55%; OLMo-13B at ":" around 45%)? Testing it needs more models that read the question in different places.
- **OLMo.** Is its role information relational, i.e. stored in the name–verb or name–name relation rather than on one name token? Test: a same-frame transplant (move both names together).
- **Chat models' first token.** Do Glimmer and Mistral really fail, or does the scoring misfit them? The generation check answers this.
- **Movement.** Does the failure need wh-*movement*? Test with English in-situ ("Someone praised who?") and yes/no ("Did Leon praise Mira?") variants in OTR-05. This is cheap and sets up the multilingual extension.
- **Question patching on cloze prompts.** Does OTR-04 on cloze prompts show no broken state?

## 9. Next steps (in order)

1. **Generation check, round 2** (§0, §2c). This decides whether the Qwen2.5/OLMo behavioural failure holds in generated answers, and whether Glimmer and Mistral stay in the paper. If the instruction gives bare names, consider adding it as an OTR-05 forced-choice variant as well.
2. **Add in-situ and yes/no variants to OTR-05** (English movement test). Run them on the cached models.
3. **Add `--test-set N` to OTR-02/03/04/05 and the generation check**, mirroring `select_items` in `otr01_behavior_gate.py`. Add tests that fail if more than one test set is selected.
4. **Freeze document** (`docs/order_to_role/FREEZE.md`):
   - claims C1–C5 from §2;
   - per-model frozen position and layer for repair_x/control_x and the OTR-03 peak layer, all chosen on dev;
   - metrics and thresholds;
   - generator, robustness and script hashes;
   - model revisions.

   Commit it, then **open test set 0 once** for the core claims. Keep test sets 1–3 for extensions such as new models and new variants.
5. **Second passing model.** Candidates:
   - Llama-3.3-70B: needs an HF token, an access request and multi-GPU loading.
   - Gemma 4 26B-A4B.
   - Granite 4.2 30B.
   - Qwen3-32B.
6. **Multilingual extension** (after the English freeze), with our own templates and native-speaker checks:
   - Mandarin: in-situ wh, no case.
   - Russian: fronted wh, case marking, free word order including OVS actives. A strong test of order vs function vs role.
   - Japanese or Turkish: in-situ wh, case marking.
   - Spanish: optional.
7. **Paper:** related-work note from §3, figures (cross-model panel, OTR-04 layer curves, OTR-05 bars), and the human-parallel discussion.

## 10. Project docs (claude.ai project "LLM Linguistics")

- `claude/order-to-role-status-2026-11-13.md`: running status note.
- `claude/order-to-role-context-package.md` and the "current state" doc: older (14 Nov) and superseded by this file.
- Plan doc: https://claude.ai/artifact/CuGEPvwkbgrP3ZaiqtsoPB
