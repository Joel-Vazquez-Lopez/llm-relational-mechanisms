[← Order-to-Role](README.md) · [Results](RESULTS.md) · [Experiment map](EXPERIMENTS.md)

# Methods

The study combines a linguistic design that separates three properties with causal interventions inside the models. This page explains both. Authoritative definitions are in the [English freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) (section 2) and the [Spanish/Galician addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md); where this page and the freeze differ, the freeze governs.

## 1. Linguistic design

Three properties usually travel together: **word order** (first or second name), **grammatical function** (subject or object) and **thematic role** (agent or patient; experiencer or stimulus for psych verbs). Four constructions make them exactly uncorrelated (Leon = agent):

| Construction | Example | First name | Subject | Agent |
|:--|:--|:--|:--|:--|
| Active | Leon praised Mira. | Leon | Leon | Leon |
| Passive | Mira was praised by Leon. | Mira | Mira | Leon |
| Object cleft | It was Mira that Leon praised. | Mira | Leon | Leon |
| By-cleft | It was by Leon that Mira was praised. | Leon | Mira | Leon |

A subject cleft serves as a control, and two relative-clause constructions are held out. Design choices that keep the test clean:

- **Reversible events, no plausibility cue.** Both names are people and either could be the agent.
- **Questions never name a participant** ("Who praised someone?"), so the answer cannot be found by elimination.
- **Focus caveat.** In the clefts the focused name comes first, so "order" in the main constructions means order-or-focus.

**The minimal pair.** Two questions ask for the same participant: the working "Who was the V-ing done by?" and the failing "Who was someone V-ed by?". The sentence is identical, so the pair isolates what the question does.

## 2. Stimulus sets

| Set | Generator | What it adds | Role in the evidence |
|:--|:--|:--|:--|
| Core English | `order_role_stimuli.py` v1.3.0 | 24 names × 16 transitive verbs (276 unordered name pairs), 7 constructions × 4 question types | Development and test0 |
| BX (broad English) | `broad_stimuli.py` | New names and verbs, 18 transitive constructions, ditransitives, psych verbs | Development; transitive block also a frozen test0 replication |
| Spanish, Galician (XL) | `multilingual_stimuli.py` v1.0.0 | Same name pairs and templates in two Romance languages | Development and a separate test0 addendum |
| Universal Dependencies | `ud_stimuli.py` | 188 EWT/GUM sentences with two human participants, names substituted | Validation only |
| WIKI-ROLES | `wiki_roles.py` v0.2 | Wikipedia sentences with two human participants (automatic parse plus two NER systems), real names or neutral replacements | Validation only |
| Swedish | `swedish_stimuli.py` | Additional language | Excluded until native-speaker check |

Eleven of the 48 BX names were dropped for non-Gemma tokenizers because they are not single tokens there (recorded in the result files).

## 3. Models, prompting and behavioural scoring

- **Models.** Gemma 4 31B-it, Qwen3.8-27B (thinking off), phi-4 (14B), OLMo-3.1-32B-Instruct, Granite-4.2-30B (thinking off). Hugging Face revisions are pinned in FREEZE.md §4. All runs use bfloat16 on one H100.
- **Inclusion gate (fixed on development data).** At least .80 on ordinary questions and .90 on the working question. gpt-oss-20b missed it (about .76 / .88) and is reported only descriptively. Earlier candidates (Qwen2.5 1.5B/7B base and instruct) failed the original behavioural gate.
- **Prompt.** `Sentence: … / Question: … / Answer:` inside each model's chat template, with the instruction "Reply with one name only." (Spanish "Responde con un solo nombre.", Galician "Responde cun só nome.").
- **Behavioural score.** Greedy generation, then a rule-based extractor that credits the gold name in a bare answer, a frame or a full sentence. Ambiguous or empty replies count as wrong. First-token forced choice is not used for behavioural claims, because it misrepresented chat models ([Results §1](RESULTS.md#1-behaviour-one-question-wording-fails-and-scoring-matters)).

## 4. Name-state transplantation (OTR-03)

*Where are roles stored?*

1. Take a base sentence and a source sentence in another construction, with a different, counterbalanced name pair (each donor name appears in each role, so name identity cancels).
2. At one layer, replace the hidden state of one name in the base with the state of a name from the source.
3. Measure the change in that recipient name's answer margin over the other name.
4. Regress the change on the source name's order, function and role (uncorrelated by design) **within each base item and slot**. This gives one coefficient per property per layer, with family-bootstrap intervals.

**Validity check.** Active and passive sources share their left context up to the first name. In this check the role slope must be exactly zero, and it is in every model. Two earlier confounds were found and fixed: a single foreign pair per base tied the source role to one name (now counterbalanced), and a pooled regression let an identity effect pass as role (now estimated within base and slot).

## 5. Answer-position and head patching (OTR-09, OTR-12)

*How is the role read out?*

- **Residual repair.** At one layer, copy the residual stream at the answer position from the working run into the failing run. Repair = (patched − failing) / (working − failing) on the first-token logit margin, per item, on items whose working–failing gap exceeds 1.0 logit. The **handover layer** is the first layer where mean repair reaches .5.
- **Head repair.** Patch single attention-head outputs instead. Heads are ranked on part of the development families and evaluated on the rest. The top-20 set is compared with 20 random heads from the same layers (5 draws).
- **Controls.** These are head outputs from another item's failing run (`other_failing`), a random push of the same size (`random_nudge`), and, in the addendum, another item's working-run output (C5b).
- **Knockout and steering (OTR-12).** Mean-ablate the top-k heads at the answer position (each head's output is replaced by its mean over the fit items) against k random heads from the same layers, in the working and in an ordinary agent question. Separately, add α × the averaged working − failing residual direction at one layer, against a random direction of the same norm.

## 6. Question-token patching and attention blocking (OTR-13, OTR-14)

*Where does the failure enter?*

- **OTR-13.** Swap the state of a single question token ("by", the verb, "someone", "?") between the two questions. *Reverse damage* = harm to the working answer when the failing token is inserted; *repair* = the opposite direction.
- **OTR-14.** On a common per-item scale: swap the whole question from a layer upward, patch the answer position, block attention from the answer position to the question or to the names, and transfer "by" from the failing question or from a patient question. This gives the question-to-answer handover curves.

## 7. Answer process (OTR-07)

*Does the model know the agent when it answers wrong?* On items whose bare answer names the patient, the model continues a neutral prefix, "Someone was V-ed by". The prediction at the by-slot is compared for the agent and the patient. An earlier prefix that named the patient was replaced by this neutral one, because the patient's name in the prefix could do the work.

## 8. Splits, freeze and confirmatory testing

- **Families** are unordered name pairs. The partition is fixed: seed 2026, 15% development, four test sets. Test0 has 59 core families, disjoint from the 41 development families.
- **All exploration used development data only.** Natural-text sets have no test split.
- **Freeze.** On 26 November eight claims (C1–C8) were committed publicly, with their pass rules, every layer, the ordered top-20 heads and the scorer, before any test0 command ran ([`f6ea0e1`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/commit/f6ea0e1)). Each claim is tested per model with 2,000 family-bootstrap resamples (seed 2026), paired where two quantities are compared. Failed and not-evaluable cells are reported, and no multiplicity correction is applied.
- **Spanish/Galician addendum.** Frozen after English test0 was seen and before any Spanish or Galician test run ([`3811d7c`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/commit/3811d7c)). It sets its own handover layers and adds the C5b content control.
- **Disclosed deviations.** Development OTR-03 donor pairs were drawn from all name pairs, including test families; test runs draw donors from their own split only and abort on any leak. The test0 attempt log is in [`test0_attempts.md`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/test0_attempts.md).
