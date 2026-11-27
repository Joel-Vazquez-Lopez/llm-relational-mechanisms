# Thematic roles live on the names — paper draft (24 Nov 2026)

Snapshot of the working draft (the live version is a Claude Doc). Numbers are dev-split results unless
marked otherwise; claims tested on the held-out set are marked **[test0]**. Open questions are marked **Q:**.
Target: ACL Rolling Review, deadline 4 Jan 2027 (long paper, 8 pages + limitations + appendix).

## Abstract

Where does a language model keep track of who did what to whom, and how does a question retrieve it? We study this with sentences that separate three properties linguistic theory keeps apart: word order, grammatical function and thematic role. Transplanting the hidden states of single names between sentences, we find that in five open model families the names carry their thematic role in the middle layers, more strongly than their position or grammatical function. With psych verbs, the code follows the experiencer whether it is the subject or the object. A question reads these roles out through a small set of attention heads at a fixed depth: the same heads, frozen on template sentences, repair answers on new verbs and names, in Spanish and Galician, and on natural sentences from Universal Dependencies and Wikipedia, with real and neutral names alike. A minimal pair of questions exposes the separation between storage and retrieval. "Who was the chasing done by?" is answered correctly, while "Who was someone chased by?" often yields the wrong name, although the stored states are identical: the error enters at the question, mainly at "by", and patching the question state or the heads repairs it. All claims, layers and heads were fixed on development data before a held-out test set was opened **[test0]**.

## 1 Introduction

Understanding a sentence means knowing who did what to whom. Linguistic theory describes this with three properties that usually travel together: word order, grammatical function (subject, object) and thematic role (agent, patient, experiencer) [cite: Dowty 1991; Fillmore 1968]. In "Anna praised Ben", Anna comes first, is the subject and is the agent. In "Ben was praised by Anna" these come apart. Models answer questions about such sentences well, but it is unclear which of the three properties their internal states encode, where in the network that encoding lives, and how a question gets to it.

We answer these questions causally rather than with probes. We build sentence sets in which order, function and role are fully decorrelated across active, passive and two cleft constructions, and transplant the hidden state of a single name from one sentence into another. If the model's answer moves with the transplanted name's role, rather than its position or function, the role is carried on that name. We then trace how a question reads the roles out, by patching the answer position, single attention heads and the question tokens across layers.

The study yields a consistent picture across five open model families (Gemma 4, Qwen 3.8, phi-4, OLMo 3.1, Granite 4.2). First, the names carry their thematic role in the middle layers, more strongly than order or function; psych verbs show the code is thematic, not merely "subject". Second, a question retrieves the role through a small set of attention heads at a fixed depth, and these heads are largely shared across inputs: heads selected on template sentences repair answers on new verbs, Spanish, Galician, Universal Dependencies sentences and Wikipedia text. Third, retrieval can fail while storage stays intact. Two questions with the same meaning, "Who was the chasing done by?" and "Who was someone chased by?", act as a minimal pair: models answer the first and often fail the second. Because the sentence precedes the question, the name states are identical under both; the failure enters at the question, mostly at "by", and is repaired by patching the question state or the heads.

Our contributions:

1. A decorrelated paradigm that separates order, grammatical function and thematic role, with causal name transplants, in English, Spanish and Galician.
2. Evidence that model states organize participants by thematic role, bound to the names in the middle layers, and that a shared set of heads reads them out at a fixed depth, across five model families and on natural text.
3. A dissociation between storage and retrieval, shown with a minimal pair of questions, and confirmed on a held-out test set with all claims, layers and heads fixed in advance **[test0]**.

**Q:** final title. Current: "Thematic roles live on the names: how language models represent who did what".

## 2 Related work

**Roles in model representations.** Grammatical role is recoverable from contextual embeddings and is computed later for non-prototypical arguments [cite: Papadimitriou et al. 2022]. Edge probing recovers semantic roles from BERT layers [cite: Tenney et al. 2019]. Sentence representations in LLMs reflect syntax more than reversed role assignments, with role information concentrated in some heads [cite: Denning et al. 2025]. Aljaafari et al. [cite: 2026] localise compact semantic-role circuits in models up to about 1B parameters. We differ in three ways: causal transplants of single names rather than probes, a design that fully decorrelates order, function and role, and models of about 30B parameters across five families.

**Binding and retrieval.** Entities and their attributes are bound in context by binding or ordering IDs [cite: Feng and Steinhardt 2024; Dai et al. 2024], and question-time retrieval often runs through a lookback from the final tokens [cite: Prakash et al. 2026]. Our heads and handover depth are an instance of such retrieval; what is new is what is retrieved (a thematic role on a name), that the same heads serve new languages and natural text, and a natural case where retrieval fails while the bound information stays intact.

**Causal interpretability.** We use activation patching and head-level interventions [cite: Meng et al. 2022; Wang et al. 2023] and follow work that tests circuit claims on held-out data with fixed hypotheses.

**Surface heuristics and prompt sensitivity.** Models can lean on word order and other surface cues [cite: McCoy et al. 2019; Ettinger 2020], and their answers can change with meaning-preserving rewording [cite: Sclar et al. 2024]. We use one such rewording as a controlled minimal pair: the sentence is held constant, and we explain the difference mechanistically instead of measuring it.

**Linguistic theory.** Thematic roles and proto-roles [cite: Fillmore 1968; Dowty 1991] and psych verbs, whose experiencer can be subject or object [cite: Belletti and Rizzi 1988], give the predictions we test. Human misinterpretation of noncanonical sentences has been described as a retrieval difficulty rather than a failure to build the meaning [cite: Ferreira 2003; Bader and Meng], which parallels our storage–retrieval dissociation.

**Q:** verify every citation entry (several come from a quick 15 Nov search); add filler–gap work [cite: Boguraev et al. 2025] if space allows.

## 3 Paradigm, models and methods

**Constructions.** Four constructions decorrelate order, grammatical function and thematic role exactly: across them, each label is uncorrelated with the other two.

| Construction | Example (Leon = agent) | First name | Subject | Agent |
| --- | --- | --- | --- | --- |
| active | Leon praised Mira. | Leon | Leon | Leon |
| passive | Mira was praised by Leon. | Mira | Mira | Leon |
| object cleft | It was Mira that Leon praised. | Mira | Leon | Leon |
| by-cleft | It was by Leon that Mira was praised. | Leon | Mira | Leon |

The core set crosses 24 names with 16 transitive verbs (276 unordered name pairs). Prompts read `Sentence: … / Question: … / Answer:` inside each model's chat template, with the instruction "Reply with one name only." Ordinary questions never name a participant ("Who praised someone?"), so the answer cannot be found by elimination. A broader English set (BX) adds new names and verbs, 18 constructions, ditransitives and psych verbs; Spanish and Galician versions use the same name pairs.

**Models.** Gemma 4 31B-it, Qwen3.8-27B (thinking off), phi-4 (14B), OLMo-3.1-32B-Instruct and Granite-4.2-30B (thinking off). Inclusion rule, fixed on dev: ≥ .80 accuracy on ordinary questions and ≥ .90 on the working question. Generated replies are scored with a rule-based extractor that credits the gold name in a bare answer, a frame or a full sentence.

**Name transplants (where roles are stored).** We replace the hidden state of one name in a base sentence, at one layer, with the state of a name from a source sentence in another construction, and measure the change in that name's answer margin over the other name. Regressing this change on the source name's order, function and role within each base item and slot gives one coefficient per property per layer. Donor name pairs are counterbalanced, so name identity cancels.

**Answer patching (how roles are read).** For the minimal pair, we patch the residual stream or single attention-head outputs at the answer position from the working run into the failing run. Repair is (patched − failing) / (working − failing) on the first-token logit margin, per item. The handover depth is the first layer where residual repair reaches .5. Heads are ranked on one part of the families and evaluated on the rest, against random heads from the same layers.

**Question patching (where the failure enters).** We swap single question tokens ("by", the verb, "someone", the final "?") between the two questions, and block attention from the answer position to the question or to the names, layer by layer.

**Splits and freeze.** Families (name pairs) are split into dev (15%) and four test sets. All exploration used dev. Claims, pass rules, layers and heads were frozen in a public commit before test set 0 was opened (§9).

## 4 What the name states encode

In all five models, a transplanted name moves the answer according to its thematic role, not its position or grammatical function. The role coefficient peaks in the middle of the network, at 29–59% of depth, and replicates on the broader English set with new names, verbs and constructions.

| Model | Layers | Role layer (depth) | Role coefficient, core | Role coefficient, BX transitive |
| --- | --- | --- | --- | --- |
| Gemma 4 31B | 60 | 28 (.47) | 1.08 | 1.13 |
| Granite 4.2 30B | 64 | 37 (.59) | 0.75 | 0.87 |
| phi-4 | 40 | 15 (.38) | 0.59 | 0.55 |
| Qwen3.8 27B | 64 | 29 (.46) | 0.58 | 0.71 |
| OLMo 3.1 32B | 64 | 18 (.29) | 0.53 | 0.49 |

Coefficients are in logit units of the name's answer margin per unit change of the source property, at the role layer. **[CI]**: family-bootstrap intervals to add. Order and function coefficients are near zero or small (function +0.08 to +0.36 on BX); a validity check, in which the left context alone predicts nothing, is zero in every model.

**The code is thematic.** Psych verbs separate thematic role from subjecthood. In "Mia fears the storm" the experiencer is the subject; in "The storm frightens Mia" it is the object. If the name code tracked subjects, it would flip between the two classes. It does not: the stimulus–experiencer coefficient is +0.65 to +1.01 in the fear class, where the subject is the experiencer, in all five models.

**Q:** per-model psych numbers and CIs into an appendix table; one sentence on the ditransitive block (agent vs recipient).

## 5 How the representation develops across layers

In every model the same order holds: roles are stored on the names first, the question's contribution is handed to the answer position next, and the answer is formed last. The stages sit at different depths in different models, but never in a different order.

| Model | Role peak on names | Question → answer crossover | Answer handover (repair ≥ .5) |
| --- | --- | --- | --- |
| OLMo 3.1 32B | 18 (.29) | ≈ .38 | 26 (.41) |
| phi-4 | 15 (.38) | ≈ .50 | 22 (.56) |
| Gemma 4 31B | 28 (.47) | ≈ .65 | 40 (.68) |
| Granite 4.2 30B | 37 (.59) | ≈ .75 | 52 (.83) |
| Qwen3.8 27B | 29 (.46) | ≈ .75 | 54 (.86) |

Layer index (fraction of depth). The crossover is where swapping the whole question stops repairing the failing answer and patching the answer position starts to.

**Figure 2 (planned): layer development in a good, a middle and a poor model.** One panel per model on a common depth axis, with three curves and family-bootstrap bands: the role code on the names, question-swap repair, and answer-position repair. Good = Gemma, middle = phi-4, poor = OLMo. All five go to the appendix.

**[OTR-14]** The per-layer attention-blocking curves (when the answer position reads the names, and when it reads the question) are running; they will place the reading windows on the same axis and give the "by" layer for §7.

**Q:** is "good / middle / poor" defined by behaviour (failing-question accuracy) or by the strength of the code? Proposal: behaviour, stated once.

## 6 How a question reads the roles

A small set of attention heads at a fixed depth carries the answer. Patching the outputs of the top 20 heads from the working run into the failing run repairs 34–82% of the margin on held-out families, against at most 11% for 20 random heads from the same layers.

**The heads are shared, not fitted to one dataset.** Ranking heads afresh on a new stimulus set recovers mostly the same ones: 13–19 of the top 20 overlap with the heads ranked on the core set, where chance overlap is 0.3–2.1 (all p < 1e-13). Heads frozen on the core templates repair other inputs about as well as heads ranked on those inputs:

| Model | BX transitive | Spanish / Galician | UD sentences | Wikipedia, real / neutral names |
| --- | --- | --- | --- | --- |
| Gemma 4 31B | .78 | .73 / .76 | .81 | .77 / .83 |
| phi-4 | .74 | .69 / .62 | .67 | .67 / .71 |
| Granite 4.2 30B | .50 | .59 / .62 | .52 | .42 / .46 |
| OLMo 3.1 32B | .35 | .44 / .39 | .39 | .37 / .45 |
| Qwen3.8 27B | .34 | .36 / .34 | .42 | .34 / .36 (v0.1) |

Repair by the 20 frozen heads; random heads ≤ .10 throughout.

**The handover depth is stable.** The first layer where residual repair reaches .5 is the same on templates, new English constructions and natural text: Gemma 40, Granite 52 and phi-4 22 on Wikipedia, exactly as on the core set; OLMo two to four layers earlier (22–24 vs 26). Qwen's handover is late everywhere (50–60 of 64).

**[test0]** C4 (handover depth) and C5 (frozen heads vs random, with two controls: head outputs from another item's failing run, and a random nudge of the same size) are tested on the held-out set.

## 7 Stress test: when the read-out fails, the stored state survives

Two questions with the same meaning give different answers. On the broad English set, accuracy drops from ordinary questions to the failing question "Who was someone {V}-ed by?" in every model: Gemma .99 → .86, Granite .90 → .85, Qwen .90 → .61, OLMo .88 → .41, phi-4 .90 → .13. The wrong answer is the patient, not noise.

**Storage cannot differ.** The sentence precedes the question, and the models attend only to earlier tokens. The hidden states of the names are therefore identical under the two questions, at every layer. Whatever goes wrong happens after the names have been encoded: in how the question retrieves them.

**The read-out reverses.** Under the failing question, the role code on the names is read with a weaker or reversed sign on the items the model gets wrong (BX transitive: phi-4 −0.75, OLMo −0.28, Qwen −0.14; Gemma +0.30 and Granite +0.27, weakened; psych verbs reversed in all five). The same stored role is fetched with the wrong polarity.

**The model still knows.** After the neutral prefix "Someone was {V}-ed by", the next-token prediction picks the agent on most items whose bare answer was wrong: Gemma 1.00, Granite .85, Qwen .83, OLMo .59, phi-4 .29. In three models the knowledge is fully available; it is the question that fails to retrieve it.

**The failure enters at "by".** Swapping the state of "by" from the failing into the working question damages the working answer (Granite 1.55, OLMo 1.12, Gemma .78, Qwen .59, phi-4 .35, in units of the working–failing gap); swapping the working "by" into the failing question repairs it (OLMo .75, phi-4 .57, Granite .49, Gemma .32, Qwen .22). The verb carries little of it (phi-4 on BX is the exception), and "someone" matters only in the earliest layers. The end of the question carries part of the difference too, so "by" is the main entry point, not the only one.

**Repair.** Patching the answer position from the working run at the handover depth, or the 20 heads of §6, restores the correct answer (§6).

**[test0]** C1 (selective failure), C3 (read-out weaker than C2), C6 ("by" carries the failure), C7 (knows but answers wrong: Gemma, Granite, Qwen) and C8.

**[question battery, running]** "Who was Mary chased by?", "By whom was someone chased?", "Who chased someone?" and related questions separate the placeholder "someone" from the stranded "by" and test whether the failure survives fully natural wording.

## 8 Generality: other languages and natural text

The same representation and the same read-out appear outside our templates: in Spanish and Galician, in natural English sentences from Universal Dependencies, and in 7,923 sentences we extracted from Wikipedia.

**Spanish and Galician.** The names carry the role in both languages (Gemma +0.68 / +0.92, Qwen +0.74 / +0.70, Granite +0.63 / +0.43, OLMo +0.46 / +0.24, phi-4 +0.37 / +0.35). Unlike English, grammatical function is also positive (+0.1 to +0.6). The passive agent question fails here too, and the read-out reverses on wrong items in 7 of 10 model–language cells. The English heads repair Spanish and Galician answers (§6).

**Natural sentences.** We take real sentences with two human participants and put names in their place (UD: 188 sentences from EWT and GUM), or keep the real names and add a neutral-name copy (WIKI-ROLES: Wikipedia sentences filtered with two NER systems, 5,887 active and 2,036 passive items, 86 and 77 verbs). The failing question fails on natural text as well:

| Model | UD, failing / working | Wikipedia actives, real / neutral names |
| --- | --- | --- |
| Granite 4.2 30B | .71 / .99 | .73 / .76 |
| Gemma 4 31B | .50 / 1.00 | .55 / .54 |
| phi-4 | .41 / 1.00 | .51 / .39 |
| OLMo 3.1 32B | .15 / .99 | .23 / .20 |
| Qwen3.8 27B | .22 / 1.00 | .28 / .25 (v0.1) |

Share of items where the bare answer prefers the agent (first-token logit comparison; Q: confirm UD used the same measure). The working question stays at ≥ .96.

**Not memory.** Real and neutral names give the same accuracy, the same handover layer and the same head repair, so the effect does not come from what the model knows about real people.

**Q:** Swedish results stay out until a native speaker has checked the materials. Spanish/Galician constructions enter the main text only after Joel's judgements (by-cleft reading; Galician OVS).

## 9 Held-out confirmation

Before opening test set 0 (59 name-pair families, disjoint from dev), we froze eight claims, their pass rules, the layers and the top-20 heads of every model in a public commit, together with the scoring script. Each claim is tested per model with 95% family-bootstrap intervals; results are reported whatever they show.

| Claim | Gemma | Granite | phi-4 | Qwen3.8 | OLMo |
| --- | --- | --- | --- | --- | --- |
| C1 The failing question fails selectively | | | | | |
| C2 Roles live on the names (role > 0, role > function) | | | | | |
| C3 The failing question weakens the read-out | | | | | |
| C4 Fixed handover depth (±2 layers) | | | | | |
| C5 Frozen heads repair, controls do not | | | | | |
| C6 The state at "by" carries the failure | | | | | |
| C7 Knows but answers wrong | | | — | | — |
| C8 "By" from the patient question damages the working answer | | | | | |

**[test0]** Cells get pass / fail / not evaluable with the estimate and interval. No multiple-comparison correction is applied to these replication criteria. The full freeze document goes in the appendix.

## 10 Discussion and conclusion

Models build something that behaves like thematic roles, and they attach it to the participants. The code follows agents, patients and experiencers rather than positions or subjects, it lives on the name tokens, and it is built in the middle layers before any question is read. In linguistic terms this is closer to a referent carrying its role than to a role read off a syntactic configuration.

Retrieval is a separate step with its own failure modes. A shared set of heads fetches the role at a fixed depth for templates, new constructions, two other languages and natural text. A small change in the question, mostly at "by", sends that fetch to the wrong participant, while the stored state is untouched. Behavioural tests of "who did what to whom" can therefore mistake a retrieval failure for missing knowledge, and probes can show knowledge that the behaviour does not use. Evaluations that want to measure understanding should separate the two.

The parallel with human comprehension is suggestive: misreadings of noncanonical sentences have been described as retrieval difficulties rather than failures to build the meaning. We do not claim the mechanisms are the same; we claim the dissociation is real in models and can be measured causally.

**Q:** one sentence on what this means for training or prompting, or leave it out?

## Limitations, appendix and figures

**Limitations.**

- Five open models of 14–32B parameters; no frontier models internally (API behaviour only, appendix).
- One family of failing questions. We characterise where it enters ("by", the end of the question), not a full account of which questions fail; the question battery narrows this.
- "Thematic" rests on psych verbs and cross-construction transfer; we say the states are organised by thematic role, not that the model has thematic roles in a theoretical sense.
- Natural-text sets have no held-out split; they are validation, not confirmation.
- Swedish is excluded pending native-speaker checks; Spanish and Galician materials were checked by one native speaker (an author).

**Appendix plan.** Freeze document; per-model layer curves for all five models; full head lists; psych and ditransitive blocks; prompt robustness (OTR-15); OTR-13 token-by-token question patching; extractor validation; WIKI-ROLES construction and check sheet; API behaviour; a note that forced-choice scoring of chat models manufactured a subject–object asymmetry that generation scoring removes.

**Figures (main text, 4 max).**

1. The minimal pair and the pipeline: names store roles → question → heads at the handover depth → answer (schematic).
2. Layer development in a good, a middle and a poor model (§5).
3. Shared heads: transfer of frozen heads across datasets, with the overlap against chance (§6).
4. Knows vs answers: neutral-prefix agent share and "by" damage/repair per model (§7).
