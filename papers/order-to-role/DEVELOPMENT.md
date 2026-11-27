[← Order-to-Role](README.md) · [Results](RESULTS.md) · [Evidence ledger](EVIDENCE.md)

# How the study developed

From the first behavioural test on 13 November to the frozen held-out evaluation on 26 November. Each stage records what was tried, what was learned and what changed as a result, including the hypotheses that failed. Commit hashes refer to [`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli). Results are on [`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev).

## Origins (late October to 12 November)

The research programme began by asking how language models work out who did what to whom. Early experiments on pretrained Qwen and Llama models (ML-001–036, ML-048–055) produced some strong local effects, but several transfer checks failed, and the training data of pretrained models could not be controlled. The work then moved to small transformers trained from scratch, which became [Paper I](../learning-relations/README.md).

On 12 November, a test built for those small models (choosing between two answer options) was tried on Qwen2.5-1.5B-Instruct. Its answers stayed the same under swapped options for only 3 of 16 sentences. Pretrained chat models needed their own design. That is the starting point of this paper; the earlier pretrained work is in the [historical record](../../investigations/pretrained_llms/README.md) and is not part of its evidence.

## Stage 1 · Design and first tests (13–15 November)

- **Design** (`d2a8b9f`, 13 Nov). A stimulus generator was built in which four constructions make order, grammatical function and thematic role exactly uncorrelated, with reversible name pairs, questions that never name a participant, and a family-level split into development and four test sets.
- **The original hypothesis** gave the study its name: models would convert linear **order** into thematic **role** in stages across layers.
- **OTR-01 (behaviour).** No Qwen2.5 model (1.5B or 7B, base or instruct) passed the gate; accuracy was .55–.65. Their fallbacks differed: the base 7B model followed focus, the instruct models followed the grammatical subject in clefts. Order alone never won.
- **OTR-02 and OTR-03 on Qwen2.5.** There was no answer information at the verb, "was", "by" or the period; the answer was read late from the name's own position. Transplants were dominated by name identity, and the role code was about 0.02 logits. **The staged order → role conversion was not seen.**
- **Two confounds** in the first transplant design were found and fixed: a single foreign pair per base, and a pooled regression that let identity pass as role (`1759d6a`, `b08947a`).
- **Gemma 4 31B** essentially solved the task in chat mode (98–100% on three of four question wordings). Every error came from one wording, "Who was someone praised by?", first treated as a stimulus problem.

## Stage 2 · Measurement crisis and the failing question (16–17 November)

- **The generation check** (`db3f5c1`, 16 Nov) showed that first-token forced choice misrepresented chat models: they restate the sentence, so forced choice records the first name of the restatement. An apparent subject–object asymmetry disappeared under generation. The [review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/README.md) (`7d528e5`) recorded the corrections and dropped claims.
- **The "stimulus problem" became the finding.** A battery of question variants showed that the stranded "by" in "Who was someone V-ed by?" fails selectively, while "Who was the V-ing done by?" asks for the same person and works. The sentence is the same, so the two questions form a minimal pair.
- **First mechanism tools:** probes (OTR-08), incremental role states (OTR-10) and steering (OTR-11). Spanish and Galician stimuli were added (`02c4299`).

## Stage 3 · Inside capable models (18–19 November)

- **OTR-03 on Gemma** showed a mid-network role code on the names, larger than order or function: the central representational result.
- **OTR-09** (`893ae00`) found that repair from the working run starts at a fixed depth, and that about 20 heads are enough to fix most of the error.
- **The model set was widened** to phi-4, OLMo-3.1, Granite-4.2 and gpt-oss-20b with an inclusion gate; gpt-oss missed it. OTR-12 tested knockout and an averaged role vector.

## Stage 4 · Generalisation (20–24 November)

- **BX** (20 Nov): new names, verbs and constructions. The failure, the role code and the repairing heads all transferred. Psych verbs indicated thematic rather than subject coding. Thirteen to nineteen of the top 20 heads were recovered when ranking afresh.
- **The neutral-prefix control** (21 Nov) replaced a prefix that named the patient. "Knows but answers wrong" held unaided in Gemma, Granite and Qwen3.8, only partly in OLMo and not in phi-4.
- **Spanish and Galician** (21 Nov): the role code and the English heads transferred. Unlike English, grammatical function was also positive.
- **Natural text:** Universal Dependencies sentences (21 Nov) and WIKI-ROLES, a dataset of Wikipedia sentences with real or neutral names (22–24 Nov, `4b8e201`). The failure and the head repair appeared on real sentences, and real and neutral names agreed.
- **OTR-13** (`24463d4`, 23 Nov). Contrary to the prediction, the instruction sat on "by", not on the verb. The whole-question swap and the answer-position patch crossed at the handover depth.

## Stage 5 · Freeze and held-out test (24–26 November)

- **OTR-14** (`93ec641`) put all intervention curves on a common scale, and the role layers were set from the development peaks.
- **Review before freezing.** The freeze was reviewed adversarially against the code (24 Nov, 11 points addressed).
- **English freeze** (`f6ea0e1`, 26 Nov): eight claims, pass rules, layers, ordered head lists and the scorer were committed before any test0 command. Forty-eight test0 jobs were then submitted.
- **Spanish/Galician addendum** (`3811d7c`, 26 Nov). It was frozen after the English test0 results were known and before any es/gl test run. It uses language-specific handover layers, adds the C5b content control prompted by phi-4's English result, and records a native-speaker check (`c40a1f7`). Forty jobs were submitted.
- **Outcome** (scored on 27 Nov for the [evidence ledger](EVIDENCE.md)): English 37 of 38, BX 15 of 15, Spanish 29 of 30, Galician 30 of 30.

## Stage 6 · Extension (27 November, ongoing)

Llama 3.3 70B and Qwen2.5 72B were added, each split over two GPUs (`9c43792`). The first development runs show the same handover-and-heads picture. A separate freeze is required before any test.

## What changed along the way

| Then | Now | Why |
|:--|:--|:--|
| A staged conversion from order to role | Role is encoded directly on the names in the middle layers; a question reads it out late | Transplants in capable models |
| Forced-choice behavioural scores | Generation with a bare-name instruction | The restatement artefact |
| "Who was someone V-ed by?" as a bad stimulus | The minimal pair at the centre of the storage–retrieval dissociation | Same sentence, same stored states, different answers |
| The verb carries the question's instruction | "by" carries most of it, and the end of the question some | OTR-13 |
| The other-failing control as a content test | The other-working control (C5b) | phi-4, English test0 |
| One model (Gemma) | Five families, 14–32B; 70B in progress | Inclusion gate |
