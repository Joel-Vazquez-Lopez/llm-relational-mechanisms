# Review brief (for an independent reviewer, e.g. Codex)

You are reviewing an exploratory LLM-linguistics project before more weeks are invested in it.
Read `README.md` first, then the code it points to. Be adversarial: we want the weaknesses a
hostile ACL/EMNLP reviewer would find, plus anything that is simply wrong. Do not take our
interpretations on trust; check them against the result files and the code.

Target: an ACL Rolling Review submission in early 2027 (main track if the evidence allows,
otherwise Findings / BlackboxNLP / CoNLL). Everything so far is dev split; no test set opened.

## A. Correctness of code and numbers (highest priority)

1. **Answer extraction** (`scripts/otr_generate_check.py`: `extract_answer`, `event_statement`).
   Can it mis-score answers? In particular: (a) statement patterns that match a reversed or
   unrelated sentence; (b) the "inferred" rule after removing copies of sentence/question;
   (c) verbatim copies counted as correct statements; (d) role mapping for the three-place
   frames (`role == "agent"` → doer, otherwise the prepositional participant). Spot-check
   stored outputs in the result JSONs (`rows[].output`).
2. **Rescoring and figures** (`scripts/otr_rescore_generation.py`,
   `docs/order_to_role/review_2026-11/make_review_figures.py`). Does each figure use the run it
   claims (`tables/provenance.json`)? Is `find_gen` choosing the right file when several runs
   share metadata? Are the within-item cross-tabs (`t3`) joined on the right item ids?
3. **Stimuli** (`order_role_stimuli.py`, `stranded_p_stimuli.py`). Are the decorrelation, the
   family split (SP reuses OTR's families), and gold answers correct? Any prompt where the
   question leaks the answer or the gold is debatable? Are any SP sentences/questions
   ungrammatical enough to explain errors (e.g. "gotten", "Who was a job offered to someone by?")?
4. **Logit lens** (`scripts/otr06_logit_lens.py`). Is the final norm / unembedding / softcap
   handling right for Gemma 4 (multimodal wrapper) and Qwen3.8? The last-layer check is
   0.19 logits for Gemma: acceptable or a sign of a mismatch? Is first-token scoring adequate
   when names are multi-token for a tokenizer?
5. **Forced-choice comparison** (`t1`): forced choice and generation use different item samples.
   Is the comparison still fair? What would make it clean?

## B. Do the conclusions follow?

6. **C3 (within-item dissociation)** is our core result. Is "the model knows the roles but the
   bare answer selects the wrong argument" warranted? Alternatives: the sentence instruction
   changes how the question is read (a prompt effect, not a format effect); the model rewrites
   the event before answering (a reasoning step a bare answer lacks); Qwen3.8's 40 reverse
   cases. What control would separate "knowledge vs retrieval" from "extra computation"?
7. **C4 (logit lens)**: is "chosen at the normal depth, not overridden" a fair reading of the
   curves, given logit-lens limits? What would you run instead (tuned lens, probes, patching)?
8. **C6 / C7**: we dropped a filler-gap account and a general voice-mismatch account. Were the
   tests adequate (verb sets, frames, item counts)? Is the remaining pattern (passive agent
   questions; errors name the filler of "someone"; verb- and cleft-conditioned) one phenomenon
   or several? Which of the candidate accounts in README §4.1 is most promising, and what is the
   cleanest experiment to decide?
9. **Statistics**: no CIs or tests in these tables. Which comparisons need family-clustered
   bootstrap CIs or mixed models before any claim? Are 14–18 items per verb enough for the
   per-verb pattern?
10. **Confounds we may have missed**: name effects (gender, frequency, token length), verb
    frequency in the passive, the indefinite "someone", the chat template, greedy decoding,
    the "Answer:" cue, answer-position effects.

## C. Is this a paper, and which one?

11. Given C1–C8, what is the strongest honest framing? Options we see: (i) knowledge-vs-answer
    dissociation in strong models with a mechanism; (ii) a methods paper on forced choice vs
    generation for chat models; (iii) both. What would a hostile reviewer say about novelty
    (related work: word-order insensitivity, filler-gap in LMs, "knows but answers wrong",
    entity tracking / lookback mechanisms)?
12. Is two models enough for anything? Which additional families (different labs) and which
    tests per model are the minimum for a credible claim?
13. Is the plan in README §4 the right priority order? What would you cut, and what is missing?

## Deliverable

A short report with: (1) bugs or mis-scorings found, with file/line and a concrete example;
(2) claims you think are not supported, and why; (3) missing controls ranked by importance;
(4) your recommended framing and target venue; (5) the three experiments you would run next.
