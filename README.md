<div align="center">
  <img src="assets/model-linguistics-banner.svg" alt="Model Linguistics — experience, computation and generalization" width="100%">
  <h3>How do Transformers learn, represent and use linguistic relationships?</h3>
</div>

Model Linguistics is a research programme investigating the internal computations that allow language models to understand relationships: **who did what to whom, how participants are bound to events, and how a question retrieves the right information to produce an answer.**

We approach this problem from two complementary directions: controlled training of Transformers to understand how relational abilities develop, and mechanistic investigation of pretrained language models to understand how those abilities are represented and used.

For example, *"Anna praised Ben"* and *"Ben was praised by Anna"* describe the same relationship despite different word order and grammatical subjects. We study how models learn that distinction, encode it across layers, and select the correct participant.

---

## Explore the research

| **01 · Research history** | **02 · Paper I** | **03 · Paper II** |
|:--|:--|:--|
| **[Historical Development](investigations/README.md)** | **[Relational Learning in Transformers](papers/learning-relations/README.md)** | **[Order-to-Role](papers/order-to-role/README.md)** |
| How the original investigation evolved: experiments, discoveries, failed hypotheses and the emergence of the two paper directions. | How grammatical experience and training shape relational representations, role assignment, and generalization in scratch-trained Transformers. | How pretrained language models encode thematic roles and retrieve them across layers, constructions, languages and natural text. |
| Research archive and chronology | Controlled learning and mechanism development | Mechanistic interpretability and multilingual transfer |

---

The two papers share a scientific motivation, but have distinct experiments, hypotheses and evaluation protocols. Their evidence is kept separate. The historical record preserves the work that led to both, including negative and inconclusive findings.

*Active research programme · Working manuscripts · Reproducible experiments and frozen evaluation protocols*
