[← Model Linguistics](../../README.md) · [Research history](../../investigations/README.md) · [Paper II](../order-to-role/README.md)

# Paper I · Relational Learning in Transformers

*Provisional navigation title; manuscript title remains unchanged.*

### Targeted experience, relational generalization and role-assignment errors in Transformers

**Model class:** small scratch-trained Transformers  
**Stage:** working manuscript; further experiments remain under review  
**Main question:** how does controlled grammatical exposure change generalization, joint-role errors and the computations that support answers?

The core experiments compare matched training histories on a restricted two-event synthetic grammar. Complementary questions ask for an agent or patient in a specified event. This supports precise exposure comparisons, but does **not** establish natural-language generalization or a universal learned role algorithm.

## Read the study

| Purpose | Document |
|:--|:--|
| **Paper text** | [Canonical working manuscript](../../paper/manuscript.md) |
| **What is supported?** | [Evidence and limitations](EVIDENCE.md) |
| **Where are the experiments?** | [Experiment index](EXPERIMENTS.md) |
| **How to inspect or rerun?** | [Reproducibility guide](REPRODUCE.md) |
| **Earlier research narrative** | [Scratch investigation](../../investigations/scratch_transformers/README.md) |
| **Experimental state** | [Current findings checkpoint](../../investigations/scratch_transformers/CURRENT_FINDINGS.md) |

## Research questions

**Learning.** Which controlled exposure conditions affect protected recombination performance? How does performance vary over seeds, training steps and lexical splits?

**Error structure.** Do agent and patient queries fail jointly, reverse roles, or select the same participant? What does each metric establish under candidate-restricted versus full-vocabulary scoring?

**Mechanism.** Where is role- or question-related information accessible across time and depth, and which specific pathway hypotheses survive frozen tests?

**Transfer.** Which observed patterns persist when grammatical structure, vocabulary or evaluation material changes? A development-only external benchmark pilot is not a protected transfer result.

## Reading the evidence responsibly

The manuscript records repeated within-family exposure effects, unresolved same-participant errors, and unsuccessful or inconclusive mechanistic predictions. Those negative results are part of the research contribution.

The newer [EXP-109–116 and PHASE2 work](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/exp102-clean-design) is not yet integrated into `main` and should not be silently interpreted as a revision to the manuscript's frozen evidence base.

**Next:** [Evidence](EVIDENCE.md) · [Experiments](EXPERIMENTS.md) · [Reproduce](REPRODUCE.md)
