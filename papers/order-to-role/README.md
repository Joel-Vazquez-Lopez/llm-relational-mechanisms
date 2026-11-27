[← Model Linguistics](../../README.md) · [Research history](../../investigations/README.md) · [Paper I](../learning-relations/README.md)

# Paper II · Order-to-Role

### Thematic roles, question-conditioned readout and answer selection in pretrained language models

**Working paper:** *Thematic roles live on the names: how language models represent who did what* (provisional title)  
**Research status:** development findings; frozen English and multilingual test0 runs archived; frozen English and XL claim-by-claim scoring review pending; [FREEZE.md §6](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md#6-scoring-failures-log) records the development dry run (C5 not evaluable there because k=20 transfer output was missing)  
**Core models:** Gemma, Qwen, Phi, OLMo and Granite; 70B-class extensions exploratory

The central question is how a language model represents participant roles independently of word order and grammatical function, how a question accesses these representations, and which computations influence the answer. The research combines controlled linguistic contrasts, name-state transplants, layerwise and token-level interventions, head-level repair and transfer.

## Read in scientific order

| Research question | Start here |
|:--|:--|
| What is the paper's argument? | [Paper architecture](PAPER_MAP.md) |
| Where is role information and how might it be read out? | [Layer investigation](LAYERS.md) |
| Does the effect transfer across languages and natural text? | [Generality and transfer](TRANSFER.md) |
| What has been frozen, and what remains exploratory? | [Evidence and validation status](EVIDENCE.md) |
| Where are the implementations? | [Experiment map: OTR-01–15](EXPERIMENTS.md) |
| How can the analyses be reproduced? | [Reproducibility guide](REPRODUCE.md) |

## Current research documents

- [24 November manuscript draft (25 November state snapshot)](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/paper_draft.md) — historical development draft; not updated by this navigation.
- [Development results log](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/results_log.md) and [result index](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/result_index.md).
- [English test0 freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) and [Spanish/Galician freeze addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md) — authoritative scope, criteria and exclusions.
- [November 17 review](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/README.md) — earlier findings and corrections, not the latest manuscript.

The key distinction is between role information in participant states, question-dependent readout and the final answer preference. The experiments test parts of this account; neither a patch effect nor an ordered sequence of layer peaks establishes an entire unique native pathway by itself.

## Where the work lives

[Active code and documentation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) · [Pinned results snapshot](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/25c6914e29473e4b1a1063600c48743f68b705c3/results) · [Repository map](../MAP.md)

PR #12 merged a November 21 results snapshot into main. Later runs and the current OTR implementations are still maintained on their research branches. No frozen file is changed by this navigation update.

*Research navigation snapshot: 27 November 2026.*
