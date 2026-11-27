[← Order-to-Role](README.md) · [Paper architecture](PAPER_MAP.md)

# Generality and transfer

The transfer question is whether components identified on controlled English materials retain causal influence elsewhere, rather than whether a new search can find some influential component on every dataset.

## Linguistic and dataset axes

| Material | What it tests | Evidence status |
|:--|:--|:--|
| Core English | Orthogonal order, subject and role contrasts | Development plus frozen English test0 |
| BX transitive | New names, verbs and constructions | Development plus prespecified test0 replication, reported separately |
| BX ditransitive / psych | More argument structures; stimulus/experiencer vs subject distinctions | Exploratory development |
| Spanish | Question form, morphology and construction transfer | Development plus frozen XL primary test0 |
| Galician | Cross-language transfer under distinct constructions | Development plus frozen XL secondary test0 |
| Swedish | Additional linguistic diversity | Keep exploratory; verify native-speaker review before reporting |
| Universal Dependencies EWT/GUM | Corpus-derived sentence contexts with participant phrases substituted | External validation, not a frozen test split |
| WIKI-ROLES | Automatically extracted Wikipedia sentences with real/neutral names | External validation; track dataset version and manual annotation quality |
| 70B-class Llama and Qwen | Scale and architecture extension | Exploratory development until a separate freeze |

## A reproducible development comparison

The table below transcribes the **20-head normalized repair** summaries in the [November 25 development log](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/results_log.md). Head rankings were chosen on the earlier English template runs, separately for each model. Values refer to eligible items and are not ordinary accuracy.

| Model | Spanish | Galician | UD | WIKI v0.2 real / neutral |
|:--|--:|--:|--:|:--|
| Gemma | .73 | .76 | .81 | .77 / .83 |
| Qwen | .36 | .34 | .42 | Pending in this historical snapshot |
| Phi-4 | .69 | .62 | .67 | .67 / .71 |
| OLMo | .44 | .39 | .39 | .37 / .45 |
| Granite | .59 | .62 | .52 | .42 / .46 |

These rounded **development** values are historical orientation, not final results. The later result archive includes additional runs, including Qwen WIKI. Use original per-item data, controls and confidence intervals for the paper. A head ranking transferred within one model is not the same as sharing head identities across architectures.

## Validation and limitations

The [XL addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md) was frozen after the English test0 was seen but before the Spanish/Galician test runs, with distinct pass rules and language-specific handover layers. The recorded native-speaker check notes formal phrasing and non-normative but widespread Galician prepositional accusatives. Retain these disclosures.

UD substitutes argument phrases with names. WIKI-ROLES uses automatic parses and named-entity recognition; record the exact version and annotation audit, and avoid claiming all gold labels are human-verified. Real vs neutral names is a useful control for memorized names, not a proof that pretraining memorization cannot contribute. Corpus validation is not a held-out pretraining guarantee.

[Current code snapshot](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e) · [Pinned result archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/25c6914e29473e4b1a1063600c48743f68b705c3/results) · [Large-model plan](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md).
