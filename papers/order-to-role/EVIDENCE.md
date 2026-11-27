[← Order-to-Role](README.md) · [Paper architecture](PAPER_MAP.md)

# Evidence and validation status

**Snapshot: 27 November 2026.** This is a navigation ledger, not an independent reproduction, statistical audit or pass/fail report.

## Research status

| Evidence stage | Source | Status |
|:--|:--|:--|
| Early exploratory study and scoring corrections | [November 17 review](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/README.md) | Historical; some claims dropped |
| Development synthesis | [Manuscript snapshot](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/paper_draft.md), [result log](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/results_log.md) | Exploratory |
| English C1–C8 | [FREEZE.md](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) | Frozen November 26; test0 files archived |
| BX transitive replication | [FREEZE.md](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) | Prespecified separate replication; outputs archived |
| Spanish/Galician XL-C1–C5b | [FREEZE_XL.md](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md) | Separate freeze after English test0; XL test0 files archived |
| Layer-pipeline ordering, OTR-15, psych/ditransitive and UD/WIKI | [Development log](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/results_log.md), [freeze scope](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) | Exploratory / external validation, not core test0 claims |
| Larger models | [70B extension](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md) | Development runs begun; separate freeze needed |

The output files exist. **This guide does not certify which claims passed**, which runs are complete, or whether every intended control was executed. The authoritative scoring rules and run-specific provenance remain in the source documents.

## Claim map (English freeze)

| Claim | Frozen question | Main evidence |
|:--|:--|:--|
| C1 | Selective generated-answer failure | Generation check |
| C2 | Positive role effect beyond grammatical function at frozen layer | OTR-03 |
| C3 | Weakened or reversed role readout in wrong-answer conditions | OTR-03 |
| C4 | Prespecified answer handover depth | OTR-09 |
| C5 | Frozen heads repair relative to random and specified controls | OTR-09 |
| C6 | Failing-question `by` state damages working answer | OTR-14 |
| C7 | Neutral-prefix preference on initially wrong items in specified models | OTR-07 |
| C8 | Patient-question `by` state damages working answer | OTR-14 |

See the [English freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) for full estimands, minimum denominators, thresholds, C7 model scope and all reporting rules. The [XL addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md) has its own definitions, including the other-working-item content control (XL-C5b). Do not substitute the English criteria for XL or retroactively change a failed criterion.

## Interpretation boundaries and corrections

- First-token forced choice misrepresented some chat-model generations in the early investigation. Use matched generated-output measures for behavioral claims.
- Identical pre-question states follow from causal masking, but this alone does not prove the required role information is stored or retrieved.
- Normalized repair is conditional on the defined margin gap; it is not an overall accuracy improvement.
- Role/function separation differs across languages; Galician materials include explicitly disclosed usage caveats.
- Development OTR-03 donors included name-pair families assigned to test splits. The freeze discloses that and prescribes split-restricted test donors.
- The English freeze reports no multiplicity correction for replication criteria. Report every model/claim, including failed and not-evaluable cells.
- Causal intervention sensitivity is not proof of an exclusive native mechanism.

[Result archive at inspection](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/25c6914e29473e4b1a1063600c48743f68b705c3/results) · [Reproduce and score](REPRODUCE.md).
