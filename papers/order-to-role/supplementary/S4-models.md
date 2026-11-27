[← Supplement](README.md) · [Paper II](../README.md) · [Results](../RESULTS.md)

# S4 · Model-by-model results

One card per model. The verdict lines are **confirmatory** (test set 0, [S1](S1-statistics.md)); everything else is **development, exploratory**. Frozen layers come from FREEZE.md §4 and FREEZE_XL.md §4, and the script recomputes each of them from the development files it names (all 15 handovers match). Sources: [`st14_model_cards.csv`](tables/st14_model_cards.csv) and the tables linked in each line. Transfer values are from the main [`head_transfer_dev.csv`](../results/tables/head_transfer_dev.csv).

**Inclusion.** A model was kept for the mechanism work only if, on development data, it scored ≥ .80 on ordinary questions and ≥ .90 on the working question (FREEZE.md §1). gpt-oss-20b failed this gate (about .76 / .88). Its C1 numbers are reported descriptively only and are not counted in any pass count. All five models run with their chat template, the instruction "Reply with one name only.", bfloat16, and one H100 per job. Qwen3.8 and Granite run with thinking turned off (`enable_thinking=False`).

| Model | Layers | Role / "by" / handover (EN) | Handover es / gl | Test0 EN · BX · es · gl | Failed claims |
|:--|--:|:--|:--|:--|:--|
| Gemma 4 31B-it | 60 | 28 / 28 / 40 | 46 / 46 | 8/8 · 3/3 · 5/6 · 6/6 | Spanish C3 |
| Qwen3.8 27B | 64 | 29 / 30 / 54 | 50 / 50 | 8/8 · 3/3 · 6/6 · 6/6 | none |
| phi-4 14B | 40 | 15 / 8 / 22 | 22 / 22 | 6/7 · 3/3 · 6/6 · 6/6 | English C5 (control) |
| OLMo 3.1 32B-Instruct | 64 | 18 / 8 / 26 | 24 / 26 | 7/7 · 3/3 · 6/6 · 6/6 | none |
| Granite 4.2 30B | 64 | 37 / 28 / 52 | 52 / 52 | 8/8 · 3/3 · 6/6 · 6/6 | none |

---

### Gemma 4 31B-it

- **Behaviour.** Failing / working question 69% / 100% on core development data. On BX the failing question scores 86%, against 99% for ordinary questions. It fails mainly on actives (62%) and object clefts (38%).
- **Role code.** The strongest of the five: 1.08 [0.92, 1.21] at L28. The order/focus coefficient is large (0.45 at L28, up to 0.74 at L40). Function is 0.09.
- **Pipeline.** The role peak and the "by" peak coincide at L28, and the handover is at L40. In Spanish and Galician the handover moves to L46.
- **Heads.** Top-20 repair is .82 against .05 for random heads, the strongest of the five. Transfer: BX .78, Spanish .73, Galician .76, UD .81, Wikipedia .77 (real names) and .83 (neutral names).
- **Answer process.** On every wrong item (100%), the prefix "Someone was V-ed by" leads to the agent (C7).
- **Issues.** It fails Spanish XL-C3 (CI includes zero). Its C5b magnitudes (4.08, 2.73) are on another scale. The English C6 interval is wide ([0.25, 1.57]). Historically, forced-choice scoring miscounted Gemma's restatements ([historical packages](historical-packages.md)).
- **Knockout.** Ablating the top-20 heads keeps 38% of the agent margin (random heads: 100%), and 92% of answers stay correct.

### Qwen3.8 27B

- **Behaviour.** Failing / working 52% / 97%; BX 61% / 90%. It fails on actives (31%) and object clefts (12%) and is near ceiling on by-clefts (97%).
- **Role code.** 0.58 [0.54, 0.62] at L29. It is the only model with a **negative** order/focus coefficient (−0.18). Function −0.01.
- **Pipeline.** Handover at L54, the latest in relative depth (.84). It moves *earlier* in Spanish and Galician (L50).
- **Heads.** Weakest repair: .36 against .11, and the highest random baseline of the five. Qwen3.8 is a hybrid model. Its linear-attention layers have no `o_proj` and are skipped in the head scan, so fewer candidate heads exist. Transfer stays low but consistent: .33–.42 across all materials.
- **Answer process.** 82% of wrong items name the agent after the prefix (C7).
- **Knockout.** 40% of the margin kept after ablating the top-20 heads, and 86% of answers stay correct.

### phi-4 14B

- **Behaviour.** The most extreme failure: failing / working 11% / 89%; BX 13% / 90%. It fails across all constructions, including 0% on passives.
- **Role code.** 0.59 [0.53, 0.65] at L15. Order/focus 0.38, function −0.20.
- **Pipeline.** The "by" layer (L8) is far before the role layer (L15); handover at L22, the same in all three languages.
- **Heads.** Top-20 repair .75 against .05. Transfer is strong: BX .74, Spanish .69, Galician .62, UD .67, Wikipedia .67 and .71.
- **Issues.** **It fails English C5 on a control.** Its top-20 repair on test0 is .72 against .01, but the `other_failing` control repaired .26, above the .10 limit ([S1 notes](S1-statistics.md#notes)). It passes the replacement control C5b in both languages. There is no C7 claim: on development data only 29% of wrong items named the agent after the prefix.
- **Knockout.** The strongest dependence: 18% of the margin kept after ablating the top-20 heads, and correct answers fall from 90% to 65%.

### OLMo 3.1 32B-Instruct

- **Behaviour.** Failing / working 68% / 100%; BX 41% / 88%. It fails on actives (20%) but not on object clefts (97%), the reverse of Gemma and Qwen3.8.
- **Role code.** 0.53 [0.46, 0.58] at L18, the shallowest role layer (.28 of depth). Function is the most negative of the five (−0.28); order 0.08. Galician role on test0 is the weakest value in the study (0.19 [0.15, 0.22]), and still passes.
- **Pipeline.** "By" at L8, role at L18, handover at L26; in Spanish the handover is L24.
- **Heads.** Top-20 repair .42 against .00. **The head scan started at L35, after the handover**, so its heads are late by construction ([S3 §3](S3-layerwise-and-heads.md#3-how-many-heads-and-where-they-are)). Transfer: .35–.45.
- **Issues.** There is no C7 claim: on development data 59% of wrong items named the agent after the prefix, too close to .5 to claim.
- **Knockout.** 84% of the margin kept, and every answer stays correct, consistent with its head scan starting after the handover.

### Granite 4.2 30B

- **Behaviour.** The mildest failure: failing / working 80% / 99%; BX 85% / 90%. On BX ditransitives there is no drop at all (96 / 96).
- **Role code.** 0.75 [0.64, 0.84] at L37, the deepest role layer (.58). Order/focus 0.24 on core; on BX the order coefficient rises to 0.93 at L52.
- **Pipeline.** "By" L28, role L37, handover L52, the same in all three languages.
- **Heads.** .50 against .01. Transfer: BX .50, Spanish .59, Galician .62, UD .52, Wikipedia .42 and .46.
- **Answer process.** 84% of wrong items name the agent after the prefix (C7). Its C1 effect on test0 is the smallest of the five (0.16 [0.13, 0.20]), and still passes.
- **Knockout.** 49% of the margin kept after ablating the top-20 heads, and 98% of answers stay correct.

---

### Across models

- **Depth varies; order is preserved.** The role and "by" stages always come before the handover. Relative handover depth ranges from .41 (OLMo) to .84 (Qwen3.8).
- **Behavioural severity does not predict mechanism strength.** phi-4 fails worst yet has strong head repair (.75). Granite fails least yet has middling repair (.50).
- **Language shifts are model-specific.** Gemma's handover moves later in Spanish and Galician (+6 layers), Qwen3.8's moves earlier (−4), OLMo's moves earlier in Spanish only (−2), and phi-4's and Granite's do not move ([S5](S5-multilingual.md)).
