[← Order-to-Role](README.md) · [Results](RESULTS.md) · [Evidence ledger](EVIDENCE.md)

# Manuscript map

A guide from the working manuscript to the evidence on these pages. The manuscript itself is the author's writing workspace. The last archived version is the [24 November draft (25 November state snapshot)](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/paper_draft.md). This page does not change the frozen claims or the manuscript.

**Target:** ACL Rolling Review (long paper: 8 pages, plus limitations and appendix).

## Sections, evidence and figures

| Section | Argument | Evidence | Figure on these pages | Confirmatory |
|:--|:--|:--|:--|:--|
| 1 Introduction | Order, function and role usually travel together; which does a model use, and how does a question reach it? | Design | – | – |
| 3 Paradigm and methods | Decorrelated constructions, causal transplants, answer and token patching, freeze | [Methods](METHODS.md) | – | – |
| 4 What the names encode | Role is the largest transplant effect, mid-network; psych verbs | OTR-03 core, BX, psych | [Fig. 2](results/figures/fig2_role_code_by_layer.svg) | C2 |
| 5 Development across layers | Role and "by" before the handover, at model-specific depths | OTR-03, 13, 09, 14 | [Fig. 3](results/figures/fig3_question_to_answer.svg), [Fig. 4](results/figures/fig4_layer_order.svg) | C4 |
| 6 How a question reads the roles | 20 heads at the handover; shared across inputs | OTR-09 | [Fig. 5](results/figures/fig5_head_transfer.svg) | C5 |
| 7 When retrieval fails | Identical stored states, different answers; "by"; the model still knows | Generation, OTR-03 failing, OTR-13/14, OTR-07 | [Fig. 1](results/figures/fig1_failing_question.svg) | C1, C3, C6, C7, C8 |
| 8 Generality | Spanish, Galician, UD, Wikipedia; real vs neutral names | OTR-03, OTR-09 transfer | [Fig. 5](results/figures/fig5_head_transfer.svg) | BX, XL-C1–C5b |
| 9 Held-out confirmation | All frozen claims, pass or fail | Frozen scorer | [Scorecards](EVIDENCE.md#1-confirmatory-results-test-set-0) | all |
| 10 Discussion, limitations | Storage vs retrieval; sufficiency vs a unique pathway; measurement lessons | [Ledger](EVIDENCE.md) | – | – |

## What changed since the 24 November draft

- **Section 9's empty table can now be filled** from the [scorecards](EVIDENCE.md#1-confirmatory-results-test-set-0): English 37/38 (phi-4 C5 fails on a control), BX 15/15, Spanish 29/30 (Gemma XL-C3 fails), Galician 30/30.
- **Section 4's phrase "order and function coefficients are near zero or small" needs a correction.** Function is near zero, but order/focus is 0.45 in Gemma, 0.38 in phi-4 and 0.24 in Granite at the role layer (0.61 for Granite on BX). The supported claim is that role is the largest coefficient and exceeds function.
- **The Qwen3.8 WIKI-ROLES v0.2 values are now available** (frozen-head repair .33 real, .35 neutral).
- **Figures 2–5 exist** as development figures generated from the archived files; the manuscript versions still need family-bootstrap bands for the transplant curves.

## Open writing questions (from the draft)

Final title; verify every citation; how to define "good / middle / poor" models for the layer figure; whether to include a sentence on implications for training or prompting.
