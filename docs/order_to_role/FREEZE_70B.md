# FREEZE addendum — two 70B-class models on test set 0 (28 Nov 2026)

Status: FROZEN by the commit that adds this file, together with scorer v1.2.0 (`scripts/otr_freeze_score.py`,
`FROZEN_70B`) and `scripts/slurm/otr_70b_test0_submit.sh`. Written after the dev pass of both models and before any
70B command on a test split. FREEZE.md and FREEZE_XL.md are unchanged; this addendum adds two models to the same
claims and pass rules. After the commit, sections 2–4 do not change; anything else is labelled exploratory.

## 0. Prior access to held-out data
- No 70B result file with a test split exists (checked 28 Nov: 0 files matching `test[0-3]` for either model).
- Disclosure: all core, BX and es/gl test0 results are known (FREEZE.md, FREEZE_XL.md). The values below come only
  from the 70B dev runs (section 6); the rules for turning dev runs into frozen values are those of FREEZE.md §4.
- Disclosure: the dev pass ran at code `9c43792` (order-to-role/stimuli). Test0 runs at this commit on `main`, which
  also contains `d2952be` ("one chat-template path for scoring and patching"). With no system prompt that path makes
  the same single-user-turn call as before (stated in `chat.apply_template`), so prompts are identical.

## 1. Scope
- Models, split over 2 x H100 with `--device-map auto` (`docs/order_to_role/BIG_MODELS.md`):

| Model | HF revision | Layers |
|---|---|---|
| meta-llama/Llama-3.3-70B-Instruct | 6f6073b423013f6a7d4d9f39144961bfbfbc386b | 80 |
| Qwen/Qwen2.5-72B-Instruct | 495f39366efef23836d0cfae4fbe635880d2be31 | 80 |

- Runtime, prompts, instruction ("Reply with one name only."), chat template, bf16 and greedy decoding as FREEZE.md §1.
  Family partition as FREEZE.md (test0 = 59 core families; BX test0 as in FREEZE.md).
- Claims: C1–C8 with the definitions and pass rules of FREEZE.md §2–3, C7 claimed for both models (both passed it on
  dev). Replication: BX C2, C3, C5 as in FREEZE.md (reported separately).
- These two models are reported as a separate block ("70B addendum"), not merged into the core 37/38.

## 2. Frozen per-model settings (dev only; never re-chosen)

| Model | handover (first dev crossing of .5) | role layer (OTR-03 dev peak, even grid) | "by" layer (OTR-13 dev peak of "by" reverse damage) | dev OTR-09 file (`--rank-from`) |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct | 38 | 34 | 14 | otr09_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T121453.json |
| Qwen2.5-72B-Instruct | 58 | 48 | 30 | otr09_Qwen__Qwen2.5-72B-Instruct_dev_20261127T121538.json |

Frozen top-20 heads (dev OTR-09 `head_scan.top20`, in this order; layer.head):
- Llama-3.3-70B: 74.9, 55.20, 77.47, 52.19, 75.26, 72.39, 75.30, 77.42, 64.27, 71.1, 73.62, 71.6, 79.16, 49.2, 53.1,
  55.17, 66.7, 47.17, 54.15, 72.37
- Qwen2.5-72B: 63.52, 60.40, 61.47, 60.30, 75.13, 50.5, 60.42, 60.26, 63.53, 71.28, 76.32, 74.32, 55.55, 74.34,
  59.8, 74.35, 78.58, 53.30, 63.55, 64.10

## 3. Commands
`bash scripts/slurm/otr_70b_test0_submit.sh` prints them; `--submit` runs them (20 jobs, 2 x H100 each, 4 h limit) and
logs every job ID in `docs/order_to_role/test0_attempts.md`. They are the FREEZE.md §5 commands with
`--device-map auto` added; C4/C5 also passes `--controls-extra` (extra controls reported, not scored).
Scoring: `python scripts/otr_freeze_score.py --results results --split test0 --set 70b` (then `--stimuli bx-transitive`).

## 4. What counts as what
- Every claim is reported for both models, pass or fail, with its CI. Technical failures as FREEZE.md §6.
- Expectations from dev, written down so a miss is visible (not pass rules):
  - **Llama C4 is fragile.** Dev repair jumps from .35 (layer 36) to .59 (38), then stays between .59 and .66 up to
    layer 66 (1.0 only at 79). If the plateau falls below .5 on test0, the first crossing moves far from 38 and C4
    fails although the jump is still at 38. The pass rule is not changed for this.
  - Qwen2.5-72B dev curve: .25 (54), .46 (56), .60 (58), .88 (64): a crossing at 56 or 58 is within the rule.
  - Dev passes for every claim except C5, which had no transfer block on dev (fresh top-20 on held-out dev items:
    .47 Llama, .44 Qwen2.5-72B; random .00; other item's failing run −.22 / −.30).
  - Dev C1: failing .69 / .71, working 1.00. Dev C7: .83 [.75, .90] / .99 [.97, 1.00].

## 5. Code used for the 70B runs
`hf_scoring.load_for_run` with `--device-map auto` (each GPU filled to at most 85%); all hooks write on the hidden
state's own device. `tests/test_device_map.py` checks that the dispatch path gives the same numbers as the plain path.
The smoke job of 27 Nov (`otr_big_smoke.sbatch`) passed five of six scripts and timed out while loading the model for
the sixth (OTR-07); OTR-07 then ran in the dev pass.

## 6. Dev files this addendum is frozen from (sha256; branch paper2/70b-dev-results @ 56b7651)
```
4f724add9668f651afe689d66983d9fe30add1c166f44fb310799d60fc018f48  otr03_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T142356.json
761fb0db76b68d59b31bf6a6ff1b6065ef84267c1b023ba8ee73b2d55970ee56  otr03_meta-llama__Llama-3.3-70B-Instruct_dev_aq_20261127T133736.json
3c5e4d46493061f67f60ed207cd0f39e44c0d99294ce6503142ecb72d910c392  otr03_Qwen__Qwen2.5-72B-Instruct_dev_20261128T061954.json
2881e0ae22a221b57839c6b0bb7d899eb010e33a6d67041f585c9a87ac8faa0a  otr03_Qwen__Qwen2.5-72B-Instruct_dev_aq_20261128T073322.json
be7eb4ded52cf4b1c6b29ae12f75cb75f5dde23360956e88bbdd9c18b5c15b83  otr07_meta-llama__Llama-3.3-70B-Instruct_dev_instr_byctl_20261128T043008.json
190d74d29c156772d2007607530328d1cd05dc4d8f251287d1bc0b5831acb5e7  otr07_Qwen__Qwen2.5-72B-Instruct_dev_instr_byctl_20261128T103232.json
7a35e9297427145edc3c606c2003c5f7bddde56e0ed8f3eb115b02ccdb90d7d6  otr09_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T121453.json
395708088197a7f813858a33c9b5badb6aa689ad8ea7e04c85389e910a6188b7  otr09_Qwen__Qwen2.5-72B-Instruct_dev_20261127T121538.json
1fd3464a9c11c8edda7194553253ab7db97de3711e678d7b0314e2a0b375b715  otr13_meta-llama__Llama-3.3-70B-Instruct_dev_20261128T033437.json
674813557b3c463c1bf3f0ec1088ba3e6db73d9b9f6047fd98ea8cd4870af924  otr13_Qwen__Qwen2.5-72B-Instruct_dev_20261128T093203.json
faef2e63817b340144ca26d5e445f7cbf64449bcc83a49c001f170035c965e49  otr14_meta-llama__Llama-3.3-70B-Instruct_dev_20261128T110310.json
823c23da3700088ba5936b76d4d84cbe4908f8aaaf2415f362b0d8685db69894  otr14_Qwen__Qwen2.5-72B-Instruct_dev_20261128T113135.json
1ddff69236073deee61d101c8efba3a0c605e1443c951f21349485bee13bc082  otrgen_meta-llama__Llama-3.3-70B-Instruct_dev_q-who-was-someone-past-by_20261128T035334.json
fa759a49d1838bd453a9bab658d73e7d1814d4bb5d44cf88c32219102e10f925  otrgen_meta-llama__Llama-3.3-70B-Instruct_dev_q-who-was-the-ing-done-by_20261128T041120.json
2142dab0921a00499d8eb0d1565acff5b05a0f7e16741fff39fbd177ee4d3b10  otrgen_Qwen__Qwen2.5-72B-Instruct_dev_q-who-was-someone-past-by_20261128T095222.json
81913fddea3178539f5aa6cde03025a3e6e54a030beab01de37c32537368e7d9  otrgen_Qwen__Qwen2.5-72B-Instruct_dev_q-who-was-the-ing-done-by_20261128T101202.json
```
