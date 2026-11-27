# 70B-class extension (planned 27 Nov 2026; exploratory until frozen)

Two larger open-weight models, each split over the node's 2 x H100 (`--device-map auto`):

| Model | HF revision (downloaded 27 Nov) | Layers | Why |
|---|---|---|---|
| meta-llama/Llama-3.3-70B-Instruct | 6f6073b423013f6a7d4d9f39144961bfbfbc386b | 80 | the standard open 70B; Meta, a lab not yet in the study |
| Qwen/Qwen2.5-72B-Instruct | 495f39366efef23836d0cfae4fbe635880d2be31 | 80 | the other standard open 70B-class dense model; full attention in every layer (our Qwen3.8-27B is a hybrid) |

No larger dense model exists in the five core families (OLMo, Gemma, phi, Granite top out near 30B; larger
Qwen releases are mixture-of-experts). Licences: Llama 3.3 Community Licence; Qwen licence for the 72B
(check the model card). Both allow this research use; cite both.

Procedure, same as the core five:
1. `scripts/slurm/otr_big_smoke.sbatch <model>`: every script on a few items over 2 GPUs (results/smoke/).
2. `scripts/slurm/otr_big_dev_submit.sh --submit`: dev pass (C1, OTR-03 ordinary/failing, OTR-09 with all
   controls, OTR-13, OTR-07); then OTR-14 with `--rank-from` the dev OTR-09 file.
3. Freeze per model with the FREEZE.md section 4 rules (role layer = OTR-03 dev peak on the even grid,
   handover = first dev crossing of .5, "by" layer = OTR-13 dev peak, heads = dev top-20) in an addendum,
   committed before any test0 command.
4. test0 with the frozen values, scored by `otr_freeze_score.py` (FROZEN entries added in the addendum).

Code: `hf_scoring.load_for_run` (device map, 85% of each GPU), all hooks write on the hidden state's own
device; `tests/test_device_map.py` checks the dispatch path gives the same numbers as the plain path.
