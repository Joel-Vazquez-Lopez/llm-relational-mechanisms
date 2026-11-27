[← Order-to-Role](README.md) · [Experiments](EXPERIMENTS.md)

# Reproducibility guide — Order-to-Role

**Repository status:** the current OTR implementation and full test collection are on `order-to-role/stimuli`; development and test0 results are principally on `order-to-role/results-dev`. PR #12 imported an earlier results snapshot into `main`; a `main` checkout does not yet include the latest OTR entrypoints.

## 1. Obtain the appropriate code

```bash
git clone https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms.git
cd llm-relational-mechanisms
git fetch origin
git switch --track origin/order-to-role/stimuli
```

If you already have a local branch with this name, check `git status` and its upstream before switching; never overwrite local work.

## 2. Lightweight tests before model inference

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m pytest -q tests/test_order_role_stimuli.py
```

Other OTR tests may require additional model or numerical dependencies; use each script's documented requirements. The command above is **not** a claim that every full-model intervention can run offline.

## 3. Reproducing analyses

Choose a specific [OTR experiment](EXPERIMENTS.md), then record the exact code commit, stimulus-generator version, model/tokenizer revision, chat template, answer extraction, seed, split and decoding/scoring rule.

The [review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) describes figure regeneration and the provenance table for its historical development results. The [results branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results) contains additional run outputs. Avoid selecting a same-named run solely by a glob match when several revisions exist.

No held-out test material should be opened merely to check whether a hypothesis works. Follow the protocol-specific freeze and access rules.

## 4. Frozen evaluation and archival scoring

English test0 is governed by the [November 26 freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md); Spanish/Galician by the [separate XL addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md). Consult the exact scorer version and frozen output manifest. Archived test0 outputs already exist; this guide does not authorize rerunning test jobs or opening test1–3. Do not select new layers or heads from test0, change thresholds after observing results, or conflate the historical development log with frozen results.

For audit, pin [code](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e) and [results](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/25c6914e29473e4b1a1063600c48743f68b705c3/results) separately. Compare all expected runs with the attempt log and report any missing outputs or technical failures. The newer 70B experiments are [exploratory until separately frozen](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md).

## 5. Validity checks

Compare matched items across response formats; keep full sentence correctness distinct from first-token preference; validate question grammaticality; avoid treating probes as causal localization; align token positions properly across prompt conditions; and maintain all negative controls. See [evidence caveats](EVIDENCE.md) and the shared [evidence policy](../../docs/EVIDENCE_POLICY.md).
