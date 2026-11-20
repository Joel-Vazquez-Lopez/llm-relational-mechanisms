[← Order-to-Role](README.md) · [Experiments](EXPERIMENTS.md)

# Reproducibility guide — Order-to-Role

**Repository status:** the current OTR implementation and full test collection are on `order-to-role/stimuli`; development results are principally on `order-to-role/results-dev`. A `main` checkout does not yet include those OTR entrypoints.

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

## 4. Validity checks

Compare matched items across response formats; keep full sentence correctness distinct from first-token preference; validate question grammaticality; avoid treating probes as causal localization; align token positions properly across prompt conditions; and maintain all negative controls. See [evidence caveats](EVIDENCE.md) and the shared [evidence policy](../../docs/EVIDENCE_POLICY.md).
