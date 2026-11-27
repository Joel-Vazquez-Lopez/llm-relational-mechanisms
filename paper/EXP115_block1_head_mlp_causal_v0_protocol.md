# EXP-115 — Block 1 attention-head / MLP residual contribution interventions

**Status:** Fixed exploratory v0 specification. We reuse EXP-114’s 48 trained final checkpoints and its **already scored** 144 new-development AP/PA pairs. These are **not** another independent confirmation or a new test dataset. No training, no L+I/protected evaluation, no open-ended choice of heads/seeds/arms. This analysis seeks *causal influence of residual writes*, not a unique interpretable circuit or a role-specific head.

## Question and motivation

EXP-114 independently replicated (i) rescue from input passive-clause position-code editing, (ii) an advantage of transferring Block 1’s learned update over late direct position editing, and (iii) reverse disruption of familiar PA. Its controls also showed that within-clause reversed position-code transfer rescues similarly and that wrong-event responses can rise. Thus the next exploratory question is **which part of the first block's learned residual update is causally influential downstream**.

The three-layer width-96 four-head causal Transformer implements pre-LN self-attention and then an MLP. For source token `i` at Block 1, exactly:

`B_i = E(token_i) + P(i)`

`A_i = attention_output_bias + sum_{h=0}^3 W_o,h @ (softmax(Q_h K_h^T / sqrt(d)) V_h)_i`

`M_i = MLP(LN2(B_i + A_i))`

`h1_i = B_i + A_i + M_i` and `U_i = A_i + M_i`.

For matched lexical source clauses, **after Block 1**, patch one residual-write component at recipient source tokens with the donor counterpart from **the same model**, keeping other recipient components unchanged; propagate through Blocks 2–3 to unchanged question/answer tokens. The fixed attention output bias is identical across donor/recipient, so it cancels from the patch difference.

### Full factorial of fixed modes

1. `passive_head0`–`passive_head3`: separately transfer each head's post-output-projection residual contribution (all four explicitly reported, no best-head selection).
2. `passive_attention_all`: transfer sum of all four projected attention-head contributions.
3. `passive_mlp`: transfer first-block MLP residual write alone.
4. `passive_update_all`: transfer all attention-head and MLP residual writes (identical to EXP-114 `late_update_passive` by algebra; verify numerically).
5. `passive_names_update_all`: transfer full first-block update **only at the two participant-name tokens** of the passive clause.
6. `passive_reverse_update_all`: transfer full first-block update with donor token-state ordering reversed within the passive clause; scrambled lexical identity is an imperfect off-manifold control.
7. `active_head0`–`active_head3`, `active_attention_all`, `active_mlp`, `active_update_all`: corresponding donor-transfer controls from the active clause.

All modes run **both AP←PA and PA←AP**, all four query types, all three training arms, eight training seeds × two numerical MPS repeats = 48 model-state analyses. No filtering by baseline competence, no selecting the best seed or head, no sample exclusions.

### Measurements and analysis

- Native candidate-restricted individual-query accuracy, strict four-query `C_query`, correct-role logit margin, wrong-role rate, wrong-event rate. Per-intervention change from exactly the same recipient model and sentence.
- Focus: AP second-agent + second-patient equally weighted and reverse PA first-agent + first-patient equally weighted. First-event AP collateral effects remain recorded.
- Average the two numerical repeats **within each seed**, then descriptive means and 95% Student-t intervals across eight independent seeds (df=7). Intervals are exploratory and unadjusted for the head/mode search.
- Report every head and modality, plus paired seed-level contrasts: sum of isolated head accuracy effects versus jointly patched attention, MLP versus attention, passive versus active full updates, passive all-token versus names-only, aligned versus reversed full updates.
- Because accuracy and downstream model computations are nonlinear, isolated-head accuracy effects **need not add**. They must not be interpreted as additive causal shares.

### Critical implementation checks

- Reconstruct only EXP-114’s previous 144 evaluation source pairs, validating hashes against the exact frozen manifest. Never create reserved or L+I examples.
- Load EXP-114 checkpoint locally from `results/exp114/independent_replication_v0/seed{seed}_repeat{rep}_mps_step6400.pt`, verify metadata and exact native EXP-114 accuracy. **Do not train or modify weights.** Check report hash, checkpoint hash and EXP-114 protocol digest.
- Manually decompose `torch.nn.MultiheadAttention` using saved packed Q/K/V projections, standard causal mask, softmax, per-head projected outputs, out-projection bias, and Block 1 MLP. Require decomposed Block 1 states and complete native logits to match `model.forward` within fixed numerical tolerance.
- Confirm EXP-115 `passive_update_all` matches EXP-114's `late_update_passive` in accuracy, margins, wrong-role and wrong-event rates for both directions and every model.
- Query suffix and answer residual states are **never directly modified**. Donor's source states come from its corresponding AP/PA same-event clause. No source texts, logits, hidden states, or protected examples written to JSON.
- Write one immutable JSON per `(seed, repeat, arm)`, refuse overwrites, skip complete existing files in the launcher, require a full 48/48 before aggregate. Manifest and per-run script SHA prevent mixing versions. Freeze code+protocol in Git before looking at results.

### Interpretation boundaries

A head-specific replacement changing output would imply **that head's residual write can causally influence subsequent predictions under this transplantation intervention**, not that this head alone computes semantic roles. Donor head outputs contain positional, lexical, context and attention-dependent information. MLP contributions are also conditioned on attention, position and context; patched post-block head/MLP writes may create off-manifold states. The source clauses differ in **absolute position and prior context**, so these interventions do not isolate abstract role variables. Large wrong-event increases or comparable active/scrambled effects would weaken grammatical specificity.

No new seeds, evaluation family or external preregistration are introduced; all effects are **exploratory** on an already unsealed, previously explored dataset. Any newly selected head or mechanism must be independently frozen and tested in a separate future study. Existing protected L+I gate remains closed.

## Commands

```bash
cd ~/Developer/exp102-clean-design
source ~/Developer/llm-relational-mechanisms/.venv/bin/activate
unzip -n ~/Downloads/exp115_block1_head_mlp_causal_v0_bundle.zip -d .
python3 scripts/exp115_block1_head_mlp_causal_v0.py --phase prepare
python3 scripts/exp115_block1_head_mlp_causal_v0.py --phase smoke --device mps
# Commit script, launcher, protocol, and EXP115 manifest before viewing scientific outcomes.
bash scripts/run_exp115_all_v0.sh
```

The command can run under `caffeinate -dimsu` on macOS with the laptop on AC power, lid open, and sufficient disk space. Check process completion and summary next morning. If interrupted, rerun the same launcher to skip completed JSON reports. Do not resume across code changes without a new versioned experiment.
