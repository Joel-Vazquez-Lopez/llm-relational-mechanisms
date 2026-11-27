# ROLE-DIVERSITY-COMPLEX-v0 — prospective new-seed full training protocol

**BEFORE FREEZE.** Implemented, but no scientific models may train or be scored until source-only and study manifests are committed. This hypothesis was motivated by old results and will be tested prospectively with fresh seeds. Keep old studies frozen and untouched.

## Scientific question

Under the same complex 5-frame × 9 named-adjunct training grammar, does seeing multiple **distinct participant-role assignments** for a familiar four-name quartet make the model generalize to role assignments it never saw? Distinguish diversity (repeated exposure to a second allowed mapping) from the separate one-shot question (exactly one exposure), not tested here.

## Four nested role arms

- roles_1: eligible assignment index 0, the original trained tuple.
- roles_2: eligible role indices 0 and 1 (original plus one cyclic alternative).
- roles_4: eligible 0–3 (all four cyclic rotations).
- roles_8: eligible 0–7 (four rotations plus four reversed rotations).

All role sets are nested. The *other 16* of the 24 possible assignments per quartet remain unseen by ALL four arms. Evaluation also includes all 24 assignments of the 60 wholly withheld four-name quartets (all from original TRAIN identifiers, never original DEV/reserved/protected). No within-event LI surfaces.

Training uses the same architecture/vocabulary as the earlier TRAIN-SURFACE-FACTORIAL-v0 (three causal layers, width 96, four heads, 48 token positions), same 240 training quartets, five neutral-prefix choices, nine bare/L/I modifier-pair cells (45 total), AP+PA voices, 6400 AdamW steps, learning rate 0.002, weight decay 0.01, eight quartets/step, four query labels/source, 64 labels/step/model. Within each seed initial model parameters and the exact quartet, voice, prefix and modifier schedule are matched across arms. Only role mapping changes.

## New, independent 16-seed cohort

Exactly: 16101, 16103, 16107, 16109, 16113, 16119, 16121, 16127, 16131, 16133, 16139, 16143, 16149, 16157, 16161, 16167.

No optional stopping, failed-seed replacement or checkpoint selection. All 16 seeds and four arms (64 new scratch models) must finish 6400 updates and be reported.

## Exact exposure balancing

For each seed, each of the 240 quartets gets a deterministic shuffled order of the 45 possible frame×adjunct cells. At its zero-indexed appearance number k in the training schedule, that quartet is presented using surface cell index k modulo 45 and each arm roles_n is presented the eligible training role index k modulo n. Because 45 and each n in {1,2,4,8} are coprime, the frequencies of every allowable (role index, surface cell) combination within a given quartet differ by at most 1. The source-only full schedule audit checks this for **every** quartet, seed and arm and saves per-seed schedule hashes before training. Role assignments may change the lexical-name positions and unique input counts; those are intentional distribution differences, not controlled mediators. Sequence length and non-name token bags match across role assignments within the same cell.

## Frozen outcome hierarchy

Primary: paired seed-wise mean (roles_8 minus roles_1) of strict native four-question joint correct, C_query, on all 16 never-trained role permutations for each of the 240 train quartets. Same deterministic **rich** prefix+named-modifier test source for all arms, AP+PA averaged. Rich source cells: with k=quartet_id×16+role_ordinal, frame=1+(k modulo 4), modifier=1+((k//4) modulo 8). Underlying source labels are independently parsed and constant across arms.

**Primary PASS requires ALL:** two-sided seed-level Student t 95% CI lower >0 (df=15); average effect at least +0.10 absolute (+10 pp); positive differences in at least 12/16 independent new seeds.

**Gated secondary:** only if primary PASSES, paired roles_2 minus roles_1 on the exact same never-trained rich-role items, CI lower >0 and mean at least +0.10. This tests benefit of one extra **repeatedly trained distinct mapping**, not one-time exposure. If primary fails, this comparison is descriptive only.

Exploratory: roles_4 dose shape; all bare tests and completely withheld quartets; original trained core bare/rich competence; accuracy on every permitted TRAIN-role assignment on one deterministic rich cell; event-pair error taxonomy; per-question/full-vocab accuracy; potential bimodality and anchor-style errors. No post-hoc promotion to confirmatory hypotheses.

Inferential unit: seed (N=16), not correlated arms or thousands of test examples. Report all results even when unfavorable.

## Pretraining freeze, safeguards and interpretive limits

- First run original source-only preflight and prospective model-code prepare/smoke; CPU CI is a no-new-scientific-model check.
- Commit **both** generated source-only and study manifests to branch along with all source, launcher and protocol scripts **BEFORE ANY** new-seed inference.
- Full runner checks Git HEAD byte equality for all frozen inputs and frozen dependency/schedule/protocol/vocabulary hashes; refuses a partial or existing result/weight.
- All original DEV, reserved, protected, and within-event LI remain excluded; only old TRAIN-derived quartets and their role variants permitted.
- Initial 149xx checkpoints never loaded, changed or overwritten; new 161xx scientific .pt checkpoints stay local.
- Any successful result shows a causal effect of the **bundled role-diversity training distribution** within this one synthetic grammar. It does not isolate a unique internal mechanism, prove pretrained LLM generality or show that each unseen role assignment must be trained.

Check independent source and numerical audits after all new-seed reports are sealed.
