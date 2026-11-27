[← Integration audit](INTEGRATION.md) · [Paper I](README.md) · [Reproducibility](REPRODUCE.md)

# Branch disposition and consolidation register

**27 November 2026 · Draft for coordinating review.** This extends the prior tree inventory; it is not a new scientific phase, a release approval, or an authorisation to retire branches. Main at audit: `da7d463b10955f02cf342205e5e0a03f450cfcb1`. #14: `910c320a3696be4a4807ef554602adff3965d091`. #22: `0bb6c18af1e2f9f4371335ddc866c93a8f917e9a`. Branch census: 57 remote branch names, including shared/Paper II refs. Original scientific paths and the #18 reading order remain unchanged.

## Integration gates

- [#14](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/pull/14) retains 1,479 additions; ordinary Python 3.10/3.12 CI passed. [Isolated audit](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/actions/runs/46067016850) failed its final scientific gate: 38 Python and 27 shell files syntax-clean; one unresolved import and one named hash failure, both the missing EXP107 script. 27 manifest protocol digests passed; 131 named path checks, one failed; 185 additional hashes matched tracked bytes. 1,356 reports (1,269 JSON, 86 CSV, 1 Markdown; 128,452 CSV rows) parsed, no report-manifest digest mismatch or flagged raw text/scope field. Existing 60 unit tests, 11 toy CPU smoke checks, five central result recomputations and construction-shift figure checks passed. Automated screening is not publication clearance.
- Required original `scripts/exp107_counterfactual_diagnostic_v0.py` SHA-256: `e1916dc30f62500d4871e790a4e8c2efa1a827c7a6094a88210e5d190498c23d`. No original bytes recovered. Search original local checkout and original distribution ZIP; do not reconstruct or change frozen manifest. EXP107/108 original reports and source artifacts are also not visible at the three baseline trees; inventory local archives separately. The eight unresolved evaluation digest fields (below) are not automatically missing files; classify digest semantics and verify through existing manifests/approved archived material, without generating or scoring protected examples. 122 unmatched checkpoint/tensor references require archive inventory and exact hash/provenance verification. Rights review outstanding. No #14 merge until all gates satisfied and coordinating approval.
- [#22](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/pull/22), [#17](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/issues/17): 17 original files (516,047 bytes), all blob SHAs/modes preserved. All 16 reports' metadata and 1,864 seed-level plus 932 aggregate checks matched (max difference 2.22e-16). Both Python versions passed. Parent manifest and implementation in #14. Remain draft; merge only after #14, clearance, independent coordinating review and current-main recheck.
- Read-only comparison with Paper II #19 (`0b323dafa738070c88ccb90d3e260ec5c8022a3c`), #20 (`3c2fee2977b03c524d6a2eda996fc02a09a1bbe5`) and #21 (`bc82767ce3feb9c63532b79de9afd702bdc101b7`) found no overlap with the 1,479 #14 and 17 #22 added paths. Recheck before any future write/merge. Their integration is independently prioritised.

### Outstanding evaluation digest fields

| Manifest | Field |
|---|---|
| reports/exp109/replication_v0/manifest.json | evaluation_audit.evaluation_sha256 |
| reports/exp114/independent_replication_v0/manifest.json | evaluation_audit.evaluation_sha256 |
| reports/exp115/block1_head_mlp_causal_v0/manifest.json | exp114_evaluation_sha256 |
| reports/phase2_01/coverage_curve_v0/manifest.json | evaluation_audit.evaluation_sha256 |
| reports/phase2_02/structural_transfer_v0/manifest.json | evaluation_audit.evaluation_sha256 |
| reports/phase2_03/distractor_event_binding_v0/manifest.json | evaluation_audit.evaluation_sha256 |
| reports/position_cb_v0/manifest.json | evaluation_audit.evaluation_sha256; protocol.evaluation_sha256 |

## Residual assets

30 distinct absent paths from the previously inventoried Paper I source snapshots after accounting for main, #14 and #22. This is a path inventory, not a claim that all local untracked work is visible; same-path historical variants and unavailable external artifacts are separate. Shared #15 adds six programme-wide documentation files and is excluded. Source links identify exact commits; no Git tags have yet been established.

### Deferred historical workflows (9)

Sources: [experiments/role-learning-origins-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702) · [analysis/role-anchor-probe-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/10ab4c7bf2b56d3e8c8e396eee22e2e732a41910).

Original YAMLs, not authorised for activation. All specify historical branch-scoped push triggers plus workflow_dispatch, contents: read, Ubuntu, Python 3.12. Static checks: position audit/core. Standard-library baseline: 12-minute timeout. Core-diversity source preflight: source-only. Runner/core-diversity, role-diversity, cross-eval, surface-factorial and role-anchor smoke use CPU Torch; several prepare manifests or encode held-out inputs, so review write/freeze behavior, scope and cost before enabling. No observed GPU job or actual scientific training command in these YAMLs; cost not measured. No workflow run requested.

- `.github/workflows/core-diversity-v1-preflight.yml`
- `.github/workflows/core-diversity-v1-runner-smoke.yml`
- `.github/workflows/paper-shortcut-baselines.yml`
- `.github/workflows/position-cb-audit-static.yml`
- `.github/workflows/position-cb-core-static.yml`
- `.github/workflows/role-diversity-complex-v0.yml`
- `.github/workflows/train-surface-cross-eval-v0.yml`
- `.github/workflows/train-surface-factorial-v0.yml`
- `.github/workflows/role-anchor-probe-v0.yml`

### Competing construction-shift additions (7)

Sources: [paper/construction-sensitive-roles-20261121](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/a261bd847b4119dcc10b26a0f9890c22f17780b5).

Archive all together; review the roadmap, analyses and figure against the selected manuscript version and #22 before selective integration. Four same-path conflicts: README.md, evidence_ledger.md, manuscript.md, scope_and_decisions.md under papers/learning-relations/construction-shift/. Do not overwrite any.

- `papers/learning-relations/construction-shift/PUBLICATION_ROADMAP_2026_11.md`
- `papers/learning-relations/construction-shift/analysis/core_diversity_v1_audit.md`
- `papers/learning-relations/construction-shift/analysis/position_cb_core_transfer_v0.md`
- `papers/learning-relations/construction-shift/analysis/position_cb_train_dev_audit_v0.md`
- `papers/learning-relations/construction-shift/analysis/position_cb_v0_audit.md`
- `papers/learning-relations/construction-shift/analysis/train_surface_factorial_v0_audit.md`
- `papers/learning-relations/construction-shift/figures/fig5_position_counterbalance.svg`

### EXP101 historical diagnostics (4)

Sources: [analysis/exp101-familiar-readonly-rc7](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/014c6fe75df2339d37187e4ef70b0878d07d1f23) · [protocol/exp101-mps-same-origin-check](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/3665871e95f28a7471380bda136bca9f1a398e86) · [protocol/exp101-mps-resume-diagnostic](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/5b7aba0c488e10792219287ea97e234c345784ae).

Preserve distinct historical source versions; require frozen fixture/manifest compatibility and explicit scientific decision before integration; do not run diagnostic.

- `scripts/exp101_familiar_readonly_diagnostic.py`
- `paper/exp101_mps_resume_diagnostic_note.md`
- `scripts/exp101_mps_resume_diagnostic.py`
- `scripts/exp101_mps_same_origin_diagnostic.py`

### ML090 correction and ML097 design (4)

Sources: [audit/ml090-correction-ml097-prospective](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/dea02af9ae15d3265512b6bcff854b1ab24d993b).

Already in draft #5, do not duplicate. Original ML090 role-evidence metrics withdrawn; corrected analysis not executed. ML097 design unexecuted, not frozen. No new experiments authorised.

- `reports/ml090/CORRECTION-001.md`
- `reports/ml097/PROTOCOL_DRAFT.md`
- `scripts/ml090_role_cue_path_v2_corrected.py`
- `tests/test_ml090_indexing.py`

### Historical ML097 calibration archive (4)

Sources: [experiments/ml097-p0-archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4e90d91832806723ac166a4eb71921c596a92813).

Exploratory reanalysis of existing ML088 cohort and ML091 calibration data, not independent replication or protected evaluation. Plan explicitly records item-level logits and predictions; raw observation JSON is ~5.1 MB. Plan and summary inspected; raw file retrieval was incomplete. Requires data/publication clearance, dependency hashes and summary parity before deciding whether canonical or external. Do not rerun.

- `reports/paper_role_consistency/calibration_observations.json`
- `reports/paper_role_consistency/frozen_plan.json`
- `reports/paper_role_consistency/summary.json`
- `scripts/paper_role_consistency_audit.py`

### Isolated validation harness (2)

Sources: [audit/paper-i-pr14-clean-checkout-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e651f89711a6b6aa729658fad74702ff6085ec9a).

Preserve exact audit harness and run evidence; not part of original research outputs; no automatic workflow integration.

- `.github/workflows/paper-i-pr14-audit.yml`
- `audit/paper_i_pr14_audit.py`

Historical same-path variants are preserved in the exact branch snapshots below: EXP101 release-candidate protocols, manifests, hardware fixtures, generators, release/runtime scripts and tests; earlier root/investigation navigation and .gitignore versions; four construction-shift conflicting files. Do not copy obsolete versions over canonical paths merely to eliminate branch names.

## Branch disposition manifest

Each source link is an independently retrievable GitHub tree at the exact current SHA. This is **not** a durable archive certificate: the tag namespace returned 0 references at this check. Archive target for non-active Paper I research refs is a reviewed annotated tag `archive/paper-i/<original-branch-name>` pointing to its exact commit, or equivalent independently verified archive; targets below are planned, not created. No branch is yet a verified retirement candidate. Shared and Paper II branches require their own coordinator. Do not delete any branch.

| Branch / exact source commit | Disposition | Canonical destination or blocker |
|---|---|---|
| [analysis/exp101-familiar-readonly-rc7](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/014c6fe75df2339d37187e4ef70b0878d07d1f23)<br>`014c6fe75df2339d37187e4ef70b0878d07d1f23` | Defer | One unique diagnostic, frozen fixtures and evaluation scope unresolved; archive original; no execution. |
| [analysis/position-cb-core-transfer-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/378342b82acdf5523299d212fa7d6fea24467678)<br>`378342b82acdf5523299d212fa7d6fea24467678` | Integrate via #14 + #22 | Manifest and runner in #14; 17 missing reports in #22; workflow archived pending CI review. |
| [analysis/position-cb-train-dev-audit](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/bbd1200835ecab5b1edef1214ce5e965d8ff9072)<br>`bbd1200835ecab5b1edef1214ce5e965d8ff9072` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-anchor-probe-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/10ab4c7bf2b56d3e8c8e396eee22e2e732a41910)<br>`10ab4c7bf2b56d3e8c8e396eee22e2e732a41910` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-binding-causal-validation-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/b1414f11ea5c218be7d3eabc30c1e767dec1f12a)<br>`b1414f11ea5c218be7d3eabc30c1e767dec1f12a` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-binding-localisation-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/7b339a816a87099560390e4a1f37d82ab08d8a35)<br>`7b339a816a87099560390e4a1f37d82ab08d8a35` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-causal-pathways-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/cc034ffa7b70cb693215b41e45a031a19695f9b0)<br>`cc034ffa7b70cb693215b41e45a031a19695f9b0` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-causal-routing-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/f667fd9a6b7c8b4dab5c97dd13879a9171f831dd)<br>`f667fd9a6b7c8b4dab5c97dd13879a9171f831dd` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-circuit-diagnostics-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/b780f381d0679a053127c7a65caa46343f0a0c98)<br>`b780f381d0679a053127c7a65caa46343f0a0c98` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-mechanism-discovery-v1](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/646be52ee330fa668947e02f63aec03752490651)<br>`646be52ee330fa668947e02f63aec03752490651` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/role-representation-localisation-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/44a8de194f17b06db55d016f84ce137ca5736940)<br>`44a8de194f17b06db55d016f84ce137ca5736940` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [analysis/train-surface-cross-eval-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/c207ea8b96ea7ab78a6c0de73a4de3afcddfa45d)<br>`c207ea8b96ea7ab78a6c0de73a4de3afcddfa45d` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [audit/ml090-correction-ml097-prospective](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/dea02af9ae15d3265512b6bcff854b1ab24d993b)<br>`dea02af9ae15d3265512b6bcff854b1ab24d993b` | Defer | Existing draft #5, four new files, withdrawn historical role-evidence claims; checkpoint parity unresolved; no prospective run. |
| [audit/paper-i-pr14-clean-checkout-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e651f89711a6b6aa729658fad74702ff6085ec9a)<br>`e651f89711a6b6aa729658fad74702ff6085ec9a` | Preserve externally (planned) | Audit script + audit workflow and run 46067016850; preserve exact snapshot, review CI separately. |
| [codex/otr-rescue-results](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/6d241ea697f890500ce9fa3019a2f00b5377d892)<br>`6d241ea697f890500ce9fa3019a2f00b5377d892` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [codex/research-front-page-refresh](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/7914fdde39780d71fa8bc1505ac0223554ec4113)<br>`7914fdde39780d71fa8bc1505ac0223554ec4113` | Preserve externally (planned; shared) | No absent paths in union; historical navigation variants remain; coordinator owns shared retirement decisions. |
| [codex/studies-index](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/5af981be3f5d86dd91ae3621f69e2ada275605e3)<br>`5af981be3f5d86dd91ae3621f69e2ada275605e3` | Preserve externally (planned; shared) | No absent paths in union; historical navigation variants remain; coordinator owns shared retirement decisions. |
| [design/position-cb-core-diversity-v1](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/ae63e94e4da9f48fad00194010ab6adfb1656702)<br>`ae63e94e4da9f48fad00194010ab6adfb1656702` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [design/role-diversity-complex-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/21c5e8d374fdaccbeedb40eb4b5bfee467dec56b)<br>`21c5e8d374fdaccbeedb40eb4b5bfee467dec56b` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [docs/paper-i-results-and-provenance](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/2db525a8f58dfc9fb4e86a03c27b908acc84ea68)<br>`2db525a8f58dfc9fb4e86a03c27b908acc84ea68` | Preserve externally (planned) | Merged documentation #13/#16/#18 respectively; historical version only; no scientific assets to import. Conditional retirement after archive verification. |
| [docs/paper-i-scientific-supplement](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e210de277ca79877a76ef1385875698324d8e7ab)<br>`e210de277ca79877a76ef1385875698324d8e7ab` | Preserve externally (planned) | Merged documentation #13/#16/#18 respectively; historical version only; no scientific assets to import. Conditional retirement after archive verification. |
| [docs/paper-i-start-here-reading-path](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/256179864d1786a53bb495c8b64c479322cac7a0)<br>`256179864d1786a53bb495c8b64c479322cac7a0` | Preserve externally (planned) | Merged documentation #13/#16/#18 respectively; historical version only; no scientific assets to import. Conditional retirement after archive verification. |
| [docs/supplementary-materials](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/5f75ac97225187106dbc71937f35fc6ee8f978b4)<br>`5f75ac97225187106dbc71937f35fc6ee8f978b4` | Defer (shared) | Open #15; programme index and communications, not Paper I data. |
| [docs/two-paper-research-navigation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/90b92b9b2e48cb2dbe79507bfeb7219c0ce420d3)<br>`90b92b9b2e48cb2dbe79507bfeb7219c0ce420d3` | Preserve externally (planned; shared) | No absent paths in union; historical navigation variants remain; coordinator owns shared retirement decisions. |
| [experiment/relational-interactions-001](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/0cc54aa02c517c52452bedc78703753652595f43)<br>`0cc54aa02c517c52452bedc78703753652595f43` | Preserve externally (planned) | Prior tree audit found no absent paths; retain historical context and variants. Conditional retirement only after durable archive verification. |
| [experiments/exp102-clean-design](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/df82ab3ad9db5e13db80cf47caf8f2142afa3dc7)<br>`df82ab3ad9db5e13db80cf47caf8f2142afa3dc7` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/ml097-p0-archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4e90d91832806723ac166a4eb71921c596a92813)<br>`4e90d91832806723ac166a4eb71921c596a92813` | Defer | Four unique exploratory calibration files including item-level logits/predictions; independent release and dependency review; retain original. |
| [experiments/ml099-replication](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/a2212fed7fee4c930ee47c08143e11b45f5a8c64)<br>`a2212fed7fee4c930ee47c08143e11b45f5a8c64` | Preserve externally (planned) | Prior tree audit found no absent paths; retain historical context and variants. Conditional retirement only after durable archive verification. |
| [experiments/ml100-bounded-screen](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e8b9c1b93766cb3f0d4feb68bc7b60afd4963b70)<br>`e8b9c1b93766cb3f0d4feb68bc7b60afd4963b70` | Preserve externally (planned) | Prior tree audit found no absent paths; retain historical context and variants. Conditional retirement only after durable archive verification. |
| [experiments/position-counterbalanced-roles-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e6db0c23ea642cb773176f2dfb76d63d1f6da6ef)<br>`e6db0c23ea642cb773176f2dfb76d63d1f6da6ef` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/role-diversity-recovery-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/b38aa3d4c916b7b62c4cb796ab6c5bdbd3f8549e)<br>`b38aa3d4c916b7b62c4cb796ab6c5bdbd3f8549e` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/role-learning-origins-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702)<br>`9c7146f88f96ad1c38de179c4ca8c3f26ebf1702` | Retain active / defer | Frozen implementation represented in #14; no new origins output; protected local worktree, checkpoints and jobs. No branch or archive operation affecting this ref. |
| [experiments/role-mechanism-repair-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/11bd5e2e895dc26432409cbff709f2f54b08f906)<br>`11bd5e2e895dc26432409cbff709f2f54b08f906` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/role-mechanism-replication-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/6860c60fd3262d070d2ac81918b452c6db5eb3eb)<br>`6860c60fd3262d070d2ac81918b452c6db5eb3eb` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/role-repair-specificity-v1](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4df81c3faad045db0bfdff2386ef022067eeae58)<br>`4df81c3faad045db0bfdff2386ef022067eeae58` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [experiments/train-surface-factorial-v0](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/5bffaf016abe2cc8c591934189f55bc6c5569420)<br>`5bffaf016abe2cc8c591934189f55bc6c5569420` | Integrate via #14 | Scientific additions represented at original paths in #14; old root navigation and branch-specific workflows remain archived. After gates and archive retrieval, reassess retirement. |
| [integration/paper-i-core-transfer-reports](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/0bb6c18af1e2f9f4371335ddc866c93a8f917e9a)<br>`0bb6c18af1e2f9f4371335ddc866c93a8f917e9a` | Defer | Draft #22; #14 dependency and coordinator/publication clearance. |
| [integration/paper-i-scientific-lineage](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091)<br>`910c320a3696be4a4807ef554602adff3965d091` | Defer | Draft #14; frozen dependency and scientific validation gates. |
| [main](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/da7d463b10955f02cf342205e5e0a03f450cfcb1)<br>`da7d463b10955f02cf342205e5e0a03f450cfcb1` | Retain canonical | Reviewed integration target, never retire. |
| [order-to-role/results-dev](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/25c6914e29473e4b1a1063600c48743f68b705c3)<br>`25c6914e29473e4b1a1063600c48743f68b705c3` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [order-to-role/stimuli](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e)<br>`9c437928f61351a72e8d7be4cac6d3a964844a7e` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [paper/construction-sensitive-roles-20261121](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/a261bd847b4119dcc10b26a0f9890c22f17780b5)<br>`a261bd847b4119dcc10b26a0f9890c22f17780b5` | Defer | Seven additions and four conflicting construction-shift files; author version/scope decision before importing; preserve snapshot. |
| [paper/slog-external-validation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/a30d070442d8b04e89744a963f2183214eb7ab37)<br>`a30d070442d8b04e89744a963f2183214eb7ab37` | Preserve externally (planned) | Prior tree audit found no absent paths; retain historical context and variants. Conditional retirement only after durable archive verification. |
| [paper2/integrate-otr-code](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/0b323dafa738070c88ccb90d3e260ec5c8022a3c)<br>`0b323dafa738070c88ccb90d3e260ec5c8022a3c` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [paper2/research-section](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/3c2fee2977b03c524d6a2eda996fc02a09a1bbe5)<br>`3c2fee2977b03c524d6a2eda996fc02a09a1bbe5` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [paper2/supplementary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/bc82767ce3feb9c63532b79de9afd702bdc101b7)<br>`bc82767ce3feb9c63532b79de9afd702bdc101b7` | Retain / Paper II ownership | Outside Paper I disposition authority; #19–21 independent; no modification. |
| [protocol/exp101-anti-shortcut](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4152e920756088ee80beabb527c2214397ea8724)<br>`4152e920756088ee80beabb527c2214397ea8724` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-mps-fixture-same-origin-rc4](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/ce336b7af0ebc147e32a03a43b01af6f458235ca)<br>`ce336b7af0ebc147e32a03a43b01af6f458235ca` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-mps-resume-diagnostic](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/5b7aba0c488e10792219287ea97e234c345784ae)<br>`5b7aba0c488e10792219287ea97e234c345784ae` | Preserve externally (planned) | Historical MPS diagnostic and differing frozen RC assets; three absent paths across both snapshots. No promotion or execution without compatibility decision. |
| [protocol/exp101-mps-same-origin-check](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/3665871e95f28a7471380bda136bca9f1a398e86)<br>`3665871e95f28a7471380bda136bca9f1a398e86` | Preserve externally (planned) | Historical MPS diagnostic and differing frozen RC assets; three absent paths across both snapshots. No promotion or execution without compatibility decision. |
| [protocol/exp101-rc7-manifest-keyfix](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/3ae081129b0c480bd6ac57cb8fa7f7b952f54b09)<br>`3ae081129b0c480bd6ac57cb8fa7f7b952f54b09` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-release-prep-rc2](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/aad47872dd0c5b95514c6e86ab475d168d6d5219)<br>`aad47872dd0c5b95514c6e86ab475d168d6d5219` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-release-prep-rc3-mps](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/ce336b7af0ebc147e32a03a43b01af6f458235ca)<br>`ce336b7af0ebc147e32a03a43b01af6f458235ca` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-release-prep-rc4-mps](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4cca9525fc202bbc51e80efbf703e5be7c9f88a3)<br>`4cca9525fc202bbc51e80efbf703e5be7c9f88a3` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-release-prep-rc5-author-waiver](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/0ba7090ff44646e2fae85cb4269f9f65af2c349c)<br>`0ba7090ff44646e2fae85cb4269f9f65af2c349c` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [protocol/exp101-release-prep-rc6-train](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/4ac50c97f16fc67562df1c3b66d24fd5892b174c)<br>`4ac50c97f16fc67562df1c3b66d24fd5892b174c` | Preserve externally (planned) | Earlier release-candidate versions or RC7 represented bytes; retain frozen originals and variants at their own commit; no overwrite. Archive/retrieval and worktree verification required. |
| [reorg/research-tracks-2026-11-05](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/e214de5248874e8c982892421214c5be49e7fe55)<br>`e214de5248874e8c982892421214c5be49e7fe55` | Preserve externally (planned; shared) | No absent paths in union; historical navigation variants remain; coordinator owns shared retirement decisions. |

## Bounded consolidation sequence

1. Recover original EXP107 bytes, verify SHA-256 before any dependency-only amendment or draft against latest main as coordinated. Re-run existing isolated offline audit on revised candidate only after authorisation. Clear named dependency, digest semantics, checkpoint archive, completeness and rights gates. No frozen-file substitutions. Then coordinator may authorise #14.
2. Recheck #22 against merged #14 and current main; verify exact report blobs, manifest and rights; coordinator may authorise #22.
3. Existing #5 remains a separate correction/design review. No duplication and no new ML097 experiment. Historical diagnostics and calibration archives require scope, release, dependencies and consistency decisions before any bounded import. Construction-shift additions require author choice, #22 dependencies and review against the four conflicting versions. Preserve unselected sources unchanged.
4. Nine workflows: separate coordinated CI PR only for approved checks, retaining original YAML snapshots. Prefer bounded static/offline checks, explicit manual gates and resource ceilings; do not activate historical push triggers by default. Isolated audit harness can remain archived. No new training or protected scoring.
5. Update evidence/index links to canonical paths only after integration; retain original source hashes and negative/exploratory outcomes. No visualisation redesign or shared navigation changes in this register.

## Durable archival verification

For each proposed historical tag, use an independent clean clone, not a protected worktree. Verify branch head has not moved; record research purpose, source commit and tree, paths/modes/blob SHAs and canonical destinations. Create a non-moving annotated tag only after coordinator approval; push exact tag ref without force. From an independent checkout fetch the remote tag, verify peeled commit equals recorded full SHA, verify tree/blob IDs and run Git object integrity checks. For large/protected non-Git artifacts record versioned archive URI/access policy, exact SHA-256, byte count and manifest linkage; independently retrieve and hash, without scoring or training. Record verification date, reviewer, actual tag/archive and retrieval result in this register. Confirm no local worktree, job, resume dependency, or collaborator requires the branch. No retirement recommendation becomes eligible until all checks pass and separate approval is recorded. A SHA link alone is not archival protection.

Estimated steady state: `main`, the protected origins branch for as long as needed, one active Paper II lineage if needed, and temporary reviewed PR branches. Exact active branch names/lifecycle subject to owners; no forced migration of ongoing work. Historic commits reside in verified archives, not indefinitely active branches.

**Current completion state:** scientific gates, recovery of original local source, archive verification, residual decisions, canonical hub link update and coordinating approvals remain outstanding. No merges, branch deletions, protocol changes, training, checkpoint loads or protected evaluation performed in preparing this register.
