# Source-grounded mechanistic notes: selected recovery models

Companion to [Mechanisms](../MECHANISMS.md) and [Results](../RESULTS.md). Exploratory archived evidence only; no new inference or evaluation. The source snapshot is PR #14 head `910c320a3696be4a4807ef554602adff3965d091`; it remains blocked. The following values are transcribed from original compact reports, not new statistical estimates.

## Component-specific causal control at 51,200 updates

Filter both [role-swap](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_routing_v0/compact/final_role_swap.csv) and [event-swap](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_routing_v0/compact/final_event_swap.csv) to arm roles_8 and population unseen_roles_rich. Selected illustrative rows (not exhaustive):

| Seed | Site | Swap | Donor accuracy (matching swap) | Original event mass | Original conditional role probability | Joint correctness delta |
|---|---|---|---:|---:|---:|---:|
| 16101 | L0.head0 | role | 1.0 | 0.9997356 | 0.0000296 | -1.0 |
| 16101 | L0.head3 | event | 1.0 | 0.0000762 | 0.5275371 | -1.0 |
| 16101 | L1.head1 | role | 0.3779297 | 0.9733415 | 0.609857 | -0.890625 |
| 16127 | L0.head2 | role | 1.0 | 0.9999959 | 0.0000036 | -1.0 |
| 16127 | L0.mlp | role | 0.9970703 | 0.9995266 | 0.0053119 | -1.0 |
| 16127 | L2.mlp | role | 0.9970703 | 0.9827152 | 0.0043196 | -1.0 |
| 16127 | L2.head2 | event | 1.0 | 0.0000297 | 0.6282658 | -1.0 |

Donor accuracy refers to role_donor_accuracy or event_donor_accuracy, respectively; it is not joint correctness. Layer indices are zero-based. These selected rows reveal different organisations: seed 16101 has distinct event and role control at L0, with additional L1 sensitivity; seed 16127 shows early role control, substantial MLP role sensitivity at L0 and L2, and event control at L2. Swap effects establish intervention-dependent control in these settings, not component necessity, exclusive native pathways or general architectural laws. MLP sensitivity does not by itself establish independent role computation.

## Identity transfer as a stronger test

Source: [16101](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_causal_validation_v0/compact/seed16101_step51200.csv), [16127](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_causal_validation_v0/compact/seed16127_step51200.csv). In AP voice, permutation=event, condition=query/cross_source/2, target_source=renamed, 16101 has target_accuracy=1 and alternate_target_accuracy=0; 16127 has target_accuracy=0.009765625 and alternate_target_accuracy=0.953125. For query/cross_voice_source/2, the corresponding values are 1/0 and 0.0078125/0.990234375. Both models have target accuracy 1 for query/cross_source/1 in that permutation. This is condition-specific evidence, not a claim that all role permutations transfer. See the [original protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/paper/ROLE_BINDING_CAUSAL_VALIDATION_V0_PROTOCOL.md) for definitions and controls.

## Developmental record and limits

The five original analysis summaries record 33,792 probe rows and 5,632 sensitivity rows (representation localisation), 24,576 routing rows, 6,944 pathway rows, 3,840 binding-validation rows and 13,376 binding-localisation rows. All classify themselves as exploratory. The per-checkpoint tables cover selected seeds 16101, 16127, 16143, 16167 at 6,400, 12,800, 25,600 and 51,200 updates. See [representation archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_representation_localisation_v0/analysis_summary.json), [routing](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_routing_v0/analysis_summary.json), [pathways](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_pathways_v0/analysis_summary.json), [binding validation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_causal_validation_v0/analysis_summary.json), [binding localisation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_localisation_v0/analysis_summary.json). Row counts are not independent samples. Full developmental effect-size profiles remain to be extracted; final-checkpoint results do not establish emergence order. The failed prespecified mechanism replication remains authoritative. Active origins work is not interpreted here.

No frozen data, code, manifests, checkpoints or shared infrastructure changed.
