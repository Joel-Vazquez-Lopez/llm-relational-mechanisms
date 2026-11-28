# Developmental mechanistic evidence: four selected recovery seeds

Extends [PR #27](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/pull/27), which records selected final-checkpoint intervention results. This is a separate additive supplement based on archived outputs at `910c320a3696be4a4807ef554602adff3965d091`, not new model experiments. All comparisons are exploratory; selected recovery seeds and fresh optimiser after 6400 updates preclude population or uninterrupted-training claims.

## Sources and extraction

All 16 seed/checkpoint combinations were read from each of the five report families: [representation JSON](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_representation_localisation_v0/per_checkpoint), [routing compact CSV](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_routing_v0/compact), [pathways compact CSV](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_causal_pathways_v0/compact), [binding validation compact CSV](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_causal_validation_v0/compact), [binding localisation compact CSV](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_binding_localisation_v0/compact). Files use seed{seed}_step{step}.json or .csv. [Circuit summary](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/910c320a3696be4a4807ef554602adff3965d091/reports/role_circuit_diagnostics_v0/analysis_summary.json) provides native behaviour. Values below are descriptive, rounded, not new inferential tests. Each vector follows steps 6400, 12800, 25600, 51200. All probabilities/accuracies are fractions.

## Native behaviour and representation

Circuit summary: voice_averaged, roles_8, unseen_roles_rich.

| Seed | Event mass | Conditional role probability | C_query |
|---|---|---|---|
|16101|.4958,.9475,1.0000,.9997|.5621,.9874,1.0000,1.0000|0,.8457,1,1|
|16127|.9905,.9970,.9435,1.0000|.5000,.9953,.8862,1.0000|0,.9900,.6250,1|
|16143|.4985,.6813,.9883,.9955|.5000,.5000,.5000,.5000|0,0,0,.000651|
|16167|.9862,.9918,.9985,.9992|.5000,.5000,.5000,.5000|0,0,0,0|

Representation JSON rows: roles_8, rich fit/test, opposite voices averaged. Probe chance event/role .5, binding_rank .25. Each cell is a vector across checkpoints.

|Seed|Site|Event|Role|Binding rank|
|---|---|---|---|---|
|16101|L0.residual|.5041,1,1,1|.8660,1,1,1|.2766,.2851,.2610,.2447|
|16101|L1.residual|.5242,1,1,1|.8563,1,1,1|.2853,.3572,.5074,.5381|
|16101|L2.residual|.5205,1,1,1|.8512,1,1,1|.2848,.4572,.5348,.5585|
|16101|norm|.5206,1,1,1|.8484,1,1,1|.2846,.4539,.5344,.5604|
|16127|L0.residual|.5132,.5318,.5227,.6774|.5131,1,1,1|.2496,.2352,.2458,.2444|
|16127|L1.residual|.9344,.7676,.7495,.8109|.5103,1,1,1|.3311,.3276,.3390,.3446|
|16127|L2.residual|.9905,.9976,.9497,.9237|.5068,1,1,1|.4068,.5332,.5230,.5949|
|16127|norm|.9984,.9962,.9566,.8748|.5077,1,1,1|.4105,.5402,.5238,.6003|
|16143|L0.residual|.5007,.5293,.8080,.9046|.5003,.5057,.5707,.5951|.2499,.2492,.2508,.2497|
|16143|L1.residual|.5041,.5172,.7779,.8929|.5025,.5058,.5352,.5663|.2506,.2488,.2536,.2556|
|16143|L2.residual|.5092,.7365,.9974,.9849|.5023,.5019,.5145,.5414|.2506,.3115,.4177,.4237|
|16143|norm|.5075,.7426,.9974,.9808|.5028,.5030,.5146,.5345|.2504,.3141,.4178,.4200|
|16167|L0.residual|.5163,.6320,.9734,1|.5000,.5042,.5013,.5012|.2496,.2502,.2516,.2550|
|16167|L1.residual|.5737,.6122,.9675,1|.5004,.5038,.5009,.5028|.2525,.2514,.2547,.2594|
|16167|L2.residual|1,1,1,1|.5004,.5019,.5006,.5020|.4190,.4141,.4205,.4212|
|16167|norm|1,1,1,1|.5008,.5020,.5011,.5017|.4236,.4165,.4260,.4234|

Embedding probes are at chance throughout these comparisons. 16101 already has role accessibility before strong event selection; 16127 has early event accessibility and retains perfect role probes during behavioural regression. Collapsed seeds exhibit late-layer event accessibility followed by earlier-layer accessibility without reliable role binding. This is not proof of backward information flow or onset between checkpoints. Probe targets may reflect query lexical identity; quartet-disjoint is not entity-disjoint. Candidate rank is a proxy, not binding. See [protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/910c320a3696be4a4807ef554602adff3965d091/paper/ROLE_REPRESENTATION_LOCALISATION_V0_PROTOCOL.md).

## Activation sensitivity

Mean AP/PA paired query RMS, roles_8 unseen_roles_rich; raw RMS, not normalised across sites.

|Seed|L0 residual event RMS|L0 residual role RMS|L2 residual event RMS|L2 residual role RMS|
|---|---|---|---|---|
|16101|.011,4.356,6.676,6.899|3.906,8.591,6.379,7.977|.243,8.023,11.332,13.820|4.450,10.867,11.119,14.125|
|16127|.020,.024,.013,.028|.028,11.152,14.786,5.851|17.376,17.998,16.341,6.886|.074,15.224,14.747,6.247|
|16143|.001,.037,.315,2.519|.0004,.017,.035,.286|.151,8.777,19.450,17.953|.071,.076,.050,.332|
|16167|.078,.977,4.802,4.885|.0002,.008,.008,.007|18.391,24.779,30.277,26.711|.047,.065,.015,.015|

16127's early-layer role RMS peaks during regression and falls at recovery: magnitude is not competence or causal strength. 16167 retains large event sensitivity with negligible role sensitivity.

## Routing and pathways

Routing compact rows: roles_8, unseen_roles_rich. Role/event candidate sites fixed from original discovery (16101 L0.head0/L0.head3; others L0.head2/L2.head2, matched coordinates not discovered role heads in collapsed models). Role donor accuracy is not role-specific effect: compare conditional role probability delta. Event swap delta is original event mass change, not donor correctness.

|Seed|Role donor accuracy|Role swap conditional role delta|Event swap event mass delta|L0 MLP role swap donor accuracy|L2 MLP role swap donor accuracy|
|---|---|---|---|---|---|
|16101|.3252,.9424,1,1|-.1029,-.9724,-1,-.9999|0,-.7649,-.9993,-.9997|.1807,.0137,.0049,.0400|.1523,.4258,.1367,0|
|16127|.5020,.9951,.8799,1|0,-.9901,-.7620,-1|-.9722,-.9900,-.9009,-1|.5020,.8213,.7266,.9971|.4971,.9561,.7656,.9971|
|16143|.2500,.3770,.4990,.4990|0,0,0,0|0,-.3323,-.9849,-.9928|.2500,.3770,.4980,.4980|.2500,.3770,.4980,.5068|
|16167|.5000,.5000,.5000,.5000|0,0,0,0|-.9675,-.9777,-.9965,-.9965|.5000,.5000,.5000,.5000|.4990,.4990,.5010,.4990|

Rounded zeros can conceal tiny nonzero effects; original rows remain authoritative. 16127's event swap is effective at 6400 even without role discrimination. 16143 develops strong event control later; 16167 has it already at 6400. Role donor accuracy around .5 in collapsed models does not establish role control: corresponding conditional-role deltas are negligible.

Pathways compact, unseen_roles_rich, voice averaged; target_joint for candidate/fixed/0 (baseline), candidate/fixed/3, candidate/dynamic/3, control/fixed/3.

|Seed|Baseline|Fixed joint swap|Dynamic joint swap|Control joint swap|
|---|---|---|---|---|
|16101|0,.8555,1,1|0,.5430,.9961,1|0,.5430,.9961,1|0,0,0,0|
|16127|0,.9883,.5586,1|0,.9805,.5156,1|0,.9805,.5156,1|0,0,0,0|
|16143|0,0,0,0|0,0,0,0|0,0,0,0|0,0,0,0|
|16167|0,0,0,0|0,0,0,0|0,0,0,0|0,0,0,0|

Selected joint intervention develops with successful binding and regresses with 16127 behaviour. Fixed and dynamic donors are distinct counterfactuals even where results agree. No proof of complete circuit sufficiency. Candidate axes and controls were chosen after routing discovery. Q/K/V, token-specific and restoration conditions remain in original reports; the table is not an exhaustive edge audit.

## Binding transfer and localisation

Binding validation: event permutation, AP/PA mean, query/cross_source/2, target and alternate_target accuracy.

|Seed|Intended target|Alternate target|
|---|---|---|
|16101|.3301,.8340,.9990,1|.3301,.1445,0,0|
|16127|.2412,.3906,.2207,.0078|.2598,.4932,.5410,.9727|
|16143|.2500,.2324,.2490,.2344|.2500,.2676,.2510,.2705|
|16167|.2490,.2500,.2500,.2500|.2510,.2510,.2500,.2500|

16127 develops increasingly strong alternate-target selection in this condition, even as native competence recovers. It retains successful cross_source/1 control (target .5,.9961,.8936,1). Thus transfer is condition-dependent, not universally absent. Binding localisation mixed source_type, event_flip conditions; recipient_union and donor_union:

|Seed|Q recipient|Non-name KV recipient|Non-name KV donor|
|---|---|---|---|
|16101|.3418,.8594,1,1|.3418,.8594,1,1|.1582,.0117,0,0|
|16127|.5000,.9941,.8477,1|.2813,.0801,.2441,0|.4355,.6504,.6074,.9336|
|16143|.2520,.3691,.5020,.5000|.2520,.2598,.2832,.2734|.2480,.2949,.3965,.4258|
|16167|.5000,.5000,.5000,.5000|.2695,.2676,.2578,.2500|.3750,.3574,.4004,.4336|

In this selected condition 16127's non-name KV patch increasingly favours donor identity, whereas 16101 does not exhibit that signature. Name-only KV rows agree with Q recipient accuracy in 16127 at all four checkpoints; this does not prove names are unnecessary natively, since contextualised non-name states can carry identity information. Collapsed models also show donor sensitivity without joint binding. Consult full condition sets before generalising.

## Limits and outstanding evidence

All four seeds are selected, all reports exploratory, and the original independent mechanism-category replication failed. No new training, inference, evaluation, checkpoint loading or frozen file changes. This supplement compares fixed slices across all 16 checkpoints and five report families; it does not exhaust every probe domain, head, arm, population, QKV hybrid, restoration, positional or distributed subset. Full developmental profiles and exact original checkpoints remain necessary for stronger conclusions. No claim about active origins study outcomes.
