[← Paper I](README.md) · [Results](RESULTS.md) · [Methods](METHODS.md) · [Mechanisms](MECHANISMS.md) · [Experiments](EXPERIMENTS.md)

# Mechanistic investigations

The evidence progresses from probability decomposition to representation accessibility, causal intervention and transfer. No universal circuit is claimed.

| Study | Design and interpretation |
|---|---|
| [Circuit diagnostics](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_CIRCUIT_DIAGNOSTICS_V0_PROTOCOL.md) | Decompose event and role probability/loss; distinguish global collapse, event-only competence, training-specific binding and generalisable binding. |
| [Representation localisation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_REPRESENTATION_LOCALISATION_V0_PROTOCOL.md) | Readout-position ridge probes across embedding, attention, MLP and residual sites, disjoint sources and cross-voice/surface tests. Decodability is not causal necessity. |
| [Causal routing](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_CAUSAL_ROUTING_V0_PROTOCOL.md) | Within-source role/event swaps and mean/zero interventions. Selected successful seeds show separable functional control; locations differ. |
| [Causal pathways](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_CAUSAL_PATHWAYS_V0_PROTOCOL.md) | Joint swaps, Q/K/V and positional replacements, downstream restoration. Query-side sufficiency is not entity retrieval or edge attribution. |
| [Binding validation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_BINDING_CAUSAL_VALIDATION_V0_PROTOCOL.md) | Cross-source and cross-voice transfer with changed names; distinguish recipient-target from donor-target answers. |
| [Binding localisation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_BINDING_LOCALISATION_V0_PROTOCOL.md) | Disjoint names, entity versus non-name positions, distributed head subsets. |
| [Independent replication](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_MECHANISM_REPLICATION_V0_PROTOCOL.md) | Frozen head-pair discovery and independent test; primary category replication failed. |
| [Expanded discovery](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/paper/ROLE_MECHANISM_DISCOVERY_V1_PROTOCOL.md) | Exploratory head/MLP sets; original positive controls recovered, one additional source-dependent mechanism. |

## Two illustrative routes

Selected seed 16101: early head interventions support event and role control; transfer across changed source identities is consistent with recipient-relative control. Selected seed 16127: role intervention transfers, whereas event intervention can import donor identities. These are within-model findings, not proof of a shared architecture-wide circuit. See [routing compact tables](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/reports/role_causal_routing_v0/compact) and [binding validation archive](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c7146f88f96ad1c38de179c4ca8c3f26ebf1702/reports/role_binding_causal_validation_v0).

## Limits and next tests

Failure of the original mechanistic classification in fresh models remains authoritative. Distributed discovery suggests single-head searches can miss control, but expanded search and different samples prevent interpreting it as retrospective confirmation. Repair targets in collapsed models are candidates, not independently proven causes. Origins gradients can motivate preventive interventions, not establish gradient conflict causally.
