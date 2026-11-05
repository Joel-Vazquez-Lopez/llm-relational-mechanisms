# Shared methods, infrastructure and standards

Both investigations share the same high standards, but not the same empirical conclusions.

- [Scientific charter](../../docs/RESEARCH_CHARTER.md)
- [Evidence and reproducibility policy](../../docs/EVIDENCE_POLICY.md)
- [Experimental template](../../docs/EXPERIMENT_TEMPLATE.md)
- [Provenance](../../docs/PROVENANCE.md)
- [Workflow](../../docs/WORKFLOW.md)
- [Research log](../../docs/RESEARCH_LOG.md)
- [Reusable Python package](../../src/relational_lab/)
- [Tests](../../tests/)
- [Archived reports](../../reports/)

**Compatibility contract:** `scripts/` keeps historical executable paths while a frozen experiment is running. New code can eventually be organized physically into packages, but that is a separately tested, versioned refactor—not a silent move of files whose paths are used by checkpoints, imports, and SHA-256-locked protocols.

Keep protected evaluation and local raw results distinct from examples and test fixtures. Preserve failures and exploratory results, and use independent units of analysis when assessing replication.
