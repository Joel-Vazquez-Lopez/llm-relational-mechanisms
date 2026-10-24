# Hands-on research development

This repository is for code authored and understood by the researcher.

1. Work on a small branch named analysis/... or experiment/....
2. Write one focused component, inspect it, test it and understand its mathematics.
3. Run tests and the synthetic demo before merging.
4. Before empirical model inference, complete docs/EXPERIMENT_TEMPLATE.md, check historical duplication and obtain explicit budget/data/holdout approval.
5. Store raw results securely, preserve failed runs and share only rights-cleared reports/aggregates.
6. Independently scrutinize extraordinary findings and mechanistic interpretations before calling them discoveries.

```bash
git clone https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms.git
cd llm-relational-mechanisms
make test
make demo
```

The existing project has three chats: scientific theory, hands-on coding, and Codex engineering HQ. No chat's independent suggestion overrides the researcher's experimental decisions.
