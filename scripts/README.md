# Experiment scripts — canonical execution directory

**Do not move these scripts during the active ML-072 protocol.** Some scripts import others by their historical filenames, and ML-072's frozen protocol records a SHA-256 of the current file.

The experiments are now organized into two **distinct research sections** with curated indices:

- [Scratch-trained Transformers: ML-037–047, ML-057–072](../investigations/scratch_transformers/EXPERIMENT_INDEX.md)
- [Pretrained LLMs: ML-001–036, ML-048–055](../investigations/pretrained_llms/EXPERIMENT_INDEX.md)
- [Scratch ML-056 feasibility report](../reports/ml056/README.md); this does not currently have a tracked `scripts/ml056_*.py` root entrypoint.

The original `scripts/mlNNN_*.py` paths remain executable. Frozen `plan → train → evaluate` or `freeze → evaluate` ordering must be followed before accessing protected results. Use `make test` for the package tests, and consult each experiment's documentation for its dependencies.
