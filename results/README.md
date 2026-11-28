# Experimental outputs

## What is tracked here

This is the **raw** output archive: JSON, PNG and Slurm logs from historical `ml*` runs and Paper II `otr*` runs (development, test0 and exploratory 70B). The folder is flat and contains **1,437 tracked files, including this README**, about 851 MiB at the 28 November 2026 `main` snapshot. Many OTR run names follow `<experiment>_<model>_<split>[_<variant>]_<timestamp>`.

- **For paper-facing citations, use curated outputs:** Paper II → [`papers/order-to-role/results/`](../papers/order-to-role/results/) and its `MANIFEST.json`; Paper I → the reports linked from [`papers/learning-relations/`](../papers/learning-relations/).
- The [Paper II archive inventory and checksums](../papers/order-to-role/archive/README.md) document the **458 files specific to the historical `order-to-role/results-dev` branch** (and one superseded version); they are not a full manifest of all 1,437 tracked files here.
- Four Wikipedia-derived WIKI-ROLES item sets/check sheets are deliberately excluded from tracked Git pending licensing and archival clearance.
- When multiple runs share a name prefix, use the exact run selected in the relevant manifest rather than assuming the latest timestamp is canonical.

## Default policy for new outputs

Newly generated arrays, model predictions, logs and checkpoint data stay local and are gitignored by default. Use small rights-cleared summaries and hash manifests in reports instead. Any reviewed exception, such as the existing Paper II imported outputs, requires explicit provenance, evidence-status and rights checks.
