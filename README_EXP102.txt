EXP-102 RESEARCH PACK — 13 NOV 2026

Contains:
- scripts/exp102_evidence_audit.py : read-only validation + analysis
- paper/exp102_next_study_protocol.md : research memo for the next test

From the root of your exp102-clean-design worktree:
  unzip -n ~/Downloads/exp102_research_pack.zip -d .
  python3 scripts/exp102_evidence_audit.py

To archive new derived summary files (without overwriting existing files):
  python3 scripts/exp102_evidence_audit.py --write

The audit script requires your locally saved 32 fresh-seed JSON result files.
It does not load checkpoints, modify experiments, or access protected data.
