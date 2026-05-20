# Hybrid Agent Exploration

This directory is the active, trimmed workspace for the PSC additive QSPR
agentic exploration.

## What Stays Active

- `src/`: reusable pipeline, worker, evaluation, data-cleaning, screening, and
  reporting code.
- `configs/`: search spaces, method registry, cleaning action space, and data
  source configuration.
- `explorations/`: reproducible layer-specific exploration scripts plus small
  summary outputs.
- `article/manuscript_v2.md`, `article/manuscript_v2.pdf`, and
  `article/figures_v2/`: current paper-facing manuscript and final figure set.
- `docs/`, `RESULTS_AGENTIC_CLEANING.md`, and `AGENTS.md`: scientific rationale,
  JPCL comparison notes, reference validation, and five-layer architecture.

## What Is Archived

Large regenerated bundles, bulk report artifacts, intermediate CSV caches, v1
paper figures, preview images, and old research-story runs were moved out of
the active tree to:

`/share/yhm/test/AutoML_EDA_archive/20260520_hybrid_cleanup/`

The active tree should remain source-first. Regenerate bulky outputs when
needed, or inspect the archive for historical runs.

## Audit

Run this before committing cleanup-sensitive changes:

```bash
python hybrid_agent_exploration/scripts/audit_cleanup.py
uv run pytest tests/test_hybrid_cleanup_audit.py -q
```
