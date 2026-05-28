# Source Of Truth

This file defines the active contract for `/share/yhm/test/AutoML_EDA`.

## Canonical Input

The canonical source table is:

`/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx`

The root `Makefile` owns the runtime default through `CANONICAL_SOURCE_TABLE`
and `SOURCE_TABLE`. Other references in documentation or YAML metadata must
mirror that Makefile default rather than introduce a competing data source.

The active target is:

`Delta_PCE = jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator`

## Canonical Entrypoint

Use the root `Makefile` for current workflows:

```bash
make research-package
make research-package-smoke
make research-package-pdf
make research-package-verify
make test-research-package
```

The canonical implementation runner is:

`hybrid_agent_exploration/src/run_research_package.py`

Generated packages are written under:

`hybrid_agent_exploration/results/verified_discovery_runs`

## Artifact Policy

Commit source code, tests, small manifests, curated configs, and current
documentation. Do not commit regenerable bulk outputs such as root `figures/`,
`tables/`, `reports/`, large processed CSV exports, model directories, or
temporary research-package run bundles.

Root-level old EDA summaries and generated plots are historical artifacts. They
can remain locally for provenance, but they are not the current source of truth
for data, metrics, figures, or manuscript generation.

## Guardrails

Run these checks before finalizing changes that touch data paths, entrypoints,
or generated outputs:

```bash
make audit-source-of-truth
python hybrid_agent_exploration/scripts/audit_cleanup.py
make test-research-package
```
