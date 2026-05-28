# Perovskite QSPR Research Package

This repository contains the current QSPR research-package workflow for PSC
additive/modulator analysis. The repository-level source-of-truth contract is
documented in [SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md).

## Canonical Contract

- **Input source**: `/share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx`
- **Primary target**: `Delta_PCE = jv_reverse_scan_pce - jv_reverse_scan_pce_without_modulator`
- **Canonical runner**: `hybrid_agent_exploration/src/run_research_package.py`
- **Canonical entrypoint**: root `Makefile`
- **Default output root**: `hybrid_agent_exploration/results/verified_discovery_runs/`

The older root-level EDA scripts and generated reports are historical artifacts.
They are useful for provenance, but they are not the current execution path.

## Run

Use the root Makefile for all current research-package work:

```bash
make research-package
make research-package-pdf
make research-package-verify
make test-research-package
```

For a quick smoke run:

```bash
make research-package-smoke
```

Common overrides:

```bash
SOURCE_TABLE=/path/to/source.xlsx \
DATASET_ID=jpcl-full-source \
EVIDENCE_MODE=external-cached \
INPUT_SCOPE=full-source \
MIN_VERIFIED_ROWS=10 \
TOP_K=100 \
make research-package
```

## Outputs

Generated run bundles are written under
`hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)`.
The key package artifacts are:

- `verified_discovery/`: verified training data, quarantine rows, candidate rankings, and model validation outputs.
- `report_bundle/main_text/`: main manuscript-style report.
- `report_bundle/si/`: supporting information.
- `package_manifest.json` and `provenance_manifest.json`: run-level provenance.
- `source_completeness/`: source-column completeness audit.

These generated outputs are ignored by default. Commit source code, tests,
configuration, small manifests, and curated documentation; keep bulk generated
artifacts outside the active source tree unless they are intentionally promoted.

## Repository Layout

```text
AutoML_EDA/
├── Makefile                         # canonical execution entrypoint
├── SOURCE_OF_TRUTH.md               # data/run/artifact policy
├── scripts/audit_source_of_truth.py # root-level SSoT guardrail
├── tests/                           # regression and contract tests
└── hybrid_agent_exploration/
    ├── src/                         # current research-package implementation
    ├── configs/                     # method/search/source metadata
    ├── docs/                        # current research and figure standards
    ├── candidate_sources/           # curated external candidate inputs
    └── scripts/audit_cleanup.py     # hybrid subtree artifact guardrail
```

## Validation

Run the source-of-truth and cleanup guardrails before finalizing repo hygiene
changes:

```bash
make audit-source-of-truth
python hybrid_agent_exploration/scripts/audit_cleanup.py
make test-research-package
```

If the full project environment is expensive to resolve, the pure audit tests
can also be run in an isolated pytest environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project --with pytest python -m pytest -q tests/test_source_of_truth_audit.py tests/test_hybrid_cleanup_audit.py
```
