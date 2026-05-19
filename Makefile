PYTHON ?= python

SOURCE_TABLE ?= /share/yhm/test/20260216_with_chemical_merge_fast/merged_llm_crossref_data_streaming_with_chemical_data_fast.xlsx
DATASET_ID ?= canonical-research-package
ARTIFACT_DIR ?= hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)
EVIDENCE_MODE ?= external-cached
INPUT_SCOPE ?= selected-subset
MIN_VERIFIED_ROWS ?= 10
TOP_K ?= 100
SMOKE_MAX_ROWS ?= 25
SMOKE_MIN_VERIFIED_ROWS ?= 1
SMOKE_TOP_K ?= 10
PANDOC_PDF_ENGINE ?= xelatex
PANDOC_MAINFONT ?= DejaVu Serif

RUN_RESEARCH_PACKAGE := hybrid_agent_exploration/src/run_research_package.py
ROOT_PROVENANCE_MANIFEST := hybrid_agent_exploration/src/reporting/root_provenance_manifest.py
VERIFY_RESEARCH_PACKAGE := hybrid_agent_exploration/src/verify_research_package.py
VERIFIED_DISCOVERY_DIR := $(ARTIFACT_DIR)/verified_discovery
REPORT_DIR := $(ARTIFACT_DIR)/report_bundle/main_text
SI_DIR := $(ARTIFACT_DIR)/report_bundle/si
CANDIDATE_LIBRARY_DIR := $(ARTIFACT_DIR)/candidate_library
SOURCE_COMPLETENESS_DIR := $(ARTIFACT_DIR)/source_completeness
PACKAGE_MANIFEST := $(ARTIFACT_DIR)/package_manifest.json
ROOT_PROVENANCE_JSON := $(ARTIFACT_DIR)/provenance_manifest.json
MAIN_TEXT_MD := $(ARTIFACT_DIR)/report_bundle/main_text/main_text_report.md
MAIN_TEXT_PDF := $(ARTIFACT_DIR)/report_bundle/main_text/main_text_report.pdf
SI_MD := $(ARTIFACT_DIR)/report_bundle/si/supporting_information.md
SI_PDF := $(ARTIFACT_DIR)/report_bundle/si/supporting_information.pdf
MAIN_TEXT_RESOURCE_PATH := $(REPORT_DIR):$(ARTIFACT_DIR)
SI_RESOURCE_PATH := $(SI_DIR):$(ARTIFACT_DIR)

CANDIDATE_SOURCE_ARGS := $(if $(CANDIDATE_SOURCE),--candidate-source $(CANDIDATE_SOURCE),)
CANDIDATE_SOURCE_NAME_ARGS := $(if $(CANDIDATE_SOURCE_NAME),--candidate-source-name $(CANDIDATE_SOURCE_NAME),)
VERIFY_CANDIDATE_LIBRARY_ARGS := $(if $(REQUIRE_CANDIDATE_LIBRARY),--require-candidate-library,)
VERIFY_EVIDENCE_CACHE_ARGS := $(if $(REQUIRE_EVIDENCE_CACHE),--require-evidence-cache,)

.PHONY: research-package research-package-smoke research-package-pdf research-package-verify test-research-package

research-package:
	$(PYTHON) $(RUN_RESEARCH_PACKAGE) \
		--input $(SOURCE_TABLE) \
		--output-dir $(ARTIFACT_DIR) \
		--dataset-id $(DATASET_ID) \
		--evidence-mode $(EVIDENCE_MODE) \
		--input-scope $(INPUT_SCOPE) \
		--min-verified-rows $(MIN_VERIFIED_ROWS) \
		--top-k $(TOP_K) \
		$(CANDIDATE_SOURCE_ARGS) \
		$(CANDIDATE_SOURCE_NAME_ARGS) \
		$(EXTRA_RUN_ARGS)

research-package-smoke:
	$(MAKE) research-package \
		DATASET_ID=$(DATASET_ID)-smoke \
		ARTIFACT_DIR=hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)-smoke \
		EVIDENCE_MODE=source-columns \
		MIN_VERIFIED_ROWS=$(SMOKE_MIN_VERIFIED_ROWS) \
		TOP_K=$(SMOKE_TOP_K) \
		EXTRA_RUN_ARGS="--max-rows $(SMOKE_MAX_ROWS)"

research-package-pdf:
	@command -v pandoc >/dev/null 2>&1 || { echo "pandoc is required for research-package-pdf. Install pandoc and rerun this target." >&2; exit 127; }
	@test -f $(MAIN_TEXT_MD) || { echo "missing main text markdown: $(MAIN_TEXT_MD). Run make research-package first or set ARTIFACT_DIR." >&2; exit 2; }
	@test -f $(SI_MD) || { echo "missing supporting information markdown: $(SI_MD). Run make research-package first or set ARTIFACT_DIR." >&2; exit 2; }
	pandoc --pdf-engine=$(PANDOC_PDF_ENGINE) -V mainfont="$(PANDOC_MAINFONT)" --resource-path=$(MAIN_TEXT_RESOURCE_PATH) $(MAIN_TEXT_MD) -o $(MAIN_TEXT_PDF)
	pandoc --pdf-engine=$(PANDOC_PDF_ENGINE) -V mainfont="$(PANDOC_MAINFONT)" --resource-path=$(SI_RESOURCE_PATH) $(SI_MD) -o $(SI_PDF)
	@candidate_library_args=""; \
	if [ -d "$(CANDIDATE_LIBRARY_DIR)" ]; then candidate_library_args="--candidate-library-dir $(CANDIDATE_LIBRARY_DIR)"; fi; \
	candidate_source_args=""; \
	if [ -n "$(CANDIDATE_SOURCE)" ]; then candidate_source_args="--candidate-source-path $(CANDIDATE_SOURCE)"; fi; \
	$(PYTHON) $(ROOT_PROVENANCE_MANIFEST) \
		--verified-discovery-artifact-dir $(VERIFIED_DISCOVERY_DIR) \
		--report-dir $(REPORT_DIR) \
		--si-dir $(SI_DIR) \
		--source-completeness-dir $(SOURCE_COMPLETENESS_DIR) \
		--package-manifest-path $(PACKAGE_MANIFEST) \
		--input-path $(SOURCE_TABLE) \
		--output-path $(ROOT_PROVENANCE_JSON) \
		$$candidate_library_args \
		$$candidate_source_args

research-package-verify:
	$(PYTHON) $(VERIFY_RESEARCH_PACKAGE) \
		--package-dir $(ARTIFACT_DIR) \
		$(VERIFY_CANDIDATE_LIBRARY_ARGS) \
		$(VERIFY_EVIDENCE_CACHE_ARGS)

test-research-package:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		tests/test_canonical_make_targets.py \
		tests/test_research_package_runner.py \
		tests/test_run_verified_discovery_cli.py \
		tests/test_root_provenance_manifest.py
