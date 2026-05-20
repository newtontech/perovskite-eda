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
CACHE_PREFLIGHT_MAX_ROWS ?=
CACHE_PREFLIGHT_SMOKE_MAX_ROWS ?= $(SMOKE_MAX_ROWS)
EVIDENCE_CACHE_DIR ?=
CACHE_COLLECT_MAX_REQUESTS ?= 100
CACHE_COLLECT_RETRY_ATTEMPTS ?= 2
CACHE_COLLECT_PROGRESS_EVERY ?= 10
CACHE_COLLECT_SHARD_COUNT ?=
CACHE_COLLECT_SHARD_INDEX ?=
CACHE_COLLECT_EFFECTIVE_SHARD_INDEX := $(if $(CACHE_COLLECT_SHARD_INDEX),$(CACHE_COLLECT_SHARD_INDEX),0)
CACHE_COLLECT_SHARD_CACHE_DIR ?= $(ARTIFACT_DIR)/evidence_cache_shards/shard-$(CACHE_COLLECT_EFFECTIVE_SHARD_INDEX)/evidence_cache
CACHE_COLLECT_SHARD_REPORT ?= $(ARTIFACT_DIR)/evidence_cache_collection_shard_$(CACHE_COLLECT_EFFECTIVE_SHARD_INDEX)_report.json
CACHE_SHARD_DIRS ?=
CACHE_MERGE_ENTITY_TYPE ?= all
CACHE_MERGE_CONFLICT_POLICY ?= fail
PANDOC_PDF_ENGINE ?= xelatex
PANDOC_MAINFONT ?= DejaVu Serif
PSC_CANDIDATE_CLOSURE_DATASET_ID ?= psc-additive-closure-candidate-loop
PSC_CANDIDATE_CLOSURE_ARTIFACT_DIR ?= hybrid_agent_exploration/results/verified_discovery_runs/$(PSC_CANDIDATE_CLOSURE_DATASET_ID)
PSC_CANDIDATE_CLOSURE_CACHE_DIR ?= $(abspath hybrid_agent_exploration/.cache/verified_discovery/psc-additive-closure-external-cached-full-source/evidence_cache)
PSC_CANDIDATE_SOURCE ?= hybrid_agent_exploration/candidate_sources/psc_passivator_seed_candidates.csv
PSC_CANDIDATE_SOURCE_NAME ?= psc-passivator-seed
EXPERIMENTAL_FEEDBACK_CSV ?=
EXPERIMENTAL_FEEDBACK_DATA_SCOPE ?= external_lab

RUN_RESEARCH_PACKAGE := hybrid_agent_exploration/src/run_research_package.py
RUN_EXPERIMENTAL_FEEDBACK_INGESTION := hybrid_agent_exploration/src/run_experimental_feedback_ingestion.py
RUN_EVIDENCE_CACHE_PREFLIGHT := hybrid_agent_exploration/src/run_evidence_cache_preflight.py
RUN_EVIDENCE_CACHE_COLLECTOR := hybrid_agent_exploration/src/run_evidence_cache_collector.py
RUN_EVIDENCE_CACHE_MERGE := hybrid_agent_exploration/src/run_evidence_cache_merge.py
ROOT_PROVENANCE_MANIFEST := hybrid_agent_exploration/src/reporting/root_provenance_manifest.py
VERIFY_RESEARCH_PACKAGE := hybrid_agent_exploration/src/verify_research_package.py
VERIFIED_DISCOVERY_DIR := $(ARTIFACT_DIR)/verified_discovery
CACHE_PREFLIGHT_DIR := $(ARTIFACT_DIR)/evidence_cache_preflight
CACHE_REQUIREMENTS_CSV := $(CACHE_PREFLIGHT_DIR)/evidence_cache_requirements.csv
CACHE_COLLECTION_REPORT := $(ARTIFACT_DIR)/evidence_cache_collection_report.json
CACHE_MERGE_REPORT := $(ARTIFACT_DIR)/evidence_cache_merge_report.json
REPORT_DIR := $(ARTIFACT_DIR)/report_bundle/main_text
SI_DIR := $(ARTIFACT_DIR)/report_bundle/si
CANDIDATE_LIBRARY_DIR := $(ARTIFACT_DIR)/candidate_library
SOURCE_COMPLETENESS_DIR := $(ARTIFACT_DIR)/source_completeness
EVIDENCE_CACHE_PACKAGE_DIR := $(ARTIFACT_DIR)/evidence_cache
EVIDENCE_CACHE_MANIFEST := $(EVIDENCE_CACHE_PACKAGE_DIR)/evidence_cache_manifest.json
PACKAGE_MANIFEST := $(ARTIFACT_DIR)/package_manifest.json
ROOT_PROVENANCE_JSON := $(ARTIFACT_DIR)/provenance_manifest.json
CLOSURE_READINESS_REPORT := $(ARTIFACT_DIR)/closure_readiness_report.json
EXPERIMENTAL_FEEDBACK_DIR := $(ARTIFACT_DIR)/closed_loop/iteration_000/experimental_feedback
MAIN_TEXT_MD := $(ARTIFACT_DIR)/report_bundle/main_text/main_text_report.md
MAIN_TEXT_PDF := $(ARTIFACT_DIR)/report_bundle/main_text/main_text_report.pdf
SI_MD := $(ARTIFACT_DIR)/report_bundle/si/supporting_information.md
SI_PDF := $(ARTIFACT_DIR)/report_bundle/si/supporting_information.pdf
MAIN_TEXT_RESOURCE_PATH := $(REPORT_DIR):$(ARTIFACT_DIR)
SI_RESOURCE_PATH := $(SI_DIR):$(ARTIFACT_DIR)

CANDIDATE_SOURCE_ARGS := $(if $(CANDIDATE_SOURCE),--candidate-source $(CANDIDATE_SOURCE),)
CANDIDATE_SOURCE_NAME_ARGS := $(if $(CANDIDATE_SOURCE_NAME),--candidate-source-name $(CANDIDATE_SOURCE_NAME),)
EXPERIMENTAL_FEEDBACK_ARGS := $(if $(EXPERIMENTAL_FEEDBACK_CSV),--experimental-feedback-csv $(EXPERIMENTAL_FEEDBACK_CSV) --experimental-feedback-data-scope $(EXPERIMENTAL_FEEDBACK_DATA_SCOPE),)
CACHE_PREFLIGHT_CACHE_DIR_ARGS := $(if $(EVIDENCE_CACHE_DIR),--cache-dir $(EVIDENCE_CACHE_DIR),)
CACHE_PREFLIGHT_MAX_ROWS_ARGS := $(if $(CACHE_PREFLIGHT_MAX_ROWS),--max-rows $(CACHE_PREFLIGHT_MAX_ROWS),)
CACHE_COLLECT_SHARD_MODE := $(if $(CACHE_COLLECT_SHARD_COUNT),1,)
CACHE_COLLECT_EFFECTIVE_CACHE_DIR := $(if $(CACHE_COLLECT_SHARD_MODE),$(CACHE_COLLECT_SHARD_CACHE_DIR),$(EVIDENCE_CACHE_DIR))
CACHE_COLLECT_CACHE_DIR_ARGS := $(if $(CACHE_COLLECT_EFFECTIVE_CACHE_DIR),--cache-dir $(CACHE_COLLECT_EFFECTIVE_CACHE_DIR),)
CACHE_COLLECT_SHARD_ARGS := $(if $(CACHE_COLLECT_SHARD_COUNT),--shard-count $(CACHE_COLLECT_SHARD_COUNT) --shard-index $(CACHE_COLLECT_EFFECTIVE_SHARD_INDEX),)
CACHE_COLLECT_OUTPUT_JSON := $(if $(CACHE_COLLECT_SHARD_MODE),$(CACHE_COLLECT_SHARD_REPORT),$(CACHE_COLLECTION_REPORT))
CACHE_MERGE_CANONICAL_DIR := $(if $(EVIDENCE_CACHE_DIR),$(EVIDENCE_CACHE_DIR),hybrid_agent_exploration/.cache/verified_discovery/$(DATASET_ID)/evidence_cache)
CACHE_MERGE_SHARD_ARGS := $(foreach dir,$(CACHE_SHARD_DIRS),--shard-cache-dir $(dir))
PSC_CANDIDATE_CLOSURE_CACHE_ARGS := $(if $(PSC_CANDIDATE_CLOSURE_CACHE_DIR),--cache-dir $(PSC_CANDIDATE_CLOSURE_CACHE_DIR),)
VERIFY_CANDIDATE_LIBRARY_ARGS := $(if $(REQUIRE_CANDIDATE_LIBRARY),--require-candidate-library,)
VERIFY_EVIDENCE_CACHE_ARGS := $(if $(REQUIRE_EVIDENCE_CACHE),--require-evidence-cache,)
VERIFY_CLOSURE_READINESS_ARGS := $(if $(REQUIRE_CLOSURE_READINESS),--require-closure-readiness,)
VERIFY_EXPERIMENTAL_FEEDBACK_ARGS := $(if $(REQUIRE_EXPERIMENTAL_FEEDBACK),--require-experimental-feedback,)

.PHONY: research-package research-package-smoke research-package-feedback-ingest research-package-cache-preflight research-package-cache-preflight-smoke research-package-cache-collect research-package-cache-collect-dry-run research-package-cache-merge research-package-psc-candidate-closure research-package-pdf research-package-verify test-research-package

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
		$(EXPERIMENTAL_FEEDBACK_ARGS) \
		$(EXTRA_RUN_ARGS)

research-package-feedback-ingest:
	@test -n "$(EXPERIMENTAL_FEEDBACK_CSV)" || { echo "EXPERIMENTAL_FEEDBACK_CSV is required for research-package-feedback-ingest" >&2; exit 2; }
	$(PYTHON) $(RUN_EXPERIMENTAL_FEEDBACK_INGESTION) \
		--iteration-dir $(ARTIFACT_DIR)/closed_loop/iteration_000 \
		--feedback-csv $(EXPERIMENTAL_FEEDBACK_CSV) \
		--output-dir $(EXPERIMENTAL_FEEDBACK_DIR) \
		--feedback-data-scope $(EXPERIMENTAL_FEEDBACK_DATA_SCOPE)

research-package-smoke:
	$(MAKE) research-package \
		DATASET_ID=$(DATASET_ID)-smoke \
		ARTIFACT_DIR=hybrid_agent_exploration/results/verified_discovery_runs/$(DATASET_ID)-smoke \
		EVIDENCE_MODE=source-columns \
		MIN_VERIFIED_ROWS=$(SMOKE_MIN_VERIFIED_ROWS) \
		TOP_K=$(SMOKE_TOP_K) \
		EXTRA_RUN_ARGS="--max-rows $(SMOKE_MAX_ROWS)"

research-package-psc-candidate-closure:
	@test -f $(PSC_CANDIDATE_CLOSURE_CACHE_DIR)/reference_cache.json || { echo "missing PSC candidate closure reference cache: $(PSC_CANDIDATE_CLOSURE_CACHE_DIR)/reference_cache.json. Run the full-source evidence cache preflight/collector first or set PSC_CANDIDATE_CLOSURE_CACHE_DIR to the completed cache." >&2; exit 2; }
	@test -f $(PSC_CANDIDATE_CLOSURE_CACHE_DIR)/molecule_cache.json || { echo "missing PSC candidate closure molecule cache: $(PSC_CANDIDATE_CLOSURE_CACHE_DIR)/molecule_cache.json. Run the full-source evidence cache preflight/collector first or set PSC_CANDIDATE_CLOSURE_CACHE_DIR to the completed cache." >&2; exit 2; }
	$(MAKE) research-package \
		DATASET_ID=$(PSC_CANDIDATE_CLOSURE_DATASET_ID) \
		ARTIFACT_DIR=$(PSC_CANDIDATE_CLOSURE_ARTIFACT_DIR) \
		EVIDENCE_MODE=external-cached \
		INPUT_SCOPE=full-source \
		CANDIDATE_SOURCE=$(PSC_CANDIDATE_SOURCE) \
		CANDIDATE_SOURCE_NAME=$(PSC_CANDIDATE_SOURCE_NAME) \
		EXTRA_RUN_ARGS="$(PSC_CANDIDATE_CLOSURE_CACHE_ARGS) $(EXTRA_RUN_ARGS)"
	$(MAKE) research-package-pdf \
		DATASET_ID=$(PSC_CANDIDATE_CLOSURE_DATASET_ID) \
		ARTIFACT_DIR=$(PSC_CANDIDATE_CLOSURE_ARTIFACT_DIR) \
		CANDIDATE_SOURCE=$(PSC_CANDIDATE_SOURCE)
	$(MAKE) research-package-verify \
		DATASET_ID=$(PSC_CANDIDATE_CLOSURE_DATASET_ID) \
		ARTIFACT_DIR=$(PSC_CANDIDATE_CLOSURE_ARTIFACT_DIR) \
		REQUIRE_CANDIDATE_LIBRARY=1 \
		REQUIRE_EVIDENCE_CACHE=1

research-package-cache-preflight:
	$(PYTHON) $(RUN_EVIDENCE_CACHE_PREFLIGHT) \
		--input $(SOURCE_TABLE) \
		--dataset-id $(DATASET_ID) \
		--source-name $(notdir $(basename $(SOURCE_TABLE))) \
		--output-dir $(CACHE_PREFLIGHT_DIR) \
		$(CACHE_PREFLIGHT_CACHE_DIR_ARGS) \
		$(CACHE_PREFLIGHT_MAX_ROWS_ARGS)

research-package-cache-preflight-smoke:
	$(MAKE) research-package-cache-preflight \
		CACHE_PREFLIGHT_MAX_ROWS=$(CACHE_PREFLIGHT_SMOKE_MAX_ROWS)

research-package-cache-collect:
	$(PYTHON) $(RUN_EVIDENCE_CACHE_COLLECTOR) \
		--requirements-csv $(CACHE_REQUIREMENTS_CSV) \
		--dataset-id $(DATASET_ID) \
		--max-requests $(CACHE_COLLECT_MAX_REQUESTS) \
		--retry-attempts $(CACHE_COLLECT_RETRY_ATTEMPTS) \
		--progress-every $(CACHE_COLLECT_PROGRESS_EVERY) \
		--output-json $(CACHE_COLLECT_OUTPUT_JSON) \
		$(CACHE_COLLECT_SHARD_ARGS) \
		$(CACHE_COLLECT_CACHE_DIR_ARGS) \
		$(EXTRA_CACHE_COLLECT_ARGS)

research-package-cache-collect-dry-run:
	$(MAKE) research-package-cache-collect \
		EXTRA_CACHE_COLLECT_ARGS=--dry-run

research-package-cache-merge:
ifeq ($(strip $(CACHE_SHARD_DIRS)),)
	@test -n "$(CACHE_SHARD_DIRS)" || { echo "CACHE_SHARD_DIRS is required for research-package-cache-merge" >&2; exit 2; }
else
	$(PYTHON) $(RUN_EVIDENCE_CACHE_MERGE) \
		--canonical-cache-dir $(CACHE_MERGE_CANONICAL_DIR) \
		$(CACHE_MERGE_SHARD_ARGS) \
		--entity-type $(CACHE_MERGE_ENTITY_TYPE) \
		--conflict-policy $(CACHE_MERGE_CONFLICT_POLICY) \
		--output-json $(CACHE_MERGE_REPORT)
endif

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
	evidence_cache_args=""; \
	if [ -f "$(EVIDENCE_CACHE_MANIFEST)" ]; then evidence_cache_args="--evidence-cache-manifest-path $(EVIDENCE_CACHE_MANIFEST)"; fi; \
	closed_loop_args=""; \
	if [ -d "$(ARTIFACT_DIR)/closed_loop/iteration_000" ]; then closed_loop_args="--closed-loop-iteration-dir $(ARTIFACT_DIR)/closed_loop/iteration_000"; fi; \
	experimental_feedback_args=""; \
	if [ -d "$(EXPERIMENTAL_FEEDBACK_DIR)" ]; then experimental_feedback_args="--experimental-feedback-dir $(EXPERIMENTAL_FEEDBACK_DIR)"; fi; \
	closure_readiness_args=""; \
	if [ -f "$(CLOSURE_READINESS_REPORT)" ]; then closure_readiness_args="--closure-readiness-report-path $(CLOSURE_READINESS_REPORT)"; fi; \
	$(PYTHON) $(ROOT_PROVENANCE_MANIFEST) \
		--verified-discovery-artifact-dir $(VERIFIED_DISCOVERY_DIR) \
		--report-dir $(REPORT_DIR) \
		--si-dir $(SI_DIR) \
		--source-completeness-dir $(SOURCE_COMPLETENESS_DIR) \
		--package-manifest-path $(PACKAGE_MANIFEST) \
		--input-path $(SOURCE_TABLE) \
		--output-path $(ROOT_PROVENANCE_JSON) \
		$$candidate_library_args \
		$$candidate_source_args \
		$$evidence_cache_args \
		$$closed_loop_args \
		$$experimental_feedback_args \
		$$closure_readiness_args

research-package-verify:
	$(PYTHON) $(VERIFY_RESEARCH_PACKAGE) \
		--package-dir $(ARTIFACT_DIR) \
		$(VERIFY_CANDIDATE_LIBRARY_ARGS) \
		$(VERIFY_EVIDENCE_CACHE_ARGS) \
		$(VERIFY_CLOSURE_READINESS_ARGS) \
		$(VERIFY_EXPERIMENTAL_FEEDBACK_ARGS)

test-research-package:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		tests/test_candidate_library_builder.py \
		tests/test_candidate_source_contract.py \
		tests/test_closed_loop_iteration_packet.py \
		tests/test_canonical_make_targets.py \
		tests/test_evidence_cache_collector.py \
		tests/test_evidence_cache_preflight.py \
		tests/test_experimental_feedback_ingestion.py \
		tests/test_research_package_contract.py \
		tests/test_research_package_runner.py \
		tests/test_run_verified_discovery_cli.py \
		tests/test_root_provenance_manifest.py \
		tests/test_verified_candidate_discovery.py
