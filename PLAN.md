# Project Cleanup Plan — AutoML_EDA (qspr-analysis branch)

## Current State Summary

The repo has **three layers of work accumulated** without clear separation:

| Layer | Directory | Origin | Status |
|-------|-----------|--------|--------|
| Initial EDA/AutoML | `automl/`, `data/`, `figures/`, `models/`, `report/`, `reports/`, `scripts/`, `tables/`, `virtual_screening/`, `fingerprints/` | Commits `df9a744`–`8b3594b` | Tracked in git |
| Hybrid Agent Pipeline | `hybrid_agent_exploration/` | Commit `80ff424` + uncommitted work | Partially staged, partially dirty |
| Third-party clone | `ML-Master/`, `autogluon_models/` | Local only (gitignored) | Untracked |

**Git status**: 20 staged files, 16 unstaged modifications (including 2 deleted files), ~30 untracked files/dirs.

**Key problems**:

1. **Top-level dirs vs `hybrid_agent_exploration/` overlap** — both contain models, features, figures, scripts for the same QSPR task.
2. **Large binary blobs tracked in git** — `perovskite_cleaned.csv` (81 MB), PDFs (~5 MB each), PNGs (~2 MB each). `.git` is 49 MB.
3. **Half-staged, half-dirty state** — `reporting/*.py` files are both staged (new) AND modified (unstaged), meaning the staged version is already stale.
4. **Reference PDFs in repo** — `hybrid_agent_exploration/ref/` has 8.3 MB of papers that should be fetched externally.
5. **No `.gitattributes`** — Git LFS not configured for large files.
6. **`ref/JPCL.md` deleted but still tracked** — incomplete cleanup.
7. **`results/comparison_reports/final_report.md` deleted but tracked** — same issue.

---

## Cleanup Actions

### Phase 1: Commit Hygiene — Fix the Staging Area

**Priority: HIGH | Risk: LOW**

The current staging area is stale — files were staged then modified further.

```bash
# 1. Unstage everything to start clean
git reset HEAD hybrid_agent_exploration/

# 2. Review all changes together
git diff hybrid_agent_exploration/          # unstaged
git diff --cached hybrid_agent_exploration/ # staged (should be empty now)

# 3. Stage only the final state of each file
git add hybrid_agent_exploration/src/reporting/
git add hybrid_agent_exploration/src/generate_reports.py
git add hybrid_agent_exploration/src/generate_research_story.py
git add hybrid_agent_exploration/src/pipeline.py
git add hybrid_agent_exploration/src/worker_agent.py
# ... add each file intentionally

# 4. Handle deletions explicitly
git add hybrid_agent_exploration/ref/JPCL.md
git add hybrid_agent_exploration/results/comparison_reports/final_report.md
```

### Phase 2: Remove Large Tracked Files from Git History

**Priority: HIGH | Risk: MEDIUM (rewrites history)**

The 81 MB CSV and multi-MB PNGs/PDFs bloat the repo. Once merged to main, every clone pays this cost.

**Option A (recommended for pre-merge):** Remove from tracking, add to `.gitignore`, accept they exist in history.

```bash
# Remove from tracking (keep locally)
git rm --cached data/processed/perovskite_cleaned.csv   # 81 MB
git rm --cached data/processed/perovskite_ml_ready.csv  # 4.5 MB

# Add to .gitignore
echo "data/processed/perovskite_cleaned.csv" >> .gitignore
echo "data/processed/perovskite_ml_ready.csv" >> .gitignore
```

**Option B (aggressive):** Use `git-filter-repo` to purge from history entirely. Only worth it if this repo will be shared widely.

```bash
pip install git-filter-repo
git filter-repo --path data/processed/perovskite_cleaned.csv --invert-paths
git filter-repo --path data/processed/perovskite_ml_ready.csv --invert-paths
```

**For images:** Consider keeping only final publication-quality figures in git. Move intermediate EDA figures to a `.gitignore`'d directory or external storage.

### Phase 3: Consolidate Directory Structure

**Priority: MEDIUM | Risk: LOW**

Two options:

#### Option A: Top-level cleanup (keep `hybrid_agent_exploration/` as the main module)

Move everything into `hybrid_agent_exploration/` and remove the stale top-level dirs:

```bash
# These top-level dirs are superseded by hybrid_agent_exploration/
git rm -r automl/
git rm -r models/            # superseded by hybrid_agent_exploration/models/
git rm -r fingerprints/      # gitignored, just confirm removal
git rm -r report/            # superseded by hybrid_agent_exploration/results/
git rm -r reports/           # superseded by hybrid_agent_exploration/src/reporting/
git rm -r scripts/           # superseded by hybrid_agent_exploration/scripts/
git rm -r tables/            # regenerated outputs
git rm -r virtual_screening/ # if no longer needed

# Keep these (shared data):
#   data/          — source datasets
#   figures/       — final EDA figures (consider gitignoring intermediates)
#   ML-Master/     — already gitignored
#   autogluon_models/ — already gitignored
```

#### Option B: Flatten into a cleaner top-level structure

Rename `hybrid_agent_exploration/` to `src/` or keep it but move shared data up:

```
AutoML_EDA/
├── data/              # Raw + processed data (gitignored CSVs > 1MB)
├── src/               # All Python source code (renamed from hybrid_agent_exploration/src/)
├── configs/           # Search spaces, experiment configs
├── article/           # Manuscript + figures
├── results/           # Experiment outputs (mostly gitignored)
├── ref/               # Reference papers (gitignored)
├── docs/              # Documentation
├── ML-Master/         # Third-party (gitignored)
├── .gitignore
└── CLAUDE.md
```

### Phase 4: Reference Files Cleanup

**Priority: MEDIUM | Risk: LOW**

```bash
# Remove tracked reference papers from git (keep locally)
git rm --cached hybrid_agent_exploration/ref/JPCL.md  # already deleted locally

# Ensure ref/ PDFs are gitignored
echo "hybrid_agent_exploration/ref/*.pdf" >> .gitignore

# Add a README in ref/ explaining where to get papers
cat > hybrid_agent_exploration/ref/README.md << 'EOF'
# Reference Papers

- JPCL article: DOI link or URL
- SI document: DOI link or URL
EOF
```

### Phase 5: Untracked Files Disposition

**Priority: MEDIUM | Risk: LOW**

Decide fate of each untracked file/directory:

| Path | Recommendation | Reason |
|------|---------------|--------|
| `hybrid_agent_exploration/.opencode/` | Add to `.gitignore` | IDE/tool state, 57 MB |
| `hybrid_agent_exploration/results/harness_logs/` | Add to `.gitignore` | Logs, transient |
| `hybrid_agent_exploration/results/research_stories/` | Review → keep or gitignore | May contain final outputs |
| `hybrid_agent_exploration/src/evaluation/` | Stage and commit | New source code |
| `hybrid_agent_exploration/src/features/` | Stage and commit | New source code |
| `hybrid_agent_exploration/src/harness/` | Stage and commit | New source code |
| `hybrid_agent_exploration/src/screening/` | Stage and commit | New source code |
| `hybrid_agent_exploration/src/reporting/claim_guardrail.py` | Stage and commit | New source code |
| `hybrid_agent_exploration/src/reporting/mechanism_dictionary.yaml` | Stage and commit | New config |
| `hybrid_agent_exploration/article/build_pdf.py` | Stage and commit | Build script |
| `hybrid_agent_exploration/article/manuscript_v2_ordered.md` | Stage and commit | Manuscript version |
| `hybrid_agent_exploration/docs/*.md` | Stage and commit | Documentation |
| `hybrid_agent_exploration/docs/references.bib` | Stage and commit | Bibliography |
| `hybrid_agent_exploration/docs/jpcl_*_raw.txt` | Gitignore or delete | Intermediate scratch files |
| `hybrid_agent_exploration/docs/references_raw.json` | Gitignore or delete | Intermediate file |

### Phase 6: `.gitignore` Consolidation

**Priority: LOW | Risk: LOW**

Currently there are **two** `.gitignore` files with overlapping rules. Consolidate:

```bash
# Root .gitignore should contain:
# - All large binary patterns (*.xlsx, *.pkl, *.pdf, etc.)
# - All tool/IDE patterns (.venv/, node_modules/, .opencode/)
# - All generated artifact patterns (autogluon_models/, ML-Master/, fingerprints/)
# - Data files (data/processed/*.csv over threshold)

# hybrid_agent_exploration/.gitignore should only contain:
# - Patterns specific to that module's workflow
```

### Phase 7: Branch Hygiene

**Priority: LOW | Risk: NONE**

```bash
# Review remote branches — many appear stale
git branch -r | grep -v main | grep -v qspr-analysis

# Delete remote branches that are merged or abandoned:
# - origin/feat/issue-13-interpretability
# - origin/feat/issue-15-generalization
# - origin/feat/issue-16-multitask
# - origin/feat/issue-18-gnn-gan
# - origin/feat/issue-9-multi-objective
# - origin/fix/ci-cleanup-api
```

---

## Recommended Execution Order

1. **Phase 1** — Fix staging area (do this first to avoid losing work)
2. **Phase 5** — Decide on untracked files
3. **Phase 4** — Clean up reference files
4. **Phase 2** — Remove large files from tracking
5. **Phase 6** — Consolidate `.gitignore`
6. **Phase 3** — Restructure directories (most disruptive, do last)
7. **Phase 7** — Clean up remote branches

After all phases: create a single clean commit (or a few logical commits) and merge to `main`.

---

## Disk Space Impact

| What | Current Size | After Cleanup |
|------|-------------|---------------|
| `.git/` | 49 MB | 49 MB (history) / ~5 MB (after filter-repo) |
| `autogluon_models/` | 109 MB | gitignored (0 in repo) |
| `ML-Master/` | 86 MB | gitignored (0 in repo) |
| `.venv/` | 249 MB | gitignored (0 in repo) |
| `ref/` PDFs | 8.3 MB | gitignored (0 in repo) |
| `data/processed/*.csv` | 85 MB | gitignored (0 in repo) |
| **Tracked repo size** | ~105 MB | ~10-15 MB |
