# Harness Engineering Comparison — Literature Review

## What is "Harness Engineering"?

Recent work on **harness engineering** (Lee et al., 2026) defines the *harness* as:
> "The surrounding system logic that governs storage, retrieval, and presentation — as well as on model weights."

In ML research, the harness problem spans literature review, hypothesis generation, experimentation, internal critique, manuscript preparation, and responses to external feedback.

---

## Key Systems Compared

### 1. RepoForge (TOSEM 2025) — SWE Agent Training
- **Domain**: Software engineering agent training
- **Harness components**:
  - Storage-efficient sandboxing (14× storage reduction)
  - Ray-powered distributed evaluation harness
  - Automated SPICE-based difficulty labeling
  - Bubble-free RL scaffold
- **Key insight**: Evaluation harness must spin up isolated Docker environments, install dependencies, apply patches, run tests, record pass/fail — essentially an automated CI/CD system
- **Performance**: 70% faster evaluation via distributed harness; 19,000× cheaper labeling

### 2. Autonomous Research via Adversarial Multi-Agent Collaboration (arXiv 2026)
- **Domain**: Fully autonomous ML research
- **Harness components**:
  - Heterogeneous multi-agent debate (generator + validator from different model families)
  - Decoupled workflows with saved intermediate states
  - Explicit system-level checks on experimental integrity
- **Key insight**: "Any long-term task performed by a single agent is unreliable"
- **Limitations addressed**: Same-model self-refinement leaves correlated errors uncaught; tightly coupled workflows resist stage replacement

### 3. AI Scientist / AI Scientist v2 (2024–2025)
- **Domain**: End-to-end autonomous research
- **Harness**: Automates idea generation → manuscript drafting
- **Gap**: No explicit quality checks; no heterogeneous review

### 4. Agent Laboratory (2025)
- **Domain**: Autonomous research with human oversight
- **Harness**: Adds human-in-the-loop checkpoints
- **Gap**: Still assembled manually across separate systems

### 5. DeepRed / CTF Benchmark Harness (2026)
- **Domain**: Cybersecurity agent evaluation
- **Harness components**:
  - Isolated virtual environments (attacker + target VMs)
  - Log-based partial scoring with binary checkpoints
  - Two-stage summarize-then-judge LLM auto-labeling
- **Key insight**: Binary success metrics are too coarse; need milestone-based partial credit

---

## Our Harness (src/harness/)

| Component | Purpose | Literature Parallel |
|-----------|---------|---------------------|
| `Guardrail` | Input/output validation, duplicate detection, reference integrity | DeepRed checkpoint validation |
| `RetryPolicy` | Auto-retry with exponential backoff on transient failures | RepoForge fault tolerance |
| `ExperimentValidator` | Post-experiment quality thresholds (R², RMSE, sample size) | Adversarial multi-agent integrity checks |
| `Sandbox` (planned) | Isolated execution environment | RepoForge Docker sandboxing |
| `Observability` (planned) | Structured logging, metrics, trajectory recording | DeepRed interaction logging |

**Gaps vs. state-of-the-art**:
1. No distributed evaluation (RepoForge uses Ray)
2. No heterogeneous multi-agent review (adversarial collaboration)
3. No partial-credit scoring for experiment steps
4. No automated difficulty assessment (SPICE equivalent)

**Strengths**:
1. Lightweight, Python-native, no container overhead
2. Integrated with RDKit/sklearn pipeline
3. Checkpoint-resilient (orchestrator supports resume)
4. Domain-specific: PSC molecular additive prediction

---

## Recommendations for Future Work

1. **Add adversarial review**: Run SHAP analysis + residual diagnostics as independent "reviewer" agents
2. **Distributed harness**: Ray-based parallel evaluation for large virtual screening libraries
3. **Partial credit scoring**: Decompose pipeline into binary checkpoints (data load → feature gen → train → evaluate → report)
4. **SPICE-like difficulty**: Auto-assess prediction difficulty from molecular scaffold diversity
