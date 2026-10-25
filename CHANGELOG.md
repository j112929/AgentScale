# Changelog

## Unreleased

- Optional deterministic retrieval plan templates and fake backend, separate
  quality/latency/cost estimators and a constrained adaptive optimizer.
- Queue-aware pending-request control, shared RL reservations, policy/adapter
  fences and opt-in same-clock simulator hooks, including safe RL rollback.
- Retrieval dry-run/simulation/benchmark commands, paired synthetic results,
  compatibility tests and explicit real-RAG evaluation boundaries.

## 0.1.0rc1

- Deterministic queued simulator, guarded dynamic scheduling and compensation.
- Optional pinned Ray Actor pool with real-process integration tests.
- Restricted ephemeral Docker execution, resource/time/output bounds and queue telemetry.
- Coding repair loop with protected file selection and containerized test feedback.
- vLLM HTTP inference and per-model Prometheus telemetry contracts.
- Native verl 0.6.1 BaseTool plugin; policy-versioned experience and training transaction contracts.
- Paired synthetic benchmarks, measured local container trials and external RL comparison tooling.
- Apache-2.0 license, contribution/security guidance, release checks and reproducible packaging.

This release candidate validates CPU/container flows. Real vLLM GPU inference, verl
optimizer updates, GPU sharing and verl-native RL performance comparisons remain unvalidated.
