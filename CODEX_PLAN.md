# AgentScale — Codex Implementation Plan

## Mission
Build an open-source, framework-extensible **global resource scheduler for Agentic Reinforcement Learning** that coordinates rollout concurrency, inference GPU capacity, sandbox CPU/memory capacity, and training GPU capacity. Optimize **valid, sufficiently fresh training samples per unit cost**, subject to resource, safety, and policy-staleness constraints.

This is an orchestration/control-plane project, **not** a new RL algorithm, inference engine, container runtime, or Kubernetes scheduler.

## Non-negotiable engineering rules
1. Inspect the repository before editing. If it is empty, bootstrap a minimal Python package.
2. Work in small, testable vertical slices. Run relevant tests after each slice.
3. Keep the simulator and scheduler usable **without GPUs, Ray, verl, vLLM, Kubernetes, or cloud credentials**.
4. Use typed protocols for runtime adapters. No hard dependency on unstable private APIs.
5. Do not claim that an integration works unless it has been executed and tested against a pinned upstream version.
6. Never execute untrusted sandbox workloads directly on the host in production mode. The local fake sandbox must be simulation-only.
7. All plans must be validated before execution; support dry-run, idempotency, timeouts, safe-point checks, and rollback or compensation.
8. Do not interrupt an active distributed training gang or evict in-flight inference without an explicit safe mechanism.
9. Maintain reproducibility: fixed seeds, configuration snapshots, workload traces, metric definitions, and benchmark scripts.
10. At the end of each milestone, report changed files, commands run, test results, limitations, and next steps. Do not silently skip failing tests.

## Proposed stack
- Python 3.11+; `pyproject.toml`; `pytest`, `ruff`, `mypy` (or pyright).
- Pydantic for configuration and data validation (optional if stdlib dataclasses suffice).
- SimPy or a small deterministic discrete-event simulator; avoid excessive framework complexity.
- Prometheus client for metrics, optional local dashboard.
- verl for RL orchestration/training; vLLM for inference; Ray for execution; sandbox backend through an adapter.
- Kubernetes/Kueue only after the Ray-based MVP is validated.
- Pin upstream versions/commits after checking compatibility; do not assume APIs exist based on documentation alone.

## Architecture

```text
                      AgentScale Control Plane
  Telemetry --> State Estimator --> Capacity Predictor --> Optimizer
       ^                                            |           |
       |                                            +--> Validator
       |                                                    |
       +----------- Runtime Adapters <-- Plan Executor <----+
                          |         |          |
                      verl/Ray    vLLM     Sandbox Pool
                          |         |          |
                     Training    Inference  CPU/Memory
```

Keep **policy** separate from **actuation**. The scheduler proposes a desired allocation; the executor reconciles it with observed state. Adapters report capabilities and unsupported operations explicitly.

## Repository layout

```text
agentscale/
  pyproject.toml
  README.md
  LICENSE
  agentscale/
    models/           # typed resource, workload, state, plan models
    telemetry/        # collectors, normalization, metrics
    simulator/        # deterministic events, queues, workers, traces
    predictor/        # service demand and bottleneck estimates
    scheduler/        # baseline, bottleneck-aware, cost-aware policies
    controller/       # planning loop, validation, reconciliation
    adapters/         # protocols, fake, ray, verl, vllm, sandbox
    api/              # optional read-only state/plan endpoints
  configs/
  examples/
  benchmarks/
  tests/
  docs/
```

## Resource and workload model

Represent capacities with explicit units:
- GPUs: whole-device count, GPU type, memory, topology, and gang-size constraints.
- CPU: millicores or cores; memory: bytes; network/IO as optional capacities.
- Inference: requests/s, prompt tokens, decode tokens, KV-cache footprint, batching and queue delay.
- Sandbox: execution slots, CPU-seconds, memory footprint, cold-start time, IO, failures and timeouts.
- Rollout: active trajectories, step count, tool calls, inference calls, completion latency, valid trajectory rate.
- Training: samples consumed per update, step time, trainer waiting time, policy version, allowed staleness.

Do not compare raw tokens/s, executions/s, and samples/s as if they were equivalent. Convert service demand to a common **valid fresh trajectories/s** or **training samples/s** basis using measured per-trajectory demands and an explicit sample-to-trajectory mapping.

### Core types
- `ClusterCapacity`
- `WorkloadDemand`
- `PipelineState`
- `PolicyVersionInfo`
- `ResourceAllocation`
- `ResourcePlan`
- `PlanValidationResult`
- `AdapterCapabilities`
- `ExecutionResult`

Every metric includes unit, timestamp, provenance, and optional confidence/validity metadata. Detect missing/stale telemetry and fail safe.

## Milestone 0 — Bootstrap and contracts
**Deliverables:** installable package, CLI (`agentscale simulate`, `agentscale benchmark`, `agentscale plan --dry-run`), config schema, typed interfaces, CI.

**Tasks:**
1. Initialize package, lint/type/test tooling and GitHub Actions.
2. Define domain models and unit conventions.
3. Define `TelemetryProvider`, `ResourceAdapter`, `SchedulingPolicy`, `PlanExecutor` protocols.
4. Implement `FakeAdapter` with explicit capability flags.
5. Write architecture decision records (ADRs) for simulator-first and Ray-first choices.

**Acceptance:** clean install; lint, type check and unit tests pass without external services; sample config validates; dry-run never mutates resources.

## Milestone 1 — Deterministic discrete-event simulator
**Deliverables:** reproducible four-stage Agentic RL simulator and workload traces.

**Tasks:**
1. Model an agent trajectory as repeated `inference -> sandbox/tool -> observation` steps followed by verification/reward and experience buffering.
2. Model inference capacity, batching approximation, request queue and variable token lengths.
3. Model sandbox slots, CPU/memory admission, cold starts, heavy-tailed durations and timeouts.
4. Model training as a consumer of sufficiently fresh experience with step durations, GPU gang-size requirements and policy updates.
5. Model asynchronous overlap, queueing, resource switching costs, and optional GPU sharing at safe boundaries.
6. Implement seeded synthetic workloads: inference-bound, sandbox-bound, training-bound, and time-varying bottlenecks.
7. Export event traces and metrics to CSV/JSON; produce a concise benchmark summary.

**Acceptance:** deterministic repeatability with fixed seed; resource conservation invariants; no negative queue lengths; no overcommit beyond configured policy; training cannot consume nonexistent samples; valid samples respect configured policy-staleness bounds.

## Milestone 2 — Telemetry and bottleneck diagnosis
**Deliverables:** normalized pipeline snapshot and explainable bottleneck reports.

**Tasks:**
1. Record queue depth, waiting time, service time, utilization, completion rate, buffer occupancy, trainer starvation and policy staleness.
2. Estimate per-trajectory inference and sandbox demand using EWMA or windowed robust statistics.
3. Calculate approximate stage capacity in comparable units and identify likely critical bottlenecks.
4. Distinguish CPU saturation, sandbox memory admission pressure, GPU inference saturation, and training data starvation.
5. Include telemetry age and confidence; avoid actions based on stale/insufficient data.
6. Expose Prometheus metrics without high-cardinality trajectory labels.

**Acceptance:** benchmark scenarios identify expected bottlenecks; tests cover missing, delayed, contradictory and bursty telemetry.

## Milestone 3 — Scheduling policies
**Deliverables:** three interchangeable policies and explainable decisions.

**Policies:**
- `StaticPolicy`: fixed allocation baseline.
- `IndependentAutoscalePolicy`: per-stage threshold scaling baseline with no cross-stage coordination.
- `BottleneckAwarePolicy`: joint rollout admission, sandbox worker count, and eligible inference capacity control.

**Algorithm v1:**
1. Read a validated `PipelineState` snapshot.
2. Generate a bounded set of feasible discrete allocation candidates.
3. Estimate valid fresh sample throughput, resource cost and transition cost for each candidate.
4. Reject candidates violating GPU/CPU/memory, training gang, policy-staleness, safety, or adapter-capability constraints.
5. Choose the best candidate under a configurable objective and stability margin.
6. Apply cooldowns, hysteresis and minimum dwell times to avoid oscillation.
7. Emit a structured explanation: bottleneck, evidence, rejected alternatives, predicted benefit and risks.

**Initial objective:** maximize `predicted_valid_fresh_samples_per_sec - lambda_cost * cost_per_sec - lambda_switch * transition_cost`, with hard quality and safety constraints. Do not use average reward as a short-window proxy for learning progress without evidence.

**Acceptance:** policies run on identical traces; no invalid allocation; bounded switching frequency; the global policy beats at least one baseline in a deliberately constructed dynamic-bottleneck scenario without regressing on safety. Do not invent a percentage improvement.

## Milestone 4 — Control loop and safe execution
**Deliverables:** controller that converts proposed plans into safe resource actions.

**Tasks:**
1. Separate desired allocation, observed allocation and in-progress transitions.
2. Implement plan validation, dry-run, idempotency keys, generation numbers and audit logs.
3. Implement a reconcile loop with timeouts and compensation on partial failures.
4. Define capability-aware operations: `set_rollout_concurrency`, `set_sandbox_replicas`, `set_inference_capacity`, `request_training_safe_point`.
5. Enforce no disruptive GPU reassignment without runtime-confirmed safe point.
6. Add fault injection: adapter timeout, stale state, partial scale-out, failed scale-in, process restart.

**Acceptance:** repeated plan application is safe; failures do not silently corrupt state; restart reconciliation works; unsupported operations fail explicitly.

## Milestone 5 — Real runtime integrations (incremental)
**Order:**
1. Ray adapter: observe worker/resource state and implement supported sandbox/rollout scaling operations.
2. Sandbox adapter: use a real isolated backend, enforce per-task CPU/memory/time/network policies; avoid host execution of untrusted code.
3. vLLM adapter: observe inference metrics; use documented deployment/replica controls rather than assuming request-level APIs.
4. verl adapter: consume rollout/training metrics and coordinate only documented supported dynamic-scheduling or safe-point hooks.

**Rules:**
- Verify each upstream API against a pinned version before implementing.
- Use feature flags for experimental operations.
- If changing GPU allocation is not safely supported, expose it as recommendation-only until a proven actuator exists.
- Never require production integrations for simulator tests.

**Acceptance:** an end-to-end small-model coding-agent RL run with real inference, isolated sandbox execution, experience collection and training; control loop demonstrably adjusts at least rollout concurrency and sandbox capacity. GPU sharing is optional until safely validated.

## Milestone 6 — Benchmark and ablations
**Compare:** StaticPolicy, IndependentAutoscalePolicy, upstream verl dynamic scheduling where applicable, AgentScale BottleneckAwarePolicy.

**Scenarios:** inference-bound; sandbox CPU-bound; sandbox memory-bound; training-bound; mixed/shifted bottlenecks; heavy-tail rollout durations; sandbox cold-start spikes; policy-sync overhead.

**Metrics:**
- Primary: valid fresh training samples/s at fixed budget; cost per valid fresh sample.
- Secondary: rollout p50/p95/p99, inference and sandbox queue latency, trainer idle time, utilization, switching overhead, allocation oscillations.
- Training-quality guardrails: policy staleness, sample distribution, failed trajectories; later time-to-target-reward with controlled seeds.

**Method:** fixed hardware and workload budgets, multiple seeds, warm-up, confidence intervals, documented upstream versions, no cherry-picked traces. Publish raw data, reproducible commands and negative results.

**Acceptance:** one-command simulator benchmark; documented real-runtime benchmark; claims limited to measurements actually obtained.

## Milestone 7 — OSS release
- Apache-2.0 or another explicitly chosen compatible license; verify dependency licenses.
- Architecture diagrams, installation guide, tutorial, design rationale, limitations, troubleshooting.
- Contribution guide, issue templates, security policy and code of conduct.
- CI matrix and example benchmark results.
- Release v0.1.0 only after Milestones 0–4; mark real integrations experimental until Milestone 5 validation.

## First Codex execution request
Start with **Milestone 0 only**, then implement the smallest vertical slice of Milestone 1: one simulated rollout alternates inference and sandbox calls, produces an experience item, and a simulated trainer consumes it. Do not attempt all milestones in one pass.

Before coding, inspect the current repository and report a concise implementation outline. Then create the files, tests and documentation, run the commands, and summarize results. If any upstream integration is unavailable, keep it behind a protocol and do not fabricate working support.

## Definition of Done for v0.1
- Fully offline deterministic four-workload simulator.
- At least three scheduling policies and four benchmark scenarios.
- Explicit resource/quality constraints and safe plan validation.
- Auditable dry-run controller with tests for failures and oscillation.
- Reproducible benchmark CLI and machine-readable results.
- Documentation that clearly distinguishes implemented features from planned real-runtime integrations.
