# Retrieval extension audit (2026-10-25)

## Baseline and unmerged work

Audited README.md, CODEX_PLAN.md, docs/architecture.md, pyproject.toml, all module
interfaces and existing test coverage before implementation. Local checkout was
clean at `64123c2d087cf9646966f85175f5170dc361296e`. GitHub PR #2 is open,
not merged: `codex/container-runtime-integrations` targets
`codex/dynamic-controller-ray-sandbox` (PR #1). Remote main remains `a82f451`.
The retrieval branch is stacked on PR #2's exact head; its PR targets that branch.
Neither existing branch is rewritten or merged by this change.

## Observed implementation

- `models.py`: frozen Capacity/Allocation/Workload/Snapshot/Plan, SchedulingPolicy
  and ResourceAdapter protocols; `validate` checks whole GPU/CPU/memory capacity.
  There is no extensible stage registry or retrieval resource field.
- `policies.py`: StaticPolicy, IndependentAutoscalePolicy and bounded discrete
  BottleneckAwarePolicy. Predictions are synthetic fresh trajectories/s; costs
  are arbitrary units. Allocation GPUs belong to inference and training.
- `controller.py`: periodic telemetry, confirmations, hysteresis, cooldown,
  safe-point deferral, compensation, measured-throughput rollback. Its objective
  is RL throughput, so retrieval-quality decisions cannot safely reuse that
  objective or its Snapshot fields.
- `simulator.py`: seeded queued four-stage RL simulation. Retrieval is absent.
- `ray_adapter.py` and `sandbox.py`: optional real CPU Actor/container pools;
  active work drains before scale-in. They do not execute retrieval requests.
- `vllm_adapter.py` and `experience.py`: HTTP contracts and training lifecycle
  hooks, not a verified GPU training or resource migration actuator.
- `cli.py`: simulate/plan/benchmark plus optional runtime commands; preserve
  their defaults and result schemas. Package has no mandatory runtime dependency.
- Tests cover simulator determinism, policies, capacity, controller failure paths,
  local HTTP, Docker and Ray. Use the existing Python 3.12 virtual environment.
  The initial system-Python run passed 111 tests, with one loopback bind error
  caused by the execution sandbox. The Python 3.12 rerun with loopback enabled
  passed all 112 tests (10 optional Ray/Docker tests deselected).

## Implementation sequence and boundary

1. Typed retrieval queries, four strategy plans, requests/results and a stable-hash
   fake backend. Explicit synthetic provenance, CPU execution by default.
2. Separate quality, latency and cost models; bounded candidate planner and hard
   constrained optimizer. No feasible candidate means reject/defer, never silently
   violate a quality, deadline, cost or resource requirement.
3. Residual capacity from an existing RL Snapshot, a shared reservation ledger,
   and an opt-in SchedulingPolicy wrapper. RL owns its allocation; reservations
   must be subtracted before proposing RL changes, and current allocations must
   be rechecked before retrieval admission. No independent GPU actuator.
4. Retrieval-only closed-loop controller: fresh queue snapshots, delayed admission,
   bounded pending queue, immutable in-flight plans, per-query confirmations and
   cooldown, deterministic completion and estimator feedback. Existing RL controller
   and simulator remain unchanged.
5. Add separate retrieval CLI/paired benchmark, tests, architecture/compatibility
   docs and synthetic raw results. Compare all fixed strategies, adaptive one-shot,
   adaptive closed-loop and a queue-awareness ablation on identical arrivals/seeds.

This slice does not implement real search engines, relevance labels, LLM answers,
GPU execution or real RAG quality evaluation. Synthetic quality/success, modeled
CPU/GPU utilization and accounting costs must never be labeled Recall@K, nDCG,
measured agent task success, real device utilization or money saved.
