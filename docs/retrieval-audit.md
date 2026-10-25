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

## Delivered files and validation

- Phase 1 commit `e56d00d`: this audit, retrieval models/protocol, stable fake backend
  and contract tests. Baseline 112 tests and 19 retrieval tests passed.
- Phase 2–3 commit `5eaad25`: estimators, bounded planner, constrained optimizer and
  constraint/feedback tests. 37 cumulative retrieval tests passed; Ruff/Mypy passed.
- Phase 4–5 commit `33ce6d4`: shared ledger, policy/adapter fences, retrieval controller,
  optional simulator callbacks and integration/failure tests. 65 cumulative retrieval
  tests passed; the preceding full offline run passed 175 tests. Final tests also
  exercise measured RL rollback deferral/recovery and observed cost misses.
- Final slice: independent retrieval CLI and combined simulator/benchmark runner,
  42 raw paired synthetic trials, CI smoke commands, architecture/compatibility/
  methodology docs and unreleased changelog. No dependency or version changes.

Executed with the repository's Python 3.12.12 virtual environment:

```bash
python -m pytest -q tests/test_retrieval*.py
# 80 passed
python -m pytest -q -m 'not ray and not docker'
# 192 passed, 10 deselected (includes all 112 baseline offline/local-HTTP tests)
python -m ruff check agentscale tests examples tools
python -m mypy agentscale
# Both passed; 26 source modules type checked.
python -m agentscale.cli retrieval-benchmark --duration 30 --seeds 42 43 44 --output benchmarks/retrieval-synthetic.json
# 42 trials; a second independent run compared equal to the checked-in parsed JSON.
python tools/build_release.py --output /private/tmp/agentscale-retrieval-release
# Source + wheel build and Twine checks passed.
```

A fresh virtual environment installed only the wheel with `--no-deps --no-index`.
From outside the checkout, retrieval-plan, retrieval-simulate and existing RL plan
commands all produced valid JSON. Import resolution was verified in site-packages.
The sandbox required loopback permission for the existing local HTTP fixture; tests
were rerun with it enabled. No test was removed or weakened to avoid that failure.

The optional runtime compatibility results are recorded separately below. Retrieval
validation itself remains fake-only; real retrieval, labeled quality evaluation and
real end-to-end RAG are pending, as are all preexisting GPU validation gates.

```bash
python -m pytest -q -m 'ray or docker'
# 10 passed (4 real Ray + 6 real Docker tests), 192 deselected.
```

Thus all 202 collected tests passed across the two complementary runs, including
80 new retrieval tests and every preexisting test. Ray emitted its upstream
accelerator-visibility FutureWarning. These runtime regressions check compatibility
of the existing integrations; they do not validate a real retrieval backend.
