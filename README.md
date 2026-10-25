# AgentScale (experimental)

**Offline-first global resource scheduling experiments for Agentic RL.**

AgentScale simulates rollout → inference → sandbox → training and continuously
replans resource allocation as demand changes. It includes a guarded control loop
and an optional **Ray 2.50.1 Actor pool** for trusted synthetic sandbox operations.
A restricted Docker backend now executes real source files and tests. vLLM HTTP and
verl tool interfaces are experimental; full GPU training and migration remain unvalidated.


## v0.1 release candidate: local runtime integrations

Version **0.1.0rc1** adds real restricted Docker execution, a coding-repair loop,
vLLM inference/metrics clients, a native verl 0.6.1 tool plugin, policy-versioned
experience, measured container benchmarks and Apache-2.0 licensing. GPU integration
validation remains open; this is a CPU/container-validated release candidate.

```bash
python -m pip install -e '.[dev,release]'
docker pull python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
python examples/container_coding.py
pytest -q -m docker
python tools/check_verl_contract.py --download --execute
agentscale container-benchmark --output container-results.json
# With a separately provisioned vLLM server:
agentscale vllm-inspect --url http://127.0.0.1:8000 --model agentscale-demo
python examples/container_coding.py --vllm-url http://127.0.0.1:8000 --model agentscale-demo
python tools/build_release.py --output dist
```

The default coding demo uses a **scripted inference fixture** and real containers.
Passing `--vllm-url` replaces that fixture with real HTTP inference. The verl contract
check loads genuine pinned BaseTool/schema modules without importing a full trainer.
See [runtime integration and limits](docs/runtime-integrations.md),
[measured container results](docs/container-benchmarks.md),
[release procedure](docs/release.md), [contribution guide](CONTRIBUTING.md) and
[security scope](SECURITY.md).

## Quick start (no Ray or GPUs required)

```bash
python -m pip install -e '.[dev]'
agentscale simulate --scenario dynamic --policy global --duration 600 --trace --output trace.json
agentscale plan --scenario sandbox --dry-run
agentscale benchmark --duration 600 --seeds 42 43 44 --output closed-loop.json
pytest -q -m 'not ray and not docker'
ruff check agentscale tests examples
mypy agentscale
```

`simulate` uses periodic closed-loop scheduling by default. `--one-shot` preserves
initial-only planning for comparison; `--interval` sets the telemetry period.
`plan` only proposes an allocation and never starts a simulation or runtime.
The `dynamic` profile switches from inference-bound to sandbox-bound to training-bound
at one-third and two-thirds of the requested duration. Other scenarios are
`balanced`, `inference`, `sandbox` and `training`.

## What is implemented

- Seeded event simulation with explicit stage queues, arrival admission, batch training,
  policy-staleness filtering and live workload shifts.
- Static, independent queue-based and global bottleneck-aware planners.
- Periodic telemetry, stale-data rejection, repeated proposal confirmation, relative
  benefit hysteresis, cooldown, switching-cost checks and safe-point draining.
- Compensation after partial apply failures and windowed performance rollback;
  failed rollback blocks further writes until operator reconciliation.
- An optional Ray pool with explicit CPU/memory reservations, readiness deadlines,
  transactional scale-out, idle-only scale-in, bounded task queues, timeout handling
  and replacement of failed busy Actors. GPU changes are rejected.
- Offline fault-injection tests, real local Ray integration tests, JSON traces,
  paired-seed benchmarks and a stability ablation.

## Try real Ray Actors

Use Python 3.11–3.13 (the pinned Ray wheel is required).

```bash
python -m pip install -e '.[dev,ray]'
python examples/ray_sandbox.py
pytest -q -m ray tests/test_ray_integration.py
```

The demo starts a local CPU-only Ray cluster, submits eight trusted synthetic tasks,
and lets the controller grow the pool from one to two Actors. It shuts the cluster
down on exit. Ray is imported only when this optional adapter is constructed.
See [the Ray guide](docs/ray-sandbox.md) for ownership, failure and capability details.

## Measured synthetic example

600 simulated seconds, seeds 42/43/44, identical capacity of 8 GPUs, 64 CPU cores
and 128 GiB; full runs include warm-up, draining and switching downtime:

| Dynamic workload policy | Mean trained samples/s | Sample standard deviation |
| --- | ---: | ---: |
| Static | 0.11333 | 0.00000 |
| Independent | 0.11778 | 0.00770 |
| Global closed loop | 0.20000 | 0.00667 |
| Global initial-only | 0.10667 | 0.00000 |
| Global without stability guards | 0.20444 | 0.00770 |

These are **synthetic results**, not GPU-cluster or learning-quality measurements.
The cost function uses arbitrary accounting units; different allocations consume
different fractions of the same capacity budget. Stability guards do not improve
throughput in every profile. See [methodology and negative results](docs/benchmarks.md)
and [raw results](benchmarks/closed-loop.json). The older `benchmark.json` is retained
as an MVP artifact and is not comparable to the revised queued simulator.

See [architecture](docs/architecture.md), [controller semantics](docs/control-loop.md)
and [milestone progress](CODEX_PLAN.md#implementation-status-2026-10-24).
