# AgentScale (experimental)

**Offline-first global resource scheduling experiments for Agentic RL.**

AgentScale currently provides a deterministic, four-workload **simulation** of rollout,
inference, sandbox execution, and training; resource-capacity validation; three scheduling
policies; and a benchmark CLI. **It does not yet connect to real Ray, verl, vLLM, GPU,
or production sandboxes.** The fake sandbox executes no code.

## Quick start

```bash
python -m pip install -e '.[dev]'
agentscale simulate --scenario sandbox --policy global
agentscale plan --scenario sandbox --dry-run
agentscale benchmark --duration 120 --output benchmark.json
pytest -q
```

## Scheduling model

- A trajectory alternates inference and sandbox service for a fixed number of steps.
- Sandbox durations are seeded lognormal draws; stages have bounded parallel workers.
- Completed trajectories enter a buffer; a trainer consumes batches and advances policy versions.
- Stale samples are dropped before training. GPU/CPU/memory overcommit is rejected.
- `StaticPolicy` keeps the initial allocation.
- `IndependentAutoscalePolicy` uses separate queue thresholds.
- `BottleneckAwarePolicy` searches feasible GPU, sandbox and concurrency allocations using
  a throughput upper-bound and explicit cost/switching penalties.

## Important limitations

The optimizer uses an approximate capacity model, not a trained predictor. The simulator
models each GPU as a single inference service slot (not vLLM batching or KV cache), does
not simulate real GPU migration, and assumes homogeneous workers. Its arrival admission
model drops arrivals when concurrency is full. Benchmark results are **synthetic only**;
no claims about real RL training performance or cost savings are supported yet.

See [CODEX_PLAN.md](CODEX_PLAN.md) for milestones and [docs/architecture.md](docs/architecture.md).
