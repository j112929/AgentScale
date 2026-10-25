# M5.1: experimental Ray Sandbox Adapter

Validated with **Ray 2.50.1**, Python 3.12.12, macOS arm64. The separate CI job runs
the same pinned Ray version on Linux; its result must be checked on GitHub.
The four local integration tests start actual Ray worker processes, not local mode.

```bash
python -m pip install -e '.[dev,ray]'
python examples/ray_sandbox.py
pytest -q -m ray tests/test_ray_integration.py
```

## Scope and ownership

`RaySandboxAdapter` owns a single-driver pool of Actors in a caller-owned Ray
cluster. The caller must initialize Ray and shut it down when appropriate; closing
the adapter kills only its own Actors. The context manager force-closes its pool
on exit, and unfinished tasks receive explicit failure results. Non-forced `close()`
defers if tasks remain. Calls must be serialized by one driver; the API is not thread-safe.

The Actor implements a trusted synthetic task: sleep for a configured duration and
compute SHA-256 for a string payload. There is no shell, arbitrary callable, code
interpreter, container, file upload, or network policy interface. **This pool is not
a security sandbox.** Real isolated coding-agent execution belongs to M5.2.

## Resources and capabilities

Each Actor explicitly reserves `sandbox_cpu` logical CPU cores and
`sandbox_memory_gb * 1024**3` bytes of logical memory. Memory values called `gb`
in the MVP model are interpreted as GiB in this integration. Both model capacity
and Ray cluster totals are checked; readiness times out if actual placement cannot
succeed (for example, resource fragmentation or contention from other jobs).

Ray resource reservations do not enforce CPU or per-process memory limits.
See the official [Ray resource documentation](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html).
Untrusted code must use a separate isolated backend with OS/container limits.

- **Sandbox scale-out:** create staged Actors, await readiness within a bounded timeout,
  then publish them. On failure, kill staged Actors and retain the existing pool.
  A cleanup failure remains visible in the observed actor count for compensation.
- **Sandbox scale-in:** remove only idle Actors. If too many Actors are busy, defer
  before making changes. Continue `poll()` until tasks finish, then retry.
- **Rollout concurrency:** limits concurrent dispatched sandbox tasks in this pool;
  it is not a real verl rollout-admission actuator. Reductions drain existing work.
- **GPU changes:** always rejected. The example has logical GPU placeholders only
  and requests no GPUs from Ray. `BottleneckAwarePolicy(allow_gpu_changes=False)`
  constrains global recommendations to the supported fixed GPU allocation.

## Tasks and telemetry

`submit(SandboxTask(...))` enqueues a typed synthetic operation. Queue depth is bounded
by `max_queue`; duplicate IDs are rejected until their result has been consumed.
`poll()` checks ready futures, reports success/failure, handles task deadlines,
replaces failed busy Actors, and dispatches queued tasks. There are no automatic task
retries: a task that failed or timed out is reported exactly once. Deadline handling
requires the caller to keep polling. Readiness waits and replacement startup are bounded.

`telemetry()` reports registered workers, logical reserved CPU/memory, queued/running
tasks and cumulative completed/failed/timed-out counts. `observe()` returns the
registered allocation; task failures are discovered by `poll()`. Idle actor deaths
are detected when a subsequent task is submitted and polled. These metrics are not
OS utilization, and no inference or training telemetry is fabricated.

Call `pop_result(id)` to consume and release a result and its ID. Completed results
remain in memory until consumed; drivers should drain them regularly. Queue waiting
time is excluded from each task's execution timeout. Killing a timed-out Actor is
process cancellation, not transactional rollback of an external task's side effects.
The bundled synthetic tasks have no external side effects.

## Validation

Offline tests inject partial scale-out failure, readiness timeout, cleanup failure,
Actor task failure, task timeout, queue overflow, duplicate IDs and unsafe scale-in.
Real Ray tests exercise controller-driven 1→2 scale-out, task digests, idle scale-in,
Actor crash/replacement, deadlines/recovery and resource rejection. The example runs
eight tasks through the same controller and emits machine-readable JSON.

This completes the M5.1 Actor-pool slice. A complete M5 RL pipeline, container isolation,
vLLM inference, verl training and GPU sharing are not implemented by this adapter.
