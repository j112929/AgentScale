# Validation record (2026-10-24)

Executed locally with Python 3.12.12 on macOS arm64:

| Check | Result |
| --- | --- |
| `python -m pytest -q -m 'not ray'` | 71 passed; 4 real-Ray tests deselected |
| `python -m pytest -q -m ray tests/test_ray_integration.py` | 4 passed with real Ray 2.50.1 processes |
| `ruff check agentscale tests examples` | Passed |
| `mypy agentscale` | Passed, 10 source files |
| `git diff --check` | Passed |
| `python examples/ray_sandbox.py` | 8 completed, 0 failed; pool scaled 1→2 |
| `agentscale benchmark --duration 600 --seeds 42 43 44` | 75 runs; raw JSON published |
| Clean base package installation with no Ray installed | Passed; dynamic CLI API works |

The original seven MVP tests pass against the revised implementation. New tests cover
controller stability, cadence, bad telemetry, dry run, partial apply compensation,
failed rollback blocking, delayed rollback safe points, performance windows, live
workload changes, deterministic arrivals, resource feasibility, conservation, queue
bounds, typed tasks and real/fake Actor lifecycle failures.

A Ray upstream FutureWarning about accelerator environment variables appeared during
CPU-only initialization; no test failed because of it. Local sandbox restrictions
required permission for Ray's process inspection and loopback worker processes.

CI now separates an offline Python 3.11/3.12/3.13 matrix from a pinned real-Ray
Python 3.12 job. Local checks do not establish that remote CI has passed; consult
the pull request's check results after push.
