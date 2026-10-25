"""Optional Ray Actor pool for trusted synthetic sandbox operations (M5.1).

This is a single-driver adapter, not a security boundary or a code executor.
Ray CPU/memory reservations are logical scheduling resources, not OS limits.
"""

import hashlib
import importlib
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Self

from .adapters import SafePointUnavailable
from .models import Allocation, Capacity, Workload, validate


@dataclass(frozen=True)
class SandboxTask:
    task_id: str
    payload: str = ""
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id or not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("task requires an id and a finite, nonnegative duration")


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    digest: str | None = None
    error: str | None = None


class SyntheticSandboxWorker:
    def ready(self) -> bool:
        return True

    def execute(self, task: SandboxTask) -> str:
        time.sleep(task.duration_s)
        return hashlib.sha256(task.payload.encode()).hexdigest()


@dataclass
class _Running:
    task: SandboxTask
    actor_id: int
    started_s: float


class RaySandboxAdapter:
    """Transactional scale-out, idle-only scale-in, bounded queue and task deadlines.

    The caller owns ray.init()/shutdown(). GPU reassignment is unsupported.
    Rollout concurrency caps dispatch of this sandbox pool only.
    """

    def __init__(
        self,
        allocation: Allocation,
        capacity: Capacity,
        workload: Workload,
        *,
        readiness_timeout_s: float = 30,
        task_timeout_s: float = 30,
        max_queue: int = 1024,
        ray_module: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not math.isfinite(readiness_timeout_s)
            or readiness_timeout_s <= 0
            or not math.isfinite(task_timeout_s)
            or task_timeout_s <= 0
            or type(max_queue) is not int
            or max_queue < 1
        ):
            raise ValueError("timeouts and max_queue must be finite and positive")
        errors = validate(allocation, capacity, workload)
        if errors:
            raise ValueError("; ".join(errors))
        try:
            self._ray = ray_module if ray_module is not None else importlib.import_module("ray")
        except ImportError as exc:
            raise ImportError("Ray adapter requires: pip install 'agentscale[ray]'") from exc
        if not self._ray.is_initialized():
            raise RuntimeError("call ray.init() before creating the adapter")
        self.capacity, self.workload = capacity, workload
        self._allocation = allocation
        self._clock = clock
        self.readiness_timeout_s = readiness_timeout_s
        self.task_timeout_s = task_timeout_s
        self.max_queue = max_queue
        self._actors: dict[int, Any] = {}
        self._next_actor = 0
        self._running: dict[Any, _Running] = {}
        self._queue: deque[SandboxTask] = deque()
        self._results: dict[str, TaskResult] = {}
        self._ids: set[str] = set()
        self._closed = False
        self.completed = self.failed = self.timed_out = 0
        self._remote_worker = self._ray.remote(SyntheticSandboxWorker)
        self._check_cluster(allocation.sandbox_workers)
        self._grow(allocation.sandbox_workers)

    def _check_cluster(self, workers: int) -> None:
        resources = self._ray.cluster_resources()
        if workers * self.workload.sandbox_cpu > resources.get("CPU", 0):
            raise ValueError("sandbox target exceeds Ray cluster CPU capacity")
        if workers * self.workload.sandbox_memory_gb * 1024**3 > resources.get("memory", 0):
            raise ValueError("sandbox target exceeds Ray cluster logical memory capacity")

    def _grow(self, count: int) -> None:
        staged: dict[int, Any] = {}
        try:
            for _ in range(count):
                actor = self._remote_worker.options(
                    num_cpus=self.workload.sandbox_cpu,
                    memory=self.workload.sandbox_memory_gb * 1024**3,
                    max_restarts=0,
                    max_task_retries=0,
                ).remote()
                staged[self._next_actor] = actor
                self._next_actor += 1
            if staged:
                self._ray.get(
                    [a.ready.remote() for a in staged.values()], timeout=self.readiness_timeout_s
                )
        except Exception:
            # Preserve handles if cleanup fails so observe() exposes partial state.
            for key, actor in staged.items():
                try:
                    self._ray.kill(actor, no_restart=True)
                except Exception:
                    self._actors[key] = actor
            raise
        self._actors.update(staged)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("adapter is closed")

    def observe(self) -> Allocation:
        self._ensure_open()
        return replace(self._allocation, sandbox_workers=len(self._actors))

    def apply(self, allocation: Allocation) -> None:
        self._ensure_open()
        errors = validate(allocation, self.capacity, self.workload)
        if errors:
            raise ValueError("; ".join(errors))
        if (allocation.inference_gpus, allocation.training_gpus) != (
            self._allocation.inference_gpus,
            self._allocation.training_gpus,
        ):
            raise ValueError("GPU reassignment unsupported by RaySandboxAdapter")
        self._check_cluster(allocation.sandbox_workers)
        delta = allocation.sandbox_workers - len(self._actors)
        if delta > 0:
            self._grow(delta)
        elif delta < 0:
            busy = {r.actor_id for r in self._running.values()}
            idle = [key for key in self._actors if key not in busy]
            if len(idle) < -delta:
                raise SafePointUnavailable("scale-in waits for idle sandbox workers; call poll()")
            for key in idle[:-delta]:
                self._ray.kill(self._actors[key], no_restart=True)
                del self._actors[key]
        self._allocation = allocation
        self._dispatch()

    def submit(self, task: SandboxTask) -> str:
        self._ensure_open()
        if task.task_id in self._ids:
            raise ValueError("duplicate task id; consume its result before reuse")
        if len(self._queue) >= self.max_queue:
            raise OverflowError("sandbox queue is full")
        self._ids.add(task.task_id)
        self._queue.append(task)
        self._dispatch()
        return task.task_id

    def _dispatch(self) -> None:
        busy = {r.actor_id for r in self._running.values()}
        for key, actor in self._actors.items():
            if not self._queue or len(self._running) >= self._allocation.rollout_concurrency:
                break
            if key not in busy:
                task = self._queue.popleft()
                try:
                    ref = actor.execute.remote(task)
                except Exception as exc:
                    self._results[task.task_id] = TaskResult(task.task_id, error=str(exc))
                    self.failed += 1
                    continue
                self._running[ref] = _Running(task, key, self._clock())

    def _replace_failed_actor(self, key: int) -> None:
        self._ray.kill(self._actors[key], no_restart=True)
        del self._actors[key]
        self._grow(1)

    def poll(self) -> dict[str, int]:
        self._ensure_open()
        refs = list(self._running)
        ready, _ = self._ray.wait(refs, num_returns=len(refs), timeout=0) if refs else ([], [])
        now = self._clock()
        for ref in refs:
            running = self._running[ref]
            error = None
            digest = None
            if ref in ready:
                try:
                    digest = self._ray.get(ref, timeout=0)
                except Exception as exc:
                    error = f"actor task failed: {exc}"
            elif now - running.started_s >= self.task_timeout_s:
                error = "sandbox task deadline exceeded"
                self.timed_out += 1
            else:
                continue
            del self._running[ref]
            self._results[running.task.task_id] = TaskResult(running.task.task_id, digest, error)
            if error:
                self.failed += 1
                self._replace_failed_actor(running.actor_id)
            else:
                self.completed += 1
        self._dispatch()
        return self.telemetry()

    def telemetry(self) -> dict[str, int]:
        return {
            "workers": len(self._actors),
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": self.completed,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "reserved_cpu_cores": len(self._actors) * self.workload.sandbox_cpu,
            "reserved_memory_bytes": len(self._actors) * self.workload.sandbox_memory_gb * 1024**3,
        }

    def pop_result(self, task_id: str) -> TaskResult | None:
        result = self._results.pop(task_id, None)
        if result is not None:
            self._ids.remove(task_id)
        return result

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        if (self._running or self._queue) and not force:
            raise SafePointUnavailable("pool has pending tasks; drain or close(force=True)")
        for key in list(self._actors):
            self._ray.kill(self._actors[key], no_restart=True)
            del self._actors[key]
        for task in list(self._queue) + [r.task for r in self._running.values()]:
            self._results[task.task_id] = TaskResult(task.task_id, error="pool closed")
            self.failed += 1
        self._queue.clear()
        self._running.clear()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close(force=True)
