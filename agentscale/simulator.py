"""Deterministic queued pipeline with telemetry, workload shifts and safe draining."""

import heapq
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field

from .adapters import FakeAdapter, SafePointUnavailable
from .controller import ControllerConfig, DynamicController, switch_units
from .models import Allocation, Capacity, SchedulingPolicy, Snapshot, Workload, validate


@dataclass
class Result:
    completed: int = 0
    trained: int = 0
    failed: int = 0
    admitted: int = 0
    admission_dropped: int = 0
    stale_dropped: int = 0
    inference_wait_s: float = 0.0
    sandbox_wait_s: float = 0.0
    trainer_busy_s: float = 0.0
    switching_s: float = 0.0
    drain_s: float = 0.0
    resource_cost: float = 0.0
    events: list[dict] = field(default_factory=list)
    telemetry: list[Snapshot] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    allocations: list[dict] = field(default_factory=list)

    def summary(self, duration: float) -> dict:
        return {
            "completed": self.completed,
            "trained": self.trained,
            "failed": self.failed,
            "admitted": self.admitted,
            "admission_dropped": self.admission_dropped,
            "stale_dropped": self.stale_dropped,
            "valid_trajectories_per_s": round(self.completed / duration, 5),
            "trained_samples_per_s": round(self.trained / duration, 5),
            "inference_wait_s": round(self.inference_wait_s, 4),
            "sandbox_wait_s": round(self.sandbox_wait_s, 4),
            "trainer_busy_s": round(self.trainer_busy_s, 4),
            "switching_s": round(self.switching_s, 4),
            "drain_s": round(self.drain_s, 4),
            "resource_cost": round(self.resource_cost, 4),
            "cost_per_trained_sample": (
                round(self.resource_cost / self.trained, 4) if self.trained else None
            ),
            "allocation_changes": len(self.allocations) - 1,
            "rollbacks": sum(d["status"] == "rolled_back" for d in self.decisions),
        }


def simulate(
    capacity: Capacity,
    allocation: Allocation,
    workload: Workload,
    duration_s: float = 120,
    seed: int = 42,
    trace: bool = False,
    *,
    policy: SchedulingPolicy | None = None,
    controller_config: ControllerConfig | None = None,
    workload_shifts: tuple[tuple[float, Workload], ...] = (),
) -> Result:
    errors = validate(allocation, capacity, workload)
    if errors:
        raise ValueError("; ".join(errors))
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    previous_shift = 0.0
    for at, demand in workload_shifts:
        if not math.isfinite(at) or not previous_shift < at < duration_s:
            raise ValueError("workload shifts must be strictly increasing within duration")
        if demand.arrival_per_s != workload.arrival_per_s:
            raise ValueError("arrival rate shifts are not supported; use a constant arrival trace")
        if (demand.sandbox_cpu, demand.sandbox_memory_gb) != (
            workload.sandbox_cpu,
            workload.sandbox_memory_gb,
        ):
            raise ValueError("worker resource shapes must be constant during a run")
        errors = validate(allocation, capacity, demand)
        if errors:
            raise ValueError("; ".join(errors))
        previous_shift = at
    cfg = controller_config or ControllerConfig()
    result = Result()
    events: list[tuple[float, int, str, int]] = []
    seq = 0

    def push(t: float, kind: str, tid: int = -1) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, tid))

    infer_q: deque[tuple[int, float]] = deque()
    sandbox_q: deque[tuple[int, float]] = deque()
    buffer: deque[int] = deque()
    # tid -> (completed steps, target steps, admission policy version)
    trajectories: dict[int, tuple[int, int, int]] = {}
    infer_busy = sandbox_busy = 0
    training_busy = False
    policy_version = 0
    now = 0.0
    ready_after = 0.0

    class SimulationAdapter(FakeAdapter):
        draining = False

        def apply(self, target: Allocation) -> None:
            nonlocal ready_after
            if target == self.allocation:
                return
            # Pause dispatch and drain all stages; never evict a training batch.
            if infer_busy or sandbox_busy or training_busy:
                self.draining = True
                raise SafePointUnavailable("draining in-flight stages before resource transition")
            before = self.allocation
            super().apply(target)
            delay = switch_units(before, target) * cfg.switch_s_per_unit
            ready_after = now + delay
            result.switching_s += min(delay, duration_s - now)
            self.draining = False
            result.allocations.append({"time_s": now, "allocation": asdict(target)})
            push(ready_after, "ready")

    adapter = SimulationAdapter(allocation, capacity, workload)
    controller = DynamicController(policy, adapter, cfg) if policy is not None else None
    result.allocations.append({"time_s": 0.0, "allocation": asdict(allocation)})

    def dispatch() -> None:
        nonlocal infer_busy, sandbox_busy, training_busy, buffer
        if adapter.draining or now < ready_after:
            return
        alloc, w = adapter.observe(), adapter.workload
        while infer_q and infer_busy < alloc.inference_gpus:
            tid, queued = infer_q.popleft()
            result.inference_wait_s += now - queued
            infer_busy += 1
            push(now + w.inference_s / w.inference_gpu_efficiency, "infer_done", tid)
        while sandbox_q and sandbox_busy < alloc.sandbox_workers:
            tid, queued = sandbox_q.popleft()
            result.sandbox_wait_s += now - queued
            sandbox_busy += 1
            # Separate per-arrival streams preserve common service draws across policies.
            steps = trajectories[tid][0]
            service_rng = random.Random(f"{seed}:{tid}:{steps}:service")
            service = w.sandbox_s * service_rng.lognormvariate(-0.125, 0.5)
            push(now + service, "sandbox_done", tid)
        if not training_busy:
            fresh = deque(v for v in buffer if policy_version - v <= w.max_staleness)
            result.stale_dropped += len(buffer) - len(fresh)
            buffer = fresh
            if len(buffer) >= w.batch_size:
                for _ in range(w.batch_size):
                    buffer.popleft()
                training_busy = True
                service = w.training_s * w.batch_size / alloc.training_gpus
                result.trainer_busy_s += min(service, duration_s - now)
                push(now + service, "train_done", w.batch_size)

    # Generate arrivals independently of scheduling and service draws.
    arrivals = random.Random(seed)
    at = 0.0
    tid = 0
    while at <= duration_s:
        push(at, "arrival", tid)
        tid += 1
        at += arrivals.expovariate(workload.arrival_per_s)
    for i, (at, _) in enumerate(workload_shifts):
        push(at, "shift", i)
    push(0, "control")
    last_time = 0.0
    last_trained = 0
    last_telemetry = 0.0
    while events:
        now, _, kind, tid = heapq.heappop(events)
        elapsed = min(now, duration_s) - last_time
        alloc = adapter.observe()
        result.resource_cost += elapsed * (
            10 * (alloc.inference_gpus + alloc.training_gpus) + 0.25 * alloc.sandbox_workers
        )
        if adapter.draining:
            result.drain_s += elapsed
        last_time = min(now, duration_s)
        if now > duration_s:
            break
        if kind == "arrival":
            if len(trajectories) >= alloc.rollout_concurrency:
                result.admission_dropped += 1
            else:
                result.admitted += 1
                trajectories[tid] = (0, adapter.workload.steps, policy_version)
                infer_q.append((tid, now))
        elif kind == "infer_done":
            infer_busy -= 1
            sandbox_q.append((tid, now))
        elif kind == "sandbox_done":
            sandbox_busy -= 1
            steps, target_steps, version = trajectories[tid]
            failure_rng = random.Random(f"{seed}:{tid}:{steps}:failure")
            if failure_rng.random() < adapter.workload.sandbox_failure_rate:
                result.failed += 1
                del trajectories[tid]
            elif steps + 1 < target_steps:
                trajectories[tid] = (steps + 1, target_steps, version)
                infer_q.append((tid, now))
            else:
                result.completed += 1
                del trajectories[tid]
                buffer.append(version)
        elif kind == "train_done":
            training_busy = False
            result.trained += tid
            policy_version += 1
        elif kind == "shift":
            adapter.workload = workload_shifts[tid][1]
            adapter.draining = False
        elif kind == "control":
            window = now - last_telemetry
            state = Snapshot(
                now,
                alloc,
                capacity,
                adapter.workload,
                len(infer_q),
                len(sandbox_q),
                len(buffer),
                now,
                (result.trained - last_trained) / window if window > 0 else None,
            )
            result.telemetry.append(state)
            last_trained, last_telemetry = result.trained, now
            if controller is not None:
                decision = controller.tick(state)
                result.decisions.append(asdict(decision))
                if decision.status not in ("deferred", "rollback_deferred"):
                    adapter.draining = False
            push(now + cfg.interval_s, "control")
        dispatch()
        if trace:
            result.events.append({"time_s": round(now, 6), "event": kind, "id": tid})
    return result
