"""Deterministic event simulator: trajectories alternate inference and sandbox.

Each worker handles one operation at a time. Training consumes fresh buffered
samples in batches, at training-GPU-proportional duration. No real code is run.
"""
import heapq
import random
from dataclasses import dataclass, field
from .models import Allocation, Capacity, Workload, validate


@dataclass
class Result:
    completed: int = 0
    trained: int = 0
    failed: int = 0
    stale_dropped: int = 0
    inference_wait_s: float = 0.0
    sandbox_wait_s: float = 0.0
    trainer_busy_s: float = 0.0
    events: list[dict] = field(default_factory=list)

    def summary(self, duration: float) -> dict:
        return {
            "completed": self.completed,
            "trained": self.trained,
            "failed": self.failed,
            "stale_dropped": self.stale_dropped,
            "valid_trajectories_per_s": round(self.completed / duration, 5),
            "trained_samples_per_s": round(self.trained / duration, 5),
            "inference_wait_s": round(self.inference_wait_s, 4),
            "sandbox_wait_s": round(self.sandbox_wait_s, 4),
            "trainer_busy_s": round(self.trainer_busy_s, 4),
        }


def simulate(capacity: Capacity, allocation: Allocation, workload: Workload,
             duration_s: float = 120, seed: int = 42, trace: bool = False) -> Result:
    errors = validate(allocation, capacity, workload)
    if errors:
        raise ValueError('; '.join(errors))
    if duration_s <= 0 or workload.arrival_per_s <= 0:
        raise ValueError("duration and arrival rate must be positive")
    rng = random.Random(seed)
    result = Result()
    events: list[tuple[float, int, str, int]] = []
    seq = 0
    def push(t: float, kind: str, tid: int) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, tid))

    # Stage availability, queues, and trajectory state.
    infer_free = [0.0] * allocation.inference_gpus
    sandbox_free = [0.0] * allocation.sandbox_workers
    steps: dict[int, int] = {}
    created: dict[int, float] = {}
    versions: dict[int, int] = {}
    buffer: list[int] = []
    active = 0
    policy_version = 0
    training_free = 0.0
    tid_counter = 0
    push(0.0, "arrival", -1)

    while events:
        now, _, kind, tid = heapq.heappop(events)
        if now > duration_s:
            break
        if kind == "arrival":
            push(now + rng.expovariate(workload.arrival_per_s), "arrival", -1)
            if active >= allocation.rollout_concurrency:
                continue  # admission drop, not an invisible unbounded queue
            tid_counter += 1
            tid = tid_counter
            active += 1
            steps[tid] = 0
            created[tid] = now
            versions[tid] = policy_version
            push(now, "infer", tid)
        elif kind == "infer":
            idx = min(range(len(infer_free)), key=lambda i: infer_free[i])
            start = max(now, infer_free[idx])
            result.inference_wait_s += start - now
            service = workload.inference_s / workload.inference_gpu_efficiency
            infer_free[idx] = start + service
            push(start + service, "sandbox", tid)
        elif kind == "sandbox":
            idx = min(range(len(sandbox_free)), key=lambda i: sandbox_free[i])
            start = max(now, sandbox_free[idx])
            result.sandbox_wait_s += start - now
            service = workload.sandbox_s * rng.lognormvariate(-0.125, 0.5)
            sandbox_free[idx] = start + service
            push(start + service, "step_done", tid)
        elif kind == "step_done":
            if rng.random() < workload.sandbox_failure_rate:
                result.failed += 1
                active -= 1
                continue
            steps[tid] += 1
            if steps[tid] < workload.steps:
                push(now, "infer", tid)
            else:
                result.completed += 1
                active -= 1
                buffer.append(versions[tid])
                if training_free <= now:
                    push(now, "train_check", -1)
        elif kind == "train_check":
            if now < training_free:
                continue
            fresh = [v for v in buffer if policy_version - v <= workload.max_staleness]
            result.stale_dropped += len(buffer) - len(fresh)
            buffer = fresh
            if len(buffer) >= workload.batch_size:
                del buffer[:workload.batch_size]
                service = workload.training_s * workload.batch_size / allocation.training_gpus
                training_free = now + service
                result.trainer_busy_s += min(service, duration_s - now)
                push(training_free, "train_done", workload.batch_size)
        elif kind == "train_done":
            result.trained += tid
            policy_version += 1
            push(now, "train_check", -1)
        if trace:
            result.events.append({"time_s": round(now, 6), "event": kind, "id": tid})
    return result
