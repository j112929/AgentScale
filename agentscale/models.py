"""Typed scheduling state and resource constraints. Units are explicit."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Capacity:
    gpus: int = 8
    cpu_cores: int = 64
    memory_gb: int = 128


@dataclass(frozen=True)
class Allocation:
    inference_gpus: int = 4
    training_gpus: int = 4
    sandbox_workers: int = 8
    rollout_concurrency: int = 32


@dataclass(frozen=True)
class Workload:
    arrival_per_s: float = 1.0
    inference_s: float = 1.0
    sandbox_s: float = 2.0
    steps: int = 3
    training_s: float = 4.0
    batch_size: int = 4
    sandbox_cpu: int = 2
    sandbox_memory_gb: int = 4
    inference_gpu_efficiency: float = 1.0
    max_staleness: int = 2
    sandbox_failure_rate: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    time_s: float
    allocation: Allocation
    capacity: Capacity
    workload: Workload
    inference_queue: int = 0
    sandbox_queue: int = 0
    training_buffer: int = 0
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class Plan:
    allocation: Allocation
    reason: str
    predicted_throughput: float
    predicted_cost_per_s: float
    rejected: tuple[str, ...] = field(default_factory=tuple)


class SchedulingPolicy(Protocol):
    def propose(self, state: Snapshot) -> Plan: ...


class ResourceAdapter(Protocol):
    def observe(self) -> Allocation: ...
    def apply(self, allocation: Allocation) -> None: ...


def validate(allocation: Allocation, capacity: Capacity, workload: Workload) -> list[str]:
    errors: list[str] = []
    if min(allocation.inference_gpus, allocation.training_gpus,
           allocation.sandbox_workers, allocation.rollout_concurrency) < 1:
        errors.append("all allocation fields must be positive")
    if allocation.inference_gpus + allocation.training_gpus > capacity.gpus:
        errors.append("GPU overcommit")
    if allocation.sandbox_workers * workload.sandbox_cpu > capacity.cpu_cores:
        errors.append("CPU overcommit")
    if allocation.sandbox_workers * workload.sandbox_memory_gb > capacity.memory_gb:
        errors.append("memory overcommit")
    if not 0 <= workload.sandbox_failure_rate <= 1:
        errors.append("invalid failure rate")
    if workload.steps < 1 or workload.batch_size < 1 or workload.max_staleness < 0:
        errors.append("invalid workload configuration")
    return errors


def estimate_throughput(a: Allocation, w: Workload) -> float:
    """Upper-bound trajectories/s using normalized per-trajectory service demand."""
    inference = a.inference_gpus * w.inference_gpu_efficiency / (w.steps * w.inference_s)
    sandbox = a.sandbox_workers / (w.steps * w.sandbox_s)
    training = a.training_gpus / w.training_s
    concurrency = a.rollout_concurrency / (w.steps * (w.inference_s + w.sandbox_s))
    return min(w.arrival_per_s, inference, sandbox, training, concurrency) * (
        1 - w.sandbox_failure_rate
    )
