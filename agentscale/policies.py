"""Resource scheduling policies with deterministic, explainable proposals."""

from dataclasses import replace
from itertools import product

from .models import Allocation, Plan, Snapshot, estimate_throughput, validate


def cost(a: Allocation) -> float:
    return 10.0 * (a.inference_gpus + a.training_gpus) + a.sandbox_workers * 0.25


class StaticPolicy:
    def propose(self, state: Snapshot) -> Plan:
        a = state.allocation
        return Plan(a, "static allocation", estimate_throughput(a, state.workload), cost(a))


class IndependentAutoscalePolicy:
    """Reactive per-stage scaling without a shared throughput model."""

    def propose(self, state: Snapshot) -> Plan:
        a = state.allocation
        w = state.workload
        sandbox = a.sandbox_workers + (2 if state.sandbox_queue > 2 else 0)
        sandbox = min(
            sandbox,
            state.capacity.cpu_cores // w.sandbox_cpu,
            state.capacity.memory_gb // w.sandbox_memory_gb,
        )
        inference = a.inference_gpus + (1 if state.inference_queue > 2 else 0)
        inference = min(inference, state.capacity.gpus - a.training_gpus)
        candidate = replace(a, sandbox_workers=sandbox, inference_gpus=inference)
        if validate(candidate, state.capacity, w):
            candidate = a
        return Plan(
            candidate,
            "independent queue-threshold scaling",
            estimate_throughput(candidate, w),
            cost(candidate),
        )


class BottleneckAwarePolicy:
    """Enumerate feasible global allocations; penalize churn and idle resources."""

    def __init__(
        self,
        cost_weight: float = 0.0002,
        switch_weight: float = 0.002,
        improvement_margin: float = 0.0001,
        allow_gpu_changes: bool = True,
    ) -> None:
        self.cost_weight = cost_weight
        self.switch_weight = switch_weight
        self.improvement_margin = improvement_margin
        self.allow_gpu_changes = allow_gpu_changes

    def propose(self, state: Snapshot) -> Plan:
        old, cap, w = state.allocation, state.capacity, state.workload

        def score(a: Allocation) -> float:
            switches = (
                abs(a.inference_gpus - old.inference_gpus)
                + abs(a.sandbox_workers - old.sandbox_workers) / 4
                + abs(a.rollout_concurrency - old.rollout_concurrency) / 16
            )
            return (
                estimate_throughput(a, w)
                - self.cost_weight * cost(a)
                - self.switch_weight * switches
            )

        best = old
        best_score = score(old)
        rejected = 0
        max_workers = min(cap.cpu_cores // w.sandbox_cpu, cap.memory_gb // w.sandbox_memory_gb)
        for igpu, workers, concurrency in product(
            range(1, cap.gpus) if self.allow_gpu_changes else (old.inference_gpus,),
            range(1, max_workers + 1),
            (8, 16, 32, 64, 128),
        ):
            training = cap.gpus - igpu if self.allow_gpu_changes else old.training_gpus
            candidate = Allocation(igpu, training, workers, concurrency)
            if validate(candidate, cap, w):
                rejected += 1
                continue
            value = score(candidate)
            if value > best_score + self.improvement_margin:
                best, best_score = candidate, value
        old_rate = estimate_throughput(old, w)
        return Plan(
            best,
            f"global candidate search: predicted {old_rate:.4f} -> "
            f"{estimate_throughput(best, w):.4f} valid trajectories/s",
            estimate_throughput(best, w),
            cost(best),
            (f"{rejected} infeasible candidates rejected",),
        )
