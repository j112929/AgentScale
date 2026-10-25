"""Single-driver reservation accounting around the existing RL policy protocol."""

from dataclasses import replace

from ..adapters import SafePointUnavailable
from ..models import (
    Allocation,
    Capacity,
    Plan,
    ResourceAdapter,
    SchedulingPolicy,
    Snapshot,
    Workload,
    validate,
)
from .models import Resources


def residual_capacity(state: Snapshot) -> Resources:
    errors = validate(state.allocation, state.capacity, state.workload)
    if errors:
        raise ValueError("; ".join(errors))
    a, w, cap = state.allocation, state.workload, state.capacity
    return Resources(
        cap.cpu_cores - a.sandbox_workers * w.sandbox_cpu,
        cap.memory_gb - a.sandbox_workers * w.sandbox_memory_gb,
        cap.gpus - a.inference_gpus - a.training_gpus,
    )


class ReservationLedger:
    """In-memory logical reservations. RL and retrieval must share one driver/ledger.

    Counts only resources present in the existing RL model. No real GPU actuator,
    distributed lock, process-memory measurement or inference CPU accounting.
    """

    def __init__(self) -> None:
        self._reservations: dict[str, Resources] = {}

    @property
    def total(self) -> Resources:
        total = Resources()
        for demand in self._reservations.values():
            total += demand
        return total

    def reserve(self, token: str, demand: Resources, state: Snapshot) -> None:
        if not token:
            raise ValueError("reservation token must be nonempty")
        old = self._reservations.get(token)
        if old is not None and old != demand:
            raise ValueError("reservation token already has a different demand")
        target = self.total if old is not None else self.total + demand
        if not target.fits(residual_capacity(state)):
            raise ValueError("retrieval reservation would overcommit shared capacity")
        self._reservations[token] = demand

    def release(self, token: str) -> None:
        self._reservations.pop(token, None)

    def contains(self, token: str) -> bool:
        return token in self._reservations

    def guard_transition(self, proposed: Snapshot) -> None:
        if not self.total.fits(residual_capacity(proposed)):
            raise SafePointUnavailable("waiting for retrieval reservations to drain")


class ReservationAwareAdapter:
    """Fence every write, including rollback, before delegating to an existing adapter."""

    def __init__(
        self,
        adapter: ResourceAdapter,
        ledger: ReservationLedger,
        capacity: Capacity,
        workload: Workload,
    ) -> None:
        self.adapter, self.ledger = adapter, ledger
        self.capacity, self.workload = capacity, workload

    def observe(self) -> Allocation:
        return self.adapter.observe()

    def apply(self, allocation: Allocation) -> None:
        self.ledger.guard_transition(Snapshot(0, allocation, self.capacity, self.workload))
        self.adapter.apply(allocation)


class ReservationAwarePolicy:
    """Opt-in wrapper; all RL choices still belong to the wrapped SchedulingPolicy."""

    def __init__(self, policy: SchedulingPolicy, ledger: ReservationLedger) -> None:
        self.policy = policy
        self.ledger = ledger

    def propose(self, state: Snapshot) -> Plan:
        used = self.ledger.total
        cap = state.capacity
        limited = replace(
            state,
            capacity=Capacity(
                cap.gpus - used.gpus, cap.cpu_cores - used.cpu_cores, cap.memory_gb - used.memory_gb
            ),
        )
        errors = validate(limited.allocation, limited.capacity, limited.workload)
        if errors:
            raise ValueError(
                "active reservations conflict with RL allocation: " + "; ".join(errors)
            )
        plan = self.policy.propose(limited)
        errors = validate(plan.allocation, limited.capacity, limited.workload)
        if errors:
            raise ValueError("RL policy ignored retrieval reservations: " + "; ".join(errors))
        return plan
