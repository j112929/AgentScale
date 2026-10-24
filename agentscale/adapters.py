"""Offline adapter; production adapters are intentionally not implemented."""
from dataclasses import dataclass
from .models import Allocation, Capacity, Workload, validate


@dataclass
class FakeAdapter:
    allocation: Allocation
    capacity: Capacity
    workload: Workload
    writes: int = 0

    def observe(self) -> Allocation:
        return self.allocation

    def apply(self, allocation: Allocation) -> None:
        errors = validate(allocation, self.capacity, self.workload)
        if errors:
            raise ValueError('; '.join(errors))
        if allocation != self.allocation:
            self.allocation = allocation
            self.writes += 1


def dry_run(adapter: FakeAdapter, allocation: Allocation) -> list[str]:
    return validate(allocation, adapter.capacity, adapter.workload)
