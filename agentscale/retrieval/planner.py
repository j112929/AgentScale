"""Bounded adaptive retrieval planning with explicit hard constraints."""

from dataclasses import dataclass

from .estimators import Estimators
from .models import (
    Constraints,
    Estimate,
    Query,
    QueueState,
    Resources,
    RetrievalPlan,
    Strategy,
    nonnegative,
)


@dataclass(frozen=True)
class Objective:
    quality_weight: float = 0.25
    success_weight: float = 1.0
    latency_weight: float = 0.08
    cost_weight: float = 0.1

    def __post_init__(self) -> None:
        weights = (self.quality_weight, self.success_weight, self.latency_weight, self.cost_weight)
        for weight in weights:
            nonnegative(weight, "objective weight")
        if not any(weights):
            raise ValueError("at least one objective weight must be positive")

    def utility(self, estimate: Estimate) -> float:
        return (
            self.quality_weight * estimate.quality
            + self.success_weight * estimate.success_probability
            - self.latency_weight * estimate.latency_s
            - self.cost_weight * estimate.cost_units
        )


@dataclass(frozen=True)
class Candidate:
    plan: RetrievalPlan
    estimate: Estimate
    utility: float


@dataclass(frozen=True)
class OptimizationResult:
    selected: Candidate | None
    rejected: tuple[str, ...]
    considered: int


def violations(
    estimate: Estimate, constraints: Constraints, capacity: Resources
) -> tuple[str, ...]:
    reasons = []
    for name, requested, available in zip(
        ("cpu", "memory", "gpu"), estimate.resources.values(), capacity.values()
    ):
        if requested > available:
            reasons.append(f"{name}: {requested} > {available}")
    for name, actual, limit, exceeds in (
        ("quality", estimate.quality, constraints.min_quality, False),
        ("success", estimate.success_probability, constraints.min_success_probability, False),
        ("latency_s", estimate.latency_s, constraints.max_latency_s, True),
        ("cost_units", estimate.cost_units, constraints.max_cost_units, True),
    ):
        if (actual > limit + 1e-12) if exceeds else (actual < limit - 1e-12):
            reasons.append(f"{name}: {actual:.6g} {'>' if exceeds else '<'} {limit:.6g}")
    return tuple(reasons)


class ConstrainedOptimizer:
    def __init__(
        self, estimators: Estimators | None = None, objective: Objective | None = None
    ) -> None:
        self.estimators = estimators or Estimators()
        self.objective = objective or Objective()

    def select(
        self,
        query: Query,
        plans: tuple[RetrievalPlan, ...],
        constraints: Constraints,
        capacity: Resources,
        queue: QueueState,
        elapsed_s: float = 0.0,
    ) -> OptimizationResult:
        best: Candidate | None = None
        rejected = []
        for plan in plans:
            estimate = self.estimators.estimate(query, plan, queue, capacity, elapsed_s)
            errors = violations(estimate, constraints, capacity)
            if errors:
                rejected.append(
                    f"{plan.strategy}/k={plan.top_k}/gpu={plan.use_gpu}: " + "; ".join(errors)
                )
                continue
            candidate = Candidate(plan, estimate, self.objective.utility(estimate))
            # Stable first-candidate tie breaking independent of hash/random state.
            if best is None or candidate.utility > best.utility + 1e-12:
                best = candidate
        return OptimizationResult(best, tuple(rejected), len(plans))


class AdaptiveRetrievalPlanner:
    def __init__(
        self, optimizer: ConstrainedOptimizer | None = None, *, allow_gpu: bool = False
    ) -> None:
        self.optimizer = optimizer or ConstrainedOptimizer()
        self.allow_gpu = allow_gpu

    def candidates(self) -> tuple[RetrievalPlan, ...]:
        plans = [
            RetrievalPlan(
                strategy, top_k=k, rerank_k=max(40, k) if strategy == Strategy.HYBRID_RERANK else 0
            )
            for strategy in Strategy
            for k in (5, 10, 20)
        ]
        if self.allow_gpu:
            plans += [RetrievalPlan(Strategy.HYBRID_RERANK, k, 40, True) for k in (5, 10, 20)]
        return tuple(plans)

    def propose(
        self,
        query: Query,
        constraints: Constraints,
        capacity: Resources,
        queue: QueueState,
        elapsed_s: float = 0.0,
    ) -> OptimizationResult:
        return self.optimizer.select(
            query, self.candidates(), constraints, capacity, queue, elapsed_s
        )
