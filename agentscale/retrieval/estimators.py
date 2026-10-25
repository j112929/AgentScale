"""Independent synthetic quality, latency and cost models with bounded EWMA feedback."""

from typing import Protocol

from .backend import synthetic_quality, synthetic_service_s
from .models import (
    Estimate,
    Query,
    QueueState,
    Resources,
    RetrievalPlan,
    RetrievalResult,
    Strategy,
    nonnegative,
    probability,
)


class QualityEstimator(Protocol):
    def predict(self, query: Query, plan: RetrievalPlan) -> tuple[float, float]: ...


class LatencyEstimator(Protocol):
    def service_s(self, query: Query, plan: RetrievalPlan) -> float: ...


class CostEstimator(Protocol):
    def predict(self, plan: RetrievalPlan, service_s: float) -> float: ...


class SyntheticQualityEstimator:
    def __init__(self, alpha: float = 0.25) -> None:
        probability(alpha, "EWMA alpha")
        if alpha == 0:
            raise ValueError("EWMA alpha must be positive")
        self.alpha = alpha
        self.corrections: dict[tuple[Strategy, bool], float] = {}

    def predict(self, query: Query, plan: RetrievalPlan) -> tuple[float, float]:
        correction = self.corrections.get((plan.strategy, plan.use_gpu), 0.0)
        quality = min(1.0, max(0.0, synthetic_quality(query, plan) + correction))
        return quality, quality * (1 - 0.15 * query.difficulty)

    def observe(self, query: Query, result: RetrievalResult) -> None:
        key = result.plan.strategy, result.plan.use_gpu
        residual = result.quality - synthetic_quality(query, result.plan)
        self.corrections[key] = (1 - self.alpha) * self.corrections.get(
            key, 0.0
        ) + self.alpha * residual


class SyntheticLatencyEstimator:
    def __init__(self, alpha: float = 0.25) -> None:
        probability(alpha, "EWMA alpha")
        if alpha == 0:
            raise ValueError("EWMA alpha must be positive")
        self.alpha = alpha
        self.multipliers: dict[tuple[Strategy, bool], float] = {}

    def service_s(self, query: Query, plan: RetrievalPlan) -> float:
        return synthetic_service_s(query, plan) * self.multipliers.get(
            (plan.strategy, plan.use_gpu), 1.0
        )

    def observe(self, query: Query, result: RetrievalResult) -> None:
        key = result.plan.strategy, result.plan.use_gpu
        ratio = result.service_s / synthetic_service_s(query, result.plan)
        # Prevent a single observation from making a model zero or unbounded.
        ratio = min(4.0, max(0.25, ratio))
        self.multipliers[key] = (1 - self.alpha) * self.multipliers.get(
            key, 1.0
        ) + self.alpha * ratio


class SyntheticCostEstimator:
    def predict(self, plan: RetrievalPlan, service_s: float) -> float:
        nonnegative(service_s, "service time")
        resources = plan.resources
        searches = 1 if plan.strategy in (Strategy.BM25, Strategy.VECTOR) else 2
        return (
            0.01 * searches
            + 0.001 * plan.top_k
            + 0.002 * plan.rerank_k
            + service_s * (0.03 * resources.cpu_cores + 0.5 * resources.gpus)
            + 0.0001 * (128 + 64 * plan.top_k)
        )


class Estimators:
    def __init__(
        self,
        quality: QualityEstimator | None = None,
        latency: LatencyEstimator | None = None,
        cost: CostEstimator | None = None,
    ) -> None:
        self.quality = quality if quality is not None else SyntheticQualityEstimator()
        self.latency = latency if latency is not None else SyntheticLatencyEstimator()
        self.cost = cost if cost is not None else SyntheticCostEstimator()
        self.observations = 0

    def estimate(
        self,
        query: Query,
        plan: RetrievalPlan,
        queue: QueueState,
        capacity: Resources,
        elapsed_s: float = 0.0,
    ) -> Estimate:
        nonnegative(elapsed_s, "elapsed time")
        quality, success = self.quality.predict(query, plan)
        service = self.latency.service_s(query, plan)
        # Fluid approximation of outstanding reserved work; not a p95 guarantee.
        cpu_wait = queue.cpu_core_s / max(1, capacity.cpu_cores)
        gpu_wait = queue.gpu_s / max(1, capacity.gpus) if plan.use_gpu else 0.0
        latency = elapsed_s + max(cpu_wait, gpu_wait) + service + queue.downstream_delay_s
        return Estimate(
            quality, success, service, latency, self.cost.predict(plan, service), plan.resources
        )

    def observe(self, query: Query, result: RetrievalResult) -> None:
        if query.query_id != result.query_id:
            raise ValueError("feedback query ID mismatch")
        if not result.synthetic or result.backend != "fake-v1":
            raise ValueError("synthetic calibration requires fake-v1 feedback")
        if isinstance(self.quality, SyntheticQualityEstimator):
            self.quality.observe(query, result)
        if isinstance(self.latency, SyntheticLatencyEstimator):
            self.latency.observe(query, result)
        self.observations += 1
