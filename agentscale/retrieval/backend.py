"""Stable fake search, never host-dependent hash(), sleeping or network access."""

import hashlib
import json
from typing import Protocol

from .models import Hit, Query, RetrievalPlan, RetrievalResult, Strategy


class RetrievalBackend(Protocol):
    def retrieve(self, query: Query, plan: RetrievalPlan) -> RetrievalResult: ...


def synthetic_quality(query: Query, plan: RetrievalPlan) -> float:
    base = {
        Strategy.BM25: 0.40 + 0.35 * query.lexicality,
        Strategy.VECTOR: 0.70 - 0.15 * query.lexicality,
        Strategy.HYBRID: 0.77,
        Strategy.HYBRID_RERANK: 0.88 + min(plan.rerank_k, 80) * 0.0005,
    }[plan.strategy]
    return min(0.99, max(0.0, base - 0.14 * query.difficulty + 0.04 * (plan.top_k / 10) ** 0.5))


def synthetic_service_s(query: Query, plan: RetrievalPlan) -> float:
    base = {
        Strategy.BM25: 0.08,
        Strategy.VECTOR: 0.16,
        Strategy.HYBRID: 0.24,
        Strategy.HYBRID_RERANK: 0.40,
    }[plan.strategy]
    rerank = plan.rerank_k * (0.004 if plan.use_gpu else 0.012)
    return (base + rerank) * (1 + 0.5 * query.difficulty) * (1 + plan.top_k / 50)


class FakeRetrievalBackend:
    """All strategies emulate search. Returned IDs are synthetic, with no corpus/qrels."""

    def __init__(self, seed: int = 42) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed

    def _unit(self, query: Query, plan: RetrievalPlan, label: str) -> float:
        payload = json.dumps(
            [
                self.seed,
                query.query_id,
                query.text,
                query.difficulty,
                query.lexicality,
                plan.strategy,
                plan.top_k,
                plan.rerank_k,
                plan.use_gpu,
                label,
            ],
            ensure_ascii=False,
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64

    def retrieve(self, query: Query, plan: RetrievalPlan) -> RetrievalResult:
        quality = min(
            1.0,
            max(
                0.0,
                synthetic_quality(query, plan) + 0.04 * (self._unit(query, plan, "quality") - 0.5),
            ),
        )
        service = synthetic_service_s(query, plan) * (0.8 + 0.4 * self._unit(query, plan, "time"))
        hits = tuple(Hit(f"synthetic-{query.query_id}-{i}", 1 / (i + 1)) for i in range(plan.top_k))
        success = self._unit(query, plan, "success") < quality * (1 - 0.15 * query.difficulty)
        return RetrievalResult(query.query_id, plan, hits, service, quality, success)
