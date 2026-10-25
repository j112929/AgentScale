"""Retrieval contracts. Seconds, whole cores/GPUs, GiB and arbitrary cost units."""

import math
from dataclasses import dataclass
from enum import StrEnum


def nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def probability(value: float, name: str) -> None:
    nonnegative(value, name)
    if value > 1:
        raise ValueError(f"{name} must be at most one")


class Strategy(StrEnum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid-rerank"


@dataclass(frozen=True)
class Resources:
    cpu_cores: int = 0
    memory_gb: int = 0
    gpus: int = 0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in self.values()):
            raise ValueError("resource counts must be nonnegative integers")

    def values(self) -> tuple[int, int, int]:
        return self.cpu_cores, self.memory_gb, self.gpus

    def fits(self, capacity: "Resources") -> bool:
        return all(a <= b for a, b in zip(self.values(), capacity.values()))

    def __add__(self, other: "Resources") -> "Resources":
        return Resources(*(a + b for a, b in zip(self.values(), other.values())))

    def __sub__(self, other: "Resources") -> "Resources":
        return Resources(*(a - b for a, b in zip(self.values(), other.values())))


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    difficulty: float = 0.5
    lexicality: float = 0.5

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.text.strip():
            raise ValueError("query ID and text must be nonempty")
        probability(self.difficulty, "difficulty")
        probability(self.lexicality, "lexicality")


@dataclass(frozen=True)
class RetrievalPlan:
    strategy: Strategy
    top_k: int = 10
    rerank_k: int = 0
    use_gpu: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, Strategy):
            raise TypeError("strategy must be a Strategy")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be an integer in [1, 100]")
        if type(self.rerank_k) is not int or type(self.use_gpu) is not bool:
            raise ValueError("invalid reranking configuration")
        if self.strategy == Strategy.HYBRID_RERANK:
            if not self.top_k <= self.rerank_k <= 200:
                raise ValueError("rerank_k must cover top_k and be at most 200")
        elif self.rerank_k or self.use_gpu:
            raise ValueError("reranking and GPU use require hybrid-rerank")

    @property
    def resources(self) -> Resources:
        # Peak reservation across a fake job, not a real engine resource profile.
        return {
            Strategy.BM25: Resources(1, 1),
            Strategy.VECTOR: Resources(1, 2),
            Strategy.HYBRID: Resources(2, 3),
            Strategy.HYBRID_RERANK: Resources(2 if self.use_gpu else 4, 4, int(self.use_gpu)),
        }[self.strategy]


@dataclass(frozen=True)
class Constraints:
    min_quality: float = 0.55
    min_success_probability: float = 0.4
    max_latency_s: float = 5.0
    max_cost_units: float = 3.0

    def __post_init__(self) -> None:
        probability(self.min_quality, "minimum quality")
        probability(self.min_success_probability, "minimum success probability")
        for name, value in (("latency", self.max_latency_s), ("cost", self.max_cost_units)):
            nonnegative(value, name)
            if value == 0:
                raise ValueError(f"maximum {name} must be positive")


@dataclass(frozen=True)
class QueueState:
    timestamp_s: float
    cpu_core_s: float = 0.0
    gpu_s: float = 0.0
    downstream_delay_s: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.timestamp_s, self.cpu_core_s, self.gpu_s, self.downstream_delay_s):
            nonnegative(value, "queue metric")


@dataclass(frozen=True)
class Estimate:
    quality: float
    success_probability: float
    service_s: float
    latency_s: float
    cost_units: float
    resources: Resources
    provenance: str = "synthetic-model-v1"

    def __post_init__(self) -> None:
        probability(self.quality, "quality")
        probability(self.success_probability, "success probability")
        for value in (self.service_s, self.latency_s, self.cost_units):
            nonnegative(value, "estimate")
        if self.service_s <= 0 or self.latency_s < self.service_s:
            raise ValueError("latency must include positive service time")


@dataclass(frozen=True)
class Hit:
    document_id: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    plan: RetrievalPlan
    hits: tuple[Hit, ...]
    service_s: float
    quality: float
    task_success: bool
    backend: str = "fake-v1"
    synthetic: bool = True

    def __post_init__(self) -> None:
        nonnegative(self.service_s, "service time")
        probability(self.quality, "quality")
        if self.service_s <= 0 or type(self.task_success) is not bool:
            raise ValueError("result must have positive service time and boolean success")
