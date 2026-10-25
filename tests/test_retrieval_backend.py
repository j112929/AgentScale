import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from agentscale.retrieval import (
    Constraints,
    FakeRetrievalBackend,
    Query,
    Resources,
    RetrievalPlan,
    Strategy,
)


@pytest.mark.parametrize("strategy", list(Strategy))
def test_all_fake_strategies_are_repeatable_and_cpu_capable(strategy):
    plan = RetrievalPlan(strategy, rerank_k=40 if strategy == Strategy.HYBRID_RERANK else 0)
    query = Query("q", "find a fact")
    result = FakeRetrievalBackend().retrieve(query, plan)
    assert result == FakeRetrievalBackend().retrieve(query, plan)
    assert result.synthetic and result.backend == "fake-v1"
    assert len(result.hits) == plan.top_k
    assert len({h.document_id for h in result.hits}) == plan.top_k
    assert result.service_s > 0 and 0 <= result.quality <= 1
    assert plan.resources.gpus == 0
    json.dumps(asdict(result), allow_nan=False)


def test_cross_process_hash_seed_does_not_change_fake_result():
    source = (
        "import json; from dataclasses import asdict; from agentscale.retrieval import *; "
        "print(json.dumps(asdict(FakeRetrievalBackend(3).retrieve("
        "Query('q', '中文检索'), RetrievalPlan(Strategy.HYBRID)))))"
    )
    results = [
        subprocess.check_output(
            [sys.executable, "-c", source], env={**os.environ, "PYTHONHASHSEED": str(seed)}
        )
        for seed in (1, 2)
    ]
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Query("", "text"),
        lambda: Query("q", ""),
        lambda: Query("q", "t", difficulty=float("nan")),
        lambda: Query("q", "t", lexicality=1.1),
        lambda: Resources(-1),
        lambda: Resources(True),
        lambda: Constraints(max_latency_s=float("inf")),
        lambda: Constraints(max_cost_units=0),
        lambda: Constraints(min_quality=-0.1),
        lambda: RetrievalPlan("bm25"),
        lambda: RetrievalPlan(Strategy.BM25, top_k=0),
        lambda: RetrievalPlan(Strategy.BM25, use_gpu=True),
        lambda: RetrievalPlan(Strategy.HYBRID_RERANK, top_k=20, rerank_k=10),
    ],
)
def test_invalid_contracts_fail_before_execution(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_resource_arithmetic_and_gpu_rerank_contract():
    a, b = Resources(2, 4, 1), Resources(4, 8, 2)
    assert a.fits(b) and not b.fits(a)
    assert a + a == b and b - a == a
    with pytest.raises(ValueError):
        _ = a - b
    assert RetrievalPlan(Strategy.HYBRID_RERANK, rerank_k=40, use_gpu=True).resources == a
