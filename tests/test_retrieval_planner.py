from dataclasses import replace

import pytest

from agentscale.retrieval import (
    Constraints,
    FakeRetrievalBackend,
    Query,
    Resources,
    RetrievalPlan,
    Strategy,
)
from agentscale.retrieval.estimators import Estimators, SyntheticLatencyEstimator
from agentscale.retrieval.models import QueueState
from agentscale.retrieval.planner import (
    AdaptiveRetrievalPlanner,
    ConstrainedOptimizer,
    Objective,
    violations,
)

QUERY = Query("q", "a semantic query", difficulty=0.6, lexicality=0.1)
CAPACITY = Resources(8, 16, 1)


def test_bounded_candidates_cpu_default_and_opt_in_gpu():
    plans = AdaptiveRetrievalPlanner().candidates()
    assert len(plans) == len(set(plans)) == 12
    assert all(not p.use_gpu for p in plans)
    assert len(AdaptiveRetrievalPlanner(allow_gpu=True).candidates()) == 15


def test_optimizer_returns_actual_best_feasible_candidate():
    planner = AdaptiveRetrievalPlanner(allow_gpu=True)
    constraints = Constraints(min_quality=0.7, max_cost_units=0.25)
    queue = QueueState(0, cpu_core_s=1)
    result = planner.propose(QUERY, constraints, CAPACITY, queue)
    candidates = []
    for plan in planner.candidates():
        estimate = planner.optimizer.estimators.estimate(QUERY, plan, queue, CAPACITY)
        if not violations(estimate, constraints, CAPACITY):
            candidates.append(planner.optimizer.objective.utility(estimate))
    assert result.selected is not None
    assert result.selected.utility == max(candidates)
    assert result.rejected
    assert result == planner.propose(QUERY, constraints, CAPACITY, queue)


@pytest.mark.parametrize(
    "constraints,capacity,reason",
    [
        (Constraints(min_quality=1), CAPACITY, "quality"),
        (Constraints(min_success_probability=1), CAPACITY, "success"),
        (Constraints(max_latency_s=0.001), CAPACITY, "latency_s"),
        (Constraints(max_cost_units=0.001), CAPACITY, "cost_units"),
        (Constraints(), Resources(0, 16), "cpu"),
        (Constraints(), Resources(8, 0), "memory"),
    ],
)
def test_infeasible_constraints_are_explained_without_fallback(constraints, capacity, reason):
    result = AdaptiveRetrievalPlanner().propose(QUERY, constraints, capacity, QueueState(0))
    assert result.selected is None
    assert result.considered == len(result.rejected) == 12
    assert all(reason in rejection for rejection in result.rejected)


def test_cpu_and_gpu_pressure_change_feasible_choices():
    optimizer = ConstrainedOptimizer()
    cpu = RetrievalPlan(Strategy.HYBRID_RERANK, rerank_k=40)
    gpu = replace(cpu, use_gpu=True)
    constraints = Constraints(min_quality=0.8)
    result = optimizer.select(QUERY, (gpu, cpu), constraints, Resources(4, 8), QueueState(0))
    assert result.selected.plan == cpu
    assert "gpu: 1 > 0" in result.rejected[0]
    result = optimizer.select(QUERY, (gpu, cpu), constraints, CAPACITY, QueueState(0, gpu_s=100))
    assert result.selected.plan == cpu


def test_queue_and_already_elapsed_time_count_toward_end_to_end_deadline():
    planner = AdaptiveRetrievalPlanner()
    constraints = Constraints(max_latency_s=0.9)
    clear = planner.propose(QUERY, constraints, CAPACITY, QueueState(0))
    busy = planner.propose(QUERY, constraints, CAPACITY, QueueState(0, cpu_core_s=8), 0.2)
    assert clear.selected is not None and busy.selected is None
    plan = RetrievalPlan(Strategy.HYBRID)
    models = Estimators()
    a = models.estimate(QUERY, plan, QueueState(0), CAPACITY)
    b = models.estimate(QUERY, plan, QueueState(0, 8, 0, 2), CAPACITY, 0.3)
    assert b.latency_s == pytest.approx(a.service_s + 3.3)
    assert b.cost_units == a.cost_units


def test_feedback_updates_service_and_quality_without_claiming_real_measurements():
    models = Estimators()
    plan = RetrievalPlan(Strategy.HYBRID)
    before = models.estimate(QUERY, plan, QueueState(0), CAPACITY)
    result = FakeRetrievalBackend().retrieve(QUERY, plan)
    models.observe(QUERY, result)
    after = models.estimate(QUERY, plan, QueueState(0), CAPACITY)
    assert models.observations == 1 and before != after
    assert after.provenance == "synthetic-model-v1"
    with pytest.raises(ValueError, match="fake-v1"):
        models.observe(QUERY, replace(result, synthetic=False))
    with pytest.raises(ValueError, match="mismatch"):
        models.observe(replace(QUERY, query_id="other"), result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency_weight": -1},
        {"cost_weight": float("nan")},
        {"quality_weight": 0, "success_weight": 0, "latency_weight": 0, "cost_weight": 0},
    ],
)
def test_invalid_objectives(kwargs):
    with pytest.raises(ValueError):
        Objective(**kwargs)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), 2])
def test_invalid_calibration(alpha):
    with pytest.raises(ValueError):
        SyntheticLatencyEstimator(alpha)
