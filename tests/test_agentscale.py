import pytest

from agentscale.adapters import FakeAdapter, dry_run
from agentscale.cli import run
from agentscale.models import Allocation, Capacity, Snapshot, Workload, validate
from agentscale.policies import BottleneckAwarePolicy, IndependentAutoscalePolicy, StaticPolicy
from agentscale.simulator import simulate


def test_repeatable_simulation():
    args = (Capacity(), Allocation(), Workload())
    a = simulate(*args, seed=12, trace=True)
    b = simulate(*args, seed=12, trace=True)
    assert a == b
    assert a.completed > 0
    assert a.trained <= a.completed


def test_no_training_without_samples():
    result = simulate(Capacity(), Allocation(), Workload(arrival_per_s=0.001), 2, 42)
    assert result.trained == 0


def test_invalid_resources_rejected():
    assert "GPU overcommit" in validate(Allocation(7, 4, 8, 32), Capacity(), Workload())
    with pytest.raises(ValueError, match="GPU overcommit"):
        simulate(Capacity(), Allocation(7, 4, 8, 32), Workload())


def test_all_policies_feasible():
    state = Snapshot(0, Allocation(), Capacity(), Workload(), 10, 10)
    for cls in (StaticPolicy, IndependentAutoscalePolicy, BottleneckAwarePolicy):
        plan = cls().propose(state)
        assert not validate(plan.allocation, state.capacity, state.workload)
        assert plan.predicted_throughput >= 0


def test_global_sandbox_case_improves_capacity():
    w = Workload(sandbox_s=8, inference_s=0.5)
    state = Snapshot(0, Allocation(), Capacity(), w, 0, 10)
    baseline = StaticPolicy().propose(state)
    candidate = BottleneckAwarePolicy().propose(state)
    assert candidate.predicted_throughput > baseline.predicted_throughput


def test_dry_run_never_mutates():
    adapter = FakeAdapter(Allocation(), Capacity(), Workload())
    assert dry_run(adapter, Allocation(3, 5, 10, 32)) == []
    assert adapter.writes == 0
    assert adapter.observe() == Allocation()
    adapter.apply(Allocation(3, 5, 10, 32))
    adapter.apply(Allocation(3, 5, 10, 32))
    assert adapter.writes == 1


def test_cli_results():
    a = run("sandbox", "global", 60, 42)
    b = run("sandbox", "global", 60, 42)
    assert a == b
    assert a["metrics"]["completed"] > 0
