from itertools import pairwise

import pytest

from agentscale.cli import run
from agentscale.controller import ControllerConfig
from agentscale.models import Allocation, Capacity, Workload, validate
from agentscale.policies import BottleneckAwarePolicy
from agentscale.scenarios import SCENARIOS
from agentscale.simulator import simulate


def test_dynamic_reproducible_and_feasible():
    a = run("dynamic", "global", 600, 42, trace=True)
    assert a == run("dynamic", "global", 600, 42, trace=True)
    allocations = a["allocations"]
    assert len(allocations) >= 4
    assert allocations[1]["allocation"]["inference_gpus"] > 4
    assert any(r["allocation"]["sandbox_workers"] > 8 for r in allocations)
    assert allocations[-1]["allocation"]["training_gpus"] > 4
    times = [r["time_s"] for r in allocations[1:]]
    assert all(b - a >= 15 for a, b in pairwise(times))
    for row in allocations:
        assert not validate(Allocation(**row["allocation"]), Capacity(), SCENARIOS["dynamic"])
    m = a["metrics"]
    assert m["trained"] + m["stale_dropped"] <= m["completed"]
    assert m["completed"] + m["failed"] <= m["admitted"]
    assert m["switching_s"] > 0 and m["drain_s"] > 0
    assert all(
        min(t["inference_queue"], t["sandbox_queue"], t["training_buffer"]) >= 0
        for t in a["telemetry"]
    )
    assert not any(d["status"] == "rollback_failed" for d in a["decisions"])


def test_constructed_dynamic_case_improves_one_baseline_on_same_arrivals():
    static = run("dynamic", "static", 600, 42, trace=True)
    global_result = run("dynamic", "global", 600, 42, trace=True)
    assert global_result["metrics"]["trained"] > static["metrics"]["trained"]

    def arrivals(row):
        return [e for e in row["events"] if e["event"] == "arrival"]

    assert arrivals(static) == arrivals(global_result)


def test_no_training_from_failed_sandboxes():
    r = simulate(Capacity(), Allocation(), Workload(sandbox_failure_rate=1), 120)
    assert r.failed > 0
    assert r.completed == r.trained == 0
    assert r.failed <= r.admitted


def test_single_shot_has_no_runtime_changes():
    r = run("dynamic", "global", 120, 42, closed_loop=False)
    assert r["metrics"]["allocation_changes"] == 0
    assert r["decisions"] == []
    assert len(r["workload_shifts"]) == 2


def test_telemetry_is_periodic():
    r = simulate(
        Capacity(), Allocation(), Workload(), 10, controller_config=ControllerConfig(interval_s=2)
    )
    assert [s.time_s for s in r.telemetry] == [0, 2, 4, 6, 8, 10]
    assert all(s.timestamp_s == s.time_s for s in r.telemetry)


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        simulate(Capacity(), Allocation(), Workload(), duration)


@pytest.mark.parametrize(
    "changes",
    [
        ((0, Workload()),),
        ((10, Workload()), (5, Workload())),
        ((10, Workload(arrival_per_s=2)),),
        ((10, Workload(sandbox_cpu=4)),),
    ],
)
def test_invalid_workload_shifts(changes):
    with pytest.raises(ValueError):
        simulate(Capacity(), Allocation(), Workload(), 20, workload_shifts=changes)


@pytest.mark.parametrize(
    "workload",
    [
        Workload(inference_s=0),
        Workload(sandbox_s=float("nan")),
        Workload(sandbox_cpu=0),
        Workload(batch_size=0),
        Workload(training_s=-1),
    ],
)
def test_invalid_demands_fail_before_simulating(workload):
    with pytest.raises(ValueError):
        simulate(Capacity(), Allocation(), workload)


def test_fixed_gpu_planning_for_ray():
    s = simulate(Capacity(), Allocation(), SCENARIOS["sandbox"], 20).telemetry[0]
    p = BottleneckAwarePolicy(allow_gpu_changes=False).propose(s)
    assert p.allocation.inference_gpus == s.allocation.inference_gpus
    assert p.allocation.training_gpus == s.allocation.training_gpus
    assert p.allocation.sandbox_workers > s.allocation.sandbox_workers
