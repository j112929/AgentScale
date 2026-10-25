"""Runs real Ray 2.50.1 worker processes; isolated from the offline CI job."""

import hashlib
import time
from dataclasses import replace

import pytest

from agentscale.adapters import SafePointUnavailable
from agentscale.controller import ControllerConfig, DynamicController
from agentscale.models import Allocation, Capacity, Snapshot, Workload
from agentscale.policies import IndependentAutoscalePolicy
from agentscale.ray_adapter import RaySandboxAdapter, SandboxTask

pytestmark = pytest.mark.ray
ALLOC = Allocation(1, 1, 1, 8)
CAP = Capacity(2, 2, 4)
WORK = Workload(inference_s=0.1, sandbox_s=8, sandbox_cpu=1, sandbox_memory_gb=1)


@pytest.fixture(scope="module")
def ray_cluster():
    ray = pytest.importorskip("ray")
    assert ray.__version__ == "2.50.1"
    ray.init(
        num_cpus=2, include_dashboard=False, log_to_driver=False, object_store_memory=100 * 1024**2
    )
    try:
        yield ray
    finally:
        ray.shutdown()


def drain(pool, count):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        metrics = pool.poll()
        if metrics["completed"] + metrics["failed"] >= count:
            return metrics
        time.sleep(0.02)
    pytest.fail(f"pool did not drain: {pool.telemetry()}")


def test_real_actor_scale_drain_and_controller(ray_cluster):
    with RaySandboxAdapter(ALLOC, CAP, WORK) as pool:
        for i in range(6):
            pool.submit(SandboxTask(str(i), "trusted test payload", duration_s=0.05))
        m = pool.telemetry()
        controller = DynamicController(
            IndependentAutoscalePolicy(), pool, ControllerConfig(confirmations=1)
        )
        state = Snapshot(0, pool.observe(), CAP, WORK, sandbox_queue=m["queued"], timestamp_s=0)
        assert controller.tick(state).status == "applied"
        assert pool.observe().sandbox_workers == 2
        with pytest.raises(SafePointUnavailable):
            pool.apply(ALLOC)
        assert drain(pool, 6)["completed"] == 6
        for i in range(6):
            assert (
                pool.pop_result(str(i)).digest
                == hashlib.sha256(b"trusted test payload").hexdigest()
            )
        pool.apply(ALLOC)
        pool.apply(ALLOC)
        assert pool.observe() == ALLOC
        assert len(pool._actors) == 1


def test_real_task_timeout_and_replacement(ray_cluster):
    with RaySandboxAdapter(ALLOC, CAP, WORK, task_timeout_s=0.2) as pool:
        pool.submit(SandboxTask("slow", duration_s=5))
        m = drain(pool, 1)
        assert m["timed_out"] == 1 and m["failed"] == 1
        assert "deadline" in pool.pop_result("slow").error
        pool.task_timeout_s = 10
        pool.submit(SandboxTask("recovered", "ok"))
        assert drain(pool, 2)["completed"] == 1
        assert pool.pop_result("recovered").error is None


def test_real_actor_failure_is_reported_and_replaced(ray_cluster):
    with RaySandboxAdapter(ALLOC, CAP, WORK) as pool:
        pool.submit(SandboxTask("crash", duration_s=5))
        ray_cluster.kill(next(iter(pool._actors.values())), no_restart=True)
        assert drain(pool, 1)["failed"] == 1
        assert pool.pop_result("crash").error is not None
        pool.submit(SandboxTask("healthy"))
        assert drain(pool, 2)["completed"] == 1
        assert pool.observe() == ALLOC


def test_real_gpu_and_capacity_rejection(ray_cluster):
    with RaySandboxAdapter(ALLOC, CAP, WORK) as pool:
        with pytest.raises(ValueError, match="GPU"):
            pool.apply(replace(ALLOC, inference_gpus=2))
        with pytest.raises(ValueError, match="CPU"):
            pool.apply(replace(ALLOC, sandbox_workers=3))
        assert pool.observe() == ALLOC
