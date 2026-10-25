"""Offline fault injection; no Ray installation or processes required."""

import hashlib
from dataclasses import replace

import pytest

from agentscale.adapters import SafePointUnavailable
from agentscale.controller import ControllerConfig, DynamicController
from agentscale.models import Allocation, Capacity, Snapshot, Workload
from agentscale.policies import IndependentAutoscalePolicy
from agentscale.ray_adapter import RaySandboxAdapter, SandboxTask

ALLOC = Allocation(1, 1, 1, 8)
CAP = Capacity(2, 4, 8)
WORK = Workload(inference_s=0.1, sandbox_s=8, sandbox_cpu=1, sandbox_memory_gb=1)


class Ref:
    def __init__(self, value=None, done=True, error=None):
        self.value, self.done, self.error = value, done, error


class Method:
    def __init__(self, fn):
        self.remote = fn


class Actor:
    def __init__(self, ray):
        self.ray = ray
        self.killed = False
        self.ready = Method(
            lambda: Ref(True, error=TimeoutError("not ready") if ray.fail_ready else None)
        )
        self.execute = Method(self.run)

    def run(self, task):
        ref = Ref(hashlib.sha256(task.payload.encode()).hexdigest(), done=False)
        self.ray.refs.append(ref)
        return ref


class RemoteClass:
    def __init__(self, ray):
        self.ray = ray

    def options(self, **options):
        self.ray.options.append(options)
        return self

    def remote(self):
        actor = Actor(self.ray)
        self.ray.actors.append(actor)
        return actor


class FakeRay:
    def __init__(self):
        self.actors, self.refs, self.options = [], [], []
        self.fail_ready = False
        self.fail_kill_once = False

    def is_initialized(self):
        return True

    def remote(self, cls):
        return RemoteClass(self)

    def cluster_resources(self):
        return {"CPU": 4, "memory": 8 * 1024**3}

    def get(self, ref, timeout):
        if isinstance(ref, list):
            return [self.get(r, timeout) for r in ref]
        if ref.error:
            raise ref.error
        return ref.value

    def wait(self, refs, num_returns, timeout):
        return [r for r in refs if r.done], [r for r in refs if not r.done]

    def kill(self, actor, no_restart):
        if self.fail_kill_once:
            self.fail_kill_once = False
            raise RuntimeError("transient kill failure")
        actor.killed = True


def make(**kwargs):
    ray = FakeRay()
    return RaySandboxAdapter(ALLOC, CAP, WORK, ray_module=ray, **kwargs), ray


def test_explicit_resources_and_idempotent_scale():
    a, ray = make()
    assert ray.options[0] == {
        "num_cpus": 1,
        "memory": 1024**3,
        "max_restarts": 0,
        "max_task_retries": 0,
    }
    target = replace(ALLOC, sandbox_workers=3)
    a.apply(target)
    a.apply(target)
    assert len(ray.actors) == 3
    assert a.observe() == target
    a.apply(ALLOC)
    assert sum(actor.killed for actor in ray.actors) == 2
    a.close()
    a.close()
    assert all(actor.killed for actor in ray.actors)


def test_busy_scale_in_defers_without_killing_inflight_tasks():
    a, ray = make()
    a.apply(replace(ALLOC, sandbox_workers=2))
    a.submit(SandboxTask("a"))
    a.submit(SandboxTask("b"))
    with pytest.raises(SafePointUnavailable):
        a.apply(ALLOC)
    assert not any(actor.killed for actor in ray.actors)
    for ref in ray.refs:
        ref.done = True
    assert a.poll()["completed"] == 2
    a.apply(ALLOC)
    assert a.observe() == ALLOC
    a.close()


def test_partial_scaleout_timeout_cleans_up_new_actors_and_preserves_old_pool():
    a, ray = make()
    ray.fail_ready = True
    with pytest.raises(TimeoutError):
        a.apply(replace(ALLOC, sandbox_workers=3))
    assert a.observe() == ALLOC
    assert not ray.actors[0].killed
    assert all(actor.killed for actor in ray.actors[1:])
    ray.fail_ready = False
    a.close()


def test_cleanup_failure_is_visible_for_compensation():
    a, ray = make()
    ray.fail_ready = True
    ray.fail_kill_once = True
    with pytest.raises(TimeoutError):
        a.apply(replace(ALLOC, sandbox_workers=2))
    assert a.observe().sandbox_workers == 2
    a.apply(ALLOC)
    assert a.observe() == ALLOC
    a.close()


def test_timeouts_and_actor_crashes_produce_errors_and_replace_workers():
    clock = [0.0]
    a, ray = make(clock=lambda: clock[0], task_timeout_s=1)
    a.submit(SandboxTask("timeout", duration_s=100))
    clock[0] = 2
    assert a.poll()["timed_out"] == 1
    assert "deadline" in a.pop_result("timeout").error
    assert ray.actors[0].killed
    a.submit(SandboxTask("crash"))
    ray.refs[-1].done = True
    ray.refs[-1].error = RuntimeError("actor died")
    assert a.poll()["failed"] == 2
    assert "actor died" in a.pop_result("crash").error
    assert a.observe() == ALLOC
    a.close()


def test_queue_bound_duplicates_concurrency_and_results():
    a, ray = make(max_queue=1)
    a.submit(SandboxTask("first", "payload"))
    with pytest.raises(ValueError, match="duplicate"):
        a.submit(SandboxTask("first"))
    a.submit(SandboxTask("second"))
    with pytest.raises(OverflowError):
        a.submit(SandboxTask("third"))
    ray.refs[0].done = True
    m = a.poll()
    assert m["completed"] == 1 and m["running"] == 1 and m["queued"] == 0
    assert a.pop_result("first").digest == hashlib.sha256(b"payload").hexdigest()
    assert a.pop_result("first") is None
    with pytest.raises(SafePointUnavailable):
        a.close()
    a.close(force=True)
    assert a.pop_result("second").error == "pool closed"
    with pytest.raises(RuntimeError, match="closed"):
        a.submit(SandboxTask("closed"))


def test_unsupported_gpu_and_overcommit_do_not_create_actors():
    a, ray = make()
    for target in (Allocation(2, 1, 1, 8), replace(ALLOC, sandbox_workers=5)):
        with pytest.raises(ValueError):
            a.apply(target)
    assert len(ray.actors) == 1
    a.close()


def test_controller_operates_ray_pool_using_queue_telemetry():
    a, ray = make()
    for i in range(6):
        a.submit(SandboxTask(str(i)))
    c = DynamicController(
        IndependentAutoscalePolicy(), a, ControllerConfig(confirmations=1, interval_s=1)
    )
    m = a.telemetry()
    s = Snapshot(0, a.observe(), CAP, WORK, sandbox_queue=m["queued"], timestamp_s=0)
    assert c.tick(s).status == "applied"
    assert a.observe().sandbox_workers == 3
    assert len(ray.refs) == 3
    a.close(force=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_queue": 0},
        {"task_timeout_s": float("nan")},
        {"readiness_timeout_s": 0},
    ],
)
def test_invalid_pool_configuration(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_task_duration(duration):
    with pytest.raises(ValueError):
        SandboxTask("bad", duration_s=duration)
