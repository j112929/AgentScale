from dataclasses import replace

import pytest

from agentscale.adapters import FakeAdapter, SafePointUnavailable
from agentscale.controller import ControllerConfig, DynamicController
from agentscale.models import Allocation, Capacity, Plan, Snapshot, Workload

OLD = Allocation()
TARGET = replace(OLD, sandbox_workers=24)
WORK = Workload(inference_s=0.5, sandbox_s=8)


class FixedPolicy:
    def __init__(self, target=TARGET):
        self.target = target

    def propose(self, state):
        return Plan(self.target, "test proposal", 999, 0)


def make(adapter=None, policy=None, **kwargs):
    adapter = adapter or FakeAdapter(OLD, Capacity(), WORK)
    cfg = ControllerConfig(interval_s=1, cooldown_s=6, observation_window_s=5, **kwargs)
    return DynamicController(policy or FixedPolicy(), adapter, cfg)


def state(controller, at, rate=None, **kwargs):
    return Snapshot(
        at,
        controller.adapter.observe(),
        Capacity(),
        WORK,
        timestamp_s=at,
        trained_samples_per_s=rate,
        **kwargs,
    )


def test_interval_confirmation_and_idempotency():
    c = make()
    assert c.tick(state(c, 0)).status == "confirming"
    assert c.tick(state(c, 0.5)).status == "not_due"
    assert c.tick(state(c, 1)).status == "applied"
    assert c.tick(state(c, 2)).status == "observing"
    assert c.tick(state(c, 6)).status == "cooldown"
    assert c.tick(state(c, 7)).status == "unchanged"
    assert c.adapter.writes == 1
    assert c.generation == 1


def test_alternating_proposals_do_not_oscillate():
    p = FixedPolicy()
    c = make(policy=p)
    for at in range(10):
        p.target = TARGET if at % 2 else replace(TARGET, sandbox_workers=28)
        assert c.tick(state(c, at)).status == "confirming"
    assert c.adapter.writes == 0


def test_stale_metrics_reset_confirmation():
    c = make()
    c.tick(state(c, 0))
    assert c.tick(replace(state(c, 20), timestamp_s=0)).status == "invalid_telemetry"
    assert c.tick(state(c, 21)).status == "confirming"
    assert c.adapter.writes == 0


@pytest.mark.parametrize(
    "change",
    [
        {"timestamp_s": float("nan")},
        {"timestamp_s": 1},
        {"trained_samples_per_s": float("nan")},
        {"trained_samples_per_s": -1},
        {"inference_queue": -1},
        {"sandbox_queue": 0.5},
    ],
)
def test_invalid_metrics_do_not_actuate(change):
    c = make(confirmations=1)
    assert c.tick(replace(state(c, 0), **change)).status == "invalid_telemetry"
    assert c.adapter.writes == 0


def test_clock_and_observed_drift_fail_safe():
    c = make(confirmations=1)
    assert c.tick(state(c, float("nan"))).status == "invalid_telemetry"
    assert c.tick(replace(state(c, 2), allocation=TARGET)).status == "drift"
    assert c.tick(state(c, 1)).status == "invalid_telemetry"
    assert c.adapter.writes == 0


def test_margin_switch_cost_and_resource_validation():
    c = make(policy=FixedPolicy(replace(OLD, sandbox_workers=9)), min_relative_gain=0.2)
    assert c.tick(state(c, 0)).status == "hysteresis"
    c = make(switch_s_per_unit=100)
    assert c.tick(state(c, 0)).status == "hysteresis"
    c = make(policy=FixedPolicy(replace(OLD, inference_gpus=20)))
    assert c.tick(state(c, 0)).status == "invalid_plan"
    assert c.adapter.writes == 0


def test_dry_run_keeps_runtime_and_generation_unchanged():
    c = make(confirmations=1)
    c.dry_run = True
    assert c.tick(state(c, 0)).status == "dry_run"
    assert c.adapter.observe() == OLD
    assert c.generation == c.adapter.writes == 0


class FaultyAdapter(FakeAdapter):
    calls = 0
    permanent = False

    def apply(self, target):
        self.calls += 1
        super().apply(target)
        if self.calls == 1 or self.permanent:
            raise TimeoutError("partial operation timed out")


def test_partial_apply_failure_compensates():
    a = FaultyAdapter(OLD, Capacity(), WORK)
    c = make(adapter=a, confirmations=1)
    assert c.tick(state(c, 0)).status == "rolled_back"
    assert a.observe() == OLD
    assert a.calls == 2
    assert not c.faulted


def test_failed_compensation_latches_and_blocks_further_writes():
    a = FaultyAdapter(OLD, Capacity(), WORK)
    a.permanent = True
    c = make(adapter=a, confirmations=1)
    assert c.tick(state(c, 0)).status == "rollback_failed"
    assert c.tick(state(c, 10)).status == "blocked"
    assert a.calls == 2


def warmup_trial(adapter=None):
    p = FixedPolicy(OLD)
    c = make(policy=p, adapter=adapter, confirmations=1)
    for at in range(6):
        c.tick(state(c, at, rate=1))
    p.target = TARGET
    assert c.tick(state(c, 6, rate=1)).status == "applied"
    return c


def test_sustained_performance_regression_rolls_back_after_full_window():
    c = warmup_trial()
    for at in range(7, 11):
        assert c.tick(state(c, at, rate=0)).status == "observing"
    assert c.tick(state(c, 11, rate=0)).status == "rolled_back"
    assert c.adapter.observe() == OLD
    assert c.tick(state(c, 12, rate=1)).status == "cooldown"


def test_one_empty_batch_interval_does_not_trigger_rollback():
    c = warmup_trial()
    for at in range(7, 12):
        d = c.tick(state(c, at, rate=0 if at == 7 else 1))
    assert d.status != "rolled_back"
    assert c.adapter.observe() == TARGET


def test_workload_shift_does_not_compare_unrelated_performance():
    c = warmup_trial()
    d = c.tick(replace(state(c, 11, rate=0), workload=replace(WORK, sandbox_s=4)))
    assert d.status != "rolled_back"
    assert c.adapter.observe() == TARGET


def test_rollback_can_wait_for_a_safe_point():
    class BusyRollback(FakeAdapter):
        busy = True

        def apply(self, target):
            if target == OLD and self.busy:
                raise SafePointUnavailable("draining")
            super().apply(target)

    a = BusyRollback(OLD, Capacity(), WORK)
    c = warmup_trial(a)
    for at in range(7, 11):
        c.tick(state(c, at, rate=0))
    assert c.tick(state(c, 11, rate=0)).status == "rollback_deferred"
    assert not c.faulted
    a.busy = False
    assert c.tick(state(c, 12, rate=0)).status == "rolled_back"
    assert a.observe() == OLD


def test_missing_performance_samples_cannot_trigger_rollback():
    c = warmup_trial()
    for at in range(7, 11):
        c.tick(state(c, at))
    assert c.tick(state(c, 11, rate=0)).status != "rolled_back"
    assert c.adapter.observe() == TARGET


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_s": 0},
        {"cooldown_s": -1},
        {"confirmations": 0},
        {"confirmations": 1.5},
        {"switch_s_per_unit": float("inf")},
        {"rollback_relative_drop": 1},
        {"observation_window_s": 0},
    ],
)
def test_invalid_controller_configuration(kwargs):
    with pytest.raises(ValueError):
        ControllerConfig(**kwargs)
