from dataclasses import replace

import pytest

from agentscale.adapters import FakeAdapter, SafePointUnavailable
from agentscale.controller import ControllerConfig, DynamicController
from agentscale.models import Allocation, Capacity, Plan, Snapshot, Workload, validate
from agentscale.policies import BottleneckAwarePolicy, IndependentAutoscalePolicy, StaticPolicy
from agentscale.retrieval import Resources
from agentscale.retrieval.integration import (
    ReservationAwareAdapter,
    ReservationAwarePolicy,
    ReservationLedger,
    residual_capacity,
)
from agentscale.simulator import simulate


def state():
    return Snapshot(0, Allocation(), Capacity(cpu_cores=20, memory_gb=48), Workload(sandbox_s=8))


def test_reservation_admission_idempotency_release_and_no_overcommit():
    snapshot = state()
    ledger = ReservationLedger()
    assert residual_capacity(snapshot) == Resources(4, 16)
    ledger.reserve("q", Resources(4, 4), snapshot)
    ledger.reserve("q", Resources(4, 4), snapshot)
    assert ledger.total == Resources(4, 4)
    for token, demand in (
        ("another", Resources(1)),
        ("q", Resources(2)),
        ("gpu", Resources(gpus=1)),
    ):
        with pytest.raises(ValueError):
            ledger.reserve(token, demand, snapshot)
    assert ledger.total == Resources(4, 4)
    ledger.release("missing")
    ledger.release("q")
    assert ledger.total == Resources()
    with pytest.raises(ValueError, match="GPU overcommit"):
        residual_capacity(replace(snapshot, allocation=Allocation(inference_gpus=7)))


@pytest.mark.parametrize("cls", [StaticPolicy, IndependentAutoscalePolicy, BottleneckAwarePolicy])
def test_empty_ledger_preserves_rl_planning_and_full_simulation(cls):
    original = cls()
    wrapped = ReservationAwarePolicy(cls(), ReservationLedger())
    snapshot = state()
    assert wrapped.propose(snapshot) == original.propose(snapshot)
    args = (snapshot.capacity, snapshot.allocation, snapshot.workload)
    assert simulate(*args, duration_s=12, seed=17, policy=original, trace=True) == simulate(
        *args, duration_s=12, seed=17, policy=wrapped, trace=True
    )


def test_active_reservations_constrain_existing_global_controller():
    snapshot = replace(state(), allocation=Allocation(sandbox_workers=4))
    ledger = ReservationLedger()
    ledger.reserve("rag", Resources(4, 4), snapshot)
    policy = ReservationAwarePolicy(BottleneckAwarePolicy(), ledger)
    adapter = FakeAdapter(snapshot.allocation, snapshot.capacity, snapshot.workload)
    controller = DynamicController(policy, adapter, ControllerConfig(confirmations=1))
    for at in (0, 5, 10, 20, 40):
        snapshot = replace(snapshot, time_s=at, timestamp_s=at, allocation=adapter.observe())
        controller.tick(snapshot)
        assert ledger.total.fits(residual_capacity(replace(snapshot, allocation=adapter.observe())))
    assert adapter.writes > 0
    assert adapter.observe().sandbox_workers <= 8
    ledger.release("rag")
    after = policy.propose(replace(snapshot, allocation=adapter.observe()))
    assert after.allocation.sandbox_workers > 8


def test_gpu_reservations_are_logical_and_rl_policy_cannot_claim_them():
    snapshot = replace(state(), allocation=Allocation(inference_gpus=3))
    ledger = ReservationLedger()
    ledger.reserve("gpu", Resources(2, 4, 1), snapshot)
    plan = ReservationAwarePolicy(BottleneckAwarePolicy(), ledger).propose(snapshot)
    assert plan.allocation.inference_gpus + plan.allocation.training_gpus <= 7
    assert not validate(plan.allocation, Capacity(7, 18, 44), snapshot.workload)


def test_drift_and_noncompliant_inner_policies_fail_closed():
    snapshot = state()
    ledger = ReservationLedger()
    ledger.reserve("rag", Resources(4, 4), snapshot)

    class BadPolicy:
        def propose(self, _state):
            return Plan(Allocation(sandbox_workers=10), "overcommit", 1, 1)

    with pytest.raises(ValueError, match="ignored"):
        ReservationAwarePolicy(BadPolicy(), ledger).propose(snapshot)
    with pytest.raises(ValueError, match="conflict"):
        ReservationAwarePolicy(StaticPolicy(), ledger).propose(
            replace(snapshot, allocation=Allocation(sandbox_workers=10))
        )


def test_allocation_fence_also_defers_rollback_until_retrieval_drains():
    snapshot = state()
    adapter = FakeAdapter(snapshot.allocation, snapshot.capacity, snapshot.workload)
    ledger = ReservationLedger()
    fenced = ReservationAwareAdapter(adapter, ledger, snapshot.capacity, snapshot.workload)
    previous = replace(snapshot.allocation, sandbox_workers=10)
    ledger.reserve("job", Resources(4, 4), snapshot)
    with pytest.raises(SafePointUnavailable, match="retrieval"):
        fenced.apply(previous)
    assert adapter.writes == 0
    ledger.release("job")
    fenced.apply(previous)
    assert fenced.observe() == previous


def test_simulator_callbacks_share_live_four_stage_queues_and_fence_all_writes():
    from agentscale.retrieval import Constraints, Query
    from agentscale.retrieval.controller import RetrievalController, RetrievalControllerConfig

    ledger = ReservationLedger()
    retrieval = RetrievalController(
        ledger=ledger, config=RetrievalControllerConfig(confirmations=1)
    )
    snapshot = replace(state(), workload=Workload(steps=1, batch_size=1, arrival_per_s=2))
    observed = []

    def collect(current):
        if current.time_s == 0:
            retrieval.submit(Query("q", "retrieval while RL trains"), Constraints(), 0)
        retrieval.tick(current)
        assert ledger.total.fits(residual_capacity(current))
        observed.append(current)

    result = simulate(
        snapshot.capacity,
        snapshot.allocation,
        snapshot.workload,
        duration_s=20,
        seed=42,
        policy=ReservationAwarePolicy(BottleneckAwarePolicy(), ledger),
        controller_config=ControllerConfig(interval_s=0.5),
        on_snapshot=collect,
        transition_guard=ledger.guard_transition,
    )
    assert retrieval.completions and result.trained > 0
    assert any(s.inference_queue or s.sandbox_queue or s.training_buffer for s in observed)


def test_measured_rl_rollback_waits_for_reservations_and_then_recovers():
    capacity = Capacity(cpu_cores=20, memory_gb=48)
    workload = Workload(
        steps=1, batch_size=1, training_s=20, inference_s=0.1, sandbox_s=0.1, arrival_per_s=100
    )
    old, target = Allocation(sandbox_workers=10), Allocation(3, 5, 8, 32)
    adapter = FakeAdapter(old, capacity, workload)
    ledger = ReservationLedger()
    fenced = ReservationAwareAdapter(adapter, ledger, capacity, workload)

    class Policy:
        desired = old

        def propose(self, snapshot):
            return Plan(self.desired, "trial", 1, 1)

    policy = Policy()
    controller = DynamicController(
        ReservationAwarePolicy(policy, ledger),
        fenced,
        ControllerConfig(
            interval_s=1,
            confirmations=1,
            cooldown_s=0,
            observation_window_s=2,
            min_relative_gain=0,
            switch_s_per_unit=0,
        ),
    )

    def snapshot(at, rate):
        return Snapshot(
            at, adapter.observe(), capacity, workload, timestamp_s=at, trained_samples_per_s=rate
        )

    for at in (0, 1, 2):
        controller.tick(snapshot(at, 1))
    policy.desired = target
    assert controller.tick(snapshot(3, 1)).status == "applied"
    ledger.reserve("rag", Resources(4, 4), snapshot(3, 1))
    assert controller.tick(snapshot(4, 0)).status == "observing"
    assert controller.tick(snapshot(5, 0)).status == "rollback_deferred"
    assert adapter.observe() == target and not controller.faulted
    ledger.release("rag")
    assert controller.tick(snapshot(6, 0)).status == "rolled_back"
    assert adapter.observe() == old
