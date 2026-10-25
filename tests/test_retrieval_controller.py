from dataclasses import replace

import pytest

from agentscale.models import Allocation, Capacity, Snapshot, Workload
from agentscale.retrieval import (
    Constraints,
    FakeRetrievalBackend,
    Query,
    Resources,
    RetrievalPlan,
    Strategy,
)
from agentscale.retrieval.controller import RetrievalController, RetrievalControllerConfig
from agentscale.retrieval.integration import residual_capacity
from agentscale.retrieval.models import QueueState
from agentscale.retrieval.planner import Objective


def state(at=0, **kwargs):
    return Snapshot(
        at, Allocation(), Capacity(cpu_cores=20, memory_gb=48), Workload(), timestamp_s=at, **kwargs
    )


def make(**kwargs):
    return RetrievalController(config=RetrievalControllerConfig(confirmations=1), **kwargs)


def submit(controller, name="q", constraints=None, at=0):
    return controller.submit(Query(name, "retrieve evidence"), constraints or Constraints(), at)


def test_fifo_mixed_requests_share_capacity_release_and_reproduce():
    def run():
        controller = make()
        for i in range(6):
            submit(controller, str(i), Constraints(max_latency_s=20))
        for i in range(201):
            controller.tick(state(i / 10))
            assert controller.ledger.total.fits(residual_capacity(state()))
        assert controller.pending_count == controller.running_count == 0
        assert controller.ledger.total == Resources()
        assert len(controller.completions) == 6
        assert controller.planner.optimizer.estimators.observations == 6
        return controller.completions, controller.history

    assert run() == run()


def test_rl_queue_pressure_changes_strategy_and_is_recorded_in_task_latency():
    clear, busy = make(), make()
    for controller in (clear, busy):
        submit(controller, constraints=Constraints(max_latency_s=1.3))
    clear.tick(state())
    busy.tick(state(inference_queue=2))
    assert clear.history[-1].plan.strategy == Strategy.HYBRID_RERANK
    assert busy.history[-1].plan.strategy == Strategy.HYBRID
    busy.tick(state(2))
    completion = busy.completions[0]
    assert completion.modeled_task_latency_s == pytest.approx(completion.result.service_s + 0.5)


def test_stale_future_or_invalid_queue_metrics_reset_confirmation():
    controller = RetrievalController()
    submit(controller)
    assert controller.tick(state())[0].status == "confirming"
    assert controller.tick(replace(state(2), timestamp_s=0))[0].status == "invalid_telemetry"
    assert controller.tick(state(2.1))[0].status == "confirming"
    assert controller.tick(state(2.2))[0].status == "admitted"
    controller = make()
    submit(controller)
    assert controller.tick(state(), QueueState(1))[0].status == "invalid_telemetry"
    assert controller.tick(state(0.1, inference_queue=-1))[0].status == "invalid_telemetry"
    assert controller.running_count == 0


def test_interval_idempotency_dry_run_and_no_feedback_before_completion():
    controller = make(dry_run=True)
    submit(controller)
    assert controller.tick(state())[0].status == "dry_run"
    assert controller.tick(state()) == ()
    assert controller.ledger.total == Resources()
    assert controller.running_count == controller.planner.optimizer.estimators.observations == 0
    controller = make()
    submit(controller)
    controller.tick(state())
    assert controller.planner.optimizer.estimators.observations == 0
    assert controller.tick(state()) == ()
    assert controller.tick(state(-1))[0].status == "invalid_telemetry"
    controller.tick(state(2))
    assert controller.planner.optimizer.estimators.observations == 1


def test_bounded_queue_duplicates_and_nonchronological_submissions():
    controller = RetrievalController(config=RetrievalControllerConfig(max_pending=1))
    assert submit(controller)
    assert not submit(controller, "full")
    assert controller.history[-1].status == "queue_full"
    with pytest.raises(ValueError, match="duplicate"):
        submit(controller)
    controller.tick(state(1))
    with pytest.raises(ValueError, match="chronological"):
        submit(controller, "old", at=0)
    with pytest.raises(ValueError):
        submit(controller, "nan", at=float("nan"))


def test_impossible_constraints_and_expired_requests_never_execute():
    controller = make()
    submit(controller, constraints=Constraints(min_quality=1))
    assert controller.tick(state())[0].status == "rejected"
    submit(controller, "late", Constraints(max_latency_s=0.1))
    assert controller.tick(state(1))[0].status == "expired"
    assert controller.ledger.total == Resources() and not controller.completions


@pytest.mark.parametrize("failure", ["exception", "mismatch", "real"])
def test_backend_failure_and_invalid_results_release_reservations(failure):
    class BadBackend:
        def retrieve(self, query, plan):
            if failure == "exception":
                raise RuntimeError("offline backend failure")
            result = FakeRetrievalBackend().retrieve(query, plan)
            return (
                replace(result, query_id="other")
                if failure == "mismatch"
                else replace(result, synthetic=False)
            )

    controller = make(backend=BadBackend())
    submit(controller)
    assert controller.tick(state())[0].status == "backend_failed"
    assert controller.ledger.total == Resources()
    assert controller.pending_count == controller.running_count == 0


def test_foreign_reservation_token_cannot_be_released_or_double_counted():
    controller = make(candidate_plans=(RetrievalPlan(Strategy.HYBRID),))
    controller.ledger.reserve("q", Resources(2, 3), state())
    submit(controller)
    assert controller.tick(state())[0].status == "reservation_conflict"
    assert controller.ledger.total == Resources(2, 3)


def test_external_capacity_drift_blocks_new_work_and_does_not_evict_running_job():
    controller = make()
    submit(controller)
    controller.tick(state())
    submit(controller, "pending", at=0.1)
    conflict = replace(state(0.1), allocation=Allocation(sandbox_workers=10))
    assert controller.tick(conflict)[0].status == "invalid_telemetry"
    assert controller.running_count == 1
    controller.tick(state(2))
    assert controller.completions[0].result.plan.strategy == Strategy.HYBRID_RERANK
    assert controller.running_count == 1


def test_observed_constraint_misses_are_reported_and_calibrated():
    class SlowBackend:
        def retrieve(self, query, plan):
            return replace(
                FakeRetrievalBackend().retrieve(query, plan),
                service_s=6,
                quality=0.1,
                task_success=False,
            )

    controller = make(backend=SlowBackend())
    submit(controller, constraints=Constraints(max_cost_units=0.5))
    controller.tick(state())
    controller.tick(state(6))
    completion = controller.completions[0]
    assert not completion.deadline_met and not completion.quality_met
    assert not completion.cost_met
    assert controller.planner.optimizer.estimators.observations == 1


@pytest.mark.parametrize("margin,should_change", [(0, True), (1000, False)])
def test_cooldown_and_benefit_margin_guard_pending_plan_changes(margin, should_change):
    controller = RetrievalController(
        config=RetrievalControllerConfig(confirmations=1, cooldown_s=0.5, min_relative_gain=margin)
    )
    controller.ledger.reserve("other", Resources(3, 1), state())
    submit(controller)
    controller.tick(state())
    initial = controller.history[-1].plan
    controller.planner.optimizer.objective = Objective(latency_weight=2)
    controller.tick(state(0.1))
    assert controller.history[-1].plan == initial
    controller.tick(state(0.6))
    assert (controller.history[-1].plan != initial) == should_change


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_s": 0},
        {"confirmations": 0},
        {"max_pending": True},
        {"cooldown_s": float("inf")},
        {"feedback": 1},
    ],
)
def test_invalid_controller_configuration(kwargs):
    with pytest.raises(ValueError):
        RetrievalControllerConfig(**kwargs)
