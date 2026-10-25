"""Clock-driven, auditable control loop; no wall-clock sleeping in scheduling logic."""

import math
from dataclasses import dataclass
from typing import Protocol

from .adapters import SafePointUnavailable
from .models import (
    Allocation,
    ResourceAdapter,
    SchedulingPolicy,
    Snapshot,
    Workload,
    estimate_throughput,
    validate,
)


class TelemetryProvider(Protocol):
    def collect(self) -> Snapshot: ...


def switch_units(before: Allocation, after: Allocation) -> float:
    return (
        abs(before.inference_gpus - after.inference_gpus)
        + abs(before.training_gpus - after.training_gpus)
        + abs(before.sandbox_workers - after.sandbox_workers) / 4
        + abs(before.rollout_concurrency - after.rollout_concurrency) / 16
    )


@dataclass(frozen=True)
class ControllerConfig:
    interval_s: float = 5.0
    cooldown_s: float = 15.0
    confirmations: int = 2
    min_relative_gain: float = 0.05
    switch_s_per_unit: float = 0.1
    benefit_horizon_s: float = 60.0
    telemetry_max_age_s: float = 10.0
    observation_window_s: float = 30.0
    rollback_relative_drop: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.interval_s,
            self.cooldown_s,
            self.min_relative_gain,
            self.switch_s_per_unit,
            self.benefit_horizon_s,
            self.telemetry_max_age_s,
            self.observation_window_s,
            self.rollback_relative_drop,
        )
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("controller parameters must be finite and nonnegative")
        if min(self.interval_s, self.benefit_horizon_s, self.observation_window_s) <= 0:
            raise ValueError("interval, horizon and observation window must be positive")
        if type(self.confirmations) is not int or self.confirmations < 1:
            raise ValueError("confirmations must be a positive integer")
        if self.rollback_relative_drop >= 1:
            raise ValueError("rollback drop must be less than one")


@dataclass(frozen=True)
class Decision:
    time_s: float
    status: str
    reason: str
    allocation: Allocation
    generation: int


@dataclass
class _Trial:
    previous: Allocation
    target: Allocation
    workload: Workload
    started_s: float
    baseline: float | None
    measured_samples: float = 0.0
    measured_s: float = 0.0
    last_observation_s: float = 0.0


class DynamicController:
    def __init__(
        self,
        policy: SchedulingPolicy,
        adapter: ResourceAdapter,
        config: ControllerConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.policy = policy
        self.adapter = adapter
        self.config = config or ControllerConfig()
        self.dry_run = dry_run
        self.history: list[Decision] = []
        self.generation = 0
        self.faulted = False
        self._last_poll = -math.inf
        self._last_change = -math.inf
        self._candidate: Allocation | None = None
        self._confirmations = 0
        self._trial: _Trial | None = None
        self._rollback_target: Allocation | None = None
        self._rollback_reason = ""
        self._workload: Workload | None = None
        self._workload_since = 0.0
        self._rates: list[tuple[float, float]] = []

    def _record(self, state: Snapshot, status: str, reason: str) -> Decision:
        decision = Decision(state.time_s, status, reason, state.allocation, self.generation)
        self.history.append(decision)
        return decision

    def _reset_candidate(self) -> None:
        self._candidate, self._confirmations = None, 0

    def _restore(self, state: Snapshot, previous: Allocation, reason: str) -> Decision:
        try:
            self.adapter.apply(previous)
            if self.adapter.observe() != previous:
                raise RuntimeError("rollback did not converge")
        except SafePointUnavailable as exc:
            self._rollback_target = previous
            self._rollback_reason = reason
            return self._record(state, "rollback_deferred", str(exc))
        except Exception as exc:
            self.faulted = True
            self._trial = None
            self._reset_candidate()
            return self._record(state, "rollback_failed", f"{reason}; compensation: {exc}")
        self.generation += 1
        self._last_change = state.time_s
        self._trial = None
        self._rollback_target = None
        self._reset_candidate()
        return self._record(state, "rolled_back", reason)

    def poll(self, provider: TelemetryProvider) -> Decision:
        return self.tick(provider.collect())

    def tick(self, state: Snapshot) -> Decision:
        cfg = self.config
        if self.faulted:
            return self._record(
                state, "blocked", "rollback failed; operator reconciliation required"
            )
        if not math.isfinite(state.time_s) or state.time_s < self._last_poll:
            return self._record(state, "invalid_telemetry", "clock must be finite and monotonic")
        if state.time_s - self._last_poll < cfg.interval_s - 1e-9:
            return self._record(state, "not_due", "waiting for telemetry interval")
        self._last_poll = state.time_s
        age = state.time_s - state.timestamp_s
        queues = (state.inference_queue, state.sandbox_queue, state.training_buffer)
        rate = state.trained_samples_per_s
        errors = validate(state.allocation, state.capacity, state.workload)
        if (
            not math.isfinite(age)
            or age < 0
            or age > cfg.telemetry_max_age_s
            or any(type(q) is not int or q < 0 for q in queues)
            or (rate is not None and (not math.isfinite(rate) or rate < 0))
            or errors
        ):
            self._reset_candidate()
            return self._record(
                state, "invalid_telemetry", "; ".join(errors) or "stale or invalid metrics"
            )
        try:
            observed = self.adapter.observe()
        except Exception as exc:
            self._reset_candidate()
            return self._record(state, "observe_failed", str(exc))
        if observed != state.allocation:
            self._reset_candidate()
            self._trial = None
            return self._record(state, "drift", "observed allocation differs; recollect telemetry")
        if self._rollback_target is not None:
            return self._restore(state, self._rollback_target, self._rollback_reason)
        if state.workload != self._workload:
            self._workload = state.workload
            self._workload_since = state.time_s
            self._rates.clear()
        self._rates = [
            (t, r) for t, r in self._rates if state.time_s - t <= cfg.observation_window_s
        ]
        if rate is not None:
            self._rates.append((state.time_s, rate))
        if self._trial is not None:
            trial = self._trial
            if rate is not None:
                elapsed = min(cfg.interval_s, state.time_s - trial.last_observation_s)
                trial.measured_samples += rate * elapsed
                trial.measured_s += elapsed
            trial.last_observation_s = state.time_s
            if trial.workload != state.workload:
                self._trial = None  # A different workload is not a valid before/after comparison.
            elif observed != trial.target:
                self._trial = None
                return self._record(state, "drift", "trial allocation changed externally")
            elif state.time_s - trial.started_s < cfg.observation_window_s:
                return self._record(
                    state, "observing", "waiting for performance observation window"
                )
            else:
                self._trial = None
                if (
                    trial.baseline is not None
                    and trial.baseline > 0
                    and trial.measured_s >= cfg.observation_window_s - 1e-9
                    and trial.measured_samples / trial.measured_s
                    < trial.baseline * (1 - cfg.rollback_relative_drop)
                ):
                    return self._restore(
                        state, trial.previous, "measured training throughput regressed"
                    )
        if state.time_s - self._last_change < cfg.cooldown_s:
            self._reset_candidate()
            return self._record(state, "cooldown", "minimum allocation dwell time")
        try:
            plan = self.policy.propose(state)
        except Exception as exc:
            self._reset_candidate()
            return self._record(state, "planning_failed", str(exc))
        errors = validate(plan.allocation, state.capacity, state.workload)
        if errors:
            self._reset_candidate()
            return self._record(state, "invalid_plan", "; ".join(errors))
        if plan.allocation == observed:
            self._reset_candidate()
            return self._record(state, "unchanged", plan.reason)
        old_rate = estimate_throughput(observed, state.workload)
        new_rate = estimate_throughput(plan.allocation, state.workload)
        switch_s = switch_units(observed, plan.allocation) * cfg.switch_s_per_unit
        net_gain = new_rate * max(0, cfg.benefit_horizon_s - switch_s) - (
            old_rate * cfg.benefit_horizon_s
        )
        if new_rate <= old_rate * (1 + cfg.min_relative_gain) or net_gain <= 0:
            self._reset_candidate()
            return self._record(state, "hysteresis", "benefit below margin or switching cost")
        if self._candidate == plan.allocation:
            self._confirmations += 1
        else:
            self._candidate, self._confirmations = plan.allocation, 1
        if self._confirmations < cfg.confirmations:
            return self._record(state, "confirming", "waiting for repeated consistent proposals")
        if self.dry_run:
            return self._record(state, "dry_run", plan.reason)
        try:
            self.adapter.apply(plan.allocation)
            if self.adapter.observe() != plan.allocation:
                raise RuntimeError("apply did not converge to desired allocation")
        except SafePointUnavailable as exc:
            return self._record(state, "deferred", str(exc))
        except Exception as exc:
            return self._restore(state, observed, f"apply failed: {exc}")
        self.generation += 1
        self._last_change = state.time_s
        baseline = (
            sum(r for _, r in self._rates) / len(self._rates)
            if self._rates
            and state.time_s - self._workload_since >= cfg.observation_window_s
            and len(self._rates) * cfg.interval_s >= cfg.observation_window_s
            else None
        )
        self._trial = _Trial(
            observed,
            plan.allocation,
            state.workload,
            state.time_s,
            baseline,
            last_observation_s=state.time_s,
        )
        self._reset_candidate()
        return self._record(state, "applied", plan.reason)
