"""Deterministic retrieval-only control loop; immutable in-flight work and no sleeps."""

import math
from collections import deque
from dataclasses import dataclass

from ..models import Snapshot
from .backend import FakeRetrievalBackend, RetrievalBackend
from .integration import ReservationLedger, residual_capacity
from .models import (
    Constraints,
    Query,
    QueueState,
    Resources,
    RetrievalPlan,
    RetrievalResult,
    nonnegative,
)
from .planner import AdaptiveRetrievalPlanner, Candidate, OptimizationResult, violations


@dataclass(frozen=True)
class RetrievalControllerConfig:
    interval_s: float = 0.1
    telemetry_max_age_s: float = 1.0
    confirmations: int = 2
    cooldown_s: float = 0.3
    min_relative_gain: float = 0.03
    max_pending: int = 64
    feedback: bool = True
    queue_aware: bool = True
    replan: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.interval_s,
            self.telemetry_max_age_s,
            self.cooldown_s,
            self.min_relative_gain,
        ):
            nonnegative(value, "controller configuration")
        if self.interval_s == 0:
            raise ValueError("interval must be positive")
        for value in (self.confirmations, self.max_pending):
            if type(value) is not int or value < 1:
                raise ValueError("confirmations and queue bound must be positive integers")
        for value in (self.feedback, self.queue_aware, self.replan):
            if type(value) is not bool:
                raise ValueError("controller flags must be booleans")


@dataclass(frozen=True)
class RetrievalDecision:
    time_s: float
    query_id: str
    status: str
    reason: str
    plan: RetrievalPlan | None = None


@dataclass(frozen=True)
class Completion:
    query: Query
    result: RetrievalResult
    submitted_s: float
    started_s: float
    finished_s: float
    modeled_task_latency_s: float
    cost_units: float
    deadline_met: bool
    quality_met: bool


@dataclass(frozen=True)
class ResourceInterval:
    started_s: float
    finished_s: float
    resources: Resources


@dataclass
class _Pending:
    query: Query
    constraints: Constraints
    submitted_s: float
    candidate: Candidate | None = None
    confirmations: int = 0
    chosen: Candidate | None = None
    changed_s: float = -math.inf


@dataclass(frozen=True)
class _Running:
    request: _Pending
    candidate: Candidate
    result: RetrievalResult
    started_s: float
    finished_s: float
    downstream_delay_s: float


class RetrievalController:
    """Synthetic executor sharing an RL reservation ledger, not a real search runtime.

    Replans only pending FIFO requests. Once admitted, a result and finish event are
    immutable. Feedback becomes visible only at completion, never at plan time.
    """

    def __init__(
        self,
        planner: AdaptiveRetrievalPlanner | None = None,
        backend: RetrievalBackend | None = None,
        ledger: ReservationLedger | None = None,
        config: RetrievalControllerConfig | None = None,
        *,
        candidate_plans: tuple[RetrievalPlan, ...] | None = None,
        dry_run: bool = False,
    ) -> None:
        if candidate_plans is not None and not candidate_plans:
            raise ValueError("candidate plans must not be empty")
        self.planner = planner or AdaptiveRetrievalPlanner()
        self.backend = backend or FakeRetrievalBackend()
        self.ledger = ledger if ledger is not None else ReservationLedger()
        self.config = config or RetrievalControllerConfig()
        self.candidate_plans = candidate_plans
        self.dry_run = dry_run
        self.history: list[RetrievalDecision] = []
        self.completions: list[Completion] = []
        self.resource_intervals: list[ResourceInterval] = []
        self._pending: deque[_Pending] = deque()
        self._running: dict[str, _Running] = {}
        self._seen: set[str] = set()
        self._last_seen_s = -math.inf
        self._last_poll_s = -math.inf
        self._last_submission_s = -math.inf

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def running_count(self) -> int:
        return len(self._running)

    def _record(
        self, now: float, query_id: str, status: str, reason: str, plan: RetrievalPlan | None = None
    ) -> None:
        self.history.append(RetrievalDecision(now, query_id, status, reason, plan))

    def submit(self, query: Query, constraints: Constraints, submitted_s: float) -> bool:
        nonnegative(submitted_s, "submission time")
        if query.query_id in self._seen:
            raise ValueError("duplicate query ID")
        if submitted_s < self._last_submission_s or submitted_s < self._last_seen_s:
            raise ValueError("submissions must be chronological and not precede the last tick")
        self._last_submission_s = submitted_s
        self._seen.add(query.query_id)
        if len(self._pending) >= self.config.max_pending:
            self._record(submitted_s, query.query_id, "queue_full", "bounded pending queue is full")
            return False
        self._pending.append(_Pending(query, constraints, submitted_s))
        return True

    def _complete(self, now: float) -> None:
        for key, job in sorted(
            self._running.items(), key=lambda item: (item[1].finished_s, item[0])
        ):
            if job.finished_s > now + 1e-12:
                continue
            self.ledger.release(key)
            del self._running[key]
            request, result = job.request, job.result
            if self.config.feedback:
                self.planner.optimizer.estimators.observe(request.query, result)
            latency = job.finished_s - request.submitted_s + job.downstream_delay_s
            cost = self.planner.optimizer.estimators.cost.predict(result.plan, result.service_s)
            self.completions.append(
                Completion(
                    request.query,
                    result,
                    request.submitted_s,
                    job.started_s,
                    job.finished_s,
                    latency,
                    cost,
                    latency <= request.constraints.max_latency_s + 1e-12,
                    result.quality >= request.constraints.min_quality - 1e-12,
                )
            )
            self._record(job.finished_s, key, "completed", "synthetic completion", result.plan)

    def complete_until(self, now: float) -> None:
        """Process finish events at a simulation horizon, without inventing new telemetry."""
        nonnegative(now, "completion horizon")
        if now < self._last_seen_s:
            raise ValueError("completion horizon must be monotonic")
        self._last_seen_s = now
        self._complete(now)

    def _queue(self, state: Snapshot, external: QueueState) -> QueueState:
        cpu = sum(
            max(0, job.finished_s - state.time_s) * job.candidate.estimate.resources.cpu_cores
            for job in self._running.values()
        )
        gpu = sum(
            max(0, job.finished_s - state.time_s) * job.candidate.estimate.resources.gpus
            for job in self._running.values()
        )
        a, w = state.allocation, state.workload
        downstream = (
            state.inference_queue * w.inference_s / (a.inference_gpus * w.inference_gpu_efficiency)
            + state.sandbox_queue * w.sandbox_s / a.sandbox_workers
        )
        return QueueState(
            state.time_s,
            cpu + external.cpu_core_s,
            gpu + external.gpu_s,
            downstream + external.downstream_delay_s,
        )

    def _select(
        self, request: _Pending, state: Snapshot, queue: QueueState, *, empty_queue: bool = False
    ) -> OptimizationResult:
        plans = self.candidate_plans or self.planner.candidates()
        if not self.config.replan and request.chosen is not None:
            plans = (request.chosen.plan,)
        planning_queue = (
            QueueState(state.time_s) if empty_queue or not self.config.queue_aware else queue
        )
        return self.planner.optimizer.select(
            request.query,
            plans,
            request.constraints,
            residual_capacity(state),
            planning_queue,
            state.time_s - request.submitted_s,
        )

    def tick(
        self, state: Snapshot, queue: QueueState | None = None
    ) -> tuple[RetrievalDecision, ...]:
        start = len(self.history)
        now, cfg = state.time_s, self.config
        if not math.isfinite(now) or now < 0 or now < self._last_seen_s:
            # Invalid numbers never enter JSON history.
            self._record(max(0.0, self._last_seen_s), "", "invalid_telemetry", "invalid clock")
            return tuple(self.history[start:])
        self._last_seen_s = now
        self._complete(now)
        if now - self._last_poll_s < cfg.interval_s - 1e-12:
            return tuple(self.history[start:])
        self._last_poll_s = now
        external = queue or QueueState(now)
        try:
            capacity = residual_capacity(state)
            ages = (now - state.timestamp_s, now - external.timestamp_s)
            if any(
                not math.isfinite(age) or not 0 <= age <= cfg.telemetry_max_age_s for age in ages
            ):
                raise ValueError("stale or future telemetry")
            if any(
                type(q) is not int or q < 0
                for q in (state.inference_queue, state.sandbox_queue, state.training_buffer)
            ):
                raise ValueError("invalid RL queue metrics")
            if not self.ledger.total.fits(capacity):
                raise ValueError("active reservations conflict with current RL allocation")
        except ValueError as exc:
            for request in self._pending:
                request.confirmations = 0
                request.candidate = None
            self._record(now, "", "invalid_telemetry", str(exc))
            return tuple(self.history[start:])
        while self._pending and self._pending[0].submitted_s <= now:
            request = self._pending[0]
            key = request.query.query_id
            if now - request.submitted_s >= request.constraints.max_latency_s:
                self._pending.popleft()
                self._record(now, key, "expired", "deadline elapsed while pending")
                continue
            metrics = self._queue(state, external)
            result = self._select(request, state, metrics)
            candidate = result.selected
            if candidate is None:
                request.candidate, request.confirmations = None, 0
                future = self._select(request, state, metrics, empty_queue=True)
                if future.selected is None:
                    self._pending.popleft()
                    self._record(now, key, "rejected", "; ".join(result.rejected))
                    continue
                self._record(now, key, "queued", "queue pressure prevents a feasible plan")
                break
            old = request.chosen
            if old is not None and candidate.plan != old.plan:
                old_estimate = self.planner.optimizer.estimators.estimate(
                    request.query,
                    old.plan,
                    metrics if cfg.queue_aware else QueueState(now),
                    capacity,
                    now - request.submitted_s,
                )
                if not violations(old_estimate, request.constraints, capacity):
                    old_utility = self.planner.optimizer.objective.utility(old_estimate)
                    gain = candidate.utility - old_utility
                    if (
                        now - request.changed_s < cfg.cooldown_s
                        or gain <= abs(old_utility) * cfg.min_relative_gain
                    ):
                        candidate = Candidate(old.plan, old_estimate, old_utility)
            if request.candidate is not None and candidate.plan == request.candidate.plan:
                request.confirmations += 1
            else:
                request.candidate, request.confirmations = candidate, 1
            if request.confirmations < cfg.confirmations:
                self._record(now, key, "confirming", "waiting for consistent plans", candidate.plan)
                break
            if old is None or old.plan != candidate.plan:
                request.changed_s = now
            request.chosen = candidate
            if not candidate.estimate.resources.fits(capacity - self.ledger.total):
                self._record(now, key, "queued", "waiting for shared resources", candidate.plan)
                break
            if self.dry_run:
                self._record(
                    now, key, "dry_run", "proposed plan; no reservation/execution", candidate.plan
                )
                break
            if self.ledger.contains(key):
                self._pending.popleft()
                self._record(now, key, "reservation_conflict", "reservation token already exists")
                continue
            self.ledger.reserve(key, candidate.estimate.resources, state)
            try:
                answer = self.backend.retrieve(request.query, candidate.plan)
                if (
                    answer.query_id != key
                    or answer.plan != candidate.plan
                    or not answer.synthetic
                    or answer.backend != "fake-v1"
                ):
                    raise ValueError("synthetic controller requires matching fake-v1 results")
            except Exception as exc:  # noqa: BLE001 -- release reservations at the backend boundary.
                self.ledger.release(key)
                self._pending.popleft()
                self._record(now, key, "backend_failed", str(exc), candidate.plan)
                continue
            self._pending.popleft()
            self._running[key] = _Running(
                request, candidate, answer, now, now + answer.service_s, metrics.downstream_delay_s
            )
            self.resource_intervals.append(
                ResourceInterval(now, now + answer.service_s, candidate.estimate.resources)
            )
            self._record(now, key, "admitted", "reserved shared resources", candidate.plan)
        return tuple(self.history[start:])
