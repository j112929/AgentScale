"""Paired synthetic retrieval experiments alongside the existing four-stage simulator."""

import math
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from ..controller import ControllerConfig
from ..models import Allocation, Capacity, Snapshot, Workload
from ..policies import BottleneckAwarePolicy, StaticPolicy
from ..runtime_benchmark import percentile
from ..simulator import simulate
from .backend import FakeRetrievalBackend
from .controller import RetrievalController, RetrievalControllerConfig
from .integration import ReservationAwarePolicy, ReservationLedger
from .models import Constraints, Query, Strategy
from .planner import AdaptiveRetrievalPlanner

VARIANTS = tuple(strategy.value for strategy in Strategy) + (
    "adaptive-one-shot",
    "adaptive-closed-loop",
    "adaptive-no-queue",
)
SCENARIOS = ("steady", "pressure")


@dataclass(frozen=True)
class ScheduledQuery:
    at_s: float
    query: Query
    constraints: Constraints


def request_trace(duration_s: float, seed: int, scenario: str) -> tuple[ScheduledQuery, ...]:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    if type(seed) is not int or scenario not in SCENARIOS:
        raise ValueError("invalid seed or retrieval scenario")
    arrivals = random.Random(f"{seed}:retrieval-arrivals")
    rows: list[ScheduledQuery] = []
    rates = (2.0, 2.0, 2.0) if scenario == "steady" else (1.0, 6.0, 2.0)
    for phase, rate in enumerate(rates):
        at = phase * duration_s / 3
        end = (phase + 1) * duration_s / 3
        while at < end:
            index = len(rows)
            features = random.Random(f"{seed}:query:{index}")
            query = Query(
                f"q{index}",
                f"synthetic research question {index}",
                difficulty=features.uniform(0.2, 0.9),
                lexicality=0.9 if index % 2 == 0 else 0.1,
            )
            constraints = Constraints(min_quality=0.5 if index % 3 else 0.7, max_latency_s=3)
            rows.append(ScheduledQuery(at, query, constraints))
            at += arrivals.expovariate(rate)
    return tuple(rows)


def run_retrieval(
    duration_s: float = 30,
    seed: int = 42,
    scenario: str = "pressure",
    variant: str = "adaptive-closed-loop",
    *,
    rl_policy: str = "static",
    trace: bool = False,
) -> dict:
    requests = request_trace(duration_s, seed, scenario)
    if variant not in VARIANTS or rl_policy not in ("static", "global"):
        raise ValueError("unknown retrieval variant or RL policy")
    capacity = Capacity(cpu_cores=20, memory_gb=48)
    allocation = Allocation()
    workload = Workload(
        arrival_per_s=0.5, inference_s=0.4, sandbox_s=0.8, steps=1, training_s=0.5, batch_size=1
    )
    planner = AdaptiveRetrievalPlanner()
    fixed = tuple(p for p in planner.candidates() if p.strategy.value == variant and p.top_k == 10)
    config = RetrievalControllerConfig(
        confirmations=1,
        feedback=variant == "adaptive-closed-loop" or variant == "adaptive-no-queue",
        replan=variant != "adaptive-one-shot",
        queue_aware=variant != "adaptive-no-queue",
    )
    # One-shot fixes the initial choice for each query and disables calibration.
    ledger = ReservationLedger()
    retrieval = RetrievalController(
        planner, FakeRetrievalBackend(seed), ledger, config, candidate_plans=fixed or None
    )
    rl_config = ControllerConfig(interval_s=0.1)
    index = 0

    def collect(state: Snapshot) -> None:
        nonlocal index
        while index < len(requests) and requests[index].at_s <= state.time_s + 1e-12:
            request = requests[index]
            retrieval.submit(request.query, request.constraints, request.at_s)
            index += 1
        retrieval.tick(state)

    policy = (
        StaticPolicy() if rl_policy == "static" else BottleneckAwarePolicy(allow_gpu_changes=False)
    )
    rl = simulate(
        capacity,
        allocation,
        workload,
        duration_s,
        seed,
        trace,
        policy=ReservationAwarePolicy(policy, ledger),
        controller_config=rl_config,
        on_snapshot=collect,
        transition_guard=ledger.guard_transition,
    )
    # Final arrivals are submitted chronologically, even when no control tick follows.
    while index < len(requests):
        request = requests[index]
        retrieval.submit(request.query, request.constraints, request.at_s)
        index += 1
    retrieval.complete_until(duration_s)
    completed = retrieval.completions
    latencies = [c.modeled_task_latency_s for c in completed]
    intervals = retrieval.resource_intervals
    cpu_s = sum(
        (min(duration_s, row.finished_s) - row.started_s) * row.resources.cpu_cores
        for row in intervals
    )
    gpu_s = sum(
        (min(duration_s, row.finished_s) - row.started_s) * row.resources.gpus for row in intervals
    )
    admitted = [d for d in retrieval.history if d.status == "admitted"]
    # Search/context accounting is charged at admission; resource accounting is clipped
    # at the horizon, so censored jobs are not billed for unexecuted modeled time.
    cost_model = retrieval.planner.optimizer.estimators.cost
    total_cost = sum(
        cost_model.predict(decision.plan, min(duration_s, interval.finished_s) - interval.started_s)
        for decision, interval in zip(admitted, intervals)
        if decision.plan is not None
    )
    statuses = Counter(d.status for d in retrieval.history)
    terminal = {
        name: statuses[name]
        for name in ("queue_full", "rejected", "expired", "backend_failed", "reservation_conflict")
    }
    metrics = {
        "submitted": len(requests),
        "admitted": len(intervals),
        "completed": len(completed),
        "pending_at_horizon": retrieval.pending_count,
        "running_at_horizon": retrieval.running_count,
        "terminal_counts": terminal,
        "synthetic_quality_mean": statistics.mean(c.result.quality for c in completed)
        if completed
        else None,
        "synthetic_successes": sum(c.result.task_success for c in completed),
        "synthetic_success_fraction_of_submitted": sum(c.result.task_success for c in completed)
        / len(requests),
        "modeled_task_latency_p50_s": percentile(latencies, 0.5) if latencies else None,
        "modeled_task_latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "modeled_task_latency_p99_s": percentile(latencies, 0.99) if latencies else None,
        "modeled_deadline_misses": sum(not c.deadline_met for c in completed),
        "synthetic_quality_misses": sum(not c.quality_met for c in completed),
        "modeled_cost_misses": sum(not c.cost_met for c in completed),
        "modeled_cost_units": total_cost,
        "modeled_cost_per_completed": total_cost / len(completed) if completed else None,
        "modeled_retrieval_cpu_utilization": cpu_s / (capacity.cpu_cores * duration_s),
        "modeled_retrieval_gpu_utilization": gpu_s / (capacity.gpus * duration_s),
        "modeled_search_calls": sum(
            1 if d.plan.strategy in (Strategy.BM25, Strategy.VECTOR) else 2
            for d in admitted
            if d.plan
        ),
        "modeled_inference_input_tokens": sum(128 + 64 * d.plan.top_k for d in admitted if d.plan),
        "strategy_counts": dict(Counter(d.plan.strategy.value for d in admitted if d.plan)),
        "calibration_observations": retrieval.planner.optimizer.estimators.observations,
    }
    output = {
        "schema_version": 1,
        "synthetic_only": True,
        "backend": "fake-v1",
        "seed": seed,
        "scenario": scenario,
        "variant": variant,
        "duration_s": duration_s,
        "capacity": asdict(capacity),
        "initial_rl_allocation": asdict(allocation),
        "rl_workload": asdict(workload),
        "rl_policy": rl_policy,
        "rl_controller_config": asdict(rl_config),
        "retrieval_controller_config": asdict(config),
        "objective": asdict(planner.optimizer.objective),
        "metrics": metrics,
        "rl_metrics": rl.summary(duration_s),
        "request_trace": [asdict(request) for request in requests],
    }
    if trace:
        output["retrieval_decisions"] = [asdict(d) for d in retrieval.history]
        output["retrieval_completions"] = [asdict(c) for c in completed]
        output["resource_intervals"] = [asdict(row) for row in intervals]
        output["rl_decisions"] = rl.decisions
        output["rl_allocations"] = rl.allocations
        output["rl_events"] = rl.events
    return output


def benchmark_retrieval(duration_s: float = 30, seeds: list[int] | None = None) -> dict:
    seeds = [42, 43, 44] if seeds is None else seeds
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("benchmark requires nonempty unique integer seeds")
    raw, traces, summaries = [], [], []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            rows = []
            for seed in seeds:
                row = run_retrieval(duration_s, seed, scenario, variant)
                requests = row.pop("request_trace")
                if variant == VARIANTS[0]:
                    traces.append({"scenario": scenario, "seed": seed, "requests": requests})
                raw.append(row)
                rows.append(row)
            counts = [row["metrics"]["completed"] for row in rows]
            summaries.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "mean_completed": statistics.mean(counts),
                    "stdev_completed": statistics.stdev(counts) if len(counts) > 1 else 0,
                    "mean_synthetic_success_fraction_of_submitted": statistics.mean(
                        r["metrics"]["synthetic_success_fraction_of_submitted"] for r in rows
                    ),
                    "mean_modeled_cost_units": statistics.mean(
                        r["metrics"]["modeled_cost_units"] for r in rows
                    ),
                    "mean_modeled_deadline_misses": statistics.mean(
                        r["metrics"]["modeled_deadline_misses"] for r in rows
                    ),
                    "mean_synthetic_quality_misses": statistics.mean(
                        r["metrics"]["synthetic_quality_misses"] for r in rows
                    ),
                }
            )
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "duration_s": duration_s,
        "seeds": seeds,
        "method": "paired explicit arrivals/query features/backend seeds; fixed shared capacity; "
        "full horizon, no warmup exclusion, unfinished jobs censored; RL policy static",
        "summary": summaries,
        "traces": traces,
        "runs": raw,
    }
