"""Offline command line interface and reproducible closed-loop benchmarks."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .controller import ControllerConfig
from .models import Allocation, Capacity, SchedulingPolicy, Snapshot
from .policies import BottleneckAwarePolicy, IndependentAutoscalePolicy, StaticPolicy
from .scenarios import SCENARIOS, shifts
from .simulator import simulate

POLICIES: dict[str, Callable[[], SchedulingPolicy]] = {
    "static": StaticPolicy,
    "independent": IndependentAutoscalePolicy,
    "global": BottleneckAwarePolicy,
}


def run(
    scenario: str,
    policy: str,
    duration: float,
    seed: int,
    trace: bool = False,
    *,
    closed_loop: bool = True,
    config: ControllerConfig | None = None,
) -> dict:
    cap, alloc, workload = Capacity(), Allocation(), SCENARIOS[scenario]
    cfg = config or ControllerConfig()
    planner = POLICIES[policy]()
    plan = planner.propose(Snapshot(0, alloc, cap, workload))
    initial = alloc if closed_loop else plan.allocation
    result = simulate(
        cap,
        initial,
        workload,
        duration,
        seed,
        trace,
        policy=planner if closed_loop else None,
        controller_config=cfg,
        workload_shifts=shifts(scenario, duration),
    )
    output = {
        "scenario": scenario,
        "policy": policy,
        "seed": seed,
        "duration_s": duration,
        "closed_loop": closed_loop,
        "capacity": asdict(cap),
        "initial_allocation": asdict(initial),
        "allocation": result.allocations[-1]["allocation"],
        "workload": asdict(workload),
        "workload_shifts": [
            {"time_s": at, "workload": asdict(w)} for at, w in shifts(scenario, duration)
        ],
        "controller_config": asdict(cfg),
        "plan_reason": plan.reason,
        "metrics": result.summary(duration),
        "allocations": result.allocations,
        "decisions": result.decisions,
    }
    if trace:
        output["events"] = result.events
        output["telemetry"] = [asdict(s) for s in result.telemetry]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentscale")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("simulate", "benchmark", "plan"):
        p = sub.add_parser(command)
        p.add_argument("--scenario", choices=SCENARIOS, default="balanced")
        p.add_argument("--policy", choices=POLICIES, default="global")
        p.add_argument("--duration", type=float, default=120)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output", type=Path)
        p.add_argument("--trace", action="store_true")
        p.add_argument("--interval", type=float, default=5)
        p.add_argument("--one-shot", action="store_true")
        if command == "plan":
            p.add_argument("--dry-run", action="store_true", default=True)
        if command == "benchmark":
            p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    cfg = ControllerConfig(interval_s=args.interval)
    if args.command == "benchmark":
        from .benchmarks import benchmark

        output = benchmark(args.duration, args.seeds, cfg)
    elif args.command == "plan":
        w = SCENARIOS[args.scenario]
        plan = POLICIES[args.policy]().propose(Snapshot(0, Allocation(), Capacity(), w))
        output = {
            "scenario": args.scenario,
            "policy": args.policy,
            "allocation": asdict(plan.allocation),
            "plan_reason": plan.reason,
            "dry_run": True,
        }
    else:
        output = run(
            args.scenario,
            args.policy,
            args.duration,
            args.seed,
            args.trace,
            closed_loop=not args.one_shot,
            config=cfg,
        )
    payload = json.dumps(output, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
