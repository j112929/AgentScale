"""Offline command line interface."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .models import Allocation, Capacity, Snapshot, Workload
from .policies import BottleneckAwarePolicy, IndependentAutoscalePolicy, StaticPolicy
from .simulator import simulate

SCENARIOS = {
    "balanced": Workload(),
    "inference": Workload(inference_s=5.0, sandbox_s=0.5),
    "sandbox": Workload(inference_s=0.5, sandbox_s=8.0),
    "training": Workload(inference_s=0.5, sandbox_s=0.5, training_s=24.0),
    "dynamic": Workload(inference_s=2.0, sandbox_s=4.0),
}
POLICIES = {
    "static": StaticPolicy,
    "independent": IndependentAutoscalePolicy,
    "global": BottleneckAwarePolicy,
}


def run(scenario: str, policy: str, duration: float, seed: int, trace: bool = False) -> dict:
    cap = Capacity()
    alloc = Allocation()
    workload = SCENARIOS[scenario]
    snapshot = Snapshot(0, alloc, cap, workload, inference_queue=10, sandbox_queue=10)
    plan = POLICIES[policy]().propose(snapshot)
    result = simulate(cap, plan.allocation, workload, duration, seed, trace)
    output = {"scenario": scenario, "policy": policy, "seed": seed,
              "duration_s": duration, "allocation": asdict(plan.allocation),
              "plan_reason": plan.reason, "metrics": result.summary(duration)}
    if trace:
        output["events"] = result.events
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
        if command == "plan":
            p.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    if args.command == "benchmark":
        output = [run(s, p, args.duration, args.seed)
                  for s in SCENARIOS for p in POLICIES]
    else:
        output = run(args.scenario, args.policy, args.duration, args.seed, args.trace)
        if args.command == "plan":
            output = {k: output[k] for k in ("scenario", "policy", "allocation", "plan_reason")}
            output["dry_run"] = True
    payload = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
