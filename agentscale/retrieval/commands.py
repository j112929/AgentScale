"""CLI registration stays lightweight; experiments load only when selected."""

import argparse
from dataclasses import asdict
from pathlib import Path


def add_commands(subparsers) -> None:
    parser = subparsers.add_parser("retrieval-plan", help="dry-run synthetic retrieval planning")
    parser.add_argument("--query", required=True)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--lexicality", type=float, default=0.5)
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--allow-gpu", action="store_true", help="include modeled GPU reranking")
    parser.add_argument("--min-quality", type=float, default=0.55)
    parser.add_argument("--min-success", type=float, default=0.4)
    parser.add_argument("--max-latency", type=float, default=5)
    parser.add_argument("--max-cost", type=float, default=3)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--output", type=Path)
    parser = subparsers.add_parser("retrieval-simulate", help="synthetic retrieval plus RL queues")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=("steady", "pressure"), default="pressure")
    parser.add_argument(
        "--variant",
        default="adaptive-closed-loop",
        choices=(
            "bm25",
            "vector",
            "hybrid",
            "hybrid-rerank",
            "adaptive-one-shot",
            "adaptive-closed-loop",
            "adaptive-no-queue",
        ),
    )
    parser.add_argument("--rl-policy", choices=("static", "global"), default="static")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    parser = subparsers.add_parser("retrieval-benchmark", help="paired fake-backend comparisons")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path)


def execute(args: argparse.Namespace) -> dict:
    if args.command == "retrieval-plan":
        from .models import Constraints, Query, QueueState, Resources
        from .planner import AdaptiveRetrievalPlanner

        query = Query("cli", args.query, args.difficulty, args.lexicality)
        capacity = Resources(args.cpu_cores, args.memory_gb, args.gpus)
        constraints = Constraints(
            args.min_quality, args.min_success, args.max_latency, args.max_cost
        )
        planner = AdaptiveRetrievalPlanner(allow_gpu=args.allow_gpu)
        result = planner.propose(query, constraints, capacity, QueueState(0))
        return {
            "synthetic_only": True,
            "dry_run": True,
            "query": asdict(query),
            "budget": asdict(capacity),
            "constraints": asdict(constraints),
            "feasible": result.selected is not None,
            **asdict(result),
        }
    from .benchmark import benchmark_retrieval, run_retrieval

    if args.command == "retrieval-benchmark":
        return benchmark_retrieval(args.duration, args.seeds)
    return run_retrieval(
        args.duration,
        args.seed,
        args.scenario,
        args.variant,
        rl_policy=args.rl_policy,
        trace=args.trace,
    )
