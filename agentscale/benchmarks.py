"""Synthetic comparisons and stability ablation with paired seeds and raw data."""

import statistics
from collections import Counter
from dataclasses import asdict, replace

from .cli import run
from .controller import ControllerConfig
from .scenarios import SCENARIOS


def benchmark(
    duration: float = 600, seeds: list[int] | None = None, config: ControllerConfig | None = None
) -> dict:
    seeds = [42, 43, 44] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("benchmark requires nonempty unique seeds")
    cfg = config or ControllerConfig()
    raw = []
    summaries = []
    for scenario in SCENARIOS:
        variants = [
            ("static", "static", True, cfg),
            ("independent", "independent", True, cfg),
            ("global", "global", True, cfg),
            ("global-one-shot", "global", False, cfg),
            (
                "global-no-stability",
                "global",
                True,
                replace(cfg, cooldown_s=0, confirmations=1, min_relative_gain=0),
            ),
        ]
        for variant, policy, closed_loop, settings in variants:
            rows = []
            for seed in seeds:
                row = run(
                    scenario, policy, duration, seed, closed_loop=closed_loop, config=settings
                )
                row["variant"] = variant
                row["decision_counts"] = dict(Counter(d["status"] for d in row["decisions"]))
                row["decisions"] = [
                    d
                    for d in row["decisions"]
                    if d["status"] in ("applied", "rolled_back", "rollback_failed")
                ]
                rows.append(row)
                raw.append(row)
            rates = [row["metrics"]["trained_samples_per_s"] for row in rows]
            summaries.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "mean_trained_samples_per_s": statistics.mean(rates),
                    "stdev_trained_samples_per_s": (
                        statistics.stdev(rates) if len(rates) > 1 else 0
                    ),
                    "mean_allocation_changes": statistics.mean(
                        r["metrics"]["allocation_changes"] for r in rows
                    ),
                    "mean_resource_cost": statistics.mean(
                        r["metrics"]["resource_cost"] for r in rows
                    ),
                }
            )
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "duration_s": duration,
        "seeds": seeds,
        "controller_config": asdict(cfg),
        "method": "paired arrival/service seeds; fixed capacity; full run including warmup",
        "summary": summaries,
        "runs": raw,
    }
