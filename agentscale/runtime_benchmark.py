"""Measured local container comparisons and strict external RL result comparisons."""

import json
import math
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict

from .controller import ControllerConfig, DynamicController
from .models import Allocation, Capacity, SchedulingPolicy, Snapshot, Workload
from .policies import BottleneckAwarePolicy, IndependentAutoscalePolicy, StaticPolicy
from .sandbox import CodeTask, ContainerSandboxPool, DockerSandbox


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def container_benchmark(seeds: list[int] | None = None, tasks: int = 6) -> dict:
    seeds = [42, 43, 44] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or type(tasks) is not int or tasks < 4:
        raise ValueError("unique seeds and at least four tasks required")
    sandbox = DockerSandbox()
    image_id = sandbox.prepare()
    rows = []
    capacity = Capacity(2, 2, 2)
    workload = Workload(
        arrival_per_s=100,
        inference_s=0.001,
        sandbox_s=1,
        steps=1,
        training_s=0.001,
        sandbox_cpu=1,
        sandbox_memory_gb=1,
        batch_size=1,
    )
    allocation = Allocation(1, 1, 1, 8)
    policies: dict[str, Callable[[], SchedulingPolicy]] = {
        "static": StaticPolicy,
        "independent": IndependentAutoscalePolicy,
        "global": lambda: BottleneckAwarePolicy(allow_gpu_changes=False),
    }
    for seed in seeds:
        # Rotate order deterministically to reduce systematic cache/order bias.
        variants = list(policies)
        random.Random(seed).shuffle(variants)
        delays = [random.Random(f"{seed}:{i}").uniform(0.01, 0.05) for i in range(tasks)]
        for policy_name in variants:
            with ContainerSandboxPool(allocation, capacity, workload, sandbox) as pool:
                controller = DynamicController(
                    policies[policy_name](),
                    pool,
                    ControllerConfig(
                        interval_s=0.1, cooldown_s=0.2, confirmations=1, observation_window_s=0.5
                    ),
                )
                start = time.monotonic()
                results: list[dict] = []
                latencies: list[float] = []
                queues: list[float] = []
                for i, delay in enumerate(delays):
                    source = f"import time\ntime.sleep({delay!r})\nassert sum([1,2,3])==6\n"
                    pool.submit(CodeTask(str(i), {"test.py": source}, ("python", "test.py")))
                deadline = start + tasks * 20
                while len(results) < tasks:
                    if time.monotonic() > deadline:
                        raise TimeoutError("container benchmark deadline exceeded")
                    metrics = pool.poll()
                    now = time.monotonic() - start
                    queues.append(metrics["oldest_queue_s"])
                    controller.tick(
                        Snapshot(
                            now,
                            pool.observe(),
                            capacity,
                            workload,
                            sandbox_queue=int(metrics["queued"]),
                            timestamp_s=now,
                        )
                    )
                    for i in range(tasks):
                        result = pool.pop_result(str(i))
                        if result:
                            results.append(asdict(result))
                            latencies.append(time.monotonic() - start)
                    time.sleep(0.005)
                elapsed = time.monotonic() - start
                rows.append(
                    {
                        "backend": "real-docker",
                        "policy": policy_name,
                        "seed": seed,
                        "tasks": tasks,
                        "elapsed_s": elapsed,
                        "passed": sum(r["status"] == "passed" for r in results),
                        "completed_tasks_per_s": len(results) / elapsed,
                        "latency_p50_s": percentile(latencies, 0.5),
                        "latency_p95_s": percentile(latencies, 0.95),
                        "latency_p99_s": percentile(latencies, 0.99),
                        "oldest_queue_p95_s": percentile(queues, 0.95),
                        "allocation_changes": sum(
                            d.status == "applied" for d in controller.history
                        ),
                        "results": results,
                    }
                )
    return {
        "schema_version": 1,
        "backend": "real-docker",
        "synthetic_inference": True,
        "rl_training": False,
        "image_id": image_id,
        "capacity": asdict(capacity),
        "limits": asdict(sandbox.limits),
        "seeds": seeds,
        "runs": rows,
        "unavailable_baselines": {"verl-native": "requires a GPU RL training environment"},
    }


def compare_rl_runs(rows: list[dict], bootstrap_samples: int = 2000) -> dict:
    """Compare measured fresh-training goodput, never container completions or tokens/s."""
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap resamples required")
    groups: dict[str, dict[int, float]] = {}
    reference = None
    for row in rows:
        signature = tuple(
            json.dumps(row[k], sort_keys=True)
            for k in (
                "hardware_id",
                "capacity",
                "model_revision",
                "dataset_sha256",
                "runtime_versions",
                "training_config",
            )
        )
        if reference is None:
            reference = signature
        if signature != reference or row.get("backend") != "measured-rl":
            raise ValueError("results must share hardware, model, dataset and runtime versions")
        duration = row["duration_s"]
        trained = row["fresh_trained_samples"]
        if (
            not math.isfinite(duration)
            or duration <= 0
            or type(trained) is not int
            or trained < 0
            or type(row["seed"]) is not int
        ):
            raise ValueError("invalid duration, sample count or seed")
        seeds = groups.setdefault(row["policy"], {})
        if row["seed"] in seeds:
            raise ValueError("duplicate policy/seed pair")
        seeds[row["seed"]] = trained / duration
    if "static" not in groups or len(groups["static"]) < 2:
        raise ValueError("at least two static baseline seeds required")
    rng = random.Random(0)
    comparisons = []
    for policy, rates in groups.items():
        if rates.keys() != groups["static"].keys():
            raise ValueError("all policies must use the same paired seeds")
        deltas = [rates[s] - groups["static"][s] for s in sorted(rates)]
        means = [
            statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(bootstrap_samples)
        ]
        comparisons.append(
            {
                "policy": policy,
                "mean_fresh_samples_per_s": statistics.mean(rates.values()),
                "paired_gain_per_s": statistics.mean(deltas),
                "bootstrap_95_percent_interval": [
                    percentile(means, 0.025),
                    percentile(means, 0.975),
                ],
            }
        )
    return {
        "schema_version": 1,
        "backend": "measured-rl",
        "comparisons": comparisons,
        "missing_baselines": sorted(
            {"static", "independent", "global", "verl-native"} - groups.keys()
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": 0,
    }
