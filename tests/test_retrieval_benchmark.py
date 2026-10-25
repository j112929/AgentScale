import json
import sys

import pytest

from agentscale import cli
from agentscale.retrieval.benchmark import (
    SCENARIOS,
    VARIANTS,
    benchmark_retrieval,
    request_trace,
    run_retrieval,
)


def test_paired_benchmark_reproducibility_budgets_and_censor_accounting():
    result = benchmark_retrieval(3, [11])
    assert result == benchmark_retrieval(3, [11])
    assert result["synthetic_only"]
    assert len(result["runs"]) == len(result["summary"]) == len(SCENARIOS) * len(VARIANTS)
    assert len(result["traces"]) == len(SCENARIOS)
    assert {r["variant"] for r in result["runs"]} == set(VARIANTS)
    assert all(
        r["capacity"] == {"gpus": 8, "cpu_cores": 20, "memory_gb": 48} for r in result["runs"]
    )
    for row in result["runs"]:
        metrics = row["metrics"]
        assert metrics["submitted"] == (
            metrics["completed"]
            + metrics["pending_at_horizon"]
            + metrics["running_at_horizon"]
            + sum(metrics["terminal_counts"].values())
        )
        assert 0 <= metrics["modeled_retrieval_cpu_utilization"] <= 0.2 + 1e-12
        assert metrics["modeled_retrieval_gpu_utilization"] == 0
        assert row["rl_policy"] == "static"
        if row["variant"] in (
            "adaptive-one-shot",
            *[s for s in VARIANTS if not s.startswith("adaptive")],
        ):
            assert metrics["calibration_observations"] == 0
    json.dumps(result, allow_nan=False)


def test_shared_timeline_trace_accounts_for_resource_time_cost_and_rl_training():
    result = run_retrieval(20, 42, trace=True)
    metrics = result["metrics"]
    cpu_s = sum(
        (min(20, row["finished_s"]) - row["started_s"]) * row["resources"]["cpu_cores"]
        for row in result["resource_intervals"]
    )
    assert metrics["modeled_retrieval_cpu_utilization"] == pytest.approx(cpu_s / (20 * 20))
    assert result["rl_metrics"]["trained"] > 0
    assert result["retrieval_completions"] and result["rl_events"]
    assert sum(metrics["strategy_counts"].values()) == metrics["admitted"]
    assert all(c["result"]["synthetic"] for c in result["retrieval_completions"])
    assert metrics["modeled_cost_units"] > 0


def test_global_rl_scheduler_can_run_with_retrieval_fences():
    result = run_retrieval(6, 42, rl_policy="global", trace=True)
    assert result["rl_decisions"] and result["retrieval_decisions"]
    assert all(d["status"] != "invalid_telemetry" for d in result["retrieval_decisions"])


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [1.2]])
def test_bad_seed_sets(seeds):
    with pytest.raises(ValueError):
        benchmark_retrieval(3, seeds)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_bad_durations(duration):
    with pytest.raises(ValueError):
        request_trace(duration, 42, "pressure")


def test_retrieval_plan_is_dry_run_and_infeasibility_has_explicit_exit(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("planning must not start a simulator")

    monkeypatch.setattr(cli, "simulate", forbidden)
    monkeypatch.setattr(sys, "argv", ["agentscale", "retrieval-plan", "--query", "evidence"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and result["synthetic_only"] and result["feasible"]
    monkeypatch.setattr(
        sys, "argv", ["agentscale", "retrieval-plan", "--query", "evidence", "--min-quality", "1"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert not json.loads(capsys.readouterr().out)["feasible"]


@pytest.mark.parametrize("command", ["retrieval-simulate", "retrieval-benchmark"])
def test_retrieval_cli_roundtrips_finite_json(command, monkeypatch, tmp_path, capsys):
    path = tmp_path / "results.json"
    monkeypatch.setattr(
        sys, "argv", ["agentscale", command, "--duration", "2", "--output", str(path)]
    )
    cli.main()
    result = json.loads(path.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["synthetic_only"]
