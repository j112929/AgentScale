import json
import sys

import pytest

from agentscale import cli
from agentscale.benchmarks import benchmark


def test_plan_does_not_run_simulation(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("plan must not simulate or actuate")

    monkeypatch.setattr(cli, "simulate", forbidden)
    monkeypatch.setattr(sys, "argv", ["agentscale", "plan", "--scenario", "sandbox", "--dry-run"])
    cli.main()
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["allocation"]["sandbox_workers"] > 8


def test_cli_simulation_writes_valid_trace_json(monkeypatch, tmp_path, capsys):
    path = tmp_path / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agentscale",
            "simulate",
            "--scenario",
            "dynamic",
            "--duration",
            "12",
            "--trace",
            "--output",
            str(path),
        ],
    )
    cli.main()
    result = json.loads(path.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["closed_loop"] is True
    assert result["workload_shifts"][0]["time_s"] == 4
    assert result["telemetry"] and result["decisions"]


def test_benchmark_paired_budget_seeds_and_ablation():
    data = benchmark(12, [42])
    assert data["synthetic_only"] is True
    assert len(data["summary"]) == len(data["runs"]) == 25
    assert {r["variant"] for r in data["runs"]} == {
        "static",
        "independent",
        "global",
        "global-one-shot",
        "global-no-stability",
    }
    assert all(
        r["capacity"] == {"gpus": 8, "cpu_cores": 64, "memory_gb": 128} for r in data["runs"]
    )
    assert all(r["seed"] == 42 for r in data["runs"])
    assert all(r["controller_config"]["switch_s_per_unit"] == 0.1 for r in data["runs"])
    assert all(sum(r["decision_counts"].values()) == 3 for r in data["runs"] if r["closed_loop"])


@pytest.mark.parametrize("seeds", [[], [42, 42]])
def test_benchmark_rejects_empty_or_duplicated_seeds(seeds):
    with pytest.raises(ValueError):
        benchmark(12, seeds)
