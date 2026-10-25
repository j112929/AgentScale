from copy import deepcopy

import pytest

from agentscale.runtime_benchmark import compare_rl_runs, percentile


def rows():
    return [
        {
            "backend": "measured-rl",
            "policy": policy,
            "seed": seed,
            "duration_s": 100,
            "fresh_trained_samples": 100 + gain + seed,
            "hardware_id": "cluster-A",
            "capacity": {"gpus": 2},
            "model_revision": "pinned-sha",
            "dataset_sha256": "a" * 64,
            "runtime_versions": {"verl": "0.6.1"},
            "training_config": {"batch": 4},
        }
        for policy, gain in [("static", 0), ("global", 100)]
        for seed in [1, 2, 3]
    ]


def test_paired_bootstrap_and_explicit_missing_native_baseline():
    r = compare_rl_runs(rows(), 100)
    global_result = next(x for x in r["comparisons"] if x["policy"] == "global")
    assert global_result["paired_gain_per_s"] == pytest.approx(1)
    assert global_result["bootstrap_95_percent_interval"] == pytest.approx([1, 1])
    assert "verl-native" in r["missing_baselines"]
    assert percentile([1, 2, 3], 0.5) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "real-docker"),
        ("capacity", {"gpus": 4}),
        ("hardware_id", "other"),
        ("duration_s", 0),
        ("fresh_trained_samples", -1),
    ],
)
def test_incomparable_or_invalid_runs_rejected(field, value):
    data = rows()
    data[-1][field] = value
    with pytest.raises(ValueError):
        compare_rl_runs(data, 100)


def test_duplicate_and_unpaired_seeds_rejected():
    data = rows()
    with pytest.raises(ValueError):
        compare_rl_runs(data + [deepcopy(data[0])], 100)
    with pytest.raises(ValueError):
        compare_rl_runs(data[:-1], 100)
