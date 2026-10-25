"""Public, versioned synthetic demand profiles. Dynamic shifts use thirds of a run."""

from dataclasses import replace

from .models import Workload

SCENARIOS = {
    "balanced": Workload(),
    "inference": Workload(inference_s=5.0, sandbox_s=0.5),
    "sandbox": Workload(inference_s=0.5, sandbox_s=8.0),
    "training": Workload(inference_s=0.5, sandbox_s=0.5, training_s=24.0),
    "dynamic": Workload(inference_s=5.0, sandbox_s=0.5),
}


def shifts(scenario: str, duration: float) -> tuple[tuple[float, Workload], ...]:
    if scenario != "dynamic":
        return ()
    initial = SCENARIOS[scenario]
    return (
        (duration / 3, replace(initial, inference_s=0.5, sandbox_s=8.0)),
        (2 * duration / 3, replace(initial, inference_s=0.5, sandbox_s=0.5, training_s=24.0)),
    )
