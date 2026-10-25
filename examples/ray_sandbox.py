"""Trusted synthetic Ray sandbox demo; no shell, arbitrary code or GPU migration."""

import json
import time
from dataclasses import asdict

import ray

from agentscale.controller import ControllerConfig, DynamicController
from agentscale.models import Allocation, Capacity, Snapshot, Workload
from agentscale.policies import IndependentAutoscalePolicy
from agentscale.ray_adapter import RaySandboxAdapter, SandboxTask


def main() -> None:
    ray.init(
        num_cpus=2, include_dashboard=False, log_to_driver=False, object_store_memory=100 * 1024**2
    )
    try:
        capacity = Capacity(gpus=2, cpu_cores=2, memory_gb=4)
        workload = Workload(inference_s=0.1, sandbox_s=8, sandbox_cpu=1, sandbox_memory_gb=1)
        allocation = Allocation(1, 1, 1, 8)
        with RaySandboxAdapter(allocation, capacity, workload) as pool:
            controller = DynamicController(
                IndependentAutoscalePolicy(), pool, ControllerConfig(confirmations=1)
            )
            for i in range(8):
                pool.submit(SandboxTask(str(i), f"trusted payload {i}", duration_s=0.05))
            start = time.monotonic()
            deadline = start + 30
            while time.monotonic() < deadline:
                m = pool.poll()
                now = time.monotonic() - start
                snapshot = Snapshot(
                    now,
                    pool.observe(),
                    capacity,
                    workload,
                    sandbox_queue=m["queued"],
                    timestamp_s=now,
                )
                decision = controller.tick(snapshot)
                if decision.status == "applied":
                    print(
                        json.dumps(
                            {"decision": asdict(decision), "allocation": asdict(pool.observe())}
                        )
                    )
                if m["completed"] + m["failed"] == 8:
                    print(
                        json.dumps(
                            {
                                "ray_version": ray.__version__,
                                "metrics": m,
                                "results": [asdict(pool.pop_result(str(i))) for i in range(8)],
                            }
                        )
                    )
                    break
                time.sleep(0.02)
            else:
                raise TimeoutError("demo did not finish within 30 seconds")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
