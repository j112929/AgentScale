import json
from pathlib import Path

from agentscale.runtime_benchmark import container_benchmark

if __name__ == "__main__":
    result = container_benchmark()
    Path("benchmarks/container-local.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "results"} for r in result["runs"]], indent=2
        )
    )
