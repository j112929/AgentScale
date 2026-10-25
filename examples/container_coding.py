"""Real container coding loop; use --vllm-url/--model for a real inference server."""

import argparse
import json
import os
from dataclasses import asdict

from agentscale.coding import EXAMPLE_PROBLEM, solve
from agentscale.sandbox import DockerSandbox
from agentscale.vllm_adapter import ChatResult, VLLMAdapter


class ScriptedFixture:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=1024):
        self.calls += 1
        op = "-" if self.calls == 1 else "+"
        return ChatResult(
            json.dumps({"files": {"solution.py": f"def add(a,b):\n    return a {op} b\n"}}),
            "scripted-test-fixture",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url")
    parser.add_argument("--model")
    args = parser.parse_args()
    if bool(args.vllm_url) != bool(args.model):
        parser.error("--vllm-url and --model must be provided together")
    client = (
        VLLMAdapter(args.vllm_url, args.model, api_key=os.getenv("VLLM_API_KEY"))
        if args.vllm_url
        else ScriptedFixture()
    )
    result = solve(EXAMPLE_PROBLEM, client, DockerSandbox())
    print(
        json.dumps(
            {
                "inference": "vllm" if args.vllm_url else "scripted-fixture",
                "real_containers": True,
                "result": asdict(result),
            },
            indent=2,
        )
    )
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
