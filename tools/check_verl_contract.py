"""Validate the optional tool against genuine pinned verl modules without loading torch.

Only BaseTool, schemas and rollout_trace are loaded. This is not a full verl trainer run.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

REVISION = "d62da4950573d7a4b7ef2362337952e7ab59e78d"
FILES = (
    "verl/tools/base_tool.py",
    "verl/tools/schemas.py",
    "verl/utils/rollout_trace.py",
    "verl/version/version",
)


def check(root, execute):
    manifest = json.loads(Path("configs/upstream-contracts.json").read_text())["verl"]
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("upstream source hash mismatch: " + name)
    if (root / "verl/version/version").read_text().strip() != "0.6.1":
        raise ValueError("verl 0.6.1 sources required")
    # Load actual lightweight upstream modules, avoiding verl.__init__'s torch import.
    for name in ("verl", "verl.tools", "verl.utils"):
        namespace = types.ModuleType(name)
        namespace.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = namespace
    from verl.tools.schemas import OpenAIFunctionToolSchema

    from agentscale.integrations.verl_tool import ContainerCodeTool
    from agentscale.sandbox import CodeResult

    base = importlib.import_module("verl.tools.base_tool").BaseTool
    assert issubclass(ContainerCodeTool, base)
    config = json.loads(Path("configs/verl-tools.json").read_text())["tools"][0]
    schema = OpenAIFunctionToolSchema.model_validate(config["tool_schema"])
    tool = ContainerCodeTool(config["config"], schema)
    if not execute:

        class ContractSandbox:
            def execute(self, task):
                return CodeResult(task.task_id, "passed", 0, "contract fixture")

        tool._sandbox = ContractSandbox()

    async def lifecycle():
        instance, _ = await tool.create()
        response, reward, metrics = await tool.execute(
            instance, {"files": {"test.py": "print(42)"}, "command": ["python", "test.py"]}
        )
        assert response.text and reward == 1 and metrics["sandbox_timeout"] == 0
        assert await tool.calc_reward(instance) == 1
        await tool.release(instance)
        assert not tool._instances

    asyncio.run(lifecycle())
    print(
        json.dumps(
            {
                "upstream": "verl",
                "version": "0.6.1",
                "revision": REVISION,
                "contract": "BaseTool lifecycle + native ToolResponse",
                "real_container": execute,
                "full_training": False,
                "status": "passed",
                "source_hashes_verified": True,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.download == bool(args.source_root):
        parser.error("choose exactly one of --source-root or --download")
    if args.source_root:
        check(args.source_root, args.execute)
    else:
        with tempfile.TemporaryDirectory(prefix="agentscale-verl-") as temp:
            root = Path(temp)
            for name in FILES:
                url = f"https://raw.githubusercontent.com/verl-project/verl/{REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=15) as response:
                    data = response.read(1024 * 1024)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            check(root, args.execute)


if __name__ == "__main__":
    main()
