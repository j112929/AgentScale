"""verl 0.6.1 BaseTool extension; requires verl in the training environment."""

import asyncio
import json
import uuid

from verl.tools.base_tool import BaseTool  # type: ignore[import-not-found]
from verl.tools.schemas import ToolResponse  # type: ignore[import-not-found]

from agentscale.sandbox import CodeTask, DockerSandbox, SandboxLimits


class ContainerCodeTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self._sandbox = DockerSandbox(SandboxLimits(**config.get("limits", {})))
        self._max_instances = config.get("max_instances", 128)
        workers = config.get("workers", 2)
        if type(workers) is not int or workers < 1 or self._max_instances < 1:
            raise ValueError("positive worker and instance bounds required")
        self._slots = asyncio.Semaphore(workers)
        self._instances: dict[str, tuple[asyncio.Lock, float]] = {}

    async def create(self, instance_id=None, **kwargs):
        instance_id = instance_id or str(uuid.uuid4())
        if instance_id in self._instances or len(self._instances) >= self._max_instances:
            raise ValueError("duplicate instance or instance limit reached")
        self._instances[instance_id] = (asyncio.Lock(), 0.0)
        return instance_id, ToolResponse(text="Ephemeral container execution ready.")

    async def execute(self, instance_id, parameters, **kwargs):
        lock, _ = self._instances[instance_id]
        async with lock, self._slots:
            if instance_id not in self._instances:
                raise KeyError("tool instance released while execution was queued")
            task = CodeTask(
                uuid.uuid4().hex,
                parameters.get("files", {}),
                tuple(parameters.get("command", ("python", "-m", "unittest", "discover"))),
                parameters.get("timeout_s", 10),
            )
            running = asyncio.create_task(asyncio.to_thread(self._sandbox.execute, task))
            try:
                result = await asyncio.shield(running)
            except asyncio.CancelledError:
                # A cancelled coroutine must not free a slot while its container still runs.
                await running
                raise
            reward = float(result.status == "passed")
            self._instances[instance_id] = (lock, reward)
            response = ToolResponse(
                text=json.dumps(
                    {
                        "status": result.status,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.exit_code,
                    }
                )
            )
            return (
                response,
                reward,
                {
                    "sandbox_duration_s": result.duration_s,
                    "sandbox_timeout": float(result.status == "timeout"),
                },
            )

    async def calc_reward(self, instance_id, **kwargs):
        return self._instances[instance_id][1]

    async def release(self, instance_id, **kwargs):
        instance = self._instances.get(instance_id)
        if instance:
            async with instance[0]:
                self._instances.pop(instance_id, None)
