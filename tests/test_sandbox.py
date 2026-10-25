import json
import threading
import time
from dataclasses import replace

import pytest

from agentscale.adapters import SafePointUnavailable
from agentscale.models import Allocation, Capacity, Workload
from agentscale.sandbox import (
    CodeResult,
    CodeTask,
    CommandResult,
    ContainerSandboxPool,
    DockerSandbox,
    SandboxCleanupError,
    SandboxLimits,
)


class FakeDocker:
    def __init__(self):
        self.calls = []
        self.cleanup_failure = False
        self.start_timeout = False
        self.oom = False

    def run(self, args, *, timeout, input_bytes=None):
        self.calls.append((args, timeout, input_bytes))
        if args[0] == "info":
            return CommandResult(
                0,
                json.dumps(
                    {
                        "OSType": "linux",
                        "MemoryLimit": True,
                        "SwapLimit": True,
                        "CpuCfsQuota": True,
                        "PidsLimit": True,
                    }
                ).encode(),
            )
        if args[:2] == ["image", "inspect"]:
            return CommandResult(0, ("sha256:" + "a" * 64).encode())
        if args[0] == "create":
            return CommandResult(0, b"container-id")
        if args[0] == "start":
            if self.start_timeout:
                raise TimeoutError("injected timeout")
            return CommandResult(
                0,
                json.dumps(
                    {
                        "exit_code": 0,
                        "timed_out": False,
                        "stdout": "hello",
                        "stderr": "",
                        "duration_s": 0.1,
                        "truncated": False,
                    }
                ).encode(),
            )
        if args[0] == "inspect":
            return CommandResult(0, json.dumps({"OOMKilled": self.oom, "ExitCode": 0}).encode())
        if args[0] == "rm":
            if self.cleanup_failure:
                raise TimeoutError("cleanup unavailable")
            return CommandResult(0, b"container-id")
        raise AssertionError(args)


def test_docker_flags_no_host_mounts_or_privileges_and_cleanup():
    runner = FakeDocker()
    sandbox = DockerSandbox(runner=runner)
    result = sandbox.execute(CodeTask("code", {"solution.py": "print(1)"}))
    assert result.status == "passed" and result.stdout == "hello"
    create = next(args for args, _, _ in runner.calls if args[0] == "create")
    for flag in (
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--pull=never",
    ):
        assert flag in create
    assert not any(a in create for a in ("--privileged", "--volume", "-v", "--device", "--gpus"))
    assert create[create.index("--memory") + 1] == create[create.index("--memory-swap") + 1]
    assert runner.calls[-1][0][0] == "rm"
    assert create[-4] == "python"


def test_timeout_and_oom_have_explicit_results_and_cleanup():
    runner = FakeDocker()
    runner.start_timeout = True
    assert DockerSandbox(runner=runner).execute(CodeTask("timeout")).status == "timeout"
    assert runner.calls[-1][0][0] == "rm"
    runner = FakeDocker()
    runner.oom = True
    assert DockerSandbox(runner=runner).execute(CodeTask("oom")).status == "oom"


def test_ambiguous_cleanup_quarantines_sandbox():
    runner = FakeDocker()
    runner.cleanup_failure = True
    sandbox = DockerSandbox(runner=runner)
    with pytest.raises(SandboxCleanupError):
        sandbox.execute(CodeTask("leak"))
    calls = len(runner.calls)
    with pytest.raises(SandboxCleanupError):
        sandbox.execute(CodeTask("blocked"))
    assert len(runner.calls) == calls


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../escape", "x/../../escape", ".", "a\\b", "a\0b"]
)
def test_source_path_validation(path):
    with pytest.raises(ValueError):
        CodeTask("bad", {path: "text"})


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), 301])
def test_task_deadline_validation(timeout):
    with pytest.raises(ValueError):
        CodeTask("bad", timeout_s=timeout)


def test_source_payload_bound():
    with pytest.raises(ValueError):
        CodeTask("big", {"big.py": "a" * (512 * 1024 + 1)})


def test_pool_queue_scaling_and_results():
    release = threading.Event()

    class BlockingSandbox:
        limits = SandboxLimits()

        def execute(self, task):
            release.wait(timeout=5)
            return CodeResult(task.task_id, "passed", 0)

    alloc = Allocation(1, 1, 1, 8)
    with ContainerSandboxPool(
        alloc,
        Capacity(2, 2, 2),
        Workload(sandbox_cpu=1, sandbox_memory_gb=1),
        BlockingSandbox(),
        max_queue=1,
    ) as pool:
        pool.submit(CodeTask("a"))
        pool.submit(CodeTask("b"))
        with pytest.raises(OverflowError):
            pool.submit(CodeTask("c"))
        with pytest.raises(ValueError):
            pool.submit(CodeTask("a"))
        pool.apply(replace(alloc, sandbox_workers=2))
        assert pool.telemetry()["running"] == 2
        with pytest.raises(SafePointUnavailable):
            pool.apply(alloc)
        release.set()
        deadline = time.monotonic() + 5
        while pool.poll()["completed"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.telemetry()["completed"] == 2
        assert pool.pop_result("a").status == "passed"
        pool.apply(alloc)
        assert pool.observe() == alloc


def test_close_cancels_queued_work_and_preserves_running_result():
    release = threading.Event()

    class Sandbox:
        limits = SandboxLimits()

        def execute(self, task):
            release.wait(timeout=5)
            return CodeResult(task.task_id, "passed", 0)

    pool = ContainerSandboxPool(
        Allocation(1, 1, 1, 8),
        Capacity(2, 2, 2),
        Workload(sandbox_cpu=1, sandbox_memory_gb=1),
        Sandbox(),
    )
    pool.submit(CodeTask("running"))
    pool.submit(CodeTask("queued"))
    release.set()
    pool.close()
    assert pool.pop_result("queued").status == "cancelled"
    assert pool.pop_result("running").status == "passed"
