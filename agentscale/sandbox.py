"""CPU-only, ephemeral Docker execution with bounded I/O and explicit limits."""

from __future__ import annotations

import json
import math
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol, Self

from .adapters import SafePointUnavailable
from .models import Allocation, Capacity, Workload, validate

DEFAULT_IMAGE = "python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"

# Sent as argv to Python inside the container. User code never runs in the driver.
_BOOTSTRAP = r"""
import json, os, pathlib, signal, subprocess, sys, tempfile, time
request = json.load(sys.stdin)
os.chdir('/work')
for name, content in request['files'].items():
    path = pathlib.Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
started = time.monotonic()
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    proc = subprocess.Popen(request['command'], stdout=out, stderr=err, start_new_session=True,
                            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/work',
                                 'TMPDIR': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1'})
    timed_out = False
    try:
        proc.wait(timeout=request['timeout_s'])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    output = {}
    truncated = False
    for name, stream in [('stdout', out), ('stderr', err)]:
        size = stream.tell()
        stream.seek(0)
        output[name] = stream.read(request['output_bytes']).decode('utf-8', errors='replace')
        truncated = truncated or size > request['output_bytes']
    print(json.dumps(dict(exit_code=proc.returncode, timed_out=timed_out,
                         duration_s=time.monotonic()-started, truncated=truncated, **output)))
"""


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    files: dict[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ("python", "-m", "unittest", "discover", "-v")
    timeout_s: float = 10

    def __post_init__(self) -> None:
        if not self.task_id or not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 300:
            raise ValueError("task requires an id and timeout in (0, 300] seconds")
        if (
            not isinstance(self.command, (tuple, list))
            or not self.command
            or len(self.command) > 128
            or any(not isinstance(a, str) or "\0" in a for a in self.command)
        ):
            raise ValueError("command must be a nonempty argv sequence without NUL bytes")
        if sum(len(a.encode()) for a in self.command) > 16 * 1024:
            raise ValueError("command argv exceeds 16 KiB")
        if len(self.files) > 128:
            raise ValueError("at most 128 source files per task")
        total = 0
        for name, content in self.files.items():
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "." == name
                or "\\" in name
                or "\0" in name
                or not isinstance(content, str)
            ):
                raise ValueError("source paths must be relative and cannot traverse directories")
            total += len(name.encode()) + len(content.encode())
        if total > 512 * 1024:
            raise ValueError("source payload exceeds 512 KiB")


@dataclass(frozen=True)
class SandboxLimits:
    image: str = DEFAULT_IMAGE
    cpus: float = 1
    memory_mb: int = 256
    pids: int = 64
    work_mb: int = 64
    output_bytes: int = 32 * 1024

    def __post_init__(self) -> None:
        if not self.image or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("image and positive finite CPU limit required")
        if any(
            type(n) is not int or n <= 0
            for n in (self.memory_mb, self.pids, self.work_mb, self.output_bytes)
        ):
            raise ValueError("memory, PID, scratch and output limits must be positive integers")
        if self.output_bytes > 1024 * 1024:
            raise ValueError("output cap cannot exceed 1 MiB per stream")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes = b""
    truncated: bool = False


class CommandRunner(Protocol):
    def run(
        self, args: list[str], *, timeout: float, input_bytes: bytes | None = None
    ) -> CommandResult: ...


class DockerCLI:
    """Bound both pipe buffers, including output written outside the task harness."""

    def run(
        self, args: list[str], *, timeout: float, input_bytes: bytes | None = None
    ) -> CommandResult:
        proc = subprocess.Popen(
            ["docker", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        buffers = [bytearray(), bytearray()]
        overflow = threading.Event()

        def read(stream, target):
            while data := stream.read(8192):
                room = max(0, 3 * 1024**2 - len(target))
                target.extend(data[:room])
                if len(data) > room:
                    overflow.set()

        def write():
            try:
                assert proc.stdin is not None
                if input_bytes:
                    proc.stdin.write(input_bytes)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        threads = [
            threading.Thread(target=read, args=(proc.stdout, buffers[0]), daemon=True),
            threading.Thread(target=read, args=(proc.stderr, buffers[1]), daemon=True),
            threading.Thread(target=write, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise TimeoutError("Docker operation exceeded its deadline") from None
        finally:
            for thread in threads:
                thread.join(timeout=1)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        return CommandResult(
            proc.returncode, bytes(buffers[0]), bytes(buffers[1]), overflow.is_set()
        )


@dataclass(frozen=True)
class CodeResult:
    task_id: str
    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0
    truncated: bool = False
    image_id: str = ""


class SandboxCleanupError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, limits: SandboxLimits | None = None, runner: CommandRunner | None = None):
        self.limits = limits or SandboxLimits()
        self.runner = runner or DockerCLI()
        self._image_id = ""
        self._prepare_lock = threading.Lock()
        self._faulted = False

    def prepare(self) -> str:
        with self._prepare_lock:
            if self._faulted:
                raise SandboxCleanupError(
                    "cleanup previously failed; reconcile owned containers first"
                )
            if self._image_id:
                return self._image_id
            info = self.runner.run(["info", "--format", "{{json .}}"], timeout=10)
            if info.returncode:
                raise RuntimeError(
                    "Docker daemon unavailable: " + info.stderr.decode(errors="replace")[:512]
                )
            data = json.loads(info.stdout)
            if data.get("OSType") != "linux" or not all(
                data.get(k) for k in ("MemoryLimit", "SwapLimit", "CpuCfsQuota", "PidsLimit")
            ):
                raise RuntimeError("Linux Docker with CPU, memory/swap and PID limits is required")
            image = self.runner.run(
                ["image", "inspect", self.limits.image, "--format", "{{.Id}}"], timeout=10
            )
            image_id = image.stdout.decode().strip()
            if image.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                raise RuntimeError(
                    "pull the configured image before execution: " + self.limits.image
                )
            self._image_id = image_id
            return image_id

    def execute(self, task: CodeTask) -> CodeResult:
        image_id = self.prepare()
        name = "agentscale-" + uuid.uuid4().hex
        cfg = self.limits
        created = False
        try:
            args = [
                "create",
                "--name",
                name,
                "--label",
                "org.agentscale.sandbox=true",
                "--pull=never",
                "--interactive",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user=65534:65534",
                "--cpus",
                str(cfg.cpus),
                "--memory",
                f"{cfg.memory_mb}m",
                "--memory-swap",
                f"{cfg.memory_mb}m",
                "--pids-limit",
                str(cfg.pids),
                "--ulimit",
                "nofile=128:128",
                "--log-driver=none",
                "--tmpfs",
                f"/work:rw,noexec,nosuid,nodev,size={cfg.work_mb}m,mode=1777",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
                "--workdir=/work",
                image_id,
                "python",
                "-I",
                "-c",
                _BOOTSTRAP,
            ]
            # Always clean by our unpredictable name, including ambiguous create timeouts.
            created = True
            result = self.runner.run(args, timeout=15)
            if result.returncode:
                raise RuntimeError(
                    "container create failed: " + result.stderr.decode(errors="replace")[:512]
                )
            payload = json.dumps(
                {
                    "files": task.files,
                    "command": task.command,
                    "timeout_s": task.timeout_s,
                    "output_bytes": cfg.output_bytes,
                }
            ).encode()
            try:
                execution = self.runner.run(
                    ["start", "--attach", "--interactive", name],
                    timeout=task.timeout_s + 5,
                    input_bytes=payload,
                )
            except TimeoutError:
                return CodeResult(task.task_id, "timeout", image_id=image_id)
            state_result = self.runner.run(
                ["inspect", "--format", "{{json .State}}", name], timeout=5
            )
            if state_result.returncode:
                raise RuntimeError("cannot inspect sandbox exit state")
            state = json.loads(state_result.stdout)
            if state.get("OOMKilled"):
                return CodeResult(task.task_id, "oom", exit_code=137, image_id=image_id)
            if execution.truncated:
                return CodeResult(task.task_id, "output_limit", truncated=True, image_id=image_id)
            try:
                output = json.loads(execution.stdout)
                if not isinstance(output, dict) or type(output.get("exit_code")) is not int:
                    raise ValueError("invalid result")
            except (ValueError, TypeError):
                return CodeResult(
                    task.task_id,
                    "runtime_error",
                    exit_code=state.get("ExitCode"),
                    stderr=execution.stderr.decode(errors="replace")[: cfg.output_bytes],
                    image_id=image_id,
                )
            status = (
                "timeout"
                if output.get("timed_out")
                else "passed"
                if output["exit_code"] == 0
                else "failed"
            )
            return CodeResult(
                task.task_id,
                status,
                output["exit_code"],
                str(output.get("stdout", ""))[: cfg.output_bytes],
                str(output.get("stderr", ""))[: cfg.output_bytes],
                float(output.get("duration_s", 0)),
                bool(output.get("truncated")),
                image_id,
            )
        finally:
            if created:
                try:
                    cleanup = self.runner.run(["rm", "--force", name], timeout=10)
                except (OSError, TimeoutError) as exc:
                    self._faulted = True
                    raise SandboxCleanupError(f"cannot confirm cleanup for {name}") from exc
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    self._faulted = True
                    raise SandboxCleanupError(f"failed to remove owned container {name}")


class ContainerSandboxPool:
    """ResourceAdapter backed by bounded task admission and real per-task containers."""

    def __init__(
        self,
        allocation: Allocation,
        capacity: Capacity,
        workload: Workload,
        sandbox: DockerSandbox,
        max_queue: int = 128,
    ):
        errors = validate(allocation, capacity, workload)
        if errors:
            raise ValueError("; ".join(errors))
        if type(max_queue) is not int or max_queue < 1:
            raise ValueError("max_queue must be positive")
        if (
            sandbox.limits.cpus > workload.sandbox_cpu
            or sandbox.limits.memory_mb > workload.sandbox_memory_gb * 1024
        ):
            raise ValueError("container limits exceed reserved worker resources")
        self.allocation, self.capacity, self.workload = allocation, capacity, workload
        self.sandbox, self.max_queue = sandbox, max_queue
        max_workers = min(
            capacity.cpu_cores // workload.sandbox_cpu,
            capacity.memory_gb // workload.sandbox_memory_gb,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue: deque[tuple[CodeTask, float]] = deque()
        self._running: dict[Future[CodeResult], str] = {}
        self._results: dict[str, CodeResult] = {}
        self._ids: set[str] = set()
        self._closed = self._faulted = False
        self.completed = self.failed = 0

    def observe(self) -> Allocation:
        if self._closed or self._faulted:
            raise RuntimeError("pool closed or requires cleanup reconciliation")
        return self.allocation

    def apply(self, allocation: Allocation) -> None:
        self.observe()
        errors = validate(allocation, self.capacity, self.workload)
        if errors:
            raise ValueError("; ".join(errors))
        if (allocation.inference_gpus, allocation.training_gpus) != (
            self.allocation.inference_gpus,
            self.allocation.training_gpus,
        ):
            raise ValueError("container pool cannot reassign GPUs")
        if allocation.sandbox_workers < len(self._running):
            raise SafePointUnavailable("wait for running containers before scale-in")
        self.allocation = allocation
        self._dispatch()

    def submit(self, task: CodeTask) -> str:
        self.observe()
        if task.task_id in self._ids:
            raise ValueError("duplicate task id")
        if len(self._queue) >= self.max_queue:
            raise OverflowError("container task queue is full")
        self._ids.add(task.task_id)
        self._queue.append((task, time.monotonic()))
        self._dispatch()
        return task.task_id

    def _dispatch(self) -> None:
        slots = min(self.allocation.sandbox_workers, self.allocation.rollout_concurrency)
        while self._queue and len(self._running) < slots:
            task, _ = self._queue.popleft()
            self._running[self._executor.submit(self.sandbox.execute, task)] = task.task_id

    def poll(self) -> dict[str, float | int]:
        self.observe()
        self._collect()
        self._dispatch()
        return self.telemetry()

    def _collect(self) -> None:
        for future in list(self._running):
            if not future.done():
                continue
            task_id = self._running.pop(future)
            try:
                result = future.result()
            except SandboxCleanupError:
                self._faulted = True
                raise
            except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                result = CodeResult(task_id, "runtime_error", stderr=str(exc))
            self._results[task_id] = result
            if result.status == "passed":
                self.completed += 1
            else:
                self.failed += 1

    def telemetry(self) -> dict[str, float | int]:
        return {
            "workers": self.allocation.sandbox_workers,
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": self.completed,
            "failed": self.failed,
            "oldest_queue_s": time.monotonic() - self._queue[0][1] if self._queue else 0.0,
        }

    def pop_result(self, task_id: str) -> CodeResult | None:
        result = self._results.pop(task_id, None)
        if result is not None:
            self._ids.remove(task_id)
        return result

    def close(self) -> None:
        if not self._closed:
            while self._queue:
                task, _ = self._queue.popleft()
                self._results[task.task_id] = CodeResult(task.task_id, "cancelled")
                self.failed += 1
            try:
                self._executor.shutdown(wait=True)
                self._collect()
            finally:
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
