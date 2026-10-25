import json
import subprocess
from dataclasses import asdict

import pytest

from agentscale.coding import EXAMPLE_PROBLEM, solve
from agentscale.sandbox import CodeTask, DockerSandbox, SandboxLimits
from agentscale.vllm_adapter import ChatResult

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def sandbox():
    return DockerSandbox()


def test_real_coding_agent_repairs_source_after_failed_feedback(sandbox):
    class ScriptedInference:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, max_tokens=1024):
            self.calls += 1
            if self.calls == 2:
                assert "failed" in messages[-1]["content"]
            op = "-" if self.calls == 1 else "+"
            return ChatResult(
                json.dumps({"files": {"solution.py": f"def add(a,b):\n    return a {op} b\n"}}),
                "scripted-test-fixture",
            )

    result = solve(EXAMPLE_PROBLEM, ScriptedInference(), sandbox)
    assert result.passed
    assert [r.status for r in result.attempts] == ["failed", "passed"]


def test_real_isolation_and_cgroup_limits(sandbox):
    code = """import os, json, pathlib, socket
try:
 pathlib.Path('/root-write-test').write_text('bad')
 readonly=False
except OSError:
 readonly=True
try:
 socket.create_connection(('1.1.1.1',80),timeout=.2)
 network=False
except OSError:
 network=True
print(json.dumps(dict(uid=os.getuid(), readonly=readonly, network=network,
 socket_exposed=pathlib.Path('/var/run/docker.sock').exists(),
 memory=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),
 pids=pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip(),
 cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip())))"""
    r = sandbox.execute(CodeTask("isolation", command=("python", "-c", code)))
    assert r.status == "passed", asdict(r)
    data = json.loads(r.stdout)
    assert data["uid"] == 65534 and data["readonly"] and data["network"]
    assert not data["socket_exposed"]
    assert int(data["memory"]) == 256 * 1024**2 and int(data["pids"]) == 64
    quota, period = map(int, data["cpu"].split())
    assert quota / period == 1


def test_real_timeout_and_container_cleanup(sandbox):
    r = sandbox.execute(
        CodeTask("timeout", command=("python", "-c", "while True: pass"), timeout_s=0.2)
    )
    assert r.status == "timeout"


def test_real_memory_limit():
    r = DockerSandbox(SandboxLimits(memory_mb=64)).execute(
        CodeTask("oom", command=("python", "-c", "x=bytearray(512*1024*1024)"), timeout_s=5)
    )
    assert r.status == "oom", asdict(r)


def test_real_bounded_output(sandbox):
    r = sandbox.execute(CodeTask("output", command=("python", "-c", 'print("x"*100000)')))
    assert r.status == "passed" and r.truncated
    assert len(r.stdout.encode()) <= sandbox.limits.output_bytes


def test_no_owned_containers_remain_after_tests(sandbox):
    r = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=org.agentscale.sandbox=true"],
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert not r.stdout.strip()
