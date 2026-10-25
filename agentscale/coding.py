"""A bounded coding-agent loop: propose source replacements, run isolated tests, retry."""

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .sandbox import CodeResult, CodeTask, DockerSandbox
from .vllm_adapter import ChatResult


class InferenceClient(Protocol):
    def chat(self, messages: list[dict[str, str]], max_tokens: int = 1024) -> ChatResult: ...


@dataclass(frozen=True)
class CodingProblem:
    problem_id: str
    instruction: str
    files: dict[str, str]
    editable: tuple[str, ...]
    command: tuple[str, ...] = ("python", "-m", "unittest", "discover", "-v")


@dataclass(frozen=True)
class CodingRun:
    problem_id: str
    passed: bool
    attempts: tuple[CodeResult, ...]
    files: dict[str, str]
    policy_version: int


def solve(
    problem: CodingProblem,
    inference: InferenceClient,
    sandbox: DockerSandbox,
    *,
    max_attempts: int = 3,
    policy_version: int = 0,
) -> CodingRun:
    if not 1 <= max_attempts <= 10 or policy_version < 0:
        raise ValueError("attempts must be 1..10 and policy version nonnegative")
    if not problem.editable or any(p not in problem.files for p in problem.editable):
        raise ValueError("editable files must exist in the problem")
    files = dict(problem.files)
    messages = [
        {
            "role": "system",
            "content": 'Repair the supplied code. Return only JSON: {"files": {"path": "full content"}}. '
            "Only modify these files: " + ", ".join(problem.editable),
        },
        {"role": "user", "content": problem.instruction + "\n" + json.dumps(files)},
    ]
    attempts = []
    for index in range(max_attempts):
        reply = inference.chat(messages)
        proposal = json.loads(reply.text)
        updates = proposal.get("files")
        if (
            not isinstance(updates, dict)
            or not updates
            or any(
                path not in problem.editable or not isinstance(content, str)
                for path, content in updates.items()
            )
        ):
            raise ValueError("inference attempted to modify protected or invalid files")
        files.update(updates)
        result = sandbox.execute(CodeTask(f"{problem.problem_id}-{index}", files, problem.command))
        attempts.append(result)
        if result.status == "passed":
            return CodingRun(problem.problem_id, True, tuple(attempts), files, policy_version)
        messages.extend(
            [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": json.dumps(asdict(result))},
            ]
        )
    return CodingRun(problem.problem_id, False, tuple(attempts), files, policy_version)


EXAMPLE_PROBLEM = CodingProblem(
    "repair-add",
    "Fix add(a, b) so it returns the sum, preserving its function signature.",
    {
        "solution.py": "def add(a, b):\n    return a - b\n",
        "test_solution.py": "import unittest\nfrom solution import add\n"
        "class Tests(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
        "        self.assertEqual(add(-2, 3), 1)\n",
    },
    ("solution.py",),
)
