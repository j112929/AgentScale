# Local runtime integrations (M5.2–M5.4)

## Validation boundaries

| Capability | Implemented | Locally exercised |
| --- | --- | --- |
| Docker source/test execution and coding repair | Yes | Real Linux containers on Docker Desktop |
| CPU, memory/swap, PID, filesystem/network limits | Yes | Cgroup values, read-only writes, network denial, OOM |
| Timeout, bounded output, cleanup and queue scaling | Yes | Real containers plus injected failures |
| vLLM HTTP chat and per-model queue/cache metrics | Yes | Local HTTP fixture; upstream 0.11.0 source checked |
| Native verl coding tool | Yes | Genuine verl 0.6.1 BaseTool/schemas plus a real container |
| Versioned experience and training/sync transactions | Contract | Offline ordered/failure-injection tests |
| Actual GPU inference/training or GPU sharing | Pending | No GPU available |

## Docker backend

`CodeTask` contains a task ID, relative text source files, an argv sequence and a
bounded execution deadline. No task can choose image, mounts, Docker flags or host
credentials. `SandboxLimits` belongs to the trusted operator. The default image is
Python 3.12.12 slim-bookworm pinned by the digest in `DEFAULT_IMAGE`.

The backend verifies Linux CPU/memory/swap/PID support, resolves an immutable local
image ID, and uses `docker create/start/inspect/rm` without a shell. It never executes
the submitted command on the host. It uses no network, root privileges, devices,
host volumes or Docker socket inside the task. Root is read-only; `/work` and `/tmp`
are bounded tmpfs with noexec/nosuid/nodev. Python can interpret source on noexec tmpfs.
Memory and memory+swap limits are equal. PID and open-file counts are bounded.

Default limits: 1 CPU, 256 MiB memory, 64 processes, 64 MiB workspace, 16 MiB temporary
files and 32 KiB captured output per stream. Source files total at most 512 KiB, with
at most 128 files; argv totals at most 16 KiB. The bootstrap captures child output in
bounded container scratch storage; driver pipes also have bounded buffers. Timeout
kills the command group, and final cleanup removes the entire container. Cleanup
failure quarantines the backend. Task-provided results cannot override Docker's OOM
state. Exit code success alone is not an adversarially secure reward oracle.

`ContainerSandboxPool` implements `ResourceAdapter` with bounded admission, logical
worker slots, busy-worker scale-in deferral and queue-age telemetry. Each active task
owns a fresh container. Reservations use conservative workload CPU/GiB counts and
must cover each container's limits. Polling collects results and dispatches queued
work. Closing cancels queued tasks and waits for running tasks to finish/clean up;
results remain consumable. GPU reassignment is unsupported.

`solve()` runs a bounded inference → source replacement → isolated tests → feedback
loop. A problem specifies editable files and trusted test command. The client refuses
model changes to protected test files. The local example deliberately uses a scripted
fixture; the tests it executes are real. Supply a vLLM endpoint to use real inference.
See SECURITY.md before running hostile submissions on shared infrastructure.

## vLLM 0.11.0 HTTP adapter

Public interfaces inspected: [OpenAI server](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html),
[metrics](https://docs.vllm.ai/en/v0.11.0/usage/metrics.html) and
[V1 metric definitions](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py).
The adapter uses `/health`, `/version`, `/v1/chat/completions` and `/metrics`.
No vLLM installation is required in the control-plane package.

Request counts sum across matching `model_name` engine series. Cache occupancy uses
the maximum matching fraction; missing optional counters remain `None`. Required
missing queue metrics, duplicate series, negative/nonfinite values and fractional
request counts fail explicitly. Optional generation tokens are cumulative, not
throughput until measured over a time window. KV-cache occupancy is not GPU compute
utilization. Callers must check sample age before acting on telemetry.

HTTP operations have deadlines and a 2 MiB response bound; redirects are rejected.
`VLLM_API_KEY` is read by CLI/examples, never printed. Worker resizing raises an
unsupported-operation error. Sleep/wake APIs are not enabled by this client: a
queue-depth check does not fence other clients, and [vLLM sleep mode](https://docs.vllm.ai/en/v0.11.0/features/sleep_mode.html)
requires a valid runtime-owned safe point and CUDA validation.

## verl 0.6.1 coding tool

`configs/verl-tools.json` is JSON compatible with the upstream YAML tool-config loader.
It references `agentscale.integrations.verl_tool.ContainerCodeTool`, implementing the
native `create/execute/calc_reward/release` lifecycle and `ToolResponse` protocol.
Relevant sources: [BaseTool](https://github.com/verl-project/verl/blob/v0.6.1/verl/tools/base_tool.py)
and [registry](https://github.com/verl-project/verl/blob/v0.6.1/verl/tools/utils/tool_registry.py).

The plugin limits active instances and concurrent containers, serializes commands
within each instance and waits for an executing container even if its coroutine is
cancelled. Each tool call is ephemeral; callers must supply all required files again.
Reward is the latest command's exit success. The plugin must be installed on each
trusted worker that owns a Docker connection. Limits are per plugin instance; cluster
admission and aggregate resources must be enforced by the deployment/control plane.

The contract checker downloads only four files from immutable verl commit
`d62da4950573d7a4b7ef2362337952e7ab59e78d`. It loads genuine lightweight tool/schema/trace
modules, avoiding the unrelated torch-heavy package initializer. This verifies the
extension lifecycle, not a complete installed verl trainer or a jointly compatible
verl/vLLM/Ray GPU stack. The training environment must pin and validate that stack
separately using upstream installation guidance.

## Experience and training transactions

`ExperienceBuffer` bounds both queued and in-flight samples, reserves IDs, drops old
policy versions, rejects future versions and advances versions only on successful
commit. `TrainingCoordinator` orders rollout quiescence, inference sleep, training,
weight sync, wake and admission resume. Hooks are supplied by a verified runtime;
this project does not fabricate an actuator for a distributed GPU gang.

Before training starts, failures restore samples and attempt reversible compensation.
After an optimizer update might have started, errors latch the coordinator and keep
rollouts fenced rather than retrying a possibly executed update or resuming unknown
weights. There is no automatic retry/reset; operator reconciliation is required.
The coordinator is a single-driver, in-memory experimental contract. Durable recovery,
real GPU resource transfer and end-to-end training quality remain open.
