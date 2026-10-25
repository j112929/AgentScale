# Security policy

AgentScale 0.1 is experimental research software. Report vulnerabilities using the
repository's private vulnerability-reporting feature if enabled. If it is unavailable,
open an issue requesting a private reporting channel without publishing exploit details.
No private reporting address is promised by this repository.

The Docker sandbox uses Linux containers, non-root execution, dropped capabilities,
no new privileges, no network, no host bind mounts, read-only root, bounded tmpfs,
cgroup CPU/memory/PID limits and execution deadlines. Docker must enforce these
features; execution fails when required limit support is unavailable. The Docker
socket belongs to the trusted driver and is never passed into task containers.

Containers share a kernel with their Docker host/VM. This configuration is not a
security certification or a hostile multi-tenant isolation service. Operate it in
a dedicated disposable VM/worker for adversarial workloads. Use approved images.
A passed task exit code is not proof against reward hacking by malicious submissions.

Source payloads and output are bounded. Cleanup failure stops further work through
that backend until the operator reconciles its `org.agentscale.sandbox=true` containers.
Do not remove unrelated containers. vLLM keys come from caller configuration or an
environment variable; HTTP redirects are rejected and keys are not logged. Do not
expose vLLM development sleep/wake endpoints to untrusted clients.
