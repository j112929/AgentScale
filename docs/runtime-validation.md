# Runtime validation record

Validated locally with Python 3.12.12, macOS arm64, Docker Desktop engine 29.8.0
(Linux cgroup v2), pinned Python 3.12.12 container and Ray 2.50.1.

- Offline/controller/HTTP suite: 112 passed, 10 real-runtime tests deselected.
- Real integrations: 6 Docker tests and 4 Ray tests passed.
- Genuine pinned verl 0.6.1 BaseTool/schema/trace lifecycle plus real Docker execution:
  passed; all four upstream source hashes verified. Full trainer was not loaded.
- Coding repair demo: first tests failed, second source replacement passed in a fresh
  real container. Inference was explicitly a scripted fixture.
- Measured container benchmarks: 9 trials, 54 task containers, all passed.
- Ruff passed across package/tests/examples/tools; Mypy passed across 17 source files.
- Source/wheel distributions build and pass Twine metadata checks. See release artifacts
  and their SHA256SUMS for exact hashes; reproducibility is verified by two builds.

Published records: `benchmarks/coding-demo.json`, `benchmarks/container-local.json`
and `benchmarks/verl-contract.json`. The prior synthetic results are retained with
separate backend provenance. Local GitHub CLI authentication is not required for
building or testing and is not bundled into artifacts.

GPU inference, optimizer updates, colocated GPU transfer, joint upstream compatibility,
learning-quality evaluation and verl-native RL comparison were not run. Those gates
remain pending GPU infrastructure. This release candidate must retain that limitation.
