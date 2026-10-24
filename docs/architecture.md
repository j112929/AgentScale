# Architecture and scope

`models.py` defines unit-bearing capacity, workload, allocation, snapshot, plan, and adapter
contracts. `simulator.py` is a seeded event queue with four stages. `policies.py` contains
three interchangeable allocation planners. `adapters.py` contains an idempotent in-memory
adapter and non-mutating dry-run validation. `cli.py` runs scenarios and emits JSON.

**ADR-001: Simulator first.** Scheduling logic is developed without infrastructure costs;
real system behavior must eventually calibrate the simulator.

**ADR-002: Ray first.** Future integrations should prefer public Ray and verl interfaces,
then add Kubernetes/Kueue support; no upstream API compatibility is claimed today.

Next: workload shifts during a run, stage queue depth telemetry, safe-point controller,
trace replay, Prometheus, and verified external runtime adapters.
