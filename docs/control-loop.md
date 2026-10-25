# M4.5: dynamic closed-loop scheduling

`DynamicController.tick(snapshot)` is a deterministic clock-driven reconcile step.
A runtime can call `poll(provider)` through the `TelemetryProvider.collect()` protocol.
The caller owns the timer; the simulator schedules ticks as events and the Ray
example uses a monotonic clock. The controller never sleeps.

## Telemetry and decisions

`Snapshot` carries allocation, capacity, service-demand estimates, stage queue depths,
a sample timestamp and an optional measured trained-samples/s rate. That rate must
cover the preceding telemetry interval in the same clock domain as `time_s`.
Missing rates disable measured-performance comparisons; capacity planning still works.
Future timestamps, old samples, nonfinite rates, negative queues, invalid demands and
allocation drift prevent actuation. Decisions retain the input allocation, reason,
time and an increasing successful-operation generation. Runtime allocation histories
record the actual result of each transition. All controller state is in memory.

## Default guards

| Setting | Default | Meaning |
| --- | ---: | --- |
| `interval_s` | 5 | Minimum time between accepted polls |
| `confirmations` | 2 | Consecutive identical, beneficial proposals |
| `min_relative_gain` | 0.05 | Require strictly more than 5% predicted throughput gain |
| `cooldown_s` | 15 | Minimum dwell time after apply or rollback |
| `switch_s_per_unit` | 0.1 | Estimated transition downtime per normalized change unit |
| `benefit_horizon_s` | 60 | Benefit must repay transition downtime within this horizon |
| `telemetry_max_age_s` | 10 | Maximum accepted sample age |
| `observation_window_s` | 30 | Post-change performance evaluation window |
| `rollback_relative_drop` | 0.25 | Roll back a sustained measured drop greater than 25% |

A normalized change unit is the sum of changed inference GPUs, changed training GPUs,
changed sandbox workers / 4, and changed rollout concurrency / 16. The controller
recomputes throughput from the proposed allocation; it does not trust a policy's
self-reported benefit. Net benefit is `new_rate * max(0, horizon - switch_seconds)
- old_rate * horizon`. Nonpositive benefit is rejected. The planner separately
penalizes resource cost and switching. This first controller deliberately rejects
cost-only plans with no throughput gain; it is not yet a cost-minimizing autoscaler.

## Execution and rollback

1. Validate the snapshot, observe the adapter, and reject stale or mismatched state.
2. Finish any pending compensation, then evaluate a previous performance trial.
3. Replan, validate capacity, apply benefit guards and require stable proposals.
4. In dry-run mode, report the eligible proposal without mutation.
5. Apply through the adapter and verify that observed allocation matches the target.
6. On an ordinary failure, compensate to the pre-apply allocation and verify it.

`SafePointUnavailable` means defer and retry at a later poll. It must be raised
before making a partial change. This applies to rollback as well as forward actions.
Other failures attempt compensation. A failed compensation latches `faulted=True`;
subsequent ticks are blocked. An operator must inspect/reconcile the runtime before
constructing a fresh controller. There is no automatic reset that could conceal failure.

A measured-performance trial waits for the full observation window. It uses the
average of recent baseline rates only after the workload has been stable for a
full window, and integrates post-change rates over the evaluation window. A known
workload shift invalidates the comparison. Zero or insufficient baseline data and incomplete observation windows cannot
trigger performance rollback. This is an operational heuristic, not causal proof
that the allocation caused a regression. While a trial is pending, new plans wait.

## Simulator actuation

The simulator stops dispatch when a transition requests a safe point, drains
already running inference, sandbox and training operations, applies at the next
eligible poll, then charges switching downtime. New arrivals still encounter the
rollout admission bound. Queued work is preserved; in-flight operations use their
original service duration and batch size. Stage slots are never resized underneath
running work. This conservative full-pipeline drain can delay changes substantially.

Telemetry is captured for every poll; `--trace` exports it plus pipeline events.
Output includes allocation history, decision audit, dropped admissions, stale samples,
resource cost, drain time and switch downtime. Resource cost accrues during idle,
draining and switching periods. Actor startup latency and GPU migration behavior
are not calibrated against real infrastructure.

## Limits

The simulator supplies known demand profiles rather than estimating them from a real
inference/training runtime. The rollback journal and generation are not persisted;
there is no distributed leader election or crash-recovery transaction log. Staleness
is checked before starting training; a batch's admitted versions remain fixed during
that batch. Prometheus, measured service-demand estimation and production reconciliation
remain follow-up work.
