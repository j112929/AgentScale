# Synthetic benchmark methodology and results

Run from the repository root:

```bash
agentscale benchmark --duration 600 --seeds 42 43 44 --output benchmarks/closed-loop.json
```

The published file contains 75 runs: five profiles × five policy variants × three
paired seeds. Every run uses a capacity budget of 8 GPUs, 64 CPU cores and 128 GiB.
Arrival timestamps are generated independently of scheduling; sandbox duration and
failure draws are keyed by arrival ID and step. Admission drops can still make the
admitted populations differ across policies. All runs include warm-up, drain time,
switch downtime and unfinished work at the cutoff. There is no excluded warm-up
period or steady-state-only measurement. No confidence interval is claimed from
just three seeds; the table reports sample standard deviations.

## Profiles

All profiles use arrival rate 1/s, three tool steps per trajectory, batch size 4,
2 CPU cores and 4 GiB per sandbox worker, zero task failures and maximum policy
staleness of two versions. Service times below are seconds per operation, except
training which is GPU-seconds per sample. Sandbox durations are seeded lognormal
draws around the listed mean.

| Profile | Inference | Sandbox | Training |
| --- | ---: | ---: | ---: |
| Balanced | 1 | 2 | 4 |
| Inference | 5 | 0.5 | 4 |
| Sandbox | 0.5 | 8 | 4 |
| Training | 0.5 | 0.5 | 24 |
| Dynamic | Inference profile to t=200 | Sandbox profile to t=400 | Training profile to t=600 |

## Comparisons

- `static`: fixed 4 inference GPUs, 4 training GPUs, 8 sandbox workers and 32 rollout slots.
- `independent`: periodic queue-threshold scaling with the same controller guards.
- `global`: periodic joint capacity search with the default controller configuration.
- `global-one-shot`: global planning at initialization, then fixed allocation through all shifts.
- `global-no-stability`: periodic global search with confirmations=1, cooldown=0 and
  minimum relative gain=0. It retains physical switch downtime, net-benefit checks,
  performance observation and rollback. This ablates three stability guards, not every guard.

All settings, workload shifts, seed-level metrics, allocation histories, action
decisions and status counts are in the raw JSON. For full per-poll decisions and
event traces, use `agentscale simulate --scenario dynamic --duration 600 --trace`.
The synthetic cost is 10 units/s per allocated GPU plus 0.25 units/s per sandbox
worker, charged even while idle or switching. It is not a cloud price. Capacity
budgets are identical, while resource cost can differ by allocation.

## Results

| Profile | Variant | Mean trained samples/s | Sample std dev | Mean changes | Mean resource cost |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced | static | 0.62000 | 0.01764 | 0.00 | 49200.00 |
| balanced | independent | 0.62000 | 0.01764 | 0.00 | 49200.00 |
| balanced | global | 0.62000 | 0.01764 | 0.00 | 49200.00 |
| balanced | global-one-shot | 0.62000 | 0.01764 | 0.00 | 49200.00 |
| balanced | global-no-stability | 0.62000 | 0.01764 | 0.00 | 49200.00 |
| inference | static | 0.08222 | 0.00385 | 0.00 | 49200.00 |
| inference | independent | 0.08222 | 0.00385 | 0.00 | 49200.00 |
| inference | global | 0.12000 | 0.00000 | 1.00 | 49200.00 |
| inference | global-one-shot | 0.12000 | 0.00000 | 0.00 | 49200.00 |
| inference | global-no-stability | 0.12000 | 0.00000 | 1.00 | 49200.00 |
| sandbox | static | 0.10222 | 0.00385 | 0.00 | 49200.00 |
| sandbox | independent | 0.14222 | 0.02341 | 11.00 | 50167.50 |
| sandbox | global | 0.32889 | 0.00770 | 1.00 | 51513.33 |
| sandbox | global-one-shot | 0.33778 | 0.01018 | 0.00 | 51600.00 |
| sandbox | global-no-stability | 0.33556 | 0.01018 | 1.00 | 51580.00 |
| training | static | 0.16000 | 0.00000 | 0.00 | 49200.00 |
| training | independent | 0.16000 | 0.00000 | 0.00 | 49200.00 |
| training | global | 0.18000 | 0.00667 | 1.67 | 49200.00 |
| training | global-one-shot | 0.18445 | 0.00385 | 0.00 | 49200.00 |
| training | global-no-stability | 0.18000 | 0.00000 | 1.00 | 49200.00 |
| dynamic | static | 0.11333 | 0.00000 | 0.00 | 49200.00 |
| dynamic | independent | 0.11778 | 0.00770 | 4.67 | 49558.33 |
| dynamic | global | 0.20000 | 0.00667 | 3.00 | 50706.67 |
| dynamic | global-one-shot | 0.10667 | 0.00000 | 0.00 | 49200.00 |
| dynamic | global-no-stability | 0.20444 | 0.00770 | 3.00 | 50780.00 |

## Interpretation and negative results

The dynamic global controller averages 0.20000 trained samples/s, compared with
0.11333 for static, 0.11778 for independent and 0.10667 for initial-only global.
Its mean synthetic resource cost is 50706.67 versus 49200.00 for static. This
demonstrates adaptation in a constructed capacity model, not real RL efficiency.

Balanced throughput is identical across variants. Initial-only global performs
better than closed-loop global in the fixed sandbox profile because it starts at
a favorable allocation and pays no later transition cost. The no-stability variant
is slightly faster on the dynamic profile; these smooth shifts do not establish
a throughput advantage for stability guards. Anti-oscillation, cooldown and rollback
are tested separately with alternating proposals and injected faults.

Policy staleness, finite batches and startup effects reduce realized training
goodput below the planner's capacity upper bound. Completed trajectories/s and
fresh trained samples/s are separate metrics. This experiment does not measure
reward quality, convergence, real GPU utilization or production operating costs.

## Runtime validation and remaining M6 work

`benchmarks/ray-demo.jsonl` records a real local Ray 2.50.1 demo with eight trusted
synthetic tasks, 1→2 Actor scale-out and zero failed tasks. It is a functional
integration check, not an RL performance benchmark. Four real integration tests
also cover idle scale-in, task timeout and Actor crash/recovery.

Follow-up work includes calibrated inference/training telemetry, isolated coding
workloads, heavy-tail/cold-start sweeps, larger seed sets and confidence intervals,
real cost accounting, reward guardrails and upstream verl comparisons. The full M6
research validation milestone remains open. The original root `benchmark.json` is
an older MVP artifact and must not be mixed with this revised simulator's results.
