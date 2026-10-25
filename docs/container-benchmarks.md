# Measured local container benchmark

Run `agentscale container-benchmark --output container-results.json` after pulling
the pinned image. The published `benchmarks/container-local.json` records 9 trials,
3 policies × seeds 42/43/44, with 6 ephemeral containers per trial. All 54 tasks passed.

The same 2-CPU / 2-GiB logical capacity budget is available to each policy. Each
container is limited to 1 CPU / 256 MiB. Static retains one slot; independent and
global start at one and can grow to two. Policy order is shuffled per seed.
Container creation, execution, cleanup and queue waiting are included. Image pull
and preflight are excluded. Task sleep draws are paired across policies.

| Policy | Mean completed tasks/s | Sample std dev | Mean p95 end-to-end latency (s) |
| --- | ---: | ---: | ---: |
| static | 2.842 | 0.031 | 2.021 |
| independent | 5.334 | 0.116 | 1.118 |
| global | 5.363 | 0.155 | 1.105 |

This exercises real CPU containers and control-plane scaling. Inference demand
is configured synthetically and there is no optimizer/trainer. Metrics are **task
completions/s**, not trained samples/s. This result establishes no GPU or learning
benefit. Independent and global scaling are nearly equal here; the workload does
not distinguish cross-workload scheduling from ordinary CPU pool scaling.

The `verl-native` baseline is explicitly unavailable. Three seeds and short trials
are sensitive to startup and machine noise. No confidence interval or broad speedup
claim is made from this data. Use more tasks/seeds and a dedicated runtime for
performance research.

## Full RL comparison tooling

`agentscale compare-rl --input records.json --output comparison.json` accepts a
list of measured RL records with these fields:

```json
{"backend":"measured-rl","policy":"static","seed":42,"duration_s":600,
 "fresh_trained_samples":100,"hardware_id":"your-cluster",
 "capacity":{"gpus":2},"model_revision":"immutable-model-revision",
 "dataset_sha256":"dataset-content-hash","runtime_versions":{"verl":"0.6.1"},
 "training_config":{"batch_size":4,"warmup_s":30}}
```

Include at least two paired static seeds and matching seeds for every policy.
Hardware, capacity, model revision, dataset hash, runtime versions and training
configuration must match. The report computes fresh-training goodput and paired
bootstrap 95% percentile intervals using 2000 resamples and fixed resampling seed 0.
Missing `independent`, `global` or `verl-native` baselines are listed explicitly.
The tool checks record consistency; it does not certify externally supplied data.
Actual verl-native runs, model quality and GPU utilization remain pending.

The two GPU entries in the planner's `Capacity` object are model placeholders only;
Docker receives no GPU request and no physical GPU was used for this benchmark.
