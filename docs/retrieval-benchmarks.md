# Synthetic retrieval benchmark

The checked-in [raw report](../benchmarks/retrieval-synthetic.json) was generated with:

```bash
agentscale retrieval-benchmark --duration 30 --seeds 42 43 44 --output benchmarks/retrieval-synthetic.json
```

The 42 runs compare seven variants on steady and pressure profiles. Within each
profile and seed, every variant uses identical query arrival timestamps, features,
constraints, RL arrivals and fake-backend seed. Query traces are saved once per
profile/seed. Each trial records capacity, initial allocation, workload, objective,
controller settings, backend provenance and metrics. Re-running produces identical
finite JSON. The backend hash also includes the selected plan, so common queries
using the same plan have common service/quality/success draws.

All runs model 8 GPUs, 20 CPU cores and 48 GiB. Static RL reserves 4 inference GPUs,
4 training GPUs and eight sandbox workers (16 cores, 32 GiB), leaving four cores,
16 GiB and zero GPUs for retrieval. The real machine requires no GPU. RL work runs
in the existing four-stage simulator. Fixing the RL policy isolates retrieval-policy
differences; `retrieval-simulate --rl-policy global` separately exercises joint control.

The steady trace has rate 2 queries/s. Pressure changes rate from 1 to 6 to 2 queries/s
at thirds of the horizon; each phase begins with a query. Query lexicality alternates
0.9/0.1, difficulty is seeded in [0.2, 0.9], minimum quality is 0.7 for every third
query and 0.5 otherwise, minimum success probability is 0.4, task deadline is 3 seconds
and cost budget is 3 accounting units. These are constructed assumptions, not a
representative labeled dataset. Request, result and estimator details are in
[retrieval architecture](retrieval.md).

## Baselines and accounting

- Fixed BM25/vector/hybrid/hybrid-rerank use top-K 10 and CPU-only templates; the
  reranker uses 40 candidates. Every variant enforces the same admission constraints.
- Adaptive one-shot chooses from the same 12-candidate grid as closed-loop at a
  query's first eligible poll, retains that plan while pending and disables feedback.
- Adaptive closed-loop replans pending requests and calibrates on completed results.
- Adaptive no-queue retains feedback/replanning but hides retrieval and downstream
  queue delay from planning. Actual elapsed pending time, admission capacity and
  reported completion latency still apply.

All benchmark variants use one confirmation and a 0.1-second telemetry interval;
the library default of two confirmations is tested separately. Cooldown is 0.3 seconds
and relative benefit margin 0.03. The FIFO queue is bounded at 64 pending queries.
Head-of-line blocking is possible. Calibration and warm-up begin empty on every run;
the full 30-second horizon is included, without a post-horizon drain.

Metrics include submitted, admitted, completed, terminal rejection/expiry/failure
counts, pending/running at horizon, synthetic quality and success, modeled latency
p50/p95/p99, quality/deadline/cost misses, resource accounting, search calls, assumed
inference input tokens and strategy counts. Requests partition into completed,
terminal outcomes and censored pending/running jobs. Censored jobs remain explicit;
success fraction uses **all submitted requests** as its denominator, so they cannot
improve success by being omitted. Latency percentiles and quality means apply only
to completed jobs and must be read alongside the completion and censor counts.

Modeled retrieval CPU/GPU utilization divides busy peak-reserved resource-seconds
by total cluster capacity times horizon. It excludes RL utilization and is not a real
device measurement. Resource usage is clipped at the horizon; search/context fees
are charged at admission. Total cost includes unfinished admitted work, so dividing
by completed queries charges that overhead to completions. Zero-completion ratios
and latency/quality statistics are `null`, not invented zeros. GPU use is modeled zero
in this benchmark because RL already reserves all GPUs.

## Observed tradeoffs

Means over seeds 42/43/44, with synthetic/model units only:

| Profile | Variant | Completed | Synthetic successes / submitted | Cost units | Deadline misses |
| --- | --- | ---: | ---: | ---: | ---: |
| Steady | Fixed hybrid | 55.33 | 0.6793 | 7.15 | 0.00 |
| Steady | Adaptive one-shot | 45.67 | 0.6051 | 11.48 | 1.67 |
| Steady | Adaptive closed-loop | 50.67 | 0.7030 | 12.25 | 3.33 |
| Steady | Adaptive no-queue | 51.67 | 0.7263 | 12.30 | 4.33 |
| Pressure | Fixed hybrid | 84.00 | 0.6516 | 10.92 | 0.00 |
| Pressure | Adaptive one-shot | 68.67 | 0.5579 | 15.00 | 3.00 |
| Pressure | Adaptive closed-loop | 73.67 | 0.5952 | 15.04 | 3.00 |
| Pressure | Adaptive no-queue | 75.67 | 0.6172 | 15.93 | 5.67 |

Closed-loop completes more queries than one-shot here, but fixed hybrid completes
more than either adaptive policy at lower cost. Disabling queue awareness completes
more queries and obtains more synthetic successes in these trials, while missing more
deadlines. These are multi-objective tradeoffs, not evidence that adaptivity dominates
or queue awareness always improves completion. The objective weights favor expensive
reranking on many queries; they are not fitted to an independent evaluation set.

The summary includes sample standard deviation of completion counts. Three seeds
are insufficient to claim general significance; no percentage improvement, real RAG
quality gain, GPU performance gain or monetary saving is claimed. The synthetic
quality formula also drives the fake backend, so these experiments validate control
logic against known assumptions, not retrieval-quality generalization.

Real evaluation requires a licensed corpus/query set, relevance annotations,
Recall@K/nDCG@K, an independently scored end-to-end agent task, measured search and
inference cost, resource telemetry, pinned backend versions and paired repetitions.
Those are separate future acceptance gates, not completed phases in this PR.
