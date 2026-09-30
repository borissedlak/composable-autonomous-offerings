# Experiment Design: Arrival Processes and Load

Draft. All parameter values below are **assumptions, chosen freely**; each comes with the reason for the choice so it
can be defended or changed. Nothing in this file is a result.

## Motivation

`why_poisson_is_loose.ipynb` shows that the tightness of the SNC latency bound depends mainly on how *bursty* the
arrivals are, not on the GP or the composition:

- With smooth (periodic) arrivals, the bound is close to the true quantile.
- With Poisson arrivals, it is loose, and the looseness grows along a chain because every node bound assumes the
  external arrival process.
- With arrivals burstier than the bound assumes, the bound can be *violated*.

So the evaluation should cover a range of burstiness with realistic scenarios, at several loads, and check both
**coverage** (does the guarantee hold?) and **tightness** (is it useful?).

## Notation

| symbol | meaning | unit |
|---|---|---|
| $\lambda$ | mean arrival rate into the pipeline | items/s |
| $\mu_b$ | mean execution time of the bottleneck (slowest) service at the chosen configuration | s |
| $\rho = \lambda \mu_b$ | utilization of the bottleneck | – |
| $\varepsilon$ | violation probability of the bound (p95 ↔ $\varepsilon = 0.05$) | – |
| $K$ | number of services on the path (chain depth) | – |

**Load is set through $\rho$, not through $\lambda$.** The simulation and the testbed have different execution times,
so the same $\lambda$ means different loads. Each scenario below has one "volume" parameter (number of cameras,
number of aggregators, …) that is scaled to reach the target $\rho$ for the configuration under test.

## Arrival scenarios

### S0 – Poisson (reference)
Many independent, unsynchronized sources. Gaps between arrivals are exponential with mean $1/\lambda$.

- *Role:* the textbook case and the middle of the burstiness range; not claimed to be realistic on its own.
- *Parameters:* $\lambda$ only.

### S1 – Camera streams (smooth)
$k$ cameras, each sending frames at $f$ frames per second. Each frame is delayed by a small random jitter (encoding,
network), and cameras start at random phase offsets, so their frames don't align.

| parameter | default | variants | why |
|---|---|---|---|
| frame rate $f$ | 10 fps | 5, 25 | typical rates for analytics pipelines; 25 fps is PAL video |
| relative jitter $j$ | 2% of $1/f$ | 0%, 10% | network/encoding jitter is small compared to the frame interval |
| cameras $k$ | set by target $\rho$ | – | volume parameter: $\lambda = k f$ |
| phase offsets | uniform in $[0, 1/f)$ | all zero (synchronized) | unsynchronized is the normal case; synchronized is a stress test |

- *Expected:* with $k = 1$ no queue forms and the bound is tight. With many cameras, the merged stream moves towards
  Poisson. Synchronized cameras deliver $k$ frames at once every $1/f$ seconds, which turns S1 into a batch process.

### S2 – Hierarchical IoT (bursty)
Sensors report periodically to an **aggregator**, which collects reports for a window and forwards them as one batch.
Some aggregators are **unstable**: they lose connectivity, buffer their batches, and flush the whole buffer on
reconnect.

| parameter | default | variants | why |
|---|---|---|---|
| sensors per aggregator $N$ | 50 | 10, 200 | a building floor or a small field deployment |
| sensor period $P$ | 10 s | 1 s, 60 s | common telemetry periods |
| aggregation window $W$ | 2 s | 0.5 s, 10 s | trades freshness against transmission overhead |
| batch size $B$ | Binomial($N W / P$, 0.95) | – | reports in one window; 5% lost or late |
| aggregators $A$ | set by target $\rho$ | – | volume parameter: $\lambda \approx 0.95 \cdot A N / P$ |
| unstable share | 20% of aggregators | 0%, 50% | 0% isolates the batching effect; 50% is a poorly connected deployment |
| up time (unstable aggregator) | exponential, mean 120 s | 30 s | with the default down time, an unstable aggregator is offline about 8% of the time |
| down time (unstable aggregator) | exponential, mean 10 s | 60 s | short outages are common; long ones are the stress case |
| flush | the whole buffer arrives at once on reconnect | spread over 1 s | worst case first |

- *Variant S2-sync ("reconnect storm"):* a shared outage (e.g. the uplink fails) takes all aggregators down at once.
  On recovery they all flush at the same time. This is the synchronized-access case of 3GPP TR 37.868.
- *Expected:* both batches and flushes put many items into the queue at once. Latency is dominated by the position
  of an item inside a batch or flush, and the tail is heavier than under Poisson at the same $\rho$.

## Load levels

For every scenario: $\rho \in \{0.2, 0.5, 0.8\}$ as the main levels. E2 adds a finer sweep.

- $0.2$: lightly loaded; queueing comes only from bursts.
- $0.5$: typical operating point.
- $0.8$: near saturation; queueing dominates.

## Experiments

### E1 – Burstiness spectrum at fixed load (main result)
- **Question:** how do coverage and tightness depend on the arrival process when the bound uses the *matching*
  arrival model?
- **Grid:** scenarios {S0, S1 ($k=1$), S1 (default), S2 (default)} × $\rho \in \{0.2, 0.5, 0.8\}$ × chain depth
  $K \in \{1, 3, 5\}$.
- **Expected:** coverage at or above $1-\varepsilon$ everywhere. Tightness is best for S1 and worst for S2, and gets
  worse with depth for S0 and S2.

### E2 – Load sweep
- **Question:** where does each scenario's bound become impractical?
- **Grid:** each scenario × $\rho \in \{0.1, 0.2, \dots, 0.9\}$, $K = 3$.
- **Report:** tightness vs. $\rho$, one line per scenario; the $\rho$ at which tightness exceeds 2× (a proposed
  threshold for "impractical").

### E3 – Wrong arrival model
- **Question:** what happens if the bound assumes a smoother process than the real one?
- **Grid:** traffic scenario × assumed model, all combinations of {S0, S1, S2} × {Poisson, periodic, matching};
  $\rho = 0.5$, $K = 3$.
- **Expected:** assuming a smoother process than the real one (e.g. Poisson for S2 traffic) breaks coverage. Assuming
  a burstier one keeps coverage but loosens the bound. This motivates learning the arrival model.

### E4 – Learned arrival model
- **Question:** can the agent estimate the arrival model from observed timestamps, and how many observations does it
  need?
- **Method:** estimate the arrival MGF from the last $n$ observed inter-arrival times (or arrival counts per window),
  with a confidence margin (e.g. bootstrap upper bound). This mirrors how the GP learns execution times.
- **Grid:** each scenario × $n \in \{50, 200, 1000, 5000\}$; $\rho = 0.5$, $K = 3$.
- **Expected:** coverage converges to the matching-model case as $n$ grows. On/off processes (S2 with unstable
  aggregators) need the most observations, because outages are rare events.

### E5 – Depth and topology
- **Question:** how does the looseness grow along a chain and in a hierarchy?
- **Grid:** chains $K \in \{1, 2, 3, 5, 10\}$ and a fan-in tree (aggregators → gateway → cloud) for S2; compare
  - (a) every node assumes the external arrival process (current PPG),
  - (b) propagated departure bounds (each node's output bound is the next node's arrival model),
  - (c) the measured-nodes union-bound oracle as the reference for the best achievable tightness.
- **Expected:** (b) closes most of the gap between (a) and (c).

### E6 – Stress: reconnect storm (optional)
- **Question:** does the guarantee survive a synchronized flush, and how long does the system take to recover?
- **Setup:** S2-sync at $\rho = 0.5$; measure latency over time around the storm, not only the stationary quantile.

## Metrics

- **Coverage:** share of configurations whose measured $(1-\varepsilon)$ latency quantile is at or below the bound.
  Target: $\ge 1-\varepsilon$ per scenario.
- **Tightness:** median over configurations of bound ÷ measured quantile, reported with the interquartile range.
- **Per-node and end-to-end:** both, so the node bound and the composition can be told apart (as in `method_v2`,
  Section 9).
- **Per item, not per batch:** in S2, latency is measured for every sensor report, from the moment its batch arrives
  at the first service.

## Implementation prerequisites

| scenario | arrival model the bound needs | status |
|---|---|---|
| S0 | renewal, exponential gaps | in `method_v2` |
| S1, $k = 1$ | renewal, jittered periodic gaps | in `method_v2` |
| S1, $k > 1$ | not renewal; arrival envelope (at most $k$ frames in any window of length $1/f$) | to do |
| S2 without outages | periodic batches: treat a batch as one arrival with work $X_1 + \dots + X_B$, i.e. MGF $\mathbb{E}[M_X(\theta)^B]$ | to do, small change |
| S2 with outages | on/off (Markov-modulated) process, SNC effective bandwidth | to do, larger change |
| E5 (b) | departure bound of each node as arrival model of the next | to do |

## Procedure

1. **One source for arrivals.** Generate arrival timestamps per scenario, load level and seed once, and store them as
   `statics/arrivals/<scenario>_rho<ρ>_seed<s>.csv` (columns: timestamp, source id, batch id). The testbed's load
   generator and the simulation both replay these files, so both see the same traffic.
2. **Seeds:** 5 per cell, reported as mean and spread.
3. **Run length:** long enough for the quantile to be estimated reliably: at least ~200 items beyond the quantile
   (≈ 4,000 items for $\varepsilon = 0.05$). Smaller $\varepsilon$ (e.g. $10^{-3}$) is evaluated in simulation only,
   because it needs ~200,000 items per run. Drop the first 10% as warm-up.
4. **Configurations:** the same held-out configurations as in `method_v2` (random draws from the training range) so
   results stay comparable.

## Open decisions

- Which service chain is used in the testbed for S1 (camera → detection → …) and S2 (aggregator → gateway → cloud),
  and whether both scenarios share services.
- Whether $\varepsilon = 0.05$ stays the main target, or $0.01$ is used in the testbed as well.
- Whether E4 (learned arrival model) is a contribution of this paper or future work; it decides whether E3 is a
  motivation or a result.
