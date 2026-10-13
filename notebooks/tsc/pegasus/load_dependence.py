"""Does the load of a service change its execution time? Helpers of `explain_6_when_load_changes_execution_times.ipynb`.

Two parts:
- Measurements of the testbed campaign (`real_data.py`): per run of a service measured alone, how often its CPU quota
  was used up (throttling); per run and node of a pipeline, how the processing time of an item depends on what the
  service was doing when the item started.
- Two small simulators that isolate one mechanism each: a worker pool under a CPU quota, and a service that is slower
  after an idle period, with the latency bounds that two kinds of benchmark of that service give.
Units: times in seconds, rates in items/s.
"""
import numpy as np
import pandas as pd

from .bounds import martingale_closed_form_node_bound
from .real_data import CAMPAIGN, experiment_folder, run_table
from .simulation import quantile_with_lower_bound


# ------------------------------------------------------------------------------
# Measurements: services alone
# ------------------------------------------------------------------------------
def single_service_runs(min_items: int = 50) -> pd.DataFrame:
    """One row per valid run of a service measured alone (experiments "*-single-*") with at least min_items items:
    experiment, service, workers (the knob `parallelism`), cores (the CPU quota of the container), throttled (share of
    the scheduling periods in which the container had used up its quota), cpu_use (CPU time used / CPU time allowed by
    the quota during the measurement), cores_per_busy_worker (CPU time used / total processing time of the items), and
    tail_ratio (95th percentile / median of the processing time of the run)."""
    rows = []
    for folder in sorted(path for path in CAMPAIGN.iterdir() if path.is_dir() and "-single-" in path.name):
        table = run_table(folder.name)
        node = table["node"].iloc[0]
        measure_seconds = table["measured_items"] / table["rate_received"]
        rows.append(pd.DataFrame(dict(
            experiment=folder.name, service=table["service"], workers=table[f"{node}.parallelism"], cores=table[f"{node}.cores"],
            throttled=table["throttled_fraction"], cpu_use=table["cpu_seconds_used"] / (measure_seconds * table[f"{node}.cores"]),
            cores_per_busy_worker=table["cpu_seconds_used"] / (table["measured_items"] * table["processing_mean_ms"] / 1e3),
            tail_ratio=table["processing_p95_ms"] / table["processing_p50_ms"], items=table["measured_items"])))
    runs = pd.concat(rows, ignore_index=True).dropna()
    return runs[runs["items"] >= min_items].drop(columns="items").reset_index(drop=True)


# ------------------------------------------------------------------------------
# Measurements: items inside pipeline runs
# ------------------------------------------------------------------------------
def _pipeline_experiments() -> tuple:
    """Names of the pipeline experiments whose dataset.csv is readable, and of those whose file is cut off."""
    readable, cut_off = [], []
    for folder in sorted(path for path in CAMPAIGN.iterdir() if path.is_dir() and path.name[0].isdigit() and "-single-" not in path.name):
        try:
            run_table(folder.name)
            readable.append(folder.name)
        except (ValueError, KeyError, pd.errors.ParserError):
            cut_off.append(folder.name)
    return readable, cut_off


def within_run_effects(idle_s: float = 0.5, back_to_back_s: float = 0.05, window_s: float = 2.0, min_items: int = 60,
                       min_group: int = 8) -> pd.DataFrame:
    """How the processing time of an item depends on the state of its service, within one run (one configuration).
    One row per valid pipeline run and node with at least min_items completed items. An item starts at
    finish - processing time, and is put into one of three groups by what the service did at that moment:
    - "next to another": another item of the node was still being processed (needs at least 2 workers);
    - "back to back": the item processed before it finished less than back_to_back_s earlier;
    - "after idle": the node had processed nothing for more than idle_s.
    Columns, each a ratio of median processing times within the run (nan if a group has fewer than min_group items):
    next_to_another = "next to another" / "back to back", after_idle = "after idle" / "back to back", and
    busy_vs_calm = the third of the items with the most arrivals at the node in the window_s before their own arrival /
    the third with the fewest. Also: experiment, run_id, node, service, workers, cores, throttled.
    `attrs["cut_off"]` lists the experiments that were left out because their dataset.csv is cut off."""
    experiments, cut_off = _pipeline_experiments()
    rows = []
    for experiment in experiments:
        table = run_table(experiment)
        for run_id, run in table.groupby("run_id"):
            file = experiment_folder(experiment) / "runs" / run_id / "trace_steps.csv"
            if not file.exists():
                continue
            steps = pd.read_csv(file, on_bad_lines="skip")
            numeric = ["processing_ms", "arrived_at_ns", "finished_at_ns"]
            steps[numeric] = steps[numeric].apply(pd.to_numeric, errors="coerce")   # a cut-off last line leaves text or gaps
            steps = steps[steps["outcome"] == "completed"].dropna(subset=numeric)
            for node, items in steps.groupby("node"):
                row = run[run["node"] == node]
                if len(row) != 1 or f"{node}.parallelism" not in row or len(items) < min_items:
                    continue
                finish = items["finished_at_ns"].to_numpy() / 1e9
                processing = items["processing_ms"].to_numpy() / 1e3
                arrived = items["arrived_at_ns"].to_numpy() / 1e9
                order = np.argsort(finish - processing)
                start, finish, processing, arrived = (finish - processing)[order], finish[order], processing[order], arrived[order]
                # time since the node last finished an item that started earlier; negative if such an item is still being processed
                idle = start - np.concatenate([[-np.inf], np.maximum.accumulate(finish)[:-1]])
                groups = {"next": idle < 0, "back": (idle >= 0) & (idle < back_to_back_s), "idle": idle > idle_s}
                median = {name: np.median(processing[chosen]) if chosen.sum() >= min_group else np.nan for name, chosen in groups.items()}
                sorted_arrivals = np.sort(arrived)
                recent = np.searchsorted(sorted_arrivals, arrived) - np.searchsorted(sorted_arrivals, arrived - window_s)
                calm, busy = processing[recent <= np.quantile(recent, 1 / 3)], processing[recent >= np.quantile(recent, 2 / 3)]
                rows.append(dict(experiment=experiment, run_id=run_id, node=node, service=row["service"].iloc[0],
                                 workers=int(row[f"{node}.parallelism"].iloc[0]), cores=float(row[f"{node}.cores"].iloc[0]),
                                 throttled=float(row["throttled_fraction"].iloc[0]), next_to_another=median["next"] / median["back"],
                                 after_idle=median["idle"] / median["back"], busy_vs_calm=np.median(busy) / np.median(calm)))
    effects = pd.DataFrame(rows)
    effects.attrs["cut_off"] = cut_off
    return effects


# ------------------------------------------------------------------------------
# Simulator 1: a worker pool under a CPU quota
# ------------------------------------------------------------------------------
def simulate_cpu_quota(workers: int, cores_per_worker: float, cores: float, rate: float, mean_s: float, sd_s: float,
                       duration_s: float, rng: np.random.Generator, period_s: float = 0.1, step_s: float = 0.001) -> tuple:
    """A FIFO pool of `workers` workers in a container with a CPU quota of `cores`, under Poisson arrivals at `rate`.
    The quota works as in Linux containers: in every scheduling period of period_s the container may use
    cores * period_s of CPU time; once that is used up, all its workers stand still until the period ends (throttling).
    A busy worker uses cores_per_worker cores. An item needs a Gaussian time (mean_s, sd_s) of undisturbed processing.
    Time advances in steps of step_s. Returns the execution times of the finished items (s, start to finish, including
    the time the worker stood still) and the share of the periods in which the container was throttled."""
    arrivals = np.cumsum(rng.exponential(1.0 / rate, int(2 * rate * duration_s) + 10))
    arrivals = arrivals[arrivals < duration_s]
    work = np.maximum(mean_s + sd_s * rng.standard_normal(len(arrivals)), step_s)
    steps_per_period = int(round(period_s / step_s))
    remaining, started = [0.0] * workers, [0.0] * workers   # undisturbed processing still needed (0: free), and the start of the item
    next_item, budget, throttled_now, throttled_periods, execution = 0, 0.0, False, 0, []
    n_steps = int(duration_s / step_s)
    for step in range(n_steps):
        now = step * step_s
        if step % steps_per_period == 0:
            throttled_periods += throttled_now
            budget, throttled_now = cores * period_s, False
        for w in range(workers):   # waiting items start at free workers, in the order of arrival
            if remaining[w] <= 0 and next_item < len(arrivals) and arrivals[next_item] <= now:
                remaining[w], started[w] = work[next_item], now
                next_item += 1
        busy = [w for w in range(workers) if remaining[w] > 0]
        if not busy:
            continue
        wanted = len(busy) * cores_per_worker * step_s   # CPU time the busy workers would use in this step
        share = min(1.0, budget / wanted)                 # below 1: the quota of the period is used up
        budget -= share * wanted
        throttled_now = throttled_now or share < 1.0
        for w in busy:
            remaining[w] -= share * step_s
            if remaining[w] <= 0:
                execution.append(now + step_s - started[w])
    return np.array(execution), throttled_periods / (n_steps / steps_per_period)


# ------------------------------------------------------------------------------
# Simulator 2: a service that is slower after an idle period
# ------------------------------------------------------------------------------
def simulate_warmup_service(arrivals: np.ndarray, warm_execution: np.ndarray, cold_factor: float, idle_s: float) -> tuple:
    """One FIFO worker whose execution time depends on its state, vectorized over rows (e.g. arrival rates).
    arrivals: (rows, items) arrival times (s, increasing); warm_execution: (rows, items) execution time of every item
    if it starts right after another one (s). An item that starts after the worker was idle for more than idle_s is
    *cold* and takes cold_factor times as long. Returns the execution times and the latencies (waiting + execution),
    both (rows, items), and the share of cold items per row."""
    execution, latency = np.empty_like(warm_execution), np.empty_like(warm_execution)
    previous_finish = np.full(len(arrivals), -np.inf)
    for i in range(arrivals.shape[1]):
        start = np.maximum(arrivals[:, i], previous_finish)
        execution[:, i] = warm_execution[:, i] * np.where(start - previous_finish > idle_s, cold_factor, 1.0)
        previous_finish = start + execution[:, i]
        latency[:, i] = previous_finish - arrivals[:, i]
    return execution, latency, np.mean(execution > warm_execution, axis=1)


def warmup_bound_comparison(cold_factor: float, rates: np.ndarray, warm_mean: float, warm_sd: float, idle_s: float, epsilon: float,
                            n_deployed: int, n_benchmark: int, seed) -> dict:
    """Latency bounds of one service with a warm-up effect (simulate_warmup_service, Gaussian warm execution times with
    warm_mean and warm_sd) from two benchmarks, next to the latency that the service has under Poisson arrivals at
    every rate of `rates`:
    - "isolated": one item at a time with gaps of 2 * idle_s, so every item starts after idle;
    - "saturated": all items available at once, so every item but the first starts right after another one.
    Each benchmark gives the mean and the variance of its n_benchmark execution times, and from them the bound of one
    service at epsilon for every rate (martingale_closed_form_node_bound; inf where the service would be overloaded).
    Returns measured and measured_low (the (1 - epsilon) latency quantile of n_deployed items per rate without the
    first tenth, and the lower edge of its confidence interval), cold_share (share of items that start after idle, per
    rate), benchmark_mean (benchmark -> s) and bounds (benchmark -> one bound per rate). The same seed gives the same
    arrivals and warm execution times for every cold_factor."""
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates, dtype=float)
    arrivals = np.cumsum(rng.exponential(1.0, (len(rates), n_deployed)) / rates[:, None], axis=1)
    warm_times = np.maximum(warm_mean + warm_sd * rng.standard_normal((len(rates), n_deployed)), 0.0)
    _, latency, cold_share = simulate_warmup_service(arrivals, warm_times, cold_factor, idle_s)
    measured, measured_low = quantile_with_lower_bound(latency[:, n_deployed // 10:], 1 - epsilon)
    benchmark_times = np.maximum(warm_mean + warm_sd * rng.standard_normal((1, n_benchmark)), 0.0)
    benchmark_arrivals = {"isolated": 2 * idle_s * np.arange(1, n_benchmark + 1)[None, :], "saturated": np.zeros((1, n_benchmark))}
    benchmark_mean, bounds = {}, {}
    for name, starts in benchmark_arrivals.items():
        times = simulate_warmup_service(starts, benchmark_times, cold_factor, idle_s)[0][0]
        benchmark_mean[name] = times.mean()
        bounds[name] = np.concatenate([martingale_closed_form_node_bound(times.mean(), times.var(), "poisson", rate, epsilon) for rate in rates])
    return dict(measured=measured, measured_low=measured_low, cold_share=cold_share, benchmark_mean=benchmark_mean, bounds=bounds)
