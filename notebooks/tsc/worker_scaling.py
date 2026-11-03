"""How much the execution time of an item depends on the number of workers of its service, on the second campaign
(`statics/results_milos_v2/b1-unthrottled/`).

Pegasus assumes that a service with w workers processes up to w items at the same time and that the execution time of
one item does not depend on w (model_inputs in data.py: no model receives the number of workers). The service then
scales linearly: its capacity is w / (mean execution time). This module measures how far that holds.

Data: the chains of CHAINS_UNTHROTTLED_V2. Every service of a run has w = 1 to 4 workers (knob parallelism), w + 0.5
cores (so no run was throttled), and quality knobs drawn per run; the arrival rate puts the pacing service at a
utilization of 0.5. Every worker is a single-threaded process (OMP_NUM_THREADS = 1, OpenCV with one thread), so one
item never uses more than one core. Every quality configuration of a service (its knobs other than cores and
parallelism) was measured at two or more worker counts, which allows matched comparisons of the same configuration.

Three measurements per service:
- Across runs (worker_effects on run_level): the ratio r_w of a statistic of the run (mean or 95th percentile of the
  execution time of the items of the run) at w workers to the same quality configuration at 1 worker. This is the
  total effect at the load of the campaign: it contains the concurrency between the items of the service, the lower
  number of cores per worker (1.5 at w = 1, 1.125 at w = 4), and the longer idle gaps of each worker at a lower load
  per worker.
- Within runs, own concurrency (item_level, own_ratio_<k>): in one run (configuration and arrival rate fixed), the
  median execution time of the items that started while k other items of the same service were being processed,
  divided by the median of the items that started alone. This is the penalty per concurrent item, independent of
  the load at which the campaign happened to run.
- Within runs, alone across worker counts (worker_effects on alone_ms): the median execution time of the items that
  started alone, matched across worker counts like the total effect. It holds the effects that are not concurrency
  within the service (cores per worker, idle gaps, more processes in the container, other services of the chain).
Execution time is the campaign's processing time: finish minus start of an item at a worker, without waiting.
Units: times in milliseconds where a name ends in _ms, else seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pegasus.load_independence import item_states
from pegasus.real_data import CHAINS_UNTHROTTLED_V2, chain_experiments, experiment_folder, node_configurations, run_table

CACHE = CHAINS_UNTHROTTLED_V2.parent.parent / "worker_scaling"   # statics/worker_scaling/
RECORD_COLUMNS = ["admission_phase", "outcome", "started_at_ns", "finished_at_ns", "worker_id"]
MAX_WORKERS = 4
MIN_GROUP = 8   # fewest items in a group of a within-run median; a run with fewer gives no ratio for that group


def _other_services_busy(start: np.ndarray, others: list) -> np.ndarray:
    """Number of items in process at the other nodes of the run at each time of `start` (s); others: (started,
    finished) arrays of every other node."""
    busy = np.zeros(len(start), dtype=int)
    for started, finished in others:
        busy += np.searchsorted(np.sort(started), start, side="right") - np.searchsorted(np.sort(finished), start, side="right")
    return busy


def _worker_gaps(started: np.ndarray, finished: np.ndarray, worker: np.ndarray) -> np.ndarray:
    """Idle time (s) of the worker of each item before it started the item: start minus the finish of the previous item
    of the same worker; nan for the first item of a worker."""
    order = np.lexsort((started, worker))
    gap = np.full(len(started), np.nan)
    same = worker[order][1:] == worker[order][:-1]
    gap[order[1:][same]] = started[order][1:][same] - finished[order][:-1][same]
    return gap


def _tercile_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """Median of y over the third of the items with the largest x / over the third with the smallest x; nan if a group
    has fewer than MIN_GROUP items or the tercile boundaries coincide."""
    finite = np.isfinite(x)
    y, x = y[finite], x[finite]
    if len(x) < 3 * MIN_GROUP:
        return np.nan
    low, high = np.quantile(x, [1 / 3, 2 / 3])
    small, large = y[x <= low], y[x >= high]
    if low == high or min(len(small), len(large)) < MIN_GROUP:
        return np.nan
    return float(np.median(large) / np.median(small))


def _run_items(folder: Path, nodes: list) -> dict:
    """node -> (execution time (s) of every completed item of the measurement phase, number of other items of the node
    in process when it started, number of items in process at the other nodes of the run when it started, idle time (s)
    of its worker before it started)."""
    times = {}
    for node in nodes:
        records = pd.read_csv(folder / node / "records.csv", usecols=RECORD_COLUMNS)
        records = records[records["outcome"] == "completed"]
        times[node] = (records["started_at_ns"].to_numpy() / 1e9, records["finished_at_ns"].to_numpy() / 1e9,
                       (records["admission_phase"] == "measure").to_numpy(), records["worker_id"].to_numpy())
    result = {}
    for node, (started, finished, measured, worker) in times.items():
        own, _ = item_states(started, finished)   # over all items, so that warm-up items still in process count
        rest = _other_services_busy(started, [(s, f) for other, (s, f, _, _) in times.items() if other != node])
        gap = _worker_gaps(started, finished, worker)
        result[node] = ((finished - started)[measured], own[measured], rest[measured], gap[measured])
    return result


def item_level(recompute: bool = False) -> pd.DataFrame:
    """One row per valid run and node of the chains of CHAINS_UNTHROTTLED_V2 (without the checks of the setup).
    Columns: experiment, run_id, node, service, configuration (quality knobs as one string, without cores and
    parallelism), workers, cores, rate (items/s at the node), utilization (rate x mean execution time / workers, as the
    campaign measured it), throttled, mean_ms and p95_ms (of the run, from dataset.csv), items (completed items of the
    measurement phase), share_alone (share of them that started with no other item of the node in process),
    alone_ms (median execution time of those items), own_ratio_<k> for k = 1 .. MAX_WORKERS - 1 (median execution time
    of the items that started next to exactly k other items of the node / alone_ms; nan if either group has fewer than
    MIN_GROUP items), and rest_ratio (among the items that started alone: median execution time of those that started
    while at least one item was in process at another node of the run / of those that started while none was),
    worker_gap_s (median idle time of the worker before an item, over the items that started alone) and idle_ratio
    (among the items that started alone: median execution time of the third with the longest idle time of their worker
    / of the third with the shortest).
    Read from CACHE/item_level.csv unless it is missing or recompute is True (reading every records.csv takes a few
    minutes)."""
    path = CACHE / "item_level.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    rows = []
    # hidden folders (e.g. a tool's lock folder) are no experiments
    for experiment in [name for name in chain_experiments(CHAINS_UNTHROTTLED_V2) if not name.startswith(".")]:
        table = run_table(experiment)
        nodes = list(table["node"].unique())
        configurations = {node: node_configurations(table, node) for node in nodes}
        for run_id, run in table.groupby("run_id"):
            items = _run_items(experiment_folder(experiment) / "runs" / run_id, nodes)
            for row in run.itertuples():
                configuration = configurations[row.node].loc[run_id]
                y, own, rest, gap = items[row.node]
                alone = own == 0
                median = lambda chosen: float(np.median(y[chosen]) * 1e3) if chosen.sum() >= MIN_GROUP else np.nan
                alone_ms = median(alone)
                rows.append(dict(experiment=experiment, run_id=run_id, node=row.node, service=row.service,
                                 configuration="|".join(f"{k}={v}" for k, v in configuration.drop(["cores", "parallelism"]).astype(str).items()),
                                 workers=int(configuration["parallelism"]), cores=float(configuration["cores"]),
                                 rate=row.rate_received, utilization=row.utilization_measured, throttled=row.throttled_fraction,
                                 mean_ms=row.processing_mean_ms, p95_ms=row.processing_p95_ms, items=len(y),
                                 share_alone=float(alone.mean()), alone_ms=alone_ms,
                                 **{f"own_ratio_{k}": median(own == k) / alone_ms for k in range(1, MAX_WORKERS)},
                                 rest_ratio=median(alone & (rest > 0)) / median(alone & (rest == 0)),
                                 worker_gap_s=float(np.nanmedian(gap[alone])) if alone.any() else np.nan,
                                 idle_ratio=_tercile_ratio(y[alone], gap[alone])))
    level = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    level.to_csv(path, index=False)
    return level


def _worker_coefficients(log_y: np.ndarray, configuration: np.ndarray, workers: np.ndarray, levels: list) -> np.ndarray:
    """Least-squares fit of log_y = a_configuration + b_w (b_1 = 0): one intercept per configuration and one offset per
    worker count; returns b_w for every w in levels[1:] (nan if a level is missing)."""
    configurations, column = np.unique(configuration, return_inverse=True)
    design = np.zeros((len(log_y), len(configurations) + len(levels) - 1))
    design[np.arange(len(log_y)), column] = 1
    for j, w in enumerate(levels[1:]):
        design[:, len(configurations) + j] = workers == w
    present = np.array([np.any(workers == w) for w in levels[1:]])
    coefficients = np.linalg.lstsq(design, log_y, rcond=None)[0][len(configurations):]
    return np.where(present, coefficients, np.nan)


def worker_effects(level: pd.DataFrame, column: str, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Ratio r_w of `column` (a run statistic of item_level, e.g. mean_ms) at w workers to the same configuration at
    1 worker, per service: exp(b_w) of _worker_coefficients on the logarithm, so r_w is a geometric mean over the
    configurations of the service. Matching on the configuration removes the effect of the quality knobs, which were
    drawn independently of the workers. The 95% interval is a cluster bootstrap over the configurations of the service
    (all runs of a configuration are resampled together, because they share whatever the knobs do not describe), with
    the given seed. Assumption: the effect of w is multiplicative and the same for every configuration of a service.
    Returns one row per service and w = 2 .. MAX_WORKERS: service, workers, ratio, low, high, configurations, runs."""
    levels = list(range(1, MAX_WORKERS + 1))
    rows = []
    for service, runs in level[np.isfinite(level[column]) & (level[column] > 0)].groupby("service"):
        log_y, configuration, workers = np.log(runs[column].to_numpy()), runs["configuration"].to_numpy(), runs["workers"].to_numpy()
        estimate = _worker_coefficients(log_y, configuration, workers, levels)
        members = [np.flatnonzero(configuration == c) for c in np.unique(configuration)]
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            drawn = rng.integers(0, len(members), len(members))
            index = np.concatenate([members[g] for g in drawn])
            # every draw of a configuration is its own cluster, so that a configuration drawn twice gets two intercepts
            draw = np.concatenate([np.full(len(members[g]), k) for k, g in enumerate(drawn)])
            boot.append(_worker_coefficients(log_y[index], draw, workers[index], levels))
        low, high = np.nanquantile(np.array(boot), [0.025, 0.975], axis=0)
        for j, w in enumerate(levels[1:]):
            rows.append(dict(service=service, workers=w, ratio=float(np.exp(estimate[j])), low=float(np.exp(low[j])), high=float(np.exp(high[j])),
                             configurations=len(members), runs=int(np.sum(workers == w))))
    return pd.DataFrame(rows)
