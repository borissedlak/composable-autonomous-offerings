"""`explain_9`: is the execution time of a service independent of its load on the second campaign
(`statics/results_milos_v2/`), and does provisioning enough cores per worker make it so?

Three comparisons, each split by how the service was provisioned in the run (provisioning_class):
- Probe against run (run_states, columns probe_ms and probe_rate): before every run, the campaign measured every
  service of the chain for a few seconds at a low arrival rate (the probe), and then ran the chain at the rate that puts
  the pacing service at the target utilization. Every run therefore holds the same configuration at two loads.
- Across runs (repeated_configurations, within_configuration_elasticity): runs of the same configuration of a service
  that happened to receive different arrival rates. Only services that did not pace the chain qualify, because the
  campaign set the arrival rate of a run from the probe of its pacing service.
- Within runs (run_states): how the processing time of an item depends on the state of its service when the item
  started: next to other items that were being processed, after a short or a long gap without processing, or after
  many or few recent arrivals. Within one run the configuration and the arrival rate are fixed.
Units: times in seconds unless a name ends in _ms, rates in items/s.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .real_data import CAMPAIGN, CAMPAIGN_V2, experiment_folder, node_configurations, run_table

CACHE = CAMPAIGN.parent / "load_independence"   # results of run_states, one folder per campaign
RECORD_COLUMNS = ["admission_phase", "outcome", "arrived_at_ns", "started_at_ns", "finished_at_ns"]
PROVISIONING = ("r < c", "r = c", "r > c")


def provisioning_class(cores, workers) -> np.ndarray:
    """Provisioning of a run from its CPU quota `cores` and its number of workers `workers` (the knob parallelism):
    "r < c" (fewer cores than workers), "r = c" (one core per worker) or "r > c" (more than one core per worker)."""
    ratio = np.asarray(cores, dtype=float) / np.asarray(workers, dtype=float)
    return np.select([ratio < 1, ratio == 1], ["r < c", "r = c"], "r > c")


def item_states(started: np.ndarray, finished: np.ndarray) -> tuple:
    """State of a service when each of its items started, from the start and finish times (s) of all items of one run
    and node. Returns, in the order of the input: others (number of other items that were being processed at that
    moment) and gap (time since the service last finished an item that had started earlier; negative if such an item
    was still being processed, -inf for the first item)."""
    order = np.argsort(started, kind="stable")
    start, finish = started[order], finished[order]
    # items that started earlier and had already finished; every item that finished before start i started before it
    finished_before = np.searchsorted(np.sort(finish), start, side="right")
    others = np.arange(len(start)) - finished_before
    gap = start - np.concatenate([[-np.inf], np.maximum.accumulate(finish)[:-1]])
    result_others, result_gap = np.empty_like(others), np.empty_like(gap)
    result_others[order], result_gap[order] = others, gap
    return result_others, result_gap


def _tercile_ratio(y: np.ndarray, x: np.ndarray, min_group: int) -> float:
    """Median of y over the third of the items with the largest x / over the third with the smallest x (ties at the
    tercile boundaries go into both groups); nan if a group has fewer than min_group items or the boundaries coincide."""
    if len(x) < 3 * min_group:
        return np.nan
    low, high = np.quantile(x, [1 / 3, 2 / 3])
    small, large = y[x <= low], y[x >= high]
    if low == high or min(len(small), len(large)) < min_group:
        return np.nan
    return float(np.median(large) / np.median(small))


def _run_node_statistics(records: pd.DataFrame, min_group: int, window_s: float) -> dict:
    """Within-run statistics of one node (see run_states)."""
    records = records[records["outcome"] == "completed"]
    started, finished = records["started_at_ns"].to_numpy() / 1e9, records["finished_at_ns"].to_numpy() / 1e9
    others, gap = item_states(started, finished)   # over all items, so that warm-up items still in process count
    arrived = np.sort(records["arrived_at_ns"].to_numpy() / 1e9)
    recent = np.searchsorted(arrived, started) - np.searchsorted(arrived, started - window_s)   # arrivals in the window before the start
    measured = (records["admission_phase"] == "measure").to_numpy()
    y, others, gap, recent = (finished - started)[measured], others[measured], gap[measured], recent[measured]
    alone = (others == 0) & np.isfinite(gap)
    enough = lambda chosen: chosen.sum() >= min_group
    return dict(items=int(measured.sum()), share_next=float(np.mean(others > 0)),
                next_ratio=np.median(y[others > 0]) / np.median(y[alone]) if enough(others > 0) and enough(alone) else np.nan,
                gap_ratio=_tercile_ratio(y[alone], gap[alone], min_group),
                window_ratio=_tercile_ratio(y, recent, min_group))


def _probe(experiment: str, run_id: str) -> tuple:
    """Probe of one run: the mean processing time (ms) that the campaign measured for every node at the probe rate, and
    that rate (items/s, the first rate of the run's schedule, run_summary.json of any node)."""
    folder = experiment_folder(experiment) / "runs" / run_id
    plan = json.loads((folder / "run.json").read_text())
    node = next(iter(plan["probed_processing_ms"]))
    summary = json.loads((folder / node / "run_summary.json").read_text())
    return plan["probed_processing_ms"], summary["rate_changes"][0]["rate_per_second"]


def run_states(campaign: Path = CAMPAIGN_V2, min_group: int = 8, window_s: float = 1.0, recompute: bool = False) -> pd.DataFrame:
    """One row per valid run and node of `campaign`. Columns:
    - experiment, run_id, node, service, configuration (all knobs of the node, i.e. quality knobs, cores and
      parallelism, as one string), cores, workers (parallelism), cores_per_worker, provisioning (provisioning_class);
    - rate (measured arrival rate at the node in the run, items/s), utilization (as measured by the campaign: rate x
      mean processing time / workers), throttled (share of the CPU scheduling periods of the run in which the container
      had used up its quota), mean_ms, p50_ms, p95_ms (mean, median and 95th percentile of the processing time of the
      items of the run);
    - probe_ms (mean processing time of the node in the probe before the run, ms), probe_rate (arrival rate of the
      probe, items/s), probe_ratio (mean_ms / probe_ms) and pacing (the node had the largest probe_ms / workers of its
      run, i.e. it is the service from whose probe the campaign set the arrival rate; decided before the run, so it
      does not depend on the run's own measurements);
    - within the run, over the completed items of the measurement phase (item_states): items; share_next (share of the
      items that started while another item of the node was being processed); next_ratio (median processing time of
      the items that started next to another item / of those that started alone; nan with one worker); gap_ratio
      (among the items that started alone: median processing time of the third with the longest gap since the service
      last finished an item / of the third with the shortest, which started right after the previous item, mostly from
      the queue); window_ratio (median processing time of the third of the items with the most arrivals at the node in
      the window_s before they started / of the third with the fewest, i.e. momentary high load / low load). A ratio is
      nan if one of its groups has fewer than min_group items.
    Read from CACHE/<campaign>/run_states.csv (<campaign>: its path below statics/ with "_" for "/") if that file exists and recompute is False (computing it reads
    every records.csv of the campaign, about a minute); otherwise computed and written there."""
    path = CACHE / "_".join(campaign.relative_to(CAMPAIGN.parent).parts) / "run_states.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    rows = []
    for folder in sorted(p for p in campaign.iterdir() if p.is_dir()):
        table = run_table(folder.name)
        workers = {node: node_configurations(table, node)["parallelism"] for node in table["node"].unique()}
        for run_id, run in table.groupby("run_id"):
            probe_ms, probe_rate = _probe(folder.name, run_id)
            pacing = max(probe_ms, key=lambda node: probe_ms[node] / workers[node].loc[run_id])
            for row in run.itertuples():
                node = row.node
                configuration = node_configurations(table, node).loc[run_id]
                records = pd.read_csv(experiment_folder(folder.name) / "runs" / run_id / node / "records.csv", usecols=RECORD_COLUMNS)
                rows.append(dict(experiment=folder.name, run_id=run_id, node=node, service=row.service,
                                 configuration="|".join(f"{k}={v}" for k, v in configuration.astype(str).items()),
                                 cores=float(configuration["cores"]), workers=int(configuration["parallelism"]),
                                 rate=row.rate_received, utilization=row.utilization_measured, throttled=row.throttled_fraction,
                                 mean_ms=row.processing_mean_ms, p50_ms=row.processing_p50_ms, p95_ms=row.processing_p95_ms,
                                 probe_ms=probe_ms[node], probe_rate=probe_rate, pacing=node == pacing,
                                 **_run_node_statistics(records, min_group, window_s)))
    states = pd.DataFrame(rows)
    states = states.assign(cores_per_worker=states["cores"] / states["workers"], probe_ratio=states["mean_ms"] / states["probe_ms"],
                           provisioning=provisioning_class(states["cores"], states["workers"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(path, index=False)
    return states


# ------------------------------------------------------------------------------
# Across loads: summaries
# ------------------------------------------------------------------------------
def ratio_summary(values, n_boot: int = 2000, seed: int = 0) -> dict:
    """Summary of a ratio measured once per run (e.g. probe_ratio or next_ratio): the median over the runs with a 95%
    bootstrap interval (resampling runs), and the p-value of a Wilcoxon signed-rank test of whether the logarithms are
    centered at 0, i.e. whether the ratio is 1. Assumptions of the test: the runs are independent, and the log ratios
    are symmetric around their median. nan values are left out; with fewer than 5 runs everything is nan."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 5:
        return dict(median=np.nan, low=np.nan, high=np.nan, p=np.nan, runs=len(values))
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, (n_boot, len(values))), axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return dict(median=float(np.median(values)), low=low, high=high, p=float(stats.wilcoxon(np.log(values)).pvalue), runs=len(values))


def ratio_table(states: pd.DataFrame, column: str, by: list) -> pd.DataFrame:
    """ratio_summary of `column` for every group of `by` (columns of run_states), one row per group."""
    rows = [dict(zip(by, key if isinstance(key, tuple) else (key,)), **ratio_summary(group[column]))
            for key, group in states.groupby(by)]
    return pd.DataFrame(rows)


def repeated_configurations(states: pd.DataFrame, min_rate_ratio: float = 1.2) -> pd.DataFrame:
    """The rows of run_states that can show an effect of the arrival rate across runs: runs in which the node did not
    pace the chain, of configurations (service and configuration) with at least two such runs whose arrival rates
    differ by at least the factor min_rate_ratio. Adds log_rate and log_mean, each minus its average over the runs of
    the configuration (so only the differences within a configuration remain)."""
    free = states[~states["pacing"]].copy()
    key = [free["service"], free["configuration"]]
    spread = free.groupby(key)["rate"].transform(lambda rate: rate.max() / rate.min())
    chosen = free[(free.groupby(key)["rate"].transform("size") >= 2) & (spread >= min_rate_ratio)].copy()
    for name, column in (("log_rate", "rate"), ("log_mean", "mean_ms")):
        value = np.log(chosen[column])
        chosen[name] = value - value.groupby([chosen["service"], chosen["configuration"]]).transform("mean")
    return chosen


def within_configuration_elasticity(chosen: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Elasticity e of the mean processing time with respect to the arrival rate, within configurations: the
    least-squares slope of log_mean on log_rate of repeated_configurations (both centered per configuration, so the
    line goes through the origin), i.e. by how many percent the mean processing time changes per percent of arrival
    rate (doubling the rate multiplies it by 2^e). The 95% interval is a cluster bootstrap over the configurations (all
    runs of a configuration are resampled together, because they share whatever no knob describes). Returns e, low,
    high, configurations and runs; nan with fewer than 5 configurations."""
    x, y = chosen["log_rate"].to_numpy(), chosen["log_mean"].to_numpy()
    groups = pd.factorize(chosen["service"] + "|" + chosen["configuration"])[0]
    members = [np.flatnonzero(groups == g) for g in range(groups.max() + 1)] if len(groups) else []
    if len(members) < 5:
        return dict(e=np.nan, low=np.nan, high=np.nan, configurations=len(members), runs=len(x))
    slope = lambda index: np.sum(x[index] * y[index]) / np.sum(x[index] ** 2)
    rng = np.random.default_rng(seed)
    boot = [slope(np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])) for _ in range(n_boot)]
    low, high = np.quantile(boot, [0.025, 0.975])
    return dict(e=slope(np.arange(len(x))), low=low, high=high, configurations=len(members), runs=len(x))
