"""Reader of the two testbed campaigns, `statics/results_milos/` (CAMPAIGN, the first campaign) and
`statics/results_milos_v2/` (CAMPAIGN_V2, the B1 experiments of `new_experiments.md`), and the emulation that turns
their measurements into chain data: the measured processing times of the real services are replayed through a chain of
FIFO single-server queues (`fifo_chain_latencies` in `simulation.py`); the second campaign needs no emulation, its
latencies are measured (chain_run_latencies). An experiment is named by its folder, and
campaign_of finds the campaign that holds it.

Why an emulation: in the first campaign every service releases its items once per second (`discipline: window`) and
processes them with several workers, so the measured latencies consist mostly of the wait for the release and do not
come from a FIFO single-server queue. The processing times do not depend on that discipline. The emulation therefore
keeps what is real about the services (how the processing time depends on the configuration, and its distribution)
and replaces the queueing by the FIFO model that the bounds describe. Its latencies are not testbed measurements. The
services of the second campaign queue FIFO, so its measured latencies are used as they are.

Held-out runs (training_exclusion): in the first campaign, the methods are trained on a service measured alone or
inside pipelines whose topology differs from the chain that is evaluated; no run of the evaluated topology is used for
training. The second campaign has one topology per application, and every service occurs only in that topology, so
there the methods are trained on the runs of the other experiments of the campaign that contain the service (e.g. the
runs with one to four workers for the chain with one worker); no run of the evaluated experiment is used for training.

Folder layout: `<experiment>/<start time>/dataset.csv` has one row per run and node with the configuration and summary
statistics; `runs/<run>/trace_steps.csv` (pipelines) has one row per request and node; `runs/<run>/<node>/records.csv`
has one row per item of that node. Runs that the campaign marked as invalid (overloaded, drain timeout) are left out.
Units: the campaign stores milliseconds and nanoseconds; every function here returns seconds.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ChainData
from .simulation import fifo_chain_latencies, quantile_with_lower_bound, sample_interarrival_times

CAMPAIGN = Path(__file__).resolve().parents[3] / "statics" / "results_milos"
CAMPAIGN_V2 = CAMPAIGN.parent / "results_milos_v2"
POOL_COLUMNS = ("run_id", "experiment", "node", "alone", "mean_s", "spread_s")   # columns of training_pool that are not knobs
NORMAL_P95 = 1.6449                                                              # 95th percentile of the standard normal distribution


# ------------------------------------------------------------------------------
# Reading the campaign
# ------------------------------------------------------------------------------
def campaign_of(experiment: str) -> Path:
    """The campaign folder that holds `experiment`: CAMPAIGN_V2 if it has a folder of that name, else CAMPAIGN."""
    return CAMPAIGN_V2 if (CAMPAIGN_V2 / experiment).is_dir() else CAMPAIGN


def experiment_folder(experiment: str) -> Path:
    """Folder of the one execution of `experiment` (e.g. "04e-chain-qr-cv-face" or "b1-vision-1-single")."""
    folders = sorted(path for path in (campaign_of(experiment) / experiment).iterdir() if path.is_dir())
    assert len(folders) == 1, f"{experiment}: expected one execution, found {len(folders)}"
    return folders[0]


@lru_cache(maxsize=None)
def run_table(experiment: str) -> pd.DataFrame:
    """The valid rows of the experiment's dataset.csv: one row per run and node, with the configuration of every node
    (columns "<node>.<knob>") and the summary statistics of that node in that run."""
    table = pd.read_csv(experiment_folder(experiment) / "dataset.csv")
    return table[table["valid"]].reset_index(drop=True)


def node_configurations(table: pd.DataFrame, node: str) -> pd.DataFrame:
    """Configuration of `node` in every run of `table`, one row per run (index: run_id) and one column per knob
    (cores, parallelism and the knobs of the service), without the prefix "<node>.". Left out: the service name, the
    memory limit and the queueing discipline (second campaign only), which are the same in every run, and "rps", the
    arrival rate of a source node in a tree, which is a property of the load and not of the service."""
    runs = table.drop_duplicates("run_id").set_index("run_id")
    columns = [column for column in runs.columns
               if column.startswith(node + ".") and not column.endswith((".service", ".memory", ".discipline", ".rps"))]
    return runs[columns].rename(columns=lambda column: column[len(node) + 1:])


@lru_cache(maxsize=None)
def campaign_nodes(campaign: Path = CAMPAIGN) -> pd.DataFrame:
    """One row per experiment of `campaign` and node: experiment, pipeline (the topology as the campaign writes it,
    e.g. "qr->cv cv->face"; the node name for a service measured alone), alone (the experiment measures one service
    alone), node and service. The folders "campaigns" and "results" of the first campaign hold no experiments."""
    rows = []
    for folder in sorted(path for path in campaign.iterdir() if path.is_dir() and path.name not in ("campaigns", "results")):
        table = run_table(folder.name)
        nodes = table.drop_duplicates("node")
        rows += [dict(experiment=folder.name, pipeline=table["pipeline"].iloc[0], alone="->" not in table["pipeline"].iloc[0], node=node, service=service)
                 for node, service in zip(nodes["node"], nodes["service"])]
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def pipeline_steps(experiment: str, run_id: str) -> pd.DataFrame:
    """The completed steps of one run of a pipeline, from trace_steps.csv: one row per request and node visit, in the
    order of arrival at the node, with the measured processing time in seconds (column "processing_s")."""
    steps = pd.read_csv(experiment_folder(experiment) / "runs" / run_id / "trace_steps.csv")
    steps = steps[steps["outcome"] == "completed"].sort_values("arrived_at_ns")
    return steps.assign(processing_s=steps["processing_ms"] / 1e3)


def chain_execution_times(experiment: str, run_id: str, path: list) -> np.ndarray:
    """Measured processing time (s) of every request at every node of `path` in one run of a chain, (requests, nodes),
    in the order in which the requests entered the chain. Only requests that completed at every node are kept."""
    steps = pipeline_steps(experiment, run_id)
    order = steps[steps["node"] == path[0]]["request_id"]
    return steps.pivot(index="request_id", columns="node", values="processing_s").loc[order, path].dropna().to_numpy()


def single_execution_times(experiment: str, run_id: str, node: str):
    """Measured processing time (s) of every item of a single-service run, in the order of arrival, from records.csv
    (finished - started, measurement phase, completed items). None if the copy of the campaign has no records for
    that run."""
    file = experiment_folder(experiment) / "runs" / run_id / node / "records.csv"
    if not file.exists() or file.stat().st_size == 0:
        return None
    records = pd.read_csv(file)
    records = records[(records["admission_phase"] == "measure") & (records["outcome"] == "completed")].sort_values("arrived_at_ns")
    return ((records["finished_at_ns"] - records["started_at_ns"]) / 1e9).to_numpy()


# ------------------------------------------------------------------------------
# Training runs of a service: alone, or in pipelines of another topology
# ------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def training_pool(service: str, exclude_pipeline: str = None, sources: tuple = ("alone", "pipeline"), campaign: Path = CAMPAIGN,
                  exclude_experiment: str = None) -> pd.DataFrame:
    """Every run of `campaign` that can train a model of `service` (e.g. "cv-analyzer:v1") for a chain with the
    topology exclude_pipeline: runs that measured the service alone and have per-item records ("alone"), and runs of
    pipelines with any other topology that contain the service ("pipeline"). With exclude_experiment instead, only the
    runs of that experiment are left out (see training_exclusion). One row per run with experiment, run_id,
    node, alone, the knobs of the service, and two statistics of the processing time of the service in that run from
    dataset.csv: mean_s, its mean (s), and spread_s, the standard deviation (s) of a Gaussian with the run's median and
    95th percentile, (p95 - median) / 1.645, which is less sensitive to single extreme items than the sample standard
    deviation; a run whose p95 does not exceed its median gets a spread of one microsecond. Experiments that do not vary
    every knob of the service are left out, because the value of the missing knob is not stored."""
    nodes = campaign_nodes(campaign)
    nodes = nodes[(nodes["service"] == service) & (nodes["pipeline"] != exclude_pipeline) & (nodes["experiment"] != exclude_experiment)]
    knobs = max((set(node_configurations(run_table(row.experiment), row.node).columns) for row in nodes.itertuples()), key=len)
    parts = []
    for row in nodes.itertuples():
        if ("alone" if row.alone else "pipeline") not in sources:
            continue
        configurations = node_configurations(run_table(row.experiment), row.node)
        if set(configurations.columns) != knobs:
            continue
        if row.alone:   # only the runs whose per-item records are in the copy of the campaign
            folder = experiment_folder(row.experiment) / "runs"
            configurations = configurations[[(folder / run_id / row.node / "records.csv").exists() and (folder / run_id / row.node / "records.csv").stat().st_size > 0
                                             for run_id in configurations.index]]
        table = run_table(row.experiment)
        statistics = table[table["node"] == row.node].set_index("run_id").loc[configurations.index]
        spread = (statistics["processing_p95_ms"] - statistics["processing_p50_ms"]) / NORMAL_P95 / 1e3
        parts.append(configurations.reset_index().assign(experiment=row.experiment, node=row.node, alone=row.alone,
                                                         mean_s=statistics["processing_mean_ms"].to_numpy() / 1e3,
                                                         spread_s=np.maximum(spread.to_numpy(), 1e-6)))
    return pd.concat(parts, ignore_index=True)


def training_exclusion(experiment: str) -> dict:
    """The arguments of training_pool that hold out the chain `experiment`: in the first campaign every run with its
    topology, in the second campaign (one topology per application) every run of the experiment itself."""
    campaign = campaign_of(experiment)
    if campaign == CAMPAIGN_V2:
        return dict(campaign=campaign, exclude_experiment=experiment)
    return dict(campaign=campaign, exclude_pipeline=chain_path(experiment)[2])


def run_execution_times(run) -> np.ndarray:
    """Measured processing times (s) of the service in one run of training_pool (a row of that table)."""
    if run.alone:
        return single_execution_times(run.experiment, run.run_id, run.node)
    steps = pipeline_steps(run.experiment, run.run_id)
    return steps[steps["node"] == run.node]["processing_s"].to_numpy()


def feature_names(experiment: str) -> dict:
    """Names of the columns of x_train and x_test in real_chain_data, per node of the chain `experiment`: the knobs of
    the service, and one name "<knob>_<value>" per value of a knob that is a word."""
    path, service_of = chain_path(experiment)[:2]
    names = {}
    for node in path:
        pool = training_pool(service_of[node], **training_exclusion(experiment))
        names[node] = list(pd.get_dummies(pool[[column for column in pool.columns if column not in POOL_COLUMNS]], dtype=float).columns)
    return names


def configuration_grid(x_train: np.ndarray, names: list, knob_a: str, knob_b: str, n_grid: int = 30) -> tuple:
    """A grid over two knobs of a service for plotting a model: knob_a and knob_b vary between their smallest and
    largest value among the training configurations x_train (columns `names`), and every other knob is fixed at its
    most frequent training value. Returns the two (n_grid, n_grid) arrays of the grid, the configurations of all grid
    points (n_grid * n_grid, features), and a dictionary other knob -> its fixed value."""
    a, b = names.index(knob_a), names.index(knob_b)
    grid_a, grid_b = np.meshgrid(np.linspace(x_train[:, a].min(), x_train[:, a].max(), n_grid),
                                 np.linspace(x_train[:, b].min(), x_train[:, b].max(), n_grid))
    most_frequent = [values[np.argmax(counts)] for values, counts in (np.unique(column, return_counts=True) for column in x_train.T)]
    x_grid = np.tile(most_frequent, (n_grid * n_grid, 1)).astype(float)
    x_grid[:, a], x_grid[:, b] = grid_a.ravel(), grid_b.ravel()
    return grid_a, grid_b, x_grid, {name: value for k, (name, value) in enumerate(zip(names, most_frequent)) if k not in (a, b)}


def encode_configurations(training: pd.DataFrame, in_chain: pd.DataFrame) -> tuple:
    """Feature matrices of one service from its configurations in the training runs and in the evaluated chain, with
    the same columns in the same order: numeric knobs as they are, and one 0/1 column per value of a knob that is a
    word (e.g. the hash algorithm)."""
    assert set(training.columns) == set(in_chain.columns), f"knobs differ: {sorted(training.columns)} and {sorted(in_chain.columns)}"
    both = pd.get_dummies(pd.concat([training, in_chain[training.columns]], keys=["training", "in_chain"]), dtype=float).astype(float)
    return both.loc["training"].to_numpy(), both.loc["in_chain"].to_numpy()


# ------------------------------------------------------------------------------
# Emulation
# ------------------------------------------------------------------------------
def emulate_runs(execution: list, rates: np.ndarray, n_items: int, rng: np.random.Generator, process: str = "poisson",
                 warmup_share: float = 0.1, **arrival_options) -> tuple:
    """Latencies of chains of FIFO single-server services whose execution times are measured ones, for several runs at
    once. execution: one array (requests, nodes) of measured processing times (s) per run, see chain_execution_times; a
    single service is a chain with one node. rates: arrival rate (items/s) of every run, for the arrival process
    `process` (as in simulation.py; arrival_options: jitter, cv2).
    The n_items items of an emulated run draw their execution times from the measured requests of that run with
    replacement, a whole request at a time: the times of one request at the different nodes stay together, and
    consecutive items are independent. Returns per-node latencies (runs, items, nodes) and end-to-end latencies (runs,
    items), without the first warmup_share of the items; with return_arrivals also the times (s) at which the kept
    items entered the chain (runs, items)."""
    return_arrivals = arrival_options.pop("return_arrivals", False)
    drawn = np.stack([times[rng.integers(len(times), size=n_items)] for times in execution])
    gaps = sample_interarrival_times(process, np.asarray(rates, dtype=float)[:, None], (len(execution), n_items), rng, **arrival_options)
    arrivals = np.cumsum(gaps, axis=1)
    node_latencies, latencies = fifo_chain_latencies(arrivals, drawn)
    kept = slice(int(warmup_share * n_items), None)
    results = node_latencies[:, kept], latencies[:, kept]
    return (*results, arrivals[:, kept]) if return_arrivals else results


def chain_path(experiment: str) -> tuple:
    """Nodes of the chain `experiment` in the order of the chain, the service of every node, and the topology as the
    campaign writes it."""
    chain = run_table(experiment)
    first_run = chain[chain["run_id"] == chain["run_id"].iloc[0]]
    return list(first_run["node"]), dict(zip(first_run["node"], first_run["service"])), chain["pipeline"].iloc[0]


def chain_execution_moments(experiment: str) -> tuple:
    """Mean (s) and variance (s^2) of the measured processing time of every service in every valid run of the chain
    `experiment`, both (runs, services), in the order of the runs in real_chain_data. An emulated run draws its
    execution times from exactly these measurements, so these are the true moments of its execution times."""
    times = chain_run_execution_times(experiment)
    return np.array([run_times.mean(axis=0) for run_times in times]), np.array([run_times.var(axis=0) for run_times in times])


def chain_run_execution_times(experiment: str) -> list:
    """Measured processing times (s) of every valid run of the chain `experiment`, one array (requests, nodes) per run,
    in the order of the runs in real_chain_data."""
    path = chain_path(experiment)[0]
    return [chain_execution_times(experiment, run_id, path) for run_id in run_table(experiment)["run_id"].unique()]


# ------------------------------------------------------------------------------
# Measured latencies (second campaign): the services queue FIFO, so the latencies need no emulation
# ------------------------------------------------------------------------------
def measured_latencies(experiment: str, run_id: str, path: list) -> tuple:
    """Measured latencies of one run of a pipeline of the second campaign, in the order in which the requests entered
    the pipeline; only requests that completed at every node of `path` are kept. Returns the end-to-end latency (s)
    of every request (traces.csv, total_ms: from entering the first service until leaving the last one, including the
    handover between services), its latency (s) at every node of `path`, waiting plus processing (requests, nodes;
    trace_steps.csv), and the time (s) at which it arrived at every node, counted from the first request that entered
    the pipeline (requests, nodes)."""
    steps = pipeline_steps(experiment, run_id)
    traces = pd.read_csv(experiment_folder(experiment) / "runs" / run_id / "traces.csv")
    traces = traces[traces["completed"]].sort_values("entered_at_ns").set_index("request_id")
    node = steps.assign(latency_s=(steps["waiting_ms"] + steps["processing_ms"]) / 1e3).pivot(index="request_id", columns="node", values="latency_s")
    node = node[path].dropna()
    arrived = steps.pivot(index="request_id", columns="node", values="arrived_at_ns")[path]
    order = traces.index[traces.index.isin(node.index)]
    return (traces.loc[order, "total_ms"].to_numpy() / 1e3, node.loc[order].to_numpy(),
            (arrived.loc[order].to_numpy() - traces["entered_at_ns"].iloc[0]) / 1e9)


def chain_run_latencies(experiment: str) -> list:
    """measured_latencies of every valid run of the chain `experiment` (second campaign), in the order of the runs in
    real_chain_data."""
    path = chain_path(experiment)[0]
    return [measured_latencies(experiment, run_id, path) for run_id in run_table(experiment)["run_id"].unique()]


def arrival_rates(experiment: str) -> pd.Series:
    """The arrival rate (items/s) of the Poisson arrivals that the campaign set for every valid run of `experiment`
    (rps_on_first_service), indexed by run_id. In the second campaign it was chosen from a short probe so that the
    slowest service of the run is busy about half of the time."""
    return run_table(experiment).drop_duplicates("run_id").set_index("run_id")["rps_on_first_service"]


def measured_quantiles(samples: list, level: float, coverage_check: dict) -> tuple:
    """quantile_with_lower_bound for series of different lengths: the `level` quantile of every array of `samples`
    (items in time order) and the lower edge of its confidence interval, two arrays with one entry per array."""
    results = [quantile_with_lower_bound(np.asarray(series)[None], level, **coverage_check) for series in samples]
    return np.array([estimate[0] for estimate, _ in results]), np.array([low[0] for _, low in results])


def emulate_services_alone(experiment: str, rates: np.ndarray, n_items: int, seed) -> np.ndarray:
    """Latencies (s) of every service of the chain `experiment` when it is alone, (runs, items, services): for every
    valid run, the processing times that the run measured at a service are replayed through one FIFO queue under
    Poisson arrivals at the arrival rate of that run (rates, items/s). Every service has its own arrivals."""
    times = chain_run_execution_times(experiment)
    rng = np.random.default_rng(seed)
    return np.stack([emulate_runs([run_times[:, [k]] for run_times in times], rates, n_items, rng)[1] for k in range(times[0].shape[1])], axis=-1)


def real_replay_data(experiment: str, seed, utilization: float = 0.5, n_train: int = 50, n_trace_items: int = 2000) -> tuple:
    """Training data of the replay baseline (ReplayComposition in baselines_chain.py) for the chain `experiment`, from
    the runs that training_exclusion allows; no run of the chain is used.
    A *training composition* puts one training run of every service of the chain together: the j-th of n_train runs
    that are drawn at random per service from training_pool. Its arrival rate is `utilization` / (mean processing time
    of its busiest service), as for the runs of the chain. Every service of a training composition is then monitored
    alone: its measured processing times are replayed through one FIFO queue under Poisson arrivals at that rate
    (n_trace_items items).
    Second campaign: every training run contains every service of the chain, so a training composition is one
    training run, drawn once for all services, and nothing is emulated: the monitored arrival times and latencies of a
    service are the ones measured at that service in that run (its first n_trace_items requests, or as many as the
    shortest of the drawn runs has), and the arrival rate is the one the campaign set (arrival_rates); `utilization` is
    not used.
    Returns x_train (training compositions, features): the configurations of all services next to each other and
    log10 of the arrival rate; the arrival times and the latencies (s) of the monitored items, each a dictionary
    service -> (training compositions, items); and x_test, the same features for every valid run of the chain."""
    rng = np.random.default_rng(seed)
    chain = run_table(experiment)
    path, service_of = chain_path(experiment)[:2]
    runs = list(chain["run_id"].unique())
    measured = campaign_of(experiment) == CAMPAIGN_V2
    pools = {node: training_pool(service_of[node], **training_exclusion(experiment)) for node in path}
    if measured:
        drawn = pools[path[0]].iloc[rng.choice(len(pools[path[0]]), size=n_train, replace=False)][["experiment", "run_id"]]
    features_train, features_test, times, chosen_runs = [], [], {}, {}
    for node in path:
        pool = pools[node]
        chosen = drawn.merge(pool, on=["experiment", "run_id"]) if measured else pool.iloc[rng.choice(len(pool), size=n_train, replace=False)]
        knobs = [column for column in pool.columns if column not in POOL_COLUMNS]
        x_train, x_test = encode_configurations(chosen[knobs], node_configurations(chain, node).loc[runs])
        features_train.append(x_train)
        features_test.append(x_test)
        chosen_runs[node] = chosen
        if not measured:
            times[node] = [run_execution_times(run) for run in chosen.itertuples()]
    arrivals, latencies = {}, {}
    if measured:
        rates = np.array([arrival_rates(run.experiment)[run.run_id] for run in drawn.itertuples()])
        test_rates = arrival_rates(experiment).loc[runs].to_numpy()
        monitored = {node: [measured_latencies(run.experiment, run.run_id, [run.node]) for run in chosen_runs[node].itertuples()] for node in path}
        n_items = min([n_trace_items] + [len(run[0]) for node in path for run in monitored[node]])
        for node in path:
            latencies[node] = np.stack([run[1][:n_items, 0] for run in monitored[node]])
            arrivals[node] = np.stack([run[2][:n_items, 0] for run in monitored[node]])
    else:
        rates = utilization / np.max([[run_times.mean() for run_times in times[node]] for node in path], axis=0)
        test_rates = utilization / np.array([run_times.mean(axis=0).max() for run_times in chain_run_execution_times(experiment)])
        for node in path:
            _, latencies[node], arrivals[node] = emulate_runs([run_times[:, None] for run_times in times[node]], rates, n_trace_items, rng,
                                                              return_arrivals=True)
    return (np.column_stack(features_train + [np.log10(rates)]), arrivals, latencies,
            np.column_stack(features_test + [np.log10(test_rates)]))


def chain_run_statistics(experiment: str) -> tuple:
    """Mean (s) and spread (s) of the processing time of every service in every valid run of the chain `experiment`,
    both (runs, services), from dataset.csv, in the order of the runs in real_chain_data. These are the two training
    targets that a run of the chain gives for each of its services, defined as in training_pool."""
    chain = run_table(experiment)
    path, runs = chain_path(experiment)[0], list(chain["run_id"].unique())
    rows = [chain[chain["node"] == node].set_index("run_id").loc[runs] for node in path]
    mean = np.column_stack([row["processing_mean_ms"].to_numpy() for row in rows]) / 1e3
    spread = np.column_stack([(row["processing_p95_ms"] - row["processing_p50_ms"]).to_numpy() for row in rows]) / NORMAL_P95 / 1e3
    return mean, np.maximum(spread, 1e-6)


def chain_tail_ratios(experiment: str) -> np.ndarray:
    """How heavy the tail of the measured processing times is in every valid run of the chain `experiment`, in the
    order of the runs in real_chain_data: the 99th percentile / the median of the processing time, for the service of
    the run at which that ratio is largest. 1 would be a processing time without any scatter."""
    path = chain_path(experiment)[0]
    times = [chain_execution_times(experiment, run_id, path) for run_id in run_table(experiment)["run_id"].unique()]
    return np.array([np.max(np.quantile(run_times, 0.99, axis=0) / np.quantile(run_times, 0.5, axis=0)) for run_times in times])


def real_chain_data(experiment: str, level: float, coverage_check: dict, seed, utilization: float = 0.5, arrival_process: str = "poisson",
                    arrival_cv2: float = 1.0, n_train: int = 50, n_items: int = 5000, n_trace_items: int = 2000,
                    sources: tuple = ("alone", "pipeline"), train_seed=None, arrival_rate: float = None) -> ChainData:
    """Data of one chain of the campaign (`experiment`, e.g. "04e-chain-qr-cv-face") for the experiments of
    `2_evaluation.ipynb`. No run of the chain is used for training (training_exclusion): in the first campaign the chain is
    an unseen composition (no run with its topology is used), in the second campaign an unseen experiment.

    Ground truth, second campaign (per valid run of the chain): the services queue FIFO, so the ground truth is the
    quantile of the measured end-to-end latencies of the run (chain_run_latencies), at the Poisson arrival rate that the
    campaign set for the run (arrival_rates); `utilization`, n_items, arrival_process and arrival_rate are not used
    for it, and node_latencies is None, because the runs measured different numbers of requests.

    Ground truth, first campaign (per valid run of the chain): the processing times that the run measured for every request at every
    service are replayed through a chain of FIFO queues (emulate_runs, n_items items). Every run has its own arrival
    rate: `utilization` / (mean processing time of its busiest service), so that the busiest service of every run is
    busy for the share `utilization` of the time. The rates of the campaign are not used, because they were chosen for
    services that release their items once per second. `level` is the latency quantile and coverage_check the
    arguments of its confidence interval.

    arrival_rate (items/s): if given, every run and every monitored training run gets this one arrival rate instead,
    and `utilization` is not used; workload["rate"] is then that number. This is the workload of an optimization,
    where the configuration is chosen for a given load. A run of the chain whose busiest service is overloaded at
    that rate (rate * mean processing time >= 1) has no latency quantile: its measured quantile is inf. A training
    run at which the service would be busy 95% of the time or more is not monitored (its latencies are nan).

    Training (per service): n_train runs drawn at random from training_pool, i.e. from runs of the service alone and
    of pipelines with another topology, or in the second campaign from the other experiments that contain the service
    (`sources` restricts the pool to runs alone or in pipelines). The configuration of a run is
    one training configuration. Its target is the mean processing time of the run (y_train), not one item: a run
    measured many items, and their mean is a far less noisy observation of the mean execution time. The scatter of
    single items, which one item per run would have told the model, comes from the spread of the run instead
    (y_train_spread, see training_pool).

    For the baselines that need a service monitored under load, the processing times of every training run are replayed
    through one FIFO queue (n_trace_items items) at an arrival rate that gives the service a utilization drawn
    uniformly between a tenth of `utilization` and `utilization`: in the chain every service is at most as busy as the
    busiest one. Because the arrival rate differs from run to run, these baselines get log10(rate) as a further input
    (x_train_load, x_test_load). In the second campaign, the monitored latencies of a training run are the ones
    measured at the service in that run (its first n_trace_items requests, or as many as the shortest run has), at the
    arrival rate of that run.

    seed: seed of the ground truth and, unless train_seed is given, of the training data; with train_seed, another
    training set is drawn for the same ground truth. x_train and x_test are dictionaries service -> configurations,
    because every service has its own knobs and runs; workload["rate"] has one arrival rate per run of the chain."""
    rng_train, rng_test, rng_monitoring = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    if train_seed is not None:
        rng_train, rng_monitoring = (np.random.default_rng(s) for s in np.random.SeedSequence(train_seed).spawn(2))
    arrival_options = {"cv2": arrival_cv2} if arrival_process == "gamma" else {}
    chain = run_table(experiment)
    path, service_of = chain_path(experiment)[:2]

    # ground truth: every valid run of the chain at its own arrival rate
    runs = list(chain["run_id"].unique())
    measured_campaign = campaign_of(experiment) == CAMPAIGN_V2
    if measured_campaign:   # the services queue FIFO: the measured latencies are the ground truth
        assert arrival_rate is None and arrival_process == "poisson", "the second campaign measured Poisson arrivals at the rate of every run"
        rates = arrival_rates(experiment).loc[runs].to_numpy()
        measured, measured_low = measured_quantiles([run[0] for run in chain_run_latencies(experiment)], level, coverage_check)
        node_latencies = None   # the runs measured different numbers of requests; chain_run_latencies has them per run
    else:
        execution = [chain_execution_times(experiment, run_id, path) for run_id in runs]
        busiest_mean = np.array([times.mean(axis=0).max() for times in execution])
        rates = utilization / busiest_mean if arrival_rate is None else np.full(len(runs), float(arrival_rate))
        node_latencies, latencies = emulate_runs(execution, rates, n_items, rng_test, arrival_process, **arrival_options)
        measured, measured_low = quantile_with_lower_bound(latencies, level, **coverage_check)
        overloaded = rates * busiest_mean >= 1
        measured, measured_low = np.where(overloaded, np.inf, measured), np.where(overloaded, np.inf, measured_low)
    workload = dict(process=arrival_process, rate=rates if arrival_rate is None else float(arrival_rate), jitter=0.1, **arrival_options)

    # training data: every service from the runs that training_exclusion allows
    x_train, y_train, y_train_spread, x_test, x_train_load, x_test_load, monitored_latencies = {}, {}, {}, {}, {}, {}, {}
    for node in path:
        pool = training_pool(service_of[node], sources=tuple(sources), **training_exclusion(experiment))
        assert len(pool) >= n_train, f"{service_of[node]}: {len(pool)} training runs available, {n_train} needed"
        chosen = pool.iloc[np.sort(rng_train.choice(len(pool), size=n_train, replace=False))]
        knobs = [column for column in pool.columns if column not in POOL_COLUMNS]
        x_train[node], x_test[node] = encode_configurations(chosen[knobs], node_configurations(chain, node).loc[runs])
        times = [run_execution_times(run) for run in chosen.itertuples()]
        y_train[node], y_train_spread[node] = chosen["mean_s"].to_numpy(), chosen["spread_s"].to_numpy()
        if measured_campaign:   # the latencies of the service as measured in the training run, at the rate of that run
            monitored_rates = np.array([arrival_rates(run.experiment)[run.run_id] for run in chosen.itertuples()])
            monitored = [measured_latencies(run.experiment, run.run_id, [run.node])[1][:, 0] for run in chosen.itertuples()]
            monitored_latencies[node] = np.stack([series[:min(n_trace_items, *map(len, monitored))] for series in monitored])
        else:
            if arrival_rate is None:
                monitored_rates = np.array([rng_monitoring.uniform(0.1, 1.0) * utilization / run_times.mean() for run_times in times])
            else:
                monitored_rates = np.full(n_train, float(arrival_rate))
            monitored_latencies[node] = emulate_runs([run_times[:, None] for run_times in times], monitored_rates, n_trace_items, rng_monitoring,
                                                     arrival_process, **arrival_options)[1]
            monitored_latencies[node][monitored_rates * np.array([run_times.mean() for run_times in times]) >= 0.95] = np.nan
        x_train_load[node] = np.column_stack([x_train[node], np.log10(monitored_rates)])
        x_test_load[node] = np.column_stack([x_test[node], np.log10(rates)])
    return ChainData(path, workload, x_train, y_train, x_test, measured, measured_low, node_latencies, monitored_latencies=monitored_latencies,
                     y_train_spread=y_train_spread, x_train_load=x_train_load, x_test_load=x_test_load)
