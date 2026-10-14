"""Prototype: every service is a FIFO pool of c workers instead of one FIFO server. c is the knob `parallelism` of the
service. Today's modules (`simulation.py`, `bounds.py`, `real_data.py`) are not changed; this module adds the pool
next to them so that both models can be compared on the same data.

Three parts:
- Ground truth: a chain of FIFO pools. An item starts as soon as it has arrived and any worker of the pool is free.
  Items can overtake each other inside a pool, so every node receives the items in the order in which the node before
  released them.
- Data: the testbed campaign replayed through the pools (as `real_chain_data`, but with pools), and four variants of
  the execution-time model (MODELS): today's GP, and three GPs whose inputs leave out `parallelism` and use the CPU that
  one worker can use, cores / parallelism, instead of cores. Of these, one is trained only on runs with cores >
  parallelism, where the execution time should not depend on the number of workers; one on every run; and one on
  every run with the load per worker as a further input. The notebook `prototype_multi_server.ipynb` compares them.
- Bounds: a pool of c workers is replaced by c single-server *lanes*; item i goes to lane i mod c. A lane receives every
  c-th item, so its gap between arrivals is the sum of c gaps of the arrival process, and E[exp(-theta T_lane)] =
  E[exp(-theta T)]^c. The pool is never slower than the lanes in the (1 - epsilon) latency quantile. For one node this
  is a classical comparison of FIFO multi-server queues with cyclic assignment. For chains it is only supported by
  simulation, not proved. Per item the comparison does not hold: a lane can let an item start before an item that
  arrived earlier and waits in another lane.
Units: times in seconds, rates in items/s.
"""
import numpy as np
import pandas as pd

from .baselines import union_bound_oracle
from .bounds import THETA_GRID, latency_bound_from_moment, log_interarrival_mgf, martingale_closed_form_node_bound
from .real_data import (POOL_COLUMNS, chain_execution_times, chain_path, encode_configurations, node_configurations, run_table,
                        training_pool)
from .service_gp import run_statistics_belief, train_run_statistics_models
from .simulation import quantile_with_lower_bound, sample_interarrival_times


# ------------------------------------------------------------------------------
# Ground truth: a chain of FIFO pools
# ------------------------------------------------------------------------------
def fifo_pool_chain_latencies(arrivals: np.ndarray, execution: np.ndarray, servers: np.ndarray) -> tuple:
    """Latencies of a chain of FIFO pools, vectorized over configurations. arrivals: (configurations, items) times at
    which the items enter the chain (s, increasing); execution: (configurations, items, nodes) execution time of every
    item at every node (s); servers: (configurations, nodes) number of workers of every node (integers >= 1).
    A node takes the items in the order in which they reach it, and an item starts at the earliest time at which it has
    arrived and a worker is free. A configuration with fewer workers than the largest one gets workers that are never
    free (free from +inf). With one worker per node the result equals fifo_chain_latencies in simulation.py.
    Returns per-node latencies (configurations, items, nodes) and end-to-end latencies (configurations, items), both
    indexed by the order in which the items entered the chain."""
    arrivals = np.asarray(arrivals, dtype=float)
    servers = np.broadcast_to(np.asarray(servers, dtype=int), (len(arrivals), execution.shape[-1]))
    rows = np.arange(len(arrivals))
    entering, node_latencies = arrivals, []
    for k in range(execution.shape[-1]):
        free = np.where(np.arange(servers[:, k].max())[None, :] < servers[:, k][:, None], 0.0, np.inf)   # time from which every worker is free
        order = np.argsort(entering, axis=1, kind="stable")
        leaving = np.empty_like(entering)
        for i in range(entering.shape[1]):
            item = order[:, i]
            worker = np.argmin(free, axis=1)
            free[rows, worker] = np.maximum(entering[rows, item], free[rows, worker]) + execution[rows, item, k]
            leaving[rows, item] = free[rows, worker]
        node_latencies.append(leaving - entering)
        entering = leaving
    return np.stack(node_latencies, axis=-1), entering - arrivals


def emulate_pool_runs(execution: list, servers: np.ndarray, rates: np.ndarray, n_items: int, rng: np.random.Generator,
                      process: str = "poisson", warmup_share: float = 0.1, **arrival_options) -> tuple:
    """emulate_runs of real_data.py with pools: the measured processing times of every run (one array (requests, nodes)
    per run) are replayed through a chain of FIFO pools with servers (runs, nodes) workers. The random draws are the same
    as in emulate_runs for the same generator state, so both emulations see the same requests and the same arrival
    pattern (scaled by the rates). Returns per-node latencies (runs, items, nodes) and end-to-end latencies (runs,
    items) without the first warmup_share of the items."""
    drawn = np.stack([times[rng.integers(len(times), size=n_items)] for times in execution])
    gaps = sample_interarrival_times(process, np.asarray(rates, dtype=float)[:, None], (len(execution), n_items), rng, **arrival_options)
    node_latencies, latencies = fifo_pool_chain_latencies(np.cumsum(gaps, axis=1), drawn, servers)
    kept = slice(int(warmup_share * n_items), None)
    return node_latencies[:, kept], latencies[:, kept]


# ------------------------------------------------------------------------------
# Bounds: a pool of c workers as c single-server lanes
# ------------------------------------------------------------------------------
def lane_martingale_node_bounds(path_beliefs: list, path_lower_means: list, path_servers: np.ndarray, process: str, rate, epsilon: float,
                                jitter: float = 0.1, cv2: float = 1.0) -> np.ndarray:
    """Closed-form martingale bound (martingale_closed_form_node_bound) of every node of a chain of pools, one row per
    node and one column per configuration; path_servers: (configurations, nodes) workers per node.
    A lane of a pool with c workers receives every c-th item. For Poisson arrivals at rate lambda, the sum of c
    exponential gaps is a gamma gap with shape c and mean c / lambda, i.e. the gamma process with rate lambda / c and
    squared coefficient of variation 1 / c; for gamma arrivals with cv2, the lane has rate lambda / c and cv2 / c. So
    the existing bound is called per value of c with that gamma process.
    Spacing by the node before (the second exponential tail of martingale_closed_form_node_bound): a node with one
    worker releases its items at least one execution time apart, so a lane of the next node with c workers receives its
    items at least c execution times of the node before apart; the walk then has increments X_k - (sum of c execution
    times of node k-1), i.e. the node before enters with c times its optimistic mean and c times its variance. A node
    with several workers can release items at almost the same time, so after it no spacing is used."""
    if process not in ("poisson", "gamma"):
        raise ValueError(f"the lane bound supports poisson and gamma arrivals, not {process!r}")
    base_cv2 = 1.0 if process == "poisson" else cv2
    path_servers = np.asarray(path_servers, dtype=int)
    rate = np.broadcast_to(np.asarray(rate, dtype=float), path_servers[:, 0].shape)
    bounds = np.empty((len(path_beliefs), len(path_servers)))
    for k, (mean, var) in enumerate(path_beliefs):
        mean, var = np.atleast_1d(mean), np.atleast_1d(var)
        for c in np.unique(path_servers[:, k]):
            group = path_servers[:, k] == c
            previous_mean = previous_var = None
            if k > 0:
                previous_single = path_servers[group, k - 1] == 1
                # spacing only where the node before has one worker; elsewhere a mean of -inf makes theta_s negative, i.e. unused
                previous_mean = np.where(previous_single, c * np.atleast_1d(path_lower_means[k - 1])[group], -np.inf)
                previous_var = c * np.atleast_1d(path_beliefs[k - 1][1])[group]
            with np.errstate(invalid="ignore"):
                bounds[k, group] = martingale_closed_form_node_bound(mean[group], var[group], "gamma", rate[group] / c, epsilon, jitter,
                                                                     previous_mean, previous_var, cv2=base_cv2 / c)
    return bounds


def lane_network_bound(path_beliefs: list, path_servers: np.ndarray, process: str, rate, epsilon: float, jitter: float = 0.1,
                       theta: np.ndarray = THETA_GRID, cv2: float = 1.0) -> np.ndarray:
    """network_service_bound with independent nodes (all Hoelder exponents 1) for a chain of pools, one value per
    configuration: node k with c_k workers enters with the lane transform E[exp(-theta T)]^(c_k). The network service
    curve writes the end-to-end latency as a maximum over paths through (item, node) of single-server queues; through
    lanes this holds when every node has the same number of workers, and is not proved when the numbers differ."""
    path_servers = np.asarray(path_servers, dtype=float)
    log_r = log_interarrival_mgf(process, rate, jitter, theta, cv2)
    total = 0.0
    for k, (mean, var) in enumerate(path_beliefs):
        log_mgf = theta * np.atleast_1d(mean)[:, None] + 0.5 * theta ** 2 * np.atleast_1d(var)[:, None]
        log_rho = log_mgf + path_servers[:, k][:, None] * log_r
        with np.errstate(divide="ignore", invalid="ignore"):
            total = total + np.where(log_rho < 0, log_mgf - np.log(-np.expm1(np.minimum(log_rho, 0.0))), np.inf)
    return latency_bound_from_moment(total, epsilon, theta)


# ------------------------------------------------------------------------------
# Ground truth on the campaign: its measured processing times replayed through pools
# ------------------------------------------------------------------------------
def campaign_rates(experiment: str) -> np.ndarray:
    """Arrival rate (items/s) that every valid run of the chain `experiment` received at its first node in the
    campaign (rate_received of dataset.csv), in the order of the runs in real_chain_data. The processing times of the
    run were measured at this load."""
    chain = run_table(experiment)
    first = chain_path(experiment)[0][0]
    return chain[chain["node"] == first].set_index("run_id").loc[list(chain["run_id"].unique()), "rate_received"].to_numpy(dtype=float)


def chain_servers(experiment: str) -> tuple:
    """Workers of every node in every valid run of the chain `experiment`, (runs, nodes), and per run whether every node
    has more cores than workers (cores > parallelism)."""
    chain = run_table(experiment)
    path, runs = chain_path(experiment)[0], list(chain["run_id"].unique())
    configurations = [node_configurations(chain, node).loc[runs] for node in path]
    servers = np.column_stack([cfg["parallelism"].to_numpy(dtype=int) for cfg in configurations])
    return servers, np.all([cfg["cores"].to_numpy() > cfg["parallelism"].to_numpy() for cfg in configurations], axis=0)


def utilization_rates(experiment: str, utilization: float) -> np.ndarray:
    """Arrival rate (items/s) of every valid run of the chain that puts its busiest pool at the share `utilization` of
    its capacity: utilization / max over nodes of (mean processing time / workers). With one worker per node this is the
    rate of real_chain_data."""
    path, servers = chain_path(experiment)[0], chain_servers(experiment)[0]
    execution = [chain_execution_times(experiment, run_id, path) for run_id in run_table(experiment)["run_id"].unique()]
    return utilization / np.array([np.max(times.mean(axis=0) / workers) for times, workers in zip(execution, servers)])


def pool_truth(experiment: str, rates: np.ndarray, level: float, coverage_check: dict, seed, epsilon: float, n_items: int = 5000) -> dict:
    """Ground truth of the chain `experiment` with pools, under Poisson arrivals at `rates` (one per valid run, items/s):
    the processing times that every run measured are replayed through FIFO pools with `parallelism` workers per node
    (emulate_pool_runs). The generator is the one real_chain_data uses for its ground truth under the same seed.
    Returns rates, servers (runs, nodes), follows_rule (per run: every node has more cores than workers), measured and
    measured_low (the `level` latency quantile and the lower edge of its confidence interval), and oracle: the
    measured node quantiles at epsilon / K, added (union_bound_oracle), the best any union bound over nodes can do.
    The per-item latencies are not kept, so that many chains and seeds fit into memory."""
    rng_test = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1])
    path = chain_path(experiment)[0]
    servers, follows_rule = chain_servers(experiment)
    execution = [chain_execution_times(experiment, run_id, path) for run_id in run_table(experiment)["run_id"].unique()]
    node_latencies, latencies = emulate_pool_runs(execution, servers, rates, n_items, rng_test)
    measured, measured_low = quantile_with_lower_bound(latencies, level, **coverage_check)
    return dict(rates=np.asarray(rates, dtype=float), servers=servers, follows_rule=follows_rule, measured=measured, measured_low=measured_low,
                oracle=union_bound_oracle(node_latencies, epsilon))


# ------------------------------------------------------------------------------
# Execution-time models: which runs train the GP, and which inputs it gets
# ------------------------------------------------------------------------------
# Every model is a RunStatisticsModel per service (a GP on the log of the run means and a GP on the log of the spread of
# single items). The models differ in their training runs and in their inputs:
# - "today": every run; the knobs as they are, with parallelism and cores (the model of real_chain_data);
# - "pool, more cores than workers": only runs with cores > parallelism; parallelism left out, cores per worker instead of cores;
# - "pool, all runs": every run; the inputs of the previous model;
# - "pool, all runs, load": every run; the inputs of the previous model and log10 of the load per worker (items/s per
#   worker), the arrival rate at which the run was measured divided by its workers. It lets the GP learn that the workers
#   of a pool slow each other down when they share too few cores and many items arrive.
MODELS = {"today": dict(pool_inputs=False, rule_only=False, load=False),
          "pool, more cores than workers": dict(pool_inputs=True, rule_only=True, load=False),
          "pool, all runs": dict(pool_inputs=True, rule_only=False, load=False),
          "pool, all runs, load": dict(pool_inputs=True, rule_only=False, load=True)}


def pool_knobs(configurations: pd.DataFrame) -> pd.DataFrame:
    """The GP inputs of the pool models from configurations with the knobs `parallelism` and `cores`: parallelism is
    left out (it only sets the number of workers), and cores becomes cores_per_worker = cores / parallelism, the CPU
    that one item can use."""
    knobs = configurations.drop(columns=["parallelism", "cores"])
    knobs.insert(0, "cores_per_worker", configurations["cores"] / configurations["parallelism"])
    return knobs


def model_inputs(configurations: pd.DataFrame, model: str, load_per_worker=None) -> pd.DataFrame:
    """The inputs of `model` (a key of MODELS) at the configurations; load_per_worker (items/s per worker, one per
    configuration) is used only by a model with a load input."""
    if not MODELS[model]["pool_inputs"]:
        return configurations
    inputs = pool_knobs(configurations)
    if MODELS[model]["load"]:
        inputs["log10_load_per_worker"] = np.log10(np.asarray(load_per_worker, dtype=float))
    return inputs


def training_rates(runs: pd.DataFrame) -> np.ndarray:
    """Arrival rate (items/s) at which every run of a training_pool table was measured at its node (rate_received)."""
    runs = runs.reset_index(drop=True)
    rates = np.full(len(runs), np.nan)
    for experiment, rows in runs.groupby("experiment"):
        table = run_table(experiment).set_index(["run_id", "node"])["rate_received"]
        rates[rows.index] = table.loc[list(zip(rows["run_id"], rows["node"]))].to_numpy(dtype=float)
    return rates


def pool_training(experiment: str, model: str, seed, n_train: int = 50) -> dict:
    """Training data of `model` for every service of the chain `experiment`, node -> dict with inputs (DataFrame of
    the n_train training runs), mean and spread (s, the two targets as in real_chain_data) and chain (the knobs of the
    node in every valid run of the chain, turned into inputs by chain_inputs). The runs come from training_pool, so no
    run with the chain's topology is used. The generator is the one real_chain_data uses for its training data under
    the same seed, so "today" draws the same runs as real_chain_data."""
    rng_train = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
    chain = run_table(experiment)
    path, service_of, pipeline = chain_path(experiment)
    runs = list(chain["run_id"].unique())
    training = {}
    for node in path:
        pool = training_pool(service_of[node], pipeline)
        if MODELS[model]["rule_only"]:
            pool = pool[pool["cores"] > pool["parallelism"]].reset_index(drop=True)
        chosen = pool.iloc[np.sort(rng_train.choice(len(pool), size=n_train, replace=False))].reset_index(drop=True)
        knobs = chosen[[column for column in pool.columns if column not in POOL_COLUMNS]]
        load = training_rates(chosen) / chosen["parallelism"].to_numpy() if MODELS[model]["load"] else None
        training[node] = dict(inputs=model_inputs(knobs, model, load), mean=chosen["mean_s"].to_numpy(), spread=chosen["spread_s"].to_numpy(),
                              chain=node_configurations(chain, node).loc[runs][knobs.columns])
    return training


def chain_inputs(training: dict, model: str, servers: np.ndarray, rates: np.ndarray) -> dict:
    """Encoded inputs of every node, node -> (x_train, x_test): the training runs, and the valid runs of the chain at
    the arrival rates `rates` (items/s, one per run) with `servers` (runs, nodes) workers; a model with a load input
    gets the load per worker, rates / workers."""
    return {node: encode_configurations(part["inputs"], model_inputs(part["chain"], model, np.asarray(rates) / servers[:, k]))
            for k, (node, part) in enumerate(training.items())}


def fit_models(training: dict, inputs: dict) -> dict:
    """One RunStatisticsModel per node (train_run_statistics_models), fitted on the encoded training inputs."""
    nodes = list(training)
    return train_run_statistics_models({node: inputs[node][0] for node in nodes}, {node: training[node]["mean"] for node in nodes},
                                       {node: training[node]["spread"] for node in nodes}, nodes)


def path_beliefs(models: dict, inputs: dict, confidence: float) -> tuple:
    """What the bounds take from the models at the runs of the chain, in the order of the chain: per node a pessimistic
    mean (s) with the variance of single items (s^2), the optimistic mean (s) that the spacing of the next node uses,
    and the predicted mean execution time (s) without margin."""
    beliefs = [run_statistics_belief(models[node], inputs[node][1], confidence) for node in models]
    lower = [run_statistics_belief(models[node], inputs[node][1], 1 - confidence)[0] for node in models]
    return beliefs, lower, [models[node].mean(inputs[node][1]) for node in models]


def load_response(model, part: dict, servers_k: np.ndarray, loads: np.ndarray) -> np.ndarray:
    """Predicted mean execution time (s) of one service by the model with a load input, at every configuration the
    service has in the runs of the chain (part: one node of pool_training; servers_k: its workers per run) and at every
    load per worker of `loads` (items/s per worker), (configurations, loads). It shows what the GP learned about load."""
    curves = []
    for load in loads:
        rates = load * servers_k   # the arrival rate that gives every configuration this load per worker
        x_test = encode_configurations(part["inputs"], model_inputs(part["chain"], "pool, all runs, load", rates / servers_k))[1]
        curves.append(model.mean(x_test))
    return np.column_stack(curves)


def model_beliefs(experiment: str, model: str, seed, truths: dict, confidence: float) -> dict:
    """Fits `model` (a key of MODELS) for the chain `experiment` and computes what the bounds need under every ground
    truth of `truths` (name -> pool_truth of this chain and seed): name -> (beliefs, lower means, predicted means), see
    path_beliefs. The fitted models do not depend on the arrival rate; a model with a load input gets the load per
    worker of each truth at its runs. Also returns the training data ("training") and the fitted models ("models")."""
    training = pool_training(experiment, model, seed)
    any_truth = next(iter(truths.values()))
    models = fit_models(training, chain_inputs(training, model, any_truth["servers"], any_truth["rates"]))
    result = {name: path_beliefs(models, chain_inputs(training, model, truth["servers"], truth["rates"]), confidence) for name, truth in truths.items()}
    return dict(result, training=training, models=models)


def case_statistics(bound: np.ndarray, truth: dict) -> dict:
    """Coverage and tightness of one bound (s, one value per run of the chain) against one ground truth: coverage (share
    of runs whose bound is not below the lower edge of the confidence interval of the measured quantile), the same on
    the runs that follow the rule (every node with more cores than workers) and on the others, tightness (median over
    the runs of bound / measured quantile; inf where half of the bounds or more are inf) and finite (share of runs with a
    finite bound)."""
    covered = bound >= truth["measured_low"]
    rule = truth["follows_rule"]
    return dict(coverage=covered.mean(), coverage_rule=covered[rule].mean() if rule.any() else np.nan,
                coverage_other=covered[~rule].mean() if (~rule).any() else np.nan,
                tightness=np.median(bound / truth["measured"]), finite=np.isfinite(bound).mean())
