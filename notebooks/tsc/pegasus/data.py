"""Data of the experiments in `2_evaluation.ipynb`, from one of two sources (`data_source` in the notebook):

- "synthetic": services whose mean execution time is a known function of the configuration (CPU, quality,
  parallelism), and a simulated chain of FIFO queues as ground truth, where the parallelism of a configuration is the
  number of workers of every service (1 to 4). A seeded generator provides as
  many service types as a chain needs, so that no chain repeats a service type.
- "real": the testbed campaign in `statics/results_milos/`, see `real_data.py`: measured processing times of real
  services, replayed through the same FIFO queues.

Both sources return a `ChainData`, so that the experiments do not depend on where the data comes from.
Units: execution times and latencies in seconds.
"""
from dataclasses import dataclass, replace

import numpy as np

from .service_gp import configurations_of
from .simulation import execution_times, quantile_with_lower_bound, simulate_chain

# Synthetic service types. The mean execution time (s) of a service at the configuration (CPU, quality, parallelism) is
# base + cpu / CPU + quality * quality_knob ** power; noise is the standard deviation of one execution time (s). CPU is
# the CPU of one worker, so the execution time of an item does not depend on the parallelism (the number of workers).
# Every service type is drawn from these ranges, which span the three hand-made services of 1_method_synthetic.ipynb. The slowest
# possible service takes 0.012 + 0.040 / 0.5 + 0.030 * 2 ** 2 = 0.212 s at the slowest configuration (CPU 0.5, quality 2).
GENERATED_RANGES = dict(base=(0.006, 0.012), cpu=(0.025, 0.040), quality=(0.015, 0.030), power=(1.5, 2.0), noise=(0.001, 0.006))


def service_index(name: str) -> int:
    """Number k of the service "s<k>"; services are kept in this order wherever the order matters."""
    return int(name[1:])


def synthetic_service(name: str, seed: int = 0) -> dict:
    """Parameters of the synthetic service type `name` ("s1", "s2", ...), drawn from GENERATED_RANGES. A service depends
    only on `seed` and on its own number, so "s7" is the same service in every chain that contains it."""
    rng = np.random.default_rng([seed, service_index(name)])
    return {parameter: rng.uniform(low, high) for parameter, (low, high) in GENERATED_RANGES.items()}


def synthetic_chain(depth: int) -> list:
    """Chain of `depth` different service types, s1 -> s2 -> ... ; no service type occurs twice."""
    return [f"s{k}" for k in range(1, depth + 1)]


@dataclass
class ChainData:
    """Everything an experiment needs about one chain. Fields that a source cannot provide are None; the methods that
    need them are then left out."""
    path: list                      # service at every position of the chain
    workload: dict                  # arrival process of the chain: process, rate (items/s; one number, or one per held-out
                                    # configuration), jitter; cv2 for gamma arrivals
    x_train: np.ndarray             # (training configurations, features); a dictionary service -> array if every service has its own
    y_train: dict                   # service -> one measured execution time per training configuration; real data: the mean of a run
    x_test: np.ndarray              # (held-out configurations, features); a dictionary like x_train
    measured: np.ndarray            # end-to-end latency quantile of every held-out configuration (ground truth)
    measured_low: np.ndarray        # lower edge of the confidence interval of `measured`
    node_latencies: np.ndarray = None         # (held-out configurations, items, positions) latency per service
    trace_node_latencies: np.ndarray = None   # the same, traced on the chain at the training configurations
    monitored_arrivals: dict = None           # service -> (training configurations, items) arrival times, service alone
    monitored_latencies: dict = None          # service -> (training configurations, items) latencies, service alone
    x_extra: np.ndarray = None                # (extra configurations, features): data beyond the training budget
    extra_monitored_latencies: dict = None    # service -> (extra configurations, items) latencies, service alone
    y_train_spread: dict = None               # real data: service -> standard deviation of single items in every training run (s)
    x_train_load: dict = None                 # inputs of the baselines monitored under load where they differ from x_train:
    x_test_load: dict = None                  # the configuration and the arrival rate, if the rate differs between configurations
    workers_column: object = None             # column of x_test that holds the number of workers of a service (an int, or a
                                              # dictionary service -> int or None like x_test); None: one worker


def true_demand(x: np.ndarray, service: dict) -> np.ndarray:
    """Mean execution time (s) of a synthetic service (parameters from synthetic_service) at the configurations
    x = (CPU, quality, parallelism)."""
    return service["base"] + service["cpu"] / x[:, 0] + service["quality"] * x[:, 1] ** service["power"]


def busiest_utilization(path: list, x: np.ndarray, rate: float, service_seed: int = 0) -> np.ndarray:
    """True utilization of the busiest service of the synthetic chain `path`, per configuration of x: the arrival rate
    (items/s) times the largest mean execution time (s) among the services, divided by the number of workers (the
    parallelism). At 1 or above, that service is overloaded."""
    return rate * np.max([true_demand(x, synthetic_service(s, service_seed)) for s in path], axis=0) / x[:, WORKERS_COLUMN]


def synthetic_execution_moments(path: list, x: np.ndarray, service_seed: int = 0) -> tuple:
    """True mean (s) and variance (s^2) of the execution time of every service of the synthetic chain `path` at the
    configurations x, both (configurations, services), for Gaussian execution times (execution_tail = 0): the mean is
    true_demand and the variance the squared noise level of the service."""
    services = [synthetic_service(s, service_seed) for s in path]
    mean = np.column_stack([true_demand(x, service) for service in services])
    return mean, np.broadcast_to([service["noise"] ** 2 for service in services], mean.shape).copy()

WORKERS_COLUMN = 2   # the parallelism, the number of workers of every service of a synthetic chain

# Range of CPU, quality and parallelism in the synthetic system; the parallelism is drawn from [1, 5) and rounded down
# to a whole number of workers, 1 to 4 (whole_workers)
CONFIGURATION_RANGES = ((0.5, 2.0), (0.5, 2.0), (1.0, 5.0))


def whole_workers(x: np.ndarray) -> np.ndarray:
    """x with its parallelism (third column) rounded down to a whole number of workers."""
    x = np.array(x, dtype=float)
    x[:, WORKERS_COLUMN] = np.floor(x[:, WORKERS_COLUMN])
    return x


def chain_workers(x: np.ndarray, depth: int) -> np.ndarray:
    """Workers of every service of a synthetic chain with `depth` services, (configurations, depth): every service has
    the parallelism of the configuration."""
    return np.repeat(x[:, WORKERS_COLUMN:WORKERS_COLUMN + 1].astype(int), depth, axis=1)


def path_workers(data: ChainData) -> list:
    """Workers of every position of the chain of `data` at its held-out configurations, one array per position (read
    from x_test, so that it follows any replacement of x_test); None if every service has one worker."""
    if data.workers_column is None:
        return None
    column = lambda s: data.workers_column[s] if isinstance(data.workers_column, dict) else data.workers_column
    n_cfgs = len(configurations_of(data.x_test, data.path[0]))
    return [np.ones(n_cfgs, dtype=int) if column(s) is None else configurations_of(data.x_test, s)[:, column(s)].astype(int) for s in data.path]


def sample_configurations(n: int, rng: np.random.Generator) -> np.ndarray:
    """n configurations drawn uniformly from the range of the synthetic system: CPU, quality, parallelism (whole
    workers)."""
    return whole_workers(np.column_stack([rng.uniform(low, high, n) for low, high in CONFIGURATION_RANGES]))


def _simulate(x_cfgs, path, workload, n_items, item_correlation, rng, service_seed, **options):
    services = [synthetic_service(s, service_seed) for s in path]
    mean_execution = np.column_stack([true_demand(x_cfgs, service) for service in services])
    return simulate_chain(mean_execution, [service["noise"] for service in services], n_items=n_items,
                          item_correlation=item_correlation, rng=rng, workers=chain_workers(x_cfgs, len(path)), **workload, **options)


def _execution_times(x_train, services, rng, service_seed, tail=0.0):
    """One noisy execution time per service and training configuration; tail: shape of the noise (execution_times)."""
    specs = {s: synthetic_service(s, service_seed) for s in services}
    return {s: execution_times(true_demand(x_train, specs[s]), specs[s]["noise"], rng.standard_normal(len(x_train)), tail)
            for s in services}


def synthetic_execution_times(x: np.ndarray, services: list, rng: np.random.Generator, service_seed: int = 0) -> dict:
    """One measured execution time (s) of every synthetic service at every configuration of x, service -> array: what
    a benchmark of these configurations gives (Gaussian execution times)."""
    return _execution_times(x, services, rng, service_seed)


def synthetic_chain_data(path: list, level: float, coverage_check: dict, seed: int, arrival_process: str = "poisson",
                         arrival_rate: float = 4.0, arrival_jitter: float = 0.2, item_correlation: float = 0.0,
                         n_train: int = 50, n_test: int = 100, n_items: int = 5000, n_trace_items: int = 2000,
                         n_extra: int = 0, service_seed: int = 0, arrival_cv2: float = 1.0, execution_tail: float = 0.0) -> ChainData:
    """Simulated data of the chain `path` (service types "s1", "s2", ...; none may occur twice), with the defaults of
    `1_method_synthetic.ipynb` except for item_correlation. service_seed: seed of the generated service types (synthetic_service).
    item_correlation: correlation of an item's execution-time noise between the services of the chain (simulate_chain);
    0 makes the execution times of an item independent between the services, which the SNC version of Pegasus assumes.
    arrival_cv2: squared coefficient of variation of the gaps between arrivals; only the arrival process "gamma" uses it.
    execution_tail: shape of the execution-time noise of every service (execution_times in simulation.py): 0 is
    Gaussian, a larger value gives a heavier right tail at the same mean execution time. It applies to the training
    data and to every simulated run.
    Training: n_train configurations with one execution time per service. Ground truth: the chain simulated with
    n_items items at n_test held-out configurations; `level` is the latency quantile and coverage_check the arguments
    of its confidence interval (quantile_with_lower_bound). The training configurations are also traced on the chain
    and monitored per service alone, with n_trace_items items each, for the baselines that need such data. n_extra
    further configurations are monitored per service alone in the same way; they are data beyond the training budget
    that every method shares, for baselines that are deliberately given more. Every part has its own random generator
    derived from `seed`, so that no part changes when another one does."""
    rng_train, rng_test, rng_traces, rng_monitoring, rng_extra = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5))
    assert len(set(path)) == len(path), "a chain must not repeat a service type"
    workload = dict(process=arrival_process, rate=arrival_rate, jitter=arrival_jitter)
    if arrival_process == "gamma":   # only the gamma process has this parameter, and only the functions that know the process take it
        workload["cv2"] = arrival_cv2
    services = sorted(path, key=service_index)   # fixed order of the random draws: the data of a service is the same in every chain

    x_train = sample_configurations(n_train, rng_train)
    y_train = _execution_times(x_train, services, rng_train, service_seed, execution_tail)

    x_test = sample_configurations(n_test, rng_test)
    node_latencies, latencies = _simulate(x_test, path, workload, n_items, item_correlation, rng_test, service_seed, tail=execution_tail)
    measured, measured_low = quantile_with_lower_bound(latencies, level, **coverage_check)

    trace_node_latencies, _ = _simulate(x_train, path, workload, n_trace_items, item_correlation, rng_traces, service_seed, tail=execution_tail)
    monitored_arrivals, monitored_latencies = {}, {}
    for s in services:
        monitored, _, monitored_arrivals[s] = _simulate(x_train, [s], workload, n_trace_items, item_correlation,
                                                        rng_monitoring, service_seed, return_arrivals=True, tail=execution_tail)
        monitored_latencies[s] = monitored[:, :, 0]

    x_extra, extra_monitored_latencies = None, None
    if n_extra > 0:
        x_extra = sample_configurations(n_extra, rng_extra)
        extra_monitored_latencies = {s: _simulate(x_extra, [s], workload, n_trace_items, item_correlation, rng_extra, service_seed,
                                                  tail=execution_tail)[0][:, :, 0]
                                     for s in services}

    return ChainData(path, workload, x_train, y_train, x_test, measured, measured_low, node_latencies,
                     trace_node_latencies, monitored_arrivals, monitored_latencies, x_extra, extra_monitored_latencies,
                     workers_column=WORKERS_COLUMN)


def synthetic_training_data(services: list, workload: dict, n_train: int, seed, item_correlation: float = 0.0,
                            n_trace_items: int = 2000, service_seed: int = 0) -> tuple:
    """A further set of single-service training data, independent of the one in synthetic_chain_data: n_train
    configurations with one execution time per service, and every service monitored alone under `workload` at these
    configurations with n_trace_items items. `seed` may be an int or a sequence of ints.
    Returns x_train, y_train (service -> execution times) and monitored_latencies (service -> (configurations, items))."""
    rng_train, rng_monitoring = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    services = sorted(services, key=service_index)
    x_train = sample_configurations(n_train, rng_train)
    y_train = _execution_times(x_train, services, rng_train, service_seed)
    monitored_latencies = {s: _simulate(x_train, [s], workload, n_trace_items, item_correlation, rng_monitoring, service_seed)[0][:, :, 0]
                           for s in services}
    return x_train, y_train, monitored_latencies


def load_chain_data(source: str, **options) -> ChainData:
    """Data of one chain from `source` ("synthetic" or "real"); the options are those of the source's loader
    (synthetic_chain_data, or real_chain_data in real_data.py)."""
    if source == "synthetic":
        return synthetic_chain_data(**options)
    if source == "real":
        from .real_data import real_chain_data   # imported here, because real_data.py uses ChainData
        return real_chain_data(**options)
    raise ValueError(f"unknown data source {source!r}")


def with_training_data(source: str, data: ChainData, n_train: int, seed, **options) -> ChainData:
    """`data` with another set of single-service training data of n_train configurations per service, drawn with
    `seed`; the chain, its held-out configurations and the ground truth stay as they are. options: those of the
    source's loader that the data was made with (synthetic: service_seed; real: everything passed to real_chain_data
    except the seed)."""
    if source == "synthetic":
        x_train, y_train, monitored_latencies = synthetic_training_data(services=data.path, workload=data.workload, n_train=n_train, seed=seed, **options)
        return replace(data, x_train=x_train, y_train=y_train, monitored_latencies=monitored_latencies)
    if source == "real":
        from .real_data import real_chain_data
        other = real_chain_data(n_train=n_train, seed=0, train_seed=seed, **options)   # only its training data is used
        return replace(data, x_train=other.x_train, y_train=other.y_train, y_train_spread=other.y_train_spread,
                       monitored_latencies=other.monitored_latencies, x_train_load=other.x_train_load)
    raise ValueError(f"unknown data source {source!r}")


def same_configurations(a, b) -> bool:
    """Whether two sets of configurations are the same; each is one array, or a dictionary service -> array."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(np.array_equal(a[service], b[service]) for service in a)
    return np.array_equal(a, b)


def assert_unseen(data: ChainData) -> None:
    """Checks that the held-out part of `data` is unseen by the training data. Synthetic data (one configuration array
    for all services): no held-out configuration occurs among the training configurations. Real data (one array per
    service): nothing to check here, because what is unseen is the composition, not the configuration of a service;
    training_pool in real_data.py leaves out every run with the topology of the chain."""
    if not isinstance(data.x_train, dict):
        assert not (data.x_test[:, None, :] == data.x_train[None, :, :]).all(axis=2).any(), "a held-out configuration occurs in the training data"
