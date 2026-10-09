"""Data of the experiments in `evaluation.ipynb`, from one of two sources (`data_source` in the notebook):

- "synthetic": services whose mean execution time is a known function of the configuration (CPU, quality,
  parallelism), and a simulated chain of FIFO single-server queues as ground truth. A seeded generator provides as
  many service types as a chain needs, so that no chain repeats a service type.
- "real": the testbed campaign in `statics/milos_results/` (not connected yet).

Both sources return a `ChainData`, so that the experiments do not depend on where the data comes from.
Units: execution times and latencies in seconds.
"""
from dataclasses import dataclass

import numpy as np

from snc_bounds import quantile_with_lower_bound, simulate_chain

# Synthetic service types. The mean execution time (s) of a service at the configuration (CPU, quality, parallelism) is
# base + cpu / CPU + quality * quality_knob ** power; noise is the standard deviation of one execution time (s).
# Every service type is drawn from these ranges, which span the three hand-made services of method_v2.ipynb. The slowest
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
    workload: dict                  # arrival process of the chain: process, rate (items/s), jitter
    x_train: np.ndarray             # (training configurations, features)
    y_train: dict                   # service -> one measured execution time per training configuration
    x_test: np.ndarray              # (held-out configurations, features)
    measured: np.ndarray            # end-to-end latency quantile of every held-out configuration (ground truth)
    measured_low: np.ndarray        # lower edge of the confidence interval of `measured`
    node_latencies: np.ndarray = None         # (held-out configurations, items, positions) latency per service
    trace_node_latencies: np.ndarray = None   # the same, traced on the chain at the training configurations
    monitored_arrivals: dict = None           # service -> (training configurations, items) arrival times, service alone
    monitored_latencies: dict = None          # service -> (training configurations, items) latencies, service alone
    x_extra: np.ndarray = None                # (extra configurations, features): data beyond the training budget
    extra_monitored_latencies: dict = None    # service -> (extra configurations, items) latencies, service alone


def true_demand(x: np.ndarray, service: dict) -> np.ndarray:
    """Mean execution time (s) of a synthetic service (parameters from synthetic_service) at the configurations
    x = (CPU, quality, parallelism)."""
    return service["base"] + service["cpu"] / x[:, 0] + service["quality"] * x[:, 1] ** service["power"]


def sample_configurations(n: int, rng: np.random.Generator) -> np.ndarray:
    """n configurations drawn uniformly from the range of the synthetic system: CPU, quality, parallelism."""
    return np.column_stack([rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n), rng.uniform(1.0, 4.0, n)])


def _simulate(x_cfgs, path, workload, n_items, item_correlation, rng, service_seed, **options):
    services = [synthetic_service(s, service_seed) for s in path]
    mean_execution = np.column_stack([true_demand(x_cfgs, service) for service in services])
    return simulate_chain(mean_execution, [service["noise"] for service in services], n_items=n_items,
                          item_correlation=item_correlation, rng=rng, **workload, **options)


def _execution_times(x_train, services, rng, service_seed):
    """One noisy execution time per service and training configuration."""
    specs = {s: synthetic_service(s, service_seed) for s in services}
    return {s: true_demand(x_train, specs[s]) + rng.normal(0, specs[s]["noise"], len(x_train)) for s in services}


def synthetic_chain_data(path: list, level: float, coverage_check: dict, seed: int, arrival_process: str = "poisson",
                         arrival_rate: float = 4.0, arrival_jitter: float = 0.2, item_correlation: float = 0.7,
                         n_train: int = 50, n_test: int = 100, n_items: int = 5000, n_trace_items: int = 2000,
                         n_extra: int = 0, service_seed: int = 0) -> ChainData:
    """Simulated data of the chain `path` (service types "s1", "s2", ...; none may occur twice), with the defaults of
    `method_v2.ipynb`. service_seed: seed of the generated service types (synthetic_service).
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
    services = sorted(path, key=service_index)   # fixed order of the random draws: the data of a service is the same in every chain

    x_train = sample_configurations(n_train, rng_train)
    y_train = _execution_times(x_train, services, rng_train, service_seed)

    x_test = sample_configurations(n_test, rng_test)
    node_latencies, latencies = _simulate(x_test, path, workload, n_items, item_correlation, rng_test, service_seed)
    measured, measured_low = quantile_with_lower_bound(latencies, level, **coverage_check)

    trace_node_latencies, _ = _simulate(x_train, path, workload, n_trace_items, item_correlation, rng_traces, service_seed)
    monitored_arrivals, monitored_latencies = {}, {}
    for s in services:
        monitored, _, monitored_arrivals[s] = _simulate(x_train, [s], workload, n_trace_items, item_correlation,
                                                        rng_monitoring, service_seed, return_arrivals=True)
        monitored_latencies[s] = monitored[:, :, 0]

    x_extra, extra_monitored_latencies = None, None
    if n_extra > 0:
        x_extra = sample_configurations(n_extra, rng_extra)
        extra_monitored_latencies = {s: _simulate(x_extra, [s], workload, n_trace_items, item_correlation, rng_extra, service_seed)[0][:, :, 0]
                                     for s in services}

    return ChainData(path, workload, x_train, y_train, x_test, measured, measured_low, node_latencies,
                     trace_node_latencies, monitored_arrivals, monitored_latencies, x_extra, extra_monitored_latencies)


def synthetic_training_data(services: list, workload: dict, n_train: int, seed, item_correlation: float = 0.7,
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


def load_training_data(source: str, **options) -> tuple:
    """Single-service training data from `source` ("synthetic" or "real"); the options are those of the source's loader."""
    if source == "synthetic":
        return synthetic_training_data(**options)
    if source == "real":
        raise NotImplementedError("the real data source (statics/milos_results) is not connected yet")
    raise ValueError(f"unknown data source {source!r}")


def load_chain_data(source: str, **options) -> ChainData:
    """Data of one chain from `source` ("synthetic" or "real"); the options are those of the source's loader."""
    if source == "synthetic":
        return synthetic_chain_data(**options)
    if source == "real":
        raise NotImplementedError("the real data source (statics/milos_results) is not connected yet")
    raise ValueError(f"unknown data source {source!r}")
