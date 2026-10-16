"""Simulator of a chain, or of a fork-join tree (trees.py), of FIFO services with one or more workers each: the ground truth that the bounds of
`bounds.py` are compared with. It samples the gaps between arrivals and the execution times, runs the chain item by item, and measures a latency
quantile with a confidence interval. The arrival processes ("poisson", "periodic", "gamma") are those of `bounds.py`.
Units: times in seconds.
"""
import numpy as np
from scipy.stats import t as student_t

from .trees import children_first


# ------------------------------------------------------------------------------
# Arrival processes
# ------------------------------------------------------------------------------
def sample_interarrival_times(process, rate, shape, rng, jitter=0.1, cv2=1.0):
    if process == "poisson":
        return rng.exponential(1.0 / rate, shape)
    if process == "periodic":
        assert 0 <= jitter < 1, "jitter must be in [0, 1)"
        return (1.0 / rate) * (1 + rng.uniform(-jitter, jitter, shape))
    if process == "gamma":
        assert cv2 > 0, "cv2 must be positive"
        return rng.gamma(1.0 / cv2, cv2 / rate, shape)   # shape 1 / cv2 and scale cv2 / rate give the mean 1 / rate
    raise ValueError(f"unknown arrival process {process!r}")


# ------------------------------------------------------------------------------
# Simulated chain of FIFO nodes with one or more workers
# ------------------------------------------------------------------------------
def execution_times(mean, noise_level, z, tail=0.0):
    """Execution times (s) of items from their standard normal draws z. mean: mean execution time (s); noise_level:
    std of the Gaussian noise of the service (s). tail = 0 gives mean + noise_level * z, a Gaussian execution time.
    tail > 0 multiplies the mean by the lognormal factor exp(tail * z - tail^2 / 2) before the Gaussian noise is added.
    That factor has mean 1, so the mean execution time is the same for every tail, and squared coefficient of variation
    exp(tail^2) - 1: with a larger tail most items are faster than the mean and a few are several times slower. `tail`
    is the shape parameter (the sigma) of the lognormal factor. The result is increasing in z and may be negative for
    tail = 0; the caller clips it."""
    if tail == 0:
        return mean + noise_level * z
    return mean * np.exp(tail * z - 0.5 * tail ** 2) + noise_level * z


def fifo_node_leaving(entering, execution, workers=1):
    """Times (s) at which the items leave one FIFO node with `workers` workers, vectorized over configurations.
    entering: (configurations, items) times at which the items reach the node, in any order (after a node with several
    workers, the items arrive out of order); execution: (configurations, items) their execution times at the node (s);
    workers: one number, or one per configuration. The node takes the items in the order in which they reach it and gives
    each to the worker that becomes free first; an item starts when it has arrived and that worker is free. With one
    worker, this is the single-server recursion: an item starts when it has arrived and the previous item has left."""
    entering, execution = np.asarray(entering, dtype=float), np.asarray(execution, dtype=float)
    n_cfgs, n_items = entering.shape
    workers = np.broadcast_to(np.asarray(workers, dtype=int), (n_cfgs,))
    order = np.argsort(entering, axis=1, kind="stable")
    arriving, executing = np.take_along_axis(entering, order, axis=1), np.take_along_axis(execution, order, axis=1)
    free = np.where(np.arange(workers.max())[None, :] < workers[:, None], 0.0, np.inf)   # time at which every worker is free
    rows, leaving_in_order = np.arange(n_cfgs), np.empty_like(arriving)
    for i in range(n_items):
        worker = np.argmin(free, axis=1)
        leaving_in_order[:, i] = np.maximum(arriving[:, i], free[rows, worker]) + executing[:, i]
        free[rows, worker] = leaving_in_order[:, i]
    leaving = np.empty_like(leaving_in_order)
    np.put_along_axis(leaving, order, leaving_in_order, axis=1)
    return leaving


def simulate_chain(mean_execution, noise_levels, process, rate, n_items, item_correlation, rng, jitter=0.1, warmup_share=0.1,
                   return_arrivals=False, cv2=1.0, tail=0.0, workers=None):
    """Simulates a chain of FIFO services, vectorized over configurations; every node has `workers` workers
    ((configurations, nodes), default one each, see fifo_node_leaving).
    mean_execution: (configurations, nodes) mean execution time of every node (s); noise_levels: std per node (s).
    The execution-time noise of an item is shared across nodes with correlation item_correlation; it is Gaussian for
    tail = 0 and has a heavier right tail at the same mean for tail > 0 (execution_times).
    Returns per-node latencies (configurations, items, nodes) and end-to-end latencies (configurations, items);
    the first warmup_share of items is dropped. With return_arrivals, the times (s) at which the kept items entered the
    chain (configurations, items) are returned as a third value."""
    mean_execution = np.atleast_2d(mean_execution)
    n_cfgs = len(mean_execution)
    arrivals = np.cumsum(sample_interarrival_times(process, rate, (n_cfgs, n_items), rng, jitter, cv2), axis=1)
    shared_size = rng.standard_normal((n_cfgs, n_items))

    entering = arrivals
    node_latencies = []
    for k, noise_level in enumerate(noise_levels):
        own_noise = rng.standard_normal((n_cfgs, n_items))
        z = item_correlation * shared_size + np.sqrt(1 - item_correlation ** 2) * own_noise
        execution = np.maximum(execution_times(mean_execution[:, k][:, None], noise_level, z, tail), 0.0)

        leaving = fifo_node_leaving(entering, execution, 1 if workers is None else workers[:, k])
        node_latencies.append(leaving - entering)
        entering = leaving

    kept_items = slice(int(warmup_share * n_items), None)
    results = np.stack(node_latencies, axis=-1)[:, kept_items], (entering - arrivals)[:, kept_items]
    return (*results, arrivals[:, kept_items]) if return_arrivals else results


def fifo_chain_latencies(arrivals, execution, workers=None):
    """Latencies of a chain of FIFO services for given arrival and execution times, vectorized over configurations.
    arrivals: (configurations, items) times at which the items enter the chain (s, increasing); execution:
    (configurations, items, nodes) execution time of every item at every node (s), e.g. measured ones; workers:
    (configurations, nodes) workers of every node, default one each (fifo_node_leaving).
    Returns per-node latencies (configurations, items, nodes) and end-to-end latencies (configurations, items)."""
    entering = np.asarray(arrivals, dtype=float)
    node_latencies = []
    for k in range(execution.shape[-1]):
        leaving = fifo_node_leaving(entering, execution[:, :, k], 1 if workers is None else np.asarray(workers)[:, k])
        node_latencies.append(leaving - entering)
        entering = leaving
    return np.stack(node_latencies, axis=-1), entering - arrivals


def simulate_tree(mean_execution, noise_levels, children, process, rate, n_items, item_correlation, rng, jitter=0.1, warmup_share=0.1,
                  workers=None, node_level=None, return_execution=False):
    """Simulates a fork-join tree of FIFO services (trees.py), vectorized over configurations. A frame enters every
    leaf at its arrival time; an aggregator receives the frame when the later of its two children releases it, and
    processes the frames in that order. mean_execution: (configurations, services) mean execution time (s);
    noise_levels: std per service (s); rate: one arrival rate or one per configuration; workers: (configurations,
    services), default one each. The execution-time noise of a frame is shared across services with correlation
    item_correlation, as in simulate_chain.
    Returns the end-to-end latencies (configurations, frames), the first warmup_share of frames dropped. With node_level,
    also the node_level quantile of every service's latency (release minus the time the frame was complete at the
    service), (configurations, services). With return_execution, also the arrival times (configurations, all frames)
    and execution times (configurations, all frames, services), e.g. to replay every path as a chain."""
    mean_execution = np.atleast_2d(mean_execution)
    n_cfgs, n_services = mean_execution.shape
    workers = np.ones((n_cfgs, n_services), dtype=int) if workers is None else workers
    rate_column = np.asarray(rate, dtype=float)[:, None] if np.ndim(rate) else rate
    arrivals = np.cumsum(sample_interarrival_times(process, 1.0, (n_cfgs, n_items), rng, jitter, 1.0) / rate_column, axis=1)
    shared_size = rng.standard_normal((n_cfgs, n_items))
    kept_items = slice(int(warmup_share * n_items), None)
    leaving, node_quantiles, executions = [None] * n_services, np.empty((n_cfgs, n_services)), [None] * n_services
    for position in children_first(children):
        z = item_correlation * shared_size + np.sqrt(1 - item_correlation ** 2) * rng.standard_normal((n_cfgs, n_items))
        execution = np.maximum(mean_execution[:, position][:, None] + noise_levels[position] * z, 0.0)
        entering = arrivals if not children[position] else np.maximum.reduce([leaving[child] for child in children[position]])
        leaving[position] = fifo_node_leaving(entering, execution, workers[:, position])
        if node_level is not None:
            node_quantiles[:, position] = np.quantile((leaving[position] - entering)[:, kept_items], node_level, axis=1)
        if return_execution:
            executions[position] = execution
        for child in children[position]:
            leaving[child] = None   # only the parent needs it; frees memory in large trees
    results = ((leaving[0] - arrivals)[:, kept_items],)
    if node_level is not None:
        results += (node_quantiles,)
    if return_execution:
        results += (arrivals, np.stack(executions, axis=-1))
    return results if len(results) > 1 else results[0]


# ------------------------------------------------------------------------------
# Measured quantile
# ------------------------------------------------------------------------------
def quantile_with_lower_bound(samples, level, n_batches=10, confidence=0.95):
    """Quantile of every row of `samples` (configurations, items in time order) and the lower edge of its two-sided
    confidence interval at `confidence`, from batch means: the items are split into n_batches consecutive blocks, the
    quantile is computed per block, and the standard error is the spread of the block quantiles / sqrt(n_batches).
    Consecutive latencies in a queue are correlated, so the blocks must be long: fewer, longer blocks keep the interval
    valid near saturation, where busy periods are long. A bound counts as covering a configuration unless it lies below
    the lower edge, i.e. unless the simulation shows that it is below the true quantile."""
    estimate = np.quantile(samples, level, axis=-1)
    block_quantiles = np.stack([np.quantile(block, level, axis=-1) for block in np.array_split(samples, n_batches, axis=-1)], axis=-1)
    standard_error = block_quantiles.std(axis=-1, ddof=1) / np.sqrt(n_batches)
    return estimate, estimate - student_t.ppf(0.5 + confidence / 2, n_batches - 1) * standard_error
