"""Simulator of a chain of FIFO single-server services: the ground truth that the bounds of `bounds.py` are compared
with. It samples the gaps between arrivals and the execution times, runs the chain item by item, and measures a latency
quantile with a confidence interval. The arrival processes ("poisson", "periodic", "gamma") are those of `bounds.py`.
Units: times in seconds.
"""
import numpy as np
from scipy.stats import t as student_t


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
# Simulated chain of FIFO servers
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


def simulate_chain(mean_execution, noise_levels, process, rate, n_items, item_correlation, rng, jitter=0.1, warmup_share=0.1,
                   return_arrivals=False, cv2=1.0, tail=0.0):
    """Simulates a chain of FIFO single-server services, vectorized over configurations.
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

        # an item starts when it has arrived and the previous item has left
        leaving = np.empty_like(entering)
        previous_leaving = np.zeros(n_cfgs)
        for i in range(n_items):
            previous_leaving = np.maximum(entering[:, i], previous_leaving) + execution[:, i]
            leaving[:, i] = previous_leaving
        node_latencies.append(leaving - entering)
        entering = leaving

    kept_items = slice(int(warmup_share * n_items), None)
    results = np.stack(node_latencies, axis=-1)[:, kept_items], (entering - arrivals)[:, kept_items]
    return (*results, arrivals[:, kept_items]) if return_arrivals else results


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
