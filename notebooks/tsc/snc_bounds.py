"""Latency bounds for chains of FIFO single-server services, and a simulator of such chains.

Shared by `method_v2.ipynb` and `why_poisson_is_loose.ipynb`. Every function takes the arrival process explicitly:
`process` is "poisson" (exponential gaps with mean 1/rate) or "periodic" (gaps of (1/rate) * (1 + U(-jitter, jitter))).
Execution times are Gaussian with the given mean and variance (s, s^2); bounds are latencies in seconds that are
exceeded with probability at most `epsilon`. Vectorized functions take one mean/variance per configuration.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

THETA_GRID = np.logspace(-1, 5, 4000)   # MGF parameter theta (1/s) over which every bound is optimized


# ------------------------------------------------------------------------------
# Arrival processes
# ------------------------------------------------------------------------------
def sample_interarrival_times(process, rate, shape, rng, jitter=0.1):
    if process == "poisson":
        return rng.exponential(1.0 / rate, shape)
    if process == "periodic":
        assert 0 <= jitter < 1, "jitter must be in [0, 1)"
        return (1.0 / rate) * (1 + rng.uniform(-jitter, jitter, shape))
    raise ValueError(f"unknown arrival process {process!r}")


def log_interarrival_mgf(process, rate, jitter=0.1, theta=THETA_GRID):
    """ln E[exp(-theta T)] of one gap T between arrivals."""
    if process == "poisson":
        return np.log(rate / (rate + theta))
    if process == "periodic":
        shortest_gap = (1 - jitter) / rate
        if jitter == 0:
            return -theta * shortest_gap
        width = 2 * jitter / rate
        return -theta * shortest_gap + np.log(-np.expm1(-theta * width)) - np.log(theta * width)
    raise ValueError(f"unknown arrival process {process!r}")


# ------------------------------------------------------------------------------
# SNC bound (union bound over queue lengths + Chernoff)
# ------------------------------------------------------------------------------
def log_latency_moment(mean, var, process, rate, jitter=0.1, theta=THETA_GRID):
    """ln of M_X(theta) / (1 - rho(theta)), one row per configuration; inf where rho >= 1."""
    mean, var = np.atleast_1d(mean)[:, None], np.atleast_1d(var)[:, None]
    log_mgf = theta * mean + 0.5 * theta ** 2 * var
    log_rho = log_mgf + log_interarrival_mgf(process, rate, jitter, theta)
    stable = log_rho < 0
    moment = np.full(log_mgf.shape, np.inf)
    moment[stable] = log_mgf[stable] - np.log1p(-np.exp(log_rho[stable]))
    return moment


def latency_bound_from_moment(log_moment, epsilon, theta=THETA_GRID):
    """d(epsilon) = min over theta of (log moment - ln epsilon) / theta."""
    return np.min((log_moment - np.log(epsilon)) / theta, axis=-1)


def snc_latency_bound(mean, var, process, rate, epsilon, jitter=0.1):
    return latency_bound_from_moment(log_latency_moment(mean, var, process, rate, jitter), epsilon)


# ------------------------------------------------------------------------------
# SNC bound with propagated arrival models (node k is fed by the departures of node k-1)
# ------------------------------------------------------------------------------
def queue_log_moment(log_mgf_node, log_spacing):
    """ln sum_n M_X(theta)^(n+1) E[exp(-theta A_n)], where log_spacing[..., n] bounds ln E[exp(-theta A_n)].
    log_spacing is a minimum of functions linear in n, so the log-terms are concave in n: beyond the last n every term
    shrinks at least by the last ratio r, and the rest of the sum is at most term_last * r / (1 - r). inf if r >= 1."""
    n = np.arange(log_spacing.shape[-1])
    log_terms = (n + 1) * log_mgf_node[..., None] + log_spacing
    log_ratio = log_terms[..., -1] - log_terms[..., -2]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_tail = log_terms[..., -1] + log_ratio - np.log(-np.expm1(np.minimum(log_ratio, 0.0)))
    moment = np.logaddexp(logsumexp(log_terms, axis=-1), log_tail)
    moment[~(log_ratio < 0)] = np.inf
    return moment


def propagated_node_bounds(path_beliefs, process, rate, epsilon, jitter=0.1, n_max=200, chunk=5, theta=THETA_GRID):
    """Latency bound of every node on the path; one row per node, one column per configuration.
    path_beliefs: one (mean, var) pair per node. n_max: queue lengths summed explicitly (the rest by a geometric tail)."""
    n = np.arange(n_max + 1)
    external = n * log_interarrival_mgf(process, rate, jitter, theta)[:, None]
    n_cfgs = len(np.atleast_1d(path_beliefs[0][0]))
    bounds = np.empty((len(path_beliefs), n_cfgs))
    for start in range(0, n_cfgs, chunk):
        cfgs = slice(start, start + chunk)
        log_spacing = external[None]
        for k, (mean, var) in enumerate(path_beliefs):
            mean, var = np.atleast_1d(mean)[cfgs, None], np.atleast_1d(var)[cfgs, None]
            log_mgf = theta * mean + 0.5 * theta ** 2 * var
            log_mgf_negative = -theta * mean + 0.5 * theta ** 2 * var
            moment = queue_log_moment(log_mgf, log_spacing)
            bounds[k, cfgs] = latency_bound_from_moment(moment, epsilon, theta)
            # arrival model of the next node: service spacing or arrival spacing, whichever bound is smaller
            with np.errstate(invalid="ignore"):
                via_service = n * log_mgf_negative[..., None]
                via_arrivals = np.nan_to_num(log_spacing + log_mgf_negative[..., None] + moment[..., None], nan=np.inf)
            log_spacing = np.minimum(np.minimum(via_service, via_arrivals), 0.0)
    return bounds


# ------------------------------------------------------------------------------
# Martingale bound (Kingman; Pollaczek-Khinchine refinement for Poisson arrivals)
# ------------------------------------------------------------------------------
def martingale_theta(mean, var, process, rate, jitter=0.1, theta=THETA_GRID):
    """Largest grid theta below the first crossing of rho(theta) = 1; nan if there is none (overloaded node)."""
    log_rho = theta * mean[:, None] + 0.5 * theta ** 2 * var[:, None] + log_interarrival_mgf(process, rate, jitter, theta)
    crossing = np.where(np.any(log_rho >= 0, axis=1), np.argmax(log_rho >= 0, axis=1), len(theta))
    return np.where(crossing > 0, theta[np.maximum(crossing - 1, 0)], np.nan)


def waiting_time_tail(w, mean, sd, theta, process, rate):
    """Bound on P(Q > w) for the waiting time Q, w >= 0: Pollaczek-Khinchine refinement for Poisson arrivals,
    Kingman's exp(-theta w) otherwise."""
    w = np.maximum(w, 0.0)
    kingman = np.exp(-theta * w)
    if process != "poisson":
        return np.minimum(1.0, kingman)
    z_w, z_0 = (w - mean) / sd, -mean / sd
    beyond = sd * (norm.pdf(z_w) - z_w * norm.sf(z_w))                  # int_w^inf P(X > x) dx
    below = (norm.sf(z_w) - kingman * norm.sf(z_0)) / theta             # e^{-theta w} int_0^w P(X > x) e^{theta x} dx ...
    mass = norm.cdf((w - mean - theta * sd ** 2) / sd) - norm.cdf((-mean - theta * sd ** 2) / sd)
    with np.errstate(divide="ignore"):
        below = below + np.exp(-theta * w + theta * mean + 0.5 * theta ** 2 * sd ** 2 + np.log(np.maximum(mass, 0.0))) / theta
    return np.minimum(1.0, np.nan_to_num(rate * (beyond + below), nan=1.0))


def martingale_node_bound(mean, var, process, rate, epsilon, jitter=0.1, bins=4000):
    """Smallest d with P(Q + X > d) <= epsilon, per configuration; inf for overloaded nodes.
    E_X[P(Q > d - X)] uses `bins` equal-probability bins of X at their upper edges, the last bin counted as a violation,
    so the discretization can only make the bound larger."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    sd, theta = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter)
    upper_edges = mean[:, None] + sd[:, None] * norm.ppf(np.arange(1, bins) / bins)

    def violation(d):
        slack = d[:, None] - upper_edges
        tail = np.where(slack < 0, 1.0, waiting_time_tail(slack, mean[:, None], sd[:, None], theta[:, None], process, rate))
        return (tail.sum(axis=1) + 1.0) / bins

    low, high = mean.copy(), mean + 60.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        too_small = violation(middle) > epsilon
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    return np.where(np.isnan(theta), np.inf, high)


# ------------------------------------------------------------------------------
# Ground truth: simulated chain of FIFO servers
# ------------------------------------------------------------------------------
def simulate_chain(mean_execution, noise_levels, process, rate, n_items, item_correlation, rng, jitter=0.1, warmup_share=0.1):
    """Simulates a chain of FIFO single-server services, vectorized over configurations.
    mean_execution: (configurations, nodes) mean execution time of every node (s); noise_levels: std per node (s).
    The execution-time noise of an item is shared across nodes with correlation item_correlation.
    Returns per-node latencies (configurations, items, nodes) and end-to-end latencies (configurations, items);
    the first warmup_share of items is dropped."""
    mean_execution = np.atleast_2d(mean_execution)
    n_cfgs = len(mean_execution)
    arrivals = np.cumsum(sample_interarrival_times(process, rate, (n_cfgs, n_items), rng, jitter), axis=1)
    shared_size = rng.standard_normal((n_cfgs, n_items))

    entering = arrivals
    node_latencies = []
    for k, noise_level in enumerate(noise_levels):
        own_noise = rng.standard_normal((n_cfgs, n_items))
        z = item_correlation * shared_size + np.sqrt(1 - item_correlation ** 2) * own_noise
        execution = np.maximum(mean_execution[:, k][:, None] + noise_level * z, 0.0)

        # an item starts when it has arrived and the previous item has left
        leaving = np.empty_like(entering)
        previous_leaving = np.zeros(n_cfgs)
        for i in range(n_items):
            previous_leaving = np.maximum(entering[:, i], previous_leaving) + execution[:, i]
            leaving[:, i] = previous_leaving
        node_latencies.append(leaving - entering)
        entering = leaving

    kept_items = slice(int(warmup_share * n_items), None)
    return np.stack(node_latencies, axis=-1)[:, kept_items], (entering - arrivals)[:, kept_items]
