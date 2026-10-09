"""Latency bounds for chains of FIFO single-server services, and a simulator of such chains.

Shared by `method_v2.ipynb` and `why_poisson_is_loose.ipynb`. Every function takes the arrival process explicitly:
`process` is "poisson" (exponential gaps with mean 1/rate) or "periodic" (gaps of (1/rate) * (1 + U(-jitter, jitter))).
Execution times are Gaussian with the given mean and variance (s, s^2); bounds are latencies in seconds that are
exceeded with probability at most `epsilon`. Vectorized functions take one mean/variance per configuration.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t

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


def propagated_node_moments(path_beliefs, process, rate, jitter=0.1, n_max=200, chunk=5, theta=THETA_GRID, path_lower_means=None):
    """ln of the latency moment sum_n M_X(theta)^(n+1) E[exp(-theta A_n)] of every node on the path, shape (nodes,
    configurations, theta). It does not depend on epsilon, and node k only depends on nodes 1..k, so the moments of the
    longest chain serve every shorter prefix and every epsilon (see propagated_node_bounds).
    The arrival spacing of node k is the smallest of three bounds: the service spacing of node k-1, the arrival spacing
    of node k-1 corrected by its latency, and the external arrival process (the assumption of the SNC bound), so the result
    is never looser than the SNC bound. path_beliefs: one (mean, var) pair per node. n_max: queue lengths summed explicitly
    (the rest by a geometric tail). path_lower_means: the lower (optimistic) mean of every node, used where its execution
    times space the items for the next node (a faster node spaces them less); defaults to the means of path_beliefs."""
    n = np.arange(n_max + 1)
    external = n * log_interarrival_mgf(process, rate, jitter, theta)[:, None]
    n_cfgs = len(np.atleast_1d(path_beliefs[0][0]))
    moments = np.empty((len(path_beliefs), n_cfgs, len(theta)))
    for start in range(0, n_cfgs, chunk):
        cfgs = slice(start, start + chunk)
        log_spacing = external[None]
        for k, (mean, var) in enumerate(path_beliefs):
            lower = mean if path_lower_means is None else path_lower_means[k]
            mean, var, lower = np.atleast_1d(mean)[cfgs, None], np.atleast_1d(var)[cfgs, None], np.atleast_1d(lower)[cfgs, None]
            log_mgf = theta * mean + 0.5 * theta ** 2 * var
            log_mgf_negative = -theta * lower + 0.5 * theta ** 2 * var
            moment = queue_log_moment(log_mgf, log_spacing)
            moments[k, cfgs] = moment
            # arrival model of the next node: service spacing, arrival spacing or the external process, whichever is smallest
            with np.errstate(invalid="ignore"):
                via_service = n * log_mgf_negative[..., None]
                via_arrivals = np.nan_to_num(log_spacing + log_mgf_negative[..., None] + moment[..., None], nan=np.inf)
            log_spacing = np.minimum(np.minimum(np.minimum(via_service, via_arrivals), external[None]), 0.0)
    return moments


def propagated_node_bounds(path_beliefs, process, rate, epsilon, jitter=0.1, n_max=200, chunk=5, theta=THETA_GRID,
                           moments=None, path_lower_means=None):
    """Latency bound of every node on the path at violation probability epsilon; one row per node, one column per
    configuration. Pass `moments` from propagated_node_moments of a longer chain with the same first nodes to skip the
    recursion (only the minimization over theta depends on epsilon)."""
    if moments is None:
        moments = propagated_node_moments(path_beliefs, process, rate, jitter, n_max, chunk, theta, path_lower_means)
    return latency_bound_from_moment(moments[:len(path_beliefs)], epsilon, theta)


# ------------------------------------------------------------------------------
# Network service curve with Hoelder's inequality (one bound for the whole chain, any dependence between nodes)
# ------------------------------------------------------------------------------
def network_log_moment(path_beliefs, process, rate, jitter=0.1, theta=THETA_GRID, holder_exponents=None, iterations=50):
    """ln of prod_k M~_k(theta) / (1 - M~_k(theta) E[exp(-theta T)]), one row per configuration, where M~_k is node k's
    Gaussian execution-time MGF with its variance multiplied by the Hoelder exponent p_k (sum_k 1/p_k = 1).
    holder_exponents: one fixed p_k per node; None tunes them per configuration and theta (see network_service_bound)."""
    n_nodes = len(path_beliefs)
    mean = np.stack([np.atleast_1d(m)[:, None] for m, _ in path_beliefs])   # (nodes, configurations, 1)
    var = np.stack([np.atleast_1d(v)[:, None] for _, v in path_beliefs])
    log_r = log_interarrival_mgf(process, rate, jitter, theta)

    def node_moments(share):
        # share = 1 / p_k; returns the log latency moment and rho of every node (inf where rho >= 1)
        log_mgf = theta * mean + 0.5 * theta ** 2 * var / share
        log_rho = np.minimum(log_mgf + log_r, 0.0)
        with np.errstate(divide="ignore"):
            moment = np.where(log_mgf + log_r < 0, log_mgf - np.log(-np.expm1(log_rho)), np.inf)
        return moment, np.exp(log_rho)

    if holder_exponents is not None:
        assert np.isclose(sum(1 / p for p in holder_exponents), 1) or all(p == 1 for p in holder_exponents), \
            "Hoelder exponents need sum(1 / p_k) = 1"
        share = np.reshape(1 / np.asarray(holder_exponents, dtype=float), (-1, 1, 1))
        return node_moments(share)[0].sum(axis=0)
    # Tuned exponents: minimizing sum_k moment_k over the shares q_k = 1 / p_k (sum 1) is convex; its optimality
    # condition gives q_k proportional to sigma_k / sqrt(1 - rho_k), solved by a damped fixed-point iteration. Any shares
    # that sum to 1 give a valid bound, so the result is the smaller of the tuned and the equal (p_k = K) shares.
    equal = node_moments(np.full((n_nodes, 1, 1), 1 / n_nodes))[0].sum(axis=0)
    share = np.full(np.broadcast_shapes(mean.shape, theta.shape), 1 / n_nodes)
    for _ in range(iterations):
        rho = node_moments(share)[1]
        weight = np.sqrt(var / np.maximum(1 - rho, 1e-12))
        share = np.maximum(0.5 * share + 0.5 * weight / weight.sum(axis=0), 1e-300)
    return np.minimum(equal, node_moments(share)[0].sum(axis=0))


def network_service_bound(path_beliefs, process, rate, epsilon, jitter=0.1, theta=THETA_GRID, holder_exponents=None):
    """End-to-end latency bound of a chain of FIFO nodes from its network service process (Section 7b), one value per
    configuration. The end-to-end latency is a maximum over paths through (item, node); a union bound over the paths and
    one Chernoff step give e^(-theta d) * prod_k M~_k(theta) / (1 - M~_k(theta) E[e^(-theta T)]). Hoelder's inequality,
    E[prod_k e^(theta S_k)] <= prod_k E[e^(p_k theta S_k)]^(1/p_k) with sum_k 1/p_k = 1, makes it hold for any dependence
    between the nodes; for Gaussian execution times it multiplies node k's variance by p_k.
    holder_exponents: fixed p_k per node (all ones gives the independent SNC convolution, not a guarantee); None tunes
    them per configuration and theta."""
    return latency_bound_from_moment(network_log_moment(path_beliefs, process, rate, jitter, theta, holder_exponents),
                                     epsilon, theta)


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


def martingale_node_bound(mean, var, process, rate, epsilon, jitter=0.1, bins=4000, iterations=30):
    """Smallest d with P(Q + X > d) <= epsilon, per configuration; inf for overloaded nodes.
    epsilon may be a scalar (result: one bound per configuration) or a sequence (result: one row per epsilon); all
    epsilons are solved in the same bisection. E_X[P(Q > d - X)] uses `bins` equal-probability bins of X at their upper
    edges, the last bin counted as a violation, so the discretization can only make the bound larger. The bisection over
    [mean, mean + 60 s] returns its upper end, so it can only err upwards, by at most 60 s / 2^iterations (56 ns for 30)."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    sd, theta = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter)
    upper_edges = mean[:, None] + sd[:, None] * norm.ppf(np.arange(1, bins) / bins)
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))[:, None]          # (epsilons, 1)

    def violation(d):                                                              # d: (epsilons, configurations)
        slack = d[..., None] - upper_edges
        tail = np.where(slack < 0, 1.0, waiting_time_tail(slack, mean[:, None], sd[:, None], theta[:, None], process, rate))
        return (tail.sum(axis=-1) + 1.0) / bins

    low = np.broadcast_to(mean, (len(epsilons), len(mean))).copy()
    high = low + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = violation(middle) > epsilons
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    bounds = np.where(np.isnan(theta), np.inf, high)
    return bounds[0] if np.ndim(epsilon) == 0 else bounds


def martingale_propagated_node_bound(mean, var, previous_lower_mean, previous_var, process, rate, epsilon, jitter=0.1,
                                     bins=4000, iterations=30):
    """Martingale bound of a node that receives the departures of the node before it (previous_lower_mean None for the
    first node). The waiting-time tail is the smaller of the martingale tail with external arrivals (waiting_time_tail)
    and Kingman's bound exp(-theta_s w) for the walk with increments X_k - X_(k-1), which bounds the wait because the node
    before cannot release items closer together than its execution times; theta_s = 2 (mean_prev - mean) / (var + var_prev).
    The node before enters with its lower (optimistic) mean, because a faster node spaces items less. epsilon may be a
    scalar or a sequence, as in martingale_node_bound."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    sd, theta_external = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter)
    if previous_lower_mean is None:
        theta_spacing = np.zeros_like(mean)
    else:
        theta_spacing = np.maximum(2 * (np.atleast_1d(previous_lower_mean) - mean) / (var + np.atleast_1d(previous_var)), 0.0)
    upper_edges = mean[:, None] + sd[:, None] * norm.ppf(np.arange(1, bins) / bins)
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))[:, None]

    def violation(d):
        slack = np.maximum(d[..., None] - upper_edges, 0.0)
        external = np.nan_to_num(waiting_time_tail(slack, mean[:, None], sd[:, None], theta_external[:, None], process, rate), nan=1.0)
        spacing = np.where(theta_spacing[:, None] > 0, np.exp(-theta_spacing[:, None] * slack), 1.0)
        tail = np.where(d[..., None] - upper_edges < 0, 1.0, np.minimum(external, spacing))
        return (tail.sum(axis=-1) + 1.0) / bins

    low = np.broadcast_to(mean, (len(epsilons), len(mean))).copy()
    high = low + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = violation(middle) > epsilons
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    bounds = np.where(np.isnan(theta_external) & (theta_spacing <= 0), np.inf, high)
    return bounds[0] if np.ndim(epsilon) == 0 else bounds


def martingale_propagated_node_bounds(path_beliefs, path_lower_means, process, rate, epsilon, jitter=0.1, bins=4000, iterations=30):
    """martingale_propagated_node_bound for every node of a path; one row per node, one column per configuration.
    path_lower_means: the lower (optimistic) mean of every node, used for the spacing it gives the next node."""
    return np.array([martingale_propagated_node_bound(mean, var, None if k == 0 else path_lower_means[k - 1],
                                                      None if k == 0 else path_beliefs[k - 1][1], process, rate, epsilon,
                                                      jitter, bins, iterations)
                     for k, (mean, var) in enumerate(path_beliefs)])


# ------------------------------------------------------------------------------
# Martingale bound in closed form (Kingman's bound with a Gaussian execution time)
# ------------------------------------------------------------------------------
def martingale_closed_form_node_bound(mean, var, process, rate, epsilon, jitter=0.1, previous_lower_mean=None, previous_var=None,
                                      iterations=40):
    """Smallest d with P(Q + X > d) <= epsilon from a closed formula, per configuration; inf for overloaded nodes.
    Q is the waiting time and X ~ N(mean, var) the item's own execution time (s), independent of Q.
    1. Waiting time: Kingman's martingale bound P(Q > w) <= exp(-theta w) for w >= 0, with theta the decay rate of
       martingale_theta (the largest grid theta with rho(theta) <= 1; any smaller theta is valid as well).
    2. Latency: P(Q + X > d) = E_X[P(Q > d - X)] <= E_X[min(1, exp(-theta (d - X)))], which for a Gaussian X is
       P(X > d) + exp(-theta (d - mean) + theta^2 var / 2) * Phi((d - mean - theta var) / sd), with Phi the standard
       normal distribution function. No discretization of X is needed; d is found by bisection on this formula.
    With previous_lower_mean and previous_var (the optimistic mean and the variance of the node before, see
    martingale_propagated_node_bound), the node receives the departures of that node: the waiting time then also obeys
    exp(-theta_s w) with theta_s = 2 (mean_prev - mean) / (var + var_prev), and the smaller of two exponential tails is
    the one with the larger rate, so theta = max(theta, theta_s). The assumptions are those of the numerical bounds
    (martingale_node_bound, martingale_propagated_node_bound).
    Unlike martingale_node_bound it does not use the Pollaczek-Khinchine refinement for Poisson arrivals (a waiting-time
    tail that starts at the share of items that wait, not at 1), so for Poisson arrivals it is somewhat larger.
    The bisection over [mean, mean + 60 s] returns its upper end, so it can only err upwards, by 60 s / 2^iterations."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    sd, theta = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter)
    if previous_lower_mean is not None:
        theta_spacing = 2 * (np.atleast_1d(previous_lower_mean) - mean) / (var + np.atleast_1d(previous_var))
        theta = np.fmax(theta, np.where(theta_spacing > 0, theta_spacing, np.nan))   # fmax ignores a nan on one side
    rate_of_decay = np.nan_to_num(theta, nan=1.0)   # placeholder where there is no valid rate; those entries become inf below

    def tail(d):   # bound on P(Q + X > d); the second term in log space, because theta^2 var can be very large
        waits = np.exp(-rate_of_decay * (d - mean) + 0.5 * rate_of_decay ** 2 * var + norm.logcdf((d - mean - rate_of_decay * var) / sd))
        return norm.sf((d - mean) / sd) + waits

    low, high = mean.copy(), mean + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = tail(middle) > epsilon
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    return np.where(np.isnan(theta), np.inf, high)


def martingale_closed_form_node_bounds(path_beliefs, path_lower_means, process, rate, epsilon, jitter=0.1):
    """martingale_closed_form_node_bound for every node of a path, each node receiving the departures of the node before
    it; one row per node, one column per configuration. path_lower_means: the lower (optimistic) mean of every node."""
    return np.array([martingale_closed_form_node_bound(mean, var, process, rate, epsilon, jitter,
                                                       None if k == 0 else path_lower_means[k - 1],
                                                       None if k == 0 else path_beliefs[k - 1][1])
                     for k, (mean, var) in enumerate(path_beliefs)])


# ------------------------------------------------------------------------------
# Ground truth: simulated chain of FIFO servers
# ------------------------------------------------------------------------------
def simulate_chain(mean_execution, noise_levels, process, rate, n_items, item_correlation, rng, jitter=0.1, warmup_share=0.1,
                   return_arrivals=False):
    """Simulates a chain of FIFO single-server services, vectorized over configurations.
    mean_execution: (configurations, nodes) mean execution time of every node (s); noise_levels: std per node (s).
    The execution-time noise of an item is shared across nodes with correlation item_correlation.
    Returns per-node latencies (configurations, items, nodes) and end-to-end latencies (configurations, items);
    the first warmup_share of items is dropped. With return_arrivals, the times (s) at which the kept items entered the
    chain (configurations, items) are returned as a third value."""
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
    results = np.stack(node_latencies, axis=-1)[:, kept_items], (entering - arrivals)[:, kept_items]
    return (*results, arrivals[:, kept_items]) if return_arrivals else results


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
