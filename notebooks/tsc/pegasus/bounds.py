"""Latency bounds for chains of FIFO services with one or more workers.

Shared by the notebooks of this folder. Every function takes the arrival process explicitly: `process` is "poisson"
(exponential gaps with mean 1/rate), "periodic" (gaps of (1/rate) * (1 + U(-jitter, jitter))) or "gamma"
(gamma-distributed gaps with mean 1/rate and squared coefficient of variation cv2 = variance / mean^2; cv2 = 1 is the
Poisson process, a smaller cv2 spaces the arrivals more evenly and a larger one makes them burstier). Only the functions
that take `cv2` know the gamma process.
Execution times Y are Gaussian with the given mean m and variance (s, s^2); bounds are latencies in seconds that are
exceeded with probability at most `epsilon`. Vectorized functions take one mean/variance per configuration.
`rate` is one number, or one arrival rate per configuration (real data: every run of a chain has its own rate).
The simulator that provides the ground truth is in `simulation.py`.

Workers. A service with c workers takes the items in the order of arrival and gives each to the first free worker
(a FIFO queue with c servers). Its waiting time is at most the waiting time under *cyclic assignment*, where every c-th
item goes to the same worker (Wolff, "An upper bound for multi-channel queues", J. Appl. Prob. 14, 1977). Under cyclic
assignment, every worker is a single-server queue whose gaps between arrivals are sums of c consecutive gaps, so
E[exp(-theta T)] becomes E[exp(-theta T)]^c for i.i.d. gaps. Every bound below applies this to a node with c workers
(`workers`: one number, or one per configuration; `path_workers`: one such entry per node; default one worker, which
gives the single-server bounds unchanged). The bound is looser than the true multi-server wait, because a worker of
the cyclic system may wait while another one is free.

Which bounds are Pegasus. Pegasus Node (evaluation.pegasus_bound) is the union bound over
martingale_closed_form_node_bounds, which bounds every node with the external arrival process; Pegasus SNC
(evaluation.pegasus_snc_bound) is network_service_bound. The propagated bounds (propagated_node_moments,
propagated_node_bounds, martingale_propagated_node_bound(s), spacing_decay_rate) are explored variants and not part of
Pegasus: they use the minimum spacing that a node gives the next one. A node with several workers can release items out
of order and several at almost the same time, so they use that spacing only after a node with one worker.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


THETA_GRID = np.logspace(-1, 5, 4000)   # MGF parameter theta (1/s) over which every bound is optimized


# ------------------------------------------------------------------------------
# Arrival processes
# ------------------------------------------------------------------------------
def log_interarrival_mgf(process, rate, jitter=0.1, theta=THETA_GRID, cv2=1.0):
    """ln E[exp(-theta T)] of one gap T between arrivals, one value per theta; with one rate per configuration, one
    row per configuration."""
    rate = np.asarray(rate, dtype=float)
    if rate.ndim:
        rate = rate[:, None]
    if process == "poisson":
        return np.log(rate / (rate + theta))
    if process == "gamma":
        # a gamma gap with shape 1 / cv2 and scale cv2 / rate has E[exp(-theta T)] = (1 + theta * scale)^(-shape)
        return -np.log1p(theta * cv2 / rate) / cv2
    if process == "periodic":
        shortest_gap = (1 - jitter) / rate
        if jitter == 0:
            return -theta * shortest_gap
        width = 2 * jitter / rate
        return -theta * shortest_gap + np.log(-np.expm1(-theta * width)) - np.log(theta * width)
    raise ValueError(f"unknown arrival process {process!r}")


def worker_counts(workers, n_cfgs: int) -> np.ndarray:
    """Number of workers of a node per configuration, as a float array of length n_cfgs (workers: one number or one per
    configuration)."""
    return np.broadcast_to(np.asarray(workers, dtype=float), (n_cfgs,)).copy()


def log_worker_gap_mgf(process, rate, workers, jitter=0.1, theta=THETA_GRID, cv2=1.0):
    """ln E[exp(-theta (T_1 + ... + T_w))], the gap between two arrivals at one worker under cyclic assignment (module
    docstring), one row per configuration; workers: one number per configuration (worker_counts)."""
    return workers[:, None] * log_interarrival_mgf(process, rate, jitter, theta, cv2)


# ------------------------------------------------------------------------------
# SNC bound (union bound over queue lengths + Chernoff)
# ------------------------------------------------------------------------------
def log_latency_moment(mean, var, process, rate, jitter=0.1, theta=THETA_GRID, workers=1):
    """ln of M_Y(theta) / (1 - phi(theta)), one row per configuration; inf where phi >= 1. With c workers,
    phi(theta) = M_Y(theta) E[exp(-theta T)]^c (cyclic assignment, module docstring)."""
    mean, var = np.atleast_1d(mean)[:, None], np.atleast_1d(var)[:, None]
    log_mgf = theta * mean + 0.5 * theta ** 2 * var
    log_rho = log_mgf + log_worker_gap_mgf(process, rate, worker_counts(workers, len(mean)), jitter, theta)
    stable = log_rho < 0
    moment = np.full(log_mgf.shape, np.inf)
    moment[stable] = log_mgf[stable] - np.log1p(-np.exp(log_rho[stable]))
    return moment


def latency_bound_from_moment(log_moment, epsilon, theta=THETA_GRID):
    """d(epsilon) = min over theta of (log moment - ln epsilon) / theta."""
    return np.min((log_moment - np.log(epsilon)) / theta, axis=-1)


def snc_latency_bound(mean, var, process, rate, epsilon, jitter=0.1, workers=1):
    return latency_bound_from_moment(log_latency_moment(mean, var, process, rate, jitter, workers=workers), epsilon)


# ------------------------------------------------------------------------------
# SNC bound with propagated arrival models (node k is fed by the departures of node k-1); explored variant, not Pegasus
# ------------------------------------------------------------------------------
def queue_log_moment(log_mgf_node, log_spacing):
    """ln sum_n M_Y(theta)^(n+1) E[exp(-theta A_n)], where log_spacing[..., n] bounds ln E[exp(-theta A_n)].
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


def propagated_node_moments(path_beliefs, process, rate, jitter=0.1, n_max=200, chunk=5, theta=THETA_GRID, path_lower_means=None,
                            path_workers=None):
    """ln of the latency moment sum_n M_Y(theta)^(n+1) E[exp(-theta A_n)] of every node on the path, shape (nodes,
    configurations, theta). It does not depend on epsilon, and node k only depends on nodes 1..k, so the moments of the
    longest chain serve every shorter prefix and every epsilon (see propagated_node_bounds).
    The arrival spacing of node k is the smallest of three bounds: the service spacing of node k-1, the arrival spacing
    of node k-1 corrected by its latency, and the external arrival process (the assumption of the SNC bound), so the result
    is never looser than the SNC bound. path_beliefs: one (mean, var) pair per node. n_max: queue lengths summed explicitly
    (the rest by a geometric tail). path_lower_means: the lower (optimistic) mean of every node, used where its execution
    times space the items for the next node (a faster node spaces them less); defaults to the means of path_beliefs.
    path_workers: workers of every node (module docstring), default one each. A worker of a node with c workers receives
    every c-th item, so its n-th item before is n * c items earlier in the stream: its spacing is A_(n c). The first two
    bounds on the spacing of node k need the items to leave node k-1 in order, one at a time; after a node with more than
    one worker, node k assumes the external arrival process."""
    n_cfgs = len(np.atleast_1d(path_beliefs[0][0]))
    path_workers = [worker_counts(1 if path_workers is None else path_workers[k], n_cfgs).astype(int) for k in range(len(path_beliefs))]
    w_max = max(int(w.max()) for w in path_workers)
    chunk = max(1, chunk // w_max)   # the spacing covers n_max * w_max items, so fewer configurations fit in memory at once
    n = np.arange(n_max * w_max + 1)   # an item and the n items before it in the stream that enters a node
    log_gap = log_interarrival_mgf(process, rate, jitter, theta)   # (theta,), or (configurations, theta)
    moments = np.empty((len(path_beliefs), n_cfgs, len(theta)))
    for start in range(0, n_cfgs, chunk):
        cfgs = slice(start, start + chunk)
        size = len(range(n_cfgs)[cfgs])
        external = np.broadcast_to(n * (log_gap if log_gap.ndim == 1 else log_gap[cfgs])[..., None], (size, len(theta), len(n)))
        log_spacing = external
        for k, (mean, var) in enumerate(path_beliefs):
            lower = mean if path_lower_means is None else path_lower_means[k]
            mean, var, lower = np.atleast_1d(mean)[cfgs, None], np.atleast_1d(var)[cfgs, None], np.atleast_1d(lower)[cfgs, None]
            workers = path_workers[k][cfgs]
            log_mgf = theta * mean + 0.5 * theta ** 2 * var
            log_mgf_negative = -theta * lower + 0.5 * theta ** 2 * var
            every_w = np.broadcast_to((np.arange(n_max + 1)[None, :] * workers[:, None])[:, None, :], (size, len(theta), n_max + 1))
            moment = queue_log_moment(log_mgf, np.take_along_axis(log_spacing, every_w, axis=-1))
            moments[k, cfgs] = moment
            # arrival model of the next node: service spacing, arrival spacing or the external process, whichever is
            # smallest; only the external process after a node with several workers
            with np.errstate(invalid="ignore"):
                via_service = n * log_mgf_negative[..., None]
                via_arrivals = np.nan_to_num(log_spacing + log_mgf_negative[..., None] + moment[..., None], nan=np.inf)
            propagated = np.minimum(np.minimum(via_service, via_arrivals), external)
            log_spacing = np.minimum(np.where((workers == 1)[:, None, None], propagated, external), 0.0)
    return moments


def propagated_node_bounds(path_beliefs, process, rate, epsilon, jitter=0.1, n_max=200, chunk=5, theta=THETA_GRID,
                           moments=None, path_lower_means=None, path_workers=None):
    """Latency bound of every node on the path at violation probability epsilon; one row per node, one column per
    configuration. Pass `moments` from propagated_node_moments of a longer chain with the same first nodes to skip the
    recursion (only the minimization over theta depends on epsilon)."""
    if moments is None:
        moments = propagated_node_moments(path_beliefs, process, rate, jitter, n_max, chunk, theta, path_lower_means, path_workers)
    return latency_bound_from_moment(moments[:len(path_beliefs)], epsilon, theta)


# ------------------------------------------------------------------------------
# Network service curve with Hoelder's inequality (one bound for the whole chain, any dependence between nodes)
# ------------------------------------------------------------------------------
def path_log_gap_mgf(path_beliefs, process, rate, jitter=0.1, theta=THETA_GRID, cv2=1.0, path_workers=None):
    """ln E[exp(-theta T)]^(c_k) of every node k of the path (one worker's gap under cyclic assignment, module
    docstring), shape (nodes, configurations, theta); without path_workers the same for every node, shape (theta,) or
    (configurations, theta)."""
    if path_workers is None:
        return log_interarrival_mgf(process, rate, jitter, theta, cv2)
    n_cfgs = len(np.atleast_1d(path_beliefs[0][0]))
    return np.stack([log_worker_gap_mgf(process, rate, worker_counts(w, n_cfgs), jitter, theta, cv2) for w in path_workers])


def network_log_moment(path_beliefs, process, rate, jitter=0.1, theta=THETA_GRID, holder_exponents=None, iterations=50, cv2=1.0,
                       path_workers=None):
    """ln of prod_k M~_k(theta) / (1 - M~_k(theta) E[exp(-theta T)]^(c_k)), one row per configuration, where M~_k is
    node k's Gaussian execution-time MGF with its variance multiplied by the Hoelder exponent p_k (sum_k 1/p_k = 1) and
    c_k its number of workers (path_workers, default one each).
    holder_exponents: one fixed p_k per node; None tunes them per configuration and theta (see network_service_bound)."""
    n_nodes = len(path_beliefs)
    mean = np.stack([np.atleast_1d(m)[:, None] for m, _ in path_beliefs])   # (nodes, configurations, 1)
    var = np.stack([np.atleast_1d(v)[:, None] for _, v in path_beliefs])
    log_r = path_log_gap_mgf(path_beliefs, process, rate, jitter, theta, cv2, path_workers)

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
    # condition gives q_k proportional to sigma_k / sqrt(1 - phi_k), solved by a damped fixed-point iteration. Any shares
    # that sum to 1 give a valid bound, so the result is the smaller of the tuned and the equal (p_k = K) shares.
    equal = node_moments(np.full((n_nodes, 1, 1), 1 / n_nodes))[0].sum(axis=0)
    share = np.full(np.broadcast_shapes(mean.shape, theta.shape), 1 / n_nodes)
    for _ in range(iterations):
        rho = node_moments(share)[1]
        weight = np.sqrt(var / np.maximum(1 - rho, 1e-12))
        share = np.maximum(0.5 * share + 0.5 * weight / weight.sum(axis=0), 1e-300)
    return np.minimum(equal, node_moments(share)[0].sum(axis=0))


def network_service_bound(path_beliefs, process, rate, epsilon, jitter=0.1, theta=THETA_GRID, holder_exponents=None, cv2=1.0,
                          path_workers=None):
    """End-to-end latency bound of a chain of FIFO nodes from its network service process (Section 7b), one value per
    configuration. The end-to-end latency is a maximum over paths through (item, node); a union bound over the paths and
    one Chernoff step give e^(-theta d) * prod_k M~_k(theta) / (1 - M~_k(theta) E[e^(-theta T)]). Hoelder's inequality,
    E[prod_k e^(theta S_k)] <= prod_k E[e^(p_k theta S_k)]^(1/p_k) with sum_k 1/p_k = 1, makes it hold for any dependence
    between the nodes; for Gaussian execution times it multiplies node k's variance by p_k.
    holder_exponents: fixed p_k per node; None tunes them per configuration and theta. All ones gives the independent
    SNC convolution, which skips the Hoelder step: a guarantee only if the execution times of an item are independent
    between the nodes. cv2: squared coefficient of variation of the gaps between arrivals (gamma arrivals only).
    path_workers: workers of every node; a node with c workers enters with the cyclic-assignment factor of the module
    docstring. The derivation over paths through (item, node) assumes that items leave every node in order, which a
    node with several workers does not guarantee; with workers, this composition is checked against the simulation
    only, not proven."""
    return latency_bound_from_moment(network_log_moment(path_beliefs, process, rate, jitter, theta, holder_exponents, cv2=cv2,
                                                        path_workers=path_workers),
                                     epsilon, theta)


def network_bound_terms(path_beliefs, process, rate, epsilon, jitter=0.1, theta=THETA_GRID, cv2=1.0, path_workers=None):
    """The three terms of network_service_bound for independent nodes (all Hoelder exponents 1), at the theta that
    minimizes their sum, per configuration (s): the pessimistic execution time sum_k ln M_k(theta) / theta, the queueing
    term sum_k -ln(1 - phi_k(theta)) / theta, and the price of confidence ln(1 / epsilon) / theta. The fourth value is
    that theta (1/s). The three terms add up to network_service_bound(..., holder_exponents=[1] * K); all are inf where
    a node is overloaded at every theta."""
    mean = np.stack([np.atleast_1d(m)[:, None] for m, _ in path_beliefs])   # (nodes, configurations, 1)
    var = np.stack([np.atleast_1d(v)[:, None] for _, v in path_beliefs])
    log_mgf = theta * mean + 0.5 * theta ** 2 * var
    log_rho = log_mgf + path_log_gap_mgf(path_beliefs, process, rate, jitter, theta, cv2, path_workers)
    stable = np.all(log_rho < 0, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        queueing = np.where(stable, -np.log(-np.expm1(np.minimum(log_rho, 0.0))).sum(axis=0), np.inf) / theta
    execution, confidence = log_mgf.sum(axis=0) / theta, np.broadcast_to(-np.log(epsilon) / theta, queueing.shape)
    best = np.argmin(execution + queueing + confidence, axis=-1)
    pick = lambda term: np.take_along_axis(term, best[:, None], axis=-1)[:, 0]
    return pick(execution), pick(queueing), pick(confidence), theta[best]


# ------------------------------------------------------------------------------
# Martingale bound (Kingman; Pollaczek-Khinchine refinement for Poisson arrivals)
# ------------------------------------------------------------------------------
def martingale_theta(mean, var, process, rate, jitter=0.1, theta=THETA_GRID, cv2=1.0, workers=1):
    """Largest grid theta below the first crossing of phi(theta) = 1; nan if there is none (overloaded node). With c
    workers, phi(theta) = M_Y(theta) E[exp(-theta T)]^c (cyclic assignment, module docstring)."""
    log_rho = theta * mean[:, None] + 0.5 * theta ** 2 * var[:, None] + log_worker_gap_mgf(process, rate, worker_counts(workers, len(mean)),
                                                                                           jitter, theta, cv2)
    crossing = np.where(np.any(log_rho >= 0, axis=1), np.argmax(log_rho >= 0, axis=1), len(theta))
    return np.where(crossing > 0, theta[np.maximum(crossing - 1, 0)], np.nan)


def waiting_time_tail(w, mean, sd, theta, process, rate, single_worker=True):
    """Bound on P(Q > w) for the waiting time Q, w >= 0: Pollaczek-Khinchine refinement for Poisson arrivals at a node
    with one worker, Kingman's exp(-theta w) otherwise (a worker of a node with c workers receives every c-th item
    under cyclic assignment, which is no longer a Poisson stream). One rate per configuration, and single_worker (bool,
    one per configuration), must have the shape of mean, sd and theta (one column)."""
    w = np.maximum(w, 0.0)
    kingman = np.exp(-theta * w)
    if process != "poisson" or not np.any(single_worker):
        return np.minimum(1.0, kingman)
    z_w, z_0 = (w - mean) / sd, -mean / sd
    beyond = sd * (norm.pdf(z_w) - z_w * norm.sf(z_w))                  # int_w^inf P(Y > x) dx
    below = (norm.sf(z_w) - kingman * norm.sf(z_0)) / theta             # e^{-theta w} int_0^w P(Y > x) e^{theta x} dx ...
    mass = norm.cdf((w - mean - theta * sd ** 2) / sd) - norm.cdf((-mean - theta * sd ** 2) / sd)
    with np.errstate(divide="ignore"):
        below = below + np.exp(-theta * w + theta * mean + 0.5 * theta ** 2 * sd ** 2 + np.log(np.maximum(mass, 0.0))) / theta
    refined = np.minimum(1.0, np.nan_to_num(rate * (beyond + below), nan=1.0))
    return np.where(single_worker, refined, np.minimum(1.0, kingman))


def martingale_node_bound(mean, var, process, rate, epsilon, jitter=0.1, bins=4000, iterations=30, workers=1):
    """Smallest d with P(Q + Y > d) <= epsilon, per configuration; inf for overloaded nodes.
    epsilon may be a scalar (result: one bound per configuration) or a sequence (result: one row per epsilon); all
    epsilons are solved in the same bisection. E_Y[P(Q > d - Y)] uses `bins` equal-probability bins of Y at their upper
    edges, the last bin counted as a violation, so the discretization can only make the bound larger. The bisection over
    [mean, mean + 60 s] returns its upper end, so it can only err upwards, by at most 60 s / 2^iterations (56 ns for 30).
    workers: workers of the node, see the module docstring."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    workers = worker_counts(workers, len(mean))
    sd, theta = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter, workers=workers)
    rate_column = np.asarray(rate, dtype=float)[:, None] if np.ndim(rate) else rate   # one rate per configuration: one row each
    upper_edges = mean[:, None] + sd[:, None] * norm.ppf(np.arange(1, bins) / bins)
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))[:, None]          # (epsilons, 1)

    def violation(d):                                                              # d: (epsilons, configurations)
        slack = d[..., None] - upper_edges
        tail = np.where(slack < 0, 1.0, waiting_time_tail(slack, mean[:, None], sd[:, None], theta[:, None], process, rate_column,
                                                           workers[:, None] == 1))
        return (tail.sum(axis=-1) + 1.0) / bins

    low = np.broadcast_to(mean, (len(epsilons), len(mean))).copy()
    high = low + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = violation(middle) > epsilons
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    bounds = np.where(np.isnan(theta), np.inf, high)
    return bounds[0] if np.ndim(epsilon) == 0 else bounds


def spacing_decay_rate(mean, var, previous_lower_mean, previous_var, workers, previous_workers):
    """Decay rate theta_s (1/s) of Kingman's bound exp(-theta_s w) on the waiting time of a node that receives the
    departures of the node before it, per configuration; 0 where there is none. A node before with one worker releases
    the items in order and no two closer together than its execution times, so the items i and i + c that a worker of
    the node receives under cyclic assignment are at least c execution times of the node before apart. The wait is then
    the maximum of a random walk with increments Y - (Y'_1 + ... + Y'_c), whose Gaussian decay rate is
    theta_s = 2 (c m'^- - m^+) / (var + c var_prev), with m^+ = mean and m'^- = previous_lower_mean. After a node with several workers there is no such spacing."""
    spacing = 2 * (workers * np.atleast_1d(previous_lower_mean) - mean) / (var + workers * np.atleast_1d(previous_var))
    return np.where(previous_workers == 1, np.maximum(spacing, 0.0), 0.0)


def martingale_propagated_node_bound(mean, var, previous_lower_mean, previous_var, process, rate, epsilon, jitter=0.1,
                                     bins=4000, iterations=30, workers=1, previous_workers=1):
    """Martingale bound of a node that receives the departures of the node before it (previous_lower_mean None for the
    first node); an explored variant, not part of Pegasus Node (module docstring). The waiting-time tail is the smaller of the martingale tail with external arrivals (waiting_time_tail)
    and Kingman's bound exp(-theta_s w) for the spacing that the node before gives (spacing_decay_rate).
    The node before enters with its lower (optimistic) mean, because a faster node spaces items less. epsilon may be a
    scalar or a sequence, as in martingale_node_bound. workers, previous_workers: workers of the node and of the node
    before, see the module docstring."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    workers = worker_counts(workers, len(mean))
    sd, theta_external = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter, workers=workers)
    rate_column = np.asarray(rate, dtype=float)[:, None] if np.ndim(rate) else rate   # one rate per configuration: one row each
    if previous_lower_mean is None:
        theta_spacing = np.zeros_like(mean)
    else:
        theta_spacing = spacing_decay_rate(mean, var, previous_lower_mean, previous_var, workers, worker_counts(previous_workers, len(mean)))
    upper_edges = mean[:, None] + sd[:, None] * norm.ppf(np.arange(1, bins) / bins)
    epsilons = np.atleast_1d(np.asarray(epsilon, dtype=float))[:, None]

    def violation(d):
        slack = np.maximum(d[..., None] - upper_edges, 0.0)
        external = np.nan_to_num(waiting_time_tail(slack, mean[:, None], sd[:, None], theta_external[:, None], process, rate_column,
                                                   workers[:, None] == 1), nan=1.0)
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


def martingale_propagated_node_bounds(path_beliefs, path_lower_means, process, rate, epsilon, jitter=0.1, bins=4000, iterations=30,
                                      path_workers=None):
    """martingale_propagated_node_bound for every node of a path; one row per node, one column per configuration.
    path_lower_means: the lower (optimistic) mean of every node, used for the spacing it gives the next node.
    path_workers: workers of every node, default one each."""
    path_workers = [1] * len(path_beliefs) if path_workers is None else path_workers
    return np.array([martingale_propagated_node_bound(mean, var, None if k == 0 else path_lower_means[k - 1],
                                                      None if k == 0 else path_beliefs[k - 1][1], process, rate, epsilon,
                                                      jitter, bins, iterations, path_workers[k], 1 if k == 0 else path_workers[k - 1])
                     for k, (mean, var) in enumerate(path_beliefs)])


# ------------------------------------------------------------------------------
# Martingale bound in closed form (Kingman's bound with a Gaussian execution time)
# ------------------------------------------------------------------------------
def martingale_closed_form_node_bound(mean, var, process, rate, epsilon, jitter=0.1, iterations=40, cv2=1.0, workers=1):
    """Smallest d with P(Q + Y > d) <= epsilon from a closed formula, per configuration; inf for overloaded nodes.
    Q is the waiting time and Y ~ N(mean, var) the item's own execution time (s), independent of Q.
    1. Waiting time: Kingman's martingale bound P(Q > w) <= exp(-theta w) for w >= 0, with theta the decay rate of
       martingale_theta (the largest grid theta with phi(theta) <= 1; any smaller theta is valid as well).
    2. Latency: P(Q + Y > d) = E_Y[P(Q > d - Y)] <= E_Y[min(1, exp(-theta (d - Y)))], which for a Gaussian Y is
       P(Y > d) + exp(-theta (d - mean) + theta^2 var / 2) * Phi((d - mean - theta var) / sd), with Phi the standard
       normal distribution function. No discretization of X is needed; d is found by bisection on this formula.
    Every node is bounded with the external arrival process, also inside a chain (unlike
    martingale_propagated_node_bound, it does not use the minimum spacing that the node before gives). The assumptions
    are those of martingale_node_bound. workers: workers of the node, see the module docstring.
    Unlike martingale_node_bound it does not use the Pollaczek-Khinchine refinement for Poisson arrivals (a waiting-time
    tail that starts at the share of items that wait, not at 1), so for Poisson arrivals it is somewhat larger.
    The bisection over [mean, mean + 60 s] returns its upper end, so it can only err upwards, by 60 s / 2^iterations."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    workers = worker_counts(workers, len(mean))
    sd, theta = np.sqrt(var), martingale_theta(mean, var, process, rate, jitter, cv2=cv2, workers=workers)
    rate_of_decay = np.nan_to_num(theta, nan=1.0)   # placeholder where there is no valid rate; those entries become inf below

    def tail(d):   # bound on P(Q + Y > d); the second term in log space, because theta^2 var can be very large
        waits = np.exp(-rate_of_decay * (d - mean) + 0.5 * rate_of_decay ** 2 * var + norm.logcdf((d - mean - rate_of_decay * var) / sd))
        return norm.sf((d - mean) / sd) + waits

    low, high = mean.copy(), mean + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = tail(middle) > epsilon
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    return np.where(np.isnan(theta), np.inf, high)


def martingale_closed_form_node_bounds(path_beliefs, process, rate, epsilon, jitter=0.1, cv2=1.0, path_workers=None):
    """martingale_closed_form_node_bound for every node of a path, each with the external arrival process; one row per
    node, one column per configuration. path_workers: workers of every node, default one each."""
    path_workers = [1] * len(path_beliefs) if path_workers is None else path_workers
    return np.array([martingale_closed_form_node_bound(mean, var, process, rate, epsilon, jitter, cv2=cv2, workers=path_workers[k])
                     for k, (mean, var) in enumerate(path_beliefs)])
