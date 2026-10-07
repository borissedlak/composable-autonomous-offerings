"""All baselines of `method_v2.ipynb`, i.e. everything that is not part of the proposed method (the node bounds and
compositions with a guarantee are in `snc_bounds.py`: the union bound over each node bound, and the network service
curve). The notebook only calls these functions.

1. Execution times only (no queue): a bound on one execution time (Section 6a) and the joint Gaussian quantile of the
   sum of execution times (Section 7c).
2. Compositions of latency bounds without a guarantee: the independent SNC convolution (Section 7c).
3. Deep GP baselines (DGP Basic, DGP Advanced): the composition of the previous paper (`rejected_paper.pdf`).
4. Fluxion-style baseline: modular learning of latency quantiles, composed into an inference graph (Liang et al.,
   NSDI 2023).
5. Replay of monitored node latencies: the composition of Geebelen et al. (Information Sciences, 2014), which builds the
   latencies of a composition from latencies that were monitored at every service on its own.
6. Oracles that compose *measured* node latencies, to separate the cost of the node bounds from that of the composition,
   and the paired test of the independence assumption.

Units: execution times and latencies in seconds; `epsilon` / `alpha` is the violation probability.
"""
import hashlib
import inspect
import pathlib
import warnings
from contextlib import contextmanager

import gpytorch
import numpy as np
import torch
from scipy.stats import t as student_t
from sklearn.exceptions import ConvergenceWarning

from service_gp import train_gp_models
from snc_bounds import latency_bound_from_moment, log_latency_moment, snc_latency_bound


# ==============================================================================
# 1. Execution times only (no queue)
# ==============================================================================
def compute_snc_mgf_delay_bound(model, x_cfg, alpha=0.01):
    """
    Chernoff bound on a single execution time, no queue (= the SNC latency bound with lambda -> 0).

    Using log M_Y(theta) = theta*mu + (theta^2 / 2)*var, Chernoff's bound gives:
    P(Y >= D) <= exp(log M_Y(theta) - theta*D) = alpha
    Solving for optimal theta* yields: theta* = sqrt(-2*ln(alpha) / var)
    Plugging theta* back yields: D_i(alpha) = mu + sqrt(-2*ln(alpha) * var)
    """
    mu, std_total, var_epistemic, var_aleatoric = model.predict(x_cfg, return_variance_components=True)
    var_total = std_total ** 2

    # Node-optimal MGF theta parameter
    theta_opt = np.sqrt(-2.0 * np.log(alpha) / var_total)

    # Closed-form log-MGF evaluated at theta*
    log_mgf_opt = theta_opt * mu + 0.5 * (theta_opt ** 2) * var_total

    # Minimum physical latency bound satisfying target violation probability alpha
    delay_bound = (log_mgf_opt - np.log(alpha)) / theta_opt

    return delay_bound, theta_opt, mu, std_total


def joint_gaussian_quantile(node_means, node_stds, z_score):
    """(1 - epsilon) quantile of the sum of independent Gaussian execution times, no queue."""
    std_total = np.sqrt(sum(std ** 2 for std in node_stds))
    return sum(node_means) + z_score * std_total


# ==============================================================================
# 2. Compositions of latency bounds without a guarantee
# ==============================================================================
def independent_snc_convolution(path_beliefs, epsilon, **workload):
    """Multiplies the node latency MGFs and optimizes one theta: only valid for independent nodes. It equals the
    network service curve (network_service_bound in snc_bounds.py) with all Hoelder exponents 1."""
    return latency_bound_from_moment(sum(log_latency_moment(*belief, **workload) for belief in path_beliefs), epsilon)


# ==============================================================================
# 3. Deep GP baselines: DGP Basic and DGP Advanced
# ==============================================================================
# The model follows `rejected_paper.pdf` (Section 3.2, Experiment 2.1) and
# `notebooks/archived/icsoc_initial/create_deepGP.py`: one variational GP per service ("layer"), chained so that layer k
# receives the configuration of service k and a sample of layer k-1's output, trained layer by layer (decoupled
# variational inference), and evaluated by pushing Monte Carlo samples through the chain. Throughput is replaced by
# latency: the output of layer k is the cumulative latency L_k of an item from entering the chain until it leaves
# service k.
# - DGP Basic: training rows built from single-service measurements without any queue, L_k = L_{k-1} + y_k, exactly as
#   the paper built its chained data (with min(.) for throughput).
# - DGP Advanced: the same model trained on traced chain runs that include queueing (per item, the measured cumulative
#   latency before and after every service).
# Both report the (1 - epsilon) quantile of the Monte Carlo samples. Neither is a guarantee: there is no tail bound, and
# the Gaussian likelihood of every layer cannot represent the latency tail.

@contextmanager
def _single_thread():
    # the matrices are tiny: torch's thread pool costs far more than it gains (5-18x slower with 16 threads)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class ServiceGP(gpytorch.models.ApproximateGP):
    """Variational GP of one service (as in the archived create_deepGP.py), inducing points taken from the data."""

    def __init__(self, inducing_points):
        distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, distribution, learn_inducing_locations=True)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1)))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class DGPLayer:
    """One service of the chain, trained on its own rows (decoupled): standardized inputs and target, Gaussian likelihood."""

    def __init__(self, inputs, targets, seed, num_inducing=64, iterations=300, learning_rate=0.05):
        self.in_mean, self.in_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-12
        self.out_mean, self.out_std = targets.mean(), targets.std() + 1e-12
        x = torch.tensor((inputs - self.in_mean) / self.in_std, dtype=torch.float64)
        y = torch.tensor((targets - self.out_mean) / self.out_std, dtype=torch.float64)

        with _single_thread(), warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)
            torch.manual_seed(seed)
            inducing = x[torch.randperm(len(x))[:min(num_inducing, len(x))]].clone()
            self.model = ServiceGP(inducing).double()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
            self.model.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.likelihood.parameters()), lr=learning_rate)
            elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(y))
            for _ in range(iterations):
                optimizer.zero_grad()
                loss = -elbo(self.model(x), y)
                loss.backward()
                optimizer.step()
            self.model.eval()
            self.likelihood.eval()

    def state(self):
        """Everything needed to rebuild the trained layer, as tensors (for torch.save / torch.load(weights_only=True))."""
        return {"in_mean": torch.tensor(self.in_mean), "in_std": torch.tensor(self.in_std),
                "out_mean": torch.tensor(self.out_mean), "out_std": torch.tensor(self.out_std),
                "model": self.model.state_dict(), "likelihood": self.likelihood.state_dict()}

    @classmethod
    def from_state(cls, state):
        layer = cls.__new__(cls)
        layer.in_mean, layer.in_std = state["in_mean"].numpy(), state["in_std"].numpy()
        layer.out_mean, layer.out_std = state["out_mean"].item(), state["out_std"].item()
        inducing_shape = state["model"]["variational_strategy.inducing_points"].shape
        layer.model = ServiceGP(torch.zeros(inducing_shape, dtype=torch.float64)).double()
        layer.model.load_state_dict(state["model"])
        layer.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        layer.likelihood.load_state_dict(state["likelihood"])
        layer.model.eval()
        layer.likelihood.eval()
        return layer

    def predict(self, inputs, chunk=50_000):
        """Predictive mean and standard deviation (including the likelihood noise) of the layer output, in seconds."""
        x_all = (inputs - self.in_mean) / self.in_std
        means, sds = [], []
        with _single_thread(), torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(x_all), chunk):
                f = self.model(torch.tensor(x_all[start:start + chunk], dtype=torch.float64))
                means.append(f.mean.numpy())
                sds.append(np.sqrt(f.variance.numpy() + self.likelihood.noise.item()))
        return self.out_mean + self.out_std * np.concatenate(means), self.out_std * np.concatenate(sds)


class ChainDGP:
    """Layers for the positions 1, 2, ... of a chain; a chain of depth K uses the first K layers."""

    def __init__(self, layers, n_samples=2000, seed=0):
        self.layers, self.n_samples, self.seed = layers, n_samples, seed
        self._cache = None

    def samples(self, x_cfgs, depth):
        """Monte Carlo samples (configurations, n_samples) of the end-to-end latency after `depth` services. Every
        position of the chain runs with the same configuration; the samples of all depths are computed once per x_cfgs."""
        x_cfgs = np.atleast_2d(x_cfgs)
        key = (x_cfgs.shape, x_cfgs.tobytes())
        if self._cache is None or self._cache[0] != key:
            rng = np.random.default_rng(self.seed)
            x_repeated = np.repeat(x_cfgs, self.n_samples, axis=0)
            per_depth, latency = [], None
            for layer in self.layers:
                inputs = x_repeated if latency is None else np.column_stack([x_repeated, latency])
                mean, sd = layer.predict(inputs)
                latency = mean + sd * rng.standard_normal(len(mean))
                per_depth.append(latency.reshape(len(x_cfgs), self.n_samples))
            self._cache = (key, per_depth)
        return self._cache[1][depth - 1]

    def latency_bound(self, x_cfgs, epsilon, depth):
        """(1 - epsilon) quantile of the sampled end-to-end latency, per configuration. Not a guarantee."""
        return np.quantile(self.samples(x_cfgs, depth), 1 - epsilon, axis=1)


def basic_training_rows(x_train, y_by_service, path, seed=0):
    """Rows of DGP Basic, built as in the paper: row i is training configuration i; L_k = L_{k-1} + y_k (no queue).
    A service that appears again in the path uses its rows in a shuffled order, so that it does not repeat the same
    measurement (the paper's shuffling for repeated services). Returns one (inputs, targets) pair per position."""
    rng = np.random.default_rng(seed)
    n_rows = len(x_train)
    seen, rows, latency = set(), [], None
    for service in path:
        order = np.arange(n_rows) if service not in seen else rng.permutation(n_rows)
        seen.add(service)
        x_k, y_k = x_train[order], np.asarray(y_by_service[service])[order]
        inputs = x_k if latency is None else np.column_stack([x_k, latency])
        latency = y_k if latency is None else latency + y_k
        rows.append((inputs, latency.copy()))
    return rows


def advanced_training_rows(x_cfgs, node_latencies, n_rows, seed=0):
    """Rows of DGP Advanced from traced chain runs. node_latencies: (configurations, items, positions) measured latency
    of every item at every service; the cumulative latency after position k is their running sum. n_rows items are
    drawn at random (the same items for every position). Returns one (inputs, targets) pair per position."""
    cumulative = np.cumsum(node_latencies, axis=2)
    n_cfgs, n_items, n_positions = cumulative.shape
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_cfgs * n_items, size=min(n_rows, n_cfgs * n_items), replace=False)
    cfg, item = np.divmod(pick, n_items)
    rows = []
    for k in range(n_positions):
        inputs = x_cfgs[cfg] if k == 0 else np.column_stack([x_cfgs[cfg], cumulative[cfg, item, k - 1]])
        rows.append((inputs, cumulative[cfg, item, k]))
    return rows


def train_chain_dgp(rows, seed=0, n_samples=2000, cache_dir=None, **layer_options):
    """Trains one layer per position, each on its own rows (decoupled), and returns the chain model.
    With cache_dir, the trained layers are stored under a hash of the training rows, the seed, the layer options and the
    source code of the layer model, and loaded instead of retrained when all of them are unchanged. A relative cache_dir
    is resolved against the folder of this file, not against the working directory."""
    cache_file = None
    if cache_dir is not None:
        cache_root = pathlib.Path(cache_dir)
        if not cache_root.is_absolute():
            cache_root = pathlib.Path(__file__).resolve().parent / cache_root
        model_source = inspect.getsource(ServiceGP) + inspect.getsource(DGPLayer)
        digest = hashlib.sha1(repr((seed, sorted(layer_options.items()), model_source)).encode())
        for inputs, targets in rows:
            digest.update(np.ascontiguousarray(inputs, dtype=np.float64).tobytes())
            digest.update(np.ascontiguousarray(targets, dtype=np.float64).tobytes())
        cache_file = cache_root / f"chain_dgp_{digest.hexdigest()[:16]}.pt"
        if cache_file.exists():
            layers = [DGPLayer.from_state(state) for state in torch.load(cache_file, weights_only=True)]
            return ChainDGP(layers, n_samples=n_samples, seed=seed)
    layers = [DGPLayer(inputs, targets, seed=seed + k, **layer_options) for k, (inputs, targets) in enumerate(rows)]
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save([layer.state() for layer in layers], cache_file)
    return ChainDGP(layers, n_samples=n_samples, seed=seed)


# ==============================================================================
# 4. Fluxion-style baseline: modular learning of latency quantiles
# ==============================================================================
# Follows Fluxion (Liang et al., "On Modular Learning of Distributed Systems for Predicting End-to-End Latency",
# NSDI 2023). Fluxion models every service with "learning assignments": one regression model per service and per latency
# metric (e.g. the p90), whose inputs are the knobs of the service and a spectrum of latency percentiles of the services it
# depends on. The assignments are wired into an "inference graph" along the service dependencies, and the end-to-end
# prediction is the output of the last assignment after a traversal that passes point predictions along the edges.
#
# Mapping to the chain of `method_v2.ipynb`:
# - In Fluxion's RPC call graphs, the latency of a caller contains the latency of its callees. The counterpart in a
#   pipeline is the cumulative latency L_k of an item from entering the chain until it leaves the service at position k
#   (as for the DGP baselines), which contains L_{k-1}. So the assignments of position k depend on those of position k-1.
# - One assignment per position and per quantile level in `levels`: a GP (Matern 5/2 as in Fluxion's evaluation; the GP
#   of service_gp.py) with inputs (configuration x, the quantiles of L_{k-1} at all `levels`) and the quantile of L_k at
#   one level as output. Position 1 has no dependency and takes x only.
# - One training row per training configuration ("benchmark" in Fluxion): the quantiles measured on a traced chain run at
#   that configuration, which includes queueing. Like DGP Advanced, the composition must be traced before it can be
#   predicted. At inference, the measured upstream quantiles are replaced by the predictions of position k-1.
#
# Left out, because they have nothing to act on here: the request rate as an input (Fluxion's "observable state"; the
# arrival rate is the same in every training and test run), input selection (the spectrum has only a few levels), output
# weighting over models from different time periods, and service-vertices that aggregate replicas.
# The output is a point prediction of a quantile, not a guarantee: nothing keeps it above the true quantile.

class FluxionChain:
    """Inference graph of a chain: assignments[k][j] predicts the levels[j] quantile of the cumulative latency (s) after
    position k + 1; a chain of depth K uses the first K positions."""

    def __init__(self, assignments, levels):
        self.assignments, self.levels = assignments, np.asarray(levels, dtype=float)

    def quantiles(self, x_cfgs, depth):
        """Predicted quantiles (configurations, levels) of the end-to-end latency after `depth` services, by graph
        traversal: every position receives the point predictions of the position before it. Every position of the chain
        runs with the same configuration."""
        x_cfgs = np.atleast_2d(x_cfgs)
        upstream = None
        for models in self.assignments[:depth]:
            inputs = x_cfgs if upstream is None else np.column_stack([x_cfgs, upstream])
            upstream = np.column_stack([model.predict(inputs) for model in models])
        return upstream

    def latency_bound(self, x_cfgs, epsilon, depth):
        """Predicted (1 - epsilon) quantile of the end-to-end latency, per configuration. Not a guarantee. 1 - epsilon
        must be one of the trained levels: Fluxion needs one assignment per metric it predicts."""
        match = np.flatnonzero(np.isclose(self.levels, 1 - epsilon))
        if len(match) == 0:
            raise ValueError(f"no assignment was trained for the quantile level {1 - epsilon:g}; trained levels: {self.levels}")
        return self.quantiles(x_cfgs, depth)[:, match[0]]


def fluxion_training_rows(x_cfgs, node_latencies, levels):
    """Rows of the Fluxion-style baseline from traced chain runs. node_latencies: (configurations, items, positions)
    measured latency of every item at every service. Row i of every position is configuration i; the targets
    (configurations, levels) are the quantiles of the cumulative latency after that position over the items of the run,
    and the inputs are the configuration and the targets of the position before. Returns one (inputs, targets) pair per
    position."""
    quantiles = np.quantile(np.cumsum(node_latencies, axis=2), levels, axis=1)   # (levels, configurations, positions)
    rows = []
    for k in range(quantiles.shape[2]):
        inputs = x_cfgs if k == 0 else np.column_stack([x_cfgs, quantiles[:, :, k - 1].T])
        rows.append((inputs, quantiles[:, :, k].T))
    return rows


def train_fluxion_chain(rows, levels) -> FluxionChain:
    """Trains one learning assignment per position and quantile level, each on its own rows only (modular: replacing a
    service retrains the assignments of its position), and returns the inference graph of the chain."""
    assignments = []
    with warnings.catch_warnings():
        # the kernel bounds of service_gp.py are set for single execution times; a quantile over a whole traced run is
        # almost free of noise, so the fitted noise level ends at its lower bound and scikit-learn warns for every model
        warnings.simplefilter("ignore", ConvergenceWarning)
        for inputs, targets in rows:
            models = train_gp_models(inputs, dict(enumerate(targets.T)), range(len(levels)))
            assignments.append([models[j] for j in range(len(levels))])
    return FluxionChain(assignments, levels)


# ==============================================================================
# 5. Replay of monitored node latencies (Geebelen et al.)
# ==============================================================================
# Follows Geebelen et al., "QoS prediction for web service compositions using kernel-based quantile estimation with online
# adaptation of the constant offset", Information Sciences 268 (2014). Their method has two steps:
# 1. Aggregation: every elementary service is monitored on its own, which gives one time series of response times per
#    service. The response times of a composition are built by replaying these series through the workflow: an execution
#    that starts at time t takes the response time the first service had at t, enters the second service when the first
#    has finished, takes the response time the second service had at that moment, and so on (a sequence adds, a parallel
#    block takes the maximum). The composition itself is never executed.
# 2. Prediction: a quantile regressor is fitted to the replayed series and predicts the quantile the SLO asks for.
#
# Mapping to the chain of `method_v2.ipynb`:
# - Monitoring: one run per service and training configuration in which the service is alone and receives the arrival
#   process of the workload; the run records the arrival time and the latency (waiting + execution) of every item.
#   A service that occurs at several positions of a chain is replayed from the same monitored series at every position,
#   as in the paper, where one service can occur twice in a workflow.
# - Replay: `replay_composite_latencies`; "the response time the service had at time t" is the latency of the monitored
#   item that arrived closest to t.
# - Prediction: the paper predicts the quantile over time from a kernel quantile regression (pinball loss) with time
#   features. Here the quantile is predicted over configurations, in two stages that estimate the same conditional
#   quantile: the empirical quantile of the replayed latencies of every training configuration, and a GP (service_gp.py)
#   from the configuration to that quantile. One regressor per composition and quantile level, trained on first use:
#   as the paper states, every change of the composition needs a new regressor.
#
# Left out: the online adaptation of the constant offset, which raises or lowers the prediction after every observed
# response time, so that the violation frequency converges to the target over a long stream. The evaluation of the
# notebook has no such stream: every held-out configuration is predicted once. The parallel, switch and loop patterns of
# the paper do not occur in a chain.
# The replay keeps what each service did on its own. It does not contain what the composition changes: a downstream
# service receives the departures of the upstream one and not the arrival process it was monitored with, and the same item
# passes through all services. The output is a point prediction of a quantile, not a guarantee.

def replay_composite_latencies(arrivals_by_service, latencies_by_service, path):
    """Latencies (s) of a chain, built from runs that monitored every service on its own (Geebelen et al., step 1).
    arrivals_by_service[s], latencies_by_service[s]: (configurations, items) arrival time (s, increasing along the items)
    and latency (s) of every monitored item of service s; path: the service at every position of the chain.
    One replayed execution starts at every monitored arrival of the first service. Returns (configurations, items);
    executions that leave the period in which all services were monitored are NaN (the paper marks them "unknown")."""
    start = arrivals_by_service[path[0]]
    period_start = np.max([arrivals_by_service[s][:, 0] for s in set(path)], axis=0)[:, None]
    period_end = np.min([arrivals_by_service[s][:, -1] for s in set(path)], axis=0)[:, None]
    time = np.where(start >= period_start, start, np.nan)
    for s_name in path:
        arrivals, latencies = arrivals_by_service[s_name], latencies_by_service[s_name]
        for i in range(len(time)):
            lookup = np.nan_to_num(time[i], nan=arrivals[i, 0])
            after = np.clip(np.searchsorted(arrivals[i], lookup), 1, arrivals.shape[1] - 1)
            closest = np.where(lookup - arrivals[i, after - 1] <= arrivals[i, after] - lookup, after - 1, after)
            time[i] = time[i] + latencies[i, closest]
        time = np.where(time <= period_end, time, np.nan)
    return time - start


class ReplayComposition:
    """Replay of monitored node latencies with a quantile regressor over configurations (Geebelen et al., steps 1 and 2).
    x_cfgs: (configurations, features) the monitored configurations; arrivals_by_service, latencies_by_service: as in
    replay_composite_latencies. No composition is executed, neither for training nor for prediction."""

    def __init__(self, x_cfgs, arrivals_by_service, latencies_by_service):
        self.x_cfgs, self.arrivals, self.latencies = np.atleast_2d(x_cfgs), arrivals_by_service, latencies_by_service
        self.regressors = {}   # (path, quantile level) -> GP from the configuration to the replayed quantile

    def monitored_quantiles(self, path, level):
        """`level` quantile of the replayed latencies of the chain `path`, per monitored configuration."""
        return np.nanquantile(replay_composite_latencies(self.arrivals, self.latencies, path), level, axis=1)

    def latency_bound(self, x_cfgs, epsilon, path):
        """Predicted (1 - epsilon) quantile of the end-to-end latency of the chain `path`, per configuration. Not a
        guarantee. The regressor of a (path, epsilon) pair is trained on first use."""
        key = (tuple(path), 1 - epsilon)
        if key not in self.regressors:
            with warnings.catch_warnings():
                # as for the Fluxion-style baseline: a quantile over a whole run is almost free of noise, so the fitted
                # noise level ends at its lower bound and scikit-learn warns
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.regressors[key] = train_gp_models(self.x_cfgs, {0: self.monitored_quantiles(path, 1 - epsilon)}, [0])[0]
        return self.regressors[key].predict(np.atleast_2d(x_cfgs))


# ==============================================================================
# 6. Oracles from measured node latencies (configurations, items, nodes)
# ==============================================================================
def union_bound_oracle(node_latencies, epsilon):
    """Measured node quantiles at epsilon / K, added: the best any union bound over nodes can do, per configuration."""
    n_nodes = node_latencies.shape[-1]
    return sum(np.quantile(node_latencies[:, :, k], 1 - epsilon / n_nodes, axis=1) for k in range(n_nodes))


def independent_node_latencies(node_latencies, rng):
    """End-to-end latencies with the dependence between nodes removed: every node's latencies are shuffled across items,
    which keeps each node's distribution."""
    return sum(rng.permuted(node_latencies[:, :, k], axis=1) for k in range(node_latencies.shape[-1]))


def paired_independence_effect(node_latencies, level, rng, n_batches=10, confidence=0.95):
    """How much assuming independent nodes changes the `level` quantile of the end-to-end latency, per configuration,
    as a paired test. Both quantiles come from the same simulated items: the real one from the items as they are, the
    independent one from the same items with every node's latencies shuffled across items within consecutive blocks
    (so that the pairs stay matched). Most of the simulation noise affects both quantiles alike and cancels in their
    difference. Returns the relative difference (independent - real) / real and the upper edge of its two-sided
    confidence interval at `confidence` from batch means; an upper edge below 0 means that assuming independence
    significantly underestimates the quantile."""
    n_cfgs, n_items, n_nodes = node_latencies.shape
    blocks = np.array_split(np.arange(n_items), n_batches)
    shuffled = np.empty_like(node_latencies)
    for items in blocks:
        for k in range(n_nodes):
            shuffled[:, items, k] = rng.permuted(node_latencies[:, items, k], axis=1)
    real, independent = node_latencies.sum(axis=2), shuffled.sum(axis=2)
    real_quantile = np.quantile(real, level, axis=1)
    difference = np.quantile(independent, level, axis=1) - real_quantile
    block_differences = np.stack([np.quantile(independent[:, items], level, axis=1) - np.quantile(real[:, items], level, axis=1)
                                  for items in blocks], axis=1)
    standard_error = block_differences.std(axis=1, ddof=1) / np.sqrt(n_batches)
    upper = difference + student_t.ppf(0.5 + confidence / 2, n_batches - 1) * standard_error
    return difference / real_quantile, upper / real_quantile
