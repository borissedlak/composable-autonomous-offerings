"""Per-service baselines of `2_evaluation.ipynb`. Like the proposed method, each is trained only on data of every service
on its own; none sees a run of the chain. They differ in what is measured per service:

1. Execution-time model per service, summed, no queue: the joint Gaussian quantile of the predicted execution times,
   with the GP of `service_gp.py` or with a neural network (ServiceDNN).
2. Latency quantile per service under load (waiting plus execution of the service alone, fed by the arrival process of
   the workload): the sum of learned quantiles, without and with a conformal step. The proposed method needs execution
   times only.
3. Queueing formula per service: a textbook approximation of the latency quantile of one queue.
4. Reference: the union bound over measured node latencies.

The baselines that are trained on runs of the chain are in `baselines_chain.py`.
Units: execution times and latencies in seconds; `epsilon` / `alpha` is the violation probability.
"""
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning

from .service_gp import train_gp_models


# ==============================================================================
# 1. Execution-time model per service, summed, no queue
# ==============================================================================
def joint_gaussian_quantile(node_means, node_stds, z_score):
    """(1 - epsilon) quantile of the sum of independent Gaussian execution times, no queue."""
    std_total = np.sqrt(sum(std ** 2 for std in node_stds))
    return sum(node_means) + z_score * std_total


@contextmanager
def _single_thread():
    # the matrices are tiny: torch's thread pool costs far more than it gains (5-18x slower with 16 threads)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class ServiceDNN:
    """Feed-forward network of one service's execution time (s) as a function of the configuration: two hidden layers,
    standardized inputs and target, trained on the mean squared error. The standard deviation it reports is that of its
    training residuals, the same at every configuration (a network has no predictive variance of its own)."""

    def __init__(self, inputs, targets, seed, hidden=16, iterations=1000, learning_rate=0.01, weight_decay=1e-3):
        self.in_mean, self.in_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-12
        self.out_mean, self.out_std = targets.mean(), targets.std() + 1e-12
        x = torch.tensor((inputs - self.in_mean) / self.in_std, dtype=torch.float64)
        y = torch.tensor((targets - self.out_mean) / self.out_std, dtype=torch.float64)
        with _single_thread():
            torch.manual_seed(seed)
            self.network = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.Tanh(),
                                               torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                               torch.nn.Linear(hidden, 1)).double()
            optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for _ in range(iterations):
                optimizer.zero_grad()
                loss = torch.mean((self.network(x).squeeze(-1) - y) ** 2)
                loss.backward()
                optimizer.step()
            self.network.eval()
        self.residual_std = np.std(targets - self.predict(inputs), ddof=1)

    def predict(self, inputs, return_std=False):
        x = torch.tensor((np.atleast_2d(inputs) - self.in_mean) / self.in_std, dtype=torch.float64)
        with _single_thread(), torch.no_grad():
            mean = self.out_mean + self.out_std * self.network(x).squeeze(-1).numpy()
        return (mean, np.full_like(mean, self.residual_std)) if return_std else mean


def train_dnn_models(x_train: np.ndarray, y_train_dict: dict, service_names: list, seed: int = 0) -> dict:
    """One ServiceDNN per service, the counterpart of train_gp_models in service_gp.py."""
    return {name: ServiceDNN(x_train, np.asarray(y_train_dict[name]), seed=seed + k) for k, name in enumerate(service_names)}


# ==============================================================================
# 2. Latency quantile per service under load
# ==============================================================================
def _service_quantile_models(x_train, monitored_latencies, services, level):
    """GP (service_gp.py) from the configuration to the `level` latency quantile of every service monitored alone."""
    targets = {s: np.quantile(monitored_latencies[s], level, axis=1) for s in services}
    with warnings.catch_warnings():
        # as for the Fluxion-style baseline: a quantile over a whole run is almost free of noise, so the fitted noise
        # level ends at its lower bound and scikit-learn warns
        warnings.simplefilter("ignore", ConvergenceWarning)
        return train_gp_models(x_train, targets, services), targets


def summed_service_quantiles(x_train: np.ndarray, monitored_latencies: dict, path: list, epsilon: float,
                             x_test: np.ndarray) -> np.ndarray:
    """Sum over the services of `path` of their predicted (1 - epsilon) latency quantile, per configuration of x_test.
    monitored_latencies[s]: (training configurations, items) latency (s) of every item of service s monitored alone under
    the arrival process of the workload. One GP per service predicts the quantile from the configuration: the modular
    learning of latency quantiles of Fluxion (part 4), with every model trained on its own service only. Not a guarantee:
    a point prediction per service, and the quantile of a sum is not the sum of the quantiles."""
    services = sorted(set(path))
    models, _ = _service_quantile_models(x_train, monitored_latencies, services, 1 - epsilon)
    predicted = {s: models[s].predict(np.atleast_2d(x_test)) for s in services}
    return sum(predicted[s] for s in path)


def conformal_service_bounds(x_train: np.ndarray, monitored_latencies: dict, path: list, epsilon: float, x_test: np.ndarray,
                             miscoverage: float, n_calibration: int = 0, seed: int = 0, x_calibration: np.ndarray = None,
                             calibration_latencies: dict = None) -> np.ndarray:
    """Split-conformal latency bound per service with the violation probability split over the services (the union
    bound), summed over `path`, per configuration of x_test.
    Per service: n_calibration training configurations, drawn at random, are set aside. A GP on the remaining ones
    predicts the (1 - epsilon / K) latency quantile of the service monitored alone (K = len(path)). On the set-aside
    configurations, the scores are relative, measured quantile / predicted quantile, so that the correction scales with
    the latency; every prediction is multiplied by their ceil((n + 1)(1 - miscoverage)) / n empirical quantile (inf if
    n = n_calibration is too small for that level). For configurations exchangeable with the training ones, the
    corrected prediction then lies above the service's quantile with probability >= 1 - miscoverage, per service; over K
    services the union bound leaves 1 - K * miscoverage. It assumes that a service behaves in the chain as it did
    alone, in particular that it receives the arrival process it was monitored with.
    With x_calibration and calibration_latencies (further configurations monitored in the same way), the model is
    trained on all training configurations and calibrated on those instead; n_calibration is then not used."""
    services, n_nodes = sorted(set(path)), len(path)
    if x_calibration is None:
        order = np.random.default_rng(seed).permutation(len(x_train))
        fit, calibrate = order[:-n_calibration], order[-n_calibration:]
        x_calibration, calibration_latencies = x_train[calibrate], {s: monitored_latencies[s][calibrate] for s in services}
        x_train, monitored_latencies = x_train[fit], {s: monitored_latencies[s][fit] for s in services}
    models, _ = _service_quantile_models(x_train, monitored_latencies, services, 1 - epsilon / n_nodes)
    predict = lambda s, x: np.maximum(models[s].predict(np.atleast_2d(x)), 1e-9)   # a latency is positive
    rank = int(np.ceil((len(x_calibration) + 1) * (1 - miscoverage)))
    bounds = {}
    for s in services:
        measured = np.quantile(calibration_latencies[s], 1 - epsilon / n_nodes, axis=1)
        scores = np.sort(measured / predict(s, x_calibration))
        correction = scores[rank - 1] if rank <= len(scores) else np.inf
        bounds[s] = predict(s, x_test) * correction
    return sum(bounds[s] for s in path)


# ==============================================================================
# 3. Queueing formula per service
# ==============================================================================
def queueing_formula_latency(mean, var, process, rate, epsilon, jitter=0.1, iterations=40, cv2=1.0):
    """(1 - epsilon) latency quantile (s) of one FIFO single-server queue from textbook queueing formulas, per
    configuration; inf where the service is overloaded. mean, var: execution time of one item (s, s^2); process, rate,
    jitter: the arrival process as in bounds.py.
    1. Mean waiting time, Kingman's approximation for a G/G/1 queue (Allen-Cunneen form):
       E[Q] = rho / (1 - rho) * (C_a^2 + C_s^2) / 2 * mean, with the utilization rho = rate * mean, the squared coefficient
       of variation of the execution time C_s^2 = var / mean^2 and that of the gaps between arrivals C_a^2 (1 for
       Poisson arrivals, where the formula is the exact Pollaczek-Khinchine mean; jitter^2 / 3 for periodic arrivals
       with uniform jitter; cv2 for gamma arrivals, see bounds.py).
    2. Waiting-time distribution, the usual exponential approximation: an item does not wait with probability 1 - rho
       and otherwise waits an exponential time with mean E[Q] / rho.
    3. Latency = waiting + own execution time (Gaussian, independent of the wait); its tail has a closed form and the
       quantile is found by bisection.
    An approximation, not a bound: nothing keeps it above the true quantile, and it uses the mean and variance as given,
    without any allowance for the error of the model that predicted them."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    sd, rho = np.sqrt(var), rate * mean
    arrival_cv2 = {"poisson": 1.0, "periodic": jitter ** 2 / 3, "gamma": cv2}[process]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_wait = rho / (1 - rho) * (arrival_cv2 + var / mean ** 2) / 2 * mean
        decay = rho / mean_wait   # rate of the exponential waiting time of the items that wait

    def tail(d):   # P(waiting + execution > d)
        with np.errstate(over="ignore", invalid="ignore"):
            waits = rho * np.exp(-decay * (d - mean) + 0.5 * decay ** 2 * var) * norm.cdf((d - mean - decay * var) / sd)
        return norm.sf((d - mean) / sd) + waits

    low, high = mean.copy(), mean + 60.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_small = tail(middle) > epsilon
        low, high = np.where(too_small, middle, low), np.where(too_small, high, middle)
    return np.where(rho < 1, high, np.inf)


# ==============================================================================
# 4. Reference: union bound over measured node latencies (configurations, items, nodes)
# ==============================================================================
def union_bound_oracle(node_latencies, epsilon):
    """Measured node quantiles at epsilon / K, added: the best any union bound over nodes can do, per configuration."""
    n_nodes = node_latencies.shape[-1]
    return sum(np.quantile(node_latencies[:, :, k], 1 - epsilon / n_nodes, axis=1) for k in range(n_nodes))
