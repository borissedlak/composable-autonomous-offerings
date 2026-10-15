"""Gaussian Process model of one service's execution time, used by `1_method_synthetic.ipynb` (Section 2).

`ScaledGPRegressor` wraps scikit-learn's GaussianProcessRegressor with feature scaling and returns its standard
prediction (mean and predictive standard deviation). The predictive variance can be split into its epistemic part
Var(f(x)) and its aleatoric part sigma_n^2 (the WhiteKernel noise), both in physical units (s^2).
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class ScaledGPRegressor:
    """
    Wrapper around GaussianProcessRegressor that handles feature (X) scaling
    and leverages native `normalize_y=True` for target scaling.

    predict() is the standard GP prediction: the mean and the predictive standard deviation of one new observation,
    V[Y(x)] = Var(f(x)) + sigma_n^2. On request it also returns the two parts of that variance:
    - Epistemic uncertainty: Var(f(x)) = V[Y(x)] - sigma_n^2
    - Aleatoric uncertainty: sigma_n^2 (from WhiteKernel, unscaled to physical units)
    """
    def __init__(self, kernel, n_restarts_optimizer=5, random_state=42):
        self.x_scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,  # Handles y-scaling natively & unscales predict() outputs
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state
        )

    def fit(self, X, y):
        X_scaled = self.x_scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        return self

    def predict(self, X, return_std=False, return_variance_components=False):
        X_scaled = self.x_scaler.transform(X)

        # Standard GP prediction: mean and predictive standard deviation of one new observation, natively unscaled by
        # sklearn. The kernel contains a WhiteKernel, so this standard deviation already includes the noise of one item.
        mu, std = self.gp.predict(X_scaled, return_std=True)

        if return_variance_components:
            # Noise level sigma_n^2 of the WhiteKernel, converted from the scaled target to physical units
            y_var_scale = getattr(self.gp, '_y_train_std', 1.0) ** 2
            if hasattr(self.gp.kernel_, 'k2') and hasattr(self.gp.kernel_.k2, 'noise_level'):
                var_aleatoric = np.full_like(mu, self.gp.kernel_.k2.noise_level * y_var_scale)
            else:
                var_aleatoric = np.zeros_like(mu)
            # What remains of the predictive variance is Var(f(x)), the uncertainty about the mean function
            var_epistemic = np.maximum(std ** 2 - var_aleatoric, 0.0)
            return mu, std, var_epistemic, var_aleatoric
        if return_std:
            return mu, std

        return mu


def configurations_of(x, service: str) -> np.ndarray:
    """Configurations of one service. x is either one array (configurations, features) that every service shares, or
    a dictionary service -> array for services that each have their own knobs and their own runs."""
    return x[service] if isinstance(x, dict) else x


# Lower limit of the length scale of every GP on real data, in standard deviations of a knob (the inputs are standardized).
# Without a limit, the hyperparameter search of some services ends at a length scale near zero: the GP then reproduces
# every training run and predicts the average of all runs everywhere else. Any limit between 0.3 and 2 gave the same
# fits on the chains of the campaign, because a search that cannot collapse finds a length scale above 2 (checked with one
# length scale for all inputs, before train_gp_models used one length scale per input).
MIN_LENGTH_SCALE = 0.3


def train_gp_models(X_train, y_train_dict: dict, service_names: list, min_length_scale: float = None, n_restarts_optimizer: int = 5,
                    ard: bool = True) -> dict:
    """Fits ScaledGPRegressor models for each service. X_train: one array for all services or one per service
    (configurations_of). min_length_scale: lower limit of the length scale (default: that of scikit-learn, 1e-5).
    n_restarts_optimizer: restarts of the hyperparameter search from random starting points.
    ard: one length scale per input (automatic relevance determination), so that the hyperparameter search can give an
    input that does not matter a long length scale; with ard=False, one length scale for all inputs (the model of the
    results before 2026-10-14; explain_7 and Section 2 of explain_8 compare the two)."""
    models = {}
    length_scale_bounds = (1e-5, 1e5) if min_length_scale is None else (min_length_scale, 1e5)
    for name in service_names:
        x = configurations_of(X_train, name)
        kernel = (
            C(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(np.shape(x)[1]) if ard else 1.0, length_scale_bounds=length_scale_bounds, nu=2.5) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )

        model = ScaledGPRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=42)
        model.fit(x, y_train_dict[name])
        models[name] = model

    return models


class RunStatisticsModel:
    """Execution-time model of one service that is trained on statistics of whole runs instead of on single items
    (real data). `center` predicts the mean execution time of a configuration and was trained on the natural logarithm
    of the mean of every training run (log_center, the standard), so its prediction is exponentiated: the effects of
    the knobs of a real service multiply, and on the logarithm they add, which a GP with one length scale per knob
    can fit.
    With log_center=False, `center` was trained on the run means themselves. `spread` is a GP on the natural logarithm
    of the standard deviation (s) of single items in a run.
    predict() has the interface of ScaledGPRegressor: the mean and, on request, the standard deviation of one item."""
    def __init__(self, center, spread, log_center: bool = True):
        self.center, self.spread, self.log_center = center, spread, log_center

    def mean(self, x_cfg, shift: float = 0.0):
        """Predicted mean execution time (s), shifted by `shift` standard deviations of the prediction for a new run:
        the uncertainty about the mean function and the scatter between runs at the same configuration (for a GP as
        center; another kind of model has no such standard deviation and is not shifted). With log_center the shift
        is applied to the logarithm, so it is relative: exp(mu + shift * std)."""
        if not hasattr(self.center, "gp"):
            return self.center.predict(x_cfg)
        mu, std = self.center.predict(x_cfg, return_std=True)
        return np.exp(mu + shift * std) if self.log_center else mu + shift * std

    def predict(self, x_cfg, return_std=False, return_variance_components=False):
        mean = self.mean(x_cfg)
        if return_variance_components:
            # as ScaledGPRegressor: mean, total standard deviation, the variance of the mean of a new run (uncertainty about
            # the mean function and scatter between runs; from the upward shift by one standard deviation) and of single items
            var_mean = (self.mean(x_cfg, 1.0) - mean) ** 2
            var_items = np.exp(self.spread.predict(x_cfg)) ** 2
            return mean, np.sqrt(var_mean + var_items), var_mean, var_items
        return (mean, np.exp(self.spread.predict(x_cfg))) if return_std else mean


def train_run_statistics_models(X_train, y_mean_dict: dict, y_spread_dict: dict, service_names: list, log_center: bool = True) -> dict:
    """One RunStatisticsModel per service: a GP on the logarithm of the mean of every training run (y_mean_dict, s;
    on the means themselves with log_center=False) and a GP on the logarithm of the standard deviation of single items
    in every training run (y_spread_dict, s). Both have the lower limit MIN_LENGTH_SCALE of the length scale."""
    targets = {name: np.log(y_mean_dict[name]) for name in service_names} if log_center else y_mean_dict
    center = train_gp_models(X_train, targets, service_names, MIN_LENGTH_SCALE)
    spread = train_gp_models(X_train, {name: np.log(y_spread_dict[name]) for name in service_names}, service_names, MIN_LENGTH_SCALE)
    return {name: RunStatisticsModel(center[name], spread[name], log_center) for name in service_names}


def run_statistics_belief(model: RunStatisticsModel, x_cfg, confidence=0.95):
    """Belief about the execution time from a RunStatisticsModel, in the form of execution_time_belief: a pessimistic
    mean (s) and the variance of individual items (s^2). The pessimistic mean is the predicted mean of a new run
    shifted up by z_gamma standard deviations of that prediction (gamma = `confidence`), so that it lies above the mean of a new run with
    probability `confidence` if the GP's predictive distribution is right. The variance is the square of the predicted
    standard deviation of single items; it has no margin of its own."""
    return model.mean(x_cfg, norm.ppf(confidence)), np.exp(model.spread.predict(x_cfg)) ** 2


def execution_time_belief(model, x_cfg, confidence=0.95):
    """Belief about the execution time of one item at the configurations x_cfg: a pessimistic mean (s) and the variance of
    individual items (s^2). The GP's error about the mean (epistemic) is shared by every item at a configuration and does
    not average out over the items in a queue, so it shifts the mean, m + z_gamma * sigma_epistemic, which lies above the true
    mean with probability gamma = `confidence`; the scatter of individual items (aleatoric) is the variance per item."""
    mu, _, var_epistemic, var_aleatoric = model.predict(x_cfg, return_variance_components=True)
    return mu + norm.ppf(confidence) * np.sqrt(var_epistemic), var_aleatoric
