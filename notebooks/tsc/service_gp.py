"""Gaussian Process model of one service's execution time, used by `method_v2.ipynb` (Section 2).

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


def train_gp_models(X_train: np.ndarray, y_train_dict: dict, service_names: list) -> dict:
    """Fits ScaledGPRegressor models for each service."""
    models = {}
    for name in service_names:
        kernel = (
            C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )

        model = ScaledGPRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        model.fit(X_train, y_train_dict[name])
        models[name] = model

    return models


def execution_time_belief(model, x_cfg, confidence=0.95):
    """Belief about the execution time of one item at the configurations x_cfg: a pessimistic mean (s) and the variance of
    individual items (s^2). The GP's error about the mean (epistemic) is shared by every item at a configuration and does
    not average out over the items in a queue, so it shifts the mean, mu + z_c * sigma_epistemic, which lies above the true
    mean with probability `confidence`; the scatter of individual items (aleatoric) is the variance per item."""
    mu, _, var_epistemic, var_aleatoric = model.predict(x_cfg, return_variance_components=True)
    return mu + norm.ppf(confidence) * np.sqrt(var_epistemic), var_aleatoric
