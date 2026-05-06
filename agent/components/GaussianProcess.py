import logging
import sys
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import utils
from agent.components import RASK
from agent.components.RASK import get_dependent_variable_mapping
from agent.components.commons import ServiceType, ServiceVar

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GP_Model")


def normalize(val, v_min, v_max):
    if v_max <= v_min:
        return 0.0
    return (val - v_min) / (v_max - v_min)



def get_empirical_variable_bounds(df) -> Dict[ServiceType, Dict[ServiceVar, Tuple[float, float]]]:
    empirical_var_bounds = {}
    for s_type in df['service_type'].unique():
        df_s_type = df[df['service_type'] == s_type]
        s_type = ServiceType(s_type)
        variable_dep = get_dependent_variable_mapping(ServiceType(s_type))

        # The list of variables you care about (extracting from the dict)
        target_vars = variable_dep['max_tp']

        # Building the min/max dictionary from the original DataFrame
        empirical_boundary = {
            ServiceVar(var): (df_s_type[var].min(), df_s_type[var].max())
            for var in [*target_vars, "max_tp"]
        }
        empirical_var_bounds[s_type] = empirical_boundary

    # print(empirical_boundaries)
    return empirical_var_bounds


class GASK:
    def __init__(self, s_type: ServiceType, create_figures=False, display_figures=False):
        self.create_figures = create_figures
        self.display_figures = display_figures
        self.models: Dict[ServiceType, Dict] = {}
        self.training_data: pd.DataFrame = None
        self.s_type = s_type

    def init_model(self, df_combined: pd.DataFrame, data_density=1.0):
        if data_density < 1.0:
            # df_combined = df_combined.sample(frac=data_density, random_state=35)
            split_idx = int(len(df_combined) * data_density)
            df_combined = df_combined.iloc[:split_idx]

        df_cleared = RASK.preprocess_data(df_combined)
        # df_normalized = normalize_df(df_cleared)
        self.training_data = df_cleared
        self.models = self.train_gp_models(df_cleared)

    @utils.print_execution_time
    def train_gp_models(self, df: pd.DataFrame) -> Dict:
        service_models = {}

        for service_val in [self.s_type.value]:  # df['service_type'].unique():
            stype = ServiceType(service_val)
            df_service = df[df['service_type'] == service_val]
            service_models[stype] = {}

            dep_map = get_dependent_variable_mapping(stype)
            for var, deps in dep_map.items():
                X = df_service[deps].values
                y = df_service[var].values.reshape(-1, 1)

                # Kernel: Constant * RBF + WhiteKernel (Noise)
                # length_scale_bounds allows the GP to adapt to the scale of different metrics
                # Expanded bounds to prevent ConvergenceWarnings
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler

                from sklearn.gaussian_process.kernels import DotProduct

                # Linear trend + Non-linear RBF + Noise
                kernel = (C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e3))
                          + C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)))
                # + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))) # We don't have any noise
                gp_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('gp', GaussianProcessRegressor(
                        kernel=kernel,
                        # n_restarts_optimizer=10, # Especially needed when your (noise) kernels might be ill configured
                        alpha=0.1,
                        normalize_y=True  # Crucial: this scales your throughput/target automatically
                    ))
                ])

                logger.info(f"Fitting GP for {stype.value} - Target: {var}")
                gp_pipeline.fit(X, y)

                service_models[stype][var] = gp_pipeline

                if self.create_figures:
                    self.draw_3d_gp_plot(df_service, var, deps, gp_pipeline, stype.value)

        return service_models

    # @utils.print_execution_time
    def predict(self, service_type: ServiceType, dep_var: str, sample_state: Dict[ServiceVar, Any]):
        """Predicts mean and uncertainty."""
        if service_type not in self.models or dep_var not in self.models[service_type]:
            return None, None

        model = self.models[service_type][dep_var]
        deps = get_dependent_variable_mapping(service_type)[dep_var]

        # Ensure inputs are sorted to match training
        input_data = np.array([[sample_state[ServiceVar(k)] for k in sorted(deps)]])
        y_pred, sigma = model.predict(input_data, return_std=True)

        mu, sigma = y_pred[0], sigma[0]
        return mu, sigma  # np.random.normal(mu, sigma, 1)[0]

    def get_model_lml(self, service_type: ServiceType, dep_var: str) -> float:
        """Extracts the Log-Marginal Likelihood from the fitted GP pipeline."""
        if service_type not in self.models or dep_var not in self.models[service_type]:
            return None

        # 1. Get the pipeline
        pipeline = self.models[service_type][dep_var]

        # 2. Access the 'gp' step from the pipeline
        gp_model = pipeline.named_steps['gp']

        # 3. Return the LML
        # Note: .log_marginal_likelihood() returns the LML of the
        # optimized hyperparameters found during .fit()
        return gp_model.log_marginal_likelihood()

    @utils.print_execution_time
    def draw_3d_gp_plot(self, df, var, deps, gp, service_name):
        """
        Visualizes GP mean surface, ±95% confidence intervals, and actual data.
        Aligned with paper figure sizes (600px width approx).
        """
        grid_res = 30

        # 1. Coordinate Handling
        if len(deps) > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(df[deps].values)
            x_axis, y_axis = "PC1", "PC2"
            x_actual, y_actual = coords[:, 0], coords[:, 1]
        else:
            x_axis, y_axis = deps[0], deps[1]
            x_actual, y_actual = df[x_axis], df[y_axis]

        # Create 2D Meshgrid
        x_range = np.linspace(x_actual.min(), x_actual.max(), grid_res)
        y_range = np.linspace(y_actual.min(), y_actual.max(), grid_res)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points_pca = np.c_[xx.ravel(), yy.ravel()]

        grid_points_orig = pca.inverse_transform(grid_points_pca) if len(deps) > 2 else grid_points_pca

        # 2. GP Prediction
        y_pred, sigma = gp.predict(grid_points_orig, return_std=True)
        y_mean_grid = y_pred.reshape(xx.shape)
        sigma_grid = sigma.reshape(xx.shape)

        # 3. Confidence Intervals
        y_upper = y_mean_grid + 1.96 * sigma_grid
        y_lower = y_mean_grid - 1.96 * sigma_grid

        # 4. Construct Plotly Figure
        fig = go.Figure()

        # Trace 1: Mean Surface
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=y_mean_grid,
            colorscale='Viridis',
            name='Mean',
            showscale=False,
            # colorbar=dict(title=f"{var}", thickness=15, len=0.5),
            opacity=0.9
        ))

        # Trace 2 & 3: Confidence Bands (Simplified to one color)
        conf_style = dict(showscale=False, opacity=0.3, colorscale=[[0, 'grey'], [1, 'grey']])
        fig.add_trace(go.Surface(x=xx, y=yy, z=y_upper, name='+95% CI', **conf_style))
        fig.add_trace(go.Surface(x=xx, y=yy, z=y_lower, name='-95% CI', **conf_style))

        # Trace 4: Observations
        fig.add_trace(go.Scatter3d(
            x=x_actual, y=y_actual, z=df[var],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.8),
            name='Obs.',
        ))

        fig.update_layout(
            width=700, height=450,
            paper_bgcolor='white',
            plot_bgcolor='white',

            # 1. Eliminate outer margins entirely
            margin=dict(l=0, r=0, t=0, b=0),

            scene=dict(
                xaxis_title="Quality",
                yaxis_title="Resources",
                zaxis_title="Performance",
                xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),

                # 2. Adjust Camera: 'eye' values closer to 1.0 (default is ~1.25-2.0)
                # Reducing these values "zooms in," effectively removing internal white space.
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8),
                    center=dict(x=0, y=0, z=-0.1)  # Slightly offset center to lift the plot
                )
            ),

            # 3. Move legend inside the plot area to save space
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.95,
                xanchor="left", x=0.05,
                bgcolor='rgba(255, 255, 255, 0.5)'  # Semi-transparent background
            ),
            font=dict(size=10)
        )

        if self.display_figures:
            fig.show()

        filename = f"../figures/gp_{service_name}_{var}.pdf"
        # 4. Use crop/tight parameters if supported by your kaleido version
        fig.write_image(filename, engine="kaleido")
        logger.info(f"Saved 3D GP plot to {filename}")


# def get_ordered_boundaries(model: GASK):
#     raw_bounds = get_empirical_variable_bounds(model.training_data)[model.s_type]
#     del raw_bounds['max_tp']
#     return list(raw_bounds.values())


# --- Execution ---
if __name__ == "__main__":
    df = pd.read_csv("../../statics/metrics_TSC_EXPLORE.csv")
    # 2. Initialize and train
    rask_gp = GASK(show_figures=True)
    rask_gp.init_model(df, data_density=1.0)
    # rask_gp.init_models(df, data_density=0.5)
    # rask_gp.init_models(df, data_density=0.1)
    sys.exit()
