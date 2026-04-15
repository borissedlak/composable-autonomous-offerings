import ast
import logging
from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from agent.components.RASK import draw_3d_plot_interactive
from agent.components.commons import ServiceType

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GP_Model")


class GASK:
    def __init__(self, show_figures=True):
        self.show_figures = show_figures
        self.models: Dict[ServiceType, Dict] = {}

    def init_models(self, df_combined: pd.DataFrame):
        """Preprocesses data and trains Gaussian Process models."""
        df_cleared = self.preprocess_data(df_combined)
        self.models = self.train_gp_models(df_cleared)

    def preprocess_data(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()

        # Handle string representation of dicts if present
        if 's_config' in df.columns:
            df['s_config'] = df['s_config'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            metadata_expanded = pd.json_normalize(df['s_config'])
            df = pd.concat([df.drop(columns=['s_config']), metadata_expanded], axis=1)

        df['model_size'] = df['model_size'].fillna(-1)

        # Calculate max throughput based on latency
        if 'avg_p_latency' in df.columns:
            df['max_tp'] = np.where(df['avg_p_latency'] > 0, (1000 / df['avg_p_latency']), 0)
            # Adjust QR service for cores
            qr_mask = df['service_type'] == ServiceType.QR.value
            if 'cores' in df.columns:
                df.loc[qr_mask, 'max_tp'] = df.loc[qr_mask, 'max_tp'] * df.loc[qr_mask, 'cores'].round()

        df.reset_index(drop=True, inplace=True)
        return df

    def get_dependent_variable_mapping(self, service_type: ServiceType):
        """Defines which independent variables influence the target variable."""
        mapping = {
            ServiceType.QR: {'max_tp': sorted(['cores', 'data_quality'])},
            ServiceType.CV: {'max_tp': sorted(['cores', 'model_size', 'data_quality'])},
            ServiceType.PC: {'max_tp': sorted(['cores', 'data_quality'])}
        }
        return mapping.get(service_type, {})

    def train_gp_models(self, df: pd.DataFrame) -> Dict:
        service_models = {}

        for service_val in ['elastic-workbench-qr-detector']: #df['service_type'].unique():
            stype = ServiceType(service_val)
            df_service = df[df['service_type'] == service_val]
            service_models[stype] = {}

            dep_map = self.get_dependent_variable_mapping(stype)
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
                kernel = (C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e3)) +
                          C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)) +
                          WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3)))
                # In your train_gp_models function:
                gp_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('gp', GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=10,
                        alpha=0.1,
                        normalize_y=True  # Crucial: this scales your throughput/target automatically
                    ))
                ])


                logger.info(f"Fitting GP for {stype.value} - Target: {var}")
                gp_pipeline.fit(X, y)

                service_models[stype][var] = gp_pipeline

                if self.show_figures:
                    self.draw_3d_gp_plot(df_service, var, deps, gp_pipeline, stype.value)

        return service_models

    def predict(self, service_type: ServiceType, dep_var: str, sample_state: Dict[str, Any]):
        """Predicts mean and uncertainty."""
        if service_type not in self.models or dep_var not in self.models[service_type]:
            return None, None

        model = self.models[service_type][dep_var]
        deps = self.get_dependent_variable_mapping(service_type)[dep_var]

        # Ensure inputs are sorted to match training
        input_data = np.array([[sample_state[k] for k in sorted(deps)]])
        y_pred, sigma = model.predict(input_data, return_std=True)

        return y_pred[0][0], sigma[0]

    def draw_3d_gp_plot(self, df, var, deps, gp, service_name):
        """
        Visualizes GP mean surface, ±95% confidence intervals, and actual data.
        """
        # Define visualization parameters
        grid_res = 30  # Adjust for surface smoothness

        # 1. Coordinate Handling (Keep existing PCA or raw logic)
        if len(deps) > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(df[deps].values)
            x_axis, y_axis = "PC1", "PC2"
            x_actual = coords[:, 0]
            y_actual = coords[:, 1]
        else:
            x_axis, y_axis = deps[0], deps[1]
            x_actual = df[x_axis]
            y_actual = df[y_axis]

        # Create 2D Meshgrid
        x_range = np.linspace(x_actual.min(), x_actual.max(), grid_res)
        y_range = np.linspace(y_actual.min(), y_actual.max(), grid_res)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points_pca = np.c_[xx.ravel(), yy.ravel()]

        # Transform visualization grid back to model space if PCA was used
        if len(deps) > 2:
            grid_points_orig = pca.inverse_transform(grid_points_pca)
        else:
            grid_points_orig = grid_points_pca

        # 2. Key GP Prediction Step: Get Mean AND StdDev
        y_pred, sigma = gp.predict(grid_points_orig, return_std=True)

        # Reshape predictions back to grid shape
        y_mean_grid = y_pred.reshape(xx.shape)
        sigma_grid = sigma.reshape(xx.shape)

        # 3. Calculate Confidence Intervals
        # We use ±1.96 * sigma for 95% confidence interval
        y_upper = y_mean_grid + 1.96 * sigma_grid
        y_lower = y_mean_grid - 1.96 * sigma_grid

        # 4. Construct Plotly Figure
        fig = go.Figure()

        # Trace 1: Predicted Mean Surface (The Function)
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=y_mean_grid,
            colorscale='Viridis',
            name='GP Mean Prediction',
            colorbar=dict(title=f"Predicted {var}", x=-0.12),
            opacity=0.9
        ))

        # Trace 2: Upper Confidence Surface (σ over function)
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=y_upper,
            colorscale=[[0, 'rgba(100, 100, 100, 0.2)'], [1, 'rgba(100, 100, 100, 0.2)']],
            name='+95% Conf. Interval',
            showscale=False,
            opacity=0.3
        ))

        # Trace 3: Lower Confidence Surface (σ over function)
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=y_lower,
            colorscale=[[0, 'rgba(100, 100, 100, 0.2)'], [1, 'rgba(100, 100, 100, 0.2)']],
            name='-95% Conf. Interval',
            showscale=False,
            opacity=0.3
        ))

        # Trace 4: Actual Observations (Markers)
        fig.add_trace(go.Scatter3d(
            x=x_actual,
            y=y_actual,
            z=df[var],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.8),
            name='Observations'
        ))

        # Define Layout
        fig.update_layout(
            title={
                'text': f'RASK GP: {service_name} Performance ({var}) with 95% Confidence Band',
                'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            scene=dict(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                zaxis_title=var,
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6)  # Flatten Z-axis slightly for better surface viewing
            ),
            width=1200, height=900,
            legend=dict(x=0, y=1)
        )

        filename = f"gp_uncertainty_{service_name}_{var}.html"
        fig.write_html(filename)
        logger.info(f"Saved uncertainty visualization to {filename}")


# --- Execution ---
if __name__ == "__main__":

    df = pd.read_csv("../../statics/metrics_20_0.csv")
    # 2. Initialize and train
    rask_gp = GASK(show_figures=True)
    rask_gp.init_models(df)

    # 3. Predict a sample
    # test_sample = {'cores': 4.0, 'data_quality': 0.8}
    # mean, std = rask_gp.predict(ServiceType.QR, 'max_tp', test_sample)
    # print(f"\nPrediction for QR Service at {test_sample}:")
    # print(f"Predicted Max Throughput: {mean:.2f}")
    # print(f"Uncertainty (StdDev): {std:.2f}")