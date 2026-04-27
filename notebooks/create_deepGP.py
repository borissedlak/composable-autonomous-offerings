# %% [markdown]
# ### Setup Data loader
# %%
from typing import List

import gpytorch
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from agent.components.commons import ServiceFeatureMapping, ServiceType
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def prepare_chained_data(df: pd.DataFrame, service_configs: List[ServiceFeatureMapping], test_size: float):

    # This splits the training samples between ALL individual services, i.e., also between different QRs
    num_services = len(service_configs)

    # 1. Split the interleaved rows based on the number of services in the chain
    # This allows the df to contain 3, 6, 9... services
    service_dfs = [
        df.iloc[i::num_services].copy().reset_index(drop=True)
        for i in range(num_services)
    ]

    # 2. Scale each service throughput to quantiles
    # Maintaining the 0-1 range for the GP targets
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
    for s_df in service_dfs:
        s_df['scaled_tp'] = qt.fit_transform(s_df[['max_tp']])

    # # 1. Split the interleaved rows
    # df_qr = df.iloc[0::3].copy().reset_index(drop=True)
    # df_cv = df.iloc[1::3].copy().reset_index(drop=True)
    # df_pc = df.iloc[2::3].copy().reset_index(drop=True)
    #
    # # 2. Scale each service throughput to quantiles
    # # Thus, the values are not truly representative as they are now
    # # Plus the y values are automatically scaled to a range of 0 to 1
    # qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
    # df_qr['scaled_tp'] = qt.fit_transform(df_qr[['max_tp']])
    # df_cv['scaled_tp'] = qt.fit_transform(df_cv[['max_tp']])
    # df_pc['scaled_tp'] = qt.fit_transform(df_pc[['max_tp']])

    # 3. Link the service performance (Bottleneck logic)
    # Each service is capped by the performance of the one immediately preceding it
    y_columns = []
    current_bottleneck = None

    for i in range(num_services):
        raw_tp = service_dfs[i]['scaled_tp'].values
        if current_bottleneck is None:
            current_bottleneck = raw_tp
        else:
            # The "story": No service can exceed the throughput of the previous link
            current_bottleneck = np.minimum(current_bottleneck, raw_tp)

        y_columns.append(current_bottleneck)

    Y_final = np.column_stack(y_columns)

    # 4. Prepare Features (X)
    # We dynamically build X based on the known column structure for QR, CV, and PC
    x_parts = []
    for i, config in enumerate(service_configs):
        s_df = service_dfs[i]

        # Logic specific to your service types:
        if config.service_type == ServiceType.CV:
            # CV has cores, data_quality, AND model_size
            features = s_df[['cores', 'data_quality', 'model_size']].values
        else:
            # QR and PC only have cores and data_quality
            features = s_df[['cores', 'data_quality']].values

        x_parts.append(features)

    X_raw = np.hstack(x_parts)

    # Standardize Features
    scaler_X = MinMaxScaler()
    X_final = scaler_X.fit_transform(X_raw)

    # 5. Split and Tensors
    x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=test_size)

    t_x_train = torch.tensor(x_train, dtype=torch.float64)
    t_y_train = torch.tensor(y_train, dtype=torch.float64)
    t_x_test = torch.tensor(x_test, dtype=torch.float64)
    t_y_test = torch.tensor(y_test, dtype=torch.float64)

    dataloader = DataLoader(
        TensorDataset(t_x_train, t_y_train),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return dataloader, t_x_test, t_y_test, scaler_X


def prepare_chained_data_error(df: pd.DataFrame, service_configs: List[ServiceFeatureMapping], test_size: float):

    # This splits the training samples between ALL individual services, i.e., also between different QRs
    num_services = len(service_configs)

    # 1. Split the interleaved rows based on the number of services in the chain
    # This allows the df to contain 3, 6, 9... services
    service_dfs = [
        df.iloc[i::num_services].copy().reset_index(drop=True)
        for i in range(num_services)
    ]

    # 2. Scale each service throughput to quantiles
    # Maintaining the 0-1 range for the GP targets
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
    for s_df in service_dfs:
        s_df['scaled_tp'] = qt.fit_transform(s_df[['max_tp']])

    # 3. Link the service performance (Bottleneck logic)
    # Each service is capped by the performance of the one immediately preceding it
    y_columns = []
    current_bottleneck = None

    for i in range(num_services):
        raw_tp = service_dfs[i]['scaled_tp'].values
        if current_bottleneck is None:
            current_bottleneck = raw_tp
        else:
            # The "story": No service can exceed the throughput of the previous link
            current_bottleneck = np.minimum(current_bottleneck, raw_tp)

        y_columns.append(current_bottleneck)

    Y_final = np.column_stack(y_columns)

    # 4. Prepare Features (X)
    # We dynamically build X based on the known column structure for QR, CV, and PC
    x_parts = []
    for i, config in enumerate(service_configs):
        s_df = service_dfs[i]

        if config.service_type == ServiceType.CV:
            features = np.ones((len(s_df), 1))
        else:
            # QR and PC only have cores and data_quality
            features = s_df[['cores', 'data_quality']].values

        x_parts.append(features)

    X_raw = np.hstack(x_parts)

    # Standardize Features
    scaler_X = MinMaxScaler()
    X_final = scaler_X.fit_transform(X_raw)

    # 5. Split and Tensors
    x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=test_size)

    t_x_train = torch.tensor(x_train, dtype=torch.float64)
    t_y_train = torch.tensor(y_train, dtype=torch.float64)
    t_x_test = torch.tensor(x_test, dtype=torch.float64)
    t_y_test = torch.tensor(y_test, dtype=torch.float64)

    dataloader = DataLoader(
        TensorDataset(t_x_train, t_y_train),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return dataloader, t_x_test, t_y_test, scaler_X


# --- 1. MODEL DEFINITION ---

class ServiceGP(gpytorch.models.ApproximateGP):
    """A standard Variational GP for an individual service."""

    def __init__(self, input_dims, num_inducing):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        )
        # self.covar_module.initialize(outputscale=torch.tensor([100.0]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
