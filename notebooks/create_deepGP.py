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


class ServiceChain(torch.nn.Module):
    """Connects 3 GPs into a chain where outputs of one become inputs to the next."""

    def __init__(self, qr_idx, cv_idx, pc_idx, num_of_points):
        super().__init__()
        self.qr_idx, self.cv_idx, self.pc_idx = qr_idx, cv_idx, pc_idx

        # input_dims = raw_features + 1 (from previous service sample)
        self.qr_gp = ServiceGP(input_dims=len(qr_idx), num_inducing=num_of_points)
        self.cv_gp = ServiceGP(input_dims=len(cv_idx) + 1, num_inducing=num_of_points)
        self.pc_gp = ServiceGP(input_dims=len(pc_idx) + 1, num_inducing=num_of_points)

        # self.qr_gp.mean_module.initialize(constant=0.5)
        # self.cv_gp.mean_module.initialize(constant=0.5) # QR(50) + CV(50)
        # self.pc_gp.mean_module.initialize(constant=0.5) # QR + CV + PC

        # self.qr_gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * len(qr_idx)])
        # self.cv_gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * (len(cv_idx) + 1)])
        # self.pc_gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.5] * (len(pc_idx) + 1)])

        self.qr_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.cv_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.pc_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        # Service 1: QR
        qr_dist = self.qr_gp(x[:, self.qr_idx])
        qr_samples = qr_dist.rsample()

        qr_tp_norm = qr_samples.unsqueeze(-1)
        # qr_qual = x[:, 1].unsqueeze(-1)

        # Service 2: CV
        cv_input = torch.cat([x[:, self.cv_idx], qr_tp_norm], dim=-1)
        cv_dist = self.cv_gp(cv_input)
        cv_samples = cv_dist.rsample()

        cv_tp_norm = cv_samples.unsqueeze(-1)

        # Service 3: PC
        pc_input = torch.cat([x[:, self.pc_idx], cv_tp_norm], dim=-1)
        pc_dist = self.pc_gp(pc_input)

        return qr_dist, cv_dist, pc_dist