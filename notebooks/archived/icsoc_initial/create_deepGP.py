from typing import List

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from agent.components import RASK
from agent.components.commons import ServiceFeatureMapping, ServiceType


def get_prepared_metrics_df(path="../statics/agent_experience/metrics_ICSOC_EXPLORE.csv",
                            offset: float = 0.0, share: float = 1.0):
    _raw_df = pd.read_csv(path)
    starting_index = int(len(_raw_df) * offset)
    first_x_percent = int(len(_raw_df) * share)
    _trimmed_df = _raw_df.iloc[starting_index:first_x_percent].reset_index(drop=True)
    converted_df = RASK.preprocess_data(_trimmed_df)
    return converted_df


class DynamicServiceChain(torch.nn.Module):
    def __init__(self, service_configs: List[ServiceFeatureMapping], num_inducing: int = 64):
        super().__init__()
        self.configs = service_configs
        self.gp_layers = torch.nn.ModuleList()
        self.likelihoods = torch.nn.ModuleList()

        for i, config in enumerate(service_configs):
            input_dims = len(config.feature_indices)
            if i > 0:
                input_dims += 1  # Previous service output

            gp = ServiceGP(input_dims=input_dims, num_inducing=num_inducing)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_layers.append(gp)
            self.likelihoods.append(likelihood)

    def forward(self, x, boundary_indices: List[int]):
        dists = []
        last_output = None

        for i, gp in enumerate(self.gp_layers):
            indices = self.configs[i].feature_indices
            current_input = x[:, indices]

            # During training, we want to block the gradients from
            # flowing from one service chunk to another
            if last_output is not None:
                if i in boundary_indices:
                    inp_sample = last_output.detach()
                else:
                    inp_sample = last_output

                current_input = torch.cat([current_input, inp_sample], dim=-1)

            dist = gp(current_input)
            dists.append(dist)
            last_output = dist.rsample().unsqueeze(-1)

        return tuple(dists)


def prepare_chained_data(df: pd.DataFrame, service_configs: List[ServiceFeatureMapping]):

    # 1. Split the interleaved rows based on the number of services in the chain
    # This allows the df to contain 3, 6, 9... services
    # service_dfs = [
    #     df.iloc[i::num_services].copy().reset_index(drop=True)
    #     for i in range(num_services)
    # ]

    df_qr = df.iloc[0::3].copy().reset_index(drop=True)
    df_cv = df.iloc[1::3].copy().reset_index(drop=True)
    df_pc = df.iloc[2::3].copy().reset_index(drop=True)

    base_dfs = {
        ServiceType.QR: df_qr,
        ServiceType.CV: df_cv,
        ServiceType.PC: df_pc
    }

    # 2. Scale each service throughput to log(max)
    # for b_df in base_dfs:
    #     # Use log1p (log(1+x)) to handle cases where throughput might be 0
    #     b_df['log_tp'] = np.log1p(b_df['max_tp'])
    #     tp_max = b_df['log_tp'].max()
    #     b_df['scaled_tp'] = b_df['log_tp'] / tp_max
    for s_type in base_dfs:
        target_df = base_dfs[s_type]
        target_df['log_tp'] = np.log1p(target_df['max_tp'])
        tp_max = target_df['log_tp'].max()
        target_df['scaled_tp'] = target_df['log_tp'] / tp_max

    # service_to_df = {ServiceType.QR: base_dfs[0], ServiceType.CV: base_dfs[1], ServiceType.PC: base_dfs[2]}
    # service_dfs = [service_to_df[service_conf.service_type] for service_conf in service_configs]

    service_dfs = []
    service_counts = {ServiceType.QR: 0, ServiceType.CV: 0, ServiceType.PC: 0}

    for config in service_configs:
        s_type = config.service_type
        base = base_dfs[s_type].copy()

        # If this service type has appeared before, shift the data
        # to ensure the model sees "different" configurations at each stage
        count = service_counts[s_type]
        if count > 0:
            # Shift rows down by 'count' and wrap around (circular shift)
            # This ensures Stage 1 CV and Stage 2 CV aren't identical
            base = base.sample(frac=1, random_state=35 + count).reset_index(drop=True)

        service_dfs.append(base)
        service_counts[s_type] += 1

    # 3. Link the service performance (Bottleneck logic)
    # Each service is capped by the performance of the one immediately preceding it
    y_columns = []
    current_bottleneck = None

    for i in range(len(service_configs)):
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

    return X_final, Y_final, scaler_X

def create_training_data(X_final, Y_final, test_size: float):
    # 5. Split and Tensors
    x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=test_size)

    t_x_train = torch.tensor(x_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    t_x_test = torch.tensor(x_test, dtype=torch.float32)
    t_y_test = torch.tensor(y_test, dtype=torch.float32)

    dataloader = DataLoader(
        TensorDataset(t_x_train, t_y_train),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return dataloader, t_x_test, t_y_test


# TODO: Ildefons recommended to inject some noise between the services
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

    t_x_train = torch.tensor(x_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    t_x_test = torch.tensor(x_test, dtype=torch.float32)
    t_y_test = torch.tensor(y_test, dtype=torch.float32)

    dataloader = DataLoader(
        TensorDataset(t_x_train, t_y_train),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return dataloader, t_x_test, t_y_test, scaler_X


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
