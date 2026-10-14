"""`explain_7`: which inputs the GP of a service needs to predict the mean execution time of an item on the testbed
campaign: only the quality knobs, the quality knobs and the cores per worker, the quality knobs with cores and
parallelism, and each of the last two with the load.

Data: every valid run of the campaign in `statics/results_milos/` that contains the service, measured alone or inside
a pipeline, one row per run and node. Target: the mean processing time of the items of the service in that run
(processing_mean_ms of dataset.csv, in ms). Model: the center GP of RunStatisticsModel (service_gp.py), i.e. a GP on
the natural logarithm of the run mean whose prediction is exponentiated. Cross-validation folds are grouped by the full
configuration of the service (quality knobs, cores and parallelism), so that no held-out configuration occurs in the
training data, also for the input sets that do not see cores or parallelism.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .real_data import CAMPAIGN, campaign_nodes, node_configurations, run_table
from .service_gp import MIN_LENGTH_SCALE, train_gp_models

# Input sets, in the order of the notebook: name -> the columns of service_runs that are added to the quality knobs
INPUT_SETS = {"quality": [],
              "cores_per_worker": ["cores_per_worker"],
              "cores_and_parallelism": ["cores", "parallelism"],
              "with_load": ["cores", "parallelism", "load"],
              "cores_per_worker_with_load": ["cores_per_worker", "load"]}
RUN_COLUMNS = ("run_id", "experiment", "alone", "mean_ms", "load")   # columns of service_runs that are neither knobs nor inputs
CACHE = CAMPAIGN.parent / "input_sets"


def services() -> list:
    """Every service of the campaign, e.g. "cv-analyzer:v1", sorted by name."""
    return sorted(campaign_nodes()["service"].unique())


def service_runs(service: str) -> pd.DataFrame:
    """Every valid run of the campaign that contains `service`, one row per run and node: run_id, experiment, alone
    (measured alone or in a pipeline), the knobs of the service (cores, parallelism and its quality knobs), mean_ms (the
    mean processing time of its items in that run, ms) and load (the measured arrival rate at the node, items/s, column
    rate_received). As in training_pool, experiments that do not vary every knob of the service are left out, because
    the value of the missing knob is not stored."""
    nodes = campaign_nodes()
    nodes = nodes[nodes["service"] == service]
    knobs = max((set(node_configurations(run_table(row.experiment), row.node).columns) for row in nodes.itertuples()), key=len)
    parts = []
    for row in nodes.itertuples():
        table = run_table(row.experiment)
        configurations = node_configurations(table, row.node)
        if set(configurations.columns) != knobs:
            continue
        statistics = table[table["node"] == row.node].set_index("run_id").loc[configurations.index]
        parts.append(configurations.reset_index().assign(experiment=row.experiment, alone=row.alone,
                                                         mean_ms=statistics["processing_mean_ms"].to_numpy(),
                                                         load=statistics["rate_received"].to_numpy()))
    runs = pd.concat(parts, ignore_index=True)
    return runs[(runs["mean_ms"] > 0) & runs["load"].notna()].reset_index(drop=True)


def input_features(runs: pd.DataFrame, input_set: str) -> np.ndarray:
    """Inputs of the GP for every run of service_runs: the quality knobs (one 0/1 column per value of a knob that is a
    word) and the columns of INPUT_SETS[input_set]; cores_per_worker = cores / parallelism."""
    quality = runs[[column for column in runs.columns if column not in RUN_COLUMNS + ("cores", "parallelism")]]
    x = pd.get_dummies(quality, dtype=float).astype(float)
    extra = runs.assign(cores_per_worker=runs["cores"] / runs["parallelism"])
    for column in INPUT_SETS[input_set]:
        x[column] = extra[column]
    return x.to_numpy()


def cross_validate(service: str, input_set: str, seeds=(0, 1, 2), n_folds: int = 5, n_restarts: int = 2) -> pd.DataFrame:
    """Grouped k-fold cross-validation of the GP of `service` with the inputs `input_set`, repeated for every seed
    (each seed assigns the configurations to the folds at random). One row per seed and fold: n_test (held-out runs),
    mse (ms^2, of the predicted run mean), var_test (ms^2, variance of the held-out run means) and mse_log (of the
    natural logarithm, approximately the mean squared relative error). The splits depend only on the service and the
    seed, so all input sets are evaluated on the same folds."""
    runs = service_runs(service)
    knobs = [column for column in runs.columns if column not in RUN_COLUMNS]
    groups = pd.factorize(runs[knobs].astype(str).agg("|".join, axis=1))[0]
    x, y = input_features(runs, input_set), runs["mean_ms"].to_numpy()
    rows = []
    for seed in seeds:
        # GroupKFold is deterministic; relabeling the groups at random gives a different split per seed
        relabeled = np.random.default_rng(seed).permutation(groups.max() + 1)[groups]
        for fold, (train, test) in enumerate(GroupKFold(n_folds).split(y, groups=relabeled)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # convergence warnings of the hyperparameter search
                model = train_gp_models(x[train], {service: np.log(y[train])}, [service], MIN_LENGTH_SCALE, n_restarts)[service]
            predicted = np.exp(model.predict(x[test]))
            rows.append(dict(service=service, input_set=input_set, seed=seed, fold=fold, n_test=len(test),
                             mse=np.mean((predicted - y[test]) ** 2), var_test=np.var(y[test]),
                             mse_log=np.mean((np.log(predicted) - np.log(y[test])) ** 2)))
    return pd.DataFrame(rows)


def cross_validate_all(input_set: str, recompute: bool = False) -> pd.DataFrame:
    """cross_validate for every service, read from CACHE/<input_set>.csv if that file exists (about 15 minutes per
    input set otherwise; the file is written after the computation)."""
    path = CACHE / f"{input_set}.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    results = pd.concat([cross_validate(service, input_set) for service in services()], ignore_index=True)
    path.parent.mkdir(exist_ok=True)
    results.to_csv(path, index=False)
    return results


def mse_per_service(results: pd.DataFrame) -> pd.DataFrame:
    """One row per service and input set: mse, mse_log and nmse (mse / variance of the run means of the service), each
    pooled over the folds of a seed (weighted by the held-out runs) and averaged over the seeds."""
    weighted = results.assign(sse=results["mse"] * results["n_test"], sse_log=results["mse_log"] * results["n_test"],
                              sst=results["var_test"] * results["n_test"])
    pooled = weighted.groupby(["service", "input_set", "seed"])[["sse", "sse_log", "sst", "n_test"]].sum()
    pooled = pooled.assign(mse=pooled["sse"] / pooled["n_test"], mse_log=pooled["sse_log"] / pooled["n_test"],
                           nmse=pooled["sse"] / pooled["sst"])
    return pooled.groupby(["service", "input_set"])[["mse", "mse_log", "nmse"]].mean().reset_index()


def replicate_scatter(service: str) -> tuple:
    """Scatter of ln(run mean) between runs of `service` with the same full configuration (quality knobs, cores and
    parallelism): the pooled within-configuration variance, sum of squared deviations from the configuration's average
    / (runs - configurations), over the configurations with at least two runs. A model whose inputs are the knobs
    cannot predict a held-out run more precisely than this on average, so it is the floor of mse_log for 1a to 1c (the
    load of 1d and 1e can explain a part of it). Returns the variance and the number of configurations with repeated
    runs; (nan, 0) if there are none."""
    runs = service_runs(service)
    knobs = [column for column in runs.columns if column not in RUN_COLUMNS]
    log_mean = np.log(runs["mean_ms"])
    groups = runs[knobs].astype(str).agg("|".join, axis=1)
    sizes = groups.map(groups.value_counts())
    repeated = sizes > 1
    if not repeated.any():
        return np.nan, 0
    deviations = log_mean[repeated] - log_mean[repeated].groupby(groups[repeated]).transform("mean")
    n_configurations = groups[repeated].nunique()
    return float((deviations ** 2).sum() / (repeated.sum() - n_configurations)), n_configurations
