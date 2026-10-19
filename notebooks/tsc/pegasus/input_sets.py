"""`explain_7` and `explain_8`: which inputs the GP of a service needs to predict the mean execution time of an item on
a testbed campaign (quality knobs only, or with cores, parallelism, the cores per worker and the load), and how strongly
each input is associated with the mean execution time once the other inputs are held fixed.

Data: every valid run of a campaign (`statics/results_milos/` for explain_7, `statics/results_milos_v2/` for
explain_8) that contains the service, measured alone or inside a pipeline, one row per run and node. Target: the mean
processing time of the items of the service in that run (processing_mean_ms of dataset.csv, in ms). Model: the center
GP of RunStatisticsModel (service_gp.py), i.e. a GP on the natural logarithm of the run mean whose prediction is
exponentiated. Cross-validation folds are grouped by the full configuration of the service (quality knobs, cores and
parallelism), so that no held-out configuration occurs in the training data, also for the input sets that do not see
cores or parallelism.
"""
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from .real_data import CAMPAIGN, CAMPAIGN_V2, node_configurations
from .service_gp import MIN_LENGTH_SCALE, train_gp_models

# Input sets: name -> the columns of service_runs that are added to the quality knobs
INPUT_SETS = {"quality": [],
              "cores": ["cores"],
              "cores_and_parallelism": ["cores", "parallelism"],
              "with_load": ["cores", "parallelism", "load"],
              "cores_per_worker": ["cores_per_worker"],
              "cores_per_worker_with_load": ["cores_per_worker", "load"]}
RUN_COLUMNS = ("run_id", "experiment", "alone", "mean_ms", "load")   # columns of service_runs that are neither knobs nor inputs
CACHE = CAMPAIGN.parent / "input_sets"   # results of cross_validate_all, one folder per campaign


@lru_cache(maxsize=None)
def campaign_nodes(campaign: Path = CAMPAIGN) -> pd.DataFrame:
    """One row per experiment of `campaign` and node: experiment, alone (the experiment measures one service alone),
    node, service, and the valid rows of the experiment's dataset.csv (column table, the same object for every node of
    an experiment). An experiment is a folder with exactly one execution "<start time>/dataset.csv"; the folders
    "campaigns" and "results" of the first campaign hold no experiments."""
    rows = []
    for folder in sorted(path for path in campaign.iterdir() if path.is_dir() and path.name not in ("campaigns", "results")):
        executions = sorted(folder.glob("*/dataset.csv"))
        assert len(executions) == 1, f"{folder.name}: expected one execution, found {len(executions)}"
        table = pd.read_csv(executions[0])
        table = table[table["valid"]].reset_index(drop=True)
        nodes = table.drop_duplicates("node")
        rows += [dict(experiment=folder.name, alone="->" not in table["pipeline"].iloc[0], node=node, service=service, table=table)
                 for node, service in zip(nodes["node"], nodes["service"])]
    return pd.DataFrame(rows)


def services(campaign: Path = CAMPAIGN) -> list:
    """Every service of the campaign, e.g. "cv-analyzer:v1", sorted by name."""
    return sorted(campaign_nodes(campaign)["service"].unique())


def service_runs(service: str, campaign: Path = CAMPAIGN) -> pd.DataFrame:
    """Every valid run of `campaign` that contains `service`, one row per run and node: run_id, experiment, alone
    (measured alone or in a pipeline), the knobs of the service (cores, parallelism and its quality knobs), mean_ms (the
    mean processing time of its items in that run, ms) and load (the measured arrival rate at the node, items/s, column
    rate_received). As in training_pool, experiments that do not vary every knob of the service are left out, because
    the value of the missing knob is not stored. Knobs with the same value in every run (e.g. the queueing discipline
    of the second campaign) are dropped; cores and parallelism are always kept."""
    nodes = campaign_nodes(campaign)
    nodes = nodes[nodes["service"] == service]
    knobs = max((set(node_configurations(row.table, row.node).columns) for row in nodes.itertuples()), key=len)
    parts = []
    for row in nodes.itertuples():
        configurations = node_configurations(row.table, row.node)
        if set(configurations.columns) != knobs:
            continue
        statistics = row.table[row.table["node"] == row.node].set_index("run_id").loc[configurations.index]
        parts.append(configurations.reset_index().assign(experiment=row.experiment, alone=row.alone,
                                                         mean_ms=statistics["processing_mean_ms"].to_numpy(),
                                                         load=statistics["rate_received"].to_numpy()))
    runs = pd.concat(parts, ignore_index=True)
    constant = [column for column in knobs - {"cores", "parallelism"} if runs[column].nunique() == 1]
    return runs[(runs["mean_ms"] > 0) & runs["load"].notna()].drop(columns=constant).reset_index(drop=True)


def input_features(runs: pd.DataFrame, input_set: str) -> pd.DataFrame:
    """Inputs of the GP for every run of service_runs: the quality knobs (one 0/1 column per value of a knob that is a
    word) and the columns of INPUT_SETS[input_set]; cores_per_worker = cores / parallelism."""
    quality = runs[[column for column in runs.columns if column not in RUN_COLUMNS + ("cores", "parallelism")]]
    x = pd.get_dummies(quality, dtype=float).astype(float)
    extra = runs.assign(cores_per_worker=runs["cores"] / runs["parallelism"])
    for column in INPUT_SETS[input_set]:
        x[column] = extra[column]
    return x


def cross_validate(service: str, input_set: str, seeds=(0, 1, 2), n_folds: int = 5, n_restarts: int = 2,
                   campaign: Path = CAMPAIGN, ard: bool = False) -> pd.DataFrame:
    """Grouped k-fold cross-validation of the GP of `service` with the inputs `input_set`, repeated for every seed
    (each seed assigns the configurations to the folds at random). One row per seed and fold: n_test (held-out runs),
    mse (ms^2, of the predicted run mean), var_test (ms^2, variance of the held-out run means) and mse_log (of the
    natural logarithm, approximately the mean squared relative error). The splits depend only on the service and the
    seed, so all input sets are evaluated on the same folds.
    With ard, the GP has one length scale per input (train_gp_models with ard=True; without, one length scale for all
    inputs, as in explain_7 and Section 2 of explain_8), inputs with the same value in every run are left
    out, and every row also holds the fitted length scale of every input (columns "length_scale_<input>", in standard
    deviations of the input), the range of every input in the training runs (columns "range_<input>", largest minus
    smallest value, in the same unit) and the fitted signal variance (column "signal_variance", in units of the variance of
    ln(run mean) in the training runs, because the GP normalizes its target); see range_effect."""
    runs = service_runs(service, campaign)
    knobs = [column for column in runs.columns if column not in RUN_COLUMNS]
    groups = pd.factorize(runs[knobs].astype(str).agg("|".join, axis=1))[0]
    features = input_features(runs, input_set)
    if ard:
        features = features[[column for column in features.columns if features[column].nunique() > 1]]
    x, y = features.to_numpy(), runs["mean_ms"].to_numpy()
    rows = []
    for seed in seeds:
        # GroupKFold is deterministic; relabeling the groups at random gives a different split per seed
        relabeled = np.random.default_rng(seed).permutation(groups.max() + 1)[groups]
        for fold, (train, test) in enumerate(GroupKFold(n_folds).split(y, groups=relabeled)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # convergence warnings of the hyperparameter search
                model = train_gp_models(x[train], {service: np.log(y[train])}, [service], MIN_LENGTH_SCALE, n_restarts, ard=ard)[service]
            predicted = np.exp(model.predict(x[test]))
            row = dict(service=service, input_set=input_set, seed=seed, fold=fold, n_test=len(test),
                       mse=np.mean((predicted - y[test]) ** 2), var_test=np.var(y[test]),
                       mse_log=np.mean((np.log(predicted) - np.log(y[test])) ** 2))
            if ard:
                scaled = model.x_scaler.transform(x[train])
                row |= {f"length_scale_{name}": value for name, value in zip(features.columns, np.atleast_1d(model.gp.kernel_.k1.k2.length_scale))}
                row |= {f"range_{name}": value for name, value in zip(features.columns, scaled.max(axis=0) - scaled.min(axis=0))}
                row["signal_variance"] = model.gp.kernel_.k1.k1.constant_value
            rows.append(row)
    return pd.DataFrame(rows)


def range_effect(length_scale, input_range, signal_variance):
    """How much a GP of train_gp_models with ard=True lets its prediction differ between the smallest and the largest value of one input
    when every other input is fixed: the standard deviation of f(x_max) - f(x_min) under the fitted kernel,
    sqrt(2 * signal_variance * (1 - k(range / length_scale))), with k the Matern correlation (nu = 2.5). Unit: standard
    deviations of ln(run mean) in the training runs. Unlike the length scale alone, it separates an input without effect
    (long length scale, small value) from one with a smooth but large effect (long length scale, large signal variance)."""
    d = np.sqrt(5) * np.asarray(input_range, dtype=float) / np.asarray(length_scale, dtype=float)
    correlation = (1 + d + d ** 2 / 3) * np.exp(-d)
    return np.sqrt(2 * np.asarray(signal_variance, dtype=float) * (1 - correlation))


def cross_validate_all(input_set: str, campaign: Path = CAMPAIGN, recompute: bool = False, ard: bool = False) -> pd.DataFrame:
    """cross_validate for every service of `campaign`, read from CACHE/<campaign>/<input_set>.csv (<campaign>: its
    path below statics/ with "_" for "/", e.g. results_milos_v2_b1; with ard: CACHE/<campaign>_ard/) if that file exists (otherwise computed, which takes up to an hour per input set on
    the first campaign, and written there)."""
    path = CACHE / ("_".join(campaign.relative_to(CAMPAIGN.parent).parts) + ("_ard" if ard else "")) / f"{input_set}.csv"
    if path.exists() and not recompute:
        return pd.read_csv(path)
    results = pd.concat([cross_validate(service, input_set, campaign=campaign, ard=ard) for service in services(campaign)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def replicate_scatter(service: str, campaign: Path = CAMPAIGN) -> tuple:
    """Scatter of ln(run mean) between runs of `service` with the same full configuration (quality knobs, cores and
    parallelism): the pooled within-configuration variance, sum of squared deviations from the configuration's average
    / (runs - configurations), over the configurations with at least two runs. A model whose inputs are the knobs
    cannot predict a held-out run more precisely than this on average, so it is the floor of mse_log for the input sets
    without the load (the load can explain a part of it). Returns the variance and the number of configurations with repeated
    runs; (nan, 0) if there are none."""
    runs = service_runs(service, campaign)
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


# ------------------------------------------------------------------------------
# Conditional independence of an input and the mean execution time
# ------------------------------------------------------------------------------
def partial_correlation_test(data: pd.DataFrame, x: str, y: str, z: list) -> tuple:
    """Test of x independent of y given z, as in constraint-based structure learning (gaussCItest of pcalg, fisherz of
    causal-learn), on ranks: every column is replaced by its ranks (Spearman), x and y are regressed on z with an
    intercept, and the correlation r of the two residuals is the partial rank correlation. Fisher's z transform
    sqrt(n - |z| - 3) * atanh(r) is standard normal under independence if the ranks are jointly Gaussian after the
    transform (the nonparanormal assumption: every relation is monotone, the runs are independent). Returns r and the
    two-sided p-value."""
    ranks = data[[x, y, *z]].rank()
    design = np.column_stack([np.ones(len(ranks)), ranks[z].to_numpy()])
    residual = lambda column: ranks[column] - design @ np.linalg.lstsq(design, ranks[column], rcond=None)[0]
    r = float(np.corrcoef(residual(x), residual(y))[0, 1])
    statistic = np.sqrt(len(ranks) - len(z) - 3) * np.arctanh(np.clip(r, -0.999999, 0.999999))
    return r, float(2 * stats.norm.sf(abs(statistic)))


def input_associations(service: str, campaign: Path = CAMPAIGN, with_load: bool = False) -> pd.DataFrame:
    """Partial rank correlation of every input of the service with ln(run mean), given all other inputs, one row per
    input: input, r, p. Inputs: the quality knobs (numeric, or the 0/1 columns of a knob that is a word, without the
    first value), cores and parallelism, and with with_load also the load. The cores per worker is a function of cores
    and parallelism, so it is tested in their place: given the quality knobs (and the load). An input with the same
    value in every run (e.g. the parallelism of a service that was only run with one worker) is left out."""
    runs = service_runs(service, campaign)
    quality = runs[[column for column in runs.columns if column not in RUN_COLUMNS + ("cores", "parallelism")]]
    data = pd.get_dummies(quality, drop_first=True, dtype=float).assign(cores=runs["cores"], parallelism=runs["parallelism"],
                                                                        cores_per_worker=runs["cores"] / runs["parallelism"],
                                                                        load=runs["load"], log_mean=np.log(runs["mean_ms"]))
    data = data[[column for column in data.columns if data[column].nunique() > 1]]
    knobs = [column for column in data.columns if column not in ("cores_per_worker", "load", "log_mean")] + (["load"] if with_load else [])
    rows = []
    for column in knobs:
        rows.append((column, *partial_correlation_test(data, column, "log_mean", [k for k in knobs if k != column])))
    if "cores_per_worker" in data and "parallelism" in data:   # with one worker, the ratio equals the cores
        given = [k for k in knobs if k not in ("cores", "parallelism")]
        rows.append(("cores_per_worker", *partial_correlation_test(data, "cores_per_worker", "log_mean", given)))
    return pd.DataFrame(rows, columns=["input", "r", "p"]).assign(service=service)


def holm_adjusted(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values of several tests (NaN entries are ignored): an adjusted p-value below alpha
    rejects with a family-wise error rate of at most alpha, without assuming that the tests are independent."""
    p = np.asarray(p, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    steps = np.maximum.accumulate((len(order) - np.arange(len(order))) * p[order])
    adjusted[order] = np.minimum(steps, 1.0)
    return adjusted
