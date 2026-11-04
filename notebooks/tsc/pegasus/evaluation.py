"""What the experiments in the evaluation (`2a_base_case.ipynb` to `2e_iterative.ipynb`) share: the per-service models, the comparison of the proposed method with
every baseline, the coverage of a method, and the figures. The notebook only calls these functions; the bounds
themselves are in `bounds.py` and the baselines in `baselines.py`.

Units: execution times and latencies in seconds; `alpha` is the violation probability epsilon and `confidence` the
probability with which the pessimistic mean of a bound lies above the true mean.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .baselines import (conformal_service_bounds, joint_gaussian_quantile, queueing_formula_latency, summed_service_quantiles, train_dnn_models,
                        union_bound_oracle)
from .service_gp import (RunStatisticsModel, configurations_of, execution_time_belief, run_statistics_belief, train_gp_models,
                         train_run_statistics_models)
from .bounds import martingale_closed_form_node_bound, martingale_closed_form_node_bounds, network_service_bound
from .data import model_inputs, path_workers
from .trees import leaf_to_root_paths, max_over_paths, max_path_sum

PROPOSED = "Pegasus Node (ours)"   # union bound over the node bounds
PROPOSED_SNC = "Pegasus SNC (ours)"    # one bound for the whole chain from its network service curve
GROUP_COLORS = {"proposed": "tab:green", "baseline": "tab:gray", "reference": "tab:purple"}
METHOD_COLORS = {PROPOSED: "tab:green", PROPOSED_SNC: "tab:olive", "GP Sum": "tab:gray", "DNN Sum": "black", "GP Queue": "tab:blue",
                 "Quantile Union": "tab:orange", "Conformal": "tab:red", "Oracle": "tab:purple"}


# ------------------------------------------------------------------------------
# Models and methods
# ------------------------------------------------------------------------------
def fit_services(data, x_train, y_train: dict, model_seed: int, confidence: float, with_dnn: bool = True) -> dict:
    """Per-service models of the execution time, trained on single-service data: a GP and a DNN per service, and the
    beliefs Pegasus builds from the GPs at the held-out configurations of `data`, per position of its chain (a
    pessimistic mean with a per-item variance, and the optimistic mean that the propagated bounds need). "seconds" holds
    the time it took to train the GPs ("gp") and the DNNs ("dnn"), and to compute the beliefs from the trained GPs
    ("beliefs"). model_seed: seed of the DNN training. with_dnn=False skips the DNNs ("dnn" is then None), for
    experiments that leave out DNN Sum.
    If the data has the spread of every training run (data.y_train_spread, real data), y_train holds run means: the GP
    of a service is then fitted on the logarithm of the run means and predicts the mean execution time, a further GP
    per service predicts the standard deviation of single items from the spreads, and the beliefs come from both
    (RunStatisticsModel in service_gp.py). The time to train the spread GPs is part of "gp". The DNN is trained on the
    run means and uses the same spread GP.
    No model receives the number of workers of a configuration (model_inputs in data.py).
    A real tree (TreeData with services) trains one model per service type and computes the beliefs at the
    configurations of every node (model_of)."""
    model = model_of(data)
    services = sorted(set(model.values()))
    x_train, x_test = model_inputs(x_train, data.workers_column), model_inputs(data.x_test, data.workers_column)
    from_runs = data.y_train_spread is not None
    belief_of = run_statistics_belief if from_runs else execution_time_belief
    started = time.perf_counter()
    if from_runs:
        gp = train_run_statistics_models(x_train, y_train, data.y_train_spread, services)
        spread = {s: gp[s].spread for s in services}
    else:
        gp = train_gp_models(x_train, y_train, services)
    gp_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dnn = train_dnn_models(x_train, y_train, services, seed=model_seed) if with_dnn else None
    if from_runs and with_dnn:
        dnn = {s: RunStatisticsModel(dnn[s], spread[s], log_center=False) for s in services}
    dnn_seconds = time.perf_counter() - started
    started = time.perf_counter()
    belief = {key: belief_of(gp[model[key]], configurations_of(x_test, key), confidence) for key in sorted(model)}
    lower_mean = {key: belief_of(gp[model[key]], configurations_of(x_test, key), 1 - confidence)[0] for key in sorted(model)}
    belief_seconds = time.perf_counter() - started
    return dict(gp=gp, dnn=dnn, seconds=dict(gp=gp_seconds, dnn=dnn_seconds, beliefs=belief_seconds),
                beliefs=[belief[key] for key in data.path], lower_means=[lower_mean[key] for key in data.path])


def model_of(data) -> dict:
    """Key of the held-out configurations of every position of `data` (an entry of data.path) -> the service type
    whose models it uses: the key itself, except for a real tree, whose path holds nodes (TreeData.services)."""
    return dict(zip(data.path, getattr(data, "services", None) or data.path))


def pegasus_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus Node: the union bound over the closed-form martingale node bounds, per held-out configuration (s). The
    waiting time at a service is bounded by Kingman's martingale bound exp(-theta w); with a Gaussian execution time,
    the latency bound of the service follows from one formula. Every service is bounded with the external arrivals
    (the paper's assumption on internal arrivals). A service with several workers is bounded by cyclic assignment (bounds.py).
    Plus data.handover, the time between the services that the measured end-to-end latency contains."""
    K = len(data.path)
    return (martingale_closed_form_node_bounds(models["beliefs"], epsilon=alpha / K, path_workers=path_workers(data), **data.workload).sum(axis=0)
            + data.handover)


def pegasus_snc_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus SNC: one bound for the whole chain from its network service curve, per held-out configuration (s). One
    Chernoff step on the product of the services' latency moments bounds the end-to-end latency at the full violation
    probability alpha, instead of alpha / K per service. Multiplying the moments requires that the execution times of
    an item are independent between the services, so no Hoelder step is applied (all exponents 1). Plus data.handover,
    as in pegasus_bound."""
    return network_service_bound(models["beliefs"], epsilon=alpha, holder_exponents=[1] * len(data.path), path_workers=path_workers(data),
                                 **data.workload) + data.handover


def compare_methods(data, models: dict, x_train, monitored_latencies: dict, alpha: float, confidence: float,
                    n_calibration: int, model_seed: int, leave_out: tuple = (), x_train_load=None) -> tuple:
    """Both versions of Pegasus, every baseline and the reference for the chain of `data` at its held-out configurations. Returns two
    dictionaries: name -> (group, end-to-end latency per configuration in s), and name -> (training time, evaluation
    time) in seconds; the reference has no time. The training time is the time to train the models the method uses:
    the GPs or DNNs of fit_services, which several methods share, or the method's own models. The evaluation time is
    the time to compute the method's latency at the held-out configurations from the trained models, including the
    predictions of these models (for both versions of Pegasus: the beliefs of fit_services).
    models: from fit_services; x_train, monitored_latencies: the training configurations and the latencies of every
    service monitored alone at them (None leaves out the baselines that need them). x_train_load: the inputs of these
    baselines if they differ from x_train (see ChainData.x_train_load); the held-out inputs are then data.x_test_load. The union bound gives every service
    the share alpha / K of the violation probability. model_seed: seed of the calibration split of Conformal.
    leave_out: names of methods that are not computed.
    With real data (data.y_train_spread is given), the GPs of Quantile Union and Conformal are fitted on the logarithm
    of the latency quantile, like the GP of the mean execution time (fit_services): the latency of a real service
    differs by orders of magnitude between its configurations. No model receives the number of workers of a
    configuration (model_inputs in data.py); GP Queue and both versions of Pegasus get it from path_workers.
    Every method and the Oracle add data.handover, the time between the services that the measured end-to-end latency
    contains and no service latency does (both versions of Pegasus inside pegasus_bound and pegasus_snc_bound)."""
    K, workload = len(data.path), data.workload
    workers_by_position = path_workers(data) or [1] * K
    log_target = data.y_train_spread is not None
    x_load, x_test_load = (x_train, data.x_test) if x_train_load is None else (x_train_load, data.x_test_load)
    x_load, x_test_load = model_inputs(x_load, data.workers_column), model_inputs(x_test_load, data.workers_column)   # the rate stays the last column
    x_test = model_inputs(data.x_test, data.workers_column)
    predictions = lambda fitted: [fitted[s].predict(configurations_of(x_test, s), return_std=True) for s in data.path]
    gp_seconds, dnn_seconds = models["seconds"]["gp"], models["seconds"]["dnn"]
    # name, group, time to train the models the method shares with others, function that computes its latencies
    candidates = [
        (PROPOSED, "proposed", gp_seconds, lambda: pegasus_bound(data, models, alpha)),
        (PROPOSED_SNC, "proposed", gp_seconds, lambda: pegasus_snc_bound(data, models, alpha)),
        ("GP Sum", "baseline", gp_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["gp"])), norm.ppf(1 - alpha))),
        ("DNN Sum", "baseline", dnn_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["dnn"])), norm.ppf(1 - alpha))),
        ("GP Queue", "baseline", gp_seconds, lambda: sum(queueing_formula_latency(mean, std ** 2, epsilon=alpha, workers=workers, **workload)
                                                        for (mean, std), workers in zip(predictions(models["gp"]), workers_by_position))),
    ]
    if monitored_latencies is not None:   # these two train their own models and measure that time themselves (None here)
        candidates.append(("Quantile Union", "baseline", None, lambda: summed_service_quantiles(
            x_load, monitored_latencies, data.path, alpha / K, x_test_load, return_seconds=True, log_target=log_target)))
        if min(len(configurations_of(x_train, s)) for s in data.path) - n_calibration >= 5:   # Conformal needs configurations left to train its model on
            candidates.append(("Conformal", "baseline", None, lambda: conformal_service_bounds(
                x_load, monitored_latencies, data.path, alpha, x_test_load, miscoverage=1 - confidence, n_calibration=n_calibration, seed=model_seed,
                return_seconds=True, log_target=log_target)))
    # the beliefs are the GP predictions of both versions of Pegasus; the baselines predict inside their own function
    earlier_seconds = {PROPOSED: models["seconds"]["beliefs"], PROPOSED_SNC: models["seconds"]["beliefs"]}
    methods, seconds = {}, {}
    for name, group, training_seconds, compute in candidates:
        if name in leave_out:
            continue
        started = time.perf_counter()
        latency = compute()
        elapsed = time.perf_counter() - started
        if training_seconds is None:
            latency, training_seconds = latency
            elapsed -= training_seconds
        # the two versions of Pegasus add the handover time themselves (pegasus_bound), because other code calls them directly
        methods[name] = (group, latency if name in (PROPOSED, PROPOSED_SNC) else latency + data.handover)
        seconds[name] = (training_seconds, elapsed + earlier_seconds.get(name, 0.0))
    if data.node_latencies is not None:         # reference: what a union bound reaches with measured instead of predicted nodes
        methods["Oracle"] = ("reference", union_bound_oracle(data.node_latencies, alpha) + data.handover)
    return methods, seconds


def compare_tree_methods(data, models: dict, alpha: float, confidence: float, n_calibration: int, model_seed: int,
                         leave_out: tuple = ()) -> tuple:
    """compare_methods for a fork-join tree (TreeData in data.py) whose services have one worker each. Every method
    extends from a chain to the tree through the P leaf-to-root paths (trees.py): the end-to-end latency of a frame is the
    maximum over the paths of its latency through the path run as a chain, so bounds of every path that each fail with
    probability at most alpha / P give, as their maximum, a bound of the tree at alpha (union bound over the paths).
    - Pegasus Node: the union bound of every path over its node bounds, with every (path, service) pair at
      alpha / sum_p K_p (K_p services on path p), every service bounded with the external arrivals. The tree gets the
      largest sum over the paths.
    - Pegasus SNC: the network service curve of every path at alpha / P, maximum over the paths.
    - GP Sum: the joint Gaussian quantile of every path at alpha, maximum over the paths (a point prediction per path).
    - GP Queue: the quantile of every service at alpha, added along every path, maximum over the paths.
    - Quantile Union, Conformal: the latency of every service at alpha / N (N services in the tree, the union bound over
      the services), added along every path, maximum over the paths.
    - Oracle: the measured node quantiles of data.node_quantiles, added along every path, maximum over the paths.
    Every node bound and every path is computed on its own, also where two have the same service types, so that the
    evaluation time is that of a tree of different services. Returns the two dictionaries of compare_methods.
    A real tree (TreeData with services, real_tree_data in real_data.py) has its held-out configurations per node: every
    model is trained once per service type and predicts at the configurations of every node of that type (model_of),
    and, as in compare_methods with real data, the GPs of Quantile Union and Conformal are fitted on the logarithm of
    the latency quantile."""
    paths, n_services, workload = leaf_to_root_paths(data.children), len(data.path), data.workload
    model = model_of(data)
    services, keys = sorted(set(model.values())), sorted(model)
    log_target = data.y_train_spread is not None
    x_test = model_inputs(data.x_test, data.workers_column)
    x_load, x_test_load = model_inputs(data.x_train_load, data.workers_column), model_inputs(data.x_test_load, data.workers_column)   # the rate stays the last column
    beliefs, gp_seconds = models["beliefs"], models["seconds"]["gp"]
    predictions = lambda: {key: models["gp"][model[key]].predict(configurations_of(x_test, key), return_std=True) for key in keys}
    epsilon_node = alpha / sum(len(path) for path in paths)

    def per_service(latency_of):
        # a baseline that trains its own models per service: latency_of(s, x) gives (latency per configuration of x, training
        # time), where x holds the held-out inputs of every key of the service type s one after the other
        outputs, training_seconds = {}, 0.0
        for s in services:
            of_type = [key for key in keys if model[key] == s]
            inputs = [configurations_of(x_test_load, key) for key in of_type]
            latency, seconds = latency_of(s, {s: np.vstack(inputs)})
            outputs.update(zip(of_type, np.split(latency, np.cumsum([len(x) for x in inputs])[:-1])))
            training_seconds += seconds
        return max_path_sum(paths, lambda k, previous: outputs[data.path[k]]), training_seconds

    def gp_queue():
        predicted = predictions()
        quantiles = {key: queueing_formula_latency(predicted[key][0], predicted[key][1] ** 2, epsilon=alpha, **workload) for key in keys}
        return max_path_sum(paths, lambda k, previous: quantiles[data.path[k]])

    def gp_sum():
        predicted = predictions()
        return max_over_paths(paths, lambda path: joint_gaussian_quantile([predicted[data.path[k]][0] for k in path],
                                                                          [predicted[data.path[k]][1] for k in path], norm.ppf(1 - alpha)))

    candidates = [
        (PROPOSED, "proposed", gp_seconds, lambda: max_path_sum(paths, lambda k, previous: martingale_closed_form_node_bound(
            *beliefs[k], epsilon=epsilon_node, **workload))),
        (PROPOSED_SNC, "proposed", gp_seconds, lambda: max_over_paths(paths, lambda path: network_service_bound(
            [beliefs[k] for k in path], epsilon=alpha / len(paths), holder_exponents=[1] * len(path), **workload))),
        ("GP Sum", "baseline", gp_seconds, gp_sum),
        ("GP Queue", "baseline", gp_seconds, gp_queue),
        ("Quantile Union", "baseline", None, lambda: per_service(lambda s, x: summed_service_quantiles(
            x_load, data.monitored_latencies, [s], alpha / n_services, x, return_seconds=True, log_target=log_target))),
        ("Conformal", "baseline", None, lambda: per_service(lambda s, x: conformal_service_bounds(
            x_load, data.monitored_latencies, [s], alpha / n_services, x, miscoverage=1 - confidence, n_calibration=n_calibration,
            seed=model_seed, return_seconds=True, log_target=log_target))),
    ]
    earlier_seconds = {PROPOSED: models["seconds"]["beliefs"], PROPOSED_SNC: models["seconds"]["beliefs"]}
    methods, seconds = {}, {}
    for name, group, training_seconds, compute in candidates:
        if name in leave_out:
            continue
        started = time.perf_counter()
        latency = compute()
        elapsed = time.perf_counter() - started
        if training_seconds is None:
            latency, training_seconds = latency
            elapsed -= training_seconds
        methods[name] = (group, latency)
        seconds[name] = (training_seconds, elapsed + earlier_seconds.get(name, 0.0))
    methods["Oracle"] = ("reference", max_path_sum(paths, lambda k, previous: data.node_quantiles[:, k]))
    return methods, seconds


def queueing_formula_inputs(data, models: dict, true_mean: np.ndarray, true_var: np.ndarray, alpha: float) -> tuple:
    """Diagnostic of the GP Queue baseline (explain_5_what_allen_cunneen_inputs_cost.ipynb): the same queueing formula (queueing_formula_latency, one quantile per service
    at alpha, added) with different inputs for the mean and the variance of the execution time. "predicted" is the
    prediction of the GP of a service, as the baseline uses it; "true" is true_mean / true_var, (held-out
    configurations, services); "pessimistic" is the mean that Pegasus uses, i.e. the prediction shifted up by the GP's
    uncertainty about the mean (models["beliefs"]).
    Returns a dictionary variant -> end-to-end latency per held-out configuration (s), and the relative error of the
    predicted mean, predicted / true - 1, at the busiest service of every held-out configuration (the service with the
    largest true mean)."""
    x_test = model_inputs(data.x_test, data.workers_column)
    predictions = [models["gp"][s].predict(configurations_of(x_test, s), return_std=True) for s in data.path]
    predicted_mean = np.column_stack([mean for mean, _ in predictions])
    predicted_var = np.column_stack([std ** 2 for _, std in predictions])
    pessimistic_mean = np.column_stack([mean for mean, _ in models["beliefs"]])
    workers = path_workers(data) or [1] * len(data.path)
    # plus the time between the services, as every method of compare_methods
    formula = lambda mean, var: sum(queueing_formula_latency(mean[:, k], var[:, k], epsilon=alpha, workers=workers[k], **data.workload)
                                    for k in range(len(data.path))) + data.handover
    variants = {"predicted mean, predicted variance (the baseline)": formula(predicted_mean, predicted_var),
                "predicted mean, true variance": formula(predicted_mean, true_var),
                "true mean, predicted variance": formula(true_mean, predicted_var),
                "true mean, true variance": formula(true_mean, true_var),
                "pessimistic mean, predicted variance": formula(pessimistic_mean, predicted_var)}
    busiest = (np.arange(len(true_mean)), true_mean.argmax(axis=1))
    return variants, predicted_mean[busiest] / true_mean[busiest] - 1


# ------------------------------------------------------------------------------
# Validity: coverage per seed, judged before tightness
# ------------------------------------------------------------------------------
def coverage(latency: np.ndarray, measured_low: np.ndarray) -> float:
    """Share of the held-out configurations at which a method's latency is not below the lower edge of the confidence
    interval of the measured quantile, i.e. at which the simulation does not show that the method is too low."""
    return float(np.mean(latency >= measured_low))


def holds(coverages: np.ndarray, confidence: float) -> np.ndarray:
    """Whether a method holds in every setting: it reaches the confidence level (coverage >= confidence) under every
    seed, so that one failing seed is enough to fail. coverages: (settings, seeds). A setting in which the method was
    left out (nan) does not count as holding."""
    return np.all(np.asarray(coverages) >= confidence, axis=-1)


def sweep_statistics(results: dict, truth: dict, settings: list, seeds: list, confidence: float) -> dict:
    """Coverage and tightness of every method of a sweep. results[(setting, seed)]: name -> (group, latency per
    held-out configuration in s); truth[(setting, seed)]: (measured quantile, lower edge of its confidence interval),
    per held-out configuration.
    Returns name -> dict with "coverage" and "tightness", both (settings, seeds), "holds" (settings, see holds) and
    "computed" (settings: the method was not left out there). Coverage and tightness are nan where the method was left
    out. The tightness of a seed is the median over its held-out configurations of latency / measured quantile. It is
    also nan where that median is not finite, i.e. where the method has no finite latency at half of the
    configurations or more."""
    def per_setting_and_seed(name, statistic):
        return np.array([[statistic(results[(value, s)][name][1], *truth[(value, s)]) if name in results[(value, s)] else np.nan for s in seeds]
                         for value in settings], dtype=float)

    statistics = {}
    for name in METHOD_COLORS:
        if not any(name in result for result in results.values()):
            continue
        covered = per_setting_and_seed(name, lambda latency, measured, low: coverage(latency, low))
        tightness = per_setting_and_seed(name, lambda latency, measured, low: np.median(latency / measured))
        statistics[name] = dict(coverage=covered, tightness=np.where(np.isfinite(tightness), tightness, np.nan), holds=holds(covered, confidence),
                                computed=~np.isnan(covered).all(axis=1))
    return statistics


def sweep_seconds(seconds: dict, settings: list, seeds: list) -> dict:
    """Training and evaluation time of every method of a sweep. seconds[(setting, seed)]: name -> (training time,
    evaluation time) in s, from compare_methods. Returns name -> dict with "training" and "evaluation", both
    (settings, seeds), nan where the method was left out."""
    names = [name for name in METHOD_COLORS if any(name in result for result in seconds.values())]
    part = lambda name, k: np.array([[seconds[(value, s)].get(name, (np.nan, np.nan))[k] for s in seeds] for value in settings], dtype=float)
    return {name: dict(training=part(name, 0), evaluation=part(name, 1)) for name in names}


def same_models(times: dict, names: list) -> list:
    """Groups the methods `names` that have the same training time in every setting and seed of times (from
    sweep_seconds), i.e. that use the same trained models: a list of lists of names."""
    groups = []
    for name in names:
        for group in groups:
            if np.array_equal(times[group[0]]["training"], times[name]["training"], equal_nan=True):
                group.append(name)
                break
        else:
            groups.append([name])
    return groups


def finite_share(results: dict, name: str, setting, seeds: list) -> float:
    """Share of the held-out configurations at which a method gives a finite latency in one setting of a sweep, median
    over the seeds. results: as in sweep_statistics."""
    return float(np.median([np.mean(np.isfinite(results[(setting, s)][name][1])) for s in seeds]))


def sweep_conclusion(settings: list, statistics: dict, names: list, label) -> str:
    """Two Markdown bullets for the conclusion of a sweep, validity first: which of the methods `names` hold in every
    setting and where the others do not, then the tightness of the methods that hold (median over the seeds) in the
    first and the last setting. Only the settings in which a method is computed count for it, and a method that is
    left out somewhere is named with these settings. statistics: from sweep_statistics; label(setting): how one
    setting is written. Methods of `names` that were not computed at all are skipped."""
    names = [name for name in names if name in statistics]
    computed = {name: [value for value, there in zip(settings, statistics[name]["computed"]) if there] for name in names}
    held = {name: statistics[name]["holds"][statistics[name]["computed"]] for name in names}
    holding = [name for name in names if held[name].all()]
    failing = {name: [label(value) for value, ok in zip(computed[name], held[name]) if not ok] for name in names if name not in holding}
    left_out = {name: [label(value) for value in settings if value not in computed[name]] for name in names if len(computed[name]) < len(settings)}

    def tightness_range(name):
        # median over the seeds; a seed without a finite tightness counts as infinitely loose
        tightness = statistics[name]["tightness"][statistics[name]["computed"]]
        typical = np.median(np.where(np.isnan(tightness), np.inf, tightness), axis=1)
        finite = np.flatnonzero(np.isfinite(typical))
        if len(finite) == 0:
            return f"{name} gives no finite bound for most seeds in any setting"
        text = f"{name} from {typical[finite[0]]:.2f}× at {label(computed[name][finite[0]])} to {typical[finite[-1]]:.2f}× at {label(computed[name][finite[-1]])}"
        if len(finite) == 1:   # a finite bound in one setting only: no range to give
            text = f"{name} {typical[finite[0]]:.2f}× at {label(computed[name][finite[0]])}"
        without = [label(value) for value, value_tightness in zip(computed[name], typical) if not np.isfinite(value_tightness)]
        return text + (f", and no finite bound for most seeds at {', '.join(without)}" if without else "")

    return (f"- **Validity.** Methods that hold in every setting in which they are computed, i.e. reach the confidence level under every seed: "
            f"{', '.join(holding) or 'none'}. Methods that do not hold, and where: "
            + ("; ".join(f"{name} at {'every setting' if len(where) == len(computed[name]) else ', '.join(where)}" for name, where in failing.items()) or "none")
            + "."
            + "".join(f" {name} is not computed at {', '.join(where)}." for name, where in left_out.items())
            + "\n"
            f"- **Tightness among the methods that hold** (median over the seeds): "
            + ("; ".join(tightness_range(name) for name in holding) or "no method holds")
            + ".")


def time_conclusion(settings: list, times: dict, names: list, label) -> str:
    """Two Markdown bullets for the conclusion of a sweep: the training time and the evaluation time of the methods
    `names` (median over the seeds, s) in the first and the last setting in which a method is computed. Methods that use
    the same trained models (same_models) share one entry of the training time. times: from sweep_seconds;
    label(setting): how one setting is written. Methods of `names` that were not computed at all are skipped."""
    names = [name for name in names if name in times]

    def ends(values):
        there = np.flatnonzero(~np.isnan(values).all(axis=1))   # the settings in which the method is computed
        first, last = np.nanmedian(values[there[0]]), np.nanmedian(values[there[-1]])
        return f"{first:.2g} s at {label(settings[there[0]])} and {last:.2g} s at {label(settings[there[-1]])}"

    return ("- **Training time** (median over the seeds; methods that share their models are named together): "
            + "; ".join(f"{', '.join(group)} {ends(times[group[0]]['training'])}" for group in same_models(times, names))
            + ".\n- **Evaluation time** (median over the seeds): "
            + "; ".join(f"{name} {ends(times[name]['evaluation'])}" for name in names) + ".")


def ground_truth_conclusion(settings: list, truth: dict, statistics: dict, seeds: list, label, quantile_name: str, condition: str = "") -> str:
    """One Markdown bullet for the conclusion of a sweep: the measured quantile of the end-to-end latency (s, median over
    the held-out configurations and then over the seeds) and the tightness of the Oracle in the first and the last
    setting. truth, statistics: as in sweep_statistics; quantile_name: e.g. "p95"; condition: what stays the same
    between the settings, appended to the first sentence (e.g. ", at the same arrival rate")."""
    measured = lambda value: np.median([np.median(truth[(value, s)][0]) for s in seeds])
    oracle = lambda k: np.nanmedian(statistics["Oracle"]["tightness"][k])
    first, last = settings[0], settings[-1]
    return (f"- **Ground truth.** The measured {quantile_name} of the end-to-end latency, the median over the held-out configurations, goes from "
            f"{measured(first):.2f} s at {label(first)} to {measured(last):.2f} s at {label(last)}{condition}. The Oracle, the union bound over "
            f"measured services, is at {oracle(0):.2f}× and {oracle(-1):.2f}×.")


# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
TIGHTNESS_RANGE = (0.2, 64)   # range of the tightness axis of every figure; a value outside lies beyond the edge of the axis
TIGHTNESS_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32]


def arrival_text(workload: dict) -> str:
    """The arrival process of a chain in words, for titles and conclusions: with one arrival rate for all held-out
    configurations, e.g. "poisson arrivals at 4/s"; with one rate per configuration, the range of the rates."""
    rate = np.asarray(workload["rate"], dtype=float)
    if rate.ndim == 0:
        return f"{workload['process']} arrivals at {rate:g}/s"
    return f"{workload['process']} arrivals at one rate per configuration ({rate.min():.3g} to {rate.max():.3g}/s)"


def tightness_figure(results: dict, data, title: str, level_label: str, seeds_reached: dict, n_seeds: int) -> None:
    """One row per method of `results` (name -> (group, end-to-end latency per held-out configuration in s)): its
    tightness, latency / measured quantile, over the configurations with a finite latency. level_label: name of the
    measured quantile (e.g. "p95"). seeds_reached: name -> number of the n_seeds seeds under which the method reaches
    the confidence level; the number is written right of the row, and the box of a method that does not reach the level
    under every seed is hatched instead of filled. The axis shows TIGHTNESS_RANGE; what lies outside is cut off."""
    from .figures import tightness_label, tightness_tick
    names = list(results)[::-1]   # first method at the top
    rows = np.arange(1, len(names) + 1)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(names) + 1.6))
    ratios = [results[name][1] / data.measured for name in names]
    boxes = ax.boxplot([ratio[np.isfinite(ratio)] for ratio in ratios], positions=rows, vert=False, patch_artist=True, widths=0.6,
                       medianprops=dict(color="black"), flierprops=dict(markersize=3))
    for box, name, row in zip(boxes["boxes"], names, rows):
        color = GROUP_COLORS[results[name][0]]
        if seeds_reached[name] == n_seeds:
            box.set(facecolor=color, alpha=0.7)
        else:
            box.set(facecolor="white", edgecolor=color, hatch="////")
        ax.text(1.01, row, f"{seeds_reached[name]}/{n_seeds}", transform=ax.get_yaxis_transform(), va="center", ha="left")
    ax.text(1.01, 1.0, "seeds", transform=ax.transAxes, va="bottom", ha="left")
    ax.axvline(1.0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(*TIGHTNESS_RANGE)
    ax.set_xticks(TIGHTNESS_TICKS, [tightness_tick(tick) for tick in TIGHTNESS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel(tightness_label(level_label))
    ax.set_yticks(rows, names)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    groups = [group for group in GROUP_COLORS if any(results[name][0] == group for name in names)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[group], alpha=0.7) for group in groups]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="dimgray", hatch="////"))
    fig.legend(handles, groups + ["does not hold: below the confidence level under at least one seed"],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
    fig.suptitle(f"{title}: {' → '.join(data.path)}, {len(data.measured)} held-out configurations,\n{arrival_text(data.workload)}")
    plt.tight_layout()
    plt.show()


def sweep_line(ax, settings, values: np.ndarray, name: str, held: np.ndarray = None, label: str = None) -> None:
    """One method in a figure over a sweep, in the style of the paper figures (pegasus/figures.py): its color, marker
    and line style. settings: the values of the horizontal axis; values: (settings, seeds), nan where the method has
    no value. The line is the median over the seeds, in which a seed without a value counts as infinitely large, so that
    a setting has no point if half of the seeds or more have no value. The band reaches from the smallest to the
    largest seed with a value. held: per setting, whether the method holds there (see holds); its marker is then
    filled, otherwise hollow. Without `held` every marker is filled. label: legend entry (default: name)."""
    from .figures import BAND_ALPHA, METHOD_STYLES, method_linewidth
    typical = np.median(np.where(np.isnan(values), np.inf, values), axis=1)
    shown = np.isfinite(typical)
    settings, values, typical = np.asarray(settings)[shown], values[shown], typical[shown]
    color = METHOD_COLORS[name]
    marker, linestyle = METHOD_STYLES.get(name, ("o", "-"))
    filled = np.ones(len(settings), dtype=bool) if held is None else np.asarray(held)[shown]
    ax.plot(settings, typical, linestyle=linestyle, color=color, linewidth=method_linewidth(name))
    ax.fill_between(settings, np.nanmin(values, axis=1), np.nanmax(values, axis=1), color=color, alpha=BAND_ALPHA, linewidth=0)
    ax.scatter(settings, typical, s=40, marker=marker, facecolors=[color if f else "white" for f in filled], edgecolors=color, linewidths=1.5,
               zorder=3)
    # the legend entry carries the line and the marker, as in the legends of figures.py
    ax.plot([], [], linestyle=linestyle, marker=marker, color=color, linewidth=method_linewidth(name), markersize=7, label=label or name)


def sweep_segments(settings, values: np.ndarray) -> list:
    """(x, y_low, y_high) of every point and band of a sweep_line, for legend_inside in figures.py."""
    shown = ~np.isnan(values).all(axis=1)
    return list(zip(np.asarray(settings, dtype=float)[shown], np.nanmin(values[shown], axis=1), np.nanmax(values[shown], axis=1)))


def tightness_sweep_figure(settings: list, statistics: dict, xlabel: str, title: str, level_label: str, tick_labels: list = None,
                           log_x: bool = True, name: str = None, half: str = None) -> None:
    """Tightness of every method over a sweep, one panel: per method the median over the seeds with the band from the
    smallest to the largest seed (sweep_line), a filled marker where the method holds and a hollow one where it does
    not. statistics: from sweep_statistics; tick_labels: text of every setting on the horizontal axis (default: its
    value); level_label: name of the measured quantile (e.g. "p95"). A method has no point where, under half of the
    seeds or more, it has no finite latency at half of the held-out configurations or more. The title is printed above
    the figure instead of drawn into it (no title in a paper figure); name: file name in figures/tsc, or None.
    half: "left" or "right" for one of two figures side by side in the width of one (half_figure), with the same
    methods. The left one has the labels of the vertical axis and the first half of the legend; the right one shares
    the vertical axis and has the rest of the legend and the text of the red zone."""
    from .figures import fit_x_tick_labels, half_figure, paper_figure, paper_style, save_figure
    print(title)
    with paper_style():
        fig, ax = half_figure(half) if half else paper_figure()
        _sweep_axis(ax, settings, xlabel, tick_labels, log_x)
        _tightness_panel(ax, settings, statistics, level_label, failure_text=half != "left", wrap_text=bool(half))
        if half == "right":
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if not log_x:
            fit_x_tick_labels(ax)
        _tightness_legend(ax, settings, statistics, ncol=1 if half else 3, half=half)
        if name is not None:
            save_figure(fig, name)
        plt.show()


def tightness_and_time_figure(settings: list, statistics: dict, times: dict, xlabel: str, title: str, level_label: str,
                              tick_labels: list = None, log_x: bool = True, name: str = None, time_halves: tuple = None,
                              overlay: tuple = None, onset: tuple = None) -> None:
    """Three separate figures over a sweep. First: the tightness of every method, as in tightness_sweep_figure.
    Second: the training time (s), one line per set of trained models; methods that use the same models (same_models)
    share a line, and its legend names them. Third: the evaluation time (s) of every method. Lines and bands of the time
    figures are the median and the range over the seeds (sweep_line). statistics: from sweep_statistics; times: from
    sweep_seconds. The title is printed above the figures; name: file name in figures/tsc, saved as <name>_tightness,
    <name>_training and <name>_evaluation, or None. time_halves: (xlabel, tick_labels) of the two time figures as
    half figures side by side in the width of one (half_figure, each with its own vertical axis), which need a shorter
    horizontal axis; None: full figures with the axis of the tightness figure.
    overlay: (settings, statistics, label) of a second ground truth for some methods of the tightness figure, drawn as
    dashed lines in their colors behind the measured ones (overlay_line), with one legend entry `label`. onset: (x, label)
    of a vertical line in the tightness figure, e.g. where the services start to slow down."""
    from .figures import fit_half_pair, fit_x_tick_labels, half_figure, legend_inside, paper_figure, paper_style, plain_log_ticks, save_figure
    print(title)
    short = lambda method: method.replace(" (ours)", "")
    with paper_style():
        fig, ax = paper_figure()
        _sweep_axis(ax, settings, xlabel, tick_labels, log_x)
        _tightness_panel(ax, settings, statistics, level_label)
        extra_segments = []
        if overlay is not None:
            overlay_settings, overlay_statistics, overlay_label = overlay
            from .figures import METHOD_STYLES
            for method in METHOD_COLORS:
                if method in overlay_statistics:
                    overlay_line(ax, overlay_settings, overlay_statistics[method]["tightness"], method, overlay_statistics[method]["holds"])
                    extra_segments += sweep_segments(overlay_settings, overlay_statistics[method]["tightness"])
                    if method not in statistics:   # only the dashed line: its own legend entry, in its color and marker
                        ax.plot([], [], color=METHOD_COLORS[method], linestyle=(0, (4, 2)), linewidth=1.5, marker=METHOD_STYLES.get(method, ("o",))[0],
                                markersize=6, label=method)
            ax.plot([], [], color="0.3", linestyle=(0, (4, 2)), linewidth=1.5, label=overlay_label)
        if onset is not None:
            x, onset_label = onset
            ax.axvline(x, color="0.25", linewidth=1.2, linestyle=(0, (1, 1.5)), zorder=1)
            ax.plot([], [], color="0.25", linewidth=1.2, linestyle=(0, (1, 1.5)), label=onset_label)
        if not log_x:
            fit_x_tick_labels(ax)
        _tightness_legend(ax, settings, statistics, extra_segments=extra_segments)
        if name is not None:
            save_figure(fig, f"{name}_tightness")
        plt.show()
        panels = []   # (what, figure, axes, segments) of the two time figures, completed together so that a pair can be sized together
        for what, side in (("training", "left"), ("evaluation", "right")):
            fig, ax = half_figure(side, shared_y=False) if time_halves else paper_figure()
            _sweep_axis(ax, settings, *(time_halves or (xlabel, tick_labels)), log_x)
            segments = []
            # training: one line per set of shared models, whose names a half figure writes one per line; evaluation: one line per method
            group_label = lambda group: (",\n" if time_halves else ", ").join(short(method) for method in group)
            lines = [(group[0], group_label(group)) for group in same_models(times, list(times))] if what == "training" \
                else [(method, method) for method in times]
            for method, label in lines:
                sweep_line(ax, settings, times[method][what], method, label=label)
                segments += sweep_segments(settings, times[method][what])
            ax.set_yscale("log")
            plain_log_ticks(ax.yaxis)
            ax.set_ylabel(f"{what} time (s, log scale)")
            panels.append((what, fig, ax, segments))
        if time_halves:
            # each half as wide as its own labels need, the rest of the width to the two plot areas
            fit_half_pair([ax for _, _, ax, _ in panels])
        for what, fig, ax, segments in panels:
            if not log_x:
                fit_x_tick_labels(ax)
            handles, labels = ax.get_legend_handles_labels()
            # the times span the whole width at the top, so the log axis may grow upward to make room for the legend; the legend
            # of the evaluation time in a half figure is too tall for that (one column, one row per method), so there the axis
            # keeps the range of the data and the legend may cover the upper end of the lines of Pegasus
            keep_range = time_halves and what == "evaluation"
            legend_inside(ax, handles, labels, segments, max_doublings=0 if keep_range else 3, title="models trained for" if what == "training" else None,
                          ncol=1 if time_halves else 2, fontsize=10, title_fontsize=10, framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
            if name is not None:
                save_figure(fig, f"{name}_{what}")
        plt.show()


def _sweep_axis(ax, settings: list, xlabel: str, tick_labels: list, log_x: bool) -> None:
    """Horizontal axis of a panel over a sweep: one tick per setting."""
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(settings, tick_labels or [f"{value:g}" for value in settings])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle=":", alpha=0.6)


def _tightness_panel(ax, settings: list, statistics: dict, level_label: str, failure_text: bool = True, wrap_text: bool = False) -> None:
    """Tightness of every method over a sweep on `ax`, with the line at 1x (the measured quantile) and the red zone
    below it, where a method underestimates the quantile (SLO violated). The axis shows TIGHTNESS_RANGE; a line leaves the panel where a method lies
    outside."""
    from .figures import FAILURE_COLOR, FAILURE_TEXT_COLOR, tightness_label, tightness_tick
    ax.axhspan(TIGHTNESS_RANGE[0], 1.0, color=FAILURE_COLOR, linewidth=0)
    ax.axhline(1.0, color="black", linewidth=1)
    for name in METHOD_COLORS:
        if name in statistics:
            sweep_line(ax, settings, statistics[name]["tightness"], name, held=statistics[name]["holds"])
    if failure_text:
        # wrap_text: in two lines, for a panel of half the width
        ax.text(0.99, 0.03, f"underestimated:{chr(10) if wrap_text else ' '}SLO violated", transform=ax.transAxes, ha="right",
                va="bottom", color=FAILURE_TEXT_COLOR)
    ax.set_yscale("log")
    ax.set_ylim(*TIGHTNESS_RANGE)
    ax.set_yticks(TIGHTNESS_TICKS, [tightness_tick(tick) for tick in TIGHTNESS_TICKS])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel(tightness_label(level_label))


def overlay_line(ax, settings, values: np.ndarray, name: str, held: np.ndarray) -> None:
    """A second ground truth for the method `name` in a tightness panel: the median over the seeds as a dashed line in the
    color of the method, behind its measured line (sweep_line) and without a band, markers filled where it holds and
    hollow where it does not. values: (settings, seeds)."""
    from .figures import METHOD_STYLES
    typical = np.median(np.where(np.isnan(values), np.inf, values), axis=1)
    shown = np.isfinite(typical)
    settings, typical, held = np.asarray(settings)[shown], typical[shown], np.asarray(held)[shown]
    color, marker = METHOD_COLORS[name], METHOD_STYLES.get(name, ("o", "-"))[0]
    ax.plot(settings, typical, color=color, linestyle=(0, (4, 2)), linewidth=1.5, zorder=1.5)
    ax.scatter(settings, typical, s=28, marker=marker, facecolors=[color if h else "white" for h in held], edgecolors=color, linewidths=1.2,
               zorder=1.6)


def _tightness_legend(ax, settings: list, statistics: dict, ncol: int = 3, half: str = None, extra_segments: list = ()) -> None:
    """The legend of the methods and of the markers inside a tightness panel, where it hides the fewest points and
    bands (legend_inside); the axis keeps TIGHTNESS_RANGE, as in B.1. Call when the axes are complete. half: "left" lists the
    first half of the methods, "right" the rest and the markers, so that the two halves of a pair share one legend.
    extra_segments: points and bands drawn besides the methods of `statistics` (e.g. overlay_line), which the legend avoids too."""
    from .figures import holds_handles, legend_inside
    handles, labels = ax.get_legend_handles_labels()
    # the versions of the proposed method first, in their order: a version drawn only as an overlay (tightness_and_time_figure)
    # then sits in the first column next to the measured one, whose label is as wide
    first_proposed = sorted(range(len(labels)), key=lambda k: (labels[k] not in (PROPOSED, PROPOSED_SNC), k))
    handles, labels = [handles[k] for k in first_proposed], [labels[k] for k in first_proposed]
    marker_handles, marker_labels = holds_handles()
    first = (len(handles) + 1) // 2
    if half == "left":
        handles, labels, marker_handles, marker_labels = handles[:first], labels[:first], [], []
    elif half == "right":
        handles, labels = handles[first:], labels[first:]
    segments = [segment for name in statistics if name in METHOD_COLORS for segment in sweep_segments(settings, statistics[name]["tightness"])]
    segments += list(extra_segments)
    legend_inside(ax, handles + marker_handles, labels + marker_labels, segments, max_doublings=0, ncol=ncol, fontsize=10, framealpha=0.95,
                  columnspacing=1.0, handletextpad=0.4)


def uncertainty_figure(uncertainty: dict, chain_pegasus: dict, services: list, title: str, confidence: float, name: str = None,
                       oracle: dict = None, level_label: str = None, execution_times: dict = None) -> None:
    """What more training runs per service do to the service models and to the bound of Pegasus Node: two figures side
    by side in the width of one (half_figure, each with its own vertical axis), with the training runs per service on
    the horizontal axis (logarithmic). All services grow together; a service with fewer runs holds all of them.
    Left (accuracy of every service model, %, logarithmic): per service, in its color and marker, the error of the
    predicted mean execution time m against the measured mean of the held-out run, |m - measured| / measured, the
    median over the held-out configurations and then over the seeds. The margin of the pessimistic mean m-bar is not
    drawn: four margin lines next to four error lines made the figure unreadable; its effect on the chain is the gap
    between Pegasus Node and the line with known means on the right, and its values are in the tables of the notebook. Right (Pegasus Node, as every tightness figure of B): its
    tightness, the end-to-end bound / the measured quantile, the median over the held-out configurations per seed, as a
    line over the seeds with the band from the smallest to the largest seed, a filled marker where it holds (coverage of
    at least `confidence` under every seed) and a hollow one where it does not; the same bound with the predicted means
    m in place of m-bar (dotted, in the color of Pegasus Node, without markers), whose distance to Pegasus Node is the
    price of the uncertainty of the models; and the Oracle, which uses no training data. Where the median over the
    seeds is infinite (the pessimistic mean overloads a service at half of the configurations or more), "∞" stands at
    the top edge. uncertainty: seed -> service -> array (levels, 6) of (training runs, median margin, uncertainty cost,
    median predicted mean, median node bound with it, median error of the predicted mean); chain_pegasus: seed ->
    array (levels, 7) of (training runs per service, median bound of Pegasus Node (s), median bound with the predicted
    means (s), median tightness of both, coverage of both); oracle: seed -> (median tightness, coverage) of the Oracle,
    or None. execution_times: service -> its measured mean execution time (s), written next to its name in the legend, so
    that the figure keeps how slow every service is; or None. The title is printed above the figures; name: file name in
    figures/tsc, saved as <name>_execution (left)
    and <name>_cost (right), the names they had before, so that references to the files stay valid, or None."""
    from .figures import fit_half_pair, half_figure, legend_inside, paper_style, plain_log_ticks, save_figure
    print(title)
    seeds = list(uncertainty)
    median = lambda service, column: np.median([uncertainty[s][service][:, column] for s in seeds], axis=0)
    runs = {service: uncertainty[seeds[0]][service][:, 0] for service in services}
    # services are no methods: their own colors and markers, so that every line also reads in grayscale
    colors = ("tab:blue", "tab:orange", "tab:brown", "tab:pink", "tab:cyan", "tab:olive")
    markers = ("s", "^", "D", "v", "P", "X")
    with paper_style():
        fig, left = half_figure("left", shared_y=False)
        segments = []
        for k, service in enumerate(services):
            color, marker = colors[k % len(colors)], markers[k % len(markers)]
            error = 100 * median(service, 5)
            time = "" if execution_times is None else (f" ({execution_times[service]:.2g} s)" if execution_times[service] >= 1
                                                       else f" ({execution_times[service] * 1e3:.2g} ms)")
            left.plot(runs[service], error, linestyle="-", marker=marker, color=color, linewidth=1.5, markersize=6, label=service + time)
            segments += [(x, e, e) for x, e in zip(runs[service], error)]
        values = [value for segment in segments for value in segment[1:] if value > 0]
        # little room above the data: the errors are largest with few runs, at the left, and the legend fits upper right
        left.set_ylim(min(values) / 1.5, max(values) * 1.5)
        handles, labels = left.get_legend_handles_labels()
        left.set_ylabel("error of predicted mean (%)")
        figures = [(fig, "execution")]

        fig, ax = half_figure("right", shared_y=False)
        chain_runs = chain_pegasus[seeds[0]][:, 0]
        per_seed = lambda column: np.column_stack([chain_pegasus[s][:, column] for s in seeds])   # (levels, seeds)
        finite = lambda values: np.where(np.isfinite(values), values, np.nan)   # sweep_line: no value where the median is infinite
        holds_of = lambda coverages: np.all(coverages >= confidence, axis=1)
        statistics = {PROPOSED: dict(tightness=finite(per_seed(3)), holds=holds_of(per_seed(5)))}
        if oracle is not None:   # the same at every number of runs: it uses no training data
            statistics["Oracle"] = dict(tightness=np.tile([oracle[s][0] for s in seeds], (len(chain_runs), 1)),
                                        holds=np.full(len(chain_runs), all(oracle[s][1] >= confidence for s in seeds)))
        # the last level: every service holds all its runs; a level close to it gets no label, which would run into "all"
        ticks = [f"{value:g}" if value * 1.5 < chain_runs[-1] else "" for value in chain_runs[:-1]] + ["all"]
        _sweep_axis(ax, chain_runs, "runs per service", ticks, log_x=True)
        _tightness_panel(ax, chain_runs, statistics, level_label or "p95", wrap_text=True)
        known = np.median(np.where(np.isnan(finite(per_seed(4))), np.inf, per_seed(4)), axis=1)
        ax.plot(chain_runs[np.isfinite(known)], known[np.isfinite(known)], linestyle=":", color=METHOD_COLORS[PROPOSED], linewidth=1.5,
                label="known means")
        typical = np.median(np.where(np.isnan(statistics[PROPOSED]["tightness"]), np.inf, statistics[PROPOSED]["tightness"]), axis=1)
        for x in chain_runs[~np.isfinite(typical)]:   # no finite bound under half of the seeds or more
            ax.annotate("∞", (x, TIGHTNESS_RANGE[1]), xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                        color=METHOD_COLORS[PROPOSED], fontsize=14, fontweight="bold")
        _tightness_legend(ax, chain_runs, statistics, ncol=1)
        figures.append((fig, "cost"))

        left.set_xscale("log")
        left.set_yscale("log")
        plain_log_ticks(left.xaxis)
        plain_log_ticks(left.yaxis)
        left.set_xlabel("runs of the service")
        left.grid(True, linestyle=":", alpha=0.6)
        # one column: the names with their execution times are too wide for two columns in a half figure
        legend_inside(left, handles, labels, segments, locations=("upper right", "upper center"),
                      max_doublings=0, ncol=1, fontsize=9, framealpha=0.95, handletextpad=0.4, labelspacing=0.3)
        fit_half_pair([left, ax])
        for fig, suffix in figures:
            if name is not None:
                save_figure(fig, f"{name}_{suffix}")
            plt.show()


# ------------------------------------------------------------------------------
# Slowdown of the services in a tree (B.3 diagnosis; here and not in real_data.py, so that it does not change the cache
# key of the results: no method uses it)
# ------------------------------------------------------------------------------
SLOWDOWN_SERVICES = {   # service type -> (label, color, marker, line style); the services of the trees of B.3, distinct in grayscale
    "cv-analyzer:v2": ("object detector (leaf)", "#4477AA", "o", "-"),
    "face-detector:v2": ("face detector (leaf)", "#CC6677", "s", "--"),
    "image-transformer:v1": ("image transformer (aggregator)", "#DDAA33", "^", "-."),
}
CONCURRENCY_BINS = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48)   # bins of the concurrency c for the medians of slowdown_figure


def tree_slowdowns(experiments: dict):
    """Slowdown of every service of the trees `experiments` ((tree set, levels h) -> experiment of VM_V2) in every run,
    against its grid benchmark on the same machine (VM_TREE_BENCHMARKS), and the concurrency it saw. One row per run and
    node: tree set, h, run_id, node, service, slowdown, the mean processing time of the node in the run divided by that of
    its benchmark at the same configuration (median over the benchmarks at u = 0.4, 0.6 and 0.8, whose processing times
    do not depend on the load on this machine); concurrency, the time-weighted mean number of the other services of the
    tree that are processing while the node processes an item (trace_steps.csv). Runs: valid, and throttled for less than
    TREE_THROTTLE_LIMIT at every service; the camera does not count, because its throttling delays the frames before the
    services and not their processing (so the small trees with h = 5 and 6 count, whose camera fell behind)."""
    import pandas as pd
    from .real_data import TREE_SOURCE, TREE_THROTTLE_LIMIT, VM_TREE_BENCHMARKS, campaign_experiments, experiment_folder, run_table
    benchmark = []
    for service, view in VM_TREE_BENCHMARKS.items():
        for experiment in campaign_experiments(view):
            if experiment.endswith(("u040", "u060", "u080")):
                table = run_table(experiment)
                table = table[table["service"] == service]
                benchmark.append(pd.DataFrame({"service": service, "quality": table["service.data_quality"],
                                               "model": table["service.model_size"] if "service.model_size" in table else 0,
                                               "benchmark_ms": table["processing_mean_ms"]}))
    benchmark = pd.concat(benchmark).fillna({"model": 0}).groupby(["service", "quality", "model"])["benchmark_ms"].median()
    rows = []
    for (tree_set, h), experiment in experiments.items():
        table = run_table(experiment)
        nodes = table[table["node"] != TREE_SOURCE].copy()
        throttled = nodes.groupby("run_id")["throttled_fraction"].max()
        nodes = nodes[nodes["run_id"].isin(throttled.index[throttled < TREE_THROTTLE_LIMIT])]
        # every service of a run has the shared configuration of the leaves; the image transformer has no model size
        model = np.where(nodes["service"] == "image-transformer:v1", 0, nodes["leaf-1.model_size"])
        keys = zip(nodes["service"], nodes["leaf-1.data_quality"], model)
        nodes["slowdown"] = nodes["processing_mean_ms"].to_numpy() / np.array([benchmark.get(key, np.nan) for key in keys])
        for run_id, run_nodes in nodes.groupby("run_id"):
            steps = pd.read_csv(experiment_folder(experiment) / "runs" / run_id / "trace_steps.csv",
                                usecols=["node", "outcome", "processing_ms", "finished_at_ns"])
            steps = steps[(steps["outcome"] == "completed") & (steps["node"] != TREE_SOURCE)]
            end = steps["finished_at_ns"].to_numpy(dtype=float)
            start = end - steps["processing_ms"].to_numpy() * 1e6
            # number of processing services over time, B(t), as a step function; its integral over the processing of an
            # item, divided by the processing time, minus the item itself, is the mean number of others processing meanwhile
            times = np.concatenate([start, end])
            order = np.argsort(times, kind="stable")
            times, steps_up = times[order], np.concatenate([np.ones_like(start), -np.ones_like(end)])[order]
            area = np.concatenate([[0.0], np.cumsum(np.cumsum(steps_up)[:-1] * np.diff(times))])
            others = (np.interp(end, times, area) - np.interp(start, times, area)) / np.maximum(end - start, 1.0) - 1.0
            concurrency = pd.Series(others).groupby(steps["node"].to_numpy()).mean()
            rows += [dict(tree_set=tree_set, h=h, run_id=run_id, node=row.node, service=row.service, slowdown=row.slowdown,
                          concurrency=concurrency.get(row.node, np.nan)) for row in run_nodes.itertuples()]
    return pd.DataFrame(rows).dropna(subset=["slowdown", "concurrency"])


def slowdown_bins(slowdowns, quantiles: tuple = (0.1, 0.5, 0.9), min_size: int = 10):
    """Per service type and bin of CONCURRENCY_BINS with at least min_size rows of tree_slowdowns: the median concurrency
    of the bin and the `quantiles` of the slowdown (columns q10, q50, q90), and the number of rows (size)."""
    import pandas as pd
    binned = slowdowns.assign(bin=pd.cut(slowdowns["concurrency"], CONCURRENCY_BINS, include_lowest=True))
    grouped = binned.groupby(["service", "bin"], observed=True)
    table = grouped["slowdown"].quantile(list(quantiles)).unstack().rename(columns=lambda q: f"q{round(100 * q)}")
    table["concurrency"], table["size"] = grouped["concurrency"].median(), grouped.size()
    return table[table["size"] >= min_size].reset_index()


def slowdown_figure(slowdowns, height_marks: dict = None, name: str = None) -> None:
    """Slowdown of the services in the trees against the concurrency they saw (tree_slowdowns): per service type the
    median slowdown in every bin of the concurrency (slowdown_bins), at the median concurrency of the bin, with the band
    from the 10th to the 90th percentile; the line at 1 is the speed of the benchmark. height_marks: label -> concurrency,
    drawn as thin vertical lines (e.g. the median concurrency of every height of a tree set). name: file in figures/tsc."""
    from matplotlib.lines import Line2D
    from .figures import BAND_ALPHA, legend_inside, paper_figure, paper_style, save_figure
    table = slowdown_bins(slowdowns)
    with paper_style():
        fig, ax = paper_figure()
        handles, labels, segments = [], [], []
        for service, (label, color, marker, style) in SLOWDOWN_SERVICES.items():
            rows = table[table["service"] == service].sort_values("concurrency")
            if rows.empty:
                continue
            ax.fill_between(rows["concurrency"], rows["q10"], rows["q90"], color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(rows["concurrency"], rows["q50"], color=color, marker=marker, linestyle=style, linewidth=2)
            handles.append(Line2D([], [], color=color, marker=marker, linestyle=style, linewidth=2))
            labels.append(label)
            segments += list(zip(rows["concurrency"], rows["q10"], rows["q90"]))
        ax.axhline(1, color="black", linewidth=0.8)
        handles.append(Line2D([], [], color="black", linewidth=0.8))
        labels.append("speed in the benchmark")
        top = max(2.0, float(table["q90"].max()) * 1.05)
        ax.set_ylim(0.9, top)
        for mark, concurrency in (height_marks or {}).items():
            ax.axvline(concurrency, color="0.5", linewidth=0.8, linestyle=":")
            ax.text(concurrency, 0.92, f" {mark}", color="0.35", fontsize=10, ha="left", va="bottom")
        ax.set_xlim(-1, 1.1 * float(table["concurrency"].max()) + 1)   # up to the largest bin, not to the empty end of the last one
        ax.set_xlabel("concurrency c: other services processing at the same time")
        ax.set_ylabel("slowdown (tree ÷ benchmark)")
        ax.grid(True, linestyle=":", alpha=0.6)
        legend_inside(ax, handles, labels, segments, locations=("upper left", "upper center"), max_doublings=0, fontsize=10)
        if name is not None:
            save_figure(fig, name)
        plt.show()


def slowdown_onset(slowdowns, tree_set: str, threshold: float):
    """The height h at which the median slowdown of the services of a tree set (tree_slowdowns, over its services and
    runs) first exceeds `threshold`, interpolated linearly between that height and the one below it; None if no height
    exceeds it, and the first height if already that one does."""
    median = slowdowns[slowdowns["tree_set"] == tree_set].groupby("h")["slowdown"].median().sort_index()
    above = median.index[median.to_numpy() > threshold]
    if not len(above):
        return None
    h = above[0]
    below = median.index[median.index < h]
    if not len(below):
        return float(h)
    previous = below[-1]
    return float(previous + (threshold - median[previous]) / (median[h] - median[previous]) * (h - previous))


def tree_training_runs(experiment: str, seed, n_train: int, benchmarks: dict = None) -> set:
    """The benchmark runs, as (experiment, run_id), that real_tree_data draws to train the models of the tree `experiment`
    under `seed`: the same generator and the same draws, service type by service type in the same order."""
    from .real_data import TREE_BENCHMARKS, VM_TREE_BENCHMARKS, VM_V2, training_pool, tree_structure, unthrottled_pool
    rng_train = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])   # as in real_tree_data
    if benchmarks is None:
        benchmarks = VM_TREE_BENCHMARKS if experiment.startswith(VM_V2.name + "/") else TREE_BENCHMARKS
    chosen = set()
    for service in sorted(set(tree_structure(experiment)[2])):
        pool = unthrottled_pool(training_pool(service, campaign=benchmarks[service]))
        rows = pool.iloc[np.sort(rng_train.choice(len(pool), size=n_train, replace=False))]
        chosen |= set(zip(rows["experiment"], rows["run_id"]))
    return chosen


def emulated_tree_quantiles(experiment: str, seeds: list, n_train: int, level: float, coverage_check: dict, n_frames: int = 5000,
                            warmup_share: float = 0.1) -> dict:
    """Ground truth of the tree `experiment` of VM_V2 as if every service ran as fast as in its benchmark. Per held-out run
    (unthrottled_runs below TREE_THROTTLE_LIMIT, in the order of real_tree_data), n_frames Poisson arrivals at the rate of
    the run pass through the fork-join tree of FIFO services with one worker each (fifo_node_leaving; an aggregator
    receives a frame when the later of its children has released it). The execution time of a frame at a service is
    drawn with replacement from the processing times of the benchmark runs of its service type at its configuration
    (records.csv, measurement phase; VM_TREE_BENCHMARKS), plus the handling overhead of the tree (tree_handling_overhead).
    Per seed only benchmark runs that do not train the models under that seed (tree_training_runs), so that no training
    run is replayed; if a configuration has no other run, its training runs are used, with a printed note. Latency of a
    frame: from its arrival until the root releases it, the first warmup_share of the frames dropped; it leaves out the
    time between the services, as the methods do. seeds: those of real_tree_data ([base seed, s]); n_train: as there.
    Returns {s: (the `level` quantile per held-out run, the lower edge of its confidence interval)} as measured_quantiles,
    keyed by the last element of each seed."""
    import pandas as pd
    from .real_data import (TREE_THROTTLE_LIMIT, VM_TREE_BENCHMARKS, arrival_rates, experiment_folder, run_table, training_pool,
                            tree_handling_overhead, tree_structure, unthrottled_pool, unthrottled_runs)
    from .simulation import fifo_node_leaving, quantile_with_lower_bound
    from .trees import children_first
    children, nodes, services = tree_structure(experiment)
    runs = unthrottled_runs(experiment, below=TREE_THROTTLE_LIMIT)
    rates = arrival_rates(experiment).loc[runs].to_numpy()
    configurations = run_table(experiment).drop_duplicates("run_id").set_index("run_id").loc[runs]
    overhead = tree_handling_overhead(experiment)
    pools = {service: unthrottled_pool(training_pool(service, campaign=VM_TREE_BENCHMARKS[service])) for service in set(services)}
    processing = {}   # (experiment, run_id) of a benchmark run -> its processing times (s) in the measurement phase

    def processing_times(benchmark: str, run_id: str) -> np.ndarray:
        if (benchmark, run_id) not in processing:
            records = pd.read_csv(experiment_folder(benchmark) / "runs" / run_id / "service" / "records.csv",
                                  usecols=["admission_phase", "outcome", "started_at_ns", "finished_at_ns"])
            records = records[(records["admission_phase"] == "measure") & (records["outcome"] == "completed")]
            processing[(benchmark, run_id)] = ((records["finished_at_ns"] - records["started_at_ns"]) / 1e9).to_numpy()
        return processing[(benchmark, run_id)]

    results = {}
    for seed in seeds:
        excluded = tree_training_runs(experiment, seed, n_train)
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])   # a generator that real_tree_data does not use
        arrivals = np.cumsum(rng.exponential(1.0, (len(runs), n_frames)) / rates[:, None], axis=1)
        leaving, fallbacks = [None] * len(nodes), 0
        for position in children_first(children):
            node, pool = nodes[position], pools[services[position]]
            execution = np.empty((len(runs), n_frames))
            for row, run_id in enumerate(runs):
                matches = pool["data_quality"] == configurations.loc[run_id, f"{node}.data_quality"]
                if "model_size" in pool:
                    matches &= pool["model_size"] == configurations.loc[run_id, f"{node}.model_size"]
                candidates = [(r.experiment, r.run_id) for r in pool[matches].itertuples()]
                replayed = [run for run in candidates if run not in excluded] or candidates
                fallbacks += not [run for run in candidates if run not in excluded]
                samples = np.concatenate([processing_times(*run) for run in replayed])
                execution[row] = rng.choice(samples, n_frames) + overhead
            entering = arrivals if not children[position] else np.maximum.reduce([leaving[child] for child in children[position]])
            leaving[position] = fifo_node_leaving(entering, execution, 1)
            for child in children[position]:
                leaving[child] = None
        if fallbacks:
            print(f"emulated_tree_quantiles: {experiment}, seed {seed}: {fallbacks} node-runs replay training runs (no other run of their configuration)")
        latency = (leaving[0] - arrivals)[:, int(warmup_share * n_frames):]
        results[seed[-1]] = quantile_with_lower_bound(latency, level, **coverage_check)
    return results
