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
    (the paper's assumption on internal arrivals). A service with several workers is bounded by cyclic assignment (bounds.py)."""
    K = len(data.path)
    return martingale_closed_form_node_bounds(models["beliefs"], epsilon=alpha / K, path_workers=path_workers(data), **data.workload).sum(axis=0)


def pegasus_snc_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus SNC: one bound for the whole chain from its network service curve, per held-out configuration (s). One
    Chernoff step on the product of the services' latency moments bounds the end-to-end latency at the full violation
    probability alpha, instead of alpha / K per service. Multiplying the moments requires that the execution times of
    an item are independent between the services, so no Hoelder step is applied (all exponents 1)."""
    return network_service_bound(models["beliefs"], epsilon=alpha, holder_exponents=[1] * len(data.path), path_workers=path_workers(data),
                                 **data.workload)


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
    configuration (model_inputs in data.py); GP Queue and both versions of Pegasus get it from path_workers."""
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
        methods[name] = (group, latency)
        seconds[name] = (training_seconds, elapsed + earlier_seconds.get(name, 0.0))
    if data.node_latencies is not None:         # reference: what a union bound reaches with measured instead of predicted nodes
        methods["Oracle"] = ("reference", union_bound_oracle(data.node_latencies, alpha))
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
    formula = lambda mean, var: sum(queueing_formula_latency(mean[:, k], var[:, k], epsilon=alpha, workers=workers[k], **data.workload)
                                    for k in range(len(data.path)))
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
        if len(finite) == 1:   # one setting, e.g. the one narrow tree of the real data of B.3
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
                              tick_labels: list = None, log_x: bool = True, name: str = None, time_halves: tuple = None) -> None:
    """Three separate figures over a sweep. First: the tightness of every method, as in tightness_sweep_figure.
    Second: the training time (s), one line per set of trained models; methods that use the same models (same_models)
    share a line, and its legend names them. Third: the evaluation time (s) of every method. Lines and bands of the time
    figures are the median and the range over the seeds (sweep_line). statistics: from sweep_statistics; times: from
    sweep_seconds. The title is printed above the figures; name: file name in figures/tsc, saved as <name>_tightness,
    <name>_training and <name>_evaluation, or None. time_halves: (xlabel, tick_labels) of the two time figures as
    half figures side by side in the width of one (half_figure, each with its own vertical axis), which need a shorter
    horizontal axis; None: full figures with the axis of the tightness figure."""
    from .figures import fit_half_pair, fit_x_tick_labels, half_figure, legend_inside, paper_figure, paper_style, plain_log_ticks, save_figure
    print(title)
    short = lambda method: method.replace(" (ours)", "")
    with paper_style():
        fig, ax = paper_figure()
        _sweep_axis(ax, settings, xlabel, tick_labels, log_x)
        _tightness_panel(ax, settings, statistics, level_label)
        if not log_x:
            fit_x_tick_labels(ax)
        _tightness_legend(ax, settings, statistics)
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


def _tightness_legend(ax, settings: list, statistics: dict, ncol: int = 3, half: str = None) -> None:
    """The legend of the methods and of the markers inside a tightness panel, where it hides the fewest points and
    bands (legend_inside); the axis keeps TIGHTNESS_RANGE, as in B.1. Call when the axes are complete. half: "left" lists the
    first half of the methods, "right" the rest and the markers, so that the two halves of a pair share one legend."""
    from .figures import holds_handles, legend_inside
    handles, labels = ax.get_legend_handles_labels()
    marker_handles, marker_labels = holds_handles()
    first = (len(handles) + 1) // 2
    if half == "left":
        handles, labels, marker_handles, marker_labels = handles[:first], labels[:first], [], []
    elif half == "right":
        handles, labels = handles[first:], labels[first:]
    segments = [segment for name in statistics if name in METHOD_COLORS for segment in sweep_segments(settings, statistics[name]["tightness"])]
    legend_inside(ax, handles + marker_handles, labels + marker_labels, segments, max_doublings=0, ncol=ncol, fontsize=10, framealpha=0.95,
                  columnspacing=1.0, handletextpad=0.4)


def uncertainty_figure(uncertainty: dict, chain_bound: dict, services: list, title: str, name: str = None) -> None:
    """How uncertain every service model is, and how much that uncertainty costs the chain, against the training runs
    of the service: two figures side by side in the width of one (half_figure, each with its own vertical axis), each
    with the training runs of the service on the horizontal axis (logarithmic) and one line per service.
    Left (relative uncertainty, s, logarithmic): the line is the predicted mean execution time m, the band reaches up to
    m * (1 + margin), the pessimistic mean of the bound, so that the height of the band on the logarithmic axis is the
    margin; every value the median over the seeds. The margin with all runs is also written as a percentage at the end
    of every line (_margin_labels). Right (effect on the chain, logarithmic): the uncertainty cost of
    uncertainty_cost in evidence.py over the end-to-end bound of the chain with the predicted means and all runs of
    every service (chain_bound), per seed; the line is the median over the seeds, the band reaches from the smallest
    to the largest seed, a level has no point where the median is infinite. A thin black line marks a cost as large as
    the whole chain bound. uncertainty: seed -> service -> array (levels, 5) of (training runs, margin, cost, mean,
    bound with the mean), each value the median over the held-out configurations; the runs of every level are the
    same under every seed. chain_bound: seed -> the median over the held-out configurations of the sum of the node
    bounds with the predicted means (s). The title is printed above the figures; name: file name in figures/tsc, saved
    as <name>_execution and <name>_share, or None."""
    from .figures import BAND_ALPHA, fit_half_pair, half_figure, legend_inside, paper_style, plain_log_ticks, save_figure
    print(title)
    seeds = list(uncertainty)
    median = lambda service, column: np.median([uncertainty[s][service][:, column] for s in seeds], axis=0)
    runs = {service: uncertainty[seeds[0]][service][:, 0] for service in services}
    # services are no methods: their own colors and markers, so that every line also reads in grayscale
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink")
    styles = (("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"), ("v", "-"), ("P", "--"))
    with paper_style():
        fig, ax = half_figure("left", shared_y=False)
        segments = []
        for k, service in enumerate(services):
            mean = median(service, 3)
            pessimistic = mean * (1 + median(service, 1))
            marker, linestyle = styles[k % len(styles)]
            color = colors[k % len(colors)]
            ax.fill_between(runs[service], mean, pessimistic, color=color, alpha=0.3, linewidth=0)
            ax.plot(runs[service], pessimistic, linestyle=linestyle, color=color, linewidth=0.8)
            ax.plot(runs[service], mean, linestyle=linestyle, marker=marker, color=color, linewidth=1.5, markersize=6, label=service)
            segments += list(zip(runs[service], mean, pessimistic))
        values = [value for segment in segments for value in segment[1:]]
        ax.set_ylim(min(values) / 2, max(values) * 10)   # room above the data for the legend
        handles, labels = ax.get_legend_handles_labels()
        left, left_segments, figures = ax, segments, [(fig, "execution")]

        fig, ax = half_figure("right", shared_y=False)
        segments = []
        ax.axhline(1.0, color="black", linewidth=1)
        for k, service in enumerate(services):
            share = np.stack([uncertainty[s][service][:, 2] / chain_bound[s] for s in seeds], axis=1)   # (levels, seeds)
            typical = np.median(share, axis=1)
            shown = np.isfinite(typical) & (typical > 0)
            low, high = np.min(share, axis=1), np.max(np.where(np.isfinite(share), share, np.nan), axis=1)
            marker, linestyle = styles[k % len(styles)]
            color = colors[k % len(colors)]
            ax.fill_between(runs[service][shown], low[shown], high[shown], color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(runs[service][shown], typical[shown], linestyle=linestyle, marker=marker, color=color, linewidth=1.5, markersize=6)
            segments += list(zip(runs[service][shown], low[shown], high[shown]))
        values = [value for segment in segments for value in segment[1:] if np.isfinite(value) and value > 0]
        ax.set_ylim(min(values) / 2, max(max(values), 1.0) * 3)
        # the reference line labeled where it is drawn, at the right end, where the costs have fallen far below it
        ax.text(0.97, 1.0, "cost = chain bound", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=10)
        figures.append((fig, "share"))

        for (fig, suffix), axis in zip(figures, (left, ax)):
            axis.set_xscale("log")
            axis.set_yscale("log")
            plain_log_ticks(axis.xaxis)
            plain_log_ticks(axis.yaxis)
            axis.set_xlabel("training runs of the service")
            axis.set_ylabel("predicted execution time (s)" if suffix == "execution" else "cost of uncertainty")
            axis.grid(True, linestyle=":", alpha=0.6)
        legend_inside(left, handles, labels, left_segments, locations=("upper right", "upper center"),
                      max_doublings=0, ncol=2, fontsize=10, framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
        fit_half_pair([left, ax])
        _margin_labels(left, {service: (runs[service], median(service, 3), median(service, 1)) for service in services},
                       {service: colors[k % len(colors)] for k, service in enumerate(services)})
        for fig, suffix, *_ in figures:
            if name is not None:
                save_figure(fig, f"{name}_{suffix}")
            plt.show()


def _margin_labels(ax, lines: dict, colors: dict, fontsize: float = 9) -> None:
    """The uncertainty margin delta with all runs as a percentage ("delta = +3%": the pessimistic mean lies 3% above the
    predicted one; one-sided, because the bound uses only the pessimistic mean)
    at the end of every line of uncertainty_figure, ending at its last point: above the band, or below the line where
    the next slower service lies less than a factor 3 higher, so that the label does not run into that service's line.
    lines: service -> (training runs, mean, margin)."""
    ends = {service: mean[-1] for service, (_, mean, _) in lines.items()}
    for service, (runs, mean, margin) in lines.items():
        above = min((value for value in ends.values() if value > ends[service]), default=np.inf) >= 3 * ends[service]
        anchor = mean[-1] * (1 + margin[-1]) if above else mean[-1]   # the top of the band, or the line
        ax.annotate(rf"$\delta$ = +{margin[-1]:.0%}", (runs[-1], anchor), xytext=(4, 3 if above else -5), textcoords="offset points", ha="right",
                    va="bottom" if above else "top", color=colors[service], fontsize=fontsize, fontweight="bold")
