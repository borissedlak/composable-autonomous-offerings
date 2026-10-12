"""What the experiments in `2_evaluation.ipynb` share: the per-service models, the comparison of the proposed method with
every baseline, the coverage of a method, and the figures. The notebook only calls these functions; the bounds
themselves are in `bounds.py` and the baselines in `baselines.py`.

Units: execution times and latencies in seconds; `alpha` is the violation probability epsilon and `confidence` the
probability with which the pessimistic mean of a bound lies above the true mean.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from .baselines import (conformal_service_bounds, joint_gaussian_quantile, queueing_formula_latency, summed_service_quantiles, train_dnn_models,
                        union_bound_oracle)
from .service_gp import (RunStatisticsModel, configurations_of, execution_time_belief, run_statistics_belief, train_gp_models,
                         train_run_statistics_models)
from .bounds import martingale_closed_form_node_bounds, network_service_bound

PROPOSED = "Pegasus per node (ours)"   # union bound over the node bounds
PROPOSED_SNC = "Pegasus SNC (ours)"    # one bound for the whole chain from its network service curve
GROUP_COLORS = {"proposed": "tab:green", "baseline": "tab:gray", "reference": "tab:purple"}
METHOD_COLORS = {PROPOSED: "tab:green", PROPOSED_SNC: "tab:olive", "GP Sum": "tab:gray", "DNN Sum": "black", "Kingman": "tab:blue",
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
    run means and uses the same spread GP."""
    services = sorted(set(data.path))
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
    belief = {s: belief_of(gp[s], configurations_of(data.x_test, s), confidence) for s in services}
    lower_mean = {s: belief_of(gp[s], configurations_of(data.x_test, s), 1 - confidence)[0] for s in services}
    belief_seconds = time.perf_counter() - started
    return dict(gp=gp, dnn=dnn, seconds=dict(gp=gp_seconds, dnn=dnn_seconds, beliefs=belief_seconds),
                beliefs=[belief[s] for s in data.path], lower_means=[lower_mean[s] for s in data.path])


def pegasus_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus per node: the union bound over the closed-form martingale node bounds with propagated service spacing, per
    held-out configuration (s). The waiting time at a service is bounded by Kingman's martingale bound exp(-theta w);
    with a Gaussian execution time, the latency bound of the service follows from one formula. The first service
    receives the external arrivals; every later one receives the departures of the service before it, which cannot be
    closer together than that service's execution times. Every position of the chain gets its own node bound."""
    K = len(data.path)
    return martingale_closed_form_node_bounds(models["beliefs"], models["lower_means"], epsilon=alpha / K, **data.workload).sum(axis=0)


def pegasus_snc_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus SNC: one bound for the whole chain from its network service curve, per held-out configuration (s). One
    Chernoff step on the product of the services' latency moments bounds the end-to-end latency at the full violation
    probability alpha, instead of alpha / K per service. Multiplying the moments requires that the execution times of
    an item are independent between the services, so no Hoelder step is applied (all exponents 1)."""
    return network_service_bound(models["beliefs"], epsilon=alpha, holder_exponents=[1] * len(data.path), **data.workload)


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
    differs by orders of magnitude between its configurations."""
    K, workload = len(data.path), data.workload
    log_target = data.y_train_spread is not None
    x_load, x_test_load = (x_train, data.x_test) if x_train_load is None else (x_train_load, data.x_test_load)
    predictions = lambda fitted: [fitted[s].predict(configurations_of(data.x_test, s), return_std=True) for s in data.path]
    gp_seconds, dnn_seconds = models["seconds"]["gp"], models["seconds"]["dnn"]
    # name, group, time to train the models the method shares with others, function that computes its latencies
    candidates = [
        (PROPOSED, "proposed", gp_seconds, lambda: pegasus_bound(data, models, alpha)),
        (PROPOSED_SNC, "proposed", gp_seconds, lambda: pegasus_snc_bound(data, models, alpha)),
        ("GP Sum", "baseline", gp_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["gp"])), norm.ppf(1 - alpha))),
        ("DNN Sum", "baseline", dnn_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["dnn"])), norm.ppf(1 - alpha))),
        ("Kingman", "baseline", gp_seconds, lambda: sum(queueing_formula_latency(mean, std ** 2, epsilon=alpha, **workload)
                                                        for mean, std in predictions(models["gp"]))),
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


def queueing_formula_inputs(data, models: dict, true_mean: np.ndarray, true_var: np.ndarray, alpha: float) -> tuple:
    """Diagnostic of the Kingman baseline (explain_5_what_kingman_inputs_cost.ipynb): the same queueing formula (queueing_formula_latency, one quantile per service
    at alpha, added) with different inputs for the mean and the variance of the execution time. "predicted" is the
    prediction of the GP of a service, as the baseline uses it; "true" is true_mean / true_var, (held-out
    configurations, services); "pessimistic" is the mean that Pegasus uses, i.e. the prediction shifted up by the GP's
    uncertainty about the mean (models["beliefs"]).
    Returns a dictionary variant -> end-to-end latency per held-out configuration (s), and the relative error of the
    predicted mean, predicted / true - 1, at the busiest service of every held-out configuration (the service with the
    largest true mean)."""
    predictions = [models["gp"][s].predict(configurations_of(data.x_test, s), return_std=True) for s in data.path]
    predicted_mean = np.column_stack([mean for mean, _ in predictions])
    predicted_var = np.column_stack([std ** 2 for _, std in predictions])
    pessimistic_mean = np.column_stack([mean for mean, _ in models["beliefs"]])
    formula = lambda mean, var: sum(queueing_formula_latency(mean[:, k], var[:, k], epsilon=alpha, **data.workload) for k in range(len(data.path)))
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
    ax.set_xticks(TIGHTNESS_TICKS, [f"{tick:g}×" for tick in TIGHTNESS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel(f"tightness: end-to-end latency ÷ measured {level_label} (log scale)")
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
    """One method in a figure over a sweep. settings: the values of the horizontal axis; values: (settings, seeds), nan
    where the method has no value. The line is the median over the seeds, in which a seed without a value counts as
    infinitely large, so that a setting has no point if half of the seeds or more have no value. The band reaches from
    the smallest to the largest seed with a value. held: per setting, whether the method holds there (see holds); its
    marker is then filled, otherwise hollow. Without `held` every marker is filled. label: legend entry (default: name)."""
    typical = np.median(np.where(np.isnan(values), np.inf, values), axis=1)
    shown = np.isfinite(typical)
    settings, values, typical = np.asarray(settings)[shown], values[shown], typical[shown]
    color = METHOD_COLORS[name]
    filled = np.ones(len(settings), dtype=bool) if held is None else np.asarray(held)[shown]
    ax.plot(settings, typical, "-", color=color, label=label or name, linewidth=2.5 if name in (PROPOSED, PROPOSED_SNC) else 1.5)
    ax.fill_between(settings, np.nanmin(values, axis=1), np.nanmax(values, axis=1), color=color, alpha=0.15)
    ax.scatter(settings, typical, s=30, facecolors=[color if f else "white" for f in filled], edgecolors=color, linewidths=1.5, zorder=3)


def tightness_sweep_figure(settings: list, statistics: dict, xlabel: str, title: str, level_label: str, tick_labels: list = None,
                           log_x: bool = True) -> None:
    """Tightness of every method over a sweep, one panel: per method the median over the seeds with the band from the
    smallest to the largest seed (sweep_line), a filled marker where the method holds and a hollow one where it does
    not. statistics: from sweep_statistics; tick_labels: text of every setting on the horizontal axis (default: its
    value); level_label: name of the measured quantile (e.g. "p95"). A method has no point where, under half of the
    seeds or more, it has no finite latency at half of the held-out configurations or more."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _sweep_axis(ax, settings, xlabel, tick_labels, log_x)
    _tightness_panel(fig, ax, settings, statistics, level_label, legend_columns=3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def tightness_and_time_figure(settings: list, statistics: dict, times: dict, xlabel: str, title: str, level_label: str,
                              tick_labels: list = None, log_x: bool = True) -> None:
    """Three panels over a sweep. Left: the tightness of every method, as in tightness_sweep_figure. Middle: the
    training time (s), one line per set of trained models; methods that use the same models (same_models) share a line,
    and the legend of the panel names them. Right: the evaluation time (s) of every method. Lines and bands of the time
    panels are the median and the range over the seeds (sweep_line). statistics: from sweep_statistics; times: from
    sweep_seconds."""
    fig, (ax_tight, ax_training, ax_evaluation) = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax in (ax_tight, ax_training, ax_evaluation):
        _sweep_axis(ax, settings, xlabel, tick_labels, log_x)
    _tightness_panel(fig, ax_tight, settings, statistics, level_label, legend_columns=4)
    short = lambda name: name.replace(" (ours)", "")
    for group in same_models(times, list(times)):
        sweep_line(ax_training, settings, times[group[0]]["training"], group[0], label=", ".join(short(name) for name in group))
    for name in times:
        sweep_line(ax_evaluation, settings, times[name]["evaluation"], name)
    for ax, what in ((ax_training, "training time"), (ax_evaluation, "evaluation time")):
        ax.set_yscale("log")
        ax.set_ylabel(f"{what} (s, log scale)")
    ax_training.legend(title="models trained for", fontsize=9, title_fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _sweep_axis(ax, settings: list, xlabel: str, tick_labels: list, log_x: bool) -> None:
    """Horizontal axis of a panel over a sweep: one tick per setting."""
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(settings, tick_labels or [f"{value:g}" for value in settings])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle=":", alpha=0.6)


def _tightness_panel(fig, ax, settings: list, statistics: dict, level_label: str, legend_columns: int) -> None:
    """Tightness of every method over a sweep on `ax`, with the line at 1× (the measured quantile) and the legend of
    the methods and of the markers below the figure. The axis shows TIGHTNESS_RANGE; a line leaves the panel where a
    method lies outside."""
    for name in METHOD_COLORS:
        if name in statistics:
            sweep_line(ax, settings, statistics[name]["tightness"], name, held=statistics[name]["holds"])
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(*TIGHTNESS_RANGE)
    ax.set_yticks(TIGHTNESS_TICKS, [f"{tick:g}×" for tick in TIGHTNESS_TICKS])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel(f"tightness: latency ÷ measured {level_label} (log scale)")
    handles, labels = ax.get_legend_handles_labels()
    marker_handles, marker_labels = validity_legend()
    fig.legend(handles + marker_handles, labels + marker_labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=legend_columns)


def validity_legend() -> tuple:
    """Handles and labels that explain the filled and the hollow markers of sweep_line."""
    marker = lambda face: Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor=face, markeredgecolor="dimgray", markeredgewidth=1.5)
    return [marker("dimgray"), marker("white")], ["holds: reaches the confidence level under every seed", "does not hold"]
