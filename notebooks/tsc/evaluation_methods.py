"""What the experiments in `evaluation.ipynb` share: the per-service models, the comparison of the proposed method with
every baseline, the coverage of a method, and the figures. The notebook only calls these functions; the bounds
themselves are in `snc_bounds.py` and the baselines in `baselines.py`.

Units: execution times and latencies in seconds; `alpha` is the violation probability epsilon and `confidence` the
probability with which the pessimistic mean of a bound lies above the true mean.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from baselines import (conformal_service_bounds, joint_gaussian_quantile, queueing_formula_latency, summed_service_quantiles,
                       train_dnn_models, union_bound_oracle)
from service_gp import execution_time_belief, train_gp_models
from snc_bounds import martingale_closed_form_node_bounds

PROPOSED = "Pegasus (ours)"
GROUP_COLORS = {"proposed": "tab:green", "baseline": "tab:gray", "reference": "tab:purple"}
METHOD_COLORS = {PROPOSED: "tab:green", "GP Sum": "tab:gray", "DNN Sum": "black", "Kingman": "tab:blue",
                 "Quantile Union": "tab:orange", "Conformal": "tab:red", "Oracle": "tab:purple"}


# ------------------------------------------------------------------------------
# Models and methods
# ------------------------------------------------------------------------------
def fit_services(data, x_train: np.ndarray, y_train: dict, model_seed: int, confidence: float) -> dict:
    """Per-service models of the execution time, trained on single-service data: a GP and a DNN per service, and the
    beliefs Pegasus builds from the GPs at the held-out configurations of `data`, per position of its chain (a
    pessimistic mean with a per-item variance, and the optimistic mean that the propagated bounds need). "seconds" holds
    the time it took to train the GPs and the DNNs. model_seed: seed of the DNN training."""
    services = sorted(set(data.path))
    started = time.perf_counter()
    gp = train_gp_models(x_train, y_train, services)
    gp_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dnn = train_dnn_models(x_train, y_train, services, seed=model_seed)
    dnn_seconds = time.perf_counter() - started
    belief = {s: execution_time_belief(gp[s], data.x_test, confidence) for s in services}
    lower_mean = {s: execution_time_belief(gp[s], data.x_test, 1 - confidence)[0] for s in services}
    return dict(gp=gp, dnn=dnn, seconds=dict(gp=gp_seconds, dnn=dnn_seconds),
                beliefs=[belief[s] for s in data.path], lower_means=[lower_mean[s] for s in data.path])


def pegasus_bound(data, models: dict, alpha: float) -> np.ndarray:
    """Pegasus: the union bound over the closed-form martingale node bounds with propagated service spacing, per
    held-out configuration (s). The waiting time at a service is bounded by Kingman's martingale bound exp(-theta w);
    with a Gaussian execution time, the latency bound of the service follows from one formula. The first service
    receives the external arrivals; every later one receives the departures of the service before it, which cannot be
    closer together than that service's execution times. Every position of the chain gets its own node bound."""
    K = len(data.path)
    return martingale_closed_form_node_bounds(models["beliefs"], models["lower_means"], epsilon=alpha / K, **data.workload).sum(axis=0)


def compare_methods(data, models: dict, x_train: np.ndarray, monitored_latencies: dict, alpha: float, confidence: float,
                    n_calibration: int, model_seed: int) -> tuple:
    """Pegasus, every baseline and the reference for the chain of `data` at its held-out configurations. Returns two
    dictionaries: name -> (group, end-to-end latency per configuration in s), and name -> (seconds to train the models
    the method uses, seconds to compute its latency at the held-out configurations); the reference has no time.
    models: from fit_services; x_train, monitored_latencies: the training configurations and the latencies of every
    service monitored alone at them (None leaves out the baselines that need them). The union bound gives every service
    the share alpha / K of the violation probability. model_seed: seed of the calibration split of Conformal."""
    K, workload = len(data.path), data.workload
    predictions = lambda fitted: [fitted[s].predict(data.x_test, return_std=True) for s in data.path]
    gp_seconds, dnn_seconds = models["seconds"]["gp"], models["seconds"]["dnn"]
    # name, group, time to train the models the method shares with others, function that computes its latencies
    candidates = [
        (PROPOSED, "proposed", gp_seconds, lambda: pegasus_bound(data, models, alpha)),
        ("GP Sum", "baseline", gp_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["gp"])), norm.ppf(1 - alpha))),
        ("DNN Sum", "baseline", dnn_seconds, lambda: joint_gaussian_quantile(*zip(*predictions(models["dnn"])), norm.ppf(1 - alpha))),
        ("Kingman", "baseline", gp_seconds, lambda: sum(queueing_formula_latency(mean, std ** 2, epsilon=alpha, **workload)
                                                        for mean, std in predictions(models["gp"]))),
    ]
    if monitored_latencies is not None:   # these two train their own models, so their training time is part of the computation
        candidates.append(("Quantile Union", "baseline", 0.0, lambda: summed_service_quantiles(x_train, monitored_latencies, data.path, alpha / K, data.x_test)))
        if len(x_train) - n_calibration >= 5:   # Conformal needs configurations left to train its model on
            candidates.append(("Conformal", "baseline", 0.0, lambda: conformal_service_bounds(
                x_train, monitored_latencies, data.path, alpha, data.x_test, miscoverage=1 - confidence, n_calibration=n_calibration, seed=model_seed)))
    methods, seconds = {}, {}
    for name, group, training_seconds, compute in candidates:
        started = time.perf_counter()
        methods[name] = (group, compute())
        seconds[name] = (training_seconds, time.perf_counter() - started)
    if data.node_latencies is not None:         # reference: what a union bound reaches with measured instead of predicted nodes
        methods["Oracle"] = ("reference", union_bound_oracle(data.node_latencies, alpha))
    return methods, seconds


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
    """Coverage and tightness of every method of a sweep in which every method is computed in every setting.
    results[(setting, seed)]: name -> (group, latency per held-out configuration in s); truth[(setting, seed)]:
    (measured quantile, lower edge of its confidence interval), per held-out configuration.
    Returns name -> dict with "coverage" and "tightness", both (settings, seeds), and "holds" (settings, see holds).
    The tightness of a seed is the median over its held-out configurations of latency / measured quantile. It is nan
    where that median is not finite, i.e. where the method has no finite latency at half of the configurations or more."""
    statistics = {}
    for name in results[(settings[0], seeds[0])]:
        covered = np.array([[coverage(results[(value, s)][name][1], truth[(value, s)][1]) for s in seeds] for value in settings])
        tightness = np.array([[np.median(results[(value, s)][name][1] / truth[(value, s)][0]) for s in seeds] for value in settings])
        statistics[name] = dict(coverage=covered, tightness=np.where(np.isfinite(tightness), tightness, np.nan), holds=holds(covered, confidence))
    return statistics


def sweep_conclusion(settings: list, statistics: dict, names: list, label) -> str:
    """Two Markdown bullets for the conclusion of a sweep, validity first: which of the methods `names` hold in every
    setting and where the others do not, then the tightness of the methods that hold (median over the seeds) in the
    first and the last setting. statistics: from sweep_statistics; label(setting): how one setting is written."""
    holding = [name for name in names if statistics[name]["holds"].all()]
    failing = {name: [label(value) for value, held in zip(settings, statistics[name]["holds"]) if not held] for name in names if name not in holding}

    def tightness_range(name):
        # median over the seeds; a seed without a finite tightness counts as infinitely loose
        typical = np.median(np.where(np.isnan(statistics[name]["tightness"]), np.inf, statistics[name]["tightness"]), axis=1)
        finite = np.flatnonzero(np.isfinite(typical))
        if len(finite) == 0:
            return f"{name} gives no finite bound for most seeds in any setting"
        text = f"{name} from {typical[finite[0]]:.2f}× at {label(settings[finite[0]])} to {typical[finite[-1]]:.2f}× at {label(settings[finite[-1]])}"
        without = [label(value) for value, tightness in zip(settings, typical) if not np.isfinite(tightness)]
        return text + (f", and no finite bound for most seeds at {', '.join(without)}" if without else "")

    return (f"- **Validity.** Methods that hold in every setting, i.e. reach the confidence level under every seed: {', '.join(holding) or 'none'}. "
            f"Methods that do not hold, and where: "
            + ("; ".join(f"{name} at {'every setting' if len(where) == len(settings) else ', '.join(where)}" for name, where in failing.items()) or "none")
            + ".\n"
            f"- **Tightness among the methods that hold** (median over the seeds): "
            + ("; ".join(tightness_range(name) for name in holding) or "no method holds")
            + ".")


# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
def tightness_figure(results: dict, data, title: str, level_label: str, seeds_reached: dict, n_seeds: int) -> None:
    """One row per method of `results` (name -> (group, end-to-end latency per held-out configuration in s)): its
    tightness, latency / measured quantile, over the configurations with a finite latency. level_label: name of the
    measured quantile (e.g. "p95"). seeds_reached: name -> number of the n_seeds seeds under which the method reaches
    the confidence level; the number is written right of the row, and the box of a method that does not reach the level
    under every seed is hatched instead of filled."""
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
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    ax.minorticks_off()
    ax.set_xlabel(f"tightness: end-to-end latency ÷ measured {level_label} (log scale)")
    ax.set_yticks(rows, names)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    groups = [group for group in GROUP_COLORS if any(results[name][0] == group for name in names)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[group], alpha=0.7) for group in groups]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="dimgray", hatch="////"))
    fig.legend(handles, groups + ["does not hold: below the confidence level under at least one seed"],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
    fig.suptitle(f"{title}: {' → '.join(data.path)}, {len(data.x_test)} held-out configurations,\n"
                 f"{data.workload['process']} arrivals at {data.workload['rate']:g}/s")
    plt.tight_layout()
    plt.show()


def sweep_line(ax, settings, values: np.ndarray, name: str, held: np.ndarray = None) -> None:
    """One method in a figure over a sweep. settings: the values of the horizontal axis; values: (settings, seeds), nan
    where the method has no value. The line is the median over the seeds, in which a seed without a value counts as
    infinitely large, so that a setting has no point if half of the seeds or more have no value. The band reaches from
    the smallest to the largest seed with a value. held: per setting, whether the method holds there (see holds); its
    marker is then filled, otherwise hollow. Without `held` every marker is filled."""
    typical = np.median(np.where(np.isnan(values), np.inf, values), axis=1)
    shown = np.isfinite(typical)
    settings, values, typical = np.asarray(settings)[shown], values[shown], typical[shown]
    color = METHOD_COLORS[name]
    filled = np.ones(len(settings), dtype=bool) if held is None else np.asarray(held)[shown]
    ax.plot(settings, typical, "-", color=color, label=name, linewidth=2.5 if name == PROPOSED else 1.5)
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
    for name in METHOD_COLORS:
        if name in statistics:
            sweep_line(ax, settings, statistics[name]["tightness"], name, held=statistics[name]["holds"])
    if log_x:
        ax.set_xscale("log")
    ax.set_xticks(settings, tick_labels or [f"{value:g}" for value in settings])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel(f"tightness: latency ÷ measured {level_label} (log scale)")
    ax.grid(True, linestyle=":", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    marker_handles, marker_labels = validity_legend()
    fig.legend(handles + marker_handles, labels + marker_labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def validity_legend() -> tuple:
    """Handles and labels that explain the filled and the hollow markers of sweep_line."""
    marker = lambda face: Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor=face, markeredgecolor="dimgray", markeredgewidth=1.5)
    return [marker("dimgray"), marker("white")], ["holds: reaches the confidence level under every seed", "does not hold"]
