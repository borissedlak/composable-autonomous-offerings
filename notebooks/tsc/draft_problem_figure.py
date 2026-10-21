"""Draft of two early figures (Section II-A) that show the two problems the paper addresses, on the real chain of B.1.

Claims of the figures:
 (a) Per-service execution times do not add up to the end-to-end tail latency: the gap is the queueing, and it grows
     with the load, so it cannot be calibrated away with a constant. Adding a per-service queueing term (GP Queue)
     follows the trend of the true latency, but for a single run it falls on both sides of the true latency.
 (b) An accurate point prediction is not a guarantee: for unseen configurations it falls on both sides of the true
     latency, and nothing tells on which side.

Run from `notebooks/tsc/` with the venv:  ../../venv/Scripts/python draft_problem_figure.py
Writes figures/tsc/problem_queueing_draft.pdf and .jpg for (a), and figures/tsc/problem_prediction_draft.pdf and .jpg for (b).

Source of the data: the testbed campaign in statics/results_milos (chain 04e, qr -> cv -> face), loaded with
pegasus/real_data.py as in the evaluation (`2a_base_case.ipynb` to `2e_iterative.ipynb`). The end-to-end latencies are emulated: the measured processing times of
every run are replayed through a chain of FIFO queues under Poisson arrivals. The methods are the baselines of
the evaluation (`2a_base_case.ipynb` to `2e_iterative.ipynb`) (pegasus/evaluation.py): GP Sum, GP Queue and Quantile Union, trained on single-service data only.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from pegasus.data import load_chain_data
from pegasus.evaluation import METHOD_COLORS, compare_methods, fit_services
from pegasus.real_data import chain_run_execution_times

plt.style.use('default')
plt.rcParams.update({'font.size': 12})

OUTPUT = Path(__file__).resolve().parents[2] / "figures" / "tsc"
experiment = "04e-chain-qr-cv-face"
alpha, confidence = 0.05, 0.95                         # violation probability epsilon (the bounded quantile is p95) and confidence, as in pegasus/evaluation_setup.py
coverage_check = dict(n_batches=10, confidence=0.95)
seed, n_calibration = 35, 20                           # base seed and calibration size, as in pegasus/evaluation_setup.py
utilizations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # panel (a): utilization of the busiest service of every run
base_utilization = 0.5                                 # panel (b): the load of B.1
seeds = list(range(5))                                 # panel (b): a seed determines the training runs and the randomness of the emulation
point_predictors = ["GP Queue", "Quantile Union"]       # panel (b): one analytic and one learned predictor of the latency quantile
p_label = f"p{100 * (1 - alpha):g}"


def load(s, utilization):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)   # GP hyperparameters at their bounds, as in pegasus/evaluation_setup.py
        data = load_chain_data("real", experiment=experiment, utilization=utilization, level=1 - alpha, coverage_check=coverage_check, seed=[seed, s])
        models = fit_services(data, data.x_train, data.y_train, model_seed=seed + s, confidence=confidence, with_dnn=False)
        methods, _ = compare_methods(data, models, data.x_train, data.monitored_latencies, alpha, confidence, n_calibration, model_seed=seed + s,
                                     leave_out=("DNN Sum",), x_train_load=data.x_train_load)
    return data, methods


# Panel (a): the true end-to-end quantile and the sum of the predicted execution times against the load, first seed.
# Every run is divided by the sum of its true mean execution times, so that runs whose latencies differ by orders of
# magnitude can be shown together: 1 means "as long as the mean execution of all services".
total_execution = np.array([times.mean(axis=0).sum() for times in chain_run_execution_times(experiment)])   # s, per run of the chain
measured_by_load, summed_by_load, queueing_by_load = [], [], []
for utilization in utilizations:
    data, methods = load(seeds[0], utilization)
    measured_by_load.append(data.measured / total_execution)
    summed_by_load.append(methods["GP Sum"][1] / total_execution)
    queueing_by_load.append(methods["GP Queue"][1] / total_execution)
measured_by_load, summed_by_load = np.array(measured_by_load), np.array(summed_by_load)   # (loads, runs)
queueing_by_load = np.array(queueing_by_load)
queueing_by_load[~np.isfinite(queueing_by_load)] = np.nan   # the queueing formula has no finite value for a service it predicts to be overloaded

# Panel (b): predicted against true quantile for every held-out run and seed at the load of B.1
truth, predicted = [], {name: [] for name in point_predictors}
for s in seeds:
    data, methods = load(s, base_utilization)
    truth.append(data.measured)
    for name in point_predictors:
        predicted[name].append(methods[name][1])
truth = np.concatenate(truth)
predicted = {name: np.concatenate(values) for name, values in predicted.items()}

OUTPUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(OUTPUT / f"{name}.jpg", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")


# --- (a) -----------------------------------------------------------------------
fig_load, ax_load = plt.subplots(figsize=(6, 4))
quartiles = lambda values: np.nanquantile(values, [0.25, 0.5, 0.75], axis=1)
for values, color, style, label in ((measured_by_load, "tab:purple", "o-", f"true end-to-end {p_label}"),
                                    (queueing_by_load, METHOD_COLORS["GP Queue"], "^-.", f"sum of waiting + execution times ({p_label}, GP Queue)"),
                                    (summed_by_load, METHOD_COLORS["GP Sum"], "s--", f"sum of the per-service execution times ({p_label})")):
    low, middle, high = quartiles(values)
    ax_load.plot(utilizations, middle, style, color=color, linewidth=1.8, markersize=6, label=label)
    ax_load.fill_between(utilizations, low, high, color=color, alpha=0.2, linewidth=0)
gap_low, gap_high = (np.median(measured_by_load[i]) / np.median(summed_by_load[i]) for i in (0, -1))
ax_load.annotate("", xy=(utilizations[-2], np.median(measured_by_load[-2])), xytext=(utilizations[-2], np.median(summed_by_load[-2])),
                 arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
ax_load.text(utilizations[-2] + 0.012, (np.median(measured_by_load[-2]) + np.median(summed_by_load[-2])) / 2, "queueing", ha="left", va="center")
ax_load.set_xlabel("load: utilization of the busiest service")
ax_load.set_ylabel("latency ÷ total mean execution time")
ax_load.grid(True, linestyle=":", alpha=0.6)
ax_load.legend(loc="upper left", fontsize=11)
save(fig_load, "problem_queueing_draft")

# --- (b) -----------------------------------------------------------------------
fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
finite = {name: np.isfinite(values) for name, values in predicted.items()}
low = 1e3 * 0.7 * min(truth.min(), *(predicted[name][finite[name]].min() for name in point_predictors))
high = 1e3 * 1.4 * max(truth.max(), *(predicted[name][finite[name]].max() for name in point_predictors))
ax_scatter.fill_between([low, high], [low, high], [high, high], color="0.9", linewidth=0)   # above the diagonal: overestimated
ax_scatter.fill_between([low, high], [low, low], [low, high], color="mistyrose", linewidth=0)   # below: underestimated
ax_scatter.plot([low, high], [low, high], color="black", linewidth=1, label="prediction = true latency")
for name, marker in zip(point_predictors, ("o", "^")):
    below = np.mean(predicted[name][finite[name]] < truth[finite[name]])
    ax_scatter.plot(1e3 * truth[finite[name]], 1e3 * predicted[name][finite[name]], marker, color=METHOD_COLORS[name], markersize=5, alpha=0.6,
                    linestyle="none", label=f"{name}: {below:.0%} of the predictions too low")
ax_scatter.text(0.03, 0.97, "latency overestimated:\nresources wasted", transform=ax_scatter.transAxes, ha="left", va="top")
ax_scatter.text(0.97, 0.36, "latency underestimated:\nSLO violated", transform=ax_scatter.transAxes, ha="right", va="bottom", color="darkred")
ax_scatter.set_xscale("log")
ax_scatter.set_yscale("log")
ax_scatter.set_xlim(low, high)
ax_scatter.set_ylim(low, high)
ax_scatter.set_xlabel(f"true end-to-end {p_label} latency (ms, log scale)")
ax_scatter.set_ylabel(f"predicted {p_label} latency (ms, log scale)")
ax_scatter.grid(True, linestyle=":", alpha=0.6)
ticks = [t for t in (20, 50, 100, 200, 500, 1000) if low <= t <= high]
ax_scatter.set_xticks(ticks, [str(t) for t in ticks])
ax_scatter.set_yticks(ticks, [str(t) for t in ticks])
ax_scatter.minorticks_off()
ax_scatter.legend(loc="lower right", fontsize=11, framealpha=0.95)
save(fig_scatter, "problem_prediction_draft")

print(f"(a) true {p_label} / sum of execution times, median over {measured_by_load.shape[1]} runs: {gap_low:.2f} at utilization {utilizations[0]:g}, "
      f"{gap_high:.2f} at utilization {utilizations[-1]:g}")
for utilization, queueing, measured in zip(utilizations, queueing_by_load, measured_by_load):
    ratio = queueing / measured
    print(f"(a) GP Queue / true {p_label} at utilization {utilization:g}: median {np.nanmedian(ratio):.2f}, "
          f"{np.mean(ratio[np.isfinite(ratio)] < 1):.0%} too low, finite at {np.isfinite(ratio).mean():.0%} of the runs")
for name in point_predictors:
    ratio = predicted[name][finite[name]] / truth[finite[name]]
    print(f"(b) {name}: {np.mean(ratio < 1):.0%} too low, median ratio {np.median(ratio):.2f}, 5th to 95th percentile {np.quantile(ratio, 0.05):.2f} to "
          f"{np.quantile(ratio, 0.95):.2f}, finite at {finite[name].mean():.0%} of {len(truth)} predictions")
print(f"written to {OUTPUT / 'problem_queueing_draft.pdf'} and {OUTPUT / 'problem_prediction_draft.pdf'}, each also as .jpg")
