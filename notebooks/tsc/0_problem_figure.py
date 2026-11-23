"""Two early figures (Section II-A) that show the two problems the paper addresses, on the real chain of B.1 and B.4a
(cv-analyzer -> image-transformer -> image-compressor), measured at several loads.

Claims of the figures:
 (a) Per-service execution times do not add up to the end-to-end tail latency: the gap is the queueing, and it grows
     with the load, so it cannot be calibrated away with a constant.
 (b) An accurate point prediction is not a guarantee: for unseen configurations it falls on both sides of the true
     latency, and nothing tells on which side.

Run from `notebooks/tsc/` with the venv:  ../../venv/Scripts/python problem_figure.py
Writes figures/tsc/problem_queueing.pdf and .jpg for (a), and figures/tsc/problem_prediction.pdf and .jpg for (b).

Source of the data: the measured runs of B.4a (statics/results_milos_v2/b4/, b4_chain_data in pegasus/real_data.py),
loaded as in the load sweep of the evaluation (`2_B4_load.ipynb`). B.4a measured the
chain with the configurations of b1u-vision-1 and b1u2-vision-1 at several utilizations u of the busiest service; every
utilization holds out the same runs (b4_held_out), at most MAX_WORKERS workers per service. The true latency of a run is
its measured end-to-end P95, which contains the handover between the services, as every method's prediction does.
Training: every service on N_TRAIN of its benchmark runs with one to four workers (WORKER_TRAINING; N_TRAIN in
pegasus/evaluation_setup.py, the number of every experiment of the evaluation except B.2), drawn at random per seed (the service alone); no run of the chain is used for
training, and every model receives the knobs, the number of workers and log10 of the arrival rate (the same information
for every method). The models of a seed are fitted once and predict at the arrival rate of every run and utilization.
 (a) the measured P95 and GP Sum at every utilization of B.4a, first seed;
 (b) the measured P95 against the predictions of GP Queue and Quantile Union at the loads PANEL_B_LOADS, every seed.
The methods are the baselines of the evaluation (pegasus/evaluation.py), trained on single-service data only.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from pegasus.data import load_aware
from pegasus.evaluation import METHOD_COLORS, compare_methods, fit_services
from pegasus.evaluation_setup import N_TRAIN
from pegasus.figures import METHOD_STYLES
from pegasus.real_data import WORKER_TRAINING, b4_chain_data

plt.style.use('default')
plt.rcParams.update({'font.size': 12})

OUTPUT = Path(__file__).resolve().parents[2] / "figures" / "tsc"
alpha, confidence = 0.05, 0.95                         # violation probability epsilon (the bounded quantile is $P_{95}$) and confidence, as in pegasus/evaluation_setup.py
coverage_check = dict(n_batches=10, confidence=0.95)
seed, n_calibration = 35, 20                           # base seed and calibration size, as in pegasus/evaluation_setup.py
utilizations = [0.1, 0.25, 0.4, 0.5, 0.7, 0.9]         # B.4a: utilization of the busiest service of every run (loads in pegasus/evaluation_setup.py)
MAX_WORKERS = 4                                        # held-out runs: at most this many workers per service, as in pegasus/evaluation_setup.py
PANEL_B_LOADS = [0.5, 0.7, 0.9]                        # panel (b): the utilizations whose runs are pooled, from the load of B.1 upward
seeds = list(range(5))                                 # a seed determines the training runs of every service
point_predictors = ["GP Queue", "Quantile Union"]       # panel (b): one analytic and one learned predictor of the latency quantile
p_label = f"$P_{{{100 * (1 - alpha):g}}}$"
FIGSIZE = (6, 2.5)                                      # (width, height) in inches of each panel; lower than the evaluation figures (6 x 3.7) so that Fig. 1 takes less space


def load(s) -> dict:
    """Utilization -> (data, methods) of the seed s: the held-out runs of B.4a at that utilization and the predictions of
    every method, as compare_methods returns them. The GPs are fitted once, on the data of the first utilization, and
    every later utilization checks that its training data is the same."""
    results, models, y_train = {}, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)   # GP hyperparameters at their bounds, as in pegasus/evaluation_setup.py
        for u in utilizations:   # the training pool of B.1 (WORKER_TRAINING): the benchmark runs with one to four workers
            data = b4_chain_data("load", u, utilizations, MAX_WORKERS, level=1 - alpha, coverage_check=coverage_check, seed=[seed, s],
                                 n_train=N_TRAIN, training_campaign=WORKER_TRAINING)
            if models is None:
                view = load_aware(data)       # the inputs of every model: the knobs, the workers and the arrival rate, as in B.1
                models, y_train = fit_services(view, view.x_train, data.y_train, model_seed=seed + s, confidence=confidence, with_dnn=False), data.y_train
            assert all(np.array_equal(data.y_train[node], y_train[node]) for node in data.path), "the training data differs between the loads"
            methods, _ = compare_methods(data, models, data.x_train, data.monitored_latencies, alpha, confidence, n_calibration, model_seed=seed + s,
                                         leave_out=("DNN Sum",), x_train_load=data.x_train_load, load_aware_models=True)
            results[u] = (data, methods)
    return results


loaded = {s: load(s) for s in seeds}

# Panel (a): the measured end-to-end quantile and GP Sum (the sum of the predicted execution times, with the handover)
# against the load, first seed, in ms per held-out run; the same runs at every load (b4_held_out)
measured_by_load = np.array([1e3 * loaded[seeds[0]][u][0].measured for u in utilizations])        # (loads, runs)
summed_by_load = np.array([1e3 * loaded[seeds[0]][u][1]["GP Sum"][1] for u in utilizations])

# Panel (b): predicted against measured quantile for every held-out run and seed at the loads PANEL_B_LOADS, pooled;
# seed_of: the seed of every point, because the scatter shows the points of one seed only (see the panel)
truth, predicted, seed_of = [], {name: [] for name in point_predictors}, []
for s in seeds:
    for u in PANEL_B_LOADS:
        data, methods = loaded[s][u]
        truth.append(data.measured)
        seed_of.append(np.full(len(data.measured), s))
        for name in point_predictors:
            predicted[name].append(methods[name][1])
truth, seed_of = np.concatenate(truth), np.concatenate(seed_of)
predicted = {name: np.concatenate(values) for name, values in predicted.items()}

OUTPUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(OUTPUT / f"{name}.jpg", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")


# --- (a) -----------------------------------------------------------------------
fig_load, ax_load = plt.subplots(figsize=FIGSIZE)
quartiles = lambda values: np.nanquantile(values, [0.25, 0.5, 0.75], axis=1)
gp_sum_style = "".join(METHOD_STYLES["GP Sum"])   # marker and line style of GP Sum in the evaluation figures
# the band of the black truth is fainter, so that it does not look like the gray band of the sum
for values, color, style, band_alpha, label in ((measured_by_load, "black", "o-", 0.1, f"true end-to-end {p_label}"),
                                                (summed_by_load, METHOD_COLORS["GP Sum"], gp_sum_style, 0.2, f"{p_label} of summed execution times (GP Sum)")):
    low, middle, high = quartiles(values)
    ax_load.plot(utilizations, middle, style, color=color, linewidth=1.8, markersize=6, label=label)
    ax_load.fill_between(utilizations, low, high, color=color, alpha=band_alpha, linewidth=0)
gap_low, gap_high = (np.median(measured_by_load[i]) / np.median(summed_by_load[i]) for i in (0, -1))
# the gap at the highest load, where it is largest; the label sits left of the arrow, inside the axes, and in the lower
# part of the gap, below the steep end of the curve
gap_bottom, gap_top = np.median(summed_by_load[-1]), np.median(measured_by_load[-1])
ax_load.annotate("", xy=(utilizations[-1], gap_top), xytext=(utilizations[-1], gap_bottom), arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
ax_load.text(utilizations[-1] - 0.015, gap_bottom + 0.3 * (gap_top - gap_bottom), "queueing", ha="right", va="center")
ax_load.set_xlabel("load: utilization of the busiest service")
ax_load.set_ylabel("latency (ms)")
ax_load.grid(True, linestyle=":", alpha=0.6)
ax_load.legend(loc="upper left", fontsize=11)
save(fig_load, "problem_queueing")

# --- (b) -----------------------------------------------------------------------
fig_scatter, ax_scatter = plt.subplots(figsize=FIGSIZE)
finite = {name: np.isfinite(values) for name, values in predicted.items()}
# The scatter shows the points of the first seed only: the seeds share the held-out runs and their true latency and
# differ only in the training runs, so every seed would add a near copy of every point. The shares in the legend and
# the printed statistics use every seed.
shown = {name: finite[name] & (seed_of == seeds[0]) for name in point_predictors}
# each axis spans the values it shows (x: the truth, y: the finite predictions), so that a few very large predictions
# do not leave most of the x axis empty; the zones and the diagonal are drawn over both ranges, so they stay correct
x_low, x_high = 1e3 * 0.7 * truth[seed_of == seeds[0]].min(), 1e3 * 1.4 * truth[seed_of == seeds[0]].max()
y_low = 1e3 * 0.7 * min(predicted[name][shown[name]].min() for name in point_predictors)
y_high = 1e3 * 1.4 * max(predicted[name][shown[name]].max() for name in point_predictors)
low, high = min(x_low, y_low), max(x_high, y_high)
ax_scatter.fill_between([low, high], [low, high], [high, high], color="0.9", linewidth=0)   # above the diagonal: overestimated
ax_scatter.fill_between([low, high], [low, low], [low, high], color="mistyrose", linewidth=0)   # below: underestimated
ax_scatter.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--", zorder=3)   # above the points (zorder 2), dashed so that the points stay visible behind it; no legend entry: the two zones it separates are labeled
for name in point_predictors:
    marker = METHOD_STYLES[name][0]   # the marker of the method in the evaluation figures
    below = np.mean(predicted[name][finite[name]] < truth[finite[name]])
    ax_scatter.plot(1e3 * truth[shown[name]], 1e3 * predicted[name][shown[name]], marker, color=METHOD_COLORS[name], markersize=5, alpha=0.6,
                    linestyle="none", label=f"{name}: {below:.0%} too low")
ax_scatter.text(0.03, 0.97, "overestimated: resources wasted", transform=ax_scatter.transAxes, ha="left", va="top")
ax_scatter.text(0.225, 0.03, "underestimated: SLO violated", transform=ax_scatter.transAxes, ha="left", va="bottom", color="darkred")   # as far left as stays inside the red zone, to leave the bottom right corner to the zoom inset
ax_scatter.set_xscale("log")
ax_scatter.set_yscale("log")
ax_scatter.set_xlim(x_low, x_high)
ax_scatter.set_ylim(y_low, y_high)
ax_scatter.set_xlabel(f"true end-to-end latency {p_label} (ms)")   # log scale, visible from the ticks
ax_scatter.set_ylabel(f"predicted {p_label} (ms)")
ax_scatter.grid(True, linestyle=":", alpha=0.6)
tick_text = lambda t: f"{t // 1000}k" if t >= 1000 else str(t)   # 1k = 1000 ms: short labels, so that neighbors on the log axis do not overlap
# x: 1-2-5 steps; y: 1-3 steps (about half a decade each), because the panel is low and spans more decades upward
for set_ticks, (a, b), candidates in ((ax_scatter.set_xticks, (x_low, x_high), (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)),
                                      (ax_scatter.set_yticks, (y_low, y_high), (10, 30, 100, 300, 1000, 3000, 10000, 30000))):
    ticks = [t for t in candidates if a <= t <= b]
    set_ticks(ticks, [tick_text(t) for t in ticks])
ax_scatter.minorticks_off()

# Zoom inset in the bottom right corner: the same points, in the window around a true latency of ZOOM_X ms, where the
# scatter is densest and the two predictors lie on both sides of the diagonal
ZOOM_X, ZOOM_Y = (150, 270), (107, 400)   # (ms) window of the inset: true latency (x) and predicted latency (y)
inset = ax_scatter.inset_axes([0.7453, 0.0391, 0.2512, 0.3636])
inset.set_facecolor("white")
zoom_line = np.array([1.0, 1e4])
inset.fill_between(zoom_line, zoom_line, 1e4, color="0.9", linewidth=0)
inset.fill_between(zoom_line, 1.0, zoom_line, color="mistyrose", linewidth=0)
inset.plot(zoom_line, zoom_line, color="black", linewidth=1, linestyle="--", zorder=3)
for name in point_predictors:
    inset.plot(1e3 * truth[shown[name]], 1e3 * predicted[name][shown[name]], METHOD_STYLES[name][0], color=METHOD_COLORS[name], markersize=5,
               alpha=0.6, linestyle="none")
inset.set_xscale("log")
inset.set_yscale("log")
inset.set_xlim(*ZOOM_X)
inset.set_ylim(*ZOOM_Y)
inset.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)   # no ticks (major and minor): the window is stated by the box in the main plot
ax_scatter.indicate_inset_zoom(inset, edgecolor="black", alpha=0.7, linewidth=1)
ax_scatter.legend(loc="upper left", bbox_to_anchor=(0, 0.88), fontsize=10, handlelength=1, borderpad=0.3, framealpha=0.8)   # in the empty corner below the zone label
save(fig_scatter, "problem_prediction")

print(f"(a) true {p_label} / sum of execution times, median over {measured_by_load.shape[1]} runs: {gap_low:.2f} at utilization {utilizations[0]:g}, "
      f"{gap_high:.2f} at utilization {utilizations[-1]:g}")
for name in point_predictors:
    ratio = predicted[name][finite[name]] / truth[finite[name]]
    print(f"(b) {name}: {np.mean(ratio < 1):.0%} too low, median ratio {np.median(ratio):.2f}, 5th to 95th percentile {np.quantile(ratio, 0.05):.2f} to "
          f"{np.quantile(ratio, 0.95):.2f}, finite at {finite[name].mean():.0%} of {len(truth)} predictions")
print(f"written to {OUTPUT / 'problem_queueing.pdf'} and {OUTPUT / 'problem_prediction.pdf'}, each also as .jpg")
