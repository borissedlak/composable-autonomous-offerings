"""Draft of the methodology overview (Fig. 1 of the Pegasus paper): learn one model per service, bound each service as a
queue, compose the service bounds for a pipeline that is never executed.

Claim of the figure: only step 1 needs measurements, and only of the individual services; steps 2 and 3 are analytic.

Run from `notebooks/tsc/` with the venv:  ../../venv/Scripts/python draft_methodology_figure.py
Writes figures/tsc/methodology_overview_draft.pdf and .jpg.

Source of the data: the synthetic service types s1, s2, s3 of pegasus/data.py (service_seed below), with one knob (CPU)
varied and the other two fixed, so that a model can be drawn as a curve. Everything shown is computed: the GP fits, the
simulated latency of the first service, the service bounds and the simulated end-to-end latency. It is an illustration
on a setting chosen for readability, not a result of the evaluation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm

from pegasus.bounds import martingale_closed_form_node_bounds, martingale_theta, network_service_bound
from pegasus.data import synthetic_service, true_demand
from pegasus.service_gp import execution_time_belief, train_gp_models
from pegasus.simulation import simulate_chain

plt.style.use('default')
plt.rcParams.update({'font.size': 12})

OUTPUT = Path(__file__).resolve().parents[2] / "figures" / "tsc"
seed, service_seed = 35, 0
services = ["s1", "s2", "s3"]
alpha, confidence = 0.05, 0.95            # violation probability epsilon and confidence c of the guarantee
workload = dict(process="poisson", rate=6.0)   # external arrivals (items/s)
quality, parallelism = 1.0, 2.0           # the two knobs that are fixed
cpu_target = 1.25                         # configuration of the target pipeline (CPU cores), inside the gap of the training data
n_train = 14                              # training configurations per service
K = len(services)

# One fixed color per quantity, reused in every panel; the line styles and markers also separate them in grayscale
C_MEAN, C_EPISTEMIC, C_ALEATORIC, C_BOUND, C_SNC, C_MEASURED = "black", "tab:blue", "tab:orange", "tab:green", "tab:olive", "tab:purple"


def configurations(cpu):
    cpu = np.atleast_1d(cpu).astype(float)
    return np.column_stack([cpu, np.full_like(cpu, quality), np.full_like(cpu, parallelism)])


# ------------------------------------------------------------------------------
# Step 1: one GP per service, trained on execution times of that service only
# ------------------------------------------------------------------------------
rng = np.random.default_rng(seed)
specs = {s: synthetic_service(s, service_seed) for s in services}
# training configurations leave a gap around the target configuration: there the model is uncertain
cpu_train = np.sort(np.concatenate([rng.uniform(0.5, 1.05, n_train // 2), rng.uniform(1.45, 2.0, n_train - n_train // 2)]))
y_train = {s: true_demand(configurations(cpu_train), specs[s]) + specs[s]["noise"] * rng.standard_normal(n_train) for s in services}
models = train_gp_models(cpu_train[:, None], y_train, services)

cpu_grid = np.linspace(0.5, 2.0, 200)
z_c = norm.ppf(confidence)
target = np.array([[cpu_target]])
beliefs = [execution_time_belief(models[s], target, confidence) for s in services]            # (pessimistic mean, variance per item)
lower_means = [execution_time_belief(models[s], target, 1 - confidence)[0] for s in services]   # optimistic means

# ------------------------------------------------------------------------------
# Step 2: the first service as a queue (simulated for the picture) and its bound
# ------------------------------------------------------------------------------
node_bounds = martingale_closed_form_node_bounds(beliefs, lower_means, epsilon=alpha / K, **workload)[:, 0]   # d_i(epsilon / K), s
true_means = np.array([[true_demand(configurations(cpu_target), specs[s])[0] for s in services]])
noise = [specs[s]["noise"] for s in services]
first_latency = simulate_chain(true_means[:, :1], noise[:1], n_items=200_000, item_correlation=0.0, rng=np.random.default_rng([seed, 1]),
                               **workload)[0][0, :, 0]
first_quantile = np.quantile(first_latency, 1 - alpha / K)
# the bound on the tail of the latency of the first service (Kingman's bound on the waiting time and the Gaussian execution time)
mean_1, var_1 = beliefs[0][0][0], beliefs[0][1][0]
theta_1 = martingale_theta(np.array([mean_1]), np.array([var_1]), **workload)[0]
latency_grid = np.linspace(0.0, 1.35 * max(node_bounds[0], first_quantile), 400)
slack = latency_grid - mean_1
tail_bound = np.minimum(1.0, norm.sf(slack / np.sqrt(var_1))
                        + np.exp(-theta_1 * slack + 0.5 * theta_1 ** 2 * var_1 + norm.logcdf((slack - theta_1 * var_1) / np.sqrt(var_1))))

# ------------------------------------------------------------------------------
# Step 3: both compositions, and the simulated pipeline for reference
# ------------------------------------------------------------------------------
per_node_bound = node_bounds.sum()
snc_bound = network_service_bound(beliefs, epsilon=alpha, holder_exponents=[1] * K, **workload)[0]
end_to_end = simulate_chain(true_means, noise, n_items=200_000, item_correlation=0.0, rng=np.random.default_rng([seed, 2]), **workload)[1][0]
measured = np.quantile(end_to_end, 1 - alpha)
print(f"service bounds at epsilon/K (ms): {np.round(1e3 * node_bounds, 1)}; per node {1e3 * per_node_bound:.0f} ms, SNC {1e3 * snc_bound:.0f} ms, "
      f"simulated p{100 * (1 - alpha):g} {1e3 * measured:.0f} ms; utilization of the services {np.round(workload['rate'] * true_means[0], 2)}")

# ------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 7.2))
grid = fig.add_gridspec(4, 3, left=0.055, right=0.985, top=0.80, bottom=0.17, wspace=0.24, hspace=0.42, width_ratios=[1.0, 1.05, 1.05],
                        height_ratios=[1, 1, 1, 0.001])


def box(ax, x, y, w, h, text, **style):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", linewidth=1.5, **{"edgecolor": "black", "facecolor": "white", **style}))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")


def arrow(ax, start, end, **style):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="black", **style))


# --- Panel 1: the three service models -----------------------------------------
model_axes = []
for row, s in enumerate(services):
    ax = fig.add_subplot(grid[row, 0], sharex=model_axes[0] if model_axes else None)
    model_axes.append(ax)
    mu, _, var_epistemic, var_aleatoric = models[s].predict(cpu_grid[:, None], return_variance_components=True)
    band = z_c * np.sqrt(var_epistemic)
    ax.fill_between(cpu_grid, 1e3 * (mu - band), 1e3 * (mu + band), color=C_EPISTEMIC, alpha=0.3, linewidth=0,
                    label="uncertainty of the model")
    ax.plot(cpu_grid, 1e3 * mu, color=C_MEAN, linewidth=1.8, label="predicted mean")
    ax.plot(cpu_train, 1e3 * y_train[s], "o", color=C_ALEATORIC, markeredgecolor="black", markersize=6, label="observed execution times")
    ax.axvline(cpu_target, color="black", linewidth=1, linestyle=":")
    ax.plot([cpu_target], [1e3 * beliefs[row][0][0]], marker="^", color=C_EPISTEMIC, markeredgecolor="black", markersize=9, linestyle="none",
            label="pessimistic mean at the target")
    ax.set_ylabel(f"{s}: time (ms)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.margins(x=0)
    if row < len(services) - 1:
        ax.tick_params(labelbottom=False)
model_axes[-1].set_xlabel("configuration of the service (CPU cores)")
handles, labels = model_axes[0].get_legend_handles_labels()

# --- Panel 2: one service as a queue --------------------------------------------
ax_queue = fig.add_subplot(grid[0, 1])
ax_queue.set_axis_off()
ax_queue.set_xlim(0, 1)
ax_queue.set_ylim(0, 1)
ax_queue.text(0.14, 0.82, "arrivals of\nthe workload", ha="center", va="center")
arrow(ax_queue, (0.02, 0.48), (0.27, 0.48))
for j in range(4):
    ax_queue.add_patch(Rectangle((0.29 + 0.055 * j, 0.31), 0.045, 0.34, facecolor="0.8" if j else "white", edgecolor="black", linewidth=1.5))
ax_queue.text(0.395, 0.12, "waiting", ha="center", va="center")
box(ax_queue, 0.54, 0.26, 0.15, 0.44, "s1", facecolor=C_ALEATORIC, alpha=0.6)
ax_queue.text(0.615, 0.12, "execution", ha="center", va="center")
arrow(ax_queue, (0.71, 0.48), (0.98, 0.48))
ax_queue.text(0.85, 0.82, "to the next\nservice", ha="center", va="center")

# share of the items that are slower than a latency, on a logarithmic axis, because the bound concerns the tail
ax_latency = fig.add_subplot(grid[1:3, 1])
ordered = np.sort(first_latency)[::200]
ax_latency.plot(1e3 * ordered, 1 - np.arange(len(ordered)) / len(ordered), color="0.35", linewidth=1.8,
                label="true latency of s1 (simulated here,\nnot available to the method)")
ax_latency.plot(1e3 * latency_grid, tail_bound, color=C_BOUND, linewidth=2.2, linestyle="-.",
                label="bound on the tail, from the model\nof s1 and the workload")
ax_latency.axhline(alpha / K, color="black", linewidth=1, label=r"share $\epsilon / K$ allowed per service")
ax_latency.axvline(1e3 * node_bounds[0], color=C_BOUND, linewidth=2.2)
ax_latency.plot([1e3 * node_bounds[0]], [alpha / K], "o", color=C_BOUND, markeredgecolor="black", markersize=9)
ax_latency.annotate("service bound\n" + r"$d_1(\epsilon / K)$", xy=(1e3 * node_bounds[0], alpha / K), xytext=(1e3 * node_bounds[0] - 12, 0.45),
                    ha="right", va="center", color="darkgreen", arrowprops=dict(arrowstyle="->", color="darkgreen", linewidth=1.2))
ax_latency.set_yscale("log")
ax_latency.set_ylim(2e-5, 1.5)
ax_latency.set_xlim(0, 1e3 * latency_grid[-1])
ax_latency.set_xlabel("latency of an item at s1 = waiting + execution (ms)")
ax_latency.set_ylabel("share of items slower (log scale)", labelpad=8)
ax_latency.grid(True, linestyle=":", alpha=0.6)
ax_latency.legend(loc="lower left", fontsize=10.5, framealpha=0.95)

# --- Panel 3: the composition ----------------------------------------------------
ax_chain = fig.add_subplot(grid[0, 2])
ax_chain.set_axis_off()
ax_chain.set_xlim(0, 1)
ax_chain.set_ylim(0, 1)
ax_chain.add_patch(FancyBboxPatch((0.03, 0.22), 0.94, 0.62, boxstyle="round,pad=0.01,rounding_size=0.03", linewidth=1.5, linestyle="--",
                                  edgecolor="black", facecolor="none"))
for j, s in enumerate(services):
    box(ax_chain, 0.10 + 0.30 * j, 0.36, 0.17, 0.34, s, facecolor=C_ALEATORIC, alpha=0.6)
    if j:
        arrow(ax_chain, (0.10 + 0.30 * (j - 1) + 0.18, 0.53), (0.10 + 0.30 * j - 0.01, 0.53))
ax_chain.text(0.5, 0.06, "target pipeline: never executed", ha="center", va="center", style="italic")

ax_compose = fig.add_subplot(grid[1:3, 2])
left = 0.0
for j, (s, bound, hatch) in enumerate(zip(services, node_bounds, ("", "//", ".."))):
    ax_compose.barh(1, 1e3 * bound, left=left, height=0.5, color=C_BOUND, alpha=0.55 + 0.15 * j, edgecolor="black", hatch=hatch, linewidth=1.2)
    ax_compose.text(left + 1e3 * bound / 2, 1, rf"$d_{j + 1}$", ha="center", va="center", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
    left += 1e3 * bound
ax_compose.barh(0, 1e3 * snc_bound, height=0.5, color=C_SNC, alpha=0.75, edgecolor="black", linewidth=1.2)
ax_compose.axvline(1e3 * measured, color=C_MEASURED, linewidth=1.8, linestyle="--",
                   label=f"true end-to-end p{100 * (1 - alpha):g}\n(simulated; not used by the method)")
ax_compose.set_yticks([])
ax_compose.text(0, 1.29, " per service: sum of the service bounds", ha="left", va="bottom")
ax_compose.text(0, 0.29, " per pipeline: one bound for the whole pipeline (SNC)", ha="left", va="bottom")
ax_compose.set_ylim(-1.15, 1.75)
ax_compose.set_xlim(0, 1e3 * max(per_node_bound, snc_bound) * 1.25)
ax_compose.set_xlabel(r"end-to-end latency bound $d_S$ of the pipeline (ms)")
ax_compose.grid(True, axis="x", linestyle=":", alpha=0.6)
ax_compose.legend(loc="lower right", fontsize=11, framealpha=0.95)

# --- Frame: inputs, step titles, the divider between measured and analytic, the return arrow
fig.text(0.5, 0.965, r"Inputs:   composition $S$ (unseen)      configurations $\mathbf{x}$      workload (arrival rate $\lambda$, variability $C_a^2$)"
                     r"      violation probability $\epsilon$      confidence $c$",
         ha="center", va="center", fontsize=13, bbox=dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="black", linewidth=1.2))
positions = [model_axes[0].get_position(), ax_queue.get_position(), ax_chain.get_position()]
titles = ["① Learn: one model per service", "② Bound each service as a queue", "③ Compose the service bounds"]
for position, title in zip(positions, titles):
    fig.text((position.x0 + position.x1) / 2, 0.835, title, ha="center", va="center", fontsize=14, fontweight="bold")
divider = (positions[0].x1 + positions[1].x0) / 2 - 0.004
fig.add_artist(plt.Line2D([divider, divider], [0.12, 0.915], color="black", linewidth=1.5, linestyle=(0, (6, 4))))
fig.text((positions[0].x0 + positions[0].x1) / 2 - 0.01, 0.895, "needs measurements: of each service alone\nor in other compositions, once",
         ha="center", va="center", fontsize=12, style="italic")
fig.text((positions[1].x0 + positions[2].x1) / 2, 0.895, "analytic: no run of the composition", ha="center", va="center", fontsize=12, style="italic")
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=2, fontsize=11, frameon=False, columnspacing=1.0, handletextpad=0.5)
fig.add_artist(FancyArrowPatch((positions[2].x1, 0.04), (positions[1].x0, 0.04), arrowstyle="-|>", mutation_scale=18, linewidth=1.5, color="black",
                               transform=fig.transFigure))
fig.text((positions[1].x0 + positions[2].x1) / 2, 0.04, "new candidate composition or configuration: only ② and ③ are repeated", ha="center", va="center",
         fontsize=12, bbox=dict(facecolor="white", edgecolor="none", pad=2))

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "methodology_overview_draft.pdf", bbox_inches="tight")
fig.savefig(OUTPUT / "methodology_overview_draft.jpg", dpi=300, bbox_inches="tight", facecolor="white")
print(f"written to {OUTPUT / 'methodology_overview_draft.pdf'} and .jpg")
