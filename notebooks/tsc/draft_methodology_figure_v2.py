"""Alternative draft of the methodology overview (Fig. 2 of the Pegasus paper, Section IV), restructured to match the
three subsections: (1) learn one GP per service, (2) turn it into a pessimistic execution-time model and the load
phi_i(theta) of every service, (3) compose: Pegasus Node bounds every service by Kingman's bound and adds the service
bounds d_i(epsilon / K); Pegasus SNC bounds the whole application in one step from the product of M_i / (1 - phi_i).
Notation as in the paper: x_i = <r_i, kappa_i>, c_i workers, K services, m_i^+ = m_i + z_gamma sigma_ep,i, load
phi_i(theta) = M_i(theta) E[exp(-theta T)]^c_i, decay rate theta_i^*, service bound d_i(epsilon), PPG d_S(x, epsilon, gamma).

Claim of the figure: only step 1 needs measurements, and only of the individual services; steps 2 and 3 are analytic,
and both versions of Pegasus build on the same execution-time models and loads of step 2.

Run from `notebooks/tsc/` with the venv:  ../../venv/Scripts/python draft_methodology_figure_v2.py
Writes figures/tsc/methodology_overview_v2_draft.pdf and .jpg (the first version, draft_methodology_figure.py, is kept).

Source of the data: the synthetic service types s1, s2, s3 of pegasus/data.py (service_seed below), with one knob (the
quality, a software knob kappa_i) varied and the other two fixed, so that a model can be drawn as a curve. Everything
shown is computed: the GP fits, the loads, the simulated latency of the first service, the bounds and the simulated
end-to-end latency. It is an illustration on a setting chosen for readability, not a result of the evaluation.
As in the first version, every service has two workers, to show how c_i enters the load. Section IV-C requires one
worker per service for Pegasus SNC (items must leave every service in the order of arrival); its bar is computed with
cyclic assignment like the node bounds, as an illustration of the composition, not as a guarantee for this setting.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm

from pegasus.bounds import log_worker_gap_mgf, martingale_closed_form_node_bounds, martingale_theta, network_service_bound
from pegasus.data import synthetic_service, true_demand
from pegasus.service_gp import execution_time_belief, train_gp_models
from pegasus.simulation import simulate_chain

plt.style.use('default')
plt.rcParams.update({'font.size': 12})

OUTPUT = Path(__file__).resolve().parents[2] / "figures" / "tsc"
seed, service_seed = 35, 0
services = ["s1", "s2", "s3"]             # synthetic service types; their training data is drawn in this order
path = ["s2", "s1", "s3"]                 # order of the types in the target pipeline; the figure names them by position, s1 to s3
alpha, confidence = 0.05, 0.95            # violation probability epsilon and confidence gamma of the guarantee
cpu, parallelism = 1.0, 2.0               # the two knobs that are fixed (CPU cores r_i, parallelism); the parallelism is the number of workers c_i of every service
workers = int(parallelism)
# external arrivals (items/s), as in the first version; with c_i = 2 workers this is a moderate load lambda m_i / c_i: the
# queue of the first service is still visible, but the bounds of step 3 are not dominated by how loose the queueing bound
# becomes near saturation
workload = dict(process="poisson", rate=7.2)
quality_target = 1.25                     # configuration of the target pipeline (quality), inside the gap of the training data
n_train = 7                               # training configurations per service
noisy_share, noisy_factor = 0.4, 2.0      # share of the training samples whose noise is noisy_factor times the service's noise level
K = len(path)

# One fixed color per quantity, reused in every panel; the line styles and markers also separate them in grayscale
C_MEAN, C_EPISTEMIC, C_ALEATORIC, C_BOUND, C_SNC, C_MEASURED = "black", "tab:blue", "tab:orange", "tab:green", "tab:olive", "tab:purple"
SERVICE_STYLES = [("0.1", "-"), ("0.4", "--"), ("0.6", ":")]   # (gray level, line style) of s1, s2, s3 in the load panel


def configurations(quality):
    quality = np.atleast_1d(quality).astype(float)
    return np.column_stack([np.full_like(quality, cpu), quality, np.full_like(quality, parallelism)])


# ------------------------------------------------------------------------------
# Step 1: one GP per service, trained on execution times of that service only (as in the first version)
# ------------------------------------------------------------------------------
rng = np.random.default_rng(seed)
specs = {s: synthetic_service(s, service_seed) for s in services}
# training configurations leave a gap around the target configuration: there the model is uncertain
quality_train = np.sort(np.concatenate([rng.uniform(0.5, 0.95, n_train // 2), rng.uniform(1.55, 2.0, n_train - n_train // 2)]))
# some measurements are noisier than the service itself (e.g., a short or disturbed benchmark run), so the observed
# execution times scatter visibly; the simulated ground truth keeps the service's own noise level
noise_scale = {s: specs[s]["noise"] * np.where(rng.random(n_train) < noisy_share, noisy_factor, 1.0) for s in services}
y_train = {s: true_demand(configurations(quality_train), specs[s]) + noise_scale[s] * rng.standard_normal(n_train) for s in services}
# with this few samples, the hyperparameter search can explain all variation as noise (a length scale near zero and a flat
# mean); a lower limit on the length scale (in units of the quality) encodes that execution time changes smoothly with the quality
min_length_scale = 0.3
models = train_gp_models(quality_train[:, None], y_train, services, min_length_scale=min_length_scale)

quality_grid = np.linspace(0.5, 2.0, 200)
z_c = norm.ppf(confidence)
target = np.array([[quality_target]])

# ------------------------------------------------------------------------------
# Step 2: pessimistic execution-time model (m_i^+, sigma_al,i) and load phi_i(theta) of every service
# ------------------------------------------------------------------------------
beliefs = [execution_time_belief(models[s], target, confidence) for s in path]   # (pessimistic mean m_i^+, variance sigma_al,i^2), s and s^2
path_workers = [workers] * K
theta_star = np.array([martingale_theta(mean, var, workers=workers, **workload)[0] for mean, var in beliefs])   # decay rates theta_i^* (1/s)
theta_grid = np.linspace(1e-3, 1.25 * theta_star.max(), 400)                      # 1/s
log_gap = log_worker_gap_mgf(workload["process"], workload["rate"], np.array([workers]), theta=theta_grid)[0]   # c_i ln E[exp(-theta T)]
log_load = [theta_grid * mean[0] + 0.5 * theta_grid ** 2 * var[0] + log_gap for mean, var in beliefs]          # ln phi_i(theta)

# ------------------------------------------------------------------------------
# Step 3: Pegasus Node (service bounds, added) and Pegasus SNC (one bound), and the simulated pipeline for reference
# ------------------------------------------------------------------------------
node_bounds = martingale_closed_form_node_bounds(beliefs, epsilon=alpha / K, path_workers=path_workers, **workload)[:, 0]   # d_i(epsilon / K), s
per_node_bound = node_bounds.sum()
snc_bound = network_service_bound(beliefs, epsilon=alpha, holder_exponents=[1] * K, path_workers=path_workers, **workload)[0]
true_means = np.array([[true_demand(configurations(quality_target), specs[s])[0] for s in path]])
noise = [specs[s]["noise"] for s in path]
# the first service alone under the external arrivals, for the tail panel of Pegasus Node
first_latency = simulate_chain(true_means[:, :1], noise[:1], n_items=200_000, item_correlation=0.0, rng=np.random.default_rng([seed, 1]),
                               workers=np.full((1, 1), workers), **workload)[0][0, :, 0]
end_to_end = simulate_chain(true_means, noise, n_items=200_000, item_correlation=0.0, rng=np.random.default_rng([seed, 2]),
                            workers=np.full((1, K), workers), **workload)[1][0]
measured = np.quantile(end_to_end, 1 - alpha)
# the bound on the tail of the latency of the first service: Kingman's bound on the waiting time and the Gaussian
# execution time (Eq. latency-tail of the paper)
mean_1, var_1 = beliefs[0][0][0], beliefs[0][1][0]
latency_grid = np.linspace(0.0, 1.6 * node_bounds[0], 400)
slack = latency_grid - mean_1
tail_bound = np.minimum(1.0, norm.sf(slack / np.sqrt(var_1))
                        + np.exp(-theta_star[0] * slack + 0.5 * theta_star[0] ** 2 * var_1 + norm.logcdf((slack - theta_star[0] * var_1) / np.sqrt(var_1))))
print(f"decay rates theta_i^* (1/s): {np.round(theta_star, 1)}; service bounds at epsilon/K (ms): {np.round(1e3 * node_bounds, 1)}; "
      f"Pegasus Node {1e3 * per_node_bound:.0f} ms, Pegasus SNC {1e3 * snc_bound:.0f} ms, simulated p{100 * (1 - alpha):g} {1e3 * measured:.0f} ms; "
      f"utilization of the services {np.round(workload['rate'] * true_means[0] / workers, 2)}")

# ------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7.2))   # narrower than the first version (15 in), so that the text is larger at the width of the page
grid = fig.add_gridspec(4, 3, left=0.055, right=0.985, top=0.80, bottom=0.17, wspace=0.24, hspace=0.42, width_ratios=[1.0, 0.85, 1.3],
                        height_ratios=[1, 1, 1, 0.001])
# the three service models are shorter than the other panels, so that their legend fits below them in the same column
model_column = grid[0:3, 0].get_position(fig)
# and start further right, so that the dataset D_i of every service can be drawn to their left
model_grid = fig.add_gridspec(3, 1, left=model_column.x0 + 0.05, right=model_column.x1 - 0.025, top=0.80, bottom=0.33, hspace=0.3)


def box(ax, x, y, w, h, text, **style):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", linewidth=1.5, **{"edgecolor": "black", "facecolor": "white", **style}))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")


def arrow(ax, start, end, **style):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="black", **style))


def narrow(ax, by=0.045):
    """Takes `by` (figure width) off ax, half on each side, so that it stays centered under its step."""
    position = ax.get_position()
    ax.set_position([position.x0 + by / 2, position.y0, position.width - by, position.height])


# --- Step 1: the three service models ----------------------------------------------
model_axes = []
for row, s in enumerate(path):
    ax = fig.add_subplot(model_grid[row, 0], sharex=model_axes[0] if model_axes else None)
    model_axes.append(ax)
    mu, _, var_epistemic, var_aleatoric = models[s].predict(quality_grid[:, None], return_variance_components=True)
    # both kinds of uncertainty as +-z_gamma bands, because step 2 treats them differently: the epistemic one shifts the
    # mean to m_i^+, the aleatoric one remains the variance per item
    band_aleatoric, band_epistemic = z_c * np.sqrt(var_aleatoric), z_c * np.sqrt(var_epistemic)
    ax.fill_between(quality_grid, 1e3 * (mu - band_aleatoric), 1e3 * (mu + band_aleatoric), color="#fdd0a2", linewidth=0,
                    label=r"aleatoric $\pm z_\gamma \sigma_{al,i}$")
    ax.fill_between(quality_grid, 1e3 * (mu - band_epistemic), 1e3 * (mu + band_epistemic), color="#a6c8e6", linewidth=0,
                    label=r"epistemic $\pm z_\gamma \sigma_{ep,i}$")
    ax.plot(quality_grid, 1e3 * mu, color=C_MEAN, linewidth=1.8, label=r"predicted mean $m_i$")
    ax.plot(quality_train, 1e3 * y_train[s], "o", color=C_ALEATORIC, markeredgecolor="black", markersize=6, label=r"observed $Y_i$ (dataset $\mathcal{D}_i$)")
    ax.axvline(quality_target, color="black", linewidth=1, linestyle=":", label=r"target configuration $x_i$")
    ax.text(0.02, 0.93, rf"$Y_{row + 1}$ (ms)", transform=ax.transAxes, ha="left", va="top")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.margins(x=0)
    if row < K - 1:
        ax.tick_params(labelbottom=False)
model_axes[-1].set_xlabel(r"quality $\kappa_i$ of the service")
model_axes[-1].set_xticks([0.5, 1.0, 1.5, 2.0])
handles, labels = model_axes[0].get_legend_handles_labels()

# the dataset D_i of every service, a database cylinder with an arrow into its model (figure coordinates)
cylinder_x, cylinder_w, cylinder_h, cap_h = 0.008, 0.034, 0.075, 0.02
for row, ax in enumerate(model_axes):
    position = ax.get_position()
    y_center = (position.y0 + position.y1) / 2
    bottom, top = y_center - cylinder_h / 2, y_center + cylinder_h / 2
    style = dict(facecolor="0.9", edgecolor="black", linewidth=1.3, transform=fig.transFigure)
    fig.add_artist(Ellipse((cylinder_x + cylinder_w / 2, bottom), cylinder_w, cap_h, **style))
    fig.add_artist(Rectangle((cylinder_x, bottom), cylinder_w, cylinder_h, **{**style, "edgecolor": "none"}))
    for x in (cylinder_x, cylinder_x + cylinder_w):
        fig.add_artist(plt.Line2D([x, x], [bottom, top], color="black", linewidth=1.3))
    fig.add_artist(Ellipse((cylinder_x + cylinder_w / 2, top), cylinder_w, cap_h, **style))
    fig.text(cylinder_x + cylinder_w / 2, y_center - cap_h / 4, rf"$\mathcal{{D}}_{row + 1}$", ha="center", va="center", fontsize=13)
    fig.add_artist(FancyArrowPatch((cylinder_x + cylinder_w + 0.004, y_center), (position.x0 - 0.03, y_center), arrowstyle="-|>",
                                   mutation_scale=14, linewidth=1.4, color="black", transform=fig.transFigure))

# --- Step 2: one service as a queue, and the load of every service ------------------
ax_queue = fig.add_subplot(grid[0, 1])
ax_queue.set_axis_off()
# taller than one row of the grid (the load panel below gives up the same height), so that the two workers and the
# two lines on the execution time fit below the queue
QUEUE_EXTRA = 0.08   # figure height
position = ax_queue.get_position()
ax_queue.set_position([position.x0, position.y0 - QUEUE_EXTRA, position.width, position.height + QUEUE_EXTRA])
ax_queue.set_xlim(0, 1)
ax_queue.set_ylim(-0.75, 1.0)
ax_queue.text(0.12, 0.84, "arrivals\n" + r"($\lambda$, $C_a^2$)", ha="center", va="center")
arrow(ax_queue, (0.02, 0.47), (0.25, 0.47))
for j in range(4):
    ax_queue.add_patch(Rectangle((0.27 + 0.045 * j, 0.33), 0.037, 0.28, facecolor="0.8" if j else "white", edgecolor="black", linewidth=1.5))
# the c_i workers of s_i; under cyclic assignment, the bound sends every c_i-th item to the same worker
for k, y in enumerate((0.58, 0.08)):
    arrow(ax_queue, (0.455, 0.47), (0.535, y + 0.14))
    box(ax_queue, 0.54, y, 0.17, 0.28, rf"$s_i$ ({k + 1})", facecolor=C_ALEATORIC, alpha=0.6)
    arrow(ax_queue, (0.715, y + 0.14), (0.80, 0.47))
arrow(ax_queue, (0.80, 0.47), (0.98, 0.47))
ax_queue.text(0.90, 0.88, "to the next\nservice", ha="center", va="center")
ax_queue.text(0.34, 0.20, r"waiting $W_i$", ha="center", va="center")
ax_queue.text(0.625, -0.12, rf"$c_i = {workers}$ workers", ha="center", va="center")
ax_queue.text(0.5, -0.40, r"execution $Y_i \sim \mathcal{N}(m_i^+, \sigma_{al,i}^2)$", ha="center", va="center")
ax_queue.text(0.5, -0.65, r"pessimistic mean $m_i^+ = m_i + z_\gamma \sigma_{ep,i}$", ha="center", va="center")

ax_load = fig.add_subplot(grid[1:3, 1])
narrow(ax_load, by=0.03)
position = ax_load.get_position()
ax_load.set_position([position.x0 + 0.012, position.y0, position.width, position.height - QUEUE_EXTRA])
ax_load.axhline(0.0, color="black", linewidth=1, label=r"$\phi_i = 1$: limit of stability")
for j, (curve, (gray, linestyle)) in enumerate(zip(log_load, SERVICE_STYLES)):
    ax_load.plot(theta_grid, curve, color=gray, linestyle=linestyle, linewidth=2.2)
    # every curve named at its right end instead of in the legend, which would otherwise hide the dip or theta_i^*
    ax_load.text(theta_grid[-1] * 0.985, curve[-1], f"s{j + 1}", ha="right", va="bottom", color=gray,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
ax_load.plot(theta_star, np.zeros(K), "o", color=C_BOUND, markeredgecolor="black", markersize=9, linestyle="none",
             label=r"decay rate $\theta_i^*$", zorder=3)
ax_load.text(0.03, 0.96, r"$\phi_i(\theta) = M_i(\theta)\,\mathbb{E}[e^{-\theta T}]^{c_i}$", transform=ax_load.transAxes, ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
ax_load.set_xlim(0, theta_grid[-1])
# room above the highest curve for its name, and below the deepest dip for the legend
lowest, highest = min(curve.min() for curve in log_load), max(curve.max() for curve in log_load)
ax_load.set_ylim(lowest - 0.55 * (highest - lowest), highest + 0.12 * (highest - lowest))
ax_load.set_xlabel(r"$\theta$ (1/s)")
ax_load.set_ylabel(r"load $\ln \phi_i(\theta)$ of service $s_i$", labelpad=4)
ax_load.grid(True, linestyle=":", alpha=0.6)
# right of the dip and below the crossings, the corner that no curve reaches
ax_load.legend(loc="lower right", fontsize=10.5, framealpha=0.95)

# --- Step 3: the target pipeline and the two compositions ---------------------------
ax_chain = fig.add_subplot(grid[0, 2])
ax_chain.set_axis_off()
ax_chain.set_xlim(0, 1)
ax_chain.set_ylim(0, 1)
ax_chain.add_patch(FancyBboxPatch((0.03, 0.22), 0.94, 0.62, boxstyle="round,pad=0.01,rounding_size=0.03", linewidth=1.5, linestyle="--",
                                  edgecolor="black", facecolor="none"))
for j, s in enumerate(path):
    box(ax_chain, 0.10 + 0.30 * j, 0.36, 0.17, 0.34, f"s{j + 1}", facecolor=C_ALEATORIC, alpha=0.6)
    if j:
        arrow(ax_chain, (0.10 + 0.30 * (j - 1) + 0.18, 0.53), (0.10 + 0.30 * j - 0.01, 0.53))
ax_chain.text(0.5, 0.06, "target pipeline: never executed", ha="center", va="center", style="italic")

compose_grid = grid[1:3, 2].subgridspec(1, 2, width_ratios=[0.9, 1.35], wspace=0.30)
# Pegasus Node, per service: Kingman's bound on the tail of the latency of s1 gives its service bound d_1(epsilon / K)
ax_tail = fig.add_subplot(compose_grid[0, 0])
# shifted right, so that its tick labels and y label keep clear of the divider to step 2
position = ax_tail.get_position()
ax_tail.set_position([position.x0 + 0.03, position.y0, position.width - 0.03, position.height])
ordered = np.sort(first_latency)[::200]
ax_tail.plot(1e3 * ordered, 1 - np.arange(len(ordered)) / len(ordered), color="0.35", linewidth=1.8, label="true latency")
ax_tail.plot(1e3 * latency_grid, tail_bound, color=C_BOUND, linewidth=2.2, linestyle="-.", label="Kingman")
ax_tail.axhline(alpha / K, color="black", linewidth=1, label=r"$\epsilon / K$")
ax_tail.plot([1e3 * node_bounds[0]], [alpha / K], "o", color=C_BOUND, markeredgecolor="black", markersize=9, zorder=3)
ax_tail.set_yscale("log")
ax_tail.set_ylim(2e-6, 1.5)
ax_tail.set_xlim(0, 1e3 * latency_grid[-1])
ax_tail.set_xlabel(r"latency $d$ at s1 (ms)")
ax_tail.set_ylabel(r"$P(L_1 > d)$", labelpad=6)
ax_tail.grid(True, linestyle=":", alpha=0.6)
ax_tail.legend(loc="lower left", fontsize=10.5, framealpha=0.95, handlelength=1.4, handletextpad=0.4, borderaxespad=0.3)   # short handles: the panel is narrow
ax_tail.set_title(r"Node: $d_1(\epsilon / K)$ of s1", fontsize=12)

# both compositions as bars against the true end-to-end quantile
ax_compose = fig.add_subplot(compose_grid[0, 1])
left = 0.0
for j, (bound, hatch) in enumerate(zip(node_bounds, ("", "//", ".."))):
    ax_compose.barh(1, 1e3 * bound, left=left, height=0.5, color=C_BOUND, alpha=0.55 + 0.15 * j, edgecolor="black", hatch=hatch, linewidth=1.2)
    ax_compose.text(left + 1e3 * bound / 2, 1, rf"$d_{j + 1}$", ha="center", va="center", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
    left += 1e3 * bound
ax_compose.barh(0, 1e3 * snc_bound, height=0.5, color=C_SNC, alpha=0.75, edgecolor="black", linewidth=1.2)
# the formula of Pegasus SNC inside its bar, as the d_i inside the bar of Pegasus Node
ax_compose.text(1e3 * snc_bound / 2, 0, r"$\Pi_i\, M_i / (1 - \phi_i)$", ha="center", va="center", fontsize=13,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
ax_compose.axvline(1e3 * measured, color=C_MEASURED, linewidth=1.8, linestyle="--", label=f"true end-to-end p{100 * (1 - alpha):g}")
ax_compose.set_yticks([])
label_box = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85)
ax_compose.text(0, 1.29, r" Node: sum of the $d_i(\epsilon / K)$", ha="left", va="bottom", bbox=label_box)
ax_compose.text(0, 0.29, r" SNC: one bound, $\epsilon$ once", ha="left", va="bottom", bbox=label_box)
ax_compose.set_ylim(-1.15, 1.75)
ax_compose.set_xlim(0, 1e3 * max(per_node_bound, snc_bound) * 1.12)
ax_compose.set_xlabel(r"PPG $d_S(\mathbf{x}, \epsilon, \gamma)$ (ms)")
ax_compose.grid(True, axis="x", linestyle=":", alpha=0.6)
ax_compose.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
# the service bound of s1 is the first part of the sum of Pegasus Node
fig.add_artist(ConnectionPatch(xyA=(1e3 * node_bounds[0], alpha / K), coordsA=ax_tail.transData, xyB=(0.0, 1.0),
                               coordsB=ax_compose.transData, arrowstyle="-|>", mutation_scale=14, color="darkgreen", linewidth=1.3,
                               connectionstyle="arc3,rad=-0.2"))

# --- Frame: inputs, step titles, the dividers between the steps
# x_i as in Section III-A of the paper: resources and knobs; the workers c_i are a separate input. "Never executed" is left
# to step 3, so that the box fits the width of the figure
fig.text(0.5, 0.925, r"Inputs:   composition $S$    configurations $\mathbf{x} = (x_1, \ldots, x_K)$, $x_i = \langle r_i, \kappa_i \rangle$"
                     r"    workers $c_i$    arrival process ($\lambda$, $C_a^2$)    violation probability $\epsilon$    confidence $\gamma$",
         ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="0.95", edgecolor="black", linewidth=1.2))
positions = [model_axes[0].get_position(), ax_queue.get_position(), ax_chain.get_position()]
# one divider between every two steps: between the columns of the grid, so that both stay clear of the panels
dividers = [(model_column.x1 + positions[1].x0) / 2 - 0.004, (positions[1].x1 + positions[2].x0) / 2]
for divider in dividers:
    fig.add_artist(plt.Line2D([divider, divider], [0.12, 0.88], color="black", linewidth=1.5, linestyle=(0, (6, 4))))
# every step title is centered in its section, the space between the figure edge and the dividers
centers = [(0.0 + dividers[0]) / 2, (dividers[0] + dividers[1]) / 2, (dividers[1] + 1.0) / 2]
titles = ["① Learning: one GP per service", "② Execution time and load", "③ Composition into the PPG"]
for center, title in zip(centers, titles):
    fig.text(center, 0.86, title, ha="center", va="center", fontsize=14, fontweight="bold")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.255), ncol=2, fontsize=11, frameon=False, columnspacing=1.0, handletextpad=0.5)

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "methodology_overview_v2_draft.pdf", bbox_inches="tight", pad_inches=0)
fig.savefig(OUTPUT / "methodology_overview_v2_draft.jpg", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
print(f"written to {OUTPUT / 'methodology_overview_v2_draft.pdf'} and .jpg")
