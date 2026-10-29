"""Experiment B.2 of `2b_evidence.ipynb`: which service of a chain should get more training data. Every service starts
with a few training runs, and one service after the other moves up to its next number of runs (`levels`); a
*schedule* decides which service moves next:

- ALL_TOGETHER: the services in turn, in the order of the chain, so that all of them grow alike (the reference).
- MOST_UNCERTAIN: the service whose uncertainty costs the bound of Pegasus Node the most (uncertainty_cost), computed
  from the models and the held-out configurations only, before any run of the chain: a rule that a user can follow.
- LEAST_UNCERTAIN: the service whose uncertainty costs the least, the control.

The training runs of every service come in one fixed random order per seed (n_train=None in real_chain_data), and a
training set of n runs is the first n of them (training_prefix), so that a step only adds runs.
"""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from .bounds import martingale_closed_form_node_bounds
from .data import model_inputs, path_workers
from .evaluation import TIGHTNESS_RANGE, TIGHTNESS_TICKS, compare_methods, fit_services, sweep_line, sweep_segments
from .figures import FAILURE_COLOR, FAILURE_TEXT_COLOR, holds_handles, legend_inside, paper_figure, paper_style, save_figure
from .service_gp import configurations_of, execution_time_belief, run_statistics_belief

ALL_TOGETHER, MOST_UNCERTAIN, LEAST_UNCERTAIN = "all services together", "most uncertain first", "least uncertain first"
SCHEDULES = (ALL_TOGETHER, MOST_UNCERTAIN, LEAST_UNCERTAIN)
# Colors and styles of the schedules, to be added to METHOD_COLORS of evaluation.py and METHOD_STYLES of figures.py by the
# notebook (sweep_line draws by name)
COLORS = {ALL_TOGETHER: "black", MOST_UNCERTAIN: "tab:blue", LEAST_UNCERTAIN: "tab:orange"}
STYLES = {ALL_TOGETHER: ("o", "-"), MOST_UNCERTAIN: ("^", "--"), LEAST_UNCERTAIN: ("v", ":")}
TRAINING_FIELDS = ("y_train", "y_train_spread", "monitored_latencies", "monitored_arrivals", "x_train_load")   # per service, one row per run


def training_prefix(data, n_by_service: dict):
    """`data` with the first n_by_service[service] training runs of every service (the training fields of
    TRAINING_FIELDS and x_train, as a dictionary service -> array); the chain, the held-out configurations and the
    ground truth stay as they are."""
    x_train = {s: configurations_of(data.x_train, s)[:n] for s, n in n_by_service.items()}
    fields = {name: {s: getattr(data, name)[s][:n] for s, n in n_by_service.items()} for name in TRAINING_FIELDS
              if getattr(data, name) is not None}
    return replace(data, x_train=x_train, **fields)


def uncertainty_cost(data, models: dict, alpha: float) -> dict:
    """service -> how much the bound of Pegasus Node at that service would shrink if the model knew the mean execution
    time: the median over the held-out configurations of the node bound (at alpha / K, as in pegasus_bound) with the
    pessimistic mean of the belief minus the node bound with the predicted mean, in s; inf where only the pessimistic
    mean overloads the service. Uses the models and the held-out configurations and arrival rates only, no measured
    latency."""
    belief_of = run_statistics_belief if data.y_train_spread is not None else execution_time_belief
    x_test = model_inputs(data.x_test, data.workers_column)   # the models receive no number of workers
    predicted = [(belief_of(models["gp"][s], configurations_of(x_test, s), 0.5)[0], var) for s, (_, var) in zip(data.path, models["beliefs"])]
    workers, K = path_workers(data), len(data.path)
    pessimistic_bounds = martingale_closed_form_node_bounds(models["beliefs"], epsilon=alpha / K, path_workers=workers, **data.workload)
    predicted_bounds = martingale_closed_form_node_bounds(predicted, epsilon=alpha / K, path_workers=workers, **data.workload)
    with np.errstate(invalid="ignore"):
        return {s: float(np.median(np.where(np.isfinite(p), p - q, np.inf)))
                for s, p, q in zip(data.path, pessimistic_bounds, predicted_bounds)}


def service_levels(data, levels: list) -> dict:
    """service -> its numbers of training runs: the values of `levels` below the number of runs that `data` holds for
    the service, and then that number (all its runs)."""
    available = {s: len(configurations_of(data.x_train, s)) for s in data.path}
    return {s: [n for n in levels if n < available[s]] + [available[s]] for s in data.path}


def incremental_training(data, schedule: str, levels: list, alpha: float, confidence: float, calibration_share: float,
                         model_seed: int, leave_out: tuple) -> list:
    """Every method on the chain of `data` while the training runs grow under `schedule` (module docstring). `data`
    holds every training run of every service in a fixed random order; every service starts at its first level of
    service_levels and moves up one level per step, one service per step, until every service has all its runs.
    Conformal sets aside the share calibration_share of the runs of the smallest service for calibration, and at least
    19, and is left out where too few runs remain (compare_methods).
    Returns one entry per state, in order: (runs per service, name -> (group, latency per held-out configuration),
    name -> (training time, evaluation time), the service that moves up next or None)."""
    steps = service_levels(data, levels)
    level, states, turn = {s: 0 for s in data.path}, [], 0
    while True:
        n = {s: steps[s][level[s]] for s in data.path}
        part = training_prefix(data, n)
        models = fit_services(part, part.x_train, part.y_train, model_seed=model_seed, confidence=confidence, with_dnn=False)
        methods, seconds = compare_methods(part, models, part.x_train, part.monitored_latencies, alpha, confidence,
                                           n_calibration=max(19, round(calibration_share * min(n.values()))), model_seed=model_seed,
                                           leave_out=leave_out, x_train_load=part.x_train_load)
        growing = [s for s in data.path if level[s] < len(steps[s]) - 1]
        if not growing:
            states.append((n, methods, seconds, None))
            return states
        if schedule == ALL_TOGETHER:   # the next service in the order of the chain that can still grow
            chosen = next(data.path[(turn + k) % len(data.path)] for k in range(len(data.path)) if data.path[(turn + k) % len(data.path)] in growing)
            turn = data.path.index(chosen) + 1
        else:
            cost = uncertainty_cost(part, models, alpha)
            chosen = (max if schedule == MOST_UNCERTAIN else min)(growing, key=lambda s: cost[s])
        states.append((n, methods, seconds, chosen))
        level[chosen] += 1


def schedule_figure(totals: dict, statistics: dict, title: str, level_label: str, name: str = None) -> None:
    """The tightness of Pegasus Node under every schedule over the total number of training runs of the chain (all
    services together, logarithmic). statistics: from sweep_statistics over the steps of the schedule, with the
    schedules in place of the methods; totals: schedule -> total runs at every step, the median over the seeds; per
    step the median over the seeds, the band from the smallest to the largest seed, a filled marker where the bound
    holds, the red zone below 1x. The title is printed above the figure; name: file name in figures/tsc, saved as
    <name>_tightness, or None."""
    from .figures import tightness_label, tightness_tick
    print(title)
    xlabel = "training runs of the chain, all services (log scale)"
    with paper_style():
        fig, ax = paper_figure()
        ax.axhspan(TIGHTNESS_RANGE[0], 1.0, color=FAILURE_COLOR, linewidth=0)
        ax.axhline(1.0, color="black", linewidth=1)
        segments = []
        for schedule in statistics:
            sweep_line(ax, totals[schedule], statistics[schedule]["tightness"], schedule, held=statistics[schedule]["holds"])
            segments += sweep_segments(totals[schedule], statistics[schedule]["tightness"])
        ax.text(0.99, 0.03, "underestimated: SLO violated", transform=ax.transAxes, ha="right", va="bottom", color=FAILURE_TEXT_COLOR)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*TIGHTNESS_RANGE)
        ax.set_yticks(TIGHTNESS_TICKS, [tightness_tick(tick) for tick in TIGHTNESS_TICKS])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel(tightness_label(level_label))
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        marker_handles, marker_labels = holds_handles()
        legend_inside(ax, handles + marker_handles, labels + marker_labels, segments, max_doublings=0, ncol=2, fontsize=10, framealpha=0.95,
                      columnspacing=1.0, handletextpad=0.4)
        if name is not None:
            save_figure(fig, f"{name}_tightness")
        plt.show()
