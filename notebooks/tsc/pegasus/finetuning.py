"""Experiment C.1 of `2e_iterative.ipynb`: fine-tuning a latency bound with end-to-end samples, and the baselines that
are trained on runs of the chain itself.

An *end-to-end sample* is one run of the chain whose end-to-end latency quantile was measured: one of the held-out
configurations of an experiment, taken out of the test set. Everything here works on the results of B.1, so nothing
of B has to be computed again.
Units: latencies in seconds; `alpha` is the violation probability epsilon.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .baselines import summed_service_quantiles
from .baselines_chain import fluxion_training_rows, train_fluxion_chain
from .evaluation import METHOD_COLORS, PROPOSED, PROPOSED_SNC
from .data import model_inputs
from .service_gp import configurations_of

QUANTILE_SUM, FLUXION = "Quantile Sum", "Fluxion-style"
# Colors of the two baselines of C.1, to be added to METHOD_COLORS of evaluation.py by the notebook
COLORS = {QUANTILE_SUM: "tab:pink", FLUXION: "tab:brown"}
# Marker and line style of every method of C.1, so that the figure also reads in grayscale (as draft_problem_figure.py)
STYLES = {PROPOSED: "o-", PROPOSED_SNC: "D-", QUANTILE_SUM: "^-.", FLUXION: "s--"}


def split_samples(n_configurations: int, n_samples_max: int, rng: np.random.Generator) -> tuple:
    """Splits the held-out configurations of a chain at random into those that can serve as end-to-end samples and the
    test set. Returns the indices of n_samples_max sample configurations, in the order in which they are used (a
    smaller number of samples takes the first ones), and the indices of the remaining configurations, the test set,
    which is the same for every number of samples."""
    order = rng.permutation(n_configurations)
    return order[:n_samples_max], np.sort(order[n_samples_max:])


def calibration_factor(bound: np.ndarray, measured: np.ndarray, miscoverage: float) -> float:
    """Factor by which a bound is multiplied so that it covers a new end-to-end sample with probability at least
    1 - miscoverage (split-conformal calibration). bound, measured: the bound and the measured latency quantile at the
    n end-to-end samples. The score of a sample is measured / bound: below 1 where the bound has room, above 1 where it
    is violated (0 where the bound is infinite). The factor is the ceil((n + 1)(1 - miscoverage))-th smallest score.
    For samples that are exchangeable with the configuration the bound is used at, bound * factor then lies above the
    measured quantile with probability at least 1 - miscoverage, over the draw of the samples and of that
    configuration. Returns inf if n is too small for that level (n < 19 for a miscoverage of 0.05)."""
    scores = np.sort(np.asarray(measured, dtype=float) / np.asarray(bound, dtype=float))
    rank = int(np.ceil((len(scores) + 1) * (1 - miscoverage)))
    return float(scores[rank - 1]) if rank <= len(scores) else np.inf


def finetuned_bound(bound: np.ndarray, measured: np.ndarray, samples: np.ndarray, test: np.ndarray, miscoverage: float) -> np.ndarray:
    """A bound at the test configurations, fine-tuned with the end-to-end samples: multiplied by calibration_factor.
    The factor can lie below 1 (the bound was looser than the samples require, and becomes tighter) or above 1 (the
    samples violate the bound, and it is raised). With too few samples for the level, the bound stays as it is and
    keeps its own guarantee. An infinite bound stays infinite. bound, measured: at all held-out configurations."""
    factor = calibration_factor(bound[samples], measured[samples], miscoverage)
    return bound[test] * (factor if np.isfinite(factor) else 1.0)


def _chain_inputs(data, service: str) -> np.ndarray:
    """Inputs of a model of one service inside the chain at all held-out configurations: its configuration, and with
    real data also the arrival rate of the run (ChainData.x_test_load); without the number of workers (model_inputs)."""
    return configurations_of(model_inputs(data.x_test if data.x_test_load is None else data.x_test_load, data.workers_column), service)


def quantile_sum(data, samples: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile Sum at the test configurations: one GP per service predicts the (1 - alpha) latency quantile that the
    service had inside the chain, trained on the end-to-end samples (data.node_latencies of the sample runs), and the
    predictions are added. Not a guarantee: a point prediction per service, and the quantile of a sum is not the sum of
    the quantiles. With real data the GPs are fitted on the logarithm of the quantile, as in B."""
    inputs = {s: _chain_inputs(data, s) for s in set(data.path)}
    latencies = {s: data.node_latencies[samples][:, :, k] for k, s in enumerate(data.path)}
    return summed_service_quantiles({s: x[samples] for s, x in inputs.items()}, latencies, data.path, alpha, {s: x[test] for s, x in inputs.items()},
                                    log_target=data.y_train_spread is not None)


def fluxion_style(data, samples: np.ndarray, test: np.ndarray, alpha: float, levels: tuple = (0.5, 0.75, 0.9)) -> np.ndarray:
    """Fluxion-style composition (baselines_chain.py, part 4) at the test configurations, trained on the end-to-end
    samples: one GP per position of the chain and quantile level, which receives the configuration of the whole chain
    and the predicted quantiles of the position before it. levels: the quantile levels passed along besides
    1 - alpha. Not a guarantee."""
    levels = sorted({*levels, 1 - alpha})
    x = np.column_stack([_chain_inputs(data, s) for s in dict.fromkeys(data.path)])   # configuration of the whole chain
    chain = train_fluxion_chain(fluxion_training_rows(x[samples], data.node_latencies[samples], levels), levels)
    return chain.latency_bound(x[test], alpha, len(data.path))


def finetuning_methods(data, methods: dict, samples: np.ndarray, test: np.ndarray, alpha: float, miscoverage: float) -> dict:
    """Every method of C.1 at the test configurations, name -> (group, latency in s): both versions of Pegasus,
    fine-tuned with the end-to-end samples (finetuned_bound; with no samples, or too few for the level, the bound of
    B.1 as it is), and the two baselines trained on these samples, which need at least one sample. methods: the results
    of B.1 for the chain, name -> (group, latency at all held-out configurations), from compare_methods."""
    results = {name: ("proposed", finetuned_bound(methods[name][1], data.measured, samples, test, miscoverage)) for name in (PROPOSED, PROPOSED_SNC)}
    if len(samples) == 0:
        return results
    results[QUANTILE_SUM] = ("baseline", quantile_sum(data, samples, test, alpha))
    results[FLUXION] = ("baseline", fluxion_style(data, samples, test, alpha))
    return results


def finetuning_figure(sizes: list, statistics: dict, level_label: str, output: Path = None) -> None:
    """Tightness of every method of C.1 over the number of end-to-end samples, in the style of the tightness figures of
    B (evaluation.py, figures.py): per method the median over the seeds with the band from the smallest to the largest
    seed (sweep_line), a filled marker where the method holds (statistics[name]["holds"]) and a hollow one where it does
    not, the measured quantile as a black line at 1x and the red zone below it, where a method is no bound. Unlike the
    figures of B, the vertical axis is linear and zoomed to the values of C.1, which lie close above 1x, so that their
    differences are visible. statistics: from sweep_statistics over `sizes`; a method without a value at a size (the
    baselines at n = 0) has no point there. output: path without suffix; its name is the file name in figures/tsc
    (save_figure), or None."""
    from matplotlib.ticker import FuncFormatter
    from .evaluation import sweep_line, sweep_segments
    from .figures import (FAILURE_COLOR, FAILURE_TEXT_COLOR, METHOD_STYLES, holds_handles, legend_inside, paper_figure, paper_style, save_figure,
                          math_label, tightness_label, tightness_tick)
    METHOD_STYLES.update({name: (style[0], style[1:]) for name, style in STYLES.items()})   # sweep_line draws a method by its name
    names = [name for name in STYLES if name in statistics]
    with paper_style():
        fig, ax = paper_figure()
        segments = []
        for name in names:
            sweep_line(ax, sizes, statistics[name]["tightness"], name, held=statistics[name]["holds"])
            segments += sweep_segments(sizes, statistics[name]["tightness"])
        # zoomed: from a little below 1x, so that the red zone shows, to a little above the highest band
        top = max(high for _, _, high in segments)
        ax.set_ylim(1 - 0.1 * (top - 1), top + 0.1 * (top - 1))
        ax.axhspan(ax.get_ylim()[0], 1.0, color=FAILURE_COLOR, linewidth=0)
        ax.axhline(1.0, color="black", linewidth=1)
        ax.text(0.99, 0.01, f"below the measured {math_label(level_label)}: no bound", transform=ax.transAxes, ha="right", va="bottom", color=FAILURE_TEXT_COLOR)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: tightness_tick(value)))
        ax.set_ylabel(tightness_label(level_label))
        ax.set_xticks(sizes)
        ax.set_xlabel("end-to-end samples n")
        ax.grid(True, linestyle=":", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        marker_handles, marker_labels = holds_handles()
        legend_inside(ax, handles + marker_handles, labels + marker_labels, segments, locations=("upper right", "upper center", "upper left"),
                      max_doublings=0, ncol=2, fontsize=10, framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
        if output is not None:
            save_figure(fig, output.name)
        plt.show()
