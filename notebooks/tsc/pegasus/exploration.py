"""Experiment B.2b of `2b_evidence.ipynb`: which configuration to benchmark next. A sampling strategy builds the
training configurations of the synthetic services one at a time; the adaptive ones choose the next configuration with
an acquisition function on the GPs that are fitted to the configurations benchmarked so far.

A *benchmark* of a configuration measures one execution time of every service of the chain at that configuration (in
the synthetic system, all services of a chain share one configuration). The synthetic system can be benchmarked at any
configuration (first part). For a real service, a strategy selects among the runs that the campaign measured (second
part): a benchmark is then one measured run of one service, with the mean and the spread of its processing times.
Units: execution times and latencies in seconds.
"""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, qmc

from .data import CONFIGURATION_RANGES, WORKERS_COLUMN, model_inputs, sample_configurations, synthetic_execution_moments, synthetic_execution_times, whole_workers
from .evaluation import TIGHTNESS_RANGE, TIGHTNESS_TICKS, fit_services, pegasus_bound, sweep_line, sweep_segments
from .figures import FAILURE_COLOR, FAILURE_TEXT_COLOR, holds_handles, legend_inside, paper_style, save_figure
from .real_data import (POOL_COLUMNS, chain_execution_moments, chain_path, encode_configurations, node_configurations, run_table,
                        training_pool)
from .service_gp import MIN_LENGTH_SCALE, train_gp_models

RANDOM, SOBOL, MAX_VARIANCE, VARIANCE_REDUCTION, UCB, EXPECTED_IMPROVEMENT = (
    "Random", "Sobol sequence", "Maximum variance", "Variance reduction", "Upper confidence bound", "Expected improvement")
STRATEGIES = (RANDOM, SOBOL, MAX_VARIANCE, VARIANCE_REDUCTION, UCB, EXPECTED_IMPROVEMENT)
# Colors of the strategies, to be added to METHOD_COLORS of evaluation.py by the notebook (sweep_line draws by name)
COLORS = {RANDOM: "tab:gray", SOBOL: "black", MAX_VARIANCE: "tab:blue", VARIANCE_REDUCTION: "tab:green", UCB: "tab:orange",
          EXPECTED_IMPROVEMENT: "tab:red"}
# marker and line style of every strategy, to be added to METHOD_STYLES of figures.py by the notebook, like COLORS
STYLES = {RANDOM: ("o", "-"), SOBOL: ("s", "--"), MAX_VARIANCE: ("^", "-."), VARIANCE_REDUCTION: ("D", ":"), UCB: ("v", "--"),
          EXPECTED_IMPROVEMENT: ("P", ":")}


def acquisition(strategy: str, models: dict, candidates: np.ndarray, reference: np.ndarray, observed_total: np.ndarray,
                ucb_weight: float = 2.0) -> np.ndarray:
    """Score of every candidate configuration under an adaptive strategy; the candidate with the largest score is
    benchmarked next. models: service -> GP of its execution time, fitted to the configurations benchmarked so far.
    All services share the configuration, so the scores of the services are combined into one.
    - MAX_VARIANCE (uncertainty sampling): the sum over the services of the variance of the GP's mean at the candidate
      (its epistemic variance, without the scatter of single items). It benchmarks where the models know least.
    - VARIANCE_REDUCTION (active learning Cohn): how much benchmarking the candidate reduces the variance of the mean,
      summed over the reference configurations and the services: sum_r cov(r, x)^2 / var(y(x)). Unlike MAX_VARIANCE it
      counts what a benchmark tells about the rest of the space, so it avoids the corners. reference: configurations
      that represent the space.
    - UCB (upper confidence bound): the predicted total execution time of the chain plus ucb_weight standard
      deviations of that prediction. It benchmarks configurations that are slow or uncertain, where the queues are
      longest.
    - EXPECTED_IMPROVEMENT: the expected amount by which the total execution time at the candidate falls below the
      smallest total observed so far (observed_total), the acquisition of Bayesian optimization for finding the fastest
      configuration."""
    predictions = [model.predict(candidates, return_variance_components=True) for model in models.values()]
    mean_total = sum(mu for mu, _, _, _ in predictions)
    var_total = sum(var_epistemic for _, _, var_epistemic, _ in predictions)
    if strategy == MAX_VARIANCE:
        return var_total
    if strategy == UCB:
        return mean_total + ucb_weight * np.sqrt(var_total)
    if strategy == EXPECTED_IMPROVEMENT:
        std = np.sqrt(np.maximum(var_total, 1e-18))
        z = (observed_total.min() - mean_total) / std
        return (observed_total.min() - mean_total) * norm.cdf(z) + std * norm.pdf(z)
    if strategy == VARIANCE_REDUCTION:
        score = np.zeros(len(candidates))
        for model in models.values():
            both = model.x_scaler.transform(np.vstack([reference, candidates]))
            _, covariance = model.gp.predict(both, return_cov=True)   # of one observation: its diagonal contains the noise of an item
            between = covariance[:len(reference), len(reference):]    # covariance of the mean at a reference point and a candidate
            score += (between ** 2).sum(axis=0) / np.diag(covariance)[len(reference):]
        return score
    raise ValueError(f"unknown adaptive strategy {strategy!r}")


def explore(strategy: str, services: list, service_seed: int, n_total: int, n_initial: int, seed, n_candidates: int = 500,
            n_reference: int = 200) -> tuple:
    """Benchmarks n_total configurations of the synthetic services one after the other under a sampling strategy.
    Returns the configurations (n_total, features) in the order of the benchmarks and service -> the measured
    execution time of every benchmark; the first n configurations are the training data after n benchmarks.
    - RANDOM: every configuration is drawn uniformly at random.
    - SOBOL: the first n_total points of a scrambled Sobol sequence, which fills the space more evenly than random
      draws; it is not adaptive.
    - adaptive strategies (acquisition): the first n_initial configurations are drawn at random; every further one is
      the candidate with the largest score among n_candidates random configurations that are not benchmarked yet. The
      GPs are fitted anew after every benchmark, with one restart of the hyperparameter search to keep the loop fast.
      The GPs do not receive the number of workers (model_inputs), but a strategy chooses whole configurations.
    seed: every strategy that is called with the same seed gets the same initial configurations and candidates."""
    rng = np.random.default_rng(seed)
    initial, candidates, reference = (sample_configurations(n, rng) for n in (n_initial, n_candidates, n_reference))
    if strategy == RANDOM:
        x = np.vstack([initial, candidates[:n_total - n_initial]])
        return x, synthetic_execution_times(x, services, rng, service_seed)
    if strategy == SOBOL:
        low, high = np.array(CONFIGURATION_RANGES).T
        x = whole_workers(low + (high - low) * qmc.Sobol(d=len(low), scramble=True, seed=rng).random(n_total))
        return x, synthetic_execution_times(x, services, rng, service_seed)
    x, y = initial, synthetic_execution_times(initial, services, rng, service_seed)
    available = np.ones(n_candidates, dtype=bool)
    while len(x) < n_total:
        models = train_gp_models(model_inputs(x, WORKERS_COLUMN), y, services, n_restarts_optimizer=1)
        scores = acquisition(strategy, models, model_inputs(candidates, WORKERS_COLUMN), model_inputs(reference, WORKERS_COLUMN), sum(y.values()))
        chosen = np.flatnonzero(available)[np.argmax(scores[available])]
        available[chosen] = False
        measured = synthetic_execution_times(candidates[[chosen]], services, rng, service_seed)
        x, y = np.vstack([x, candidates[[chosen]]]), {s: np.append(y[s], measured[s]) for s in services}
    return x, y


def exploration_results(data, x: np.ndarray, y: dict, sizes: list, service_seed: int, alpha: float, confidence: float, model_seed: int) -> dict:
    """What the first n benchmarked configurations give, for every n in sizes: n -> (end-to-end latency of Pegasus per
    node at the held-out configurations of `data` in s, relative error of the predicted mean execution time). The
    models are those of the other experiments (fit_services), fitted to the first n configurations. The error is the
    median over the held-out configurations and the services of |predicted mean / true mean - 1|."""
    true_mean, _ = synthetic_execution_moments(data.path, data.x_test, service_seed)
    results = {}
    for size in sizes:
        subset = replace(data, x_train=x[:size], y_train={s: values[:size] for s, values in y.items()})
        models = fit_services(subset, subset.x_train, subset.y_train, model_seed=model_seed, confidence=confidence, with_dnn=False)
        predicted = np.column_stack([models["gp"][s].predict(model_inputs(data.x_test, data.workers_column)) for s in data.path])
        results[size] = (pegasus_bound(subset, models, alpha), float(np.median(np.abs(predicted / true_mean - 1))))
    return results


def exploration_figure(sizes: list, statistics: dict, errors: dict, title: str, level_label: str,
                       xlabel: str = "benchmarked configurations (log scale)", name: str = None) -> None:
    """Two separate figures over the number of benchmarked configurations, one line per sampling strategy (median over
    the seeds, band from the smallest to the largest seed, see sweep_line), in the style of the paper figures
    (figures.py). First: tightness of the bound that is trained on the configurations of the strategy, with a filled
    marker where the bound holds and the red zone below 1x. Second: relative error of the predicted mean execution
    time. statistics: from sweep_statistics, with the strategies in place of the methods; errors: strategy -> (sizes,
    seeds); xlabel: what the sizes count. The title is printed above the figures; name: file name in figures/tsc, saved
    as <name>_tightness and <name>_error, or None."""
    from .figures import math_label, paper_figure, tightness_label, tightness_tick
    print(title)

    def axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle=":", alpha=0.6)

    with paper_style():
        fig, ax = paper_figure()
        ax.axhspan(TIGHTNESS_RANGE[0], 1.0, color=FAILURE_COLOR, linewidth=0)
        ax.axhline(1.0, color="black", linewidth=1)
        segments = []
        for strategy in statistics:
            sweep_line(ax, sizes, statistics[strategy]["tightness"], strategy, held=statistics[strategy]["holds"])
            segments += sweep_segments(sizes, statistics[strategy]["tightness"])
        axis(ax)
        ax.text(0.99, 0.03, f"below the measured {math_label(level_label)}: no bound", transform=ax.transAxes, ha="right", va="bottom", color=FAILURE_TEXT_COLOR)
        ax.set_yscale("log")
        ax.set_ylim(*TIGHTNESS_RANGE)
        ax.set_yticks(TIGHTNESS_TICKS, [tightness_tick(tick) for tick in TIGHTNESS_TICKS])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel(tightness_label(level_label))
        handles, labels = ax.get_legend_handles_labels()
        marker_handles, marker_labels = holds_handles()
        legend_inside(ax, handles + marker_handles, labels + marker_labels, segments, max_doublings=0, ncol=2, fontsize=10, framealpha=0.95,
                      columnspacing=1.0, handletextpad=0.4)
        if name is not None:
            save_figure(fig, f"{name}_tightness")
        plt.show()

        fig, ax = paper_figure()
        segments = []
        for strategy in statistics:
            sweep_line(ax, sizes, 100 * errors[strategy], strategy)
            segments += sweep_segments(sizes, 100 * errors[strategy])
        axis(ax)
        ax.set_ylabel("error of the predicted mean (%)")   # of the execution time; a longer label does not fit the height
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        legend_inside(ax, handles, labels, segments, max_doublings=0, ncol=2, fontsize=10, framealpha=0.95, columnspacing=1.0, handletextpad=0.4)
        if name is not None:
            save_figure(fig, f"{name}_error")
        plt.show()


# ------------------------------------------------------------------------------
# Real data: selecting among the measured runs of a service
# ------------------------------------------------------------------------------
FARTHEST = "Farthest point"
POOL_STRATEGIES = (RANDOM, FARTHEST, MAX_VARIANCE, VARIANCE_REDUCTION, UCB, EXPECTED_IMPROVEMENT)
COLORS[FARTHEST], STYLES[FARTHEST] = "black", ("X", "--")


def explore_pool(strategy: str, x_pool: np.ndarray, y_pool: np.ndarray, n_total: int, n_initial: int, seed, n_reference: int = 200) -> np.ndarray:
    """Chooses n_total of the measured runs of one real service one after the other under a sampling strategy, and
    returns their indices in the pool in the order of the choices. x_pool: (runs, features) the inputs of the model at
    the configurations of all runs that may train the service (without the number of workers, model_inputs); y_pool: the mean processing time (s) of every run, which a strategy only sees for
    the runs it has chosen. A run is one benchmark; several runs of the pool can have the same configuration.
    - RANDOM: runs drawn uniformly at random.
    - FARTHEST: after n_initial random runs, always the run whose configuration is farthest (in standardized knobs)
      from the configurations chosen so far. It spreads the runs over the space without looking at a model, the
      counterpart of the Sobol sequence for a fixed set of runs.
    - adaptive strategies: after n_initial random runs, the run with the largest score of `acquisition` on a GP of the
      logarithm of the run means (the mean model of the real data, with one restart of the hyperparameter search).
      Scores and improvements are therefore on the logarithm of the mean processing time. The reference
      configurations of VARIANCE_REDUCTION are n_reference random runs of the pool.
    seed: every strategy that is called with the same seed starts from the same initial runs."""
    rng = np.random.default_rng(seed)
    chosen = list(rng.choice(len(x_pool), size=n_initial, replace=False))
    reference = x_pool[rng.choice(len(x_pool), size=min(n_reference, len(x_pool)), replace=False)]
    if strategy == RANDOM:
        rest = np.setdiff1d(np.arange(len(x_pool)), chosen)
        return np.array(chosen + list(rng.choice(rest, size=n_total - n_initial, replace=False)))
    scaled = (x_pool - x_pool.mean(axis=0)) / np.where(x_pool.std(axis=0) > 0, x_pool.std(axis=0), 1.0)
    while len(chosen) < n_total:
        if strategy == FARTHEST:
            scores = np.min(np.linalg.norm(scaled[:, None, :] - scaled[None, chosen, :], axis=2), axis=1)
        else:
            log_means = np.log(y_pool[chosen])
            model = train_gp_models(x_pool[chosen], {0: log_means}, [0], MIN_LENGTH_SCALE, n_restarts_optimizer=1)
            scores = acquisition(strategy, model, x_pool, reference, log_means)
        scores = np.array(scores, dtype=float)
        scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
    return np.array(chosen)


def real_exploration_results(experiment: str, data, strategy: str, sizes: list, n_initial: int, seed, alpha: float, confidence: float,
                             model_seed: int) -> dict:
    """The counterpart of explore and exploration_results for the real chain `experiment`: every service chooses its
    training runs one after the other from its own pool (training_pool in real_data.py: runs of the service alone or in
    pipelines of another topology) under the sampling strategy, and after n choices per service the models of the
    other experiments (fit_services) are fitted to the chosen runs, for every n in sizes. data: the chain data of the
    experiment (real_chain_data), of which the held-out runs and the ground truth are used. Returns n -> (end-to-end
    latency of Pegasus Node at the held-out runs in s, relative error of the predicted mean execution time: the
    median over the held-out runs and the services of |predicted mean / true mean - 1|)."""
    chain = run_table(experiment)
    path, service_of, pipeline = chain_path(experiment)
    runs = list(chain["run_id"].unique())
    true_mean, _ = chain_execution_moments(experiment)
    pools, x_pool, x_test, order = {}, {}, {}, {}
    for k, node in enumerate(path):
        pools[node] = training_pool(service_of[node], pipeline)
        knobs = [column for column in pools[node].columns if column not in POOL_COLUMNS]
        x_pool[node], x_test[node] = encode_configurations(pools[node][knobs], node_configurations(chain, node).loc[runs])
        order[node] = explore_pool(strategy, model_inputs(x_pool[node], data.workers_column[node]), pools[node]["mean_s"].to_numpy(), max(sizes), n_initial, seed=[*np.atleast_1d(seed), k])
    results = {}
    for size in sizes:
        take = {node: order[node][:size] for node in path}
        subset = replace(data, x_test=x_test, x_train={node: x_pool[node][take[node]] for node in path},
                         y_train={node: pools[node]["mean_s"].to_numpy()[take[node]] for node in path},
                         y_train_spread={node: pools[node]["spread_s"].to_numpy()[take[node]] for node in path})
        models = fit_services(subset, subset.x_train, subset.y_train, model_seed=model_seed, confidence=confidence, with_dnn=False)
        predicted = np.column_stack([models["gp"][node].predict(model_inputs(x_test[node], data.workers_column[node])) for node in path])
        results[size] = (pegasus_bound(subset, models, alpha), float(np.median(np.abs(predicted / true_mean - 1))))
    return results
