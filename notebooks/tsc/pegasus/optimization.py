"""Experiment C.2 of `2_evaluation.ipynb`: choosing the configuration of a chain with a latency model, among the runs
of the chain that exist.

The *objective* is the SLO fulfillment of a configuration, as in the multidimensional scaling paper (Sedlak et al.,
IEEE TSC): an SLO relates a metric m to a target t and is fulfilled by min(1, m / t) if a larger metric is better
(quality) and by min(1, t / m) if a smaller one is better (latency); an SLO cannot be overfulfilled. The objective is
the weighted mean of the fulfillment of one end-to-end latency SLO and of the quality SLOs of the services.

An *optimizer* evaluates one existing run of the chain after the other. To choose the next run it asks a latency
*method* (a bound or a baseline of B) for the latency of configurations, which gives their predicted objective. When a
run is evaluated, its measured latency becomes known, and what the run measured about every service is added to the
training data of the method, whose models are fitted anew before the next choice.
Units: latencies in seconds.
"""
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import METHOD_COLORS, PROPOSED, PROPOSED_SNC, compare_methods, fit_services, sweep_line

BEST_OF_RUNS, GRADIENT = "best of the runs", "gradient steps"
RANDOM_SEARCH = "Random search"
COLORS = {RANDOM_SEARCH: "black"}                              # to be added to METHOD_COLORS of evaluation.py by the notebook
EXECUTION_TIME_METHODS = (PROPOSED, PROPOSED_SNC, "GP Sum", "Allen-Cunneen")   # methods that use the execution-time models of fit_services


@dataclass
class ChainProblem:
    """One optimization problem: the existing runs of a chain, the SLOs, and what an evaluated run reveals."""
    data: object                 # ChainData of the chain at one arrival rate for all runs; its held-out configurations are the existing runs
    quality_slos: list           # (service or None for a configuration shared by all services, column of its configuration, target)
    latency_target: float        # target of the end-to-end latency SLO (s)
    latency_weight: float        # weight of the latency SLO; the quality SLOs share the rest equally
    observe: object              # function (run index, random generator) -> (service -> execution-time target, service -> spread or None)
    alpha: float                 # violation probability of the latency quantile
    confidence: float
    n_calibration: int
    model_seed: int

    def __post_init__(self):
        x = self.data.x_test
        self.shared = not isinstance(x, dict)
        self.nodes = [None] if self.shared else list(dict.fromkeys(self.data.path))
        parts = [x] if self.shared else [x[node] for node in self.nodes]
        edges = np.cumsum([0] + [part.shape[1] for part in parts])
        self.columns = {node: slice(edges[k], edges[k + 1]) for k, node in enumerate(self.nodes)}
        self.z = np.column_stack(parts)                        # (runs, features of all services): the configuration of the whole chain
        self.scale = np.where(self.z.std(axis=0) > 0, self.z.std(axis=0), 1.0)
        self.varied = np.flatnonzero(self.z.std(axis=0) > 0)   # knobs that differ between the existing runs

    def split(self, z: np.ndarray):
        """Configurations of the whole chain as the configurations of its services (the form of ChainData.x_test)."""
        return z if self.shared else {node: z[:, self.columns[node]] for node in self.nodes}

    def fulfillment(self, latency: np.ndarray, z: np.ndarray) -> np.ndarray:
        """SLO fulfillment of the configurations z of the whole chain, given their end-to-end latency (s; inf for a
        configuration without a finite latency, whose latency SLO is then not fulfilled at all)."""
        latency_part = np.minimum(1.0, self.latency_target / np.asarray(latency, dtype=float))
        quality = [np.minimum(1.0, z[:, self.columns[node]][:, column] / target) for node, column, target in self.quality_slos]
        return self.latency_weight * latency_part + (1 - self.latency_weight) * np.mean(quality, axis=0)

    def true_fulfillment(self) -> np.ndarray:
        """SLO fulfillment of every existing run with its measured latency quantile."""
        return self.fulfillment(self.data.measured, self.z)


def synthetic_observation(data, service_seed: int):
    """What an evaluated run of a synthetic chain reveals about its services (ChainProblem.observe): one measured
    execution time of every service at the configuration of the run, as in the training data of the synthetic system."""
    from .data import synthetic_execution_times
    return lambda run, rng: ({s: values[0] for s, values in synthetic_execution_times(data.x_test[[run]], data.path, rng, service_seed).items()}, None)


def real_observation(experiment: str, data):
    """What an evaluated run of a real chain reveals about its services (ChainProblem.observe): the mean and the spread
    of the processing time that the run measured for every service, as in the training data of the real services."""
    from .real_data import chain_run_statistics
    mean, spread = chain_run_statistics(experiment)
    return lambda run, rng: ({s: mean[run, k] for k, s in enumerate(data.path)}, {s: spread[run, k] for k, s in enumerate(data.path)})


def real_quality_slos(experiment: str, targets: dict) -> list:
    """Quality SLOs of a real chain in the form of ChainProblem.quality_slos, from (service, knob) -> target."""
    from .real_data import feature_names
    names = feature_names(experiment)
    return [(service, names[service].index(knob), target) for (service, knob), target in targets.items()]


def _training_state(data) -> dict:
    return dict(x=data.x_train, y=dict(data.y_train), spread=None if data.y_train_spread is None else dict(data.y_train_spread),
                monitored=dict(data.monitored_latencies), x_load=None if data.x_train_load is None else dict(data.x_train_load))


def _add_run(problem: ChainProblem, state: dict, run: int, rng: np.random.Generator) -> dict:
    """The training data after one more run of the chain was evaluated. Every service gets one more training
    configuration: its configuration in that run with the execution-time target that the run measured (problem.observe),
    and, for the baselines monitored under load, the latencies that the service had in that run."""
    data, row = problem.data, [run]
    y_new, spread_new = problem.observe(run, rng)
    services = list(state["y"])
    stack = lambda x, x_run: (np.vstack([x, x_run[row]]) if problem.shared else {s: np.vstack([x[s], x_run[s][row]]) for s in services})
    width = next(iter(state["monitored"].values())).shape[1]
    latencies = {s: (data.node_latencies[run, :width, data.path.index(s)] if np.isfinite(data.measured[run]) else np.full(width, np.nan)) for s in services}
    return dict(x=stack(state["x"], data.x_test), y={s: np.append(state["y"][s], y_new[s]) for s in services},
                spread=None if state["spread"] is None else {s: np.append(state["spread"][s], spread_new[s]) for s in services},
                monitored={s: np.vstack([state["monitored"][s], latencies[s]]) for s in services},
                x_load=None if state["x_load"] is None else {s: np.vstack([state["x_load"][s], data.x_test_load[s][row]]) for s in services})


def predicted_latency(problem: ChainProblem, state: dict, method: str, z: np.ndarray) -> np.ndarray:
    """End-to-end latency (s) that `method` gives for the configurations z of the whole chain, with models fitted to
    the training data `state`."""
    data, x_test = problem.data, problem.split(z)
    x_test_load = None if state["x_load"] is None else {
        s: np.column_stack([x_test[s], np.full(len(z), np.log10(data.workload["rate"]))]) for s in x_test}
    probe = replace(data, x_test=x_test, x_test_load=x_test_load, node_latencies=None, x_train=state["x"], y_train=state["y"],
                    y_train_spread=state["spread"], monitored_latencies=state["monitored"], x_train_load=state["x_load"])
    if method in EXECUTION_TIME_METHODS:
        models = fit_services(probe, probe.x_train, probe.y_train, model_seed=problem.model_seed, confidence=problem.confidence, with_dnn=False)
    else:   # the baselines monitored under load train their own models inside compare_methods
        models = dict(seconds=dict(gp=0.0, dnn=0.0, beliefs=0.0))
    methods, _ = compare_methods(probe, models, probe.x_train, probe.monitored_latencies, problem.alpha, problem.confidence, problem.n_calibration,
                                 model_seed=problem.model_seed, leave_out=tuple(name for name in METHOD_COLORS if name != method),
                                 x_train_load=probe.x_train_load)
    return methods[method][1]


def optimize(problem: ChainProblem, method, optimizer: str, start: int, budget: int, rng: np.random.Generator, step: float = 1.0,
             difference: float = 0.5) -> tuple:
    """Evaluates `budget` existing runs of the chain one after the other. Returns their indices in that order, and for
    every one of them the end-to-end latency (s) that the method gave for it when it was chosen (nan for the first run
    and for RANDOM_SEARCH, which are chosen without a model). The first run is `start`. Before every further choice, the models of `method` are fitted to the training data including
    what the evaluated runs measured.
    - method None (RANDOM_SEARCH): the next run is drawn at random among the runs that are not evaluated yet.
    - optimizer BEST_OF_RUNS: the next run is the one with the largest predicted SLO fulfillment among the runs that are
      not evaluated yet (the random optimizer of Fluxion, with the existing runs as its candidate settings).
    - optimizer GRADIENT: the gradient of the predicted SLO fulfillment with respect to the configuration is computed
      at the run evaluated last, by central differences of `difference` standard deviations in every knob that differs
      between the runs. The next run is the not yet evaluated one that is closest to the point `step` standard
      deviations along the gradient (Euclidean distance, every knob in units of its standard deviation over the
      existing runs). Without a gradient (e.g. no finite latency around the current run), it is the closest run."""
    z, scale = problem.z, problem.scale
    visited, promised, state = [start], [np.nan], _add_run(problem, _training_state(problem.data), start, rng)
    while len(visited) < budget:
        remaining = np.setdiff1d(np.arange(len(z)), visited)
        if method is None:
            chosen, latency = int(rng.choice(remaining)), np.nan
        elif optimizer == BEST_OF_RUNS:
            latencies = predicted_latency(problem, state, method, z[remaining])
            best = int(np.argmax(problem.fulfillment(latencies, z[remaining])))
            chosen, latency = int(remaining[best]), float(latencies[best])
        elif optimizer == GRADIENT:
            current = z[visited[-1]]
            offsets = np.zeros((len(problem.varied), z.shape[1]))
            offsets[np.arange(len(problem.varied)), problem.varied] = difference * scale[problem.varied]
            probes = np.vstack([current + offsets, current - offsets])
            # one fit of the models for the probes and for the runs, of which the chosen one needs its predicted latency
            latencies = predicted_latency(problem, state, method, np.vstack([probes, z[remaining]]))
            predicted = problem.fulfillment(latencies[:len(probes)], probes)
            gradient = np.zeros(z.shape[1])   # per standard deviation of every knob
            gradient[problem.varied] = (predicted[:len(problem.varied)] - predicted[len(problem.varied):]) / (2 * difference)
            length = np.linalg.norm(gradient)
            target = current / scale + (step * gradient / length if length > 0 and np.isfinite(length) else 0.0)
            closest = int(np.argmin(np.linalg.norm(z[remaining] / scale - target, axis=1)))
            chosen, latency = int(remaining[closest]), float(latencies[len(probes) + closest])
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        visited.append(chosen)
        promised.append(latency)
        state = _add_run(problem, state, chosen, rng)
    return visited, promised


def best_found(problem: ChainProblem, visited: list) -> np.ndarray:
    """After every evaluation: the largest true SLO fulfillment among the runs evaluated so far, as a share of the
    largest true SLO fulfillment of any existing run (1 means that the best existing run was found)."""
    true = problem.true_fulfillment()
    return np.maximum.accumulate(true[visited]) / true.max()


def latency_promises(problem: ChainProblem, visited: list, promised: list) -> dict:
    """How the runs that an optimizer chose with a method did on the latency SLO; the first run, which is not chosen
    with the method, is left out. Returns counts: "chosen" (runs chosen with the method), "violated" (of these, the
    runs whose measured latency quantile exceeds the latency target), "promised" (runs for which the method gave a
    latency within the target when it chose them, i.e. it promised that the latency SLO is met) and "broken"
    (promised runs whose measured latency quantile exceeds the target). RANDOM_SEARCH gives no latency and promises
    nothing."""
    chosen, latency = np.array(visited[1:]), np.array(promised[1:], dtype=float)
    violated = problem.data.measured[chosen] > problem.latency_target
    kept_promise = latency <= problem.latency_target   # False where no latency was given
    return dict(chosen=len(chosen), violated=int(violated.sum()), promised=int(kept_promise.sum()), broken=int((kept_promise & violated).sum()))


def optimization_figure(found: dict, title: str) -> None:
    """One panel per optimizer over the number of evaluated runs, one line per method (median over the seeds, band from
    the smallest to the largest seed, see sweep_line): the best true SLO fulfillment found so far as a share of the
    best existing run. found: optimizer -> method -> (evaluations, seeds), from best_found; RANDOM_SEARCH is drawn in
    every panel."""
    fig, axes = plt.subplots(1, len(found), figsize=(7.5 * len(found), 4.8), sharey=True)
    for ax, (optimizer, results) in zip(np.atleast_1d(axes), found.items()):
        steps = np.arange(1, len(next(iter(results.values()))) + 1)
        for name, values in results.items():
            sweep_line(ax, steps, 100 * values, name)
        ax.axhline(100, color="black", linewidth=1)
        ax.set_xticks(steps)
        ax.set_xlabel("evaluated runs of the chain")
        ax.set_title(f"optimizer: {optimizer}")
        ax.grid(True, linestyle=":", alpha=0.6)
    np.atleast_1d(axes)[0].set_ylabel("best SLO fulfillment found (% of the best existing run)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
