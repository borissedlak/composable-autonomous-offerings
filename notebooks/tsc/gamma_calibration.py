"""Calibration of the confidence gamma of the pessimistic mean of Pegasus (Eq. 7), on the real data, for the Pegasus of
the paper: the load-aware GPs of load_aware_b4.py (inputs: the quality knobs, the number of workers and log10 of the
arrival rate; trained on the worker benchmarks benchmark/w1 to w4, no run of any chain).

What gamma claims (RunStatisticsModel.mean, run_statistics_belief in pegasus/service_gp.py): the center GP of a service
predicts the natural logarithm of the mean execution time of a new run at input x as N(mu(x), sigma(x)^2), and the
pessimistic mean exp(mu + z_gamma * sigma) lies above the measured mean y_bar of a new run with probability gamma. A
run is therefore covered at level gamma if and only if
    u = Phi((ln y_bar - mu) / sigma) <= gamma,
so the share of runs with u <= gamma is the observed coverage at every gamma at once.

The data and the GPs come from the pipeline of B.1 on all chains (2a_base_case.ipynb): load_chain_data inside
extended_training with training_campaign=EXTENDED, the view load_aware(data), and fit_services on it. Two tests:
- chains: every held-out run of the 15 chains of the paper (DEFAULT_CHAINS), one u per node, with the measured mean
  execution time of the node in the run (processing mean plus handling overhead, as training_pool defines it);
- benchmark: the worker-benchmark runs that were not drawn for training (only for n_train below the pool size).
n_train: training runs per service; N_TRAIN of load_aware_b4.py is the setting of the paper.

The script lives outside pegasus/ on purpose: a new file there would change the cache key of the evaluation notebooks.
Run from notebooks/tsc with the project venv: python gamma_calibration.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning

from load_aware_b4 import EXTENDED, N_TRAIN, extended_pool, extended_training
from pegasus.data import load_aware, load_chain_data
from pegasus.evaluation import fit_services
from pegasus.evaluation_setup import DEFAULT_CHAINS, alpha, configure, confidence, coverage_check, seed, seeds
from pegasus.real_data import (NOT_CONFIGURATION, POOL_COLUMNS, arrival_rates, chain_path, combined_runs, encode_configurations,
                               handling_overhead, run_table, service_rng, unthrottled_pool)

N_TRAINS = (25, 50, 100, None)   # training runs per service; None: every run of the worker benchmarks; the paper: N_TRAIN
OUT = Path(__file__).resolve().parents[2] / "statics" / "gamma_calibration"


def measured_node_means(chain: str, path: list) -> np.ndarray:
    """(held-out runs, nodes): the measured mean execution time (s) of every node in every held-out run of the combined
    chain, in the order of real_chain_data (combined_runs): the processing mean plus the handling overhead."""
    rows = []
    for experiment, run in combined_runs(chain):
        statistics = run_table(experiment)
        statistics = statistics[statistics["run_id"] == run].set_index("node")
        rows.append([statistics.loc[node, "processing_mean_ms"] / 1e3 + handling_overhead(experiment, run, node) for node in path])
    return np.array(rows)


def training_draw(service: str, s: int, n_train) -> tuple:
    """(training runs, the other runs) of the service's worker-benchmark pool under seed=[seed, s], as real_chain_data
    draws them (service_rng from the training sequence)."""
    with extended_training():
        pool = unthrottled_pool(extended_pool(service))
    rng = service_rng(np.random.SeedSequence([seed, s]).spawn(3)[0], service)
    index = rng.permutation(len(pool)) if n_train is None else np.sort(rng.choice(len(pool), size=n_train, replace=False))
    return pool.iloc[index], pool.drop(index=pool.index[index])


def log_rates(runs: pd.DataFrame) -> np.ndarray:
    """log10 of the arrival rate that the benchmark set for every run: the load input of the load-aware GPs."""
    return np.log10([arrival_rates(run.experiment)[run.run_id] for run in runs.itertuples()])


def pit(model, x: np.ndarray, measured_mean: np.ndarray) -> tuple:
    """Per input row, under the center GP of a RunStatisticsModel: u, the ratio y_bar / exp(mu) of the measured to the
    predicted mean, and sigma."""
    mu, sd = model.center.predict(x, return_std=True)
    return norm.cdf((np.log(measured_mean) - mu) / sd), measured_mean / np.exp(mu), sd


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    chains = configure("real", DEFAULT_CHAINS)["chain_names"]
    chain_rows, benchmark_rows = [], []
    for n_train in N_TRAINS:
        for s in seeds:
            done = set()
            for chain in chains:
                with extended_training():
                    data = load_chain_data("real", experiment=chain, level=1 - alpha, coverage_check=coverage_check, seed=[seed, s],
                                           n_train=n_train, training_campaign=EXTENDED)
                aware = load_aware(data)
                models = fit_services(aware, aware.x_train, data.y_train, model_seed=seed + s, confidence=confidence, with_dnn=False)
                path, service_of = chain_path(chain)[:2]
                y, runs = measured_node_means(chain, path), combined_runs(chain)
                for k, node in enumerate(path):
                    column = data.workers_column[node]
                    workers = data.x_test[node][:, column] if column is not None else np.ones(len(runs))
                    u, ratio, sd = pit(models["gp"][node], aware.x_test[node], y[:, k])
                    chain_rows += [dict(n_train=n_train or "all", seed=s, chain=chain, experiment=e, run=r, K=len(path), position=k,
                                        service=service_of[node], workers=w, u=a, ratio=b, sigma=c)
                                   for (e, r), w, a, b, c in zip(runs, workers, u, ratio, sd)]
                    # the held-out benchmark runs of the service, once per service and seed
                    if n_train is None or service_of[node] in done:
                        continue
                    done.add(service_of[node])
                    chosen, held = training_draw(service_of[node], s, n_train)
                    assert np.allclose(np.sort(chosen["mean_s"]), np.sort(data.y_train[node])), "the draw differs from real_chain_data"
                    knobs = [c for c in chosen.columns if c not in POOL_COLUMNS and c not in NOT_CONFIGURATION]
                    x_train, x_held = encode_configurations(chosen[knobs], held[knobs])
                    assert np.allclose(np.column_stack([x_train, log_rates(chosen)]), aware.x_train[node]), "the encoding differs from the pipeline"
                    u, ratio, sd = pit(models["gp"][node], np.column_stack([x_held, log_rates(held)]), held["mean_s"].to_numpy())
                    benchmark_rows += [dict(n_train=n_train, seed=s, service=service_of[node], experiment=e, run=r, u=a, ratio=b, sigma=c)
                                       for e, r, a, b, c in zip(held["experiment"], held["run_id"], u, ratio, sd)]
            print(f"n_train {n_train or 'all'}, seed {s} done", flush=True)
    pd.DataFrame(chain_rows).to_csv(OUT / "load_aware_chains.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(OUT / "load_aware_benchmark.csv", index=False)
    print("wrote", OUT, flush=True)


def coverage_curves(u: pd.Series, groups: list, gammas: np.ndarray) -> np.ndarray:
    """(groups, gammas): the share of runs with u <= gamma, per group (e.g. per seed)."""
    return np.array([[(u[g] <= gamma).mean() for gamma in gammas] for g in groups])


def calibration_figure(name: str = "gamma_calibration", n_train: int = N_TRAIN) -> dict:
    """Observed vs. target confidence of the pessimistic mean (Eq. 7) of the load-aware GPs, from load_aware_benchmark.csv
    and load_aware_chains.csv (main). Lines: the median over the seeds; bands: their range.
    - benchmark runs, per service: the worker-benchmark runs not drawn for training, with n_train training runs per service;
    - chain runs, per service: every node of every held-out chain run, with n_train training runs per service;
    n_train: one of N_TRAINS; the default is the setting of the paper (N_TRAIN).
    - chain runs, all K services: the share of chain runs in which every node is covered at gamma, and next to it the
      share that the model implies if the errors of the K services are independent, the mean of gamma^K over the runs.
    Returns the curves (gammas and name -> (seeds, gammas)) for the description of the figure."""
    from pegasus.evaluation import METHOD_COLORS, PROPOSED
    from pegasus.figures import BAND_ALPHA, FAILURE_COLOR, FAILURE_TEXT_COLOR, paper_figure, paper_style, save_figure
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    benchmark = pd.read_csv(OUT / "load_aware_benchmark.csv")
    benchmark = benchmark[benchmark["n_train"] == n_train]
    chains = pd.read_csv(OUT / "load_aware_chains.csv", dtype={"n_train": str})
    chains = chains[chains["n_train"] == str(n_train or "all")]
    gammas = np.round(np.arange(0.5, 0.995, 0.01), 2)
    seeds_of = lambda table: sorted(table["seed"].unique())
    per_run = chains.groupby(["seed", "experiment", "run"]).agg(u_max=("u", "max"), K=("u", "size")).reset_index()
    curves = {
        "benchmark": coverage_curves(benchmark.set_index("seed")["u"], seeds_of(benchmark), gammas),
        "chain node": coverage_curves(chains.set_index("seed")["u"], seeds_of(chains), gammas),
        "chain run": coverage_curves(per_run.set_index("seed")["u_max"], seeds_of(per_run), gammas),
        "chain run claimed": np.array([[np.mean(gamma ** per_run.loc[per_run["seed"] == s, "K"]) for gamma in gammas] for s in seeds_of(per_run)]),
    }
    green = METHOD_COLORS[PROPOSED]
    # label, color, marker, line style, line width
    styles = {"benchmark": ("benchmark, per service", "0.35", "s", "--", 1.8),
              "chain node": ("chain, per service", green, "o", "-", 2.5),
              "chain run": ("chain, all $K$", green, "D", ":", 2.5),
              "chain run claimed": (r"model: all $K$ ($\gamma^K$)", green, "", (0, (5, 2)), 1.5)}
    marked = np.isin(gammas, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    with paper_style():
        fig, ax = paper_figure()
        ax.fill_between([0.5, 1], [0.5, 1], 0, color=FAILURE_COLOR, linewidth=0, zorder=0)
        ax.text(0.99, 0.03, "below the target: γ overstated", ha="right", va="bottom", color=FAILURE_TEXT_COLOR)
        ax.plot([0.5, 1], [0.5, 1], color="black", linewidth=1, zorder=1)
        ax.text(0.905, 0.925, "target: observed = γ", ha="right", va="bottom", fontsize=10)
        handles, labels = [], []
        for key, (label, color, marker, linestyle, width) in styles.items():
            values = curves[key]
            middle = np.median(values, axis=0)
            if len(values) > 1 and np.ptp(values, axis=0).max() > 0:
                ax.fill_between(gammas, values.min(axis=0), values.max(axis=0), color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(gammas, middle, color=color, linestyle=linestyle, linewidth=width, marker=marker or None,
                    markevery=list(np.flatnonzero(marked)), markersize=6, markerfacecolor=color)
            handles.append(Line2D([], [], color=color, linestyle=linestyle, linewidth=width, marker=marker or None, markersize=6))
            labels.append(label)
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        ax.set_xlabel("target confidence γ")
        ax.set_ylabel("observed coverage")
        # the region above the diagonal at small gamma is empty: the legend goes there (a linear axis, so not legend_inside,
        # which makes room by doubling a logarithmic one)
        ax.legend(handles, labels, loc="upper left", fontsize=9, handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
        save_figure(fig, name)
        plt.show()
    return dict(gammas=gammas, **curves)


if __name__ == "__main__":
    import sys
    calibration_figure() if sys.argv[1:] == ["figure"] else main()
