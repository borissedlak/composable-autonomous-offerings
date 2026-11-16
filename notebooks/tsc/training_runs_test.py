"""Short test: how coverage and tightness of every method change with the number of training runs per service
(25, 50 or every run of the worker benchmarks), for one seed, on the real data of B.1 (all chains), B.3 (trees),
B.4a/b (load and arrival variability) and C.1 (fine-tuning, Pegasus only; its baselines train on end-to-end samples).

Everything else is the setup of the notebooks (same information for every method, same_information_methods in pegasus/evaluation.py). DNN Sum is left
out (B.1 only, and slow). B.3 trains on its own VM benchmarks, where "all" is not an option of real_tree_data, so B.3
runs 25 and 50 only. Results go to statics/training_runs_test/ (one joblib per experiment and n_train, then a CSV).
Outside pegasus/ on purpose: a file there would change the cache key of the evaluation notebooks.
Run from notebooks/tsc: python training_runs_test.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from pegasus.data import load_aware, load_chain_data
from pegasus.evaluation import PROPOSED, compare_tree_methods, fit_services, same_information_methods
from pegasus.evaluation_setup import DEFAULT_CHAINS, MAX_WORKERS, alpha, configure, confidence, coverage_check, n_calibration, only_base_case, seed, seeds
from pegasus.finetuning import finetuned_bound, split_samples
from pegasus.real_data import WORKER_TRAINING, real_tree_data

S = seeds[0]                      # the one seed of the test
N_TRAINS = (25, 50, None)         # training runs per service; None: every run of the worker benchmarks (the current setting)
OUT = Path(__file__).resolve().parents[2] / "statics" / "training_runs_test"
C1_CHAIN = f"b1-vision-1-single+b1u-max{MAX_WORKERS}"   # the chain of C.1 (2e_finetuning.ipynb)
C1_SIZES = [0, 5, 10, 20, 30, 40, 50]


def label(n_train) -> str:
    return "all" if n_train is None else str(n_train)


def run(name: str, n_train, compute):
    """compute() -> list of dict(setting, method, latency, measured, measured_low), stored per experiment and n_train."""
    path = OUT / f"{name}_n{label(n_train)}_seed{S}.joblib"
    if path.exists():
        return joblib.load(path)
    rows = compute()
    joblib.dump(rows, path)
    print(f"{name}, n_train {label(n_train)} done", flush=True)
    return rows


def rows_of(setting, methods: dict, measured, measured_low) -> list:
    return [dict(setting=setting, method=m, latency=np.asarray(v[1] if isinstance(v, tuple) else v),
                 measured=measured, measured_low=measured_low) for m, v in methods.items()]


def b1(options, n_train):
    rows = []
    for chain in options["chain_names"]:
        data = load_chain_data("real", experiment=chain, level=1 - alpha, coverage_check=coverage_check, seed=[seed, S],
                               n_train=n_train, training_campaign=WORKER_TRAINING)
        rows += rows_of(chain, same_information_methods(data, seed + S, alpha, confidence, n_calibration), data.measured, data.measured_low)
    return rows


def b4(options, n_train, sweep):
    rows = []
    for setting in options["loads"] if sweep == "load" else options["arrival_cv2s"]:
        data = options["b4_data"](S, sweep, setting, training_campaign=WORKER_TRAINING, n_train=n_train)
        rows += rows_of(setting, same_information_methods(data, seed + S, alpha, confidence, n_calibration), data.measured, data.measured_low)
    return rows


def b3(options, n_train):
    rows = []
    for (tree_set, h), experiment in sorted(options["tree_experiments"].items()):
        data = real_tree_data(experiment, level=1 - alpha, node_level=1 - alpha / len(data_path(experiment)), coverage_check=coverage_check,
                              seed=[seed, S], n_train=n_train)
        view = load_aware(data)
        models = fit_services(view, view.x_train, data.y_train, model_seed=seed + S, confidence=confidence, with_dnn=False)
        methods, _ = compare_tree_methods(data, models, alpha, confidence, n_calibration, model_seed=seed + S, leave_out=only_base_case,
                                          load_aware_models=True)
        rows += rows_of(h, methods, data.measured, data.measured_low)
    return rows


def data_path(experiment):
    from pegasus.real_data import tree_structure
    return tree_structure(experiment)[1]


def c1(b1_rows):
    """Pegasus of C.1: the B.1 bound of C1_CHAIN fine-tuned with end-to-end samples, as in 2e_finetuning.ipynb."""
    row = next(r for r in b1_rows if r["setting"] == C1_CHAIN and r["method"] == PROPOSED)
    samples, test = split_samples(len(row["measured"]), C1_SIZES[-1], np.random.default_rng([seed, S, 100]))
    return [dict(setting=size, method=PROPOSED, latency=finetuned_bound(row["latency"], row["measured"], samples[:size], test, miscoverage=1 - confidence),
                 measured=row["measured"][test], measured_low=row["measured_low"][test]) for size in C1_SIZES]


def summary(rows: list, experiment: str, n_train) -> list:
    out = []
    for r in rows:
        ok = np.isfinite(r["measured"])   # a run without a measured quantile does not count
        latency, measured, low = r["latency"][ok], r["measured"][ok], r["measured_low"][ok]
        out.append(dict(experiment=experiment, n_train=label(n_train), setting=r["setting"], method=r["method"],
                        coverage=float(np.mean(latency >= low)), tightness=float(np.median(latency / measured)),
                        finite=float(np.mean(np.isfinite(latency))), runs=int(ok.sum())))
    return out


def main(only=None):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    options = configure("real", DEFAULT_CHAINS)
    table = []
    for n_train in N_TRAINS:
        experiments = {"B.1": lambda: b1(options, n_train), "B.4a": lambda: b4(options, n_train, "load"),
                       "B.4b": lambda: b4(options, n_train, "arrivals")}
        if n_train is not None:
            experiments["B.3"] = lambda: b3(options, n_train)
        for name, compute in experiments.items():
            if only and name not in only:
                continue
            try:
                rows = run(name, n_train, compute)
            except AssertionError as error:   # e.g. a tree benchmark with fewer runs than n_train
                print(f"{name}, n_train {label(n_train)} skipped: {error}", flush=True)
                continue
            table += summary(rows, name, n_train)
            if name == "B.1":
                table += summary(c1(rows), "C.1", n_train)
    frame = pd.DataFrame(table)
    frame.to_csv(OUT / f"summary_seed{S}.csv", index=False)
    print("wrote", OUT / f"summary_seed{S}.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
