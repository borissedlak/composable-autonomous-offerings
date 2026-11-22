"""Overload detection of B.4a: do the methods predict an infinite latency exactly for the runs that really overloaded?

The load sweep of 2_B4_load.ipynb holds out only the runs that the testbed sustained at every load (b4_held_out in
pegasus/real_data.py), so there an infinite bound is always a false alarm and a missed overload cannot occur. This
module evaluates every run of one load instead, at the loads in `LOADS`, each on its own runs (as utilization_one.py):
every unthrottled run with at most b4_max_workers workers per service, valid or not. A run that the testbed marked
invalid (flag "detector:overloaded": the queue of the busiest service kept growing) has an infinite ground truth.

The methods are those of the sweep (same_information_methods in pegasus/evaluation.py) and receive the same inputs for
an overloaded run as for a sustained one: the knobs, the workers of every service and the arrival rate that the testbed
set for the run (rps_on_first_service, recorded for invalid runs too). The Oracle is left out, because its bound is
built from the latencies measured in the run, which have no steady state in an overloaded run.

To build the held-out runs with the loader of the sweep, run_table is replaced, for the evaluated experiments only, by
a version that keeps the invalid rows; the training data (other experiments) is read unchanged. The measured quantile
that the loader computes for an invalid run is replaced by inf.

Not hashed by the cache key of the evaluation; its own cache is statics/load_aware_b4/overload_detection.joblib, keyed
on the code of pegasus/ (code_fingerprint) and on this file.
"""
import hashlib
import warnings
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pegasus import evaluation_setup as setup
from pegasus import real_data
from pegasus.evaluation import same_information_methods
from pegasus.real_data import CAMPAIGN, WORKER_TRAINING, b4_campaigns, b4_experiment, chain_runs, experiment_folder, join_chain_data, real_chain_data

LOADS = (0.9, 1.0)   # the loads with overloaded runs; below 0.9 the testbed sustained every run
CACHE = CAMPAIGN.parent / "load_aware_b4"   # statics/load_aware_b4/, next to utilization_one.joblib


@contextmanager
def _with_invalid_runs(experiments: set):
    """run_table keeps the invalid rows of `experiments` (and filters every other experiment as before)."""
    original = real_data.run_table

    def run_table(experiment: str) -> pd.DataFrame:
        if experiment not in experiments:
            return original(experiment)
        return pd.read_csv(experiment_folder(experiment) / "dataset.csv").reset_index(drop=True)

    real_data.run_table = run_table
    try:
        yield
    finally:
        real_data.run_table = original


def _overloaded(experiment: str) -> pd.Series:
    """Whether the testbed marked every run of `experiment` invalid, indexed by run_id. Every invalid run of B.4a is
    flagged "detector:overloaded" (checked)."""
    table = pd.read_csv(experiment_folder(experiment) / "dataset.csv")
    invalid = ~table.groupby("run_id")["valid"].all()
    flags = table[~table["valid"]]["flags"].astype(str).unique()
    assert all(flag.endswith(":overloaded") for flag in flags), f"{experiment}: an invalid run with another reason: {flags}"
    return invalid


def _load_data(load: float, seed_value: int, options: dict, base: dict):
    """ChainData of every unthrottled run (valid or not, at most b4_max_workers workers per service) at `load`, both
    measurements one after the other, with an infinite ground truth for the overloaded runs; and the overloaded mask
    in the same order."""
    parts, overloaded = [], []
    for campaign in b4_campaigns("load", load):
        experiment = b4_experiment("load", load, campaign)
        with _with_invalid_runs({experiment}):
            data = real_chain_data(experiment, max_workers=options["b4_max_workers"],
                                   **{**base, "seed": [setup.seed, seed_value], "training_campaign": WORKER_TRAINING, "n_train": setup.N_TRAIN})
            # the order of the held-out runs in real_chain_data: chain_runs, filtered (unthrottled, max_workers)
            runs = [run for run in chain_runs(experiment) if run in set(real_data.unthrottled_runs(experiment))]
            workers = real_data.chain_run_workers(experiment, runs)
            runs = [run for run, w in zip(runs, workers) if w.max() <= options["b4_max_workers"]]
        assert len(runs) == len(data.measured), f"{experiment}: the run order does not match the loader"
        mask = _overloaded(experiment).loc[runs].to_numpy()
        data.measured = np.where(mask, np.inf, data.measured)
        data.measured_low = np.where(mask, np.inf, data.measured_low)
        parts.append(data)
        overloaded.append(mask)
    data = parts[0]
    for part in parts[1:]:
        data = join_chain_data(data, part)
    data.node_latencies = None   # no Oracle: it needs the steady-state latencies of every run, which an overloaded run lacks
    return data, np.concatenate(overloaded)


def overload_detection(recompute: bool = False) -> dict:
    """load -> dict(methods: seed -> name -> (group, latency per run in s), measured: seed -> measured quantile (s, inf
    where overloaded), measured_low: seed -> lower edge of its confidence interval, overloaded: mask per run)."""
    path = CACHE / "overload_detection.joblib"
    inputs = dict(code=setup.code_fingerprint(), module=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16])
    if path.exists() and not recompute:
        stored = joblib.load(path)
        if stored["inputs"] == inputs:
            return stored["result"]
    options = setup.configure("real")
    base = dict(level=1 - setup.alpha, coverage_check=setup.coverage_check, **options["base_options"])
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for load in LOADS:
            methods, measured, measured_low, overloaded = {}, {}, {}, None
            for s in setup.seeds:
                data, mask = _load_data(load, s, options, base)
                assert overloaded is None or np.array_equal(mask, overloaded), "the runs differ between the seeds"
                overloaded = mask
                methods[s] = same_information_methods(data, setup.seed + s, setup.alpha, setup.confidence, setup.n_calibration)
                measured[s], measured_low[s] = data.measured, data.measured_low
            result[load] = dict(methods=methods, measured=measured, measured_low=measured_low, overloaded=overloaded)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(inputs=inputs, result=result), path)
    return result


def detection_table(result: dict, names: list, confidence: float) -> pd.DataFrame:
    """One column per load, one block of rows per method: the 2×2 of predicted infinite / finite against overloaded /
    sustained (run counts, median over the seeds), and coverage and tightness on the runs that the method gives a
    finite bound and the testbed sustained (median over the seeds; ● where the coverage reaches `confidence` under
    every seed)."""
    columns = {}
    for load, r in result.items():
        seeds, overloaded = list(r["methods"]), r["overloaded"]
        column = {("runs", "overloaded / all"): f"{overloaded.sum()} / {len(overloaded)}"}
        for name in [name for name in names if name in r["methods"][seeds[0]]]:
            infinite = np.array([~np.isfinite(r["methods"][s][name][1]) for s in seeds])   # (seeds, runs)
            count = lambda mask: f"{np.median(mask.sum(axis=1)):g}"
            column[(name, "∞ predicted, overloaded (correct alarm)")] = count(infinite & overloaded)
            column[(name, "∞ predicted, sustained (false alarm)")] = count(infinite & ~overloaded)
            column[(name, "finite, overloaded (missed overload)")] = count(~infinite & overloaded)
            column[(name, "finite, sustained")] = count(~infinite & ~overloaded)
            coverages, tightnesses = [], []
            for k, s in enumerate(seeds):
                both = ~infinite[k] & ~overloaded
                latency = r["methods"][s][name][1][both]
                coverages.append(np.mean(latency >= r["measured_low"][s][both]) if both.any() else np.nan)
                tightnesses.append(np.median(latency / r["measured"][s][both]) if both.any() else np.nan)
            coverages = np.array(coverages)
            column[(name, "coverage, finite and sustained")] = f"{np.nanmedian(coverages):.0%}" + (" ●" if np.all(coverages >= confidence) else " ○")
            column[(name, "tightness, finite and sustained")] = f"{np.nanmedian(tightnesses):.2f}×"
        columns[f"u = {load:g}"] = column
    return pd.DataFrame(columns)
