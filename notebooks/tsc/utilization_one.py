"""The point at utilization 1 of the load sweep B.4a, evaluated on its own held-out runs.

The sweep of 2c_load.ipynb leaves utilization 1 out: there the busiest service of about half of the runs was overloaded
and those runs have no latency quantile (evaluation_setup.py). Putting 1 into `loads` would restrict the held-out runs of
every other load to the runs that are valid at 1 (b4_held_out), so this module evaluates utilization 1 alone, on the
unthrottled valid runs of the measurements (settings=(1,)), and 2c_load.ipynb draws the result as a marked point beside
the sweep. The models are those of the sweep (the execution times do not depend on the arrivals), so the point differs
from the others only in the runs it holds out. Not hashed by the cache key of the evaluation; its own cache is
statics/load_aware_b4/utilization_one.joblib.
"""
import warnings

import joblib
import numpy as np

from load_aware_b4 import CACHE, EXTENDED, N_TRAIN, code_fingerprint, extended_training, variant_latency
from pegasus import evaluation_setup as setup
from pegasus.evaluation import PROPOSED, compare_methods
from pegasus.real_data import BENCHMARKS_V2, b4_chain_data

LOAD = 1


def utilization_one(load_models: dict, load_y_train: dict, leave_out: tuple, load_n_train: int, recompute: bool = False) -> dict:
    """(methods, truth): as load_methods and load_truth of 2c_load.ipynb, with the key (LOAD, seed). load_models,
    load_y_train: those of the sweep (per seed), leave_out and load_n_train: those of its compute()."""
    path = CACHE / "utilization_one.joblib"
    inputs = dict(code=code_fingerprint(), leave_out=tuple(leave_out), load_n_train=load_n_train)
    if path.exists() and not recompute:
        stored = joblib.load(path)
        if stored["inputs"] == inputs:
            return stored["result"]
    options = setup.configure("real")
    base = dict(level=1 - setup.alpha, coverage_check=setup.coverage_check, **options["base_options"])
    methods, truth, variant_fits = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in setup.seeds:
            data = b4_chain_data("load", LOAD, (LOAD,), options["b4_max_workers"], **{**base, "seed": [setup.seed, s], "training_campaign": BENCHMARKS_V2,
                                                                                    "n_train": load_n_train})
            assert all(np.array_equal(data.y_train[name], load_y_train[s][name]) for name in data.path), "the training data differs from the sweep"
            methods[(LOAD, s)], _ = compare_methods(data, load_models[s], data.x_train, data.monitored_latencies, setup.alpha, setup.confidence,
                                                    setup.n_calibration, model_seed=setup.seed + s, leave_out=leave_out, x_train_load=data.x_train_load)
            truth[(LOAD, s)] = (data.measured, data.measured_low)
            with extended_training():
                variant_data = b4_chain_data("load", LOAD, (LOAD,), options["b4_max_workers"], **{**base, "seed": [setup.seed, s], "training_campaign": EXTENDED,
                                                                                                "n_train": N_TRAIN})
            assert np.array_equal(variant_data.measured, data.measured), "Pegasus holds out other runs"
            methods[(LOAD, s)] = {PROPOSED: ("proposed", variant_latency(variant_data, variant_fits, s, setup.seed + s, setup.alpha, setup.confidence)),
                                  **methods[(LOAD, s)]}
    result = dict(methods=methods, truth=truth)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(inputs=inputs, result=result), path)
    return result
