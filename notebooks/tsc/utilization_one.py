"""The point at utilization 1 of the load sweep B.4a, evaluated on its own held-out runs.

The sweep of 2c_load.ipynb leaves utilization 1 out: there the busiest service of about half of the runs was overloaded
and those runs have no latency quantile (evaluation_setup.py). Putting 1 into `loads` would restrict the held-out runs of
every other load to the runs that are valid at 1 (b4_held_out), so this module evaluates utilization 1 alone, on the
unthrottled valid runs of the measurements (settings=(1,)), and 2c_load.ipynb draws the result as a marked point beside
the sweep. The methods are those of the sweep: every method on the same information (same_information_methods in load_aware_b4.py), so the point differs
from the others only in the runs it holds out. Not hashed by the cache key of the evaluation; its own cache is
statics/load_aware_b4/utilization_one.joblib.
"""
import warnings

import joblib
import numpy as np

from load_aware_b4 import CACHE, EXTENDED, N_TRAIN, code_fingerprint, extended_training, same_information_methods
from pegasus import evaluation_setup as setup
from pegasus.real_data import b4_chain_data

LOAD = 1


def utilization_one(recompute: bool = False) -> dict:
    """(methods, truth): as load_methods and load_truth of 2c_load.ipynb, with the key (LOAD, seed)."""
    path = CACHE / "utilization_one.joblib"
    inputs = dict(code=code_fingerprint())
    if path.exists() and not recompute:
        stored = joblib.load(path)
        if stored["inputs"] == inputs:
            return stored["result"]
    options = setup.configure("real")
    base = dict(level=1 - setup.alpha, coverage_check=setup.coverage_check, **options["base_options"])
    methods, truth = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in setup.seeds:
            with extended_training():
                data = b4_chain_data("load", LOAD, (LOAD,), options["b4_max_workers"], **{**base, "seed": [setup.seed, s], "training_campaign": EXTENDED,
                                                                                         "n_train": N_TRAIN})
            methods[(LOAD, s)] = same_information_methods(data, s, setup.seed + s, setup.alpha, setup.confidence, setup.n_calibration)
            truth[(LOAD, s)] = (data.measured, data.measured_low)
    result = dict(methods=methods, truth=truth)
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(inputs=inputs, result=result), path)
    return result
