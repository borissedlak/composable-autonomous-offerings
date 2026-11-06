"""Shared on-disk cache of fitted per-service models, so that every experiment that fits a service on the same training
runs reuses one fit instead of training it again (e.g. the models of B.1 in B.5, B.6, C.1 and C.2, the base case of
B.1 in 2a and in b1_base, and the training states that the schedules of B.2 share).

A fitted model is identified by what determines it: the kind of model, its training inputs and targets (the arrays
themselves, hashed), the training code (service_gp.py, where the GPs are built and fitted) and the scikit-learn version.
GP training is deterministic (a fixed random_state per GP, every service fitted on its own data), so a cached fit is
the fit that training would give again. Names of chains, seeds or experiments are not part of the key: two experiments
share a fit exactly when they train a service on the same runs.

Files: statics/model_cache/<kind>_<key>.joblib (never committed, like every *.joblib), never deleted automatically;
the folder can be removed at any time, which only costs training time. The evaluation cache (evaluation_setup.cached)
stays as it is: a change of the code in pegasus/ still recomputes every experiment, but outside service_gp.py it no
longer retrains the GPs.
"""
import hashlib
import os
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn

MODEL_CACHE = Path(__file__).resolve().parents[3] / "statics" / "model_cache"
_TRAINING_CODE = Path(__file__).resolve().parent / "service_gp.py"


def _array_bytes(values) -> bytes:
    """The bytes of an array of training data with its shape, so that arrays with the same values but another shape differ."""
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    return repr(array.shape).encode() + array.tobytes()


def cached_model(kind: str, arrays: tuple, train: Callable[[], object]) -> tuple:
    """(model, training time in s) of the model of kind `kind` fitted on `arrays` (its inputs and targets): loaded from
    MODEL_CACHE if it was fitted before, else trained by train() and stored. The training time is the time that the
    first training took, so that time tables report the cost of training and not of loading."""
    digest = hashlib.sha256(kind.encode() + sklearn.__version__.encode() + _TRAINING_CODE.read_bytes())
    for values in arrays:
        digest.update(_array_bytes(values))
    path = MODEL_CACHE / f"{kind}_{digest.hexdigest()[:24]}.joblib"
    if path.exists():
        return joblib.load(path)
    started = time.perf_counter()
    model = train()
    entry = (model, time.perf_counter() - started)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    # written under a temporary name and renamed, so that a notebook running at the same time never reads a partial file
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    joblib.dump(entry, temporary, compress=3)
    os.replace(temporary, path)
    return entry
