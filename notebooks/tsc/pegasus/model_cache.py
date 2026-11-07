"""Shared on-disk cache of fitted models, so that every experiment that fits a model on the same training data reuses
one fit instead of training it again. Every GP of the repository is fitted through it (train_gp_models in
service_gp.py: the execution-time GPs of Pegasus and of GP Sum and GP Queue, and the latency-quantile GPs of Quantile
Union, Conformal, Quantile Sum and the Fluxion-style baseline), and so is every network of DNN Sum (train_dnn_models
in baselines.py).

A fitted model is identified by what determines it: the kind of model, its training inputs and targets (the arrays
themselves, hashed), its training parameters (`params`, e.g. the seed of a network or the lower limit of a length
scale), the training code (service_gp.py, and `code` for a model defined elsewhere) and the library versions.
Training is deterministic (a fixed random_state per GP, a fixed seed and one thread per network), so a cached fit is
the fit that training would give again. Names of chains, seeds or experiments are not part of the key: two
experiments share a fit exactly when they train a model on the same data.

Training times. A cached fit keeps the time its first training took. Code that reports training times measures them
with training_clock: `seconds` adds up the first training time of every fit inside the block, whether it was trained
or loaded, and `wall` the time the fits actually took there (training, or loading the file), so that the time of the
rest of the block (e.g. the evaluation of a method) is the elapsed time minus `wall`.

Files: statics/cache/models/<kind>_<key>.joblib (never committed, like every *.joblib), never deleted automatically;
the folder can be removed at any time, which only costs training time. The evaluation cache (evaluation_setup.cached)
stays as it is: a change of the code in pegasus/ still recomputes every experiment, but outside service_gp.py (and
the network of baselines.py) it no longer retrains a model.
"""
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import sklearn

MODEL_CACHE = Path(__file__).resolve().parents[3] / "statics" / "cache" / "models"
_TRAINING_CODE = Path(__file__).resolve().parent / "service_gp.py"


@dataclass
class TrainingClock:
    """Times of the fits inside a training_clock block (s): `seconds`, the first training time of every fit, also of
    one loaded from the cache; `wall`, the time the fits took in this block (training or loading)."""
    seconds: float = 0.0
    wall: float = 0.0


_clocks = []   # the training_clock blocks that are open, innermost last
_depth = 0     # cached_model calls in progress; only the outermost adds to the clocks, so that a fit inside a fit counts once


@contextmanager
def training_clock():
    """Block that measures the training time of the fits inside it (TrainingClock)."""
    clock = TrainingClock()
    _clocks.append(clock)
    try:
        yield clock
    finally:
        _clocks.remove(clock)


def _array_bytes(values) -> bytes:
    """The bytes of an array of training data with its shape, so that arrays with the same values but another shape differ."""
    array = np.ascontiguousarray(np.asarray(values, dtype=float))
    return repr(array.shape).encode() + array.tobytes()


def cached_model(kind: str, arrays: tuple, train: Callable[[], object], params: tuple = (), code: str = "") -> tuple:
    """(model, training time in s) of the model of kind `kind` fitted on `arrays` (its inputs and targets) with the
    training parameters `params` and the further training code `code`: loaded from MODEL_CACHE if it was fitted
    before, else trained by train() and stored. The training time is the time that the first training took, so that
    time tables report the cost of training and not of loading; it is also added to every open training_clock."""
    global _depth
    digest = hashlib.sha256(kind.encode() + sklearn.__version__.encode() + _TRAINING_CODE.read_bytes() + repr(params).encode() + code.encode())
    for values in arrays:
        digest.update(_array_bytes(values))
    path = MODEL_CACHE / f"{kind}_{digest.hexdigest()[:24]}.joblib"
    started = time.perf_counter()
    _depth += 1
    try:
        if path.exists():
            entry = joblib.load(path)
        else:
            model = train()
            entry = (model, time.perf_counter() - started)
            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            # written under a temporary name and renamed, so that a notebook running at the same time never reads a partial file
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            joblib.dump(entry, temporary, compress=3)
            os.replace(temporary, path)
    finally:
        _depth -= 1
    if _depth == 0:
        for clock in _clocks:
            clock.seconds += entry[1]
            clock.wall += time.perf_counter() - started
    return entry
