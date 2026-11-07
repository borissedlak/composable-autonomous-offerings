"""On-disk cache of the values that real_data.py derives from the raw files of one run (per-item records, traces and
trace steps), so that every notebook and every kernel reads a raw CSV file only once. Parsing the raw files is most of
the time of loading the training and held-out data of a chain: the training pool of the load-aware models alone has
about 1,100 benchmark runs, each with its own per-item files.

A cached value is identified by what determines it: the function, its source code and that of the functions it
depends on (`depends`), its arguments, and the path, size and modification time of every raw file it reads. A
changed raw file or reader therefore gives a new key, and the value is computed again; the result is the value that
the function would compute, so the cache never changes a result.

Files: statics/cache/runs/<function>/<key>.joblib (never committed, like every *.joblib), never deleted
automatically; the folder can be removed at any time, which only costs reading time.
"""
import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

RUN_CACHE = Path(__file__).resolve().parents[3] / "statics" / "cache" / "runs"


def _file_signature(path: Path) -> str:
    """Path, size and modification time of a raw file, or that it is missing."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return f"{path}:missing"
    return f"{path}:{status.st_size}:{status.st_mtime_ns}"


def _read_only(value):
    """The value with every array in it (also inside a tuple) made read-only, as the cached readers return them."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, tuple):
        for item in value:
            _read_only(item)
    return value


def disk_cached(files: Callable[..., list], depends: tuple = (), read_only: bool = False):
    """Decorator: cache the return value of a reader of raw run files in RUN_CACHE. files(*args) lists the raw files
    that the reader reads for these arguments; depends lists further functions whose code the value depends on.
    read_only: make the arrays of the value read-only also when it is loaded from the cache."""
    def decorate(function):
        code = hashlib.sha256("".join(inspect.getsource(f) for f in (function, *depends)).encode()).hexdigest()

        @functools.wraps(function)
        def wrapper(*args):
            signature = "|".join([code, repr(args), *(_file_signature(Path(file)) for file in files(*args))])
            path = RUN_CACHE / function.__name__ / f"{hashlib.sha256(signature.encode()).hexdigest()[:24]}.joblib"
            if path.exists():
                value = joblib.load(path)
            else:
                value = function(*args)
                path.parent.mkdir(parents=True, exist_ok=True)
                # written under a temporary name and renamed, so that a notebook running at the same time never reads a partial file
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                joblib.dump(value, temporary)
                os.replace(temporary, path)
            return _read_only(value) if read_only else value
        return wrapper
    return decorate
