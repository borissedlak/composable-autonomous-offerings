"""What the evaluation notebooks 2a to 2e share: the parameters of every experiment, the options that differ between
the two data sources, and the base case of B.1 (the data and models of the chain of B.1 for every seed, and every
method on it), which B.2 to C.2 build on.

Every experiment cell wraps its computation in a function and calls cached (below), so that a notebook that is run
again only to change a figure or a text reads its results instead of computing them anew.

Every notebook starts with
    data_source = "real"   # or "synthetic"
    from pegasus.evaluation_setup import *
    globals().update(configure(data_source))
    globals().update(b1_base(data_source))   # where the notebook needs the base case of B.1
so that a parameter is defined once for all notebooks. b1_base computes the base case once and keeps it in
statics/evaluation_cache/ (never committed); see b1_base for when it is computed anew.
"""
import ast
import hashlib
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.exceptions import ConvergenceWarning

from .bounds import network_service_bound, propagated_node_bounds, snc_latency_bound
from .data import (assert_unseen, busiest_utilization, load_chain_data, model_inputs, path_workers, same_configurations, synthetic_chain,
                   synthetic_tree_data, with_training_data)
from .evaluation import (METHOD_COLORS, PROPOSED, PROPOSED_SNC, TIGHTNESS_RANGE, TIGHTNESS_TICKS, arrival_text, compare_methods,
                         compare_tree_methods, coverage,
                         finite_share, fit_services, ground_truth_conclusion, sweep_conclusion, sweep_seconds, sweep_statistics,
                         tightness_and_time_figure, tightness_figure, tightness_sweep_figure, time_conclusion)
from .real_data import (BENCHMARKS_V2, CAMPAIGN_V2, campaign_nodes, chain_path, chain_tail_ratios, training_pool, unthrottled_pool,
                        unthrottled_runs)
from .service_gp import configurations_of
from .trees import balanced_tree, narrow_tree

alpha = 0.05                                           # violation probability epsilon: a bound at alpha bounds the (1 - alpha) latency quantile
confidence = 0.95                                      # the pessimistic mean of a bound lies above the true mean with this probability
coverage_check = dict(n_batches=10, confidence=0.95)   # confidence interval of a measured quantile (batch means)
seed = 35                                              # base seed of the data source
seeds = list(range(3))                                 # every experiment is repeated for these seeds; a seed determines the training data and the
                                                       # randomness of the ground truth, and with synthetic data also the generated service types
n_calibration = 20                                     # training configurations that Conformal sets aside for calibration; the rest trains its model
p_label = f"p{100 * (1 - alpha):g}"
pegasus_versions = [PROPOSED, PROPOSED_SNC]            # the two versions of the proposed method: union bound over node bounds, network service curve
competitors = [name for name in METHOD_COLORS if name != "Oracle"]   # the proposed method and the baselines; the Oracle is a reference
# B.2 to B.6 keep one of the two execution-time baselines. GP Sum stays, because it is the faster one: it uses the GPs that
# Pegasus and GP Queue train anyway, while DNN Sum needs a network trained per service.
only_base_case = ("DNN Sum",)
sweep_competitors = [name for name in competitors if name not in only_base_case]  # the methods of B.2 to B.6 without the Oracle

CACHE = Path(__file__).resolve().parents[3] / "statics" / "evaluation_cache"


def configure(data_source: str) -> dict:
    """What differs between the two data sources ("synthetic" or "real"): which chain an experiment uses, which loads
    and trees exist, and which experiments are skipped. A dictionary name -> value for globals().update."""
    options = dict(data_source=data_source)
    if data_source == "synthetic":
        options.update(
            chain_options=lambda s: dict(path=synthetic_chain(3), service_seed=s),               # the chain of B.1 under the seed s
            training_options=lambda s: dict(service_seed=s),                                     # B.2: what with_training_data needs besides the data
            tree_levels=list(range(1, 9)),                                                       # B.3: levels h of the balanced trees (2^h - 1 services)
            narrow_levels=list(range(5, 9)),                                                     # B.3: levels h of the narrow trees (2h - 1 services)
            tree_utilization=0.5,                                                                # B.3: utilization of the slowest service of a tree
            load_option="arrival_rate", loads=[1, 2, 3, 4, 5],                                   # B.4a: arrival rate (items/s) of all configurations
            load_label=lambda load: f"{load:g}/s",
            base_options={},                                                                     # B.1: the default number of training configurations
        )
    else:
        base_experiment = "b1-vision-1-single"                                                   # the chain of B.1: detector -> transformer -> compressor, 1 worker each
        chain_options = lambda s: dict(experiment=base_experiment, training_campaign=BENCHMARKS_V2)   # every service trained on its benchmark
        options.update(
            base_experiment=base_experiment,
            base_options=dict(n_train=25),                                                       # B.1: benchmark runs per service and seed (all chains: the smallest
                                                                                                 # benchmark of a service has barely more, see B.1 on all chains)
            chain_options=chain_options,
            # the most benchmark runs that every service of the chain has (unthrottled, with a measurable handling overhead): the largest size of B.2
            n_pool=min(len(unthrottled_pool(training_pool(service, campaign=BENCHMARKS_V2))) for service in chain_path(base_experiment)[1].values()),
            training_options=lambda s: dict(level=1 - alpha, coverage_check=coverage_check, **chain_options(s)),
        )
    # Experiments that the data source does not support; their cells do nothing (the section of each says why). The real part
    # of B.2b uses the first campaign, so it is skipped with either source.
    options["skipped"] = {"B.2b real"} if data_source == "synthetic" else {"B.2b real", "B.3", "B.4a", "B.4b", "C.2"}

    def chain_data(s, **changes):
        # Data of the chain of B.1 under the seed s; `changes` replace options of the loader of the data source
        return load_chain_data(data_source, **{**dict(level=1 - alpha, coverage_check=coverage_check, seed=[seed, s]), **options["chain_options"](s),
                                               **options["base_options"], **changes})

    options["chain_data"] = chain_data
    return options


def base_case(bases: dict, base_models: dict, s: int) -> tuple:
    """Every method, and every composition of the ablation, for the chain of the seed s: two dictionaries
    name -> (group, end-to-end latency per held-out configuration in s)."""
    data, models = bases[s], base_models[s]
    K = len(data.path)
    methods, _ = compare_methods(data, models, data.x_train, data.monitored_latencies, alpha, confidence, n_calibration, model_seed=seed + s,
                                 x_train_load=data.x_train_load)
    workload = data.workload
    beliefs, lower_means, workers = models["beliefs"], models["lower_means"], path_workers(data)   # workers of every service per configuration
    # Ablation: the other ways to bound the nodes and to compose them, all on the same GP beliefs
    compositions = {
        "union bound, Chernoff (SNC) nodes": ("proposed", sum(snc_latency_bound(*b, epsilon=alpha / K, workers=w, **workload) for b, w in zip(beliefs, workers))),
        "union bound, propagated Chernoff (SNC) nodes": ("proposed", propagated_node_bounds(beliefs, epsilon=alpha / K, path_lower_means=lower_means,
                                                                                            path_workers=workers, **workload).sum(axis=0)),
        "union bound, martingale nodes = Pegasus Node": ("proposed", methods[PROPOSED][1]),
        "network service curve, independent services = Pegasus SNC": ("proposed", methods[PROPOSED_SNC][1]),
        "network service curve, Hölder (any dependence)": ("proposed", network_service_bound(beliefs, epsilon=alpha, path_workers=workers, **workload)),
    }
    return methods, compositions


# The modules whose code computes the base case of B.1; of evaluation.py only the part above its figures, so that a change
# of a figure does not compute the base case anew. Of this module, only the functions that compute the base case count
# (configure, base_case, b1_base; see _cache_key)
CACHED_CODE = ("baselines.py", "bounds.py", "data.py", "evaluation.py", "real_data.py", "service_gp.py", "simulation.py", "trees.py")


def _cache_key(data_source: str, options: dict) -> str:
    """Hash of what the base case depends on: the parameters above, the code of CACHED_CODE and what configure,
    base_case and b1_base compute (_function_code). A change of any of them computes the base case anew."""
    package = Path(__file__).resolve().parent
    code = b""
    for name in CACHED_CODE:
        text = (package / name).read_bytes()
        if name == "evaluation.py":
            text = text.split(b"\n# Figures\n")[0]
        code += text
    parameters = repr((data_source, alpha, confidence, coverage_check, seed, seeds, n_calibration, options["base_options"])).encode()
    code += b"".join(_function_code(function) for function in (configure, base_case, b1_base))
    return hashlib.sha256(parameters + code).hexdigest()[:16]


def b1_base(data_source: str, refresh: bool = False) -> dict:
    """The base case of B.1 for every seed, as a dictionary name -> value for globals().update:
    bases (seed -> data of the chain, no model sees a run of it), base_models (seed -> per-service models, with DNN),
    per_seed (one (methods, compositions) per seed, base_case), base (the data of the first seed, for what is the same
    under every seed), K (services in the chain), workload, n_train (training configurations per service) and arrivals
    (the arrival process in words).
    Computed once and kept in statics/evaluation_cache/b1_<data source>_<key>.joblib; the key (_cache_key) changes with
    the parameters and the code of the package, so a stale file is never read. refresh=True computes it anew anyway."""
    options = configure(data_source)
    path = CACHE / f"b1_{data_source}_{_cache_key(data_source, options)}.joblib"
    if path.exists() and not refresh:
        bases, base_models, per_seed = joblib.load(path)
        print(f"base case of B.1 loaded from {path.relative_to(CACHE.parents[1])}")
    else:
        bases, base_models = {}, {}
        with warnings.catch_warnings():
            # for some services the fitted GP hyperparameters end at their bounds, and scikit-learn warns for every such model
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in seeds:
                bases[s] = options["chain_data"](s)
                assert_unseen(bases[s])
                base_models[s] = fit_services(bases[s], bases[s].x_train, bases[s].y_train, model_seed=seed + s, confidence=confidence)
            per_seed = [base_case(bases, base_models, s) for s in seeds]
        CACHE.mkdir(parents=True, exist_ok=True)
        for stale in CACHE.glob(f"b1_{data_source}_*.joblib"):   # files of older parameters or code
            stale.unlink()
        joblib.dump((bases, base_models, per_seed), path)
        print(f"base case of B.1 computed and saved to {path.relative_to(CACHE.parents[1])}")
    base = bases[seeds[0]]
    return dict(bases=bases, base_models=base_models, per_seed=per_seed, base=base, K=len(base.path), workload=base.workload,
                n_train=len(configurations_of(base.x_train, base.path[0])), arrivals=arrival_text(base.workload))


def _computation_code() -> bytes:
    """The code of the package without what only draws: figures.py, the part of evaluation.py below its figures, and
    every top-level function whose name ends in _figure."""
    package = Path(__file__).resolve().parent
    code = b""
    for path in sorted(package.glob("*.py")):
        if path.name == "figures.py":
            continue
        text = path.read_text()
        if path.name == "evaluation.py":
            text = text.split("\n# Figures\n")[0]
        lines = text.splitlines(keepends=True)
        for node in reversed(ast.parse(text).body):   # from the end, so that the line numbers of the earlier ones stay valid
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_figure"):
                start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list]) - 1
                del lines[start:node.end_lineno]
        code += path.name.encode() + "".join(lines).encode()
    return code


def _function_code(function) -> bytes:
    """What a function computes: its bytecode, the names it uses and its constants, also of the functions and lambdas
    defined inside it, but not its comments and line numbers."""
    def constant_bytes(constant) -> bytes:
        if hasattr(constant, "co_code"):
            return describe(constant)
        if isinstance(constant, tuple):
            return b"(" + b",".join(constant_bytes(item) for item in constant) + b")"
        if isinstance(constant, frozenset):
            # a set literal of constants compiles to a frozenset, whose repr lists the elements in the order of their
            # string hashes; these differ between Python processes, so the key would change with every new kernel
            return b"frozenset(" + b",".join(sorted(constant_bytes(item) for item in constant)) + b")"
        return repr(constant).encode()

    def describe(code) -> bytes:
        return code.co_code + repr(code.co_names).encode() + b"".join(constant_bytes(constant) for constant in code.co_consts)
    return describe(function.__code__)


def cached(name: str, data_source: str, compute, inputs: dict = None, refresh: bool = False) -> dict:
    """The results of the experiment `name`: compute() returns them as a dictionary name -> value (for
    globals().update). They are kept in statics/evaluation_cache/<name>_<data source>_<key>.joblib, where the key
    covers the shared parameters above, `inputs` (the settings of the experiment that compute reads from outside, e.g.
    the swept sizes), what compute itself computes (_function_code) and the code of the package (_computation_code). A
    change of any of them computes the results anew; refresh=True does so anyway. The base case of B.1 that compute
    builds on has its own key (b1_base) that follows from the same parameters and code."""
    options = configure(data_source)
    key_source = repr((data_source, alpha, confidence, coverage_check, seed, seeds, n_calibration, options["base_options"],
                       sorted((inputs or {}).items()))).encode()
    key = hashlib.sha256(key_source + _function_code(compute) + _computation_code()).hexdigest()[:16]
    path = CACHE / f"{name}_{data_source}_{key}.joblib"
    if path.exists() and not refresh:
        print(f"{name}: results loaded from {path.relative_to(CACHE.parents[1])}")
        return joblib.load(path)
    results = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale in CACHE.glob(f"{name}_{data_source}_*.joblib"):   # results of older parameters or code
        stale.unlink()
    joblib.dump(results, path)
    print(f"{name}: results computed and saved to {path.relative_to(CACHE.parents[1])}")
    return results
