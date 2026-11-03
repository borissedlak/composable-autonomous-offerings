"""B.4 with load- and worker-aware execution-time models: a variant of Pegasus Node for `2c_load.ipynb`, compared
with the current Pegasus Node.

Current Pegasus (PROPOSED in pegasus/evaluation.py): every service's GPs are trained on the service's benchmark
(BENCHMARKS_V2: the service alone, one worker, five utilizations) and receive only the quality knobs; model_inputs in
pegasus/data.py drops the number of workers, and the arrival rate is no input at all. The worker-scaling analysis
(worker_scaling_analysis.py) and the B.4 diagnosis showed that the execution time of several services grows with the
number of workers and at low load, which the current models cannot represent.

The variant (VARIANT below) changes two things and nothing else:
- Inputs: the GPs of the mean execution time and of the spread of single items receive the quality knobs, the number
  of workers (knob parallelism) and log10 of the arrival rate at the service (items/s): ChainData.x_train_load and
  x_test_load of real_chain_data, with the worker column kept (workers_column=None for the fit). The arrival rate is
  the load input rather than the utilization, because the utilization contains the execution time that the model
  predicts; the rate of every held-out run is known in advance (the workload of the run).
- Training data: the service's benchmark runs plus its runs in the other chains of the second campaign, one worker
  (CAMPAIGN_V2, `b1/`) and one to four workers (CHAINS_UNTHROTTLED_V2, `b1-unthrottled/`), without any experiment of
  the chain of B.4 (B4_EXCLUDED); the benchmarks alone have one worker only, so the worker effect could not be learned
  from them. Every chain run has the rate that the campaign set for it.
The bound itself is unchanged (pegasus_bound): it reads the workers of every service from the held-out configurations
for the queueing part, as before.

ABLATION trains on the same extended data with the inputs of the current Pegasus (no workers, no rate), so that the
comparison separates the effect of more data from the effect of the new inputs.

Training runs per service: every run of the extended pool (N_TRAIN = None, real_chain_data), the same for the
variant and the ablation; the current Pegasus of B.4a uses every benchmark run that all three services have (n_pool).
Everything else (seeds, alpha, confidence, held-out runs, ground truth, handover) is that of 2c_load.ipynb.

The wider training pool is set up at call time (extended_training) instead of in pegasus/real_data.py, because the
cache key of the evaluation hashes every module of pegasus/: a change there would recompute every cached experiment.
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pegasus.real_data as real_data
from pegasus.data import model_inputs
from pegasus.evaluation import fit_services, pegasus_bound
from pegasus.real_data import BENCHMARKS_V2, CAMPAIGN, CAMPAIGN_V2, CHAINS_UNTHROTTLED_V2, b4_chain_data, run_table
from pegasus.service_gp import configurations_of, run_statistics_belief

EXTENDED = "benchmark+all-chains"   # training_campaign of real_chain_data under extended_training
B4_EXCLUDED = ("b1-vision-1-single", "b1u-vision-1", "b1u2-vision-1")   # the chain of B.4 in B.1 and its two measurements with workers
VARIANT = "Pegasus Node, load and workers"
ABLATION = "Pegasus Node, more data"
N_TRAIN = None   # every run of the extended pool, in a random order (seed)
CACHE = CAMPAIGN.parent / "load_aware_b4"   # statics/load_aware_b4/

_original = dict(training_pool=real_data.training_pool, training_exclusion=real_data.training_exclusion, campaign_nodes=real_data.campaign_nodes)


@lru_cache(maxsize=None)
def _campaign_nodes(campaign: Path = CAMPAIGN) -> pd.DataFrame:
    """campaign_nodes without hidden folders (e.g. a tool's lock folder inside a campaign), which hold no experiment."""
    rows = []
    for folder in sorted(path for path in campaign.iterdir() if path.is_dir() and path.name not in ("campaigns", "results")
                         and not path.name.startswith(("check-", "."))):
        table = run_table(folder.name)
        nodes = table.drop_duplicates("node")
        rows += [dict(experiment=folder.name, pipeline=table["pipeline"].iloc[0], alone="->" not in table["pipeline"].iloc[0], node=node, service=service)
                 for node, service in zip(nodes["node"], nodes["service"])]
    return pd.DataFrame(rows)


def extended_pool(service: str, sources: tuple = ("alone", "pipeline")) -> pd.DataFrame:
    """Training runs of `service`: its benchmark, then its runs in the chains of CAMPAIGN_V2 and CHAINS_UNTHROTTLED_V2,
    without B4_EXCLUDED (rows of training_pool)."""
    pool = pd.concat([_original["training_pool"](service, sources=sources, campaign=campaign) for campaign in (BENCHMARKS_V2, CAMPAIGN_V2, CHAINS_UNTHROTTLED_V2)],
                     ignore_index=True)
    return pool[~pool["experiment"].isin(B4_EXCLUDED)].reset_index(drop=True)


@contextmanager
def extended_training():
    """Within the block, real_chain_data(..., training_campaign=EXTENDED) trains every service on extended_pool."""
    def training_exclusion(experiment, training_campaign=None):
        return dict(campaign=EXTENDED) if training_campaign == EXTENDED else _original["training_exclusion"](experiment, training_campaign)

    def training_pool(service, exclude_pipeline=None, sources=("alone", "pipeline"), campaign=CAMPAIGN, exclude_experiment=None):
        if campaign == EXTENDED:
            return extended_pool(service, tuple(sources))
        return _original["training_pool"](service, exclude_pipeline, sources, campaign, exclude_experiment)

    patched = dict(training_pool=training_pool, training_exclusion=training_exclusion, campaign_nodes=_campaign_nodes)
    for name, function in patched.items():
        setattr(real_data, name, function)
    try:
        yield
    finally:
        for name, function in _original.items():
            setattr(real_data, name, function)


def load_aware(data):
    """The ChainData that the variant's models see: the inputs with the arrival rate (x_train_load, x_test_load) and
    with the number of workers kept (workers_column None, so model_inputs drops nothing)."""
    return replace(data, x_train=data.x_train_load, x_test=data.x_test_load, workers_column=None)


def beliefs_at(models: dict, data, confidence: float) -> dict:
    """models (from fit_services) with their beliefs recomputed at the held-out configurations of `data`, as
    fit_services computes them (run_statistics_belief: the GPs are trained on run statistics), without refitting the
    GPs. `data` must present the configurations with the inputs the GPs were trained on: load_aware(data) for the
    variant, the plain data for the ablation (model_inputs drops the workers here). Only the beliefs change, which is
    all that pegasus_bound reads."""
    x_test = model_inputs(data.x_test, data.workers_column)
    return {**models, "beliefs": [run_statistics_belief(models["gp"][node], configurations_of(x_test, node), confidence) for node in data.path]}


def sweep_bounds(sweep: str, settings: list, seeds: list, seed: int, alpha: float, confidence: float, coverage_check: dict,
                 b4_max_workers: int, base_options: dict) -> dict:
    """The variant and the ablation on every setting of one sweep of B.4 ("load" or "arrivals") and every seed:
    (setting, seed) -> dict(bounds=name -> latency per held-out run (s), measured, measured_low). The data of a setting
    is that of 2c_load.ipynb (b4_chain_data with seed [seed, s]) with the extended training pool. The GPs of a seed are
    fitted once, on the data of the first setting, and every later setting checks that its training data is the same;
    the beliefs are recomputed at every setting's held-out runs (beliefs_at), because the variant's inputs contain
    the arrival rate of the run. The bound is pegasus_bound on the original data, so that it reads the workers of
    every service for the queueing part."""
    results, fitted = {}, {}
    with extended_training():
        for setting in settings:
            for s in seeds:
                data = b4_chain_data(sweep, setting, settings, b4_max_workers, **{**dict(level=1 - alpha, coverage_check=coverage_check, seed=[seed, s],
                                     training_campaign=EXTENDED), **base_options, "n_train": N_TRAIN})
                aware = load_aware(data)
                if s not in fitted:
                    fitted[s] = (data.y_train,
                                 fit_services(aware, aware.x_train, data.y_train, model_seed=seed + s, confidence=confidence, with_dnn=False),
                                 fit_services(data, data.x_train, data.y_train, model_seed=seed + s, confidence=confidence, with_dnn=False))
                y_train, variant_models, ablation_models = fitted[s]
                assert all(np.array_equal(data.y_train[node], y_train[node]) for node in data.path), "the training data differs between the settings"
                results[(setting, s)] = dict(bounds={VARIANT: pegasus_bound(data, beliefs_at(variant_models, aware, confidence), alpha),
                                                     ABLATION: pegasus_bound(data, beliefs_at(ablation_models, data, confidence), alpha)},
                                             measured=data.measured, measured_low=data.measured_low)
    return results


CURRENT = {"load": ("b4a_load", "load_methods", "load_truth"), "arrivals": ("b4b_arrivals", "arrival_methods", "arrival_truth")}
FROM_CURRENT = ("Pegasus Node (ours)", "Oracle")   # read from the cached results of 2c_load.ipynb; the Oracle uses no model


def current_results(sweep: str, evaluation_cache: Path) -> dict:
    """The cached results of the sweep in 2c_load.ipynb (the newest real-data file for the chains of DEFAULT_CHAINS):
    (setting, seed) -> dict(bounds=name -> latency for FROM_CURRENT, measured, measured_low). Read only, never recomputed."""
    name, methods_key, truth_key = CURRENT[sweep]
    path = max(evaluation_cache.glob(f"{name}_real+b1+b1u-max3_*.joblib"), key=lambda p: p.stat().st_mtime)
    print(f"current Pegasus, {sweep}: {path}")
    cached = joblib.load(path)
    return {key: dict(bounds={n: np.asarray(cached[methods_key][key][n][1]) for n in FROM_CURRENT},
                      measured=cached[truth_key][key][0], measured_low=cached[truth_key][key][1]) for key in cached[methods_key]}


def comparison(results: dict, settings: list, seeds: list, confidence: float) -> pd.DataFrame:
    """One row per method and setting (finite_min: the smallest share of finite bounds over the seeds): coverage per seed (share of held-out runs whose bound is not below the lower edge
    of the measured quantile's confidence interval, evaluation.coverage), its minimum over the seeds, holds (coverage
    >= confidence under every seed, the criterion of the evaluation), and tightness (bound / measured quantile: the
    median over the held-out runs per seed, then the median over the seeds, as in sweep_table)."""
    rows = []
    for name in results[(settings[0], seeds[0])]["bounds"]:
        for setting in settings:
            per_seed = [results[(setting, s)] for s in seeds]
            coverage = [np.mean(r["bounds"][name] >= r["measured_low"]) for r in per_seed]
            tightness = [np.median(r["bounds"][name] / r["measured"]) for r in per_seed]
            # an infinite bound (the model expects the busiest service to be overloaded: no bound exists) counts as covered,
            # so the share of finite bounds belongs next to the coverage
            finite = [np.mean(np.isfinite(r["bounds"][name])) for r in per_seed]
            rows.append(dict(method=name, setting=setting, coverage_min=min(coverage), holds=min(coverage) >= confidence,
                             tightness=float(np.median(tightness)), finite_min=min(finite), runs=len(per_seed[0]["measured"])))
    return pd.DataFrame(rows)


def main(evaluation_cache: Path, recompute: bool = False) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from pegasus.evaluation_setup import alpha, confidence, configure, coverage_check, seed, seeds
    options = configure("real")
    CACHE.mkdir(parents=True, exist_ok=True)
    tables = {}
    for sweep, settings in (("load", options["loads"]), ("arrivals", options["arrival_cv2s"])):
        path = CACHE / f"variant_{sweep}.joblib"
        if path.exists() and not recompute:
            variant = joblib.load(path)
        else:
            with warnings.catch_warnings():
                # for some services the fitted GP hyperparameters end at their bounds, and scikit-learn warns for every such model
                warnings.simplefilter("ignore", ConvergenceWarning)
                variant = sweep_bounds(sweep, settings, seeds, seed, alpha, confidence, coverage_check, options["b4_max_workers"], options["base_options"])
            joblib.dump(variant, path)
        current = current_results(sweep, evaluation_cache)
        for key in variant:   # the same held-out runs and ground truth, so the bounds compare run by run
            assert np.array_equal(variant[key]["measured"], current[key]["measured"]), f"{sweep} {key}: the ground truth differs"
            variant[key]["bounds"] = {**current[key]["bounds"], **variant[key]["bounds"]}
        tables[sweep] = comparison(variant, settings, seeds, confidence).assign(sweep=sweep)
    table = pd.concat(tables.values(), ignore_index=True)
    table.to_csv(CACHE / "comparison.csv", index=False)
    pd.set_option("display.width", 220)
    for sweep, frame in tables.items():
        print(f"\n{sweep}: worst-seed coverage (holds if >= {confidence:g} under every seed)")
        print(frame.pivot(index="method", columns="setting", values="coverage_min").round(3).to_string())
        print(f"{sweep}: tightness (bound / measured quantile, median)")
        print(frame.pivot(index="method", columns="setting", values="tightness").round(2).to_string())
    print(f"\nwritten to {CACHE}")


if __name__ == "__main__":
    import sys
    from pegasus.evaluation_setup import CACHE as EVALUATION_CACHE
    arguments = [a for a in sys.argv[1:] if a != "--recompute"]
    main(Path(arguments[0]) if arguments else EVALUATION_CACHE, recompute="--recompute" in sys.argv)
