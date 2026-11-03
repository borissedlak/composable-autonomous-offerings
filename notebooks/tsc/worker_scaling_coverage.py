"""Does the one-worker assumption cost coverage in B.1? (step after worker_scaling_analysis.py)

The models of every method are trained on one-worker benchmarks and never see the number of workers w; the
worker-scaling analysis found that many services are slower at w > 1. This script reads the cached results of
"B.1 on all chains" (2a_base_case.ipynb, experiment b1_all_chains) and splits coverage and tightness of every method
by how much the held-out run should be affected.

Per held-out run (seconds unless stated):
- E, the expected slowdown of the run's total execution time over what one-worker models predict:
  E = sum_k m_k / sum_k (m_k / r_k(w_k)), where m_k is the measured mean execution time of the service at position k
  in the run (dataset.csv), w_k its workers, and r_k(w) the mean-execution-time ratio of that service at w workers to
  1 worker (statics/worker_scaling/worker_ratios.csv, statistic mean_ms; r = 1 at w = 1). E = 1: no effect expected.
  E reuses ratios estimated on the same campaign folder (b1-unthrottled) that holds the multi-worker held-out runs, so
  it is an exposure index from service-level averages over all runs of a service, not an outcome of the run itself.
- covered: bound >= the lower edge of the confidence interval of the measured quantile (evaluation.coverage);
- tightness: bound / measured quantile.
Outputs (statics/worker_scaling/): coverage_runs.csv (one row per chain, seed, held-out run and method),
coverage_by_exposure.csv, coverage_per_chain.csv, tightness_elasticity.csv.
Run from notebooks/tsc/: ../../.venv/bin/python worker_scaling_coverage.py [path of the b1_all_chains cache file]
(default: the newest b1_all_chains_real+b1+b1u-max3_*.joblib in pegasus.evaluation_setup.CACHE).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pegasus.evaluation_setup import CACHE as EVALUATION_CACHE, confidence, seeds
from pegasus.real_data import chain_path, combined_runs, node_configurations, run_table
from worker_scaling import CACHE

METHODS = ["Pegasus Node (ours)", "Pegasus SNC (ours)", "GP Queue", "Oracle"]   # the Oracle needs no model: a control
EXPOSURE_BINS = [0, 1.0001, 1.05, 1.10, np.inf]
EXPOSURE_LABELS = ["E = 1", "1 < E <= 1.05", "1.05 < E <= 1.10", "E > 1.10"]
N_BOOT, SEED = 1000, 0


def held_out_exposure(chain: str, ratio: dict) -> pd.DataFrame:
    """One row per held-out run of the combined chain, in the order of real_chain_data: source (b1: one worker per
    service; b1u: the twin with up to the cap of workers), the largest number of workers, and E (module docstring)."""
    nodes, services, _ = chain_path(chain)
    rows = []
    for experiment, run_id in combined_runs(chain):
        table = run_table(experiment)
        mean_s = table[table["run_id"] == run_id].set_index("node")["processing_mean_ms"] / 1e3
        workers = [int(node_configurations(table, node).loc[run_id, "parallelism"]) for node in nodes]
        r = [1.0 if w == 1 else ratio[(services[node], w)] for node, w in zip(nodes, workers)]
        m = mean_s.loc[nodes].to_numpy()
        rows.append(dict(source="b1u" if experiment.startswith("b1u") else "b1", max_workers=max(workers), E=m.sum() / np.sum(m / r)))
    return pd.DataFrame(rows)


def elasticity(runs: pd.DataFrame) -> dict:
    """Least-squares slope b of log(tightness) on log(E) over the multi-worker held-out runs, both centered within
    (chain, largest number of workers, seed), so only differences in E at the same chain and worker count count:
    tightness changes by the factor E^b. b = -1 if the bound ignores the slowdown and the measured quantile grows in
    proportion to the execution time; queueing makes it steeper. 95% interval: cluster bootstrap over chains."""
    g = runs[(runs["source"] == "b1u") & np.isfinite(runs["tightness"]) & (runs["tightness"] > 0)]
    key = g["chain"] + "|" + g["max_workers"].astype(str) + "|" + g["seed"].astype(str)
    y, x = np.log(g["tightness"]), np.log(g["E"])
    y, x = (y - y.groupby(key).transform("mean")).to_numpy(), (x - x.groupby(key).transform("mean")).to_numpy()
    slope = lambda index: np.sum(x[index] * y[index]) / np.sum(x[index] ** 2)
    chains = g["chain"].to_numpy()
    members = [np.flatnonzero(chains == c) for c in np.unique(chains)]
    rng = np.random.default_rng(SEED)
    boot = [slope(np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])) for _ in range(N_BOOT)]
    low, high = np.quantile(boot, [0.025, 0.975])
    return dict(slope=slope(np.arange(len(x))), low=low, high=high, runs=len(x))


def main(cache_file: Path = None) -> None:
    cache_file = cache_file or max(EVALUATION_CACHE.glob("b1_all_chains_real+b1+b1u-max3_*.joblib"), key=lambda p: p.stat().st_mtime)
    print(f"B.1 results: {cache_file}")
    results = joblib.load(cache_file)
    ratios = pd.read_csv(CACHE / "worker_ratios.csv")
    ratio = {(s, w): v for s, w, v in ratios[ratios["statistic"] == "mean_ms"][["service", "workers", "ratio"]].itertuples(index=False)}
    frames = []
    for chain in results["unseen_chains"]:
        exposure = held_out_exposure(chain, ratio)
        for s in seeds:
            entry = results["unseen"][(chain, s)]
            assert len(entry.measured) == len(exposure), f"{chain}: held-out runs do not match combined_runs"
            for name in METHODS:
                frames.append(exposure.assign(chain=chain, seed=s, method=name, bound=entry.latencies[name],
                                              measured=entry.measured, measured_low=entry.measured_low))
    runs = pd.concat(frames, ignore_index=True)
    runs["covered"] = runs["bound"] >= runs["measured_low"]
    runs["tightness"] = runs["bound"] / runs["measured"]
    runs["exposure"] = pd.cut(runs["E"], EXPOSURE_BINS, labels=EXPOSURE_LABELS)

    by_exposure = runs.groupby(["method", "exposure"], observed=True).agg(
        coverage=("covered", "mean"), tightness=("tightness", "median"), runs=("covered", "size")).reset_index()
    by_exposure["runs"] //= len(seeds)
    # the paper's criterion per chain: coverage >= confidence under every seed; here the worst seed, over all held-out
    # runs and over the multi-worker runs alone
    worst = lambda frame: frame.groupby(["method", "chain", "seed"])["covered"].mean().groupby(["method", "chain"]).min()
    per_chain = pd.DataFrame({"all_runs": worst(runs), "multi_worker_runs": worst(runs[runs["source"] == "b1u"])}).reset_index()
    per_chain["holds_all"] = per_chain["all_runs"] >= confidence
    per_chain["holds_multi"] = per_chain["multi_worker_runs"] >= confidence
    slopes = pd.DataFrame([dict(method=name, **elasticity(runs[runs["method"] == name])) for name in METHODS])
    # where the trend leads: the median tightness of the one-worker runs times E^slope, at the largest E that the
    # per-service ratios reach at 4 workers within the chains of B.1 (outside the held-out runs: an extrapolation)
    e4 = max(ratio[(s, 4)] for s in set(svc for chain in results["unseen_chains"] for svc in chain_path(chain)[1].values()))
    base = runs[runs["source"] == "b1"].groupby("method")["tightness"].median()
    slopes["E_at_4_workers"] = e4
    slopes["tightness_projected"] = [base[m] * e4 ** b for m, b in zip(slopes["method"], slopes["slope"])]

    for name, frame in (("coverage_runs", runs), ("coverage_by_exposure", by_exposure), ("coverage_per_chain", per_chain),
                        ("tightness_elasticity", slopes)):
        frame.to_csv(CACHE / f"{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(by_exposure.pivot(index="method", columns="exposure", values="coverage").round(3).to_string())
    print(by_exposure.pivot(index="method", columns="exposure", values="tightness").round(2).to_string())
    print(per_chain.groupby("method")[["holds_all", "holds_multi"]].sum().astype(int).to_string(), f"of {per_chain['chain'].nunique()} chains")
    print(slopes.round(3).to_string())
    print(f"written to {CACHE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
