"""How much the execution time of each Pegasus service depends on its number of workers (worker_scaling.py).

Pegasus assumes that the execution time of an item does not depend on the number of workers w of its service, so that
the service scales linearly (capacity w / mean execution time). This script measures, per service of the second
campaign's unthrottled chains (statics/results_milos_v2/b1-unthrottled/):
- total:   ratio of the run's mean and 95th-percentile execution time at w workers to the same quality configuration at
           1 worker, at the load of the campaign (worker_effects);
- own:     within a run, median execution time of items that started next to k other items of the same service /
           of items that started alone (k = 1, 2, 3), median over runs (ratio_summary);
- alone:   ratio of the median execution time of the items that started alone at w workers to 1 worker, matched on
           the configuration like the total: the part that is not concurrency within the service;
- rest, idle: within-run checks of two other explanations (items in process at other services of the chain; idle time
           of the worker before the item);
- speedup: projected capacity at full load relative to 1 worker, w / (alone_w x own_{w-1}), the speedup that linear
           scaling would put at w. Assumption: the two factors multiply, and own_{w-1} measured at the campaign's
           loads holds when all w workers are busy.
Outputs in statics/worker_scaling/: item_level.csv (cache, one row per run and node), worker_ratios.csv (total and
alone, one row per service, statistic and w), within_run.csv (own, rest and idle, one row per service and ratio),
summary.csv (one row per service), summary.json (everything), report.html (worker_scaling_report.html filled with summary.json).
Run from notebooks/tsc/ with the venv: ../../.venv/bin/python worker_scaling_analysis.py [--recompute]
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from pegasus.load_independence import ratio_summary
from worker_scaling import CACHE, MAX_WORKERS, item_level, worker_effects

N_BOOT = 1000
SEED = 0
WITHIN_RUN = [f"own_ratio_{k}" for k in range(1, MAX_WORKERS)] + ["rest_ratio", "idle_ratio"]


def main(recompute: bool = False) -> None:
    level = item_level(recompute=recompute)
    level = level[level["throttled"] == 0]   # the folder holds unthrottled runs only; this keeps it explicit
    ratios = pd.concat([worker_effects(level, column, n_boot=N_BOOT, seed=SEED).assign(statistic=column)
                        for column in ("mean_ms", "p95_ms", "alone_ms")], ignore_index=True)
    within = pd.DataFrame([dict(service=service, ratio=column, **ratio_summary(runs[column], seed=SEED))
                           for service, runs in level.groupby("service") for column in WITHIN_RUN])

    wide = ratios.pivot_table(index="service", columns=["statistic", "workers"], values="ratio")
    wide.columns = [f"{statistic.removesuffix('_ms')}_w{w}" for statistic, w in wide.columns]
    own = within.pivot(index="service", columns="ratio", values="median")
    summary = wide.join(own)
    for w in range(2, MAX_WORKERS + 1):
        summary[f"speedup_w{w}"] = w / (summary[f"alone_w{w}"] * summary[f"own_ratio_{w - 1}"])
    per_worker = level.groupby(["service", "workers"])["utilization"].median().unstack()
    summary = summary.join(per_worker.add_prefix("utilization_w")).join(
        level.groupby("service").agg(configurations=("configuration", "nunique"), runs=("run_id", "size"),
                                     experiments=("experiment", lambda e: ", ".join(sorted(set(e)))))).reset_index()

    CACHE.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(CACHE / "worker_ratios.csv", index=False)
    within.to_csv(CACHE / "within_run.csv", index=False)
    summary.to_csv(CACHE / "summary.csv", index=False)
    clean = lambda frame: json.loads(frame.replace({np.nan: None}).to_json(orient="records"))
    (CACHE / "summary.json").write_text(json.dumps(dict(
        ratios=clean(ratios), within=clean(within), summary=clean(summary),
        totals=dict(runs=int(len(level)), configurations=int(level.groupby("service")["configuration"].nunique().sum()),
                    services=int(level["service"].nunique()), n_boot=N_BOOT, seed=SEED)), indent=1))

    # the report page: the template next to this script with the results in place of its data placeholder
    template = (Path(__file__).parent / "worker_scaling_report.html").read_text()
    (CACHE / "report.html").write_text(template.replace("/*DATA*/null", (CACHE / "summary.json").read_text()))

    show = summary.set_index("service")[[f"mean_w{MAX_WORKERS}", f"p95_w{MAX_WORKERS}", f"alone_w{MAX_WORKERS}",
                                         f"own_ratio_{MAX_WORKERS - 1}", f"speedup_w{MAX_WORKERS}"]]
    print(show.sort_values(f"mean_w{MAX_WORKERS}").round(3).to_string())
    print(f"\nwritten to {CACHE}")


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
