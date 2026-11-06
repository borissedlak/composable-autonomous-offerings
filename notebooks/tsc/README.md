# notebooks/tsc

Notebooks and helper code for the Pegasus paper (IEEE TSC): latency bounds for chains of services that are each
modeled on their own.

## Notebooks, in reading order

| Notebook | What it is |
|---|---|
| `1_method_synthetic.ipynb` | The method: GP model per service, node bounds, compositions, and the baselines of the method section. |
| `1_method_real.ipynb` | The method on real services: the same steps as `1_method_synthetic.ipynb` on the testbed campaign (training on run statistics, emulated queueing), without the sections that need a synthetic system. |
| `2a_base_case.ipynb` | The paper's evaluation (Section V), part 1, on synthetic or real data (`data_source`): the two data sources, the setup (A), the metrics and methods of B.1, and B.1 on all chains of the campaign (real data). The base case of B.1 that 2b to 2e build on (`b1_base`) is cached in `statics/evaluation_cache/` by the first notebook that needs it; `1_method_real.ipynb` derives the bounds and compositions on its chain. |
| `2b_evidence.ipynb` | Evaluation B.2 (varying evidence per service: on the chain `b1-vision-3-single`, the training runs of its services grow one service at a time under three schedules, all services together, most uncertain service first and least uncertain first; real data trains on the benchmarks and the other chains, `pegasus/evidence.py`). |
| `2c_depth.ipynb` | Evaluation B.3 (varying depth of a map-reduce application: fork-join trees of the services of B.1, why the latency of a tree is the maximum over its leaf-to-root paths, every method applied path by path, balanced trees) on the trees that the second campaign measured (`statics/results_milos_v2/b3/`; `tree_source`, synthetic trees as an option). |
| `2c_load.ipynb` | Evaluation B.4 (varying load and arrival pattern) on the chain of B.1 that the second campaign measured again under other loads and arrival patterns (`statics/results_milos_v2/b4/`; `data_source`, synthetic data as an option). |
| `2d_tails_and_percentiles.ipynb` | Evaluation B.5 (varying tail behaviors) and B.6 (varying percentiles). |
| `2e_finetuning.ipynb` | Evaluation C.1: fine-tuning a bound with end-to-end samples on one chain of B.1 (vision-1, fixed in `C1_CHAIN`, with runs of up to four workers per service held out): the bound of Pegasus Node with its load-aware models (the default of B.1, `load_aware_b4.py`), read from the cache of `2a_base_case.ipynb`, multiplied by one split-conformal factor (below the smallest number of samples at which it exists, the bound stays that of B.1), next to Quantile Sum and Fluxion-style trained on the samples. |
| `2f_optimization.ipynb` | Evaluation C.2: SLO-based composition planning on real data. Five scenarios per application (vision, audio, lidar) constrain cost (total workers), latency (Pegasus Node bound on the p95, load-aware models of B.1) and quality (data quality, detector model size) and name the capabilities they need (a functional requirement that excludes compositions without them); SLSQP on a continuous relaxation, as in the RASK paper, chooses the composition among the five chains `b1u-<application>-<i>` and its configuration, checked against an exhaustive search of the knob grid. Writes the deployment table `statics/slo_planning/c2_plans.csv`. Helpers in `slo_planning.py`. |
| `explain_1_why_chernoff_mgf.ipynb` | Why the bounds use moment generating functions and Chernoff bounds, and which alternatives exist. |
| `explain_2_why_poisson_is_loose.ipynb` | Why the SNC node bound is loose under Poisson arrivals. |
| `explain_3_how_pegasus_bounds_compose.ipynb` | The two versions of Pegasus (per-node union bound, SNC): the math from one queue upward, strengths and weaknesses. |
| `explain_4_how_fluxion_composes.ipynb` | How the Fluxion baseline composes latency percentiles. |
| `explain_5_what_allen_cunneen_inputs_cost.ipynb` | Why the GP Queue baseline does not hold in B.1: its queueing formula with predicted, true and pessimistic inputs. |
| `explain_6_when_load_changes_execution_times.ipynb` | When the load changes the execution time of a service (CPU throttling, warm-up after idle), shown on the testbed campaign, and two remedies: enough cores for the workers, and a benchmark in the slowest state. |
| `explain_7_which_inputs_predict_execution_time.ipynb` | Which inputs the GP of a service needs to predict the mean execution time of a run on the testbed campaign: quality knobs only, with cores per worker, with cores and parallelism, and each of the last two with the load; cross-validated MSE per service. |
| `explain_8_which_inputs_predict_execution_time_v2.ipynb` | The question of `explain_7` on the second campaign (`statics/results_milos_v2/`) with six input sets, and a partial-correlation test (as in PC structure learning) of how strongly each input is associated with the mean execution time; the same GPs with one length scale per input (ARD); why the load of that campaign cannot be judged. |
| `explain_9_does_provisioning_make_execution_times_load_independent.ipynb` | Whether the execution time depends on the load on the second campaign, split by provisioning (fewer cores than workers, one core per worker, more than one): throttling, the probe before each run against the run, the same configuration at different rates, and the state of a service when an item starts (next to other items, many recent arrivals, after a pause); whether over-provisioning ensures Assumption 3. |

The numbered notebooks belong to the paper; the `explain_` notebooks each answer one question and can be read on
their own. Start with `explain_3` for the fundamentals of the two bounds.

## Figure scripts

Scripts that draw one figure of the paper each. `draft_methodology_figure.py` is an earlier draft of the methodology
figure, kept for reference: the file name of its output ends in `_draft`. Run a script from `notebooks/tsc/` with the venv, e.g. `../../venv/Scripts/python problem_figure.py`; the
script writes a `.pdf` and a `.jpg` to `figures/tsc/`.

| Script | Figure | Data | Output in `figures/tsc/` |
|---|---|---|---|
| `problem_figure.py` | The two problems the paper addresses (Section II-A): per-service execution times do not add up to the end-to-end tail latency, and an accurate point prediction is not a guarantee. | Real: the chain of B.1 (`b1-vision-1-single`) of the second campaign, trained on the benchmarks as in B.1. (a) emulates queueing at the swept loads from the measured processing times, because the campaign ran every run at one load; (b) uses the measured latencies. | `problem_queueing`, `problem_prediction` |
| `methodology_figure.py` | The methodology overview, structured like Section IV: learn one model per service; turn it into a pessimistic execution-time model and the load $\phi_i(\theta)$ of every service; compose per service (Pegasus Node: Kingman's bound, the $d_i(\epsilon/K)$ added) or for the whole pipeline (Pegasus SNC). Every service has two workers, as in the earlier draft; the SNC bar is then an illustration only (the paper requires one worker for Pegasus SNC). | Synthetic: the services `s1`, `s2`, `s3` of `pegasus/data.py`, on a setting chosen for readability; an illustration, not a result of the evaluation. | `methodology_overview` |
| `draft_methodology_figure.py` | Earlier draft of the methodology overview: learn one model per service, bound each service as a queue, compose the service bounds for a pipeline that is never executed. | As `methodology_figure.py`. | `methodology_overview_draft` |

## Helper code: `pegasus/`

The notebooks and figure scripts only call these functions; no method is implemented inside a notebook. Import them
as `from pegasus.bounds import ...`, with the working directory at `notebooks/tsc/`.

| Module | Contents |
|---|---|
| `bounds.py` | The latency bounds: SNC node bound, martingale node bounds, propagated arrivals, network service curve. Pegasus Node is the closed-form martingale node bound with the external arrivals at every service; the propagated bounds (minimum spacing of the departures) are explored variants, not part of Pegasus. |
| `simulation.py` | The ground truth: simulator of a chain and of a fork-join tree of FIFO services, and the measured quantile with its confidence interval. |
| `trees.py` | Fork-join trees (B.3 in `2c_depth`): balanced topologies, their leaf-to-root paths, and the maximum over paths through which a chain composition extends to a tree. |
| `service_gp.py` | The GP model of one service's execution time, and the belief the bounds take from it. |
| `baselines.py` | Per-service baselines of the evaluation: GP Sum, DNN Sum, GP Queue (Allen-Cunneen formula), Quantile Union, Conformal, Oracle. |
| `baselines_chain.py` | Baselines of the method notebook that need runs of the chain: Deep GP, Fluxion-style, replay; independence tests. |
| `data.py` | Data of the evaluation: the seeded generator of synthetic services, the simulated chain data, the simulated fork-join trees of B.3 (`synthetic_tree_data`; the measured ones are in `real_data.py`), and the entry points for both data sources. |
| `real_data.py` | The real data source: reader of the testbed campaign in `statics/results_milos/`, and the emulation that replays its measured processing times through FIFO queues; the measured fork-join trees of the second campaign for B.3 (`real_tree_data`). |
| `evaluation_setup.py` | What the evaluation notebooks 2a to 2e share: the parameters (ε, confidence, seeds, ...), the options per data source, and the base case of B.1 (`b1_base`), computed once and cached in `statics/evaluation_cache/`; every experiment cell keeps its results there too (`cached`). A cache is recomputed when the parameters, the settings or the code of the experiment, or the computation code of `pegasus/` change (not when only a figure changes); `refresh=True` forces it. |
| `model_cache.py` | The shared cache of fitted per-service GPs (`statics/model_cache/`), used by `fit_services`: a fit is keyed by its training data and the training code (`service_gp.py`), so every experiment that trains a service on the same runs reuses one fit (B.1 in B.4 to C.2, the training states shared by the schedules of B.2), and a recomputed experiment retrains no GP unless `service_gp.py` or its data changed. Never deleted automatically; removing the folder only costs training time. |
| `evaluation.py` | What the experiments of the evaluation share: model fitting, the comparison of all methods on a chain and on a fork-join tree (`compare_tree_methods`), coverage, figures. |
| `exploration.py` | Experiment B.2b: sampling strategies that pick the next configuration to benchmark (random, Sobol sequence, and four acquisition functions on the GPs), for the synthetic services and for the measured runs of a real service. |
| `finetuning.py` | Experiment C.1: split-conformal calibration of a bound with end-to-end samples, and the baselines that are trained on runs of the chain itself (Quantile Sum, Fluxion-style). |
| `optimization.py` | Former experiment C.2 (no longer used by `2f_optimization.ipynb`; kept because removing a module of `pegasus/` recomputes every cached experiment): the SLO-fulfillment objective of a chain, and the optimizers that choose the next run with a latency method and refit its models after every evaluated run. |
| `load_dependence.py` | `explain_6`: how often a service of the campaign used up its CPU quota, how the processing time of an item depends on the state of its service (next to another item, back to back, after idle), and two simulators (a worker pool under a CPU quota, a service that is slower after idle). |
| `load_independence.py` | `explain_9`: per run and service of the second campaign, the provisioning class, the probe before the run, and within-run ratios of execution times by the state of the service at the start of an item; summaries of ratios over runs and the elasticity of the execution time with respect to the rate within configurations. |
Dependencies point one way: `load_independence` depends on `real_data`; `load_dependence` depends on `bounds`, `real_data` and `simulation`; `bounds` and `trees` depend on nothing here; `simulation` on `trees`; `service_gp` on nothing; `baselines` on
`service_gp`; `baselines_chain` on `baselines`, `bounds` and `service_gp`; `data` on `simulation` and `trees`; `real_data` on `data` and
`simulation`; `evaluation` on `baselines`, `bounds`, `data`, `service_gp` and `trees`; `exploration` on `data`, `evaluation`, `real_data`
and `service_gp`; `finetuning` on `baselines`, `baselines_chain`, `evaluation` and `service_gp`; `optimization` on
`evaluation`.

## Analysis scripts

| Script | Question | Output |
|---|---|---|
| `worker_scaling_analysis.py` | How much the execution time of each service depends on its number of workers, which Pegasus assumes it does not (no model receives the worker count): the ratio to 1 worker of the mean and the 95th percentile at 2 to 4 workers, split into concurrency within the service and a part that remains when an item runs alone, and the projected capacity at full load. Helpers in `worker_scaling.py` next to it, outside `pegasus/` on purpose: the cache key of the evaluation hashes every module of `pegasus/`, so a module there would recompute every cached experiment. Run from `notebooks/tsc/`; `--recompute` rereads the records. | `statics/worker_scaling/`: CSV tables, `summary.json`, and `report.html` (the template `worker_scaling_report.html` filled with the results) |
| `worker_scaling_coverage.py` | Whether that assumption costs coverage in B.1: reads the cached results of B.1 on all chains (`b1_all_chains` in `2a_base_case.ipynb`, never recomputes them) and splits coverage and tightness of Pegasus Node, Pegasus SNC, GP Queue and the Oracle by E, the slowdown of a held-out run's total execution time that the per-service ratios of `worker_scaling_analysis.py` predict; per chain on all held-out runs and on the multi-worker runs alone; the elasticity of tightness with respect to E. Needs the outputs of `worker_scaling_analysis.py`. Optional argument: the cache file. | `statics/worker_scaling/coverage_*.csv`, `tightness_elasticity.csv` |
| `load_aware_b4.py` | The load-aware variant of Pegasus Node, also imported by `2a_base_case.ipynb` (B.1 on all chains) and `2e_finetuning.ipynb` (C.1). As a script: B.4 with a variant of Pegasus Node whose GPs also receive the number of workers and log10 of the arrival rate, trained on the benchmarks plus the runs of the other chains of `b1/` and `b1-unthrottled/` (none of the chain of B.4), and an ablation on the same data with the current inputs; compared with the current Pegasus Node and the Oracle, which it reads from the cached results of `2c_load.ipynb` (never recomputes them). The training pool is widened at call time, not in `pegasus/`, for the reason above. Optional arguments: the folder of the evaluation cache, `--recompute`. | `statics/load_aware_b4/`: `variant_<sweep>.joblib` (cache), `comparison.csv` |
| `slo_planning.py` | C.2 in `2f_optimization.ipynb`: the knob grid and the node bounds of every composition at the arrival rate of its application (load-aware models of `load_aware_b4.py`, fitted without any run of the composition), the scenarios' objective and constraints, the SLSQP solver of the RASK paper and the exhaustive reference, the trade-off figure and the deployment table. Outside `pegasus/` for the cache key, like `load_aware_b4.py`. | `statics/slo_planning/`: `models_<chain>_<key>.joblib` (cache, about a minute per composition to fit), `c2_plans.csv`; `figures/tsc/c2_slo_plans_<application>` |

## Data and outputs outside this folder

| Path | Contents |
|---|---|
| `statics/results_milos/` | The testbed campaign that `real_data.py` reads: one folder per experiment (single services, chains, trees, deep chains). The real data source of `1_method_real.ipynb`, the evaluation notebooks (`2a` to `2e`), `explain_5` and `explain_6` needs this folder. The folder is not part of the repository: it is not committed, so a fresh clone does not contain it. |
| `statics/results_milos_v2/` | The second testbed campaign, the real data of the evaluation notebooks (`2a` to `2e`): `b1/` (the chains of B1, one worker per service), `b1-unthrottled/` (the same chains with one to four workers per service, none throttled), `benchmark/` (every service alone, the training data), `b3/` and `b3-heights/` (fork-join trees). The held-out runs of a chain are its runs in `b1/` and those of its twin in `b1-unthrottled/` with at most `MAX_WORKERS` = 4 workers per service, i.e. every measured worker count (`DEFAULT_CHAINS` in `pegasus/evaluation_setup.py`; the same cap holds in B.4). Not committed, like `statics/results_milos/`. |
| `statics/load_independence/results_milos_v2_b1/run_states.csv` | Cache of `run_states` (`load_independence.py`) for `explain_9`: one row per run and service of the second campaign. Recomputed from `statics/results_milos_v2/b1/` when missing. |
| `statics/worker_scaling/` | Outputs of `worker_scaling_analysis.py`; `item_level.csv` caches the per-run statistics (recomputed when missing). Not committed. |
| `figures/tsc/` | Outputs of the figure scripts. |

## Other files

`new_experiments.md` is the specification of the testbed campaign: which applications, chains, arrival rates and
arrival patterns the experiments B.1 to B.6, C.1 and C.2 need as real data. `figure_b3_experiment.py` draws the
schematic of the proposed experiment B3 that this file shows (`b3_experiment_setup` in `figures/tsc/`); the script
needs no data and is run like the figure scripts above.

## `archive/`

Older files that nothing else uses: `proofs.ipynb` and `why_arrival_difficult.ipynb` (early standalone notebooks). They
are kept for reference and are not maintained; relative paths inside them refer to their former location one folder up.
