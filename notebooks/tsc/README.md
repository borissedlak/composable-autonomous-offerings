# notebooks/tsc

Notebooks and helper code for the Pegasus paper (IEEE TSC): latency bounds for chains of services that are each
modeled on their own.

## Notebooks, in reading order

| Notebook | What it is |
|---|---|
| `1_method_synthetic.ipynb` | The method: GP model per service, node bounds, compositions, and the baselines of the method section. |
| `1_method_real.ipynb` | The method on real services: the same steps as `1_method_synthetic.ipynb` on the testbed campaign (training on run statistics, emulated queueing), without the sections that need a synthetic system. |
| `2_evaluation.ipynb` | The paper's evaluation (Section V) on synthetic or real data (`data_source`): setup (A), the coverage and tightness experiments B.1 to B.6 including B.2b (which configuration to benchmark next), and the iterative experiments C.1 (fine-tuning a bound with end-to-end samples) and C.2 (choosing a faster configuration of a chain). A last section holds real-data results that do not fit the paper's structure. |
| `explain_1_why_chernoff_mgf.ipynb` | Why the bounds use moment generating functions and Chernoff bounds, and which alternatives exist. |
| `explain_2_why_poisson_is_loose.ipynb` | Why the SNC node bound is loose under Poisson arrivals. |
| `explain_3_how_pegasus_bounds_compose.ipynb` | The two versions of Pegasus (per-node union bound, SNC): the math from one queue upward, strengths and weaknesses. |
| `explain_4_how_fluxion_composes.ipynb` | How the Fluxion baseline composes latency percentiles. |
| `explain_5_what_allen_cunneen_inputs_cost.ipynb` | Why the Allen-Cunneen baseline does not hold in B.1: its queueing formula with predicted, true and pessimistic inputs. |
| `explain_6_when_load_changes_execution_times.ipynb` | When the load changes the execution time of a service (CPU throttling, warm-up after idle), shown on the testbed campaign, and two remedies: enough cores for the workers, and a benchmark in the slowest state. |
| `explain_7_which_inputs_predict_execution_time.ipynb` | Which inputs the GP of a service needs to predict the mean execution time of a run on the testbed campaign: quality knobs only, with cores per worker, with cores and parallelism, and each of the last two with the load; cross-validated MSE per service. |
| `explain_8_which_inputs_predict_execution_time_v2.ipynb` | The question of `explain_7` on the second campaign (`statics/results_milos_v2/`) with six input sets, and a partial-correlation test (as in PC structure learning) of how strongly each input is associated with the mean execution time; the same GPs with one length scale per input (ARD); why the load of that campaign cannot be judged. |
| `explain_9_does_provisioning_make_execution_times_load_independent.ipynb` | Whether the execution time depends on the load on the second campaign, split by provisioning (fewer cores than workers, one core per worker, more than one): throttling, the probe before each run against the run, the same configuration at different rates, and the state of a service when an item starts (next to other items, many recent arrivals, after a pause); whether over-provisioning ensures Assumption 3. |

The numbered notebooks belong to the paper; the `explain_` notebooks each answer one question and can be read on
their own. Start with `explain_3` for the fundamentals of the two bounds.

## Figure scripts

Scripts that draw one figure of the paper each. Both are drafts: the file names of their outputs end in `_draft`.
Run a script from `notebooks/tsc/` with the venv, e.g. `../../venv/Scripts/python draft_problem_figure.py`; the
script writes a `.pdf` and a `.jpg` to `figures/tsc/`.

| Script | Figure | Data | Output in `figures/tsc/` |
|---|---|---|---|
| `draft_problem_figure.py` | The two problems the paper addresses (Section II-A): per-service execution times do not add up to the end-to-end tail latency, and an accurate point prediction is not a guarantee. | Real: chain `04e-chain-qr-cv-face` of the testbed campaign, with emulated queueing as in `2_evaluation.ipynb`. | `problem_overview_draft` |
| `draft_methodology_figure.py` | The methodology overview (Fig. 1): learn one model per service, bound each service as a queue, compose the service bounds for a pipeline that is never executed. | Synthetic: the services `s1`, `s2`, `s3` of `pegasus/data.py`, on a setting chosen for readability; an illustration, not a result of the evaluation. | `methodology_overview_draft` |

## Helper code: `pegasus/`

The notebooks and figure scripts only call these functions; no method is implemented inside a notebook. Import them
as `from pegasus.bounds import ...`, with the working directory at `notebooks/tsc/`.

| Module | Contents |
|---|---|
| `bounds.py` | The latency bounds: SNC node bound, martingale node bounds, propagated arrivals, network service curve. |
| `simulation.py` | The ground truth: simulator of a chain of FIFO services, and the measured quantile with its confidence interval. |
| `service_gp.py` | The GP model of one service's execution time, and the belief the bounds take from it. |
| `baselines.py` | Per-service baselines of the evaluation: GP Sum, DNN Sum, Allen-Cunneen formula, Quantile Union, Conformal, Oracle. |
| `baselines_chain.py` | Baselines of the method notebook that need runs of the chain: Deep GP, Fluxion-style, replay; independence tests. |
| `data.py` | Data of the evaluation: the seeded generator of synthetic services, the simulated chain data, and the entry points for both data sources. |
| `real_data.py` | The real data source: reader of the testbed campaign in `statics/results_milos/`, and the emulation that replays its measured processing times through FIFO queues. |
| `evaluation.py` | What the experiments of the evaluation share: model fitting, the comparison of all methods, coverage, figures. |
| `exploration.py` | Experiment B.2b: sampling strategies that pick the next configuration to benchmark (random, Sobol sequence, and four acquisition functions on the GPs), for the synthetic services and for the measured runs of a real service. |
| `finetuning.py` | Experiment C.1: split-conformal calibration of a bound with end-to-end samples, and the baselines that are trained on runs of the chain itself (Quantile Sum, Fluxion-style). |
| `optimization.py` | Experiment C.2: the SLO-fulfillment objective of a chain, and the optimizers that choose the next run with a latency method and refit its models after every evaluated run. |
| `load_dependence.py` | `explain_6`: how often a service of the campaign used up its CPU quota, how the processing time of an item depends on the state of its service (next to another item, back to back, after idle), and two simulators (a worker pool under a CPU quota, a service that is slower after idle). |
| `load_independence.py` | `explain_9`: per run and service of the second campaign, the provisioning class, the probe before the run, and within-run ratios of execution times by the state of the service at the start of an item; summaries of ratios over runs and the elasticity of the execution time with respect to the rate within configurations. |
Dependencies point one way: `load_independence` depends on `real_data`; `load_dependence` depends on `bounds`, `real_data` and `simulation`; `bounds` and `simulation` depend on nothing here; `service_gp` on nothing; `baselines` on
`service_gp`; `baselines_chain` on `baselines`, `bounds` and `service_gp`; `data` on `simulation`; `real_data` on `data` and
`simulation`; `evaluation` on `baselines`, `bounds` and `service_gp`; `exploration` on `data`, `evaluation`, `real_data`
and `service_gp`; `finetuning` on `baselines`, `baselines_chain`, `evaluation` and `service_gp`; `optimization` on
`evaluation`.

## Data and outputs outside this folder

| Path | Contents |
|---|---|
| `statics/results_milos/` | The testbed campaign that `real_data.py` reads: one folder per experiment (single services, chains, trees, deep chains). The real data source of `1_method_real.ipynb`, `2_evaluation.ipynb`, `explain_5`, `explain_6` and `draft_problem_figure.py` needs this folder. The folder is not part of the repository: it is not committed, so a fresh clone does not contain it. |
| `statics/load_independence/results_milos_v2/run_states.csv` | Cache of `run_states` (`load_independence.py`) for `explain_9`: one row per run and service of the second campaign. Recomputed from `statics/results_milos_v2/` when missing. |
| `figures/tsc/` | Outputs of the figure scripts. |

## Other files

`new_experiments.md` is the specification of the testbed campaign: which applications, chains, arrival rates and
arrival patterns the experiments B.1 to B.6, C.1 and C.2 need as real data. `figure_b3_experiment.py` draws the
schematic of the proposed experiment B3 that this file shows (`b3_experiment_setup` in `figures/tsc/`); the script
needs no data and is run like the figure scripts above.

## `archive/`

Older files that nothing else uses: `proofs.ipynb` and `why_arrival_difficult.ipynb` (early standalone notebooks). They
are kept for reference and are not maintained; relative paths inside them refer to their former location one folder up.
