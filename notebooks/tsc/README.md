# notebooks/tsc: Pegasus (IEEE TSC)

This folder holds the code of the paper on **Pegasus**, a method that gives an upper bound on the tail latency
(for example the 95th percentile, p95) of a chain of services that was never run as a whole. Pegasus learns one
model per service from benchmarks of that service alone: a Gaussian process (GP) that predicts the execution time
of the service from its configuration (quality knobs, number of workers, arrival rate). It then turns each model
into a pessimistic execution-time model and composes the per-service models with a queueing bound (Kingman's bound,
or stochastic network calculus for Pegasus SNC) into a bound on the end-to-end latency of the chain.

The folder contains:

- **The evaluation (Section V of the paper):** six notebooks, `2_B1_base_case.ipynb` to `2_C2_optimization.ipynb`,
  one per experiment, all on the measured data of a physical testbed. These are the notebooks that reproduce the
  paper's results; see [Running the experiments](#running-the-experiments-b1-to-c2).
- **The method:** `1_method_synthetic.ipynb` and `1_method_real.ipynb` walk through the steps of Pegasus and its
  baselines on one chain.
- **The library:** the package `pegasus/` implements every method, baseline, data reader and cache; the notebooks
  only call it.
- **Supporting material:** figure scripts for Sections II and IV and the `explain_`
  notebooks, which each answer one question that came up while developing the method. These are listed at the end
  of this file.

## Setup

Requires **Python 3.12 or newer**. Run the setup and download commands below **from the repository root**. The
experiments themselves run from `notebooks/tsc/` with the project's virtual environment, `venv/` at the repository
root (`../../venv/Scripts/python` on Windows, `../../venv/bin/python` on Linux and macOS):

```bash
python3.12 -m venv venv                          # at the repository root; any Python >= 3.12 works
venv/bin/python -m pip install -r requirements.txt jupyterlab
```

The testbed itself (Docker containers, Prometheus) is not needed: every experiment reads recorded measurements.

## Downloading the testbed measurements

The experiments read the measurements of the testbed campaign in `statics/results_milos_v2/` at the repository
root. The folder is too large for the repository and is not committed, so a fresh clone does not contain it.

The measurements are a .zip archive at **https://tucloud.tuwien.ac.at/index.php/s/p8gj3payEFCdts7** (appending
`/download` to the link fetches the file directly). Download it and extract it at the repository
root, into a new folder `statics/results_milos_v2/` (the archive holds `b1/`, `b4/`, ... directly, without a top folder).
**Run these commands from the repository root**, not from `notebooks/tsc/`. The commands are chained with `&&`, so
the archive is deleted only after it was extracted successfully:

```bash
# Linux / macOS
mkdir -p statics/results_milos_v2 && \
curl -L -o results_milos_v2.zip https://tucloud.tuwien.ac.at/index.php/s/p8gj3payEFCdts7/download && \
unzip results_milos_v2.zip -d statics/results_milos_v2/ && \
rm results_milos_v2.zip
```

```bat
:: Windows (cmd or PowerShell; curl.exe and tar ship with Windows 10 and later. Write curl.exe, not curl:
:: in Windows PowerShell, curl is an alias of Invoke-WebRequest, which does not accept these options)
mkdir statics\results_milos_v2 && curl.exe -L -o results_milos_v2.zip https://tucloud.tuwien.ac.at/index.php/s/p8gj3payEFCdts7/download && tar -xf results_milos_v2.zip -C statics\results_milos_v2 && del results_milos_v2.zip
```

To check the extraction, list `statics/results_milos_v2/b1` (`ls statics/results_milos_v2/b1` on Linux / macOS,
`dir statics\results_milos_v2\b1` on Windows): it must contain files, otherwise the archive was not extracted.

The result must have this layout:

```
statics/results_milos_v2/
├── benchmark/w1/ … w4/   every service alone, with one to four workers: the training data of every method
├── b1/                   the chains of B.1, one worker per service: held-out runs
├── b1-unthrottled/       the same chains with one to four workers per service: held-out runs
├── b3-vm/                the fork-join trees of B.3 and their own benchmarks (measured on a second machine)
├── b4/                   the chain of B.1 under other loads and arrival patterns (B.4)
└── c2/                   the testbed runs of the deployed plans of C.2 (C.2d)
```

The readers in `pegasus/real_data.py` locate the folder relative to the repository, so no path needs to be set.
`statics/results_milos/`, the first, smaller campaign, is read only by `explain_5` to `explain_7` and is not needed
for the experiments.

## Running the experiments B.1 to C.2

Change into `notebooks/tsc/` (`cd notebooks/tsc`), then open the notebooks in Jupyter (`../../venv/bin/jupyter lab`,
or the venv as kernel in an IDE) with `notebooks/tsc/` as the working directory, and run each one from top to bottom. To run one without opening it:

```bash
../../venv/bin/jupyter nbconvert --to notebook --execute --inplace 2_B1_base_case.ipynb
```

Run `2_B1_base_case.ipynb` first. The notebooks after it build on its base case and read it from the cache; a
notebook whose cache is missing computes the base case itself, so any order works, but the first one then takes
longer.

| Notebook | Section | What the experiment measures | Data |
|---|---|---|---|
| `2_B1_base_case.ipynb` | A, B.1 | The setup (training data, held-out runs, methods, metrics) and the base case: coverage and tightness of Pegasus and every baseline on every chain of the campaign. Coverage is the share of held-out runs whose measured p95 lies below the predicted bound; tightness is the ratio of the bound to the measured p95. | `benchmark/`, `b1/`, `b1-unthrottled/` |
| `2_B2_evidence.ipynb` | B.2 | How coverage and tightness change with the number of training runs per service, on one chain, when the services receive more runs together, most uncertain first, or least uncertain first. | as B.1 |
| `2_B3_depth.ipynb` | B.3 | Map-reduce applications of growing depth: balanced fork-join trees, bounded path by path (the latency of a tree is the maximum over its leaf-to-root paths). | `b3-vm/` |
| `2_B4_load.ipynb` | B.4 | The chain of B.1 under growing utilization and under arrival patterns other than Poisson, and whether the methods detect the runs that overloaded. | `b4/` |
| `2_C1_finetuning.ipynb` | C.1 | Fine-tuning the bound of Pegasus with a few end-to-end samples of the chain (split-conformal calibration), against baselines that train on those samples alone. | as B.1 |
| `2_C2_optimization.ipynb` | C.2 | SLO-based composition planning: for five scenarios per application, a grid search picks the composition and configuration that meet the client's constraints on cost, latency and quality; C.2d compares the plans with their testbed runs. | as B.1, `c2/` |

What to know before running:

- **Parameters.** Every notebook shares the parameters in `pegasus/evaluation_setup.py`: the seeds (`seeds`; every
  experiment is repeated once per seed, and the seed fixes which training runs are drawn), the number of training
  runs per service (`N_TRAIN`), the violation probability ε and the confidence γ. Changing one of them changes the
  results.
- **Data source.** The first code cell of B.1 to B.4 sets `data_source` (`tree_source` in B.3) to `"real"`, the
  setting of the paper. `"synthetic"` runs the same experiment on generated services and a simulated chain of
  queues, which needs no download.
- **Caches.** Results are cached under `statics/cache/` (`*.joblib`, never committed): `evaluation/` holds the result
  of every experiment cell, `models/` the fitted GPs, `runs/` the values parsed from the raw measurement files. A
  cached result is recomputed automatically when the parameters, the settings or the computation code in `pegasus/`
  change (not when only a figure changes); `refresh=True` in a cell forces it. The first run on a fresh clone fits
  every model and is slow; later runs load from the cache. Deleting `statics/cache/` only costs computation time.
- **B.4** imports two scripts next to the notebooks, `utilization_one.py` (the point at utilization 1, evaluated on
  its own sustained runs) and `overload_detection.py` (every run at high load, including the overloaded ones); both
  keep their own cache in `statics/load_aware_b4/`.
- **C.2** uses `slo_planning.py` for the knob grid, the scenarios and the grid search. Its results are predicted
  bounds of inferred configurations, not measurements, except in C.2d. It writes the plans to
  `statics/slo_planning/c2_plans_new.csv`; with `WRITE_DEPLOYED_FILES = False` (the default) it keeps the figures of
  the plans that were deployed on the testbed (see `statics/slo_planning/README.md`).
- **Outputs.** The figures of the paper go to `figures/tsc/experimental-results/` (`save_figure` in
  `pegasus/figures.py`); the tables are printed in the notebooks.

## Further material

### Method and explanation notebooks

| Notebook | What it is |
|---|---|
| `0_GP_Demo.ipynb` | The GP model of one service (the image denoiser) as a 3D surface with its 95% band and training runs. |
| `1_method_synthetic.ipynb` | The method step by step: GP model per service, node bounds, compositions, and the baselines of the method section, on synthetic services. |
| `1_method_real.ipynb` | The same steps on one real chain of the campaign, without the sections that need a synthetic system. |
| `explain_1_why_chernoff_mgf.ipynb` | Why the bounds use moment generating functions and Chernoff bounds, and which alternatives exist. |
| `explain_2_why_poisson_is_loose.ipynb` | Why the SNC node bound is loose under Poisson arrivals. |
| `explain_3_how_pegasus_bounds_compose.ipynb` | The two versions of Pegasus (per-node union bound, SNC): the math from one queue upward. Start here for the fundamentals. |
| `explain_4_how_fluxion_composes.ipynb` | How the Fluxion baseline composes latency percentiles. |
| `explain_5_what_allen_cunneen_inputs_cost.ipynb` | Why the GP Queue baseline (Allen-Cunneen formula) does not hold in B.1. |
| `explain_6_when_load_changes_execution_times.ipynb` | When the load changes the execution time of a service (CPU throttling, warm-up after idle), and two remedies. |
| `explain_7_which_inputs_predict_execution_time.ipynb` | Which inputs the GP of a service needs (knobs, cores, parallelism, load), on the first campaign. |
| `explain_8_which_inputs_predict_execution_time_v2.ipynb` | The same question on the second campaign, with a partial-correlation test per input. |
| `explain_9_does_provisioning_make_execution_times_load_independent.ipynb` | Whether over-provisioning cores makes the execution time independent of the load. |

### Figure scripts

Each script draws one figure of the paper and writes a `.pdf` and a `.jpg` to `figures/tsc/`; run it from
`notebooks/tsc/`, for example `../../venv/bin/python 0_problem_figure.py`.

| Script | Figure |
|---|---|
| `0_problem_figure.py` | Section II-A: per-service execution times do not add up to the end-to-end tail latency (`problem_queueing`), and an accurate point prediction is not a guarantee (`problem_prediction`). Real data: the chain of B.1. |
| `0_methodology_figure.py` | Section IV: the methodology overview (`methodology_overview`), on synthetic services chosen for readability. |

### The library `pegasus/`

Import as `from pegasus.bounds import ...` with `notebooks/tsc/` as the working directory.

| Module | Contents |
|---|---|
| `bounds.py` | The latency bounds: SNC node bound, martingale node bounds (Pegasus), propagated arrivals, network service curve. |
| `simulation.py` | The ground truth of synthetic data: simulator of a chain and of a fork-join tree of FIFO queues. |
| `trees.py` | Fork-join trees (B.3): balanced topologies and their leaf-to-root paths. |
| `service_gp.py` | The GP model of one service's execution time, and the pessimistic belief the bounds take from it. |
| `baselines.py` | Per-service baselines: GP Sum, DNN Sum, GP Queue (Allen-Cunneen formula), Quantile Union, Conformal, Oracle. |
| `baselines_chain.py` | Baselines that need runs of the chain: Deep GP, Fluxion-style, replay; independence tests. |
| `data.py` | Synthetic data, and the load-aware model inputs (`load_aware`: knobs, number of workers, log10 of the arrival rate). |
| `real_data.py` | Readers of the testbed campaigns and the training pool of every real-data experiment (`WORKER_TRAINING`). |
| `run_cache.py`, `model_cache.py` | Caches of parsed run files (`statics/cache/runs/`) and of fitted GPs (`statics/cache/models/`). |
| `evaluation_setup.py` | Shared parameters of the evaluation notebooks, the base case of B.1 (`b1_base`) and the result cache (`cached`). |
| `evaluation.py` | Model fitting, the comparison of all methods on a chain or tree, every method on the same information (`same_information_methods`), coverage. |
| `evidence.py`, `exploration.py` | B.2: the schedules of training runs; sampling strategies for the next configuration to benchmark. |
| `finetuning.py` | C.1: split-conformal calibration and the baselines trained on end-to-end samples. |
| `optimization.py` | Former C.2, no longer used; kept because removing a module of `pegasus/` invalidates every cached experiment. |
| `figures.py` | The style of the paper figures and the export to `figures/tsc/experimental-results/`. |
| `input_sets.py`, `load_dependence.py`, `load_independence.py` | Helpers of `explain_6` to `explain_9`. |

### `archive/`

Earlier notebooks that nothing else uses (`proofs.ipynb`, `why_arrival_difficult.ipynb`, and
`2d_tails_and_percentiles.ipynb` with B.5 and B.6, which the paper does not use). They are kept for reference and
are not maintained.
