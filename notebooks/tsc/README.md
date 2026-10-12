# notebooks/tsc

Notebooks and helper code for the Pegasus paper (IEEE TSC): latency bounds for chains of services that are each
modeled on their own.

## Notebooks, in reading order

| Notebook | What it is |
|---|---|
| `1_method_synthetic.ipynb` | The method: GP model per service, node bounds, compositions, and the baselines of the method section. |
| `1_method_real.ipynb` | The method on real services: the same steps as `1_method_synthetic.ipynb` on the testbed campaign (training on run statistics, emulated queueing), without the sections that need a synthetic system. |
| `2_evaluation.ipynb` | The paper's evaluation (Section V): setup, coverage and tightness experiments B.1 to B.6 and the iterative experiments C.1 and C.2 on synthetic or real data (`data_source`). |
| `explain_1_why_chernoff_mgf.ipynb` | Why the bounds use moment generating functions and Chernoff bounds, and which alternatives exist. |
| `explain_2_why_poisson_is_loose.ipynb` | Why the SNC node bound is loose under Poisson arrivals. |
| `explain_3_how_pegasus_bounds_compose.ipynb` | The two versions of Pegasus (per-node union bound, SNC): the math from one queue upward, strengths and weaknesses. |
| `explain_4_how_fluxion_composes.ipynb` | How the Fluxion baseline composes latency percentiles. |
| `explain_5_what_kingman_inputs_cost.ipynb` | Why the Kingman baseline does not hold in B.1: its queueing formula with predicted, true and pessimistic inputs. |

The numbered notebooks belong to the paper; the `explain_` notebooks each answer one question and can be read on
their own. Start with `explain_3` for the fundamentals of the two bounds.

## Helper code: `pegasus/`

The notebooks only call these functions; no method is implemented inside a notebook. Import them as
`from pegasus.bounds import ...`, with the notebook's working directory at `notebooks/tsc/`.

| Module | Contents |
|---|---|
| `bounds.py` | The latency bounds: SNC node bound, martingale node bounds, propagated arrivals, network service curve. |
| `simulation.py` | The ground truth: simulator of a chain of FIFO services, and the measured quantile with its confidence interval. |
| `service_gp.py` | The GP model of one service's execution time, and the belief the bounds take from it. |
| `baselines.py` | Per-service baselines of the evaluation: GP Sum, DNN Sum, Kingman formula, Quantile Union, Conformal, Oracle. |
| `baselines_chain.py` | Baselines of the method notebook that need runs of the chain: Deep GP, Fluxion-style, replay; independence tests. |
| `data.py` | Data of the evaluation: the seeded generator of synthetic services, the simulated chain data, and the entry points for both data sources. |
| `real_data.py` | The real data source: reader of the testbed campaign in `statics/results_milos/`, and the emulation that replays its measured processing times through FIFO queues. |
| `evaluation.py` | What the experiments of the evaluation share: model fitting, the comparison of all methods, coverage, figures. |

Dependencies point one way: `bounds` and `simulation` depend on nothing here; `service_gp` on nothing; `baselines` on
`service_gp`; `baselines_chain` on `baselines`, `bounds` and `service_gp`; `data` on `simulation`; `real_data` on `data` and
`simulation`; `evaluation` on `baselines`, `bounds` and `service_gp`.

## `archive/`

Older files that nothing else uses: `proofs.ipynb` and `why_arrival_difficult.ipynb` (early standalone notebooks). They
are kept for reference and are not maintained; relative paths inside them refer to their former location one folder up.
