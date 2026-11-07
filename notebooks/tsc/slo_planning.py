"""Experiment C.2 of `2f_optimization.ipynb`: choosing the service composition of an application, and its
configuration, that meets the SLOs of a client, with a numerical solver on the predictions of Pegasus Node.

Setting. An application type (vision, audio, lidar) has five service compositions, the chains b1u-<application>-<i>
of the second campaign. A configuration of a composition sets
- q, the data quality that every service of the composition processes (knob data_quality; the image width in pixels
  for vision, the quality level of the campaign for audio and lidar),
- w_k, the number of workers of service k (knob parallelism, 1 to 4), and
- m_k, the model size of every detector that has one (knob model_size, 1 or 2; vision only).
The client sends items at one fixed arrival rate (items/s, Poisson arrivals) per application.

Three SLO dimensions, each one number per configuration:
- cost: the total number of workers of the composition, sum_k w_k;
- latency: the Pegasus Node bound (s) on the p95 of the end-to-end latency at the arrival rate of the application,
  with the load-aware models (load_aware_b4.py, the default of B.1);
- quality: the data quality q, and, for vision, the smallest model size among the detectors of the composition.
A scenario poses one to three of these as constraints (max workers, max latency, min quality) and names one goal that
the solver optimizes among the configurations that meet them (min cost, min latency or max quality). Ties of the goal
are broken by the other two dimensions with a small weight (TIE_WEIGHT).

Prediction. The models of every composition are those of B.1: every service is trained on its benchmarks with one to
four workers (worker_benchmark_pool in pegasus/real_data.py), without any run of a chain, so that a composition is
planned as if it had never run. Pegasus Node bounds every service on its own (with the external
arrival rate and epsilon = alpha / K for K services), so the end-to-end bound is the sum of the node bounds. A node bound depends only on the knobs of its own service: `composition` computes it once on the grid
of every service's knobs, and the bound of any configuration of the composition is a sum of table entries.

Solver. As in the RASK paper (SummerSOC, agent/components/Optimizer_v2.py): SLSQP (scipy.optimize.minimize) on the
knobs normalized to [0, 1], restarted from RESTARTS random points per composition. SLSQP needs continuous knobs: the
node bounds are linearly interpolated between the grid points, and an overloaded node (no finite bound) gets
OVERLOAD_LATENCY instead. Every SLSQP solution is rounded to the grid and repaired: among the grid configurations
within one grid step of it in every knob, the best that meets the constraints is kept. The plan is the best
composition. Solver "exhaustive" enumerates every grid configuration of every composition instead, as a reference that
tells whether SLSQP found the optimum of the grid.

Units: latencies in seconds, rates in items/s.
"""
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

import load_aware_b4 as la
import pegasus.real_data as real_data
from pegasus.bounds import martingale_closed_form_node_bound
from pegasus.evaluation import fit_services
from pegasus.evaluation_setup import alpha, confidence, coverage_check, seed
from pegasus.service_gp import run_statistics_belief

APPLICATIONS = {app: [f"b1u-{app}-{i}" for i in range(1, 6)] for app in ("vision", "audio", "lidar")}
CACHE = real_data.CAMPAIGN.parent / "cache" / "slo_planning"   # statics/cache/slo_planning/: the fitted models of every composition (joblib)
PLANS = real_data.CAMPAIGN.parent / "slo_planning"             # statics/slo_planning/: the plans to deploy (csv)
RESTARTS = 10              # SLSQP starting points per composition, as run_optimizer_multi of the RASK paper
OVERLOAD_LATENCY = 10.0    # s; latency that the continuous relaxation gives an overloaded node, so that SLSQP moves away from it
TIE_WEIGHT = 1e-3          # weight of the two other dimensions in the objective, which only breaks ties of the goal
GOALS = ("min cost", "min latency", "max quality")
# Functional capability -> the service (image, without version) that provides it. A scenario requires capabilities;
# a composition meets the requirement if it contains a service for every one of them.
CAPABILITIES = {
    "object detection": "cv-analyzer", "face detection": "face-detector", "QR codes": "qr-detector", "motion detection": "motion-detector",
    "edge detection": "edge-detector", "image denoising": "image-denoiser", "image compression": "image-compressor", "image transformation": "image-transformer",
    "noise suppression": "audio-denoiser", "resampling": "audio-resampler", "audio features": "audio-features", "voice activity": "voice-detector",
    "pitch tracking": "pitch-tracker", "transcription": "speech-transcriber", "audio encoding": "audio-encoder",
    "downsampling": "pc-downsampler", "ground removal": "ground-remover", "clustering": "pc-clusterer", "normal estimation": "normal-estimator",
    "registration": "pc-registrar", "visualization": "pc-visualizer", "point cloud encoding": "pc-encoder",
}


@dataclass
class Scenario:
    """The SLOs of one client. None: no constraint on that dimension."""
    name: str
    goal: str                          # one of GOALS
    max_workers: int = None            # cost: largest total number of workers of the composition
    max_latency: float = None          # latency: largest Pegasus bound on the p95 end-to-end latency (s)
    min_quality: float = None          # quality: smallest data quality q
    min_model: int = None              # quality (vision): smallest model size of every detector of the composition
    note: str = ""                     # who would pose these SLOs
    requires: tuple = ()               # functional requirement: capabilities (keys of CAPABILITIES) the composition must provide

    def dimensions(self) -> list:
        """The SLO dimensions that the scenario constrains."""
        return [name for name, value in (("cost", self.max_workers), ("latency", self.max_latency),
                                         ("quality", self.min_quality if self.min_model is None else self.min_model)) if value is not None]

    def constraints(self) -> str:
        parts = [f"≤ {self.max_workers} workers" if self.max_workers is not None else None,
                 f"p95 ≤ {1e3 * self.max_latency:.0f} ms" if self.max_latency is not None else None,
                 f"quality ≥ {self.min_quality:g}" if self.min_quality is not None else None,
                 f"model ≥ {self.min_model}" if self.min_model is not None else None]
        return ", ".join(part for part in parts if part)


# ------------------------------------------------------------------------------
# Compositions and their node bounds
# ------------------------------------------------------------------------------
@dataclass
class Composition:
    """One composition with its knob grid and its node bounds at the arrival rate of the application."""
    experiment: str
    path: list                         # nodes of the chain, in order
    services: dict                     # node -> service (image)
    rate: float                        # arrival rate (items/s) of the tables
    qualities: np.ndarray              # grid of the shared data quality
    workers: np.ndarray                # grid of the workers of every service
    model_nodes: list                  # nodes with a model_size knob
    models: np.ndarray                 # grid of the model size
    tables: dict = field(repr=False)   # node -> node bound (s), shape (qualities, workers[, models]); inf if overloaded
    extrapolated: list = field(default_factory=list)   # nodes whose training runs never had the arrival rate `rate`

    @property
    def n_dims(self) -> int:
        return 1 + len(self.path) + len(self.model_nodes)

    def grid_axes(self) -> list:
        """The grid values of every knob of the solver vector: q, w_k per node, m_k per model node."""
        return [self.qualities] + [self.workers] * len(self.path) + [self.models] * len(self.model_nodes)

    def latency(self, q_index: np.ndarray, w_index: np.ndarray, m_index: np.ndarray) -> np.ndarray:
        """End-to-end bound (s) at grid configurations given by grid indices: q_index (n,), w_index (n, nodes),
        m_index (n, model nodes)."""
        total = np.zeros(len(q_index))
        for k, node in enumerate(self.path):
            table = self.tables[node]
            if node in self.model_nodes:
                total = total + table[q_index, w_index[:, k], m_index[:, self.model_nodes.index(node)]]
            else:
                total = total + table[q_index, w_index[:, k]]
        return total


def chain_space(experiment: str) -> dict:
    """The knob space of a composition as the campaign sampled it (summary.json of the experiment)."""
    return json.loads((real_data.experiment_folder(experiment) / "summary.json").read_text())["workflow"]["space"]


def _cache_file(experiment: str, model_seed: int) -> Path:
    key = hashlib.sha256(f"{experiment}|{model_seed}|{alpha}|{confidence}|{la.code_fingerprint()}".encode()).hexdigest()[:12]
    return CACHE / f"models_{experiment}_{key}.joblib"


def fit_composition(experiment: str, model_seed: int = seed, refresh: bool = False) -> dict:
    """The load-aware models of every service of a composition (as in B.1, without any run of a chain), its
    path and the input names of every service; cached in CACHE (about a minute per composition)."""
    file = _cache_file(experiment, model_seed)
    if file.exists() and not refresh:
        return joblib.load(file)
    with la.extended_training():
        data = real_data.real_chain_data(experiment, level=1 - alpha, coverage_check=coverage_check, seed=[model_seed, 0],
                                         training_campaign=la.EXTENDED, n_train=None)
        names = real_data.feature_names(experiment, la.EXTENDED)
    aware = la.load_aware(data)
    models = fit_services(aware, aware.x_train, data.y_train, model_seed=model_seed, confidence=confidence, with_dnn=False)
    fitted = dict(path=list(data.path), services=real_data.chain_path(experiment)[1], names=names, gp=models["gp"],
                  log_rates={node: (data.x_train_load[node][:, -1].min(), data.x_train_load[node][:, -1].max()) for node in data.path})
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, file)
    return fitted


def composition(experiment: str, rate: float, model_seed: int = seed, refresh: bool = False) -> Composition:
    """A composition with the node bounds of every service on its knob grid at the arrival rate `rate` (items/s)."""
    fitted, space = fit_composition(experiment, model_seed, refresh), chain_space(experiment)
    path = fitted["path"]
    # services whose training runs never had this arrival rate: the GP extrapolates their load input
    extrapolated = [node for node in path if not fitted["log_rates"][node][0] <= np.log10(rate) <= fitted["log_rates"][node][1]]
    qualities = np.array(space["shared.data_quality"], dtype=float)
    workers = np.arange(1, 5)
    model_nodes = [node for node in path if f"{node}.model_size" in space]
    models = np.array(space[model_nodes[0] + ".model_size"]) if model_nodes else np.array([1])
    epsilon = alpha / len(path)
    tables = {}
    for node in path:
        names = fitted["names"][node]
        axes = [qualities, workers] + ([models] if node in model_nodes else [])
        grid = np.array(list(itertools.product(*axes)), dtype=float)   # (points, q / w / m)
        x = np.zeros((len(grid), len(names) + 1))                      # the inputs of the load-aware GPs: knobs, log10 rate
        x[:, names.index("data_quality")], x[:, names.index("parallelism")] = grid[:, 0], grid[:, 1]
        if node in model_nodes:
            x[:, names.index("model_size")] = grid[:, 2]
        x[:, -1] = np.log10(rate)
        mean, var = run_statistics_belief(fitted["gp"][node], x, confidence)
        bound = martingale_closed_form_node_bound(mean, var, "poisson", rate, epsilon, workers=grid[:, 1].astype(int))
        tables[node] = np.asarray(bound, dtype=float).reshape([len(a) for a in axes])
    return Composition(experiment, path, fitted["services"], rate, qualities, workers, model_nodes, models, tables, extrapolated)


# ------------------------------------------------------------------------------
# Objective and constraints
# ------------------------------------------------------------------------------
@dataclass
class Scales:
    """Normalization of the three dimensions within one application, so that the objective weighs them alike."""
    max_workers: float
    latency: float      # s, a typical latency of the application
    q_low: float
    q_high: float


def objective(scenario: Scenario, scales: Scales, cost, latency, quality):
    """Value that the solver minimizes: the goal of the scenario, plus TIE_WEIGHT times each of the other two
    dimensions, all in normalized units (cost / max workers, latency / typical latency, quality on [0, 1] over its
    range, negated because a larger quality is better)."""
    terms = {"min cost": np.asarray(cost) / scales.max_workers, "min latency": np.asarray(latency) / scales.latency,
             "max quality": -(np.asarray(quality) - scales.q_low) / (scales.q_high - scales.q_low)}
    return sum((1.0 if goal == scenario.goal else TIE_WEIGHT) * term for goal, term in terms.items())


def feasible(scenario: Scenario, cost, latency, quality, model) -> np.ndarray:
    """Whether configurations meet every constraint of the scenario (and have a finite bound, i.e. no overloaded service)."""
    ok = np.isfinite(latency)
    if scenario.max_workers is not None:
        ok &= np.asarray(cost) <= scenario.max_workers
    if scenario.max_latency is not None:
        ok &= np.asarray(latency) <= scenario.max_latency
    if scenario.min_quality is not None:
        ok &= np.asarray(quality) >= scenario.min_quality
    if scenario.min_model is not None:
        ok &= np.asarray(model) >= scenario.min_model
    return ok


def grid_metrics(comp: Composition, indices: np.ndarray) -> dict:
    """cost, latency (s), quality and smallest detector model size of grid configurations given as index vectors
    (n, n_dims). A composition without a detector has model size 0: it runs no detection model, so it cannot meet a
    model constraint."""
    q_index, w_index, m_index = indices[:, 0], indices[:, 1:1 + len(comp.path)], indices[:, 1 + len(comp.path):]
    model = comp.models[m_index].min(axis=1) if comp.model_nodes else np.zeros(len(indices))
    return dict(cost=comp.workers[w_index].sum(axis=1), latency=comp.latency(q_index, w_index, m_index), quality=comp.qualities[q_index], model=model)


def all_grid_indices(comp: Composition) -> np.ndarray:
    """Every grid configuration of a composition as index vectors (n, n_dims)."""
    return np.array(list(itertools.product(*[range(len(a)) for a in comp.grid_axes()])))


# ------------------------------------------------------------------------------
# Solvers
# ------------------------------------------------------------------------------
def _relaxation(comp: Composition):
    """Continuous latency (s) of a normalized knob vector x in [0, 1]^n_dims: the node bounds interpolated linearly
    between grid points, an overloaded node capped at OVERLOAD_LATENCY."""
    interpolators = {}
    for node in comp.path:
        axes = [np.linspace(0, 1, len(comp.qualities)), np.linspace(0, 1, len(comp.workers))] + \
               ([np.linspace(0, 1, len(comp.models))] if node in comp.model_nodes else [])
        values = np.minimum(np.nan_to_num(comp.tables[node], nan=OVERLOAD_LATENCY, posinf=OVERLOAD_LATENCY), OVERLOAD_LATENCY)
        interpolators[node] = RegularGridInterpolator(axes, values)

    def latency(x: np.ndarray) -> float:
        x = np.clip(x, 0, 1)
        total = 0.0
        for k, node in enumerate(comp.path):
            point = [x[0], x[1 + k]] + ([x[1 + len(comp.path) + comp.model_nodes.index(node)]] if node in comp.model_nodes else [])
            total += float(interpolators[node](point)[0])
        return total

    return latency


def _denormalize(comp: Composition, x: np.ndarray) -> tuple:
    """Quality, workers per service and smallest model size of a normalized knob vector, as continuous values."""
    real = np.array([a[0] + xi * (a[-1] - a[0]) for xi, a in zip(np.clip(x, 0, 1), comp.grid_axes())])
    model = real[1 + len(comp.path):].min() if comp.model_nodes else 0.0
    return real[0], real[1:1 + len(comp.path)], model


def _best_of(comp: Composition, scenario: Scenario, scales: Scales, indices: np.ndarray):
    """The best feasible configuration among grid configurations (index vectors), or None."""
    metrics = grid_metrics(comp, indices)
    ok = feasible(scenario, **metrics)
    if not ok.any():
        return None
    values = objective(scenario, scales, metrics["cost"], metrics["latency"], metrics["quality"])
    k = int(np.flatnonzero(ok)[np.argmin(values[ok])])
    return dict(objective=float(values[k]), indices=indices[k], **{name: metric[k] for name, metric in metrics.items()})


def slsqp_composition(comp: Composition, scenario: Scenario, scales: Scales, rng: np.random.Generator, restarts: int = RESTARTS) -> dict:
    """The best configuration of one composition for a scenario: SLSQP on the continuous relaxation from `restarts`
    random starts, every solution rounded to the grid and repaired within one grid step per knob. Returns the best
    feasible grid configuration with its metrics, or None if no restart led to one."""
    latency = _relaxation(comp)
    n = comp.n_dims

    def fun(x):
        q, w, _ = _denormalize(comp, x)
        return float(objective(scenario, scales, w.sum(), latency(x), q))

    constraints = []   # SLSQP: every function must be >= 0 where the constraint holds; scaled to comparable units
    if scenario.max_workers is not None:
        constraints.append(dict(type="ineq", fun=lambda x: (scenario.max_workers - _denormalize(comp, x)[1].sum()) / scales.max_workers))
    if scenario.max_latency is not None:
        constraints.append(dict(type="ineq", fun=lambda x: (scenario.max_latency - latency(x)) / scales.latency))
    if scenario.min_quality is not None:
        constraints.append(dict(type="ineq", fun=lambda x: (_denormalize(comp, x)[0] - scenario.min_quality) / (scales.q_high - scales.q_low)))
    if scenario.min_model is not None and not comp.model_nodes:
        return None   # no detector: the model constraint cannot be met
    if scenario.min_model is not None:
        constraints.append(dict(type="ineq", fun=lambda x: _denormalize(comp, x)[2] - scenario.min_model))

    axes = comp.grid_axes()
    last = np.array([len(a) - 1 for a in axes])
    steps = np.array(list(itertools.product((-1, 0, 1), repeat=n)))   # the repair neighborhood: one grid step per knob
    best = None
    for _ in range(restarts):
        result = minimize(fun, rng.uniform(size=n), method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=constraints, options=dict(maxiter=200))
        rounded = np.rint(np.clip(result.x, 0, 1) * last).astype(int)
        candidate = _best_of(comp, scenario, scales, np.unique(np.clip(rounded + steps, 0, last), axis=0))
        if candidate is not None and (best is None or candidate["objective"] < best["objective"]):
            best = candidate
    return best


def exhaustive_composition(comp: Composition, scenario: Scenario, scales: Scales) -> dict:
    """The best feasible grid configuration of one composition, by enumerating every grid configuration (reference)."""
    return _best_of(comp, scenario, scales, all_grid_indices(comp))


def configuration_row(comp: Composition, best: dict) -> dict:
    """The deployable configuration of a solution: data quality, workers and model size per service, and its metrics."""
    indices = best["indices"]
    row = dict(objective=best["objective"], workers=int(best["cost"]), latency_ms=1e3 * float(best["latency"]),
               data_quality=float(comp.qualities[indices[0]]))
    for k, node in enumerate(comp.path):
        row[f"{node}.parallelism"] = int(comp.workers[indices[1 + k]])
    for j, node in enumerate(comp.model_nodes):
        row[f"{node}.model_size"] = int(comp.models[indices[1 + len(comp.path) + j]])
    return row


def provides(comp: Composition, capabilities: tuple) -> bool:
    """Whether a composition contains a service for every capability (keys of CAPABILITIES)."""
    images = {service.split(":")[0] for service in comp.services.values()}
    return all(CAPABILITIES[capability] in images for capability in capabilities)


def plan(comps: list, scenario: Scenario, scales: Scales, solver: str = "slsqp", rng: np.random.Generator = None) -> pd.DataFrame:
    """The best configuration of every composition for a scenario (one row each; feasible False where none meets the
    SLOs), sorted by the objective: the first row is the plan."""
    rows = []
    for comp in comps:
        functional = provides(comp, scenario.requires)
        if not functional:
            best = None   # the composition lacks a required capability: no configuration can meet the scenario
        else:
            best = slsqp_composition(comp, scenario, scales, rng) if solver == "slsqp" else exhaustive_composition(comp, scenario, scales)
        row = dict(scenario=scenario.name, composition=comp.experiment, services=" → ".join(comp.path), functional=functional,
                   feasible=best is not None, objective=np.nan)
        if best is not None:
            row |= configuration_row(comp, best)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feasible", "objective"], ascending=[False, True], na_position="last").reset_index(drop=True)


def deployment_table(plans: dict, rates: dict) -> pd.DataFrame:
    """One row per (application, scenario) with the chosen composition and its configuration in the knob names of the
    campaign (shared.data_quality, <node>.parallelism, <node>.model_size), for deploying it on the testbed."""
    rows = []
    for (app, scenario), table in plans.items():
        chosen = table.iloc[0]
        row = dict(application=app, scenario=scenario, arrival_rate=rates[app], feasible=bool(chosen["feasible"]))
        if chosen["feasible"]:
            knobs = {key: int(value) for key, value in chosen.items() if key.endswith((".parallelism", ".model_size")) and pd.notna(value)}
            row |= dict(composition=chosen["composition"], services=chosen["services"], workers=int(chosen["workers"]),
                        predicted_p95_ms=round(float(chosen["latency_ms"]), 1), knobs=json.dumps({"shared.data_quality": int(chosen["data_quality"]), **knobs}))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
def solver_agreement(slsqp: dict, exhaustive: dict, seconds: dict) -> pd.DataFrame:
    """Per (application, scenario): the composition that SLSQP and the exhaustive search chose, the objective of each,
    the gap of SLSQP (its objective minus the exhaustive one, in normalized units; 0 means that SLSQP found the optimum of
    the grid) and the time of each solver (s). slsqp, exhaustive: (application, scenario) -> plan table;
    seconds: solver -> (application, scenario) -> s."""
    rows = []
    for key, table in slsqp.items():
        a, b = table.iloc[0], exhaustive[key].iloc[0]
        rows.append(dict(application=key[0], scenario=key[1], slsqp=a["composition"] if a["feasible"] else None,
                         exhaustive=b["composition"] if b["feasible"] else None, same_composition=a["composition"] == b["composition"],
                         objective_gap=a["objective"] - b["objective"] if a["feasible"] and b["feasible"] else np.nan,
                         slsqp_s=seconds["slsqp"][key], exhaustive_s=seconds["exhaustive"][key]))
    return pd.DataFrame(rows)


def tradeoff_figure(comps: list, plans: dict, scenarios: list, app: str, worker_budget: int, name: str = None) -> None:
    """One figure for an application. Lines: for every composition, the smallest Pegasus bound on the p95 end-to-end
    latency (ms, log scale) over its configurations with at most `worker_budget` workers, at every data quality (x
    axis); a gap where no such configuration has a finite bound. Stars: the plan of every scenario (plans: (app,
    scenario name) -> plan table), at its data quality and bound, labeled with the scenario's short name and its
    workers; a plan may use more or fewer workers than worker_budget. name: file name in figures/tsc/experimental-results, or None."""
    import matplotlib.pyplot as plt
    from pegasus.figures import paper_figure, paper_style, save_figure
    with paper_style():
        fig, ax = paper_figure()
        colors = dict(zip([c.experiment for c in comps], plt.get_cmap("tab10").colors))
        for comp in comps:
            metrics = grid_metrics(comp, all_grid_indices(comp))
            within = metrics["cost"] <= worker_budget
            best = [np.min(metrics["latency"][within & (metrics["quality"] == q)], initial=np.inf) for q in comp.qualities]
            best = np.where(np.isfinite(best), best, np.nan)
            ax.plot(comp.qualities, 1e3 * best, marker="o", markersize=3, color=colors[comp.experiment],
                    label=f"{comp.experiment.removeprefix('b1u-')} ({len(comp.path)} services)")
        stars = {}   # (composition, quality, bound, workers) -> short names of the scenarios with that plan, one label per star
        for scenario in scenarios:
            chosen = plans[(app, scenario.name)].iloc[0]
            if chosen["feasible"]:
                key = (chosen["composition"], chosen["data_quality"], round(chosen["latency_ms"], 6), int(chosen["workers"]))
                stars.setdefault(key, []).append(scenario.name.split()[0])
        q_low, q_high = comps[0].qualities[[0, -1]]
        for (experiment, quality, latency, workers), names in stars.items():
            ax.plot(quality, latency, marker="*", markersize=13, color=colors[experiment], markeredgecolor="black", linestyle="none", zorder=5)
            right = quality > q_low + 0.8 * (q_high - q_low)   # label above and left of the star, so that it stays inside the axes
            ax.annotate(f"{', '.join(names)} ({workers}w)", (quality, latency), textcoords="offset points", xytext=(-6, 8) if right else (6, -10),
                        ha="right" if right else "left", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("data quality")
        ax.set_ylabel(f"p95 bound (ms), ≤ {worker_budget} workers")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="best", framealpha=0.95)
        if name is not None:
            save_figure(fig, name)
        plt.show()
