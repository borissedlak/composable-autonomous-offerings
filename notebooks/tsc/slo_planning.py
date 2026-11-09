"""Experiment C.2 of `2f_optimization.ipynb`: choosing the service composition of an application, and its
configuration, that meets the SLOs of a client, with a numerical solver on the predictions of Pegasus Node.

Setting. An application type (vision, audio, lidar) has five service compositions, the chains b1u-<application>-<i>
of the second campaign. A configuration of a composition sets
- q, the data quality that every service of the composition processes (knob data_quality; the image width in pixels
  for vision, the clip length in seconds for audio, the radius of the point cloud in meters for lidar),
- w_k, the number of workers of service k (knob parallelism, 1 to 4), and
- m_k, the model size of every detector that has one (knob model_size, 1 or 2; vision only).
The client of a scenario sends items at one fixed arrival rate (items/s, Poisson arrivals).

Three SLO dimensions, each one number per configuration:
- cost: the total number of workers of the composition, sum_k w_k;
- latency: the Pegasus Node bound (s) on the p95 of the end-to-end latency at the arrival rate of the scenario,
  with the load-aware models (load_aware_b4.py, the default of B.1);
- quality: the data quality q, and, for vision, the smallest model size among the detectors of the composition.
A scenario poses one or two of these as constraints (max workers, max latency, min quality). Every dimension that it
does not constrain is free and is optimized (fewer workers, lower latency, higher quality): the objective is a weighted
sum of the free dimensions, each divided by its typical value in the scenario (median over the configurations that
meet the constraints), with weights that the scenario sets, so that a weight applies to a relative change. The constrained dimensions enter
with a small weight (TIE_WEIGHT) that only breaks ties. A configuration with an overloaded service (no finite bound:
its queue grows without limit) is never feasible.

Prediction. The models of every composition are those of B.1: every service is trained on its benchmarks with one to
four workers (worker_benchmark_pool in pegasus/real_data.py), without any run of a chain, so that a composition is
planned as if it had never run. Pegasus Node bounds every service on its own (with the external
arrival rate and epsilon = alpha / K for K services), so the end-to-end bound is the sum of the node bounds. A node bound depends only on the knobs of its own service: `composition` computes it once on the grid
of every service's knobs, and the bound of any configuration of the composition is a sum of table entries.

Solver. The default is the exhaustive search: it enumerates every grid configuration of every composition and keeps the
best that meets the constraints (the grids are small enough, and the bounds are table lookups). The solver of the RASK
paper (SummerSOC, agent/components/Optimizer_v2.py) is kept but not used by the notebook: solver "slsqp" runs SLSQP
(scipy.optimize.minimize) on the knobs normalized to [0, 1], restarted from RESTARTS random points per composition, with
the node bounds linearly interpolated between the grid points (an overloaded node gets OVERLOAD_LATENCY) and every
solution rounded to the grid and repaired within one grid step per knob.

Units: latencies in seconds, rates in items/s.
"""
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

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
QUALITY_UNITS = dict(vision="image width (px)", audio="clip length (s)", lidar="point cloud rad. (m)")   # meaning of q
QUALITY_SHORT = dict(vision="px", audio="s", lidar="m")   # its unit
THIRD_MARGINS = (0.50, 0.025, 0.04)   # inches: below (tick labels and x label), above and right of the plot area of a third-width figure
CORE_HEADROOM = 0.5        # cores per service = workers + CORE_HEADROOM, the rule of the campaign runs (knob <node>.cores)
CACHE = real_data.CAMPAIGN.parent / "cache" / "slo_planning"   # statics/cache/slo_planning/: the fitted models of every composition (joblib)
PLANS = real_data.CAMPAIGN.parent / "slo_planning"             # statics/slo_planning/: the plans to deploy (csv)
C2_RESULTS = real_data.CAMPAIGN.parent / "results_milos_v2" / "c2"   # statics/results_milos_v2/c2/: testbed runs of the deployed plans (not committed)
RESTARTS = 10              # SLSQP starting points per composition, as run_optimizer_multi of the RASK paper
OVERLOAD_LATENCY = 10.0    # s; latency that the continuous relaxation gives an overloaded node, so that SLSQP moves away from it
TIE_WEIGHT = 1e-3          # weight of a constrained dimension in the objective, which only breaks ties of the free dimensions
DIMENSIONS = ("cost", "latency", "quality")
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


def label(experiment: str) -> str:
    """Short name of a composition (application instance): b1u-vision-1 -> V1."""
    app, i = experiment.removeprefix("b1u-").split("-")
    return f"{app[0].upper()}{i}"


@dataclass
class Scenario:
    """The SLOs of one client. None: no constraint on that dimension."""
    name: str                          # "S-<application letter>-<i> <title>", e.g. "S-V-1 edge box"
    rate: float                        # arrival rate of the client (items/s, Poisson arrivals)
    weights: dict                      # free (unconstrained) dimension -> its weight in the objective; every free dimension needs one
    max_workers: int = None            # cost: largest total number of workers of the composition
    max_latency: float = None          # latency: largest Pegasus bound on the p95 end-to-end latency (s)
    min_quality: float = None          # quality: smallest data quality q
    min_model: int = None              # quality (vision): smallest model size of every detector; does not constrain q
    note: str = ""                     # who would pose these SLOs
    requires: tuple = ()               # functional requirement: capabilities (keys of CAPABILITIES) the composition must provide

    def __post_init__(self):
        if set(self.weights) != set(self.free()):
            raise ValueError(f"{self.name}: weights {sorted(self.weights)} must name exactly the free dimensions {self.free()}")

    @property
    def short(self) -> str:
        return self.name.split()[0]

    def constrained(self) -> list:
        """The SLO dimensions that the scenario constrains (quality: by a smallest data quality)."""
        return [name for name, value in (("cost", self.max_workers), ("latency", self.max_latency), ("quality", self.min_quality)) if value is not None]

    def free(self) -> list:
        """The SLO dimensions that the scenario does not constrain, which the solver optimizes."""
        return [name for name in DIMENSIONS if name not in self.constrained()]

    def goal(self) -> str:
        """The objective in words, e.g. "min latency (1) + max quality (0.5)"."""
        verbs = dict(cost="min cost", latency="min latency", quality="max quality")
        return " + ".join(f"{verbs[name]} ({self.weights[name]:g})" for name in self.free())

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
    q_range: dict = field(default_factory=dict)        # node -> (smallest, largest) data quality of its training runs, any model size

    @property
    def label(self) -> str:
        return label(self.experiment)

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


def rate_range(experiment: str, model_seed: int = seed) -> tuple:
    """The arrival rates (items/s) that every service of a composition had in its training runs: (low, high), the
    intersection of the ranges of its services; low > high if they do not overlap."""
    ranges = np.array(list(fit_composition(experiment, model_seed)["log_rates"].values()))
    return 10 ** ranges[:, 0].max(), 10 ** ranges[:, 1].min()


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
    q_range = {node: (lambda q: (q.min(), q.max()))(la.extended_pool(fitted["services"][node])["data_quality"]) for node in path}
    return Composition(experiment, path, fitted["services"], rate, qualities, workers, model_nodes, models, tables, extrapolated, q_range)


# ------------------------------------------------------------------------------
# Objective and constraints
# ------------------------------------------------------------------------------
@dataclass
class Scales:
    """The typical value of every dimension in one scenario, which normalizes the dimension in the objective: a weight
    then applies to a relative change. Example: typical quality 500 px and latency 0.5 s, weights quality 1 and latency
    0.5: 50 px more quality (10 % of 500) are worth 100 ms more latency (20 % of 0.5 s, times 0.5)."""
    cost: float         # workers
    latency: float      # s
    quality: float     # unit of the application's data quality


def typical_scales(comps: list, scenario: Scenario) -> Scales:
    """The typical values of a scenario: the median of every dimension over the grid configurations of the candidate
    compositions (those that provide the required capabilities) that meet the scenario's constraints."""
    values = {name: [] for name in DIMENSIONS}
    for comp in comps:
        if not provides(comp, scenario.requires):
            continue
        metrics = grid_metrics(comp, all_grid_indices(comp))
        ok = feasible(scenario, **metrics)
        for name, key in (("cost", "cost"), ("latency", "latency"), ("quality", "quality")):
            values[name].append(np.asarray(metrics[key], dtype=float)[ok])
    return Scales(**{name: float(np.median(np.concatenate(parts))) for name, parts in values.items()})


def objective(scenario: Scenario, scales: Scales, cost, latency, quality):
    """Value that the solver minimizes: the weighted sum of the free dimensions of the scenario (weights of the
    scenario), plus TIE_WEIGHT times each constrained dimension, every dimension divided by its typical value in the
    scenario (quality negated because a larger quality is better)."""
    terms = {"cost": np.asarray(cost) / scales.cost, "latency": np.asarray(latency) / scales.latency,
             "quality": -np.asarray(quality) / scales.quality}
    return sum(scenario.weights.get(name, TIE_WEIGHT) * term for name, term in terms.items())


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


def _closest_of(comp: Composition, scenario: Scenario, indices: np.ndarray):
    """The best-effort configuration of a scenario that no configuration meets: among the grid configurations that meet every
    constraint except the latency limit and have a finite bound, the one with the smallest bound (the closest to the limit),
    or None. Its `objective` is that bound (s)."""
    metrics = grid_metrics(comp, indices)
    without_latency_limit = SimpleNamespace(max_workers=scenario.max_workers, max_latency=None, min_quality=scenario.min_quality,
                                            min_model=scenario.min_model)   # not replace(): a Scenario checks its weights against the free dimensions
    ok = feasible(without_latency_limit, **metrics)
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmin(metrics["latency"][ok])])
    return dict(objective=float(metrics["latency"][k]), indices=indices[k], **{name: metric[k] for name, metric in metrics.items()})


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
        constraints.append(dict(type="ineq", fun=lambda x: (scenario.max_workers - _denormalize(comp, x)[1].sum()) / scales.cost))
    if scenario.max_latency is not None:
        constraints.append(dict(type="ineq", fun=lambda x: (scenario.max_latency - latency(x)) / scales.latency))
    if scenario.min_quality is not None:
        constraints.append(dict(type="ineq", fun=lambda x: (_denormalize(comp, x)[0] - scenario.min_quality) / scales.quality))
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


def plan(comps: list, scenario: Scenario, scales: Scales, solver: str = "exhaustive", rng: np.random.Generator = None) -> pd.DataFrame:
    """The best configuration of every composition for a scenario (one row each; feasible False where none meets the
    SLOs), sorted by the objective: the first row is the plan. If no composition meets the SLOs and the scenario has a
    latency limit, every composition that provides the required capabilities instead gets its best-effort configuration
    (`_closest_of`, the smallest predicted bound, feasible False, best_effort True) sorted by that bound, so that the plan
    is the closest one to the limit."""
    def rows_of(best_of) -> list:
        rows = []
        for comp in comps:
            functional = provides(comp, scenario.requires)
            best = best_of(comp) if functional else None   # a composition without a required capability cannot meet the scenario
            row = dict(scenario=scenario.name, composition=comp.experiment, services=" → ".join(comp.path), functional=functional,
                       feasible=False, best_effort=False, objective=np.nan)
            if best is not None:
                row |= configuration_row(comp, best)
            rows.append(row)
        return rows
    rows = rows_of(lambda comp: slsqp_composition(comp, scenario, scales, rng) if solver == "slsqp" else exhaustive_composition(comp, scenario, scales))
    for row in rows:
        row["feasible"] = bool(np.isfinite(row["objective"]))
    if not any(row["feasible"] for row in rows) and solver != "slsqp" and scenario.max_latency is not None:
        rows = rows_of(lambda comp: _closest_of(comp, scenario, all_grid_indices(comp)))
        for row in rows:
            row["best_effort"] = bool(np.isfinite(row["objective"]))
    return pd.DataFrame(rows).sort_values(["feasible", "best_effort", "objective"], ascending=[False, False, True], na_position="last").reset_index(drop=True)


def deployment_table(plans: dict, rates: dict) -> pd.DataFrame:
    """One row per (application, scenario) with the chosen composition and its configuration in the knob names of the
    campaign (shared.data_quality, <node>.parallelism, <node>.cores = parallelism + CORE_HEADROOM, <node>.model_size),
    for deploying it on the testbed at the arrival rate of the scenario (rates: scenario name -> items/s). feasible: the
    configuration meets every SLO; best_effort: no configuration does, and this one has the smallest predicted bound."""
    rows = []
    for (app, scenario), table in plans.items():
        chosen = table.iloc[0]
        row = dict(application=app, scenario=scenario, arrival_rate=rates[scenario], feasible=bool(chosen["feasible"]),
                   best_effort=bool(chosen.get("best_effort", False)))
        if chosen["feasible"] or row["best_effort"]:
            knobs = {}
            for key, value in chosen.items():
                if key.endswith(".parallelism") and pd.notna(value):
                    knobs[key], knobs[key.replace(".parallelism", ".cores")] = int(value), int(value) + CORE_HEADROOM
                elif key.endswith(".model_size") and pd.notna(value):
                    knobs[key] = int(value)
            row |= dict(composition=label(chosen["composition"]), experiment=chosen["composition"], services=chosen["services"], workers=int(chosen["workers"]),
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
        rows.append(dict(application=key[0], scenario=key[1], slsqp=label(a["composition"]) if a["feasible"] else None,
                         exhaustive=label(b["composition"]) if b["feasible"] else None, same_composition=a["composition"] == b["composition"],
                         objective_gap=a["objective"] - b["objective"] if a["feasible"] and b["feasible"] else np.nan,
                         slsqp_s=seconds["slsqp"][key], exhaustive_s=seconds["exhaustive"][key]))
    return pd.DataFrame(rows)


def _tradeoff_panel(ax, scenario: Scenario, comps: list, chosen: pd.Series, app: str, worker_budget: int, colors: dict, markers: dict,
                    measurement: tuple = None, single: bool = False, third: bool = False) -> None:
    """One panel of `tradeoff_figure`: scenario, its compositions `comps` at its rate, its plan row `chosen`. measurement:
    (data quality, predicted bound, measured p95, measured p99), all of one earlier deployment, drawn at its data quality (ms)
    or None; with a predicted bound of None only the measurement. single: the panel is a figure of its own (larger fonts, no
    title, which the caption gives). third: for a figure one third of the usual width: short x label, a legend of the compositions
    only (the measurement and the extrapolated stretch are explained in the text)."""
    small, big = (9, 12) if single else (8, 10)
    budget = scenario.max_workers if scenario.max_workers is not None else worker_budget
    extrapolated_q = False
    for comp in comps:
        if not provides(comp, scenario.requires):
            continue
        metrics = grid_metrics(comp, all_grid_indices(comp))
        within = (metrics["cost"] <= budget) & (metrics["model"] >= (scenario.min_model or 0))
        best = np.array([np.min(metrics["latency"][within & (metrics["quality"] == q)], initial=np.inf) for q in comp.qualities])
        y = 1e3 * np.where(np.isfinite(best), best, np.nan)
        # a quality outside the training runs of any service of the composition: its GP extrapolates the quality input
        outside = np.array([any(not lo <= q <= hi for lo, hi in comp.q_range.values()) for q in comp.qualities])
        near = outside.copy()   # the dashed stretch also reaches the neighboring trained point, so that it joins the solid line
        near[1:] |= outside[:-1]
        near[:-1] |= outside[1:]
        ax.plot(comp.qualities, np.where(~outside, y, np.nan), marker=markers[comp.experiment], markersize=3, linewidth=1.5,
                color=colors[comp.experiment], label=comp.label)
        ax.plot(comp.qualities, np.where(near, y, np.nan), linewidth=1.5, linestyle="--", color=colors[comp.experiment])
        ax.plot(comp.qualities, np.where(outside, y, np.nan), marker=markers[comp.experiment], markersize=3, linestyle="none",
                color=colors[comp.experiment], markerfacecolor="white")
        extrapolated_q = extrapolated_q or bool(outside.any())
    if scenario.max_latency is not None:
        ax.axhline(1e3 * scenario.max_latency, color="black", linestyle="--", linewidth=1)
    if scenario.min_quality is not None:
        ax.axvline(scenario.min_quality, color="black", linestyle="--", linewidth=1)
    if chosen["feasible"] or chosen.get("best_effort", False):   # hollow star: best effort, no configuration meets the SLOs
        ax.plot(chosen["data_quality"], chosen["latency_ms"], marker="*", markersize=14, color=colors[chosen["composition"]],
                markerfacecolor=colors[chosen["composition"]] if chosen["feasible"] else "white",
                markeredgecolor="black", linestyle="none", zorder=5)
        note = ax.annotate(f"{int(chosen['workers'])}w", (chosen["data_quality"], chosen["latency_ms"]), textcoords="offset points",
                           xytext=(0, 9), ha="center", fontsize=big)
        low, high = ax.get_xlim()
        if single and chosen["data_quality"] > high - 0.08 * (high - low):
            note.set_ha("right")   # a star near the right edge: its label would reach beyond the figure
            note.xyann = (5, 9)
    if measurement is not None:
        q, bound, p95, p99 = measurement
        if bound is not None:
            ax.plot(q, bound, marker="D", markersize=6, markerfacecolor="white", markeredgecolor="black", linestyle="none", zorder=6,
                    label=None if third else "bound, current models")
        ax.plot([q, q], [p95, p99], color="black", linewidth=1.5, marker="_", markersize=6, zorder=6)   # not errorbar: its 0 lower error stretches a log axis
        ax.plot(q, p95, marker="o", markersize=5, color="black", linestyle="none", zorder=6, label=None if third else "measured p95 (bar to p99)")
    if extrapolated_q and not third:
        ax.plot([], [], color="gray", linestyle="--", marker="o", markersize=3, markerfacecolor="white", label="extrapolated")
    if not single:
        ax.set_title(f"{scenario.short}, {scenario.rate:g}/s, ≤ {budget} w", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlabel(QUALITY_UNITS[app], fontsize=big)   # the meaning and the unit of q, also in the narrow figures
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=small if single else 8, loc="best", framealpha=0.95,
              **(dict(handlelength=1.0, labelspacing=0.2, borderpad=0.3, borderaxespad=0.3) if third else {}))


def _testbed_panel(ax, scenario: Scenario, latency: np.ndarray, bound: float) -> None:
    """One panel of `testbed_figure`: the requests (ms) of a scenario and the predicted bound (ms) of its deployed plan."""
    latency = np.sort(latency)
    n = len(latency)
    ax.step(latency[:-1], 1 - np.arange(1, n) / n, where="post", color="tab:blue", linewidth=1.5)   # P(latency > x)
    ax.axhline(0.05, color="black", linewidth=1, linestyle=":")
    ax.axvline(bound, color="tab:red", linewidth=1.5)
    if scenario.max_latency is not None:
        ax.axvline(1e3 * scenario.max_latency, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.05 * max(bound, latency[-1], 1e3 * (scenario.max_latency or 0)))
    ax.set_yscale("log")
    ax.set_ylim(0.5 / n, 1)
    ax.set_xlabel("latency x (ms)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)


def _testbed_legend(ax) -> None:
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="tab:blue", linewidth=1.5, label="measured requests"),
                       Line2D([], [], color="tab:red", linewidth=1.5, label="predicted p95 bound"),
                       Line2D([], [], color="black", linewidth=1, linestyle="--", label="latency SLO"),
                       Line2D([], [], color="black", linewidth=1, linestyle=":", label="0.05")],
              fontsize=8, loc="lower left", framealpha=0.95)


def bound_of_knobs(comp: Composition, knobs: dict) -> float:
    """The predicted p95 bound (ms) that the models of `comp` give a configuration in the knob names of the campaign
    (`knobs` as in the deployment table)."""
    q = list(comp.qualities).index(knobs["shared.data_quality"])
    w = [knobs[f"{node}.parallelism"] - 1 for node in comp.path]
    m = [list(comp.models).index(knobs[f"{node}.model_size"]) for node in comp.model_nodes]
    return 1e3 * float(grid_metrics(comp, np.array([[q] + w + m]))["latency"][0])


def _measurement(scenario: Scenario, comps: list, measured: dict, deployed: pd.DataFrame) -> tuple:
    """(data quality, bound, measured p95, measured p99) of the deployed plan of a scenario: the bound (ms) is the one that
    the current models give its configuration, not the one that it was planned with."""
    row = deployed[deployed["scenario"] == scenario.name].iloc[0]
    knobs = json.loads(row["knobs"])
    comp = next(c for c in comps if c.experiment == row["experiment"])
    latency = measured[scenario.name][0]
    return knobs["shared.data_quality"], bound_of_knobs(comp, knobs), np.percentile(latency, 95), np.percentile(latency, 99)


def _trade_colors(comps: dict) -> tuple:
    import matplotlib.pyplot as plt
    experiments = [c.experiment for c in next(iter(comps.values()))]
    return dict(zip(experiments, plt.get_cmap("tab10").colors)), dict(zip(experiments, "osD^v"))


def tradeoff_figure(comps: dict, plans: dict, scenarios: list, app: str, worker_budget: int, name: str = None,
                    measured: dict = None, deployed: pd.DataFrame = None) -> None:
    """One figure for an application with one panel per scenario (comps: scenario name -> the compositions at the
    scenario's arrival rate; plans: (app, scenario name) -> plan table). In a panel, one line per composition that
    provides the scenario's capabilities: the smallest Pegasus bound on the p95 end-to-end latency (ms, log scale) over
    its configurations within the scenario's worker limit (worker_budget if it has none) and its model constraint, at
    every data quality (x axis); a gap where no such configuration has a finite bound; dashed with hollow markers where the
    data quality lies outside the training runs of a service of the composition (`Composition.q_range`). Dashed lines: the scenario's
    latency and quality constraints. Star: the plan, labeled with its total workers (hollow: best effort, no configuration
    meets the SLOs). measured, deployed (testbed requests and plans of an earlier deployment, see `measured_c2`): also a
    diamond (the bound that the current models give that deployment's configuration) and a dot with a bar from the measured p95 to the measured p99 (ms), at its data
    quality. name: file name in figures/tsc/experimental-results, or None."""
    import matplotlib.pyplot as plt
    from pegasus.figures import paper_style, save_figure
    colors, markers = _trade_colors(comps)
    with paper_style():
        fig, axes = plt.subplots(1, len(scenarios), figsize=(3.0 * len(scenarios), 3.3), sharey=True, layout="constrained")
        for ax, scenario in zip(axes, scenarios):
            _tradeoff_panel(ax, scenario, comps[scenario.name], plans[(app, scenario.name)].iloc[0], app, worker_budget, colors, markers,
                            None if measured is None else _measurement(scenario, comps[scenario.name], measured, deployed))
        axes[0].set_ylabel("p95 bound (ms)")
        if name is not None:
            save_figure(fig, name)
        plt.show()


def _short_log_ticks(axis) -> None:
    """Tick labels of a logarithmic axis as plain numbers, thousands as 1k, 10k (like `pegasus.figures.plain_log_ticks`,
    narrower), at the ticks that matplotlib labels anyway."""
    from matplotlib.ticker import LogFormatterSciNotation

    class Short(LogFormatterSciNotation):
        def __call__(self, x, pos=None):
            if not super().__call__(x, pos):
                return ""
            return f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"
    axis.set_major_formatter(Short())
    axis.set_minor_formatter(Short(labelOnlyBase=False, minor_thresholds=(1, 0.4)))


def tradeoff_panel_figure(comps: list, plan: pd.Series, scenario: Scenario, app: str, worker_budget: int, measured: dict = None,
                          deployed: pd.DataFrame = None, show_bound: bool = True, third: bool = False, ylabel: bool = True,
                          name: str = None) -> None:
    """One panel of `tradeoff_figure` as a figure of its own, at the size of every evaluation figure (comps: the five
    compositions at the scenario's rate; plan: its plan row). measured, deployed: the measurement of the deployed plan as in
    `tradeoff_figure`; show_bound False: without the diamond, for a deployed configuration that is the plan (the star is then
    its bound). third: exactly one third of the width of an evaluation figure and its height, so that three of them stand side
    by side in the width of one (the left margin shrinks to the labels of the vertical axis, whose ticks are short: 1k).
    ylabel False: without the axis label, which the first of the figures gives for all. name: file name in
    figures/tsc/experimental-results, or None."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from pegasus.figures import FIGSIZE, paper_figure, paper_style, plain_log_ticks, save_figure
    colors, markers = _trade_colors({scenario.name: comps})
    measurement = None if measured is None else _measurement(scenario, comps, measured, deployed)
    if measurement is not None and not show_bound:
        measurement = (measurement[0], None, measurement[2], measurement[3])
    with paper_style():
        if third:
            width, height = FIGSIZE[0] / 3, FIGSIZE[1]
            bottom, top, right = THIRD_MARGINS
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_axes((0.5 / width, bottom / height, 1 - (0.5 + right) / width, 1 - (bottom + top) / height))
        else:
            fig, ax = paper_figure()
        _tradeoff_panel(ax, scenario, comps, plan, app, worker_budget, colors, markers, measurement, single=True, third=third)
        if ylabel:
            ax.set_ylabel("bound (ms)" if third else "p95 bound (ms)")
        (_short_log_ticks if third else plain_log_ticks)(ax.yaxis)
        if third:
            ax.xaxis.set_major_locator(MaxNLocator(4, prune="upper"))
            fig.canvas.draw()
            left = (ax.bbox.x0 - ax.yaxis.get_tightbbox(fig.canvas.get_renderer()).x0) / fig.dpi + 0.005   # the labels of the vertical axis, no more
            ax.set_position((left / width, bottom / height, 1 - (left + right) / width, 1 - (bottom + top) / height))
        if name is not None:
            save_figure(fig, name)
        plt.show()


# ------------------------------------------------------------------------------
# Testbed runs of the deployed plans
# ------------------------------------------------------------------------------
def measured_c2(deployed: pd.DataFrame) -> dict:
    """The testbed runs of the deployed plans (`deployed`: c2_plans_old.csv). Returns scenario name -> (latencies, as_planned):
    the end-to-end latency (ms) of every completed request of the scenario's runs, pooled over the runs, and whether the
    deployed configuration (data quality, workers, cores, model sizes, arrival rate in experiment.yaml) is the planned one."""
    import yaml
    out = {}
    for _, plan in deployed[deployed["feasible"]].iterrows():
        sid = plan["scenario"].split()[0].lower()   # "S-V-1 edge box" -> "s-v-1", the prefix of the experiment folder
        folder = sorted(C2_RESULTS.glob(f"c2-{sid}-*/*/"))[-1]
        experiment = yaml.safe_load((folder / "experiment.yaml").read_text())
        ran = {"shared.data_quality": experiment["shared_knobs"]["data_quality"]}
        for node, spec in experiment["nodes"].items():
            ran[f"{node}.parallelism"], ran[f"{node}.cores"] = spec["parallelism"], spec["cores"]
            if "model_size" in spec.get("knobs", {}):
                ran[f"{node}.model_size"] = spec["knobs"]["model_size"]
        as_planned = ran == json.loads(plan["knobs"]) and experiment["load"]["rps"] == plan["arrival_rate"]
        traces = (pd.read_csv(f) for f in sorted(folder.glob("runs/*/traces.csv")))
        out[plan["scenario"]] = (np.concatenate([t.loc[t["completed"], "total_ms"].to_numpy() for t in traces]), as_planned)
    return out


def testbed_table(deployed: pd.DataFrame, measured: dict, scenarios: dict) -> pd.DataFrame:
    """One row per deployed scenario at the grain of the requests: their number, the predicted p95 bound (ms), the measured
    p95 and p99 (ms), the share of requests slower than the bound (%), the tightness bound / measured p95, and the latency
    SLO of the scenario (ms, if it has one). scenarios: scenario name -> Scenario."""
    rows = []
    for _, plan in deployed[deployed["feasible"]].iterrows():
        latency, as_planned = measured[plan["scenario"]]
        bound, p95 = plan["predicted_p95_ms"], np.percentile(latency, 95)
        limit = scenarios[plan["scenario"]].max_latency
        rows.append(dict(application=plan["application"], scenario=plan["scenario"], requests=len(latency), as_planned=as_planned,
                         bound_ms=bound, measured_p95_ms=p95, measured_p99_ms=np.percentile(latency, 99),
                         above_bound_pct=100 * (latency > bound).mean(), tightness=bound / p95,
                         slo_ms=None if limit is None else 1e3 * limit))
    return pd.DataFrame(rows)


def testbed_figure(measured: dict, deployed: pd.DataFrame, scenarios: list, app: str, name: str = None) -> None:
    """One figure for an application with one panel per scenario, at the grain of the requests. x axis: end-to-end latency
    x (ms). y axis (log scale): the share of the scenario's requests that were slower than x (the complementary empirical
    distribution of all requests of its runs). Horizontal line: the share 0.05 that a p95 bound may leave above it.
    Solid vertical line: the predicted p95 bound of the deployed plan. Dashed vertical line: the latency SLO, if any.
    The bound is at least the measured p95 where the vertical line lies right of the crossing of the curve with 0.05.
    name: file name in figures/tsc/experimental-results, or None."""
    import matplotlib.pyplot as plt
    from pegasus.figures import paper_style, save_figure
    with paper_style():
        fig, axes = plt.subplots(1, len(scenarios), figsize=(3.0 * len(scenarios), 3.3), sharey=True, layout="constrained")
        for ax, scenario in zip(axes, scenarios):
            bound = float(deployed.loc[deployed["scenario"] == scenario.name, "predicted_p95_ms"].iloc[0])
            _testbed_panel(ax, scenario, measured[scenario.name][0], bound)
            ax.set_title(f"{scenario.short}, {scenario.rate:g}/s", fontsize=11)
        axes[0].set_ylabel("share of requests slower than x")
        _testbed_legend(axes[0])
        if name is not None:
            save_figure(fig, name)
        plt.show()
