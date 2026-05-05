from typing import Dict, List, Any

from scipy.optimize import minimize

from agent.components.GaussianProcess import GASK, get_empirical_variable_bounds
from agent.components.SLORegistry_v2 import calculate_weighted_SLO_F
from agent.components.commons import ServiceType, ServiceVar

import logging
logger = logging.getLogger(__name__)

def local_obj(x_norm, s_type: ServiceType, slos: Dict[ServiceVar, float], gp: GASK,
              simple_param_bounds, conservative: bool = True):
    """
        :param x_norm: values that MUST BE normalized
        :param s_type: type of service to investigate
        :param slos: just weights, no thresholds
        :param gp: gaussian process expressing system dynamics
        :param simple_param_bounds: min/max feature bounds according to dataset
        :param conservative: describes if the numerical solver should penalize high variance (when True)
        :return: SLO fulfillment
        """

    # Store these to use for de-normalization inside the objective
    # simplified_bounds = list((a[0], a[1]) for a in ordered_bounds.values())

    # Translate 0-1 back to Real Units
    x_real = []
    for i, (mini, maxi) in enumerate(simple_param_bounds):
        x_real.append(x_norm[i] * (maxi - mini) + mini)

    x_state = {ServiceVar.COST: x_real[0], ServiceVar.QUALITY: x_real[1]}
    if s_type == ServiceType.CV:
        x_state[ServiceVar.MODEL] = x_real[2]

    # Now the GP receives the units it expects (or uses its internal scaler)
    mu, sigma = gp.predict(s_type, "max_tp", x_state)

    # Gives me the 5th percentile, meaning 95% of the time, tp is larger; thus, solutions that have a high mean,
    # but also a high sd, are not as likely chosen because they might fail this also quite often.
    conservative_max_tp = (mu - 1.645 * sigma) if conservative else mu
    max_tp = {ServiceVar.PERFORMANCE: conservative_max_tp}

    empirical_boundaries = get_empirical_variable_bounds(gp.training_data)[s_type]
    slo_f = calculate_weighted_SLO_F(x_state | max_tp, slos, empirical_boundaries)

    # print(f"Calculated SLO-F for {x_state}: {slo_f}")
    return -slo_f


def run_optimizer_multi(s_type: ServiceType, slos, gp: GASK, bounds, runs: int = 10):
    results = []
    for _ in range(runs):
        fitness, x = solve_global(s_type, slos, gp, bounds, conservative=True)
        results.append((fitness, x))
    return results


def solve_global(s_type: ServiceType, slos, gp: GASK, parameter_bounds,
                 last_assignments=None, conservative: bool = True):

    simplified_bounds = list(parameter_bounds.copy().values())

    # Normalize your starting point x0
    if last_assignments:
        # Convert [5.0, 590] -> [normalized_cores, normalized_dq]
        x0 = []
        for i, (mini, maxi) in enumerate(simplified_bounds):
            norm_val = (last_assignments[i] - mini) / (maxi - mini)
            x0.append(norm_val)
    else:
        x0 = [np.random.uniform()] + [np.random.uniform()] + ([np.random.uniform()] if s_type == ServiceType.CV else [])

    # THE SOLVER BOX: Everything is 0 to 1
    normalized_param_bounds = [(0.0, 1.0) for _ in simplified_bounds]

    # Pass ordered_bounds to the objective so it can "un-scale"
    result = minimize(local_obj, x0, method='SLSQP', bounds=normalized_param_bounds,
                      args=(s_type, slos, gp, simplified_bounds, conservative), options={'maxiter': 150})

    # if not result.success:
    #     raise RuntimeWarning("Solver failed: " + result.message)
    if not result.success:
        logger.warning(f"Solver failed: {result.message}. Returning best found so far.")
        # Instead of raising, return the result anyway or a default
        return -result.fun, result.x

    # Convert the optimal 0-1 answer BACK to real units
    final_x = []
    for i, (mini, maxi) in enumerate(simplified_bounds):
        final_x.append(result.x[i] * (maxi - mini) + mini)

    return -result.fun, final_x


import numpy as np


class VersatileMapElites:
    def __init__(self, s_type: ServiceType, bins=10):
        self.bins = bins
        self.s_type = s_type

        # Determine the shape of the archive
        # If CV, the 3rd dimension has half the resolution (bins // 2)
        if s_type == ServiceType.CV:
            self.dims = 3
            self.table_shape = (bins, bins, max(1, bins // 2))
        else:
            self.dims = 2
            self.table_shape = (bins, bins)

        # Initialize tables with dynamic shape
        self.fitness_table = np.full(self.table_shape, -np.inf)
        self.solution_params_table = np.empty(self.table_shape, dtype=object)
        self.elite_group: List[Any] = []

    def get_bin(self, x_norm):
        """
        Converts normalized coordinates (0-1) to integer indices.
        """
        indices = []
        for i in range(self.dims):
            # Use the specific bin count for this dimension from table_shape
            dim_bins = self.table_shape[i]
            idx = int(x_norm[i] * (dim_bins - 1))
            indices.append(np.clip(idx, 0, dim_bins - 1))
        return tuple(indices)

    def run_search(self, slos, gp, simple_param_bounds, iterations=1000):
        for i in range(iterations):
            # --- 1. SELECTION & MUTATION ---
            if i < 100 or np.all(self.fitness_table == -np.inf):
                # Sample based on the number of dimensions (2 or 3)
                x_new = np.random.uniform(0, 1, self.dims)
            else:
                # Select from existing elites
                occupied_indices = np.argwhere(self.fitness_table > -np.inf)
                random_idx = tuple(occupied_indices[np.random.choice(len(occupied_indices))])
                parent = self.solution_params_table[random_idx]

                # Mutate: Gaussian nudge for all dimensions
                x_new = np.clip(parent + np.random.normal(0, 0.05, self.dims), 0, 1)

            # --- 2. EVALUATION ---
            # Pass s_type (stored in self) to the objective function
            fitness = -local_obj(x_new, self.s_type, slos, gp, simple_param_bounds)

            # --- 3. THE COMPETITION ---
            bin_idx = self.get_bin(x_new)

            if fitness > self.fitness_table[bin_idx]:
                self.fitness_table[bin_idx] = fitness
                self.solution_params_table[bin_idx] = x_new

                # if i % 100 == 0:
                #     print(f"Iteration {i}: Elite found in bin {bin_idx} with fitness {fitness:.4f}")

    def get_diverse_set(self, n_solutions=5, versatility=0.2):
        # 1. Flatten the archive (works for both 2D and 3D shapes)
        candidates = []
        it = np.nditer(self.fitness_table, flags=['multi_index'])
        for f in it:
            idx = it.multi_index
            if self.fitness_table[idx] > -np.inf:
                candidates.append({
                    'coord': self.solution_params_table[idx],
                    'fitness': self.fitness_table[idx]
                })

        candidates = sorted(candidates, key=lambda x: x['fitness'], reverse=True)

        selected = []
        for cand in candidates:
            if len(selected) >= n_solutions:
                break

            is_diverse_enough = True
            for sel in selected:
                # Euclidean distance works regardless of 2D or 3D
                dist = np.linalg.norm(cand['coord'] - sel['coord'])
                if dist < versatility:
                    is_diverse_enough = False
                    break

            if is_diverse_enough:
                selected.append(cand)

        self.elite_group = selected
        return selected
