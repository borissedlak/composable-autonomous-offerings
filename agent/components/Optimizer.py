from typing import Dict

from scipy.optimize import minimize

from agent.components.GaussianProcess import GASK, get_empirical_boundaries
from agent.components.SLORegistry_v2 import calculate_weighted_SLO_F
from agent.components.commons import ServiceType



def local_obj(x_norm, slos: Dict[str, float], gp: GASK, ordered_bounds):
    """
        :param x_norm: values that are NOT normalized
        :param slos: just weights, no thresholds
        :param gp: gaussian process expressing system dynamics
        :param ordered_bounds: min/max feature bounds according to dataset
        :return: SLO fulfillment
        """

    # Translate 0-1 back to Real Units
    x_real = []
    for i, (mini, maxi) in enumerate(ordered_bounds):
        x_real.append(x_norm[i] * (maxi - mini) + mini)

    x_state = {'cores': x_real[0], 'data_quality': x_real[1]}

    # Now the GP receives the units it expects (or uses its internal scaler)
    mu, sigma = gp.predict(ServiceType.QR, "max_tp", x_state)

    # Gives me the 5th percentile, meaning 95% of the time, tp is larger; thus, solutions that have a high mean,
    # but also a high sd, are not as likely chosen because they might fail this also quite often.
    conservative_max_tp = mu - 1.645 * sigma
    max_tp = {'max_tp': conservative_max_tp}

    empirical_boundaries = get_empirical_boundaries(gp.training_data)[ServiceType.QR]
    slo_f = calculate_weighted_SLO_F(x_state | max_tp, slos, empirical_boundaries)

    # print(f"Calculated SLO-F for {x_state}: {slo_f}")
    return -slo_f


def solve_global(slos, gp, last_assignments):
    raw_bounds = get_empirical_boundaries(gp.training_data)[ServiceType.QR]
    del raw_bounds['max_tp']

    # Store these to use for de-normalization inside the objective
    ordered_bounds = list(raw_bounds.values())

    # THE SOLVER BOX: Everything is 0 to 1
    flat_bounds = [(0.0, 1.0) for _ in ordered_bounds]

    # Normalize your starting point x0
    if last_assignments:
        # Convert [5.0, 590] -> [normalized_cores, normalized_dq]
        x0 = []
        for i, (mini, maxi) in enumerate(ordered_bounds):
            norm_val = (last_assignments[i] - mini) / (maxi - mini)
            x0.append(norm_val)
    else:
        x0 = [0.5, 0.5]  # Start in the middle

    # Pass ordered_bounds to the objective so it can "un-scale"
    result = minimize(local_obj, x0, method='SLSQP', bounds=flat_bounds,
                      args=(slos, gp, ordered_bounds), options={'maxiter': 150, 'eps': 1e-2})

    if not result.success:
        raise RuntimeWarning("Solver failed: " + result.message)

    # Convert the optimal 0-1 answer BACK to real units
    final_x = []
    for i, (mini, maxi) in enumerate(ordered_bounds):
        final_x.append(result.x[i] * (maxi - mini) + mini)

    return final_x

import numpy as np
from sklearn.cluster import KMeans


class VersatileMapElites:
    def __init__(self, bins=10):
        self.bins = bins
        self.fitness_table = np.full((bins, bins), -np.inf) # max solution fitness at cell (x, y)
        self.solution_params_table = np.empty((bins, bins), dtype=object) # respective param. assignments (x, y)

    def get_bin(self, x_norm):
        idx = (x_norm * (self.bins - 1)).astype(int)
        return np.clip(idx, 0, self.bins - 1)

    def run_search(self, slos, gp, ordered_bounds, iterations=1000):
        """
        The illumination loop.
        1. Select a parent from the archive (or start random).
        2. Mutate the parent to create a child.
        3. Evaluate the child using the GP (Conservative Mean).
        4. Attempt to store the child in the grid.
        """
        for i in range(iterations):
            # --- 1. SELECTION & MUTATION ---
            if i < 100 or np.all(self.fitness_table == -np.inf):
                # Bootstrap phase: Randomly sample the space to find the first 'Elites'
                x_new = np.random.uniform(0, 1, 2)
            else:
                # Picking phase: Randomly select a solution that already exists in the archive
                occupied_indices = np.argwhere(self.fitness_table > -np.inf)
                random_idx = occupied_indices[np.random.choice(len(occupied_indices))]
                parent = self.solution_params_table[random_idx[0], random_idx[1]]

                # Mutate: Add a small Gaussian nudge (Sigma 0.05 = 5% of range)
                # This allows us to 'crawl' from a known good cell into a neighbor
                x_new = np.clip(parent + np.random.normal(0, 0.05, 2), 0, 1)

            # --- 2. EVALUATION ---
            # We use your 'local_obj' but flip the sign because MAP-Elites maximizes fitness
            # Note: Ensure local_obj uses the Conservative Mean (mu - 1.96*sigma)
            fitness = -local_obj(x_new, slos, gp, ordered_bounds)

            # --- 3. THE COMPETITION (The 'Elite' part) ---
            bin_idx = self.get_bin(x_new)
            row, col = bin_idx[0], bin_idx[1]

            # Check if this new solution is better than what's currently in that niche
            if fitness > self.fitness_table[row, col]:
                self.fitness_table[row, col] = fitness
                self.solution_params_table[row, col] = x_new

                if i % 100 == 0:
                    print(f"Iteration {i}: Discovered new Elite in bin {row, col} with fitness {fitness:.4f}")

    def get_diverse_set(self, n_solutions=5, versatility=0.2):
        """
        :param n_solutions: How many candidates to return.
        :param versatility: 0.0 (only optimal) to 1.0 (maximum spread).
                            Acts as the 'exclusion radius' in the normalized 0-1 space.
        """
        # 1. Flatten the archive into a list of valid candidates
        candidates = []
        for r in range(self.bins):
            for c in range(self.bins):
                if self.fitness_table[r, c] > -np.inf:
                    candidates.append({
                        'coord': self.solution_params_table[r, c],
                        'fitness': self.fitness_table[r, c]
                    })

        # Sort by fitness descending (best first)
        candidates = sorted(candidates, key=lambda x: x['fitness'], reverse=True)

        selected = []
        if not candidates:
            return selected

        # 2. Iterative Selection with Distance Constraint
        for cand in candidates:
            if len(selected) >= n_solutions:
                break

            # Check distance against all currently selected solutions
            is_diverse_enough = True
            for sel in selected:
                dist = np.linalg.norm(cand['coord'] - sel['coord'])
                if dist < versatility:
                    is_diverse_enough = False
                    break

            if is_diverse_enough:
                selected.append(cand)

        return selected