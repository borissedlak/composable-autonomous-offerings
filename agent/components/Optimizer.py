from typing import Dict

from scipy.optimize import minimize

from agent.components.GaussianProcess import GASK, get_empirical_boundaries
from agent.components.SLORegistry_v2 import calculate_weighted_SLO_F
from agent.components.commons import ServiceType


# TODO:
#  1. Translate SLO weight to variable min/max; simply aim for max and put weight: Check
#  2. Evaluate fitness (= SLO fulfillment) of one solution: Check
#  3. Extract one optimal value: Check
#  4. Relax :-) and extract entire front


def local_obj(x_norm, slos: Dict[str, float], gp: GASK, ordered_bounds):
    """
        :param x_norm: values that are NOT normalized
        :param slos: just weights, no thresholds
        :param gp:
        :param ordered_bounds:
        :return: SLO fulfillment
        """

    # Translate 0-1 back to Real Units
    x_real = []
    for i, (mini, maxi) in enumerate(ordered_bounds):
        x_real.append(x_norm[i] * (maxi - mini) + mini)

    x_state = {'cores': x_real[0], 'data_quality': x_real[1]}

    # Now the GP receives the units it expects (or uses its internal scaler)
    mu, sigma = gp.predict(ServiceType.QR, "max_tp", x_state)
    conservative_max_tp = mu - 1.96 * sigma # TODO: Try to understand better what this does
    max_tp = {'max_tp': conservative_max_tp}

    empirical_boundaries = get_empirical_boundaries(gp.training_data)[ServiceType.QR]
    slo_f = calculate_weighted_SLO_F(x_state | max_tp, slos, empirical_boundaries)

    print(f"Calculated SLO-F for {x_state}: {slo_f}")
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
                      args=(slos, gp, ordered_bounds), options={'maxiter': 150}) # TODO: Read more about epsilon step

    if not result.success:
        raise RuntimeWarning("Solver failed: " + result.message)

    # Convert the optimal 0-1 answer BACK to real units
    final_x = []
    for i, (mini, maxi) in enumerate(ordered_bounds):
        final_x.append(result.x[i] * (maxi - mini) + mini)

    return final_x
