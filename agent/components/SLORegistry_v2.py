from typing import Dict

import yaml

from utils import smoothstep


class SLO_Registry:
    def __init__(self, slo_config_path):
        with open(slo_config_path, "r") as f:
            self.slo_lib = yaml.safe_load(f)

    def get_slo_for_client(self, experiment_id: str, client_id: str) -> Dict[str, float]:
        result = {}
        client_data = self.slo_lib.get(experiment_id, {}).get(client_id, {})

        for var_name, weight in client_data.items():
            result[var_name] = float(weight)

        return result


def calculate_weighted_SLO_F(full_state, slos: Dict[str, float], empirical_boundaries):
    if sum(slos.values()) != 1.0:
        raise RuntimeError("The sum of the SLO weights must equal 1")

    weighted_slo_f = 0.0
    for slo_var, slo_weight in slos.items():

        if slo_var not in full_state:
            raise RuntimeError(f"Missing variable '{slo_var}' in the state for evaluating SLOs")
        var_value = full_state[slo_var]

        threshold = empirical_boundaries[slo_var][1]
        slo_value = var_value / threshold

        slo_f_single_slo = float(smoothstep(slo_value) * slo_weight)

        if slo_f_single_slo > 1.0:
            raise RuntimeError(f"How can '{slo_var}' exceed the maximum known value?")

        weighted_slo_f += slo_f_single_slo
        # print(value, slo_weight)

    return weighted_slo_f
