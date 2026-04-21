from typing import Dict

import yaml

from agent.components.commons import ServiceVar
from utils import smoothstep


class SLO_Registry:
    def __init__(self, slo_config_path):
        with open(slo_config_path, "r") as f:
            self.slo_lib = yaml.safe_load(f)

    def get_slo_for_client(self, experiment_id: str, client_id: str) -> Dict[ServiceVar, float]:
        result = {}
        client_data = self.slo_lib.get(experiment_id, {}).get(client_id, {})

        for var_name, weight in client_data.items():
            result[ServiceVar(var_name)] = float(weight)

        return result


def calculate_weighted_SLO_F(full_state, slos: Dict[ServiceVar, float], empirical_boundaries):
    if sum(slos.values()) != 1.0:
        raise RuntimeError("The sum of the SLO weights must equal 1")

    local_slos = slos.copy()
    if 'model_size' in full_state.keys():  # We're having the CV service
        original_quality_slo = slos[ServiceVar.QUALITY]
        local_slos[ServiceVar.QUALITY] = original_quality_slo * 0.5
        local_slos[ServiceVar.MODEL] = original_quality_slo * 0.5

    weighted_slo_f = 0.0
    for slo_var, slo_weight in local_slos.items():
        slo_var_text = slo_var.value

        # TODO: Refactor, this is the str value of the slo variable name
        if slo_var_text not in full_state:
            raise RuntimeError(f"Missing variable '{slo_var_text}' in the state for evaluating SLOs")
        var_value = full_state[slo_var_text]

        threshold = empirical_boundaries[slo_var_text][1]
        slo_value = var_value / threshold

        if slo_var_text == 'cores': # We want low cost!!
            slo_value = 1 - slo_value

        slo_f_single_slo = float(smoothstep(slo_value) * slo_weight)

        if slo_f_single_slo > 1.0:
            raise RuntimeError(f"How can '{slo_var}' exceed the maximum known value?")

        weighted_slo_f += slo_f_single_slo
        # print(value, slo_weight)

    # Heavily penalize if we don't have any output
    if full_state["max_tp"] < 1.0:
        return weighted_slo_f * 0.1

    return weighted_slo_f
