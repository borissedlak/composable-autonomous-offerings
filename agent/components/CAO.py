from typing import List, Dict

import numpy as np
import pandas as pd

from agent.components.GaussianProcess import GASK, get_ordered_boundaries
from agent.components.Optimizer import VersatileMapElites
from agent.components.SLORegistry_v2 import SLO_Registry
from agent.components.commons import ServiceType, ServiceVar


class CAO:
    def __init__(self, service_type: ServiceType):
        self.service_type = service_type
        self.upstream_CAOs: List[CAO] = []
        self.downstream_CAOs: List[CAO] = []  # Might be omitted, to just have a single-linked list

        slo_lib = SLO_Registry("../statics/config/service_level_objectives.yml")
        slos = slo_lib.get_slo_for_client("experiment-1", "client-1")

        self.model: GASK = GASK(service_type, show_figures=False)
        self.slos: Dict[ServiceVar, float] = slos

        # TODO: Actually, the config is only defined by the actionable params
        self.config: Dict[ServiceVar, float] = get_random_CAO_config()
        self.pfo: VersatileMapElites = VersatileMapElites(self.service_type, bins=10)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.service_type}"

    def add_upstream_CAO(self, cao):
        self.upstream_CAOs.append(cao)
        cao.add_downstream_CAO(self)

    def add_downstream_CAO(self, cao):
        self.downstream_CAOs.append(cao)

    # TODO: Create the local fitness as a combination from upstream services and the local one
    def get_CAO_fitness(self):
        # upstream_fitness = 0.0
        # for upstream_cao in self.upstream_CAOs:
        #     upstream_fitness += upstream_cao.get_CAO_fitness()

        print("Cumulative cost:", self.get_cumulative_cost())
        print("Min performance:", self.get_min_performance())
        print("Min quality:", self.get_min_quality())

        return 0.0

    # TODO: Add data to GP, recalculate it, and propagate changes (by doing xyz)
    def integrate_observations(self, new_obs: pd.DataFrame, data_density=1.0):

        training_df = new_obs if self.model.training_data is None else (self.model.training_data + new_obs)
        self.model.init_model(training_df, data_density)

        ordered_bounds = get_ordered_boundaries(self.model)
        self.pfo.run_search(self.slos, self.model, ordered_bounds, iterations=800)
        self.pfo.get_diverse_set()
        # visualize_ndarray(self.pfo.fitness_table)


    def get_cumulative_cost(self):
        return self.config[ServiceVar.COST] + sum(cao.get_cumulative_cost() for cao in self.upstream_CAOs)

    def get_min_performance(self):
        local_perf = self.config[ServiceVar.PERFORMANCE]
        for upstream in self.upstream_CAOs:
            local_perf = min(local_perf, upstream.get_min_performance())
        return local_perf

    def get_min_quality(self):
        local_quality = self.config[ServiceVar.QUALITY]
        for upstream in self.upstream_CAOs:
            local_quality = min(local_quality, upstream.get_min_quality())
        return local_quality


def get_random_CAO_config() -> Dict[ServiceVar, float]:
    keys = [ServiceVar.QUALITY, ServiceVar.PERFORMANCE, ServiceVar.COST]
    rng = np.random.default_rng()
    values = rng.dirichlet(alpha=[1.0, 1.0, 1.0])

    return {k: float(v) for k, v in zip(keys, values)}


# if __name__ == "__main__":
#     producer: CAO = CAO(ServiceType.CV)
#     worker: CAO = CAO(ServiceType.QR)
#     consumer: CAO = CAO(ServiceType.PC)
#
#     consumer.add_upstream_CAO(worker)
#     worker.add_upstream_CAO(producer)
#
#     producer.integrate_observations(data_density=0.1)
#     worker.integrate_observations(data_density=0.1)
#     consumer.integrate_observations(data_density=0.1)
#
#     pass
