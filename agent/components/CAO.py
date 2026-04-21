from typing import List, Dict

import numpy as np
import pandas as pd

from agent.components.GaussianProcess import GASK
from agent.components.commons import ServiceType, ServiceVar


# TODO: The idea is that we have some recursive CAO structure and from it I can infer things

class CAO:
    def __init__(self, service_type: ServiceType):
        self.service_type = service_type
        self.upstream_CAOs: List[CAO] = []
        self.downstream_CAOs: List[CAO] = [] # Might be omitted, to just have a single-linked list

        self.model: GASK  # 1. Based on the experience
        self.slos: Dict[ServiceVar, float]  # 2. and the SLOs
        self.fitness_table = None  # 3. you can calculate the best configs

        self.config: Dict[ServiceVar, float] = get_random_CAO_config()

        df = pd.read_csv("../../statics/metrics_20_0.csv")
        gp = GASK(service_type, show_figures=False)
        gp.init_models(df, data_density=1.0)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.service_type}"

    # I can assume that they are already wired in some sense!
    def add_upstream_CAO(self, cao):
        self.upstream_CAOs.append(cao)

    def add_downstream_CAO(self, cao):
        self.downstream_CAOs.append(cao)
        cao.add_upstream_CAO(self)

    # TODO: Create the local fitness as a combination from upstream services and the local one
    def get_local_fitness(self):
        pass

    # TODO: Add data to GP, recalculate it, and propagate changes (by doing xyz)
    def integrate_observations(self, new_obs: pd.DataFrame):
        pass

    # TODO: Get the values from all upstream services
    def get_cumulative_cost(self):
        pass

    def get_max_latency(self):
        pass

    def get_min_quality(self):
        pass

def get_random_CAO_config() -> Dict[ServiceVar, float]:
    keys = [ServiceVar.QUALITY, ServiceVar.TIME, ServiceVar.COST]
    values = np.random.dirichlet(alpha=[1.0, 1.0, 1.0])

    return {k: float(v) for k, v in zip(keys, values)}


if __name__ == "__main__":
    producer: CAO = CAO(ServiceType.CV)
    # worker: CAO = CAO(ServiceType.QR)
    # consumer: CAO = CAO(ServiceType.PC)

    # producer.add_downstream_CAO(worker)
    # worker.add_downstream_CAO(consumer)

    pass
