from enum import Enum
from typing import NamedTuple, Dict, Tuple

from dataclasses import dataclass
from typing import List


class ServiceType(Enum):
    QR = "elastic-workbench-qr-detector"
    CV = "elastic-workbench-cv-analyzer"
    PC = "elastic-workbench-pc-visualizer"
    LS = "elastic-workbench-linked-service"
    UNKNOWN = "unknown"

@dataclass
class ServiceFeatureMapping:
    service_type: ServiceType
    feature_indices: List[int]

class ServiceVar(Enum):
    QUALITY = "data_quality"
    PERFORMANCE = "max_tp"
    COST = "cores"
    MODEL = "model_size"


class ESType(Enum):
    STARTUP = 'startup'
    QUALITY_SCALE = 'quality_scaling'
    PARALLELISM_SCALE = 'parallelism_scaling'
    RESOURCE_SCALE = 'resource_scaling'
    MODEL_SCALE = 'model_scaling'
    RESOURCE_SWAP = 'resource_swapping'
    OFFLOADING = 'offloading'
    IDLE = 'idle'
    UNKNOWN = 'unknown'


class ServiceID(NamedTuple):
    host: str
    service_type: ServiceType
    container_id: str
    port: str = 8080

theoretical_param_bounds: Dict[ServiceType, Dict[ServiceVar, Tuple[float, float]]] = {
    ServiceType.QR: {
        ServiceVar.COST: (1.0, 8.0),
        ServiceVar.QUALITY: (100.0, 1000.0),
    },
    ServiceType.CV: {
        ServiceVar.COST: (1.0, 8.0),
        ServiceVar.QUALITY: (128.0, 320.0),
        ServiceVar.MODEL: (1.0, 4.0),
    },
    ServiceType.PC: {
        ServiceVar.COST: (1.0, 8.0),
        ServiceVar.QUALITY: (6.0, 60.0),
    }
}

FIG_SIZE_SINGLE = (6, 3)

SERVICE_STYLE = {
    'elastic-workbench-qr-detector': {'color': '#d62728', 'linestyle': '-', 'marker': 'o'},
    'elastic-workbench-cv-analyzer': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's'},
    'elastic-workbench-pc-visualizer': {'color': '#2ca02c', 'linestyle': ':', 'marker': 'D'}
}

from agent.components.SLORegistry_v2 import SLO_Registry

slo_lib = SLO_Registry("../statics/config/service_level_objectives.yml")
_slos_default = slo_lib.get_slo_for_client("experiment-1", "default")
_slos_high_perf = slo_lib.get_slo_for_client("experiment-1", "high_perf")
_slos_low_cost = slo_lib.get_slo_for_client("experiment-1", "low_cost")
_slos_high_quality = slo_lib.get_slo_for_client("experiment-1", "high_quality")

class SloSet(Enum):
    DEFAULT = _slos_default
    HIGH_PERF = _slos_high_perf
    LOW_COST = _slos_low_cost
    HIGH_QUALITY = _slos_high_quality