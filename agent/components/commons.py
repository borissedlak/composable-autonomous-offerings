from enum import Enum
from typing import NamedTuple

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
