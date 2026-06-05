from dataclasses import dataclass
import numpy as np


@dataclass
class ClusterInfo:
    cluster_id: int
    centroid: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray
    voxel_count: int
    cluster_type: str = 'normal'


@dataclass
class Track:
    track_id: int
    centroid: np.ndarray
    velocity: np.ndarray
    age: int
    missed_updates: int
    last_stamp_sec: float
    barcode_id: str = ''
    class_id: int = -1
    class_name: str = ''
    state: str = 'confirmed'
    confidence: float = 0.0
    yaw: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    hit_count: int = 0
    source_missed_count: int = 0
    frame_id: str = ''
