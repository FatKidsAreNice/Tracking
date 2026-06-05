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
