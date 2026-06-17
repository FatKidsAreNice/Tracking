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
    motion_state: str = 'newly_appeared'
    static_streak: int = 0
    moving_streak: int = 0
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    last_confirmed_sec: float = 0.0
    lost_transition_count: int = 0
    occluded_transition_count: int = 0
    reappeared_count: int = 0
    last_motion_state_change_sec: float = 0.0
    source_track_id: int = 0
    last_source_track_id: int = 0
    identity_recovered_count: int = 0
    identity_confidence: float = 0.0
    identity_state: str = 'new'
    last_strict_identity_match: bool = False
