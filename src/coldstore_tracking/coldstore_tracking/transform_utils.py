from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


PoseTuple = Tuple[float, float, float, float, float, float]


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return rz @ ry @ rx


def pose_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = euler_to_rotation_matrix(roll, pitch, yaw)
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return matrix


def transform_points(points_xyz: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz

    rotated = points_xyz @ transform_matrix[:3, :3].T
    translated = rotated + transform_matrix[:3, 3]
    return translated.astype(np.float32, copy=False)


def build_single_sensor_transform(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    return pose_to_matrix(x, y, z, roll, pitch, yaw)


def build_sensor_transform_map() -> Dict[str, np.ndarray]:
    # Derived from the original simulation world poses.
    # Pose order: x, y, z, roll, pitch, yaw.
    sensor_poses: Dict[str, PoseTuple] = {
        'lidar_01/link/s': (-7.7, 3.1, 4.0, 0.0, math.pi, 0.0),
        'lidar_02/link/s': (0.0, 3.1, 4.0, 0.0, math.pi, 0.0),
        'lidar_03/link/s': (7.7, 3.1, 4.0, 0.0, math.pi, 0.0),
        'lidar_04/link/s': (-7.7, -3.1, 4.0, 0.0, math.pi, 0.0),
        'lidar_05/link/s': (0.0, -3.1, 4.0, 0.0, math.pi, 0.0),
        'lidar_06/link/s': (7.7, -3.1, 4.0, 0.0, math.pi, 0.0),
    }

    return {frame_name: pose_to_matrix(*pose) for frame_name, pose in sensor_poses.items()}
