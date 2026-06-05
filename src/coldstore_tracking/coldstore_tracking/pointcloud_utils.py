from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


VoxelKey = Tuple[int, int, int]


def extract_xyz_points(cloud_msg: PointCloud2) -> np.ndarray:
    points = point_cloud2.read_points_numpy(
        cloud_msg,
        field_names=['x', 'y', 'z'],
        skip_nans=True,
    )

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    if points.ndim == 1:
        points = points.reshape(-1, 3)

    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]

    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    return points.astype(np.float32, copy=False)


def crop_points(points_xyz: np.ndarray, roi_min: np.ndarray, roi_max: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    finite_mask = np.all(np.isfinite(points_xyz), axis=1)
    points_xyz = points_xyz[finite_mask]

    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    mask = np.all((points_xyz >= roi_min) & (points_xyz <= roi_max), axis=1)
    return points_xyz[mask].astype(np.float32, copy=False)


def voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    finite_mask = np.all(np.isfinite(points_xyz), axis=1)
    points_xyz = points_xyz[finite_mask]

    if points_xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if voxel_size <= 0.0:
        return points_xyz.astype(np.float32, copy=False)

    voxel_indices = np.floor(points_xyz / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    unique_idx.sort()

    return points_xyz[unique_idx].astype(np.float32, copy=False)


def cloud_header(frame_id: str, stamp: Time) -> Header:
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp
    return header


def create_xyz_cloud(frame_id: str, stamp: Time, points_xyz: np.ndarray) -> PointCloud2:
    header = cloud_header(frame_id, stamp)
    return point_cloud2.create_cloud_xyz32(header, points_xyz.tolist())


def points_to_voxel_keys(points_xyz: np.ndarray, voxel_size: float) -> Set[VoxelKey]:
    if points_xyz.size == 0:
        return set()

    finite_mask = np.all(np.isfinite(points_xyz), axis=1)
    points_xyz = points_xyz[finite_mask]

    if points_xyz.size == 0:
        return set()

    if voxel_size <= 0.0:
        voxel_size = 1.0

    voxel_indices = np.floor(points_xyz / voxel_size).astype(np.int32)
    unique = np.unique(voxel_indices, axis=0)

    return {tuple(int(v) for v in row) for row in unique}


def voxel_keys_to_centers(voxel_keys: Iterable[VoxelKey], voxel_size: float) -> np.ndarray:
    keys_list: List[VoxelKey] = list(voxel_keys)

    if not keys_list:
        return np.empty((0, 3), dtype=np.float32)

    if voxel_size <= 0.0:
        voxel_size = 1.0

    centers = (np.asarray(keys_list, dtype=np.float32) + 0.5) * voxel_size
    return centers.astype(np.float32, copy=False)


def bounds_from_points(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        zero = np.zeros(3, dtype=np.float32)
        return zero, zero

    finite_mask = np.all(np.isfinite(points_xyz), axis=1)
    points_xyz = points_xyz[finite_mask]

    if points_xyz.size == 0:
        zero = np.zeros(3, dtype=np.float32)
        return zero, zero

    return (
        points_xyz.min(axis=0).astype(np.float32, copy=False),
        points_xyz.max(axis=0).astype(np.float32, copy=False),
    )