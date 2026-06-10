from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


@dataclass(frozen=True)
class BevGeometry:
    roi_min: np.ndarray
    roi_max: np.ndarray
    resolution_m_per_px: float
    width_px: int
    height_px: int

    @staticmethod
    def from_values(
        roi_min: list[float],
        roi_max: list[float],
        resolution_m_per_px: float,
    ) -> "BevGeometry":
        min_values = np.asarray(roi_min, dtype=np.float32)
        max_values = np.asarray(roi_max, dtype=np.float32)

        width_px = int(math.ceil((float(max_values[0]) - float(min_values[0])) / resolution_m_per_px))
        height_px = int(math.ceil((float(max_values[1]) - float(min_values[1])) / resolution_m_per_px))

        return BevGeometry(
            roi_min=min_values,
            roi_max=max_values,
            resolution_m_per_px=float(resolution_m_per_px),
            width_px=width_px,
            height_px=height_px,
        )

    @staticmethod
    def from_roi(
        roi_min: list[float],
        roi_max: list[float],
        resolution_m_per_px: float,
    ) -> "BevGeometry":
        return BevGeometry.from_values(roi_min, roi_max, resolution_m_per_px)

    def expand_xy(self, padding_x_m: float, padding_y_m: float) -> "BevGeometry":
        padding_x_m = max(float(padding_x_m), 0.0)
        padding_y_m = max(float(padding_y_m), 0.0)

        expanded_min = self.roi_min.copy()
        expanded_max = self.roi_max.copy()
        expanded_min[0] -= padding_x_m
        expanded_max[0] += padding_x_m
        expanded_min[1] -= padding_y_m
        expanded_max[1] += padding_y_m

        return BevGeometry.from_values(
            roi_min=expanded_min.tolist(),
            roi_max=expanded_max.tolist(),
            resolution_m_per_px=self.resolution_m_per_px,
        )

    def filter_points_to_roi(self, points_xyz: np.ndarray) -> np.ndarray:
        if points_xyz.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        x_values = points_xyz[:, 0]
        y_values = points_xyz[:, 1]
        z_values = points_xyz[:, 2]

        roi_mask = (
            (x_values >= self.roi_min[0])
            & (x_values <= self.roi_max[0])
            & (y_values >= self.roi_min[1])
            & (y_values <= self.roi_max[1])
            & (z_values >= self.roi_min[2])
            & (z_values <= self.roi_max[2])
        )

        return points_xyz[roi_mask].astype(np.float32, copy=False)

    def points_to_pixels(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        filtered_points = self.filter_points_to_roi(points_xyz)

        if filtered_points.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                filtered_points,
            )

        u_values = np.floor((filtered_points[:, 0] - self.roi_min[0]) / self.resolution_m_per_px).astype(np.int32)
        v_values = np.floor((self.roi_max[1] - filtered_points[:, 1]) / self.resolution_m_per_px).astype(np.int32)

        image_mask = (
            (u_values >= 0)
            & (u_values < self.width_px)
            & (v_values >= 0)
            & (v_values < self.height_px)
        )

        return u_values[image_mask], v_values[image_mask], filtered_points[image_mask]

    def world_to_pixel(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.points_to_pixels(points_xyz)

    def pixel_to_world_xy(self, pixel_xy: np.ndarray) -> np.ndarray:
        pixel_xy = np.asarray(pixel_xy, dtype=np.float32)

        u_values = pixel_xy[:, 0]
        v_values = pixel_xy[:, 1]

        x_values = self.roi_min[0] + (u_values + 0.5) * self.resolution_m_per_px
        y_values = self.roi_max[1] - (v_values + 0.5) * self.resolution_m_per_px

        return np.stack((x_values, y_values), axis=1).astype(np.float32)


class PointCloudReader:
    @staticmethod
    def to_xyz_array(msg: PointCloud2) -> np.ndarray:
        if hasattr(point_cloud2, "read_points_numpy"):
            try:
                points = point_cloud2.read_points_numpy(
                    msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )

                if isinstance(points, np.ndarray):
                    if points.dtype.names:
                        xyz = np.vstack((points["x"], points["y"], points["z"])).T
                    else:
                        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)

                    return PointCloudReader.filter_finite_xyz(xyz)
            except Exception:
                pass

        points_iter = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        xyz = np.asarray([[point[0], point[1], point[2]] for point in points_iter], dtype=np.float32)
        return PointCloudReader.filter_finite_xyz(xyz)

    @staticmethod
    def filter_finite_xyz(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        finite_mask = np.all(np.isfinite(points), axis=1)
        return points[finite_mask]


class BevImageBuilder:
    def __init__(self, geometry: BevGeometry, density_clip_count: int) -> None:
        self.geometry = geometry
        self.density_clip_count = max(int(density_clip_count), 1)

    def build(self, points_xyz: np.ndarray) -> tuple[np.ndarray, dict]:
        u_values, v_values, filtered_points = self.geometry.points_to_pixels(points_xyz)

        occupancy = np.zeros((self.geometry.height_px, self.geometry.width_px), dtype=np.uint8)
        height_image = np.zeros_like(occupancy)
        density_counts = np.zeros((self.geometry.height_px, self.geometry.width_px), dtype=np.uint16)

        if filtered_points.size == 0:
            image = np.dstack((occupancy, height_image, occupancy))
            return image, self.build_stats(points_xyz, filtered_points, occupancy, density_counts)

        occupancy[v_values, u_values] = 255
        np.add.at(density_counts, (v_values, u_values), 1)

        z_min = float(self.geometry.roi_min[2])
        z_max = float(self.geometry.roi_max[2])
        z_range = max(z_max - z_min, 1e-6)

        z_normalized = np.clip(((filtered_points[:, 2] - z_min) / z_range) * 255.0, 0.0, 255.0).astype(np.uint8)
        np.maximum.at(height_image, (v_values, u_values), z_normalized)

        density_normalized = np.clip(
            (np.log1p(density_counts.astype(np.float32)) / math.log1p(self.density_clip_count)) * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        image = np.dstack((occupancy, height_image, density_normalized))
        return image, self.build_stats(points_xyz, filtered_points, occupancy, density_counts)

    def build_stats(
        self,
        raw_points: np.ndarray,
        filtered_points: np.ndarray,
        occupancy: np.ndarray,
        density_counts: np.ndarray,
    ) -> dict:
        return {
            "points_raw": int(raw_points.shape[0]),
            "points_in_roi": int(filtered_points.shape[0]),
            "occupied_pixels": int(np.count_nonzero(occupancy)),
            "max_density_per_pixel": int(density_counts.max()) if density_counts.size else 0,
        }
