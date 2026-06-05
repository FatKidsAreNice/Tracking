from __future__ import annotations

import math

from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .pointcloud_utils import create_xyz_cloud, extract_xyz_points, points_to_voxel_keys, voxel_keys_to_centers
from .tracking_types import ClusterInfo

VoxelKey = Tuple[int, int, int]


@dataclass
class ClusterCandidate:
    index: int
    keys: List[VoxelKey]
    info: ClusterInfo


class ClusterDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('cluster_detector_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('detection_mode', 'background_subtraction')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('min_cluster_voxels', 80)
        self.declare_parameter('max_cluster_voxels', 7000)
        self.declare_parameter('min_cluster_size', [0.30, 0.12, 0.06])
        self.declare_parameter('max_cluster_size', [1.30, 1.10, 1.00])
        self.declare_parameter('min_cluster_height', -0.60)
        self.declare_parameter('max_cluster_height', 2.40)

        self.declare_parameter('background_capture_duration_sec', 5.0)
        self.declare_parameter('background_min_occupancy_ratio', 0.85)
        self.declare_parameter('background_expansion_voxels', 1)
        self.declare_parameter('background_expansion_axes', [1, 1, 0])
        self.declare_parameter('publish_only_clustered_dynamic', True)
        self.declare_parameter('keep_largest_cluster_only', False)
        self.declare_parameter('publish_oriented_cluster_markers', True)
        self.declare_parameter('oriented_marker_min_voxels', 10)
        self.declare_parameter('oriented_marker_min_scale', 0.02)
        self.declare_parameter('oriented_marker_use_fixed_size', True)
        self.declare_parameter('oriented_marker_fixed_size', [1.30, 1.30, 2.00])
        self.declare_parameter('oriented_marker_center_mode', 'robust_centroid')
        self.declare_parameter('top_footprint_z_keep_ratio', 0.25)
        self.declare_parameter('top_footprint_bounds_quantile', 0.05)
        self.declare_parameter('top_footprint_min_points', 20)
        self.declare_parameter('top_footprint_z_center_mode', 'measured_mid')
        self.declare_parameter('oriented_marker_yaw_mode', 'visible_edge')
        self.declare_parameter('visible_edge_keep_ratio', 0.30)
        self.declare_parameter('visible_edge_sensor_position', [0.0, 0.0, 0.0])
        self.declare_parameter('object_center_smoothing_enabled', True)
        self.declare_parameter('object_center_smoothing_alpha', 0.15)
        self.declare_parameter('object_center_max_step', 0.08)
        self.declare_parameter('oriented_marker_yaw_smoothing_enabled', True)
        self.declare_parameter('oriented_marker_yaw_smoothing_alpha', 0.20)
        self.declare_parameter('oriented_marker_pca_keep_ratio', 0.90)
        self.declare_parameter('oriented_marker_bounds_quantile', 0.03)
        self.declare_parameter('oriented_marker_min_inlier_points', 20)

        self.declare_parameter('background_storage_path', '~/ros2_ws/backgrounds/latest_background.npz')
        self.declare_parameter('auto_save_background', True)
        self.declare_parameter('auto_load_background', True)

        self.declare_parameter('allow_touched_clusters', True)
        self.declare_parameter('touched_min_cluster_voxels', 60)
        self.declare_parameter('touched_max_cluster_voxels', 25000)
        self.declare_parameter('touched_min_cluster_size', [0.20, 0.08, 0.05])
        self.declare_parameter('touched_max_cluster_size', [2.50, 2.00, 2.20])
        self.declare_parameter('publish_touched_markers', True)

        self.declare_parameter('allow_footprint_projection', True)
        self.declare_parameter('footprint_min_cluster_voxels', 20)
        self.declare_parameter('footprint_max_cluster_voxels', 20000)
        self.declare_parameter('footprint_min_size', [0.20, 0.20, 0.01])
        self.declare_parameter('footprint_max_size', [1.40, 1.40, 0.70])
        self.declare_parameter('footprint_min_height', -0.20)
        self.declare_parameter('footprint_max_height', 2.40)
        self.declare_parameter('projection_floor_z', -0.50)
        self.declare_parameter('projection_height', 1.30)
        self.declare_parameter('publish_footprint_markers', True)

        self.declare_parameter('allow_paired_clusters', True)
        self.declare_parameter('pair_min_lower_cluster_voxels', 15)
        self.declare_parameter('pair_max_lower_cluster_voxels', 12000)
        self.declare_parameter('pair_min_upper_cluster_voxels', 15)
        self.declare_parameter('pair_max_upper_cluster_voxels', 16000)
        self.declare_parameter('pair_min_lower_size', [0.20, 0.20, 0.01])
        self.declare_parameter('pair_max_lower_size', [1.50, 1.50, 0.45])
        self.declare_parameter('pair_min_upper_size', [0.20, 0.20, 0.01])
        self.declare_parameter('pair_max_upper_size', [1.50, 1.50, 0.80])
        self.declare_parameter('pair_lower_min_height', -0.70)
        self.declare_parameter('pair_lower_max_height', 0.80)
        self.declare_parameter('pair_upper_min_height', 0.20)
        self.declare_parameter('pair_upper_max_height', 2.40)
        self.declare_parameter('pair_min_z_separation', 0.25)
        self.declare_parameter('pair_max_z_separation', 2.20)
        self.declare_parameter('pair_max_xy_center_distance', 0.75)
        self.declare_parameter('pair_min_xy_overlap_ratio', 0.10)
        self.declare_parameter('pair_project_to_rack_height', True)
        self.declare_parameter('pair_projection_floor_z', -0.50)
        self.declare_parameter('pair_projection_height', 1.30)
        self.declare_parameter('publish_pair_markers', True)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.detection_mode = str(self.get_parameter('detection_mode').value).strip().lower()
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.min_cluster_voxels = int(self.get_parameter('min_cluster_voxels').value)
        self.max_cluster_voxels = int(self.get_parameter('max_cluster_voxels').value)
        self.min_cluster_size = np.asarray(self.get_parameter('min_cluster_size').value, dtype=np.float32)
        self.max_cluster_size = np.asarray(self.get_parameter('max_cluster_size').value, dtype=np.float32)
        self.min_cluster_height = float(self.get_parameter('min_cluster_height').value)
        self.max_cluster_height = float(self.get_parameter('max_cluster_height').value)

        self.background_capture_duration_sec = float(self.get_parameter('background_capture_duration_sec').value)
        self.background_min_occupancy_ratio = float(self.get_parameter('background_min_occupancy_ratio').value)
        self.background_expansion_voxels = int(self.get_parameter('background_expansion_voxels').value)
        self.background_expansion_axes = [int(value) for value in self.get_parameter('background_expansion_axes').value]
        self.publish_only_clustered_dynamic = bool(self.get_parameter('publish_only_clustered_dynamic').value)
        self.keep_largest_cluster_only = bool(self.get_parameter('keep_largest_cluster_only').value)
        self.publish_oriented_cluster_markers = bool(self.get_parameter('publish_oriented_cluster_markers').value)
        self.oriented_marker_min_voxels = int(self.get_parameter('oriented_marker_min_voxels').value)
        self.oriented_marker_min_scale = float(self.get_parameter('oriented_marker_min_scale').value)
        self.oriented_marker_use_fixed_size = bool(self.get_parameter('oriented_marker_use_fixed_size').value)
        self.oriented_marker_fixed_size = [float(value) for value in self.get_parameter('oriented_marker_fixed_size').value]
        self.oriented_marker_center_mode = str(self.get_parameter('oriented_marker_center_mode').value).strip().lower()
        self.top_footprint_z_keep_ratio = float(self.get_parameter('top_footprint_z_keep_ratio').value)
        self.top_footprint_bounds_quantile = float(self.get_parameter('top_footprint_bounds_quantile').value)
        self.top_footprint_min_points = int(self.get_parameter('top_footprint_min_points').value)
        self.top_footprint_z_center_mode = str(self.get_parameter('top_footprint_z_center_mode').value).strip().lower()
        self.oriented_marker_yaw_mode = str(self.get_parameter('oriented_marker_yaw_mode').value).strip().lower()
        self.visible_edge_keep_ratio = float(self.get_parameter('visible_edge_keep_ratio').value)
        self.visible_edge_sensor_position = [float(value) for value in self.get_parameter('visible_edge_sensor_position').value]
        self.object_center_smoothing_enabled = bool(self.get_parameter('object_center_smoothing_enabled').value)
        self.object_center_smoothing_alpha = float(self.get_parameter('object_center_smoothing_alpha').value)
        self.object_center_max_step = float(self.get_parameter('object_center_max_step').value)
        self.oriented_marker_yaw_smoothing_enabled = bool(self.get_parameter('oriented_marker_yaw_smoothing_enabled').value)
        self.oriented_marker_yaw_smoothing_alpha = float(self.get_parameter('oriented_marker_yaw_smoothing_alpha').value)
        self.smoothed_object_center = None
        self.smoothed_oriented_yaw = None
        self.oriented_marker_pca_keep_ratio = float(self.get_parameter('oriented_marker_pca_keep_ratio').value)
        self.oriented_marker_bounds_quantile = float(self.get_parameter('oriented_marker_bounds_quantile').value)
        self.oriented_marker_min_inlier_points = int(self.get_parameter('oriented_marker_min_inlier_points').value)

        self.background_storage_path = str(self.get_parameter('background_storage_path').value)
        self.auto_save_background = bool(self.get_parameter('auto_save_background').value)
        self.auto_load_background = bool(self.get_parameter('auto_load_background').value)

        self.allow_touched_clusters = bool(self.get_parameter('allow_touched_clusters').value)
        self.touched_min_cluster_voxels = int(self.get_parameter('touched_min_cluster_voxels').value)
        self.touched_max_cluster_voxels = int(self.get_parameter('touched_max_cluster_voxels').value)
        self.touched_min_cluster_size = np.asarray(self.get_parameter('touched_min_cluster_size').value, dtype=np.float32)
        self.touched_max_cluster_size = np.asarray(self.get_parameter('touched_max_cluster_size').value, dtype=np.float32)
        self.publish_touched_markers = bool(self.get_parameter('publish_touched_markers').value)

        self.allow_footprint_projection = bool(self.get_parameter('allow_footprint_projection').value)
        self.footprint_min_cluster_voxels = int(self.get_parameter('footprint_min_cluster_voxels').value)
        self.footprint_max_cluster_voxels = int(self.get_parameter('footprint_max_cluster_voxels').value)
        self.footprint_min_size = np.asarray(self.get_parameter('footprint_min_size').value, dtype=np.float32)
        self.footprint_max_size = np.asarray(self.get_parameter('footprint_max_size').value, dtype=np.float32)
        self.footprint_min_height = float(self.get_parameter('footprint_min_height').value)
        self.footprint_max_height = float(self.get_parameter('footprint_max_height').value)
        self.projection_floor_z = float(self.get_parameter('projection_floor_z').value)
        self.projection_height = float(self.get_parameter('projection_height').value)
        self.publish_footprint_markers = bool(self.get_parameter('publish_footprint_markers').value)

        self.allow_paired_clusters = bool(self.get_parameter('allow_paired_clusters').value)
        self.pair_min_lower_cluster_voxels = int(self.get_parameter('pair_min_lower_cluster_voxels').value)
        self.pair_max_lower_cluster_voxels = int(self.get_parameter('pair_max_lower_cluster_voxels').value)
        self.pair_min_upper_cluster_voxels = int(self.get_parameter('pair_min_upper_cluster_voxels').value)
        self.pair_max_upper_cluster_voxels = int(self.get_parameter('pair_max_upper_cluster_voxels').value)
        self.pair_min_lower_size = np.asarray(self.get_parameter('pair_min_lower_size').value, dtype=np.float32)
        self.pair_max_lower_size = np.asarray(self.get_parameter('pair_max_lower_size').value, dtype=np.float32)
        self.pair_min_upper_size = np.asarray(self.get_parameter('pair_min_upper_size').value, dtype=np.float32)
        self.pair_max_upper_size = np.asarray(self.get_parameter('pair_max_upper_size').value, dtype=np.float32)
        self.pair_lower_min_height = float(self.get_parameter('pair_lower_min_height').value)
        self.pair_lower_max_height = float(self.get_parameter('pair_lower_max_height').value)
        self.pair_upper_min_height = float(self.get_parameter('pair_upper_min_height').value)
        self.pair_upper_max_height = float(self.get_parameter('pair_upper_max_height').value)
        self.pair_min_z_separation = float(self.get_parameter('pair_min_z_separation').value)
        self.pair_max_z_separation = float(self.get_parameter('pair_max_z_separation').value)
        self.pair_max_xy_center_distance = float(self.get_parameter('pair_max_xy_center_distance').value)
        self.pair_min_xy_overlap_ratio = float(self.get_parameter('pair_min_xy_overlap_ratio').value)
        self.pair_project_to_rack_height = bool(self.get_parameter('pair_project_to_rack_height').value)
        self.pair_projection_floor_z = float(self.get_parameter('pair_projection_floor_z').value)
        self.pair_projection_height = float(self.get_parameter('pair_projection_height').value)
        self.publish_pair_markers = bool(self.get_parameter('publish_pair_markers').value)

        self.background_keys: Set[VoxelKey] = set()
        self.last_points = np.empty((0, 3), dtype=np.float32)
        self.last_stamp = None

        self.is_capturing_background = False
        self.background_capture_start_sec = 0.0
        self.background_frame_count = 0
        self.background_voxel_counter: Counter[VoxelKey] = Counter()

        self.dynamic_cloud_pub = self.create_publisher(PointCloud2, '/tracking/dynamic_cloud', 10)
        self.cluster_pose_pub = self.create_publisher(PoseArray, '/tracking/cluster_centroids', 10)
        self.touched_cluster_pose_pub = self.create_publisher(PoseArray, '/tracking/touched_cluster_centroids', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/tracking/cluster_markers', 10)
        self.oriented_cluster_marker_pub = self.create_publisher(MarkerArray, '/tracking/oriented_cluster_markers', 10)

        self.cloud_sub = self.create_subscription(PointCloud2, '/tracking/merged_cloud', self.cloud_callback, 10)

        self.capture_background_srv = self.create_service(Trigger, '/tracking/capture_background', self.capture_background)
        self.clear_background_srv = self.create_service(Trigger, '/tracking/clear_background', self.clear_background)
        self.save_background_srv = self.create_service(Trigger, '/tracking/save_background', self.save_background)
        self.load_background_srv = self.create_service(Trigger, '/tracking/load_background', self.load_background)

        self.neighbor_offsets = [
            offset for offset in product([-1, 0, 1], repeat=3)
            if offset != (0, 0, 0)
        ]
        self.background_expansion_offsets = self.build_background_expansion_offsets()

        self.get_logger().info('cluster_detector_node started.')
        self.get_logger().info(f'Detection mode: {self.detection_mode}')
        self.get_logger().info('Call /tracking/capture_background while the hall is empty.')

        if self.auto_load_background:
            self.try_auto_load_background()

    def cloud_callback(self, cloud_msg: PointCloud2) -> None:
        self.last_points = extract_xyz_points(cloud_msg)
        self.last_stamp = cloud_msg.header.stamp

        if self.is_capturing_background:
            self.collect_background_frame(cloud_msg)
            self.publish_empty_outputs(cloud_msg.header.stamp)
            return


        current_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        dynamic_keys = self.select_detection_keys(current_keys)
        if dynamic_keys is None:
            return
        all_clusters = self.cluster_voxels(dynamic_keys)

        if self.keep_largest_cluster_only and all_clusters:
            all_clusters = [max(all_clusters, key=len)]

        if self.publish_oriented_cluster_markers:
            self.publish_oriented_cluster_markers_from_voxel_clusters(all_clusters, cloud_msg.header.stamp)
        accepted_cluster_infos, accepted_keys = self.build_accepted_cluster_infos_and_keys(all_clusters)

        if self.publish_only_clustered_dynamic:
            dynamic_points = voxel_keys_to_centers(accepted_keys, self.voxel_size)
        else:
            dynamic_points = voxel_keys_to_centers(dynamic_keys, self.voxel_size)

        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, cloud_msg.header.stamp, dynamic_points))
        self.publish_cluster_outputs(accepted_cluster_infos, cloud_msg.header.stamp)

    def capture_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.last_points.size == 0 or self.last_stamp is None:
            response.success = False
            response.message = 'No merged cloud received yet.'
            return response

        self.is_capturing_background = True
        self.background_capture_start_sec = self.current_time_sec()
        self.background_frame_count = 0
        self.background_voxel_counter.clear()
        self.background_keys.clear()

        response.success = True
        response.message = (
            f'Background capture started for {self.background_capture_duration_sec:.2f} seconds. '
            'Keep the scene empty and still.'
        )
        self.get_logger().info(response.message)
        return response

    def clear_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.background_keys.clear()
        self.is_capturing_background = False
        self.background_voxel_counter.clear()
        self.background_frame_count = 0
        response.success = True
        response.message = 'Background cleared.'
        self.get_logger().info(response.message)
        return response

    def save_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        success, message = self.save_background_to_disk()
        response.success = success
        response.message = message
        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)
        return response

    def load_background(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        success, message = self.load_background_from_disk()
        response.success = success
        response.message = message
        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)
        return response



    def publish_oriented_cluster_markers_from_voxel_clusters(self, voxel_clusters: list, stamp) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.target_frame
        delete_marker.header.stamp = stamp
        delete_marker.ns = 'oriented_clusters'
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 1

        for voxel_cluster in voxel_clusters:
            if len(voxel_cluster) < self.oriented_marker_min_voxels:
                continue

            cluster_points = voxel_keys_to_centers(set(voxel_cluster), self.voxel_size)
            if cluster_points.size == 0:
                continue

            center, scale, yaw = self.compute_oriented_xy_box(cluster_points)

            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = stamp
            marker.ns = 'oriented_clusters'
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = float(math.sin(yaw * 0.5))
            marker.pose.orientation.w = float(math.cos(yaw * 0.5))

            marker.scale.x = float(max(scale[0], self.oriented_marker_min_scale))
            marker.scale.y = float(max(scale[1], self.oriented_marker_min_scale))
            marker.scale.z = float(max(scale[2], self.oriented_marker_min_scale))

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.45

            marker_array.markers.append(marker)
            marker_id += 1

        self.oriented_cluster_marker_pub.publish(marker_array)





    def compute_oriented_xy_box(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        finite_points = self.filter_finite_points(points)

        if finite_points.shape[0] == 0:
            center = np.zeros(3, dtype=np.float32)
            scale = np.ones(3, dtype=np.float32) * self.oriented_marker_min_scale
            return center, scale, 0.0

        yaw = self.compute_marker_yaw(finite_points)
        center = self.estimate_cluster_object_center(finite_points)

        if self.oriented_marker_use_fixed_size:
            scale = self.get_oriented_marker_scale(np.asarray(self.oriented_marker_fixed_size, dtype=np.float32))
        else:
            scale = self.compute_measured_oriented_scale(finite_points, yaw)

        return center.astype(np.float32), scale.astype(np.float32), yaw


    def filter_finite_points(self, points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.all(np.isfinite(points), axis=1)
        return points[finite_mask].astype(np.float32)

    def estimate_cluster_object_center(self, points: np.ndarray) -> np.ndarray:
        finite_points = self.filter_finite_points(points)

        if finite_points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        if self.oriented_marker_center_mode == 'box_center':
            return self.compute_box_center(finite_points)

        if self.oriented_marker_center_mode == 'mean':
            return finite_points.mean(axis=0).astype(np.float32)

        if self.oriented_marker_center_mode == 'median':
            return np.median(finite_points, axis=0).astype(np.float32)

        if self.oriented_marker_center_mode == 'top_footprint':
            return self.smooth_object_center(self.compute_top_footprint_center(finite_points))

        return self.smooth_object_center(self.compute_top_footprint_center(finite_points))

    def compute_top_footprint_center(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        if points.shape[0] < 3:
            return self.compute_box_center(points)

        yaw = self.compute_cluster_yaw_robust(points)
        top_points = self.select_top_footprint_points(points)

        if top_points.shape[0] < max(3, self.top_footprint_min_points):
            top_points = points

        center_xy = self.compute_oriented_xy_center_from_points(top_points, yaw)
        center_z = self.compute_vertical_center(points, top_points)

        return np.asarray([center_xy[0], center_xy[1], center_z], dtype=np.float32)

    def select_top_footprint_points(self, points: np.ndarray) -> np.ndarray:
        keep_ratio = float(np.clip(self.top_footprint_z_keep_ratio, 0.05, 1.00))

        if points.shape[0] < max(3, self.top_footprint_min_points):
            return points

        z_values = points[:, 2].astype(np.float64)
        z_threshold = float(np.quantile(z_values, 1.0 - keep_ratio))

        top_points = points[z_values >= z_threshold]

        if top_points.shape[0] < max(3, self.top_footprint_min_points):
            return points

        return top_points


    def compute_marker_yaw(self, points: np.ndarray) -> float:
        if self.oriented_marker_yaw_mode in ('visible_edge', 'edge', 'front_edge'):
            yaw = self.compute_visible_edge_yaw(points)
        else:
            yaw = self.compute_cluster_yaw_robust(points)

        return self.smooth_oriented_yaw(yaw)

    def compute_visible_edge_yaw(self, points: np.ndarray) -> float:
        finite_points = self.filter_finite_points(points)

        if finite_points.shape[0] < 3:
            return self.compute_cluster_yaw_robust(finite_points)

        top_points = self.select_top_footprint_points(finite_points)

        if top_points.shape[0] < max(3, self.top_footprint_min_points):
            top_points = finite_points

        sensor_position = np.asarray(self.visible_edge_sensor_position, dtype=np.float32)

        if sensor_position.shape[0] < 2:
            return self.compute_cluster_yaw_robust(top_points)

        xy_points = top_points[:, :2].astype(np.float64)
        sensor_xy = sensor_position[:2].astype(np.float64)

        distances = np.linalg.norm(xy_points - sensor_xy, axis=1)

        keep_ratio = float(np.clip(self.visible_edge_keep_ratio, 0.05, 1.00))
        distance_threshold = float(np.quantile(distances, keep_ratio))

        edge_points = top_points[distances <= distance_threshold]

        if edge_points.shape[0] < max(3, self.top_footprint_min_points):
            edge_points = top_points

        return self.compute_cluster_yaw_robust(edge_points)

    def smooth_object_center(self, measured_center: np.ndarray) -> np.ndarray:
        center = np.asarray(measured_center, dtype=np.float32)

        if not self.object_center_smoothing_enabled:
            self.smoothed_object_center = center.copy()
            return center

        alpha = float(np.clip(self.object_center_smoothing_alpha, 0.01, 1.00))
        max_step = float(max(self.object_center_max_step, 0.0))

        if self.smoothed_object_center is None:
            self.smoothed_object_center = center.copy()
            return center

        previous = np.asarray(self.smoothed_object_center, dtype=np.float32)
        delta = center - previous

        if max_step > 0.0:
            xy_step = float(np.linalg.norm(delta[:2]))

            if xy_step > max_step:
                delta[:2] = delta[:2] * (max_step / xy_step)

        smoothed = previous + alpha * delta
        self.smoothed_object_center = smoothed.copy()

        return smoothed.astype(np.float32)

    def smooth_oriented_yaw(self, measured_yaw: float) -> float:
        yaw = float(measured_yaw)

        if not self.oriented_marker_yaw_smoothing_enabled:
            self.smoothed_oriented_yaw = yaw
            return yaw

        alpha = float(np.clip(self.oriented_marker_yaw_smoothing_alpha, 0.01, 1.00))

        if self.smoothed_oriented_yaw is None:
            self.smoothed_oriented_yaw = yaw
            return yaw

        previous = float(self.smoothed_oriented_yaw)
        delta = self.shortest_angle_delta(previous, yaw)

        smoothed = previous + alpha * delta
        smoothed = self.normalize_angle(smoothed)

        self.smoothed_oriented_yaw = smoothed

        return smoothed

    def shortest_angle_delta(self, source: float, target: float) -> float:
        return math.atan2(math.sin(target - source), math.cos(target - source))

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute_cluster_yaw_robust(self, points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0

        xy_points = points[:, :2].astype(np.float64)

        pca_keep_ratio = float(np.clip(getattr(self, 'oriented_marker_pca_keep_ratio', 0.90), 0.50, 1.00))
        min_inlier_points = int(max(getattr(self, 'oriented_marker_min_inlier_points', 20), 3))

        xy_median = np.median(xy_points, axis=0)
        distance_to_median = np.linalg.norm(xy_points - xy_median, axis=1)

        if xy_points.shape[0] >= min_inlier_points:
            distance_threshold = float(np.quantile(distance_to_median, pca_keep_ratio))
            inlier_xy = xy_points[distance_to_median <= distance_threshold]
        else:
            inlier_xy = xy_points

        if inlier_xy.shape[0] < min_inlier_points:
            inlier_xy = xy_points

        xy_mean = inlier_xy.mean(axis=0)
        centered_xy = inlier_xy - xy_mean

        covariance = np.cov(centered_xy, rowvar=False)

        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            return 0.0

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]

        return math.atan2(float(main_axis[1]), float(main_axis[0]))

    def compute_oriented_xy_center_from_points(self, points: np.ndarray, yaw: float) -> np.ndarray:
        bounds_quantile = float(np.clip(self.top_footprint_bounds_quantile, 0.00, 0.45))

        xy_points = points[:, :2].astype(np.float64)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        local_x = xy_points[:, 0] * cos_yaw + xy_points[:, 1] * sin_yaw
        local_y = -xy_points[:, 0] * sin_yaw + xy_points[:, 1] * cos_yaw

        local_min_x = float(np.quantile(local_x, bounds_quantile))
        local_max_x = float(np.quantile(local_x, 1.0 - bounds_quantile))
        local_min_y = float(np.quantile(local_y, bounds_quantile))
        local_max_y = float(np.quantile(local_y, 1.0 - bounds_quantile))

        local_center_x = (local_min_x + local_max_x) * 0.5
        local_center_y = (local_min_y + local_max_y) * 0.5

        world_center_x = local_center_x * cos_yaw - local_center_y * sin_yaw
        world_center_y = local_center_x * sin_yaw + local_center_y * cos_yaw

        return np.asarray([world_center_x, world_center_y], dtype=np.float32)

    def compute_vertical_center(self, points: np.ndarray, top_points: np.ndarray) -> float:
        z_mode = self.top_footprint_z_center_mode

        if z_mode == 'fixed_from_top' and self.oriented_marker_use_fixed_size:
            fixed_height = float(self.oriented_marker_fixed_size[2])
            top_z = float(np.quantile(top_points[:, 2].astype(np.float64), 0.95))
            return top_z - fixed_height * 0.5

        bounds_quantile = float(np.clip(self.top_footprint_bounds_quantile, 0.00, 0.45))
        z_values = points[:, 2].astype(np.float64)

        z_min = float(np.quantile(z_values, bounds_quantile))
        z_max = float(np.quantile(z_values, 1.0 - bounds_quantile))

        return (z_min + z_max) * 0.5

    def compute_box_center(self, points: np.ndarray) -> np.ndarray:
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        return ((min_bound + max_bound) * 0.5).astype(np.float32)

    def compute_measured_oriented_scale(self, points: np.ndarray, yaw: float) -> np.ndarray:
        bounds_quantile = float(np.clip(getattr(self, 'oriented_marker_bounds_quantile', 0.03), 0.00, 0.45))

        xy_points = points[:, :2].astype(np.float64)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        local_x = xy_points[:, 0] * cos_yaw + xy_points[:, 1] * sin_yaw
        local_y = -xy_points[:, 0] * sin_yaw + xy_points[:, 1] * cos_yaw

        local_min_x = float(np.quantile(local_x, bounds_quantile))
        local_max_x = float(np.quantile(local_x, 1.0 - bounds_quantile))
        local_min_y = float(np.quantile(local_y, bounds_quantile))
        local_max_y = float(np.quantile(local_y, 1.0 - bounds_quantile))

        z_values = points[:, 2].astype(np.float64)
        z_min = float(np.quantile(z_values, bounds_quantile))
        z_max = float(np.quantile(z_values, 1.0 - bounds_quantile))

        return np.asarray(
            [
                local_max_x - local_min_x,
                local_max_y - local_min_y,
                z_max - z_min,
            ],
            dtype=np.float32,
        )

    def get_oriented_marker_scale(self, measured_scale: np.ndarray) -> np.ndarray:
        if self.oriented_marker_use_fixed_size:
            fixed_size = np.asarray(self.oriented_marker_fixed_size, dtype=np.float32)

            if fixed_size.shape[0] != 3:
                fixed_size = np.asarray([1.30, 1.30, 2.00], dtype=np.float32)

            return np.asarray(
                [
                    max(float(fixed_size[0]), self.oriented_marker_min_scale),
                    max(float(fixed_size[1]), self.oriented_marker_min_scale),
                    max(float(fixed_size[2]), self.oriented_marker_min_scale),
                ],
                dtype=np.float32,
            )

        return np.asarray(
            [
                max(float(measured_scale[0]), self.oriented_marker_min_scale),
                max(float(measured_scale[1]), self.oriented_marker_min_scale),
                max(float(measured_scale[2]), self.oriented_marker_min_scale),
            ],
            dtype=np.float32,
        )


    def compute_oriented_marker_center(
        self,
        points: np.ndarray,
        inlier_points: np.ndarray,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
    ) -> np.ndarray:
        if self.oriented_marker_center_mode == 'box_center':
            return (min_bound + max_bound) * 0.5

        if self.oriented_marker_center_mode == 'mean':
            return points.mean(axis=0)

        if self.oriented_marker_center_mode == 'median':
            return np.median(points, axis=0)

        if inlier_points.size > 0:
            return np.median(inlier_points, axis=0)

        return np.median(points, axis=0)

    def compute_measured_oriented_marker_scale(
        self,
        inlier_points: np.ndarray,
        xy_mean: np.ndarray,
        yaw: float,
        bounds_quantile: float,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
    ) -> np.ndarray:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        bounds_xy_points = inlier_points[:, :2].astype(np.float64)
        bounds_centered_xy = bounds_xy_points - xy_mean

        local_x = bounds_centered_xy[:, 0] * cos_yaw + bounds_centered_xy[:, 1] * sin_yaw
        local_y = -bounds_centered_xy[:, 0] * sin_yaw + bounds_centered_xy[:, 1] * cos_yaw

        local_min_x = float(np.quantile(local_x, bounds_quantile))
        local_max_x = float(np.quantile(local_x, 1.0 - bounds_quantile))
        local_min_y = float(np.quantile(local_y, bounds_quantile))
        local_max_y = float(np.quantile(local_y, 1.0 - bounds_quantile))

        z_values = inlier_points[:, 2].astype(np.float64)
        z_min = float(np.quantile(z_values, bounds_quantile))
        z_max = float(np.quantile(z_values, 1.0 - bounds_quantile))

        return np.asarray(
            [
                local_max_x - local_min_x,
                local_max_y - local_min_y,
                z_max - z_min,
            ],
            dtype=np.float32,
        )

    def get_oriented_marker_scale(self, measured_scale: np.ndarray) -> np.ndarray:
        if self.oriented_marker_use_fixed_size:
            fixed_size = np.asarray(self.oriented_marker_fixed_size, dtype=np.float32)

            if fixed_size.shape[0] != 3:
                fixed_size = np.asarray([1.30, 1.30, 2.00], dtype=np.float32)

            return np.asarray(
                [
                    max(float(fixed_size[0]), self.oriented_marker_min_scale),
                    max(float(fixed_size[1]), self.oriented_marker_min_scale),
                    max(float(fixed_size[2]), self.oriented_marker_min_scale),
                ],
                dtype=np.float32,
            )

        return np.asarray(
            [
                max(float(measured_scale[0]), self.oriented_marker_min_scale),
                max(float(measured_scale[1]), self.oriented_marker_min_scale),
                max(float(measured_scale[2]), self.oriented_marker_min_scale),
            ],
            dtype=np.float32,
        )


    def select_detection_keys(self, current_keys: Set[VoxelKey]) -> Set[VoxelKey] | None:
        if self.detection_mode in ('live', 'live_roi', 'no_background'):
            return current_keys

        if self.detection_mode in ('background', 'background_subtraction'):
            if not self.background_keys:
                self.get_logger().warning(
                    'Background not captured yet. Call /tracking/capture_background, '
                    'or set detection_mode to live_roi for ROI-only tests.',
                    throttle_duration_sec=5.0,
                )
                return None

            return current_keys - self.background_keys

        self.get_logger().warning(
            f'Invalid detection_mode="{self.detection_mode}". '
            'Use background_subtraction or live_roi.',
            throttle_duration_sec=5.0,
        )
        return None

    def collect_background_frame(self, cloud_msg: PointCloud2) -> None:
        elapsed_sec = self.current_time_sec() - self.background_capture_start_sec
        current_keys = points_to_voxel_keys(self.last_points, self.voxel_size)
        self.background_voxel_counter.update(current_keys)
        self.background_frame_count += 1

        self.get_logger().info(
            f'Capturing background: {elapsed_sec:.2f}s / {self.background_capture_duration_sec:.2f}s, '
            f'frames={self.background_frame_count}, current_voxels={len(current_keys)}.',
            throttle_duration_sec=1.0,
        )

        if elapsed_sec < self.background_capture_duration_sec:
            return

        self.finish_background_capture()

    def finish_background_capture(self) -> None:
        if self.background_frame_count <= 0:
            self.background_keys.clear()
            self.is_capturing_background = False
            self.get_logger().warning('Background capture finished without frames.')
            return

        ratio = min(max(self.background_min_occupancy_ratio, 0.0), 1.0)
        min_count = max(1, int(np.ceil(float(self.background_frame_count) * ratio)))
        stable_keys = {
            key for key, count in self.background_voxel_counter.items()
            if count >= min_count
        }
        expanded_keys = self.expand_background_keys(stable_keys)

        self.background_keys = expanded_keys
        self.is_capturing_background = False
        self.background_voxel_counter.clear()

        self.get_logger().info(
            f'Background captured from {self.background_frame_count} frames. '
            f'Stable voxels={len(stable_keys)}, expanded voxels={len(self.background_keys)}, '
            f'min_count={min_count}.'
        )

        if self.auto_save_background:
            success, message = self.save_background_to_disk()
            if success:
                self.get_logger().info(message)
            else:
                self.get_logger().warning(message)

    def try_auto_load_background(self) -> None:
        path = self.resolved_background_storage_path()
        if not path.exists():
            self.get_logger().info(f'No stored background found at "{path}".')
            return

        success, message = self.load_background_from_disk()
        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(message)

    def resolved_background_storage_path(self) -> Path:
        return Path(self.background_storage_path).expanduser()

    def save_background_to_disk(self) -> tuple[bool, str]:
        if not self.background_keys:
            return False, 'Cannot save background: no background captured yet.'

        path = self.resolved_background_storage_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        keys_array = np.asarray(sorted(self.background_keys), dtype=np.int32)
        np.savez_compressed(
            path,
            keys=keys_array,
            voxel_size=np.asarray([self.voxel_size], dtype=np.float32),
            saved_at_sec=np.asarray([self.current_time_sec()], dtype=np.float64),
        )

        return True, f'Background saved to "{path}" with {len(self.background_keys)} voxels.'

    def load_background_from_disk(self) -> tuple[bool, str]:
        path = self.resolved_background_storage_path()
        if not path.exists():
            return False, f'Cannot load background: file does not exist: "{path}".'

        with np.load(path, allow_pickle=False) as data:
            if 'keys' not in data.files:
                return False, f'Cannot load background: file "{path}" does not contain keys.'

            keys_array = np.asarray(data['keys'], dtype=np.int32)
            stored_voxel_size = float(data['voxel_size'][0]) if 'voxel_size' in data.files else self.voxel_size

        if abs(stored_voxel_size - self.voxel_size) > 1e-6:
            return False, (
                f'Cannot load background: voxel_size mismatch. '
                f'File={stored_voxel_size:.6f}, current={self.voxel_size:.6f}.'
            )

        if keys_array.ndim != 2 or keys_array.shape[1] != 3:
            return False, f'Cannot load background: invalid key array shape {keys_array.shape}.'

        self.background_keys = {tuple(int(value) for value in row) for row in keys_array}
        self.is_capturing_background = False
        self.background_voxel_counter.clear()
        self.background_frame_count = 0

        return True, f'Background loaded from "{path}" with {len(self.background_keys)} voxels.'

    def build_background_expansion_offsets(self) -> List[VoxelKey]:
        radius = max(self.background_expansion_voxels, 0)
        axes = self.background_expansion_axes
        if len(axes) != 3:
            axes = [1, 1, 1]

        ranges = []
        for axis_index in range(3):
            if axes[axis_index]:
                ranges.append(range(-radius, radius + 1))
            else:
                ranges.append(range(0, 1))

        return [tuple(int(value) for value in offset) for offset in product(*ranges)]

    def expand_background_keys(self, stable_keys: Set[VoxelKey]) -> Set[VoxelKey]:
        if self.background_expansion_voxels <= 0:
            return set(stable_keys)

        expanded: Set[VoxelKey] = set()
        for key in stable_keys:
            for offset in self.background_expansion_offsets:
                expanded.add((key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]))
        return expanded

    def cluster_voxels(self, voxel_keys: Set[VoxelKey]) -> List[List[VoxelKey]]:
        clusters: List[List[VoxelKey]] = []
        remaining = set(voxel_keys)

        while remaining:
            seed = remaining.pop()
            queue = [seed]
            cluster = [seed]

            while queue:
                key = queue.pop()
                for offset in self.neighbor_offsets:
                    neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cluster.append(neighbor)

            clusters.append(cluster)

        return clusters

    def build_accepted_cluster_infos_and_keys(
        self,
        clusters: Iterable[List[VoxelKey]],
    ) -> Tuple[List[ClusterInfo], Set[VoxelKey]]:
        candidates = self.build_cluster_candidates(clusters)
        accepted_cluster_infos: List[ClusterInfo] = []
        accepted_keys: Set[VoxelKey] = set()
        used_indices: Set[int] = set()

        if self.allow_paired_clusters:
            pair_infos, pair_keys, pair_used_indices = self.build_pair_cluster_infos(candidates)
            for pair_info in pair_infos:
                pair_info.cluster_id = len(accepted_cluster_infos)
                accepted_cluster_infos.append(pair_info)
            accepted_keys.update(pair_keys)
            used_indices.update(pair_used_indices)

        for candidate in candidates:
            if candidate.index in used_indices:
                continue

            cluster_type = self.classify_cluster(candidate.info)
            if cluster_type is None:
                continue

            cluster_info = candidate.info
            if cluster_type == 'footprint':
                cluster_info = self.project_footprint_cluster(cluster_info)

            cluster_info.cluster_id = len(accepted_cluster_infos)
            cluster_info.cluster_type = cluster_type
            accepted_cluster_infos.append(cluster_info)
            accepted_keys.update(candidate.keys)

        return accepted_cluster_infos, accepted_keys

    def build_cluster_candidates(self, clusters: Iterable[List[VoxelKey]]) -> List[ClusterCandidate]:
        candidates: List[ClusterCandidate] = []
        for cluster_index, cluster in enumerate(clusters):
            cluster_info = self.build_cluster_info_from_keys(
                cluster=cluster,
                cluster_id=cluster_index,
                cluster_type='candidate',
            )
            candidates.append(ClusterCandidate(index=cluster_index, keys=cluster, info=cluster_info))
        return candidates

    def build_pair_cluster_infos(
        self,
        candidates: List[ClusterCandidate],
    ) -> Tuple[List[ClusterInfo], Set[VoxelKey], Set[int]]:
        lower_candidates = [candidate for candidate in candidates if self.is_pair_lower_candidate(candidate.info)]
        upper_candidates = [candidate for candidate in candidates if self.is_pair_upper_candidate(candidate.info)]

        pair_options: List[Tuple[float, int, int, ClusterCandidate, ClusterCandidate]] = []
        for lower in lower_candidates:
            for upper in upper_candidates:
                if lower.index == upper.index:
                    continue
                if float(upper.info.centroid[2]) <= float(lower.info.centroid[2]):
                    continue
                if not self.is_valid_pair(lower.info, upper.info):
                    continue
                score = self.pair_score(lower.info, upper.info)
                pair_options.append((score, lower.index, upper.index, lower, upper))

        pair_options.sort(key=lambda item: item[0])

        pair_infos: List[ClusterInfo] = []
        accepted_keys: Set[VoxelKey] = set()
        used_indices: Set[int] = set()

        for _, lower_index, upper_index, lower, upper in pair_options:
            if lower_index in used_indices or upper_index in used_indices:
                continue
            pair_info = self.build_pair_info(lower.info, upper.info)
            pair_infos.append(pair_info)
            accepted_keys.update(lower.keys)
            accepted_keys.update(upper.keys)
            used_indices.add(lower_index)
            used_indices.add(upper_index)

        return pair_infos, accepted_keys, used_indices

    def build_cluster_info_from_keys(self, cluster: List[VoxelKey], cluster_id: int, cluster_type: str) -> ClusterInfo:
        cluster_points = voxel_keys_to_centers(cluster, self.voxel_size)
        centroid = self.estimate_cluster_object_center(cluster_points)
        min_bound = np.min(cluster_points, axis=0)
        max_bound = np.max(cluster_points, axis=0)
        return ClusterInfo(
            cluster_id=cluster_id,
            centroid=centroid.astype(np.float32),
            min_bound=min_bound.astype(np.float32),
            max_bound=max_bound.astype(np.float32),
            voxel_count=len(cluster),
            cluster_type=cluster_type,
        )

    def classify_cluster(self, cluster: ClusterInfo) -> str | None:
        if self.is_normal_cluster(cluster):
            return 'normal'

        if self.allow_footprint_projection and self.is_footprint_cluster(cluster):
            return 'footprint'

        if self.allow_touched_clusters and self.is_touched_cluster(cluster):
            return 'touched'

        return None

    def is_normal_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        if cluster.voxel_count < self.min_cluster_voxels or cluster.voxel_count > self.max_cluster_voxels:
            return False
        if not np.all(size >= self.min_cluster_size):
            return False
        if not np.all(size <= self.max_cluster_size):
            return False
        if float(cluster.min_bound[2]) < self.min_cluster_height:
            return False
        if float(cluster.max_bound[2]) > self.max_cluster_height:
            return False
        return True

    def is_footprint_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        centroid_z = float(cluster.centroid[2])

        if cluster.voxel_count < self.footprint_min_cluster_voxels:
            return False
        if cluster.voxel_count > self.footprint_max_cluster_voxels:
            return False
        if not np.all(size >= self.footprint_min_size):
            return False
        if not np.all(size <= self.footprint_max_size):
            return False
        if centroid_z < self.footprint_min_height:
            return False
        if centroid_z > self.footprint_max_height:
            return False
        return True

    def project_footprint_cluster(self, cluster: ClusterInfo) -> ClusterInfo:
        projected_min = cluster.min_bound.copy()
        projected_max = cluster.max_bound.copy()
        floor_z = self.projection_floor_z
        top_z = self.projection_floor_z + max(self.projection_height, self.voxel_size)

        projected_min[2] = floor_z
        projected_max[2] = top_z
        projected_centroid = self.center_from_bounds(projected_min, projected_max)

        return ClusterInfo(
            cluster_id=cluster.cluster_id,
            centroid=projected_centroid,
            min_bound=projected_min.astype(np.float32),
            max_bound=projected_max.astype(np.float32),
            voxel_count=cluster.voxel_count,
            cluster_type=cluster.cluster_type,
        )

    def is_touched_cluster(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        if cluster.voxel_count < self.touched_min_cluster_voxels:
            return False
        if cluster.voxel_count > self.touched_max_cluster_voxels:
            return False
        if not np.all(size >= self.touched_min_cluster_size):
            return False
        if not np.all(size <= self.touched_max_cluster_size):
            return False
        if float(cluster.min_bound[2]) < self.min_cluster_height:
            return False
        if float(cluster.max_bound[2]) > self.touched_max_cluster_size[2]:
            return False
        return True

    def is_pair_lower_candidate(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        centroid_z = float(cluster.centroid[2])
        if cluster.voxel_count < self.pair_min_lower_cluster_voxels:
            return False
        if cluster.voxel_count > self.pair_max_lower_cluster_voxels:
            return False
        if not np.all(size >= self.pair_min_lower_size):
            return False
        if not np.all(size <= self.pair_max_lower_size):
            return False
        if centroid_z < self.pair_lower_min_height or centroid_z > self.pair_lower_max_height:
            return False
        return True

    def is_pair_upper_candidate(self, cluster: ClusterInfo) -> bool:
        size = cluster.max_bound - cluster.min_bound
        centroid_z = float(cluster.centroid[2])
        if cluster.voxel_count < self.pair_min_upper_cluster_voxels:
            return False
        if cluster.voxel_count > self.pair_max_upper_cluster_voxels:
            return False
        if not np.all(size >= self.pair_min_upper_size):
            return False
        if not np.all(size <= self.pair_max_upper_size):
            return False
        if centroid_z < self.pair_upper_min_height or centroid_z > self.pair_upper_max_height:
            return False
        return True

    def is_valid_pair(self, lower: ClusterInfo, upper: ClusterInfo) -> bool:
        z_separation = float(upper.centroid[2] - lower.centroid[2])
        if z_separation < self.pair_min_z_separation:
            return False
        if z_separation > self.pair_max_z_separation:
            return False

        center_distance_xy = float(np.linalg.norm(lower.centroid[:2] - upper.centroid[:2]))
        overlap_ratio = self.xy_overlap_ratio(lower, upper)

        if center_distance_xy <= self.pair_max_xy_center_distance:
            return True
        if overlap_ratio >= self.pair_min_xy_overlap_ratio:
            return True
        return False

    def pair_score(self, lower: ClusterInfo, upper: ClusterInfo) -> float:
        center_distance_xy = float(np.linalg.norm(lower.centroid[:2] - upper.centroid[:2]))
        overlap_ratio = self.xy_overlap_ratio(lower, upper)
        return center_distance_xy - overlap_ratio

    def xy_overlap_ratio(self, first: ClusterInfo, second: ClusterInfo) -> float:
        overlap_min_x = max(float(first.min_bound[0]), float(second.min_bound[0]))
        overlap_min_y = max(float(first.min_bound[1]), float(second.min_bound[1]))
        overlap_max_x = min(float(first.max_bound[0]), float(second.max_bound[0]))
        overlap_max_y = min(float(first.max_bound[1]), float(second.max_bound[1]))

        overlap_x = max(0.0, overlap_max_x - overlap_min_x)
        overlap_y = max(0.0, overlap_max_y - overlap_min_y)
        overlap_area = overlap_x * overlap_y

        first_size = first.max_bound - first.min_bound
        second_size = second.max_bound - second.min_bound
        first_area = max(float(first_size[0] * first_size[1]), 1e-6)
        second_area = max(float(second_size[0] * second_size[1]), 1e-6)
        return overlap_area / min(first_area, second_area)

    def build_pair_info(self, lower: ClusterInfo, upper: ClusterInfo) -> ClusterInfo:
        pair_min = np.minimum(lower.min_bound, upper.min_bound).astype(np.float32)
        pair_max = np.maximum(lower.max_bound, upper.max_bound).astype(np.float32)

        if self.pair_project_to_rack_height:
            pair_min[2] = self.pair_projection_floor_z
            pair_max[2] = self.pair_projection_floor_z + max(self.pair_projection_height, self.voxel_size)

        lower_weight = max(float(lower.voxel_count), 1.0)
        upper_weight = max(float(upper.voxel_count), 1.0)
        centroid_xy = (
            lower.centroid[:2] * lower_weight + upper.centroid[:2] * upper_weight
        ) / (lower_weight + upper_weight)
        centroid = self.center_from_bounds(pair_min, pair_max)
        centroid[0] = float(centroid_xy[0])
        centroid[1] = float(centroid_xy[1])

        return ClusterInfo(
            cluster_id=0,
            centroid=centroid.astype(np.float32),
            min_bound=pair_min.astype(np.float32),
            max_bound=pair_max.astype(np.float32),
            voxel_count=int(lower.voxel_count + upper.voxel_count),
            cluster_type='paired',
        )

    @staticmethod
    def center_from_bounds(min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float((min_bound[0] + max_bound[0]) * 0.5),
                float((min_bound[1] + max_bound[1]) * 0.5),
                float((min_bound[2] + max_bound[2]) * 0.5),
            ],
            dtype=np.float32,
        )

    def publish_cluster_outputs(self, cluster_infos: List[ClusterInfo], stamp) -> None:
        normal_clusters = [
            cluster for cluster in cluster_infos
            if cluster.cluster_type in ('normal', 'footprint', 'paired')
        ]
        touched_clusters = [cluster for cluster in cluster_infos if cluster.cluster_type == 'touched']

        self.cluster_pose_pub.publish(self.build_pose_array(normal_clusters, stamp))
        self.touched_cluster_pose_pub.publish(self.build_pose_array(touched_clusters, stamp))
        self.cluster_marker_pub.publish(self.build_marker_array(cluster_infos, stamp))

    def build_pose_array(self, cluster_infos: List[ClusterInfo], stamp) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp

        for cluster in cluster_infos:
            pose = Pose()
            pose.position.x = float(cluster.centroid[0])
            pose.position.y = float(cluster.centroid[1])
            pose.position.z = float(cluster.centroid[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        return pose_array

    def build_marker_array(self, cluster_infos: List[ClusterInfo], stamp) -> MarkerArray:
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for cluster in cluster_infos:
            if cluster.cluster_type == 'touched' and not self.publish_touched_markers:
                continue
            if cluster.cluster_type == 'footprint' and not self.publish_footprint_markers:
                continue
            if cluster.cluster_type == 'paired' and not self.publish_pair_markers:
                continue

            bbox = Marker()
            bbox.header.frame_id = self.target_frame
            bbox.header.stamp = stamp
            bbox.ns = 'clusters'
            bbox.id = marker_id
            bbox.type = Marker.CUBE
            bbox.action = Marker.ADD
            bbox.pose.position.x = float((cluster.min_bound[0] + cluster.max_bound[0]) * 0.5)
            bbox.pose.position.y = float((cluster.min_bound[1] + cluster.max_bound[1]) * 0.5)
            bbox.pose.position.z = float((cluster.min_bound[2] + cluster.max_bound[2]) * 0.5)
            bbox.pose.orientation.w = 1.0
            bbox.scale.x = max(float(cluster.max_bound[0] - cluster.min_bound[0]), self.voxel_size)
            bbox.scale.y = max(float(cluster.max_bound[1] - cluster.min_bound[1]), self.voxel_size)
            bbox.scale.z = max(float(cluster.max_bound[2] - cluster.min_bound[2]), self.voxel_size)

            if cluster.cluster_type == 'normal':
                bbox.color.r = 0.0
                bbox.color.g = 1.0
                bbox.color.b = 0.0
            elif cluster.cluster_type == 'footprint':
                bbox.color.r = 0.0
                bbox.color.g = 0.65
                bbox.color.b = 1.0
            elif cluster.cluster_type == 'paired':
                bbox.color.r = 0.65
                bbox.color.g = 0.0
                bbox.color.b = 1.0
            else:
                bbox.color.r = 1.0
                bbox.color.g = 0.55
                bbox.color.b = 0.0
            bbox.color.a = 0.45
            marker_array.markers.append(bbox)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = stamp
            text.ns = 'cluster_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(cluster.centroid[0])
            text.pose.position.y = float(cluster.centroid[1])
            text.pose.position.z = float(cluster.max_bound[2] + 0.25)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.22
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'C{cluster.cluster_id} {cluster.cluster_type} vox:{cluster.voxel_count}'
            marker_array.markers.append(text)
            marker_id += 1

        return marker_array

    def publish_empty_outputs(self, stamp) -> None:
        self.dynamic_cloud_pub.publish(create_xyz_cloud(self.target_frame, stamp, np.empty((0, 3), dtype=np.float32)))
        self.cluster_pose_pub.publish(self.build_pose_array([], stamp))
        self.touched_cluster_pose_pub.publish(self.build_pose_array([], stamp))

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.cluster_marker_pub.publish(marker_array)

    def current_time_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return float(now.sec) + float(now.nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClusterDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
