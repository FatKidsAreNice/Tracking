from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from .pointcloud_utils import create_xyz_cloud, crop_points, extract_xyz_points, voxel_downsample
from .transform_utils import build_sensor_transform_map, build_single_sensor_transform, transform_points


class CloudTransformMergeNode(Node):
    def __init__(self) -> None:
        super().__init__('cloud_transform_merge_node')

        self.declare_parameter('mode', 'single')
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('voxel_size', 0.04)
        self.declare_parameter('merged_voxel_size', 0.04)
        self.declare_parameter('stale_cloud_timeout_sec', 0.6)
        self.declare_parameter('roi_min', [-1.5, -1.5, 0.0])
        self.declare_parameter('roi_max', [1.5, 1.5, 2.5])

        self.declare_parameter('input_topic', '/rslidar_points')
        self.declare_parameter('expected_frame_id', '')
        self.declare_parameter('sensor_pose', [0.0, 0.0, 2.4, 3.141592653589793, 0.0, 0.0])

        self.declare_parameter(
            'lidar_topics',
            [
                '/lidar_01/points',
                '/lidar_02/points',
                '/lidar_03/points',
                '/lidar_04/points',
                '/lidar_05/points',
                '/lidar_06/points',
            ],
        )

        self.mode = str(self.get_parameter('mode').value).strip().lower()
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.merged_voxel_size = float(self.get_parameter('merged_voxel_size').value)
        self.stale_cloud_timeout_sec = float(self.get_parameter('stale_cloud_timeout_sec').value)
        self.roi_min = np.asarray(self.get_parameter('roi_min').value, dtype=np.float32)
        self.roi_max = np.asarray(self.get_parameter('roi_max').value, dtype=np.float32)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.expected_frame_id = str(self.get_parameter('expected_frame_id').value)
        self.sensor_pose = np.asarray(self.get_parameter('sensor_pose').value, dtype=np.float32)
        if self.sensor_pose.shape != (6,):
            raise ValueError('sensor_pose must contain exactly 6 values: x, y, z, roll, pitch, yaw')

        self.lidar_topics: List[str] = list(self.get_parameter('lidar_topics').value)

        self.single_sensor_transform = build_single_sensor_transform(
            float(self.sensor_pose[0]),
            float(self.sensor_pose[1]),
            float(self.sensor_pose[2]),
            float(self.sensor_pose[3]),
            float(self.sensor_pose[4]),
            float(self.sensor_pose[5]),
        )
        self.sensor_transform_map = build_sensor_transform_map()

        self.latest_points_by_topic: Dict[str, np.ndarray] = {}
        self.latest_stamp_by_topic: Dict[str, float] = {}
        self.latest_msg_stamp = None

        self.merged_cloud_pub = self.create_publisher(PointCloud2, '/tracking/merged_cloud', 10)
        self.subscribers = []

        if self.mode == 'single':
            self.subscribers.append(
                self.create_subscription(PointCloud2, self.input_topic, partial(self.cloud_callback, self.input_topic), 10)
            )
        elif self.mode == 'multi':
            self.subscribers.extend(
                self.create_subscription(PointCloud2, topic, partial(self.cloud_callback, topic), 10)
                for topic in self.lidar_topics
            )
        else:
            raise ValueError('mode must be either "single" or "multi"')

        publish_period = 1.0 / max(self.publish_rate_hz, 0.1)
        self.timer = self.create_timer(publish_period, self.publish_merged_cloud)

        self.get_logger().info('cloud_transform_merge_node started.')
        self.get_logger().info(f'Mode: {self.mode}')
        self.get_logger().info(f'Target frame: {self.target_frame}')
        if self.mode == 'single':
            self.get_logger().info(f'Input topic: {self.input_topic}')
            self.get_logger().info(f'Sensor pose xyzrpy: {self.sensor_pose.tolist()}')
        else:
            self.get_logger().info(f'Lidar topics: {self.lidar_topics}')

    def cloud_callback(self, topic_name: str, cloud_msg: PointCloud2) -> None:
        transform_matrix = self.resolve_transform(cloud_msg)
        if transform_matrix is None:
            return

        points_xyz = extract_xyz_points(cloud_msg)
        if points_xyz.size == 0:
            self.latest_points_by_topic[topic_name] = np.empty((0, 3), dtype=np.float32)
            self.latest_stamp_by_topic[topic_name] = self.msg_time_to_sec(cloud_msg)
            self.latest_msg_stamp = cloud_msg.header.stamp
            return

        world_points = transform_points(points_xyz, transform_matrix)
        cropped_points = crop_points(world_points, self.roi_min, self.roi_max)
        downsampled_points = voxel_downsample(cropped_points, self.voxel_size)

        self.latest_points_by_topic[topic_name] = downsampled_points
        self.latest_stamp_by_topic[topic_name] = self.msg_time_to_sec(cloud_msg)
        self.latest_msg_stamp = cloud_msg.header.stamp

    def resolve_transform(self, cloud_msg: PointCloud2) -> Optional[np.ndarray]:
        if self.mode == 'single':
            if self.expected_frame_id and cloud_msg.header.frame_id != self.expected_frame_id:
                self.get_logger().warning(
                    f'Unexpected frame_id "{cloud_msg.header.frame_id}" received on {self.input_topic}. '
                    f'Expected "{self.expected_frame_id}".',
                    throttle_duration_sec=5.0,
                )
            return self.single_sensor_transform

        frame_id = cloud_msg.header.frame_id
        transform_matrix = self.sensor_transform_map.get(frame_id)
        if transform_matrix is None:
            self.get_logger().warning(
                f'No static transform configured for frame "{frame_id}". '
                'Known frames: ' + ', '.join(sorted(self.sensor_transform_map.keys())),
                throttle_duration_sec=5.0,
            )
            return None
        return transform_matrix

    def publish_merged_cloud(self) -> None:
        if self.latest_msg_stamp is None:
            return

        if not self.latest_stamp_by_topic:
            return

        reference_time_sec = max(self.latest_stamp_by_topic.values())
        merged_parts = []

        for topic_name, points_xyz in self.latest_points_by_topic.items():
            msg_stamp_sec = self.latest_stamp_by_topic.get(topic_name)
            if msg_stamp_sec is None:
                continue
            if (reference_time_sec - msg_stamp_sec) > self.stale_cloud_timeout_sec:
                continue
            if points_xyz.size == 0:
                continue
            merged_parts.append(points_xyz)

        if not merged_parts:
            return

        merged_points = np.concatenate(merged_parts, axis=0)
        merged_points = voxel_downsample(merged_points, self.merged_voxel_size)
        merged_msg = create_xyz_cloud(self.target_frame, self.latest_msg_stamp, merged_points)
        self.merged_cloud_pub.publish(merged_msg)

    @staticmethod
    def msg_time_to_sec(cloud_msg: PointCloud2) -> float:
        stamp = cloud_msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudTransformMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
