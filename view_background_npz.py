#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class BackgroundPreviewNode(Node):
    def __init__(self, npz_path: Path) -> None:
        super().__init__("background_preview_node")

        self.npz_path = npz_path
        self.publisher = self.create_publisher(
            PointCloud2,
            "/tracking/background_preview",
            10,
        )

        self.points = self.load_background_points(npz_path)
        self.timer = self.create_timer(1.0, self.publish_cloud)

        self.get_logger().info(f"Loaded background: {npz_path}")
        self.get_logger().info(f"Publishing {len(self.points)} points on /tracking/background_preview")

        if len(self.points) > 0:
            self.get_logger().info(f"min xyz: {self.points.min(axis=0)}")
            self.get_logger().info(f"max xyz: {self.points.max(axis=0)}")

    def load_background_points(self, npz_path: Path) -> np.ndarray:
        if not npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            keys = np.asarray(data["keys"], dtype=np.float32)
            voxel_size = float(data["voxel_size"][0])

        if keys.ndim != 2 or keys.shape[1] != 3:
            raise ValueError(f"Invalid background key shape: {keys.shape}")

        points = (keys + 0.5) * voxel_size
        return points.astype(np.float32)

    def publish_cloud(self) -> None:
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        msg = point_cloud2.create_cloud_xyz32(header, self.points.tolist())
        self.publisher.publish(msg)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 ~/ros2_ws/view_background_npz.py ~/ros2_ws/backgrounds/background_test.npz")
        sys.exit(1)

    npz_path = Path(sys.argv[1]).expanduser()

    rclpy.init()
    node = BackgroundPreviewNode(npz_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
