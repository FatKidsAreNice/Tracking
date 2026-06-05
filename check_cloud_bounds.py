#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class CloudBoundsNode(Node):
    def __init__(self):
        super().__init__('cloud_bounds_node')
        self.sub1 = self.create_subscription(PointCloud2, '/rslidar_points', self.cb_rslidar, 10)
        self.sub2 = self.create_subscription(PointCloud2, '/tracking/merged_cloud', self.cb_merged, 10)
        self.last_rslidar = False
        self.last_merged = False

    def _extract(self, msg: PointCloud2):
        pts = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        return pts

    def _print_stats(self, name: str, msg: PointCloud2):
        pts = self._extract(msg)
        if pts.size == 0:
            self.get_logger().info(f'{name}: no points')
            return

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        self.get_logger().info(
            f'{name}: frame={msg.header.frame_id} points={len(pts)} '
            f'min=({mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}) '
            f'max=({mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f})'
        )

    def cb_rslidar(self, msg: PointCloud2):
        if not self.last_rslidar:
            self._print_stats('/rslidar_points', msg)
            self.last_rslidar = True

    def cb_merged(self, msg: PointCloud2):
        if not self.last_merged:
            self._print_stats('/tracking/merged_cloud', msg)
            self.last_merged = True

        if self.last_rslidar and self.last_merged:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = CloudBoundsNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()