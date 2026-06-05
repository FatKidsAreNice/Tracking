#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray


@dataclass
class BoxMarker:
    marker_id: int
    namespace: str
    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float

    @property
    def min_x(self) -> float:
        return self.x - self.sx * 0.5

    @property
    def max_x(self) -> float:
        return self.x + self.sx * 0.5

    @property
    def min_y(self) -> float:
        return self.y - self.sy * 0.5

    @property
    def max_y(self) -> float:
        return self.y + self.sy * 0.5

    @property
    def min_z(self) -> float:
        return self.z - self.sz * 0.5

    @property
    def max_z(self) -> float:
        return self.z + self.sz * 0.5

    @property
    def xy_area(self) -> float:
        return max(self.sx, 0.0) * max(self.sy, 0.0)


class ClusterMarkerDebugNode(Node):
    def __init__(self) -> None:
        super().__init__('cluster_marker_debug_node')
        self.subscription = self.create_subscription(
            MarkerArray,
            '/tracking/cluster_markers',
            self.marker_callback,
            10,
        )
        self.done = False

    def marker_callback(self, msg: MarkerArray) -> None:
        if self.done:
            return

        boxes: List[BoxMarker] = []

        for marker in msg.markers:
            # Marker.CUBE == 1. DELETEALL has action=3 and should be ignored.
            if marker.type != 1:
                continue
            if marker.action != 0:
                continue

            boxes.append(
                BoxMarker(
                    marker_id=int(marker.id),
                    namespace=str(marker.ns),
                    x=float(marker.pose.position.x),
                    y=float(marker.pose.position.y),
                    z=float(marker.pose.position.z),
                    sx=float(marker.scale.x),
                    sy=float(marker.scale.y),
                    sz=float(marker.scale.z),
                )
            )

        if not boxes:
            self.get_logger().info('No cube markers received.')
            self.done = True
            rclpy.shutdown()
            return

        self.get_logger().info('=== Cluster cube markers ===')
        for box in boxes:
            self.get_logger().info(
                f'id={box.marker_id} ns={box.namespace} '
                f'center=({box.x:.3f}, {box.y:.3f}, {box.z:.3f}) '
                f'size=({box.sx:.3f}, {box.sy:.3f}, {box.sz:.3f}) '
                f'z_range=({box.min_z:.3f}, {box.max_z:.3f}) '
                f'xy_area={box.xy_area:.3f}'
            )

        self.get_logger().info('=== Possible lower/upper pairs ===')
        for i, first in enumerate(boxes):
            for second in boxes[i + 1:]:
                lower, upper = self.order_by_height(first, second)
                xy_distance = math.hypot(lower.x - upper.x, lower.y - upper.y)
                z_separation = upper.z - lower.z
                overlap_ratio = self.xy_overlap_ratio(lower, upper)

                self.get_logger().info(
                    f'lower_id={lower.marker_id} upper_id={upper.marker_id} '
                    f'xy_distance={xy_distance:.3f} '
                    f'z_separation={z_separation:.3f} '
                    f'xy_overlap_ratio={overlap_ratio:.3f} '
                    f'lower_size=({lower.sx:.3f}, {lower.sy:.3f}, {lower.sz:.3f}) '
                    f'upper_size=({upper.sx:.3f}, {upper.sy:.3f}, {upper.sz:.3f})'
                )

        self.done = True
        rclpy.shutdown()

    @staticmethod
    def order_by_height(first: BoxMarker, second: BoxMarker) -> Tuple[BoxMarker, BoxMarker]:
        if first.z <= second.z:
            return first, second
        return second, first

    @staticmethod
    def xy_overlap_ratio(first: BoxMarker, second: BoxMarker) -> float:
        overlap_x = max(0.0, min(first.max_x, second.max_x) - max(first.min_x, second.min_x))
        overlap_y = max(0.0, min(first.max_y, second.max_y) - max(first.min_y, second.min_y))
        overlap_area = overlap_x * overlap_y
        reference_area = min(first.xy_area, second.xy_area)

        if reference_area <= 0.0:
            return 0.0

        return overlap_area / reference_area


def main() -> None:
    rclpy.init()
    node = ClusterMarkerDebugNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()