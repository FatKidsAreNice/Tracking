from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .event_utils import build_scan_event_payload, make_string_message, parse_string_message


@dataclass(frozen=True)
class ScannerZone:
    scanner_id: str
    direction: str
    center: np.ndarray
    size: np.ndarray


class VirtualScannerNode(Node):
    def __init__(self) -> None:
        super().__init__('virtual_scanner_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('barcode_prefix', 'SIM')
        self.declare_parameter('marker_publish_rate_hz', 1.0)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.barcode_prefix = str(self.get_parameter('barcode_prefix').value)
        self.marker_publish_rate_hz = float(self.get_parameter('marker_publish_rate_hz').value)

        self.zones: List[ScannerZone] = self.build_default_zones()
        self.active_tracks_by_zone: Dict[str, Set[int]] = {zone.scanner_id: set() for zone in self.zones}

        self.next_barcode_index = 1
        self.next_event_index = 1

        self.track_state_sub = self.create_subscription(
            String,
            '/tracking/track_states',
            self.track_state_callback,
            10,
        )

        self.scan_event_pub = self.create_publisher(String, '/tracking/scan_events', 10)
        self.scanner_marker_pub = self.create_publisher(MarkerArray, '/tracking/scanner_markers', 10)

        timer_period = 1.0 / max(self.marker_publish_rate_hz, 0.1)
        self.marker_timer = self.create_timer(timer_period, self.publish_zone_markers)

        self.get_logger().info('virtual_scanner_node started.')
        for zone in self.zones:
            self.get_logger().info(
                f'Zone "{zone.scanner_id}" direction={zone.direction} '
                f'center={zone.center.tolist()} size={zone.size.tolist()}'
            )

    def build_default_zones(self) -> List[ScannerZone]:
        return [
            ScannerZone(
                scanner_id='entry_south',
                direction='entry',
                center=np.array([-6.35, -5.20, 0.90], dtype=np.float32),
                size=np.array([1.80, 1.20, 2.20], dtype=np.float32),
            ),
            ScannerZone(
                scanner_id='exit_east_top',
                direction='exit',
                center=np.array([10.60, 2.25, 0.90], dtype=np.float32),
                size=np.array([1.20, 1.80, 2.20], dtype=np.float32),
            ),
            ScannerZone(
                scanner_id='exit_east_bottom',
                direction='exit',
                center=np.array([10.60, -2.25, 0.90], dtype=np.float32),
                size=np.array([1.20, 1.80, 2.20], dtype=np.float32),
            ),
        ]

    def track_state_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = payload.get('tracks', [])
        stamp_sec = float(payload.get('stamp_sec', 0.0))

        current_track_ids = {int(item.get('track_id', 0)) for item in tracks}

        for zone in self.zones:
            stale_ids = {track_id for track_id in self.active_tracks_by_zone[zone.scanner_id] if track_id not in current_track_ids}
            self.active_tracks_by_zone[zone.scanner_id] -= stale_ids

        for item in tracks:
            track_id = int(item.get('track_id', 0))
            barcode_id = str(item.get('barcode_id', ''))
            centroid = np.array(
                [
                    float(item.get('x', 0.0)),
                    float(item.get('y', 0.0)),
                    float(item.get('z', 0.0)),
                ],
                dtype=np.float32,
            )

            for zone in self.zones:
                inside = self.is_inside_zone(centroid, zone)
                active_set = self.active_tracks_by_zone[zone.scanner_id]

                if inside and track_id not in active_set:
                    self.handle_zone_entry(zone, track_id, barcode_id, centroid, stamp_sec)
                    active_set.add(track_id)
                elif not inside and track_id in active_set:
                    active_set.remove(track_id)

    def is_inside_zone(self, position: np.ndarray, zone: ScannerZone) -> bool:
        half_size = zone.size * 0.5
        min_bound = zone.center - half_size
        max_bound = zone.center + half_size
        return bool(np.all(position >= min_bound) and np.all(position <= max_bound))

    def handle_zone_entry(
        self,
        zone: ScannerZone,
        track_id: int,
        barcode_id: str,
        centroid: np.ndarray,
        stamp_sec: float,
    ) -> None:
        if zone.direction == 'entry':
            if barcode_id:
                return
            barcode_id = self.generate_barcode_id()
        else:
            if not barcode_id:
                return

        event_id = self.generate_event_id(zone.scanner_id)
        payload = build_scan_event_payload(
        event_id=event_id,
        scanner_id=zone.scanner_id,
        direction=zone.direction,
        barcode_id=barcode_id,
        stamp_sec=stamp_sec,
        position_x=float(centroid[0]),
        position_y=float(centroid[1]),
        position_z=float(centroid[2]),
        track_id=int(track_id),
    )
        self.scan_event_pub.publish(make_string_message(payload))

        self.get_logger().info(
            f'Virtual scan event: scanner={zone.scanner_id} direction={zone.direction} '
            f'barcode={barcode_id} near track T{track_id}.'
        )

    def generate_barcode_id(self) -> str:
        barcode_id = f'{self.barcode_prefix}_{self.next_barcode_index:04d}'
        self.next_barcode_index += 1
        return barcode_id

    def generate_event_id(self, scanner_id: str) -> str:
        event_id = f'{scanner_id}_{self.next_event_index:06d}'
        self.next_event_index += 1
        return event_id

    def publish_zone_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for zone in self.zones:
            box = Marker()
            box.header.frame_id = self.target_frame
            box.header.stamp = stamp
            box.ns = 'scanner_zones'
            box.id = marker_id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(zone.center[0])
            box.pose.position.y = float(zone.center[1])
            box.pose.position.z = float(zone.center[2])
            box.pose.orientation.w = 1.0
            box.scale.x = float(zone.size[0])
            box.scale.y = float(zone.size[1])
            box.scale.z = float(zone.size[2])

            if zone.direction == 'entry':
                box.color.r = 0.0
                box.color.g = 0.4
                box.color.b = 1.0
            else:
                box.color.r = 1.0
                box.color.g = 0.4
                box.color.b = 0.0

            box.color.a = 0.15
            marker_array.markers.append(box)
            marker_id += 1

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = stamp
            text.ns = 'scanner_zone_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(zone.center[0])
            text.pose.position.y = float(zone.center[1])
            text.pose.position.z = float(zone.center[2] + zone.size[2] * 0.5 + 0.25)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.28
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'{zone.scanner_id} ({zone.direction})'
            marker_array.markers.append(text)
            marker_id += 1

        self.scanner_marker_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VirtualScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
