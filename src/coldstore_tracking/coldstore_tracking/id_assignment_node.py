from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .event_utils import build_assignment_payload, make_string_message, parse_string_message


class IdAssignmentNode(Node):
    def __init__(self) -> None:
        super().__init__('id_assignment_node')

        self.declare_parameter('max_assignment_distance', 1.5)
        self.max_assignment_distance = float(self.get_parameter('max_assignment_distance').value)

        self.latest_tracks: Dict[int, Dict] = {}

        self.track_state_sub = self.create_subscription(
            String,
            '/tracking/track_states',
            self.track_state_callback,
            10,
        )
        self.scan_event_sub = self.create_subscription(
            String,
            '/tracking/scan_events',
            self.scan_event_callback,
            10,
        )

        self.id_assignment_pub = self.create_publisher(String, '/tracking/id_assignments', 10)
        self.remove_track_pub = self.create_publisher(String, '/tracking/remove_track_events', 10)

        self.get_logger().info('id_assignment_node started.')

    def track_state_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = payload.get('tracks', [])

        self.latest_tracks = {}
        for item in tracks:
            track_id = int(item.get('track_id', 0))
            if track_id <= 0:
                continue
            self.latest_tracks[track_id] = item

    def scan_event_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)

        event_id = str(payload.get('event_id', ''))
        scanner_id = str(payload.get('scanner_id', ''))
        direction = str(payload.get('direction', ''))
        barcode_id = str(payload.get('barcode_id', ''))
        stamp_sec = float(payload.get('stamp_sec', 0.0))
        event_track_id = int(payload.get('track_id', 0))

        position = payload.get('position', {})
        event_position = np.array(
            [
                float(position.get('x', 0.0)),
                float(position.get('y', 0.0)),
                float(position.get('z', 0.0)),
            ],
            dtype=np.float32,
        )

        if direction == 'entry':
            self.handle_entry_event(
                event_id=event_id,
                scanner_id=scanner_id,
                barcode_id=barcode_id,
                stamp_sec=stamp_sec,
                event_position=event_position,
                event_track_id=event_track_id,
            )
            return

        if direction == 'exit':
            self.handle_exit_event(
                event_id=event_id,
                scanner_id=scanner_id,
                barcode_id=barcode_id,
                stamp_sec=stamp_sec,
                event_position=event_position,
                event_track_id=event_track_id,
            )
            return

    def handle_entry_event(
        self,
        event_id: str,
        scanner_id: str,
        barcode_id: str,
        stamp_sec: float,
        event_position: np.ndarray,
        event_track_id: int,
    ) -> None:
        best_track_id: Optional[int] = None

        if event_track_id > 0:
            best_track_id = event_track_id

        if best_track_id is None:
            best_track_id = self.find_nearest_track(
                event_position=event_position,
                require_empty_barcode=True,
                barcode_id='',
            )

        if best_track_id is None:
            self.get_logger().warning(
                f'No matching unassigned track found for entry event {event_id}.',
                throttle_duration_sec=5.0,
            )
            return

        payload = build_assignment_payload(
            event_id=event_id,
            scanner_id=scanner_id,
            direction='entry',
            barcode_id=barcode_id,
            track_id=best_track_id,
            stamp_sec=stamp_sec,
        )
        self.id_assignment_pub.publish(make_string_message(payload))
        self.get_logger().info(
            f'Assigned barcode "{barcode_id}" to track T{best_track_id} from {scanner_id}.'
        )

    def handle_exit_event(
        self,
        event_id: str,
        scanner_id: str,
        barcode_id: str,
        stamp_sec: float,
        event_position: np.ndarray,
        event_track_id: int,
    ) -> None:
        best_track_id: Optional[int] = None

        if event_track_id > 0:
            best_track_id = event_track_id

        if best_track_id is None and barcode_id:
            best_track_id = self.find_track_by_barcode(barcode_id)

        if best_track_id is None:
            best_track_id = self.find_nearest_track(
                event_position=event_position,
                require_empty_barcode=False,
                barcode_id=barcode_id,
            )

        if best_track_id is None:
            self.get_logger().warning(
                f'No matching assigned track found for exit event {event_id}.',
                throttle_duration_sec=5.0,
            )
            return

        payload = build_assignment_payload(
            event_id=event_id,
            scanner_id=scanner_id,
            direction='exit',
            barcode_id=barcode_id,
            track_id=best_track_id,
            stamp_sec=stamp_sec,
        )
        self.remove_track_pub.publish(make_string_message(payload))
        self.get_logger().info(
            f'Remove request for barcode "{barcode_id}" on track T{best_track_id} from {scanner_id}.'
        )

    def find_track_by_barcode(self, barcode_id: str) -> Optional[int]:
        if not barcode_id:
            return None

        for track_id, track in self.latest_tracks.items():
            if str(track.get('barcode_id', '')) == barcode_id:
                return track_id

        return None

    def find_nearest_track(
        self,
        event_position: np.ndarray,
        require_empty_barcode: bool,
        barcode_id: str,
    ) -> Optional[int]:
        best_track_id: Optional[int] = None
        best_distance: Optional[float] = None

        for track_id, track in self.latest_tracks.items():
            track_barcode = str(track.get('barcode_id', ''))

            if require_empty_barcode and track_barcode:
                continue

            if not require_empty_barcode and barcode_id and track_barcode and track_barcode != barcode_id:
                continue

            track_position = np.array(
                [
                    float(track.get('x', 0.0)),
                    float(track.get('y', 0.0)),
                    float(track.get('z', 0.0)),
                ],
                dtype=np.float32,
            )

            distance = float(np.linalg.norm(track_position - event_position))
            if distance > self.max_assignment_distance:
                continue

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        return best_track_id


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IdAssignmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()