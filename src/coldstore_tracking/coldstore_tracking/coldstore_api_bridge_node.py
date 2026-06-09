from __future__ import annotations

import json
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

import cv2
import numpy as np
import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .event_utils import build_scan_event_payload, get_payload_list, make_string_message, parse_string_message


class ColdstoreApiBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('coldstore_api_bridge_node')

        self.declare_parameter('api_host', '0.0.0.0')
        self.declare_parameter('api_port', 8000)
        self.declare_parameter('public_base_url', 'http://10.10.121.30:8000')
        self.declare_parameter('track_state_topic', '/tracking/track_states')
        self.declare_parameter('bev_image_topic', '/detection/bev_image')
        self.declare_parameter('scan_event_topic', '/tracking/scan_events')
        self.declare_parameter('lookup_mode', 'track_id')
        self.declare_parameter('map_roi_min', [-14.5, -15.0])
        self.declare_parameter('map_roi_max', [9.0, 6.0])
        self.declare_parameter('highlighted_racks', [])
        self.declare_parameter('coldstore_sections_json', '[]')
        self.declare_parameter(
            'scanner_config_json',
            json.dumps(
                {
                    'coldstore-entry-01': {
                        'direction': 'entry',
                        'position': {'x': -12.8, 'y': 4.1, 'z': 0.0},
                    }
                }
            ),
        )

        self.api_host = str(self.get_parameter('api_host').value).strip() or '0.0.0.0'
        self.api_port = int(self.get_parameter('api_port').value)
        self.public_base_url = str(self.get_parameter('public_base_url').value).strip()
        self.track_state_topic = str(self.get_parameter('track_state_topic').value)
        self.bev_image_topic = str(self.get_parameter('bev_image_topic').value)
        self.scan_event_topic = str(self.get_parameter('scan_event_topic').value)
        self.lookup_mode = str(self.get_parameter('lookup_mode').value).strip() or 'track_id'
        self.map_roi_min = [float(value) for value in self.get_parameter('map_roi_min').value]
        self.map_roi_max = [float(value) for value in self.get_parameter('map_roi_max').value]
        self.highlighted_racks = self.load_integer_list_parameter('highlighted_racks')
        self.coldstore_sections = self.parse_json_parameter(
            str(self.get_parameter('coldstore_sections_json').value),
            default=[],
        )
        self.scanner_config = self.parse_json_parameter(
            str(self.get_parameter('scanner_config_json').value),
            default={},
        )

        self.data_lock = threading.Lock()
        self.latest_track_payload: Dict[str, Any] = {'stamp_sec': 0.0, 'frame_id': '', 'tracks': []}
        self.latest_bev_stamp_sec = 0.0
        self.latest_bev_png_bytes: bytes | None = None

        self.create_subscription(String, self.track_state_topic, self.track_state_callback, 10)
        self.create_subscription(Image, self.bev_image_topic, self.bev_image_callback, 10)
        self.scan_event_pub = self.create_publisher(String, self.scan_event_topic, 10)

        self.http_server = self.create_http_server()
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()

        self.get_logger().info(
            f'coldstore_api_bridge_node started. host={self.api_host}, port={self.api_port}, '
            f'track_state_topic={self.track_state_topic}, bev_image_topic={self.bev_image_topic}, '
            f'scan_event_topic={self.scan_event_topic}'
        )

    def load_integer_list_parameter(self, name: str) -> list[int]:
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            return []

        if value is None:
            return []

        return [int(item) for item in value]

    def parse_json_parameter(self, raw_value: str, default):
        if not raw_value:
            return default
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            self.get_logger().warning('Failed to parse JSON parameter, using default.')
            return default
        return parsed

    def create_http_server(self) -> ThreadingHTTPServer:
        node = self

        class Handler(BaseHTTPRequestHandler):
            server_version = 'ColdstoreApiBridge/0.1'

            def do_OPTIONS(self) -> None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_cors_headers()
                self.end_headers()

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in ('/overview', '/api/coldstore/overview'):
                    node.handle_overview_request(self)
                    return
                if parsed.path in ('/bev-image', '/api/coldstore/bev-image'):
                    node.handle_bev_image_request(self)
                    return
                self.send_error_response(HTTPStatus.NOT_FOUND, 'Endpoint not found.')

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in ('/barcode-scan', '/api/coldstore/barcode-scan'):
                    node.handle_barcode_scan_request(self)
                    return
                self.send_error_response(HTTPStatus.NOT_FOUND, 'Endpoint not found.')

            def send_json_response(self, status: int, payload: Dict[str, Any]) -> None:
                response_body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_cors_headers()
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def send_image_response(self, status: int, image_bytes: bytes) -> None:
                self.send_response(status)
                self.send_cors_headers()
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(image_bytes)))
                self.end_headers()
                self.wfile.write(image_bytes)

            def send_error_response(self, status: int, message: str) -> None:
                self.send_json_response(status, {'accepted': False, 'message': message})

            def send_cors_headers(self) -> None:
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

            def log_message(self, format: str, *args) -> None:
                node.get_logger().debug(format % args)

        return ThreadingHTTPServer((self.api_host, self.api_port), Handler)

    def track_state_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = [track for track in get_payload_list(payload, 'tracks') if isinstance(track, dict)]
        normalized_payload = {
            'stamp_sec': float(payload.get('stamp_sec', 0.0)),
            'frame_id': str(payload.get('frame_id', '')),
            'tracks': tracks,
        }
        with self.data_lock:
            self.latest_track_payload = normalized_payload

    def bev_image_callback(self, msg: Image) -> None:
        if msg.encoding != 'rgb8' or msg.height <= 0 or msg.width <= 0:
            return

        try:
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except ValueError:
            return

        success, png_buffer = cv2.imencode('.png', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        if not success:
            return

        bev_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self.data_lock:
            self.latest_bev_stamp_sec = bev_stamp_sec
            self.latest_bev_png_bytes = png_buffer.tobytes()

    def handle_overview_request(self, handler: BaseHTTPRequestHandler) -> None:
        with self.data_lock:
            payload = deepcopy(self.latest_track_payload)
            bev_stamp_sec = float(self.latest_bev_stamp_sec)

        payload['bev_stamp_sec'] = bev_stamp_sec
        payload['lookup_mode'] = self.lookup_mode
        payload['map_roi_min'] = list(self.map_roi_min)
        payload['map_roi_max'] = list(self.map_roi_max)
        payload['highlighted_racks'] = list(self.highlighted_racks)
        payload['coldstore'] = {'sections': self.coldstore_sections}
        payload['bev_image_url'] = f'{self.resolve_base_url(handler)}/api/coldstore/bev-image'

        handler.send_json_response(HTTPStatus.OK, payload)

    def handle_bev_image_request(self, handler: BaseHTTPRequestHandler) -> None:
        with self.data_lock:
            image_bytes = self.latest_bev_png_bytes

        if image_bytes is None:
            handler.send_error_response(HTTPStatus.NOT_FOUND, 'No BEV image available yet.')
            return

        handler.send_image_response(HTTPStatus.OK, image_bytes)

    def handle_barcode_scan_request(self, handler: BaseHTTPRequestHandler) -> None:
        content_length = int(handler.headers.get('Content-Length', '0') or '0')
        if content_length <= 0:
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'Request body is required.')
            return

        try:
            request_body = handler.rfile.read(content_length)
            payload = json.loads(request_body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'Request body must be valid JSON.')
            return

        if not isinstance(payload, dict):
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'JSON body must be an object.')
            return

        barcode_id = str(payload.get('barcode_id') or payload.get('barcode') or '').strip()
        scanner_id = str(payload.get('scanner_id', '')).strip()
        direction = str(payload.get('direction', '')).strip().lower()
        scanned_at = str(payload.get('scanned_at', '')).strip()

        if not barcode_id:
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'barcode_id is required.')
            return
        if not scanner_id:
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'scanner_id is required.')
            return

        scanner_info = self.scanner_config.get(scanner_id)
        if not isinstance(scanner_info, dict):
            handler.send_error_response(HTTPStatus.BAD_REQUEST, f'Unknown scanner_id: {scanner_id}')
            return

        if not direction:
            direction = str(scanner_info.get('direction', '')).strip().lower()
        if direction not in ('entry', 'exit'):
            handler.send_error_response(HTTPStatus.BAD_REQUEST, 'direction must be entry or exit.')
            return

        position = scanner_info.get('position', {})
        if not isinstance(position, dict):
            handler.send_error_response(HTTPStatus.BAD_REQUEST, f'Scanner {scanner_id} has invalid position config.')
            return

        stamp_sec = self.parse_scan_timestamp(scanned_at)
        event_id = f'scan-{uuid.uuid4().hex[:12]}'
        scan_event_payload = build_scan_event_payload(
            event_id=event_id,
            scanner_id=scanner_id,
            direction=direction,
            barcode_id=barcode_id,
            stamp_sec=stamp_sec,
            position_x=float(position.get('x', 0.0)),
            position_y=float(position.get('y', 0.0)),
            position_z=float(position.get('z', 0.0)),
        )
        self.scan_event_pub.publish(make_string_message(scan_event_payload))

        handler.send_json_response(
            HTTPStatus.ACCEPTED,
            {
                'accepted': True,
                'event_id': event_id,
                'barcode_id': barcode_id,
                'scanner_id': scanner_id,
                'direction': direction,
                'stamp_sec': stamp_sec,
                'message': 'Scan accepted and forwarded to tracking.',
            },
        )

    def resolve_base_url(self, handler: BaseHTTPRequestHandler) -> str:
        if self.public_base_url:
            return self.public_base_url.rstrip('/')
        host_header = str(handler.headers.get('Host', '')).strip()
        if host_header:
            return f'http://{host_header}'
        return f'http://{self.api_host}:{self.api_port}'

    @staticmethod
    def parse_scan_timestamp(scanned_at: str) -> float:
        if not scanned_at:
            return time.time()
        try:
            return datetime.fromisoformat(scanned_at.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return time.time()

    def destroy_node(self) -> bool:
        if hasattr(self, 'http_server') and self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None
        if hasattr(self, 'http_thread') and self.http_thread.is_alive():
            self.http_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColdstoreApiBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
