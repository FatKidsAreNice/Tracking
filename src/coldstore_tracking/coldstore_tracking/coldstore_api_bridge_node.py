from __future__ import annotations

import json
import math
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

from .event_utils import (
    as_float,
    as_int,
    build_scan_event_payload,
    get_payload_list,
    make_string_message,
    parse_string_message,
)
from .floorplan_overlay_utils import clamp_float, load_floorplan_image, render_floorplan_background


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
        self.declare_parameter('map_pixels_per_meter', 35.0)
        self.declare_parameter('map_rotation_deg', 0.0)
        self.declare_parameter('auto_trim_rotation_borders', False)
        self.declare_parameter('background_mode', 'floorplan')
        self.declare_parameter('floorplan_image_path', 'floorplan_background.png')
        self.declare_parameter('floorplan_rotation_deg', 157.0)
        self.declare_parameter('floorplan_fit_mode', 'contain')
        self.declare_parameter('floorplan_scale', 1.0)
        self.declare_parameter('floorplan_offset_x_ratio', 0.0)
        self.declare_parameter('floorplan_offset_y_ratio', 0.0)
        self.declare_parameter('show_lost_tracks', True)
        self.declare_parameter('show_track_heading_arrows', False)
        self.declare_parameter('overview_image_include_tracks', False)
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
        self.map_pixels_per_meter = float(self.get_parameter('map_pixels_per_meter').value)
        self.map_rotation_deg = float(self.get_parameter('map_rotation_deg').value)
        self.map_rotation_rad = math.radians(self.map_rotation_deg)
        self.auto_trim_rotation_borders = bool(self.get_parameter('auto_trim_rotation_borders').value)
        self.background_mode = str(self.get_parameter('background_mode').value).strip().lower() or 'floorplan'
        self.floorplan_image_path = str(self.get_parameter('floorplan_image_path').value).strip()
        self.floorplan_rotation_deg = float(self.get_parameter('floorplan_rotation_deg').value)
        self.floorplan_fit_mode = str(self.get_parameter('floorplan_fit_mode').value).strip().lower() or 'contain'
        self.floorplan_scale = clamp_float(self.get_parameter('floorplan_scale').value, 0.2, 5.0, 1.0)
        self.floorplan_offset_x_ratio = clamp_float(self.get_parameter('floorplan_offset_x_ratio').value, -1.0, 1.0, 0.0)
        self.floorplan_offset_y_ratio = clamp_float(self.get_parameter('floorplan_offset_y_ratio').value, -1.0, 1.0, 0.0)
        self.show_lost_tracks = bool(self.get_parameter('show_lost_tracks').value)
        self.show_track_heading_arrows = bool(self.get_parameter('show_track_heading_arrows').value)
        self.overview_image_include_tracks = bool(self.get_parameter('overview_image_include_tracks').value)
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
        self.latest_bev_image: np.ndarray | None = None
        self.floorplan_image = load_floorplan_image(self.floorplan_image_path, self.floorplan_rotation_deg)
        self.canvas_padding = 20

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
                if parsed.path in ('/overview-image', '/api/coldstore/overview-image'):
                    node.handle_overview_image_request(self)
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
            self.latest_bev_image = rgb_image.copy()

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
        payload['overview_image_url'] = f'{self.resolve_base_url(handler)}/api/coldstore/overview-image'

        handler.send_json_response(HTTPStatus.OK, payload)

    def handle_overview_image_request(self, handler: BaseHTTPRequestHandler) -> None:
        with self.data_lock:
            payload = deepcopy(self.latest_track_payload)
            bev_image = None if self.latest_bev_image is None else self.latest_bev_image.copy()

        rendered = self.render_overview_image(payload, bev_image)
        if rendered is None:
            handler.send_error_response(HTTPStatus.NOT_FOUND, 'No overview image available yet.')
            return

        success, png_buffer = cv2.imencode('.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        if not success:
            handler.send_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to encode overview image.')
            return

        handler.send_image_response(HTTPStatus.OK, png_buffer.tobytes())

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

    def render_overview_image(self, payload: Dict[str, Any], bev_image: np.ndarray | None) -> np.ndarray | None:
        tracks = [track for track in get_payload_list(payload, 'tracks') if isinstance(track, dict)]
        if not self.show_lost_tracks:
            tracks = [track for track in tracks if str(track.get('state', '')) != 'lost']

        canvas_width, canvas_height = self.get_canvas_size()
        background = self.render_overview_background(canvas_width, canvas_height, bev_image)
        if background is None:
            return None

        image = background.copy()
        if self.overview_image_include_tracks:
            for track in sorted(tracks, key=lambda item: as_int(item.get('track_id', 0))):
                track_id = as_int(track.get('track_id', 0))
                x_px, y_px = self.world_to_canvas(
                    as_float(track.get('x', 0.0)),
                    as_float(track.get('y', 0.0)),
                    canvas_width,
                    canvas_height,
                )
                is_highlighted = track_id in self.highlighted_racks
                color_rgb = self.get_track_color(track, is_highlighted)
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                radius = 10 if is_highlighted else 5
                cv2.circle(image, (int(round(x_px)), int(round(y_px))), radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
                if is_highlighted:
                    cv2.circle(image, (int(round(x_px)), int(round(y_px))), radius + 4, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                if self.show_track_heading_arrows:
                    yaw = as_float(track.get('yaw', 0.0)) + self.map_rotation_rad
                    arrow_length = 24 if is_highlighted else 18
                    arrow_x = int(round(x_px + math.cos(yaw) * arrow_length))
                    arrow_y = int(round(y_px - math.sin(yaw) * arrow_length))
                    cv2.arrowedLine(
                        image,
                        (int(round(x_px)), int(round(y_px))),
                        (arrow_x, arrow_y),
                        color_bgr,
                        thickness=3 if is_highlighted else 2,
                        tipLength=0.28,
                        line_type=cv2.LINE_AA,
                    )

                label_text = self.build_display_label(track)
                text_origin = (int(round(x_px + 10)), int(round(y_px - 10)))
                cv2.putText(image, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, label_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        return image

    def render_overview_background(self, canvas_width: int, canvas_height: int, bev_image: np.ndarray | None) -> np.ndarray | None:
        target_size = (
            max(canvas_width - 2 * self.canvas_padding, 1),
            max(canvas_height - 2 * self.canvas_padding, 1),
        )
        if self.background_mode == 'floorplan' and self.floorplan_image is not None:
            background = render_floorplan_background(
                floorplan_image=self.floorplan_image,
                target_size=target_size,
                scale=self.floorplan_scale,
                offset_x_ratio=self.floorplan_offset_x_ratio,
                offset_y_ratio=self.floorplan_offset_y_ratio,
                fit_mode=self.floorplan_fit_mode,
            )
        elif bev_image is not None:
            background = cv2.resize(bev_image, target_size, interpolation=cv2.INTER_AREA)
        else:
            background = None

        if background is None:
            return None

        canvas = np.full((canvas_height, canvas_width, 3), (243, 247, 244), dtype=np.uint8)
        x_start = self.canvas_padding
        y_start = self.canvas_padding
        canvas[y_start:y_start + background.shape[0], x_start:x_start + background.shape[1]] = background
        return canvas

    def get_canvas_size(self) -> tuple[int, int]:
        roi_width_m = max(self.map_roi_max[0] - self.map_roi_min[0], 1.0)
        roi_height_m = max(self.map_roi_max[1] - self.map_roi_min[1], 1.0)
        canvas_width = int(round(roi_width_m * self.map_pixels_per_meter + 2 * self.canvas_padding))
        canvas_height = int(round(roi_height_m * self.map_pixels_per_meter + 2 * self.canvas_padding))
        return max(canvas_width, 64), max(canvas_height, 64)

    def build_display_label(self, track: Dict[str, Any]) -> str:
        track_id = as_int(track.get('track_id', 0))
        if self.lookup_mode == 'barcode_id':
            barcode_id = str(track.get('barcode_id', '')).strip()
            if barcode_id:
                return barcode_id
        return f'T{track_id}'

    def get_track_color(self, track: Dict[str, Any], is_highlighted: bool) -> tuple[int, int, int]:
        if is_highlighted:
            return (255, 34, 85)
        motion_state = str(track.get('motion_state', ''))
        if motion_state == 'moving':
            return (0, 162, 75)
        if motion_state == 'newly_appeared':
            return (47, 125, 255)
        if motion_state == 'occluded':
            return (209, 123, 0)
        if motion_state == 'disappeared':
            return (125, 125, 125)
        return (204, 34, 34)

    def world_to_canvas(self, x_world: float, y_world: float, canvas_width: int, canvas_height: int) -> tuple[float, float]:
        x_world, y_world = self.rotate_world_xy(x_world, y_world)
        min_x, min_y = self.map_roi_min
        max_x, max_y = self.map_roi_max
        usable_width = canvas_width - 2 * self.canvas_padding
        usable_height = canvas_height - 2 * self.canvas_padding

        x_ratio = 0.0 if max_x <= min_x else (x_world - min_x) / (max_x - min_x)
        y_ratio = 0.0 if max_y <= min_y else (max_y - y_world) / (max_y - min_y)
        x_ratio = min(max(x_ratio, 0.0), 1.0)
        y_ratio = min(max(y_ratio, 0.0), 1.0)

        x_local = x_ratio * usable_width
        y_local = y_ratio * usable_height
        return self.canvas_padding + x_local, self.canvas_padding + y_local

    def rotate_world_xy(self, x_world: float, y_world: float) -> tuple[float, float]:
        if abs(self.map_rotation_rad) <= 1e-6:
            return x_world, y_world

        center_x = 0.5 * (self.map_roi_min[0] + self.map_roi_max[0])
        center_y = 0.5 * (self.map_roi_min[1] + self.map_roi_max[1])
        delta_x = x_world - center_x
        delta_y = y_world - center_y
        cos_angle = math.cos(self.map_rotation_rad)
        sin_angle = math.sin(self.map_rotation_rad)
        rotated_x = cos_angle * delta_x - sin_angle * delta_y + center_x
        rotated_y = sin_angle * delta_x + cos_angle * delta_y + center_y
        return rotated_x, rotated_y

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
