from __future__ import annotations

import base64
import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .event_utils import parse_string_message


class TrackOverviewGuiNode(Node):
    def __init__(self) -> None:
        super().__init__('track_overview_gui_node')

        self.declare_parameter('track_state_topic', '/tracking/track_states')
        self.declare_parameter('map_roi_min', [-14.5, -15.0])
        self.declare_parameter('map_roi_max', [9.0, 6.0])
        self.declare_parameter('map_pixels_per_meter', 35.0)
        self.declare_parameter('show_lost_tracks', True)
        self.declare_parameter('refresh_rate_hz', 5.0)
        self.declare_parameter('lookup_mode', 'track_id')
        self.declare_parameter('bev_image_topic', '/detection/bev_image')
        self.declare_parameter('show_bev_background', True)

        self.track_state_topic = str(self.get_parameter('track_state_topic').value)
        self.map_roi_min = [float(value) for value in self.get_parameter('map_roi_min').value]
        self.map_roi_max = [float(value) for value in self.get_parameter('map_roi_max').value]
        self.map_pixels_per_meter = float(self.get_parameter('map_pixels_per_meter').value)
        self.show_lost_tracks = bool(self.get_parameter('show_lost_tracks').value)
        self.refresh_rate_hz = max(float(self.get_parameter('refresh_rate_hz').value), 1.0)
        self.lookup_mode = str(self.get_parameter('lookup_mode').value).strip() or 'track_id'
        self.bev_image_topic = str(self.get_parameter('bev_image_topic').value)
        self.show_bev_background = bool(self.get_parameter('show_bev_background').value)

        self.latest_payload: Dict[str, Any] = {}
        self.latest_tracks_by_id: Dict[int, Dict[str, Any]] = {}
        self.selected_track_id: int | None = None
        self.canvas_padding = 20
        self.latest_bev_image: np.ndarray | None = None
        self.bev_photo_image: tk.PhotoImage | None = None

        self.create_subscription(String, self.track_state_topic, self.track_state_callback, 10)
        self.create_subscription(Image, self.bev_image_topic, self.bev_image_callback, 10)

        self.root = tk.Tk()
        self.root.title('Coldstore Track Overview')
        self.root.geometry('1420x860')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.lookup_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Waiting for /tracking/track_states ...')
        self.detail_var = tk.StringVar(value='No track selected.')

        self.build_ui()
        self.get_logger().info(
            f'track_overview_gui_node started. topic={self.track_state_topic}, '
            f'bev_image_topic={self.bev_image_topic}, lookup_mode={self.lookup_mode}'
        )

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.grid(row=0, column=0, columnspan=2, sticky='ew')
        toolbar.columnconfigure(4, weight=1)

        ttk.Label(toolbar, text='Track ID Lookup').grid(row=0, column=0, sticky='w')
        lookup_entry = ttk.Entry(toolbar, textvariable=self.lookup_var, width=18)
        lookup_entry.grid(row=0, column=1, padx=(8, 8), sticky='w')
        lookup_entry.bind('<Return>', self.on_lookup)
        ttk.Button(toolbar, text='Find', command=self.on_lookup).grid(row=0, column=2, sticky='w')
        ttk.Button(toolbar, text='Clear', command=self.clear_selection).grid(row=0, column=3, padx=(8, 0), sticky='w')
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=4, padx=(16, 0), sticky='e')

        left_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        left_frame.grid(row=1, column=0, sticky='nsew')
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        columns = (
            'track_id',
            'class_name',
            'state',
            'motion_state',
            'confidence',
            'position',
            'yaw',
            'hit_count',
            'source_missed_count',
        )
        self.tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=24)
        headings = {
            'track_id': 'Track ID',
            'class_name': 'Class',
            'state': 'State',
            'motion_state': 'Motion',
            'confidence': 'Conf',
            'position': 'x / y / z',
            'yaw': 'Yaw',
            'hit_count': 'Hits',
            'source_missed_count': 'Missed',
        }
        widths = {
            'track_id': 80,
            'class_name': 110,
            'state': 90,
            'motion_state': 120,
            'confidence': 65,
            'position': 210,
            'yaw': 70,
            'hit_count': 65,
            'source_missed_count': 70,
        }
        for key in columns:
            self.tree.heading(key, text=headings[key])
            self.tree.column(key, width=widths[key], anchor='center')
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')

        right_frame = ttk.Frame(self.root, padding=(0, 0, 10, 10))
        right_frame.grid(row=1, column=1, sticky='nsew')
        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        roi_width_m = max(self.map_roi_max[0] - self.map_roi_min[0], 1.0)
        roi_height_m = max(self.map_roi_max[1] - self.map_roi_min[1], 1.0)
        initial_canvas_width = int(roi_width_m * self.map_pixels_per_meter + 2 * self.canvas_padding)
        initial_canvas_height = int(roi_height_m * self.map_pixels_per_meter + 2 * self.canvas_padding)
        self.canvas = tk.Canvas(
            right_frame,
            width=initial_canvas_width,
            height=initial_canvas_height,
            background='#f3f7f4',
            highlightthickness=1,
            highlightbackground='#c4d0c8',
        )
        self.canvas.grid(row=0, column=0, sticky='nsew')

        detail_frame = ttk.LabelFrame(right_frame, text='Track Details', padding=12)
        detail_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        detail_frame.columnconfigure(0, weight=1)
        ttk.Label(detail_frame, textvariable=self.detail_var, justify='left').grid(row=0, column=0, sticky='nw')

    def track_state_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = payload.get('tracks', [])

        if not self.show_lost_tracks:
            tracks = [track for track in tracks if str(track.get('state', '')) != 'lost']

        self.latest_payload = payload
        self.latest_tracks_by_id = {
            int(track.get('track_id', 0)): track
            for track in tracks
            if int(track.get('track_id', 0)) > 0
        }

        if self.selected_track_id is not None and self.selected_track_id not in self.latest_tracks_by_id:
            self.selected_track_id = None

    def on_lookup(self, _event=None) -> None:
        query = self.lookup_var.get().strip()
        if not query:
            self.status_var.set('Enter a track ID.')
            return

        try:
            track_id = int(query)
        except ValueError:
            self.status_var.set('Track ID must be numeric.')
            return

        if track_id not in self.latest_tracks_by_id:
            self.status_var.set(f'Track ID {track_id} not found.')
            return

        self.selected_track_id = track_id
        self.select_tree_row(track_id)
        self.status_var.set(f'Selected track T{track_id}.')

    def bev_image_callback(self, msg: Image) -> None:
        if msg.encoding != 'rgb8':
            return

        if msg.height <= 0 or msg.width <= 0:
            return

        bev_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_bev_image = bev_image.copy()

    def clear_selection(self) -> None:
        self.selected_track_id = None
        self.lookup_var.set('')
        for item_id in self.tree.selection():
            self.tree.selection_remove(item_id)
        self.status_var.set('Selection cleared.')
        self.update_detail_panel(None)

    def on_tree_select(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            self.selected_track_id = None
            self.update_detail_panel(None)
            return

        try:
            track_id = int(selection[0])
        except ValueError:
            return

        self.selected_track_id = track_id
        self.lookup_var.set(str(track_id))
        self.status_var.set(f'Selected track T{track_id}.')

    def select_tree_row(self, track_id: int) -> None:
        item_id = str(track_id)
        if not self.tree.exists(item_id):
            return
        self.tree.selection_set(item_id)
        self.tree.focus(item_id)
        self.tree.see(item_id)

    def refresh_ui(self) -> None:
        self.populate_tree()
        self.draw_map()
        selected_track = self.latest_tracks_by_id.get(self.selected_track_id) if self.selected_track_id is not None else None
        self.update_detail_panel(selected_track)
        self.root.after(int(1000 / self.refresh_rate_hz), self.refresh_ui)

    def populate_tree(self) -> None:
        existing_ids = set(self.tree.get_children())
        sorted_tracks = sorted(self.latest_tracks_by_id.values(), key=lambda track: int(track.get('track_id', 0)))
        active_ids = set()

        for track in sorted_tracks:
            track_id = int(track.get('track_id', 0))
            item_id = str(track_id)
            active_ids.add(item_id)
            values = (
                track_id,
                str(track.get('class_name', '-')),
                str(track.get('state', '-')),
                str(track.get('motion_state', '-')),
                f"{float(track.get('confidence', 0.0)):.2f}",
                f"{float(track.get('x', 0.0)):.2f} / {float(track.get('y', 0.0)):.2f} / {float(track.get('z', 0.0)):.2f}",
                f"{float(track.get('yaw', 0.0)):.2f}",
                int(track.get('hit_count', 0)),
                int(track.get('source_missed_count', 0)),
            )
            if item_id in existing_ids:
                self.tree.item(item_id, values=values)
            else:
                self.tree.insert('', 'end', iid=item_id, values=values)

        for item_id in existing_ids - active_ids:
            self.tree.delete(item_id)

        if self.selected_track_id is not None:
            self.select_tree_row(self.selected_track_id)

    def draw_map(self) -> None:
        self.canvas.delete('all')
        canvas_width = max(self.canvas.winfo_width(), 420)
        canvas_height = max(self.canvas.winfo_height(), 420)

        self.draw_bev_background(canvas_width, canvas_height)
        self.canvas.create_rectangle(
            self.canvas_padding,
            self.canvas_padding,
            canvas_width - self.canvas_padding,
            canvas_height - self.canvas_padding,
            outline='#3d5a40',
            width=2,
        )
        self.canvas.create_text(
            self.canvas_padding + 8,
            self.canvas_padding - 6,
            text='BEV Rack Positions',
            anchor='sw',
            fill='#2f4f35',
            font=('TkDefaultFont', 11, 'bold'),
        )

        for track in sorted(self.latest_tracks_by_id.values(), key=lambda item: int(item.get('track_id', 0))):
            track_id = int(track.get('track_id', 0))
            x_px, y_px = self.world_to_canvas(
                float(track.get('x', 0.0)),
                float(track.get('y', 0.0)),
                canvas_width,
                canvas_height,
            )
            is_selected = track_id == self.selected_track_id
            radius = 13 if is_selected else 5
            fill = self.get_track_color(track, is_selected)
            outline = '#111111' if is_selected else ''
            self.canvas.create_oval(x_px - radius, y_px - radius, x_px + radius, y_px + radius, fill=fill, outline=outline, width=2 if is_selected else 0)

            yaw = float(track.get('yaw', 0.0))
            arrow_length = 26 if is_selected else 18
            arrow_x = x_px + math.cos(yaw) * arrow_length
            arrow_y = y_px - math.sin(yaw) * arrow_length
            self.canvas.create_line(x_px, y_px, arrow_x, arrow_y, fill=fill, width=3 if is_selected else 2, arrow=tk.LAST)

            label_text = self.build_display_label(track)
            self.canvas.create_text(
                x_px + 10,
                y_px - 12,
                text=label_text,
                anchor='sw',
                fill='#102511',
                font=('TkDefaultFont', 10, 'bold' if is_selected else 'normal'),
            )

    def draw_bev_background(self, canvas_width: int, canvas_height: int) -> None:
        if not self.show_bev_background or self.latest_bev_image is None:
            return

        target_width = max(canvas_width - 2 * self.canvas_padding, 1)
        target_height = max(canvas_height - 2 * self.canvas_padding, 1)
        resized_image = cv2.resize(self.latest_bev_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        success, png_buffer = cv2.imencode('.png', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
        if not success:
            self.status_var.set('Failed to encode BEV background image.')
            return

        try:
            self.bev_photo_image = tk.PhotoImage(data=base64.b64encode(png_buffer.tobytes()).decode('ascii'))
            self.canvas.create_image(
                self.canvas_padding,
                self.canvas_padding,
                anchor='nw',
                image=self.bev_photo_image,
            )
        except tk.TclError as exc:
            self.status_var.set(f'Failed to render BEV background: {exc}')

    def update_detail_panel(self, track: Dict[str, Any] | None) -> None:
        if track is None:
            track_count = len(self.latest_tracks_by_id)
            self.detail_var.set(f'No track selected.\nVisible tracks: {track_count}')
            return

        detail_lines = [
            f"Display ID: {self.build_display_label(track)}",
            f"Track ID: {int(track.get('track_id', 0))}",
            f"Class: {str(track.get('class_name', '-'))}",
            f"State: {str(track.get('state', '-'))}",
            f"Motion: {str(track.get('motion_state', '-'))}",
            f"Confidence: {float(track.get('confidence', 0.0)):.3f}",
            f"Frame: {str(self.latest_payload.get('frame_id', '-'))}",
            f"Position: x={float(track.get('x', 0.0)):.3f}, y={float(track.get('y', 0.0)):.3f}, z={float(track.get('z', 0.0)):.3f}",
            f"Yaw: {float(track.get('yaw', 0.0)):.3f}",
            f"Size: L={float(track.get('length', 0.0)):.3f}, W={float(track.get('width', 0.0)):.3f}, H={float(track.get('height', 0.0)):.3f}",
            f"Hits: {int(track.get('hit_count', 0))} | Missed: {int(track.get('source_missed_count', 0))}",
            f"Lost transitions: {int(track.get('lost_transition_count', 0))}",
            f"Occluded transitions: {int(track.get('occluded_transition_count', 0))}",
            f"Reappeared count: {int(track.get('reappeared_count', 0))}",
            f"Motion state changed at: {float(track.get('last_motion_state_change_sec', 0.0)):.3f}",
            f"Last update: {float(track.get('last_stamp_sec', 0.0)):.3f}",
        ]
        self.detail_var.set('\n'.join(detail_lines))

    def build_display_label(self, track: Dict[str, Any]) -> str:
        track_id = int(track.get('track_id', 0))
        if self.lookup_mode == 'barcode_id':
            barcode_id = str(track.get('barcode_id', '')).strip()
            if barcode_id:
                return barcode_id
        return f'T{track_id}'

    def get_track_color(self, track: Dict[str, Any], is_selected: bool) -> str:
        if is_selected:
            return '#ff2255'
        motion_state = str(track.get('motion_state', ''))
        if motion_state == 'moving':
            return '#00a24b'
        if motion_state == 'newly_appeared':
            return '#2f7dff'
        if motion_state == 'occluded':
            return '#d17b00'
        if motion_state == 'disappeared':
            return '#7d7d7d'
        return '#cc2222'

    def world_to_canvas(self, x_world: float, y_world: float, canvas_width: int, canvas_height: int) -> tuple[float, float]:
        min_x, min_y = self.map_roi_min
        max_x, max_y = self.map_roi_max
        usable_width = canvas_width - 2 * self.canvas_padding
        usable_height = canvas_height - 2 * self.canvas_padding

        x_ratio = 0.0 if max_x <= min_x else (x_world - min_x) / (max_x - min_x)
        y_ratio = 0.0 if max_y <= min_y else (max_y - y_world) / (max_y - min_y)

        x_ratio = min(max(x_ratio, 0.0), 1.0)
        y_ratio = min(max(y_ratio, 0.0), 1.0)

        x_px = self.canvas_padding + x_ratio * usable_width
        y_px = self.canvas_padding + y_ratio * usable_height
        return x_px, y_px

    def spin_once(self) -> None:
        if not rclpy.ok():
            return
        rclpy.spin_once(self, timeout_sec=0.0)
        self.root.after(20, self.spin_once)

    def on_close(self) -> None:
        self.destroy_node()
        self.root.quit()
        self.root.destroy()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackOverviewGuiNode()
    try:
        node.refresh_ui()
        node.spin_once()
        node.root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
