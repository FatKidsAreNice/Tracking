from __future__ import annotations

import base64
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .floorplan_overlay_utils import clamp_float, load_floorplan_image, render_floorplan_background


class FloorplanCalibrationToolNode(Node):
    SAFE_FIT_MODE = 'contain'
    SAFE_SCALE = 1.0
    CRITICAL_SCALE = 1.10
    CONTROLS_PANEL_WIDTH = 430

    def __init__(self) -> None:
        super().__init__('floorplan_calibration_tool_node')

        self.declare_parameter('bev_image_topic', '/detection/bev_image')
        self.declare_parameter('map_roi_min', [-14.5, -15.0])
        self.declare_parameter('map_roi_max', [9.0, 6.0])
        self.declare_parameter('map_pixels_per_meter', 35.0)
        self.declare_parameter('floorplan_image_path', 'floorplan_background.png')
        self.declare_parameter('floorplan_rotation_deg', 180.0)
        self.declare_parameter('floorplan_fit_mode', 'contain')
        self.declare_parameter('floorplan_scale', 1.0)
        self.declare_parameter('floorplan_offset_x_ratio', 0.0)
        self.declare_parameter('floorplan_offset_y_ratio', 0.0)
        self.declare_parameter('bev_preview_scale', 1.0)
        self.declare_parameter('overlay_alpha', 0.55)
        self.declare_parameter('refresh_rate_hz', 5.0)

        self.bev_image_topic = str(self.get_parameter('bev_image_topic').value)
        self.map_roi_min = [float(value) for value in self.get_parameter('map_roi_min').value]
        self.map_roi_max = [float(value) for value in self.get_parameter('map_roi_max').value]
        self.map_pixels_per_meter = float(self.get_parameter('map_pixels_per_meter').value)
        self.floorplan_image_path = str(self.get_parameter('floorplan_image_path').value).strip()
        configured_fit_mode = str(self.get_parameter('floorplan_fit_mode').value).strip().lower()
        self.floorplan_fit_mode = self.SAFE_FIT_MODE
        self.refresh_rate_hz = max(float(self.get_parameter('refresh_rate_hz').value), 1.0)

        self.initial_floorplan_rotation_deg = float(self.get_parameter('floorplan_rotation_deg').value)
        configured_scale = clamp_float(self.get_parameter('floorplan_scale').value, 0.2, 5.0, self.SAFE_SCALE)
        self.initial_floorplan_scale = self.SAFE_SCALE
        self.initial_floorplan_offset_x_ratio = clamp_float(
            self.get_parameter('floorplan_offset_x_ratio').value,
            -1.0,
            1.0,
            0.0,
        )
        self.initial_floorplan_offset_y_ratio = clamp_float(
            self.get_parameter('floorplan_offset_y_ratio').value,
            -1.0,
            1.0,
            0.0,
        )
        self.initial_bev_preview_scale = clamp_float(
            self.get_parameter('bev_preview_scale').value,
            0.5,
            2.0,
            1.0,
        )
        self.initial_overlay_alpha = clamp_float(self.get_parameter('overlay_alpha').value, 0.0, 1.0, 0.55)

        self.latest_bev_image: np.ndarray | None = None
        self.latest_bev_version = 0
        self.rendered_preview_version = -1
        self.rendered_preview_params: tuple[float, float, float, float] | None = None
        self.preview_photo_image: tk.PhotoImage | None = None
        self.warning_color = '#9d2200'
        self.safe_color = '#245c2f'

        self.create_subscription(Image, self.bev_image_topic, self.bev_image_callback, 10)

        self.root = tk.Tk()
        self.root.title('Floorplan Calibration Tool')
        self.root.geometry('1560x920')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.floorplan_rotation_var = tk.DoubleVar(master=self.root, value=self.initial_floorplan_rotation_deg)
        self.floorplan_scale_var = tk.DoubleVar(master=self.root, value=self.initial_floorplan_scale)
        self.floorplan_offset_x_var = tk.DoubleVar(master=self.root, value=self.initial_floorplan_offset_x_ratio)
        self.floorplan_offset_y_var = tk.DoubleVar(master=self.root, value=self.initial_floorplan_offset_y_ratio)
        self.bev_preview_scale_var = tk.DoubleVar(master=self.root, value=self.initial_bev_preview_scale)
        self.overlay_alpha_var = tk.DoubleVar(master=self.root, value=self.initial_overlay_alpha)

        self.status_var = tk.StringVar(value='Waiting for BEV image ...')
        self.warning_var = tk.StringVar()
        self.yaml_var = tk.StringVar()
        self.floorplan_image = self.reload_floorplan_image()
        self.build_ui()
        self.update_yaml_preview()
        self.update_warning_state()

        self.get_logger().info(
            f'floorplan_calibration_tool_node started. bev_image_topic={self.bev_image_topic}, '
            f'floorplan_image_path={self.floorplan_image_path}, '
            f'configured_fit_mode={self.floorplan_fit_mode}, configured_scale={configured_scale:.4f}, '
            f'safe_scale_start={self.initial_floorplan_scale:.4f}'
        )

    def reload_floorplan_image(self) -> np.ndarray | None:
        image = load_floorplan_image(self.floorplan_image_path, self.floorplan_rotation_var.get())
        if image is None:
            self.get_logger().warning(f'Failed to load floorplan image: {self.floorplan_image_path}')
        return image

    def build_ui(self) -> None:
        roi_width_m = max(self.map_roi_max[0] - self.map_roi_min[0], 1.0)
        roi_height_m = max(self.map_roi_max[1] - self.map_roi_min[1], 1.0)
        self.preview_width = int(roi_width_m * self.map_pixels_per_meter)
        self.preview_height = int(roi_height_m * self.map_pixels_per_meter)

        self.root.columnconfigure(0, weight=5, minsize=max(self.preview_width + 20, 720))
        self.root.columnconfigure(1, weight=0, minsize=self.CONTROLS_PANEL_WIDTH)
        self.root.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(self.root, padding=10)
        preview_frame.grid(row=0, column=0, sticky='nsew')
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=self.preview_width,
            height=self.preview_height,
            background='#f3f7f4',
            highlightthickness=1,
            highlightbackground='#c4d0c8',
        )
        self.preview_canvas.grid(row=0, column=0, sticky='nsew')

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=1, sticky='nsew')
        controls.configure(width=self.CONTROLS_PANEL_WIDTH)
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(controls, text='Floorplan Calibration', font=('TkDefaultFont', 12, 'bold')).grid(row=row, column=0, sticky='w')
        row += 1
        ttk.Label(
            controls,
            textvariable=self.status_var,
            justify='left',
            wraplength=self.CONTROLS_PANEL_WIDTH - 24,
        ).grid(row=row, column=0, sticky='w', pady=(0, 8))
        row += 1
        ttk.Label(
            controls,
            text=(
                'Kalibrierreihenfolge:\n'
                '1. Rotation setzen\n'
                '2. floorplan_fit_mode = contain verwenden\n'
                '3. floorplan_scale = 1.0000 lassen\n'
                '4. offset_x_ratio / offset_y_ratio anpassen\n'
                '5. BEV Preview Scale nur fuer den Vergleich anpassen\n'
                '6. floorplan_scale nur wenn unbedingt noetig minimal erhoehen'
            ),
            justify='left',
            wraplength=self.CONTROLS_PANEL_WIDTH - 24,
        ).grid(row=row, column=0, sticky='w', pady=(0, 8))
        row += 1
        self.warning_label = ttk.Label(
            controls,
            textvariable=self.warning_var,
            justify='left',
            foreground=self.warning_color,
            wraplength=self.CONTROLS_PANEL_WIDTH - 24,
        )
        self.warning_label.grid(row=row, column=0, sticky='w', pady=(0, 8))
        row += 1

        row = self.add_slider(controls, row, 'Rotation (deg)', self.floorplan_rotation_var, -360.0, 360.0, 1.0)
        row = self.add_slider(controls, row, 'Scale', self.floorplan_scale_var, 0.5, 2.5, 0.01)
        row = self.add_slider(controls, row, 'Offset X', self.floorplan_offset_x_var, -0.5, 0.5, 0.005)
        row = self.add_slider(controls, row, 'Offset Y', self.floorplan_offset_y_var, -0.5, 0.5, 0.005)
        row = self.add_slider(
            controls,
            row,
            'BEV Preview Scale (not saved)',
            self.bev_preview_scale_var,
            0.5,
            2.0,
            0.01,
        )
        row = self.add_slider(controls, row, 'Overlay Alpha', self.overlay_alpha_var, 0.0, 1.0, 0.01)

        button_row = ttk.Frame(controls)
        button_row.grid(row=row, column=0, sticky='ew', pady=(8, 8))
        ttk.Button(button_row, text='Reload Floorplan', command=self.on_reload_floorplan).pack(side='left')
        ttk.Button(button_row, text='Copy YAML', command=self.copy_yaml_to_clipboard).pack(side='left', padx=(8, 0))
        ttk.Button(button_row, text='Reset', command=self.reset_controls).pack(side='left', padx=(8, 0))
        ttk.Button(
            button_row,
            text='Normalize Floorplan Calibration',
            command=self.normalize_floorplan_calibration,
        ).pack(side='left', padx=(8, 0))
        row += 1

        ttk.Label(controls, text='YAML Parameters').grid(row=row, column=0, sticky='w')
        row += 1
        yaml_label = ttk.Label(
            controls,
            textvariable=self.yaml_var,
            justify='left',
            wraplength=self.CONTROLS_PANEL_WIDTH - 24,
        )
        yaml_label.grid(row=row, column=0, sticky='nw')

    def add_slider(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        resolution: float,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
        row += 1
        scale = tk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            orient='horizontal',
            resolution=resolution,
            variable=variable,
            command=self.on_control_change,
            length=340,
        )
        scale.grid(row=row, column=0, sticky='ew')
        row += 1
        return row

    def on_control_change(self, _value=None) -> None:
        self.update_yaml_preview()
        self.update_warning_state()
        self.rendered_preview_params = None

    def on_reload_floorplan(self) -> None:
        self.floorplan_image = self.reload_floorplan_image()
        self.rendered_preview_params = None
        self.update_yaml_preview()
        self.update_warning_state()

    def reset_controls(self) -> None:
        self.floorplan_rotation_var.set(self.initial_floorplan_rotation_deg)
        self.floorplan_scale_var.set(self.SAFE_SCALE)
        self.floorplan_offset_x_var.set(0.0)
        self.floorplan_offset_y_var.set(0.0)
        self.bev_preview_scale_var.set(1.0)
        self.overlay_alpha_var.set(self.initial_overlay_alpha)
        self.on_reload_floorplan()

    def normalize_floorplan_calibration(self) -> None:
        self.floorplan_scale_var.set(self.SAFE_SCALE)
        self.floorplan_offset_x_var.set(0.0)
        self.floorplan_offset_y_var.set(0.0)
        self.bev_preview_scale_var.set(1.0)
        self.on_reload_floorplan()
        self.status_var.set(
            'Floorplan calibration normalized: fit_mode=contain, floorplan_scale=1.0000, '
            'bev_preview_scale=1.0000, offsets reset, rotation preserved.'
        )

    def copy_yaml_to_clipboard(self) -> None:
        self.log_scale_warning_if_needed(prefix='Copy YAML')
        self.root.clipboard_clear()
        self.root.clipboard_append(self.yaml_var.get())
        self.status_var.set('YAML parameters copied to clipboard.')

    def update_yaml_preview(self) -> None:
        self.yaml_var.set(
            'track_overview_gui_node:\n'
            '  ros__parameters:\n'
            '    background_mode: floorplan\n'
            f'    floorplan_image_path: {self.floorplan_image_path}\n'
            f'    floorplan_rotation_deg: {self.floorplan_rotation_var.get():.2f}\n'
            f'    floorplan_fit_mode: {self.SAFE_FIT_MODE}\n'
            f'    floorplan_scale: {self.floorplan_scale_var.get():.4f}\n'
            f'    floorplan_offset_x_ratio: {self.floorplan_offset_x_var.get():.4f}\n'
            f'    floorplan_offset_y_ratio: {self.floorplan_offset_y_var.get():.4f}\n'
            '    show_bev_background: false\n'
            '    map_rotation_deg: 0.0\n'
            '    auto_trim_rotation_borders: false'
        )

    def update_warning_state(self) -> None:
        scale_value = float(self.floorplan_scale_var.get())
        warning_text = self.build_scale_warning_text(scale_value)
        self.warning_var.set(warning_text)
        if self.is_scale_risky(scale_value):
            self.status_var.set(
                f'Warning active | rotation={self.floorplan_rotation_var.get():.1f} | '
                f'scale={scale_value:.3f} | '
                f'bev_preview_scale={self.bev_preview_scale_var.get():.3f} | '
                f'offset=({self.floorplan_offset_x_var.get():.3f}, {self.floorplan_offset_y_var.get():.3f})'
            )
        elif self.latest_bev_image is not None:
            self.status_var.set(
                f'Live BEV active | rotation={self.floorplan_rotation_var.get():.1f} | '
                f'scale={scale_value:.3f} | '
                f'bev_preview_scale={self.bev_preview_scale_var.get():.3f} | '
                f'offset=({self.floorplan_offset_x_var.get():.3f}, {self.floorplan_offset_y_var.get():.3f})'
            )

    def build_scale_warning_text(self, scale_value: float) -> str:
        if scale_value >= self.CRITICAL_SCALE:
            return (
                'Achtung: contain haelt das rotierte Bild sichtbar. '
                'floorplan_scale >= 1.10 zoomt sehr wahrscheinlich wieder hinein und schneidet Ecken ab. '
                'Fuer saubere Kalibrierung zuerst scale = 1.0000 verwenden und nur mit '
                'offset_x_ratio / offset_y_ratio arbeiten.'
            )
        if scale_value > self.SAFE_SCALE + 1e-6:
            return (
                'Achtung: contain haelt das rotierte Bild sichtbar. '
                'floorplan_scale > 1.0 zoomt wieder hinein und kann Ecken abschneiden. '
                'Fuer saubere Kalibrierung zuerst scale = 1.0000 verwenden und nur mit '
                'offset_x_ratio / offset_y_ratio arbeiten.'
            )
        return (
            'Sicherer Kalibrierzustand aktiv: contain + scale = 1.0000. '
            'Jetzt Rotation und Offsets feinjustieren.'
        )

    def is_scale_risky(self, scale_value: float) -> bool:
        return scale_value > self.SAFE_SCALE + 1e-6

    def log_scale_warning_if_needed(self, prefix: str) -> None:
        scale_value = float(self.floorplan_scale_var.get())
        if scale_value >= self.CRITICAL_SCALE:
            self.get_logger().warning(
                f'{prefix}: floorplan_fit_mode=contain with floorplan_scale={scale_value:.4f} '
                'will very likely crop corners again.'
            )
        elif scale_value > self.SAFE_SCALE + 1e-6:
            self.get_logger().warning(
                f'{prefix}: floorplan_fit_mode=contain with floorplan_scale={scale_value:.4f} '
                'can crop corners. Prefer scale=1.0000 and use offsets first.'
            )

    def bev_image_callback(self, msg: Image) -> None:
        if msg.encoding != 'rgb8' or msg.height <= 0 or msg.width <= 0:
            return
        try:
            bev_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except ValueError:
            return
        self.latest_bev_image = bev_image.copy()
        self.latest_bev_version += 1

    def render_preview(self) -> None:
        if self.latest_bev_image is None:
            self.root.after(int(1000 / self.refresh_rate_hz), self.render_preview)
            return

        params = (
            self.floorplan_rotation_var.get(),
            self.floorplan_scale_var.get(),
            self.floorplan_offset_x_var.get(),
            self.floorplan_offset_y_var.get(),
            self.bev_preview_scale_var.get(),
            self.overlay_alpha_var.get(),
        )
        if self.rendered_preview_version == self.latest_bev_version and self.rendered_preview_params == params:
            self.root.after(int(1000 / self.refresh_rate_hz), self.render_preview)
            return

        if self.floorplan_image is None:
            self.floorplan_image = self.reload_floorplan_image()
            if self.floorplan_image is None:
                self.status_var.set('Floorplan image missing.')
                self.root.after(int(1000 / self.refresh_rate_hz), self.render_preview)
                return

        target_size = (self.preview_width, self.preview_height)
        bev_background = self.render_bev_preview_background(self.latest_bev_image, target_size)
        if abs(self.floorplan_rotation_var.get() - self.initial_floorplan_rotation_deg) > 1e-6:
            self.floorplan_image = self.reload_floorplan_image()
        floorplan_background = render_floorplan_background(
            floorplan_image=self.floorplan_image,
            target_size=target_size,
            scale=self.floorplan_scale_var.get(),
            offset_x_ratio=self.floorplan_offset_x_var.get(),
            offset_y_ratio=self.floorplan_offset_y_var.get(),
            fit_mode=self.SAFE_FIT_MODE,
        )

        alpha = self.overlay_alpha_var.get()
        composite = cv2.addWeighted(bev_background, 1.0 - alpha, floorplan_background, alpha, 0.0)
        success, png_buffer = cv2.imencode('.png', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        if success:
            self.preview_photo_image = tk.PhotoImage(data=base64.b64encode(png_buffer.tobytes()).decode('ascii'))
            self.preview_canvas.delete('all')
            self.preview_canvas.create_image(0, 0, anchor='nw', image=self.preview_photo_image)
            self.preview_canvas.create_text(
                8,
                8,
                anchor='nw',
                text='BEV / Floorplan Overlay',
                fill='#1d2c1f',
                font=('TkDefaultFont', 11, 'bold'),
            )
        if not self.warning_var.get():
            self.status_var.set(
                f'Live BEV active | rotation={self.floorplan_rotation_var.get():.1f} | '
                f'scale={self.floorplan_scale_var.get():.3f} | '
                f'offset=({self.floorplan_offset_x_var.get():.3f}, {self.floorplan_offset_y_var.get():.3f})'
            )
        self.rendered_preview_version = self.latest_bev_version
        self.rendered_preview_params = params
        self.root.after(int(1000 / self.refresh_rate_hz), self.render_preview)

    def render_bev_preview_background(
        self,
        bev_image: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        return render_floorplan_background(
            floorplan_image=bev_image,
            target_size=target_size,
            scale=float(self.bev_preview_scale_var.get()),
            offset_x_ratio=0.0,
            offset_y_ratio=0.0,
            fit_mode=self.SAFE_FIT_MODE,
            background_rgb=(243, 247, 244),
        )

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
    node = FloorplanCalibrationToolNode()
    try:
        node.render_preview()
        node.spin_once()
        node.root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
