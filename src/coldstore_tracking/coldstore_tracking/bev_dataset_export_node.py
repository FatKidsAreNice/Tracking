#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

try:
    from .bev_utils import BevCoordinateRotation, BevGeometry, BevImageBuilder, PointCloudReader
except ImportError:
    from bev_utils import BevCoordinateRotation, BevGeometry, BevImageBuilder, PointCloudReader

try:
    import cv2
except ImportError:
    cv2 = None


class KeyframeDecision:
    def __init__(
        self,
        keyframe_change_ratio: float,
        min_save_interval_sec: float,
        max_save_interval_sec: float,
    ) -> None:
        self.keyframe_change_ratio = float(keyframe_change_ratio)
        self.min_save_interval_sec = float(min_save_interval_sec)
        self.max_save_interval_sec = float(max_save_interval_sec)

        self.last_saved_occupancy: Optional[np.ndarray] = None
        self.last_saved_time_sec: Optional[float] = None

    def should_save(self, occupancy: np.ndarray, now_sec: float) -> tuple[bool, float, str]:
        if self.last_saved_occupancy is None or self.last_saved_time_sec is None:
            return True, 1.0, "first_frame"

        elapsed = now_sec - self.last_saved_time_sec

        if elapsed < self.min_save_interval_sec:
            return False, 0.0, "min_interval"

        current_mask = occupancy > 0
        previous_mask = self.last_saved_occupancy > 0

        if current_mask.shape != previous_mask.shape:
            return True, 1.0, "shape_changed"

        changed_pixels = int(np.count_nonzero(current_mask != previous_mask))
        change_ratio = changed_pixels / float(current_mask.size)

        if change_ratio >= self.keyframe_change_ratio:
            return True, change_ratio, "scene_changed"

        if elapsed >= self.max_save_interval_sec:
            return True, change_ratio, "max_interval_reference"

        return False, change_ratio, "duplicate"

    def mark_saved(self, occupancy: np.ndarray, now_sec: float) -> None:
        self.last_saved_occupancy = occupancy.copy()
        self.last_saved_time_sec = now_sec


class BevDatasetWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser()
        self.images_dir = self.output_root / "raw" / "images"
        self.metadata_dir = self.output_root / "raw" / "metadata"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def write(self, index: int, image: np.ndarray, metadata: dict) -> tuple[Path, Path]:
        timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"frame_{index:06d}_{timestamp_label}"

        image_path = self.images_dir / f"{stem}.png"
        metadata_path = self.metadata_dir / f"{stem}.json"

        if cv2 is None:
            raise RuntimeError("OpenCV ist nicht installiert. Bitte python3-opencv installieren.")

        success = cv2.imwrite(str(image_path), image)

        if not success:
            raise RuntimeError(f"Bild konnte nicht geschrieben werden: {image_path}")

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return image_path, metadata_path


class BevDatasetExportNode(Node):
    def __init__(self) -> None:
        super().__init__("bev_dataset_export_node")

        self.declare_parameter("input_topic", "/rslidar_points")
        self.declare_parameter("expected_frame_id", "rslidar")
        self.declare_parameter("output_root", "~/ros2_ws/bev_dataset")

        self.declare_parameter("roi_min", [-7.5, -7.5, -3.0])
        self.declare_parameter("roi_max", [7.5, 7.5, 3.0])
        self.declare_parameter("resolution_m_per_px", 0.02)
        self.declare_parameter("bev_rotation_clockwise_deg", 0.0)

        self.declare_parameter("sample_interval_sec", 5.0)
        self.declare_parameter("min_save_interval_sec", 30.0)
        self.declare_parameter("max_save_interval_sec", 300.0)
        self.declare_parameter("keyframe_change_ratio", 0.005)

        self.declare_parameter("density_clip_count", 12)
        self.declare_parameter("min_points_to_save", 1000)
        self.declare_parameter("max_saved_frames", 0)
        self.declare_parameter("capture_duration_sec", 0.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.expected_frame_id = str(self.get_parameter("expected_frame_id").value).strip()
        self.output_root = Path(str(self.get_parameter("output_root").value)).expanduser()

        roi_min = [float(value) for value in self.get_parameter("roi_min").value]
        roi_max = [float(value) for value in self.get_parameter("roi_max").value]
        resolution_m_per_px = float(self.get_parameter("resolution_m_per_px").value)
        self.bev_rotation_clockwise_deg = float(
            self.get_parameter("bev_rotation_clockwise_deg").value
        )

        self.sample_interval_sec = float(self.get_parameter("sample_interval_sec").value)
        self.min_points_to_save = int(self.get_parameter("min_points_to_save").value)
        self.max_saved_frames = int(self.get_parameter("max_saved_frames").value)
        self.capture_duration_sec = float(self.get_parameter("capture_duration_sec").value)

        self.geometry = BevGeometry.from_roi(roi_min, roi_max, resolution_m_per_px)
        self.bev_builder = BevImageBuilder(
            geometry=self.geometry,
            density_clip_count=int(self.get_parameter("density_clip_count").value),
        )
        self.bev_rotation = BevCoordinateRotation(
            geometry=self.geometry,
            clockwise_deg=self.bev_rotation_clockwise_deg,
        )
        self.keyframe_decision = KeyframeDecision(
            keyframe_change_ratio=float(self.get_parameter("keyframe_change_ratio").value),
            min_save_interval_sec=float(self.get_parameter("min_save_interval_sec").value),
            max_save_interval_sec=float(self.get_parameter("max_save_interval_sec").value),
        )
        self.writer = BevDatasetWriter(self.output_root)

        self.saved_count = 0
        self.received_count = 0
        self.skipped_wrong_frame_count = 0
        self.last_processed_time_sec: Optional[float] = None
        self.capture_started_time_sec = self.get_clock().now().nanoseconds * 1e-9
        self.capture_started_local = datetime.now().isoformat(timespec="seconds")
        self.capture_duration_timer = None
        if self.capture_duration_sec > 0.0:
            check_period_sec = min(max(self.capture_duration_sec, 1.0), 60.0)
            self.capture_duration_timer = self.create_timer(
                check_period_sec,
                self.check_capture_duration,
            )

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.handle_cloud,
            10,
        )

        self.get_logger().info("bev_dataset_export_node started.")
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Expected frame_id: {self.expected_frame_id}")
        self.get_logger().info(f"Output root: {self.output_root}")
        self.get_logger().info(f"ROI min: {self.geometry.roi_min.tolist()}")
        self.get_logger().info(f"ROI max: {self.geometry.roi_max.tolist()}")
        self.get_logger().info(
            f"BEV image: {self.geometry.width_px}x{self.geometry.height_px} px "
            f"@ {self.geometry.resolution_m_per_px:.3f} m/px"
        )
        self.get_logger().info(
            f"BEV coordinate rotation before ROI crop: "
            f"{self.bev_rotation_clockwise_deg:.1f} deg clockwise"
        )
        if self.capture_duration_sec > 0.0:
            self.get_logger().info(
                f"Capture duration: {self.capture_duration_sec:.0f}s "
                f"({self.capture_duration_sec / 86400.0:.2f} days)"
            )
        else:
            self.get_logger().info("Capture duration: unlimited")

    def check_capture_duration(self) -> None:
        elapsed_sec = self.get_clock().now().nanoseconds * 1e-9 - self.capture_started_time_sec
        if elapsed_sec < self.capture_duration_sec:
            return

        self.get_logger().info(
            f"Capture duration reached after {elapsed_sec:.1f}s with "
            f"{self.saved_count} saved frames. Shutting down exporter."
        )
        rclpy.shutdown()

    def handle_cloud(self, msg: PointCloud2) -> None:
        self.received_count += 1

        if self.expected_frame_id and msg.header.frame_id != self.expected_frame_id:
            self.skipped_wrong_frame_count += 1

            if self.skipped_wrong_frame_count <= 5:
                self.get_logger().warn(
                    f"Skipping cloud with frame_id={msg.header.frame_id!r}; "
                    f"expected {self.expected_frame_id!r}."
                )

            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.last_processed_time_sec is not None:
            if now_sec - self.last_processed_time_sec < self.sample_interval_sec:
                return

        self.last_processed_time_sec = now_sec

        points_xyz = PointCloudReader.to_xyz_array(msg)
        bev_points_xyz = self.bev_rotation.forward_points_xyz(points_xyz)
        image, stats = self.bev_builder.build(bev_points_xyz)

        if stats["points_in_roi"] < self.min_points_to_save:
            self.get_logger().warn(
                f"Skipping frame: only {stats['points_in_roi']} ROI points "
                f"(min_points_to_save={self.min_points_to_save})."
            )
            return

        occupancy = image[:, :, 0]
        should_save, change_ratio, reason = self.keyframe_decision.should_save(occupancy, now_sec)

        if not should_save:
            self.get_logger().info(
                f"Skipped BEV frame: reason={reason}, change_ratio={change_ratio:.5f}, "
                f"points_in_roi={stats['points_in_roi']}"
            )
            return

        self.saved_count += 1

        metadata = self.build_metadata(
            msg=msg,
            stats=stats,
            change_ratio=change_ratio,
            save_reason=reason,
        )

        image_path, metadata_path = self.writer.write(self.saved_count, image, metadata)
        self.keyframe_decision.mark_saved(occupancy, now_sec)

        self.get_logger().info(
            f"Saved BEV frame #{self.saved_count}: {image_path.name}, "
            f"reason={reason}, change_ratio={change_ratio:.5f}, "
            f"points_in_roi={stats['points_in_roi']}, occupied_pixels={stats['occupied_pixels']}"
        )

        if self.max_saved_frames > 0 and self.saved_count >= self.max_saved_frames:
            self.get_logger().info(f"max_saved_frames={self.max_saved_frames} reached. Shutting down.")
            rclpy.shutdown()

    def build_metadata(
        self,
        msg: PointCloud2,
        stats: dict,
        change_ratio: float,
        save_reason: str,
    ) -> dict:
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        return {
            "source_topic": self.input_topic,
            "frame_id": msg.header.frame_id,
            "stamp_sec": stamp_sec,
            "saved_at_local": datetime.now().isoformat(timespec="milliseconds"),
            "capture_started_local": self.capture_started_local,
            "capture_elapsed_sec": (
                self.get_clock().now().nanoseconds * 1e-9 - self.capture_started_time_sec
            ),
            "save_reason": save_reason,
            "change_ratio": change_ratio,
            "roi_min": self.geometry.roi_min.tolist(),
            "roi_max": self.geometry.roi_max.tolist(),
            "resolution_m_per_px": self.geometry.resolution_m_per_px,
            "image_width_px": self.geometry.width_px,
            "image_height_px": self.geometry.height_px,
            "bev_rotation_clockwise_deg": self.bev_rotation_clockwise_deg,
            "world_xy_to_bev_xy_matrix": self.bev_rotation.forward_matrix.tolist(),
            "bev_xy_to_world_xy_matrix": self.bev_rotation.inverse_matrix.tolist(),
            "channels": {
                "0": "occupancy",
                "1": "max_height_normalized",
                "2": "density_log_normalized",
            },
            "world_to_pixel": {
                "bev_xy": "world_xy_to_bev_xy_matrix * [world_x, world_y, 1]",
                "u": "floor((bev_x - roi_min_x) / resolution_m_per_px)",
                "v": "floor((roi_max_y - bev_y) / resolution_m_per_px)",
            },
            "pixel_to_world": {
                "bev_x": "roi_min_x + (u + 0.5) * resolution_m_per_px",
                "bev_y": "roi_max_y - (v + 0.5) * resolution_m_per_px",
                "world_xy": "bev_xy_to_world_xy_matrix * [bev_x, bev_y, 1]",
            },
            "stats": stats,
        }


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)

    node = BevDatasetExportNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
