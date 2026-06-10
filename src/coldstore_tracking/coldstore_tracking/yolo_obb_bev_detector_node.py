#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from .bev_utils import BevGeometry, BevImageBuilder, PointCloudReader
except ImportError:
    from bev_utils import BevGeometry, BevImageBuilder, PointCloudReader

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None

@dataclass(frozen=True)
class TileWindow:
    x_min: int
    y_min: int
    size: int

    @property
    def x_max(self) -> int:
        return self.x_min + self.size

    @property
    def y_max(self) -> int:
        return self.y_min + self.size


@dataclass(frozen=True)
class DetectionCandidate:
    class_id: int
    class_name: str
    confidence: float
    pixel_corners: np.ndarray
    source_tile: TileWindow


@dataclass(frozen=True)
class RackDetection:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    center_z: float
    yaw: float
    length: float
    width: float
    height: float
    corners_xy: list[list[float]]
    track_state: str = "raw"
    hit_count: int = 1
    missed_count: int = 0


@dataclass
class RackTrack:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    center_z: float
    yaw: float
    length: float
    width: float
    height: float
    corners_xy: list[list[float]]
    first_seen_sec: float
    last_seen_sec: float
    last_update_sec: float
    hit_count: int
    missed_count: int
    state: str
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def to_detection(self) -> RackDetection:
        return RackDetection(
            detection_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            center_x=self.center_x,
            center_y=self.center_y,
            center_z=self.center_z,
            yaw=self.yaw,
            length=self.length,
            width=self.width,
            height=self.height,
            corners_xy=self.corners_xy,
            track_state=self.state,
            hit_count=self.hit_count,
            missed_count=self.missed_count,
        )


class TrackStabilizer:
    STATE_TENTATIVE = "tentative"
    STATE_CONFIRMED = "confirmed"
    STATE_LOST = "lost"
    STATE_DELETED = "deleted"

    def __init__(
        self,
        match_distance_m: float,
        yaw_gate_deg: float,
        min_iou: float,
        max_match_cost: float,
        confirm_hits: int,
        instant_confirm_confidence: float,
        max_missed_sec: float,
        tentative_timeout_sec: float,
        position_alpha: float,
        yaw_alpha: float,
        confidence_alpha: float,
        moving_match_distance_m: float,
        lost_match_distance_m: float,
        moving_yaw_gate_deg: float,
        lost_yaw_gate_deg: float,
        reid_max_distance_m: float,
        reid_max_size_ratio_delta: float,
        reid_max_yaw_diff_deg: float,
        moving_speed_threshold_mps: float,
    ) -> None:
        self.match_distance_m = float(match_distance_m)
        self.yaw_gate_rad = math.radians(float(yaw_gate_deg))
        self.min_iou = float(min_iou)
        self.max_match_cost = float(max_match_cost)
        self.confirm_hits = max(int(confirm_hits), 1)
        self.instant_confirm_confidence = float(instant_confirm_confidence)
        self.max_missed_sec = float(max_missed_sec)
        self.tentative_timeout_sec = float(tentative_timeout_sec)
        self.position_alpha = float(np.clip(position_alpha, 0.0, 1.0))
        self.yaw_alpha = float(np.clip(yaw_alpha, 0.0, 1.0))
        self.confidence_alpha = float(np.clip(confidence_alpha, 0.0, 1.0))
        self.moving_match_distance_m = max(float(moving_match_distance_m), self.match_distance_m)
        self.lost_match_distance_m = max(float(lost_match_distance_m), self.moving_match_distance_m)
        self.moving_yaw_gate_rad = math.radians(float(moving_yaw_gate_deg))
        self.lost_yaw_gate_rad = math.radians(float(lost_yaw_gate_deg))
        self.reid_max_distance_m = float(reid_max_distance_m)
        self.reid_max_size_ratio_delta = float(reid_max_size_ratio_delta)
        self.reid_max_yaw_gate_rad = math.radians(float(reid_max_yaw_diff_deg))
        self.moving_speed_threshold_mps = float(moving_speed_threshold_mps)

        self.tracks: list[RackTrack] = []
        self.recently_deleted_tracks: list[RackTrack] = []
        self.next_track_id = 1

    def update(self, detections: list[RackDetection], now_sec: float) -> list[RackDetection]:
        active_track_indices = [
            index for index, track in enumerate(self.tracks)
            if track.state != self.STATE_DELETED
        ]

        matches = self.match_tracks_to_detections(active_track_indices, detections)

        matched_track_indices = {track_index for track_index, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}

        for track_index, detection_index in matches:
            self.update_matched_track(
                track=self.tracks[track_index],
                detection=detections[detection_index],
                now_sec=now_sec,
            )

        for track_index in active_track_indices:
            if track_index not in matched_track_indices:
                self.update_unmatched_track(self.tracks[track_index], now_sec)

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detection_indices:
                if self.try_recover_lost_track(detection, now_sec):
                    continue
                if self.try_reidentify_track(detection, now_sec):
                    continue
                self.create_track(detection, now_sec)

        self.suppress_lost_tracks_shadowed_by_confirmed_tracks()
        self.remove_deleted_tracks(now_sec)
        return self.get_publishable_detections()

    def match_tracks_to_detections(
        self,
        active_track_indices: list[int],
        detections: list[RackDetection],
    ) -> list[tuple[int, int]]:
        pair_costs: list[tuple[float, int, int]] = []

        for track_index in active_track_indices:
            track = self.tracks[track_index]

            for detection_index, detection in enumerate(detections):
                cost = self.compute_match_cost(track, detection)

                if cost is not None and cost <= self.max_match_cost:
                    pair_costs.append((cost, track_index, detection_index))

        pair_costs.sort(key=lambda value: value[0])

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for _, track_index, detection_index in pair_costs:
            if track_index in used_tracks or detection_index in used_detections:
                continue

            matches.append((track_index, detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        return matches

    def compute_match_cost(self, track: RackTrack, detection: RackDetection) -> Optional[float]:
        predicted_center_x, predicted_center_y = self.predict_track_xy(track)
        distance = math.hypot(
            detection.center_x - predicted_center_x,
            detection.center_y - predicted_center_y,
        )

        yaw_difference = abs(self.normalize_angle(detection.yaw - track.yaw))
        iou = self.oriented_iou_from_corners(track.corners_xy, detection.corners_xy)
        match_distance_gate = self.get_match_distance_gate(track)
        yaw_gate_rad = self.get_yaw_gate_rad(track)

        distance_gate_ok = distance <= match_distance_gate
        yaw_gate_ok = yaw_difference <= yaw_gate_rad
        iou_gate_ok = iou >= self.min_iou

        if not ((distance_gate_ok and yaw_gate_ok) or iou_gate_ok):
            return None

        size_cost = self.compute_size_cost(track, detection)
        distance_cost = distance / max(match_distance_gate, 1e-6)
        yaw_cost = yaw_difference / max(yaw_gate_rad, 1e-6)
        iou_cost = 1.0 - iou
        class_penalty = 0.0 if detection.class_id == track.class_id else 0.65

        return distance_cost + 0.25 * yaw_cost + 0.50 * iou_cost + 0.20 * size_cost + class_penalty

    def update_matched_track(self, track: RackTrack, detection: RackDetection, now_sec: float) -> None:
        previous_center_x = track.center_x
        previous_center_y = track.center_y
        previous_update_sec = track.last_update_sec
        track.center_x = self.ema(track.center_x, detection.center_x, self.position_alpha)
        track.center_y = self.ema(track.center_y, detection.center_y, self.position_alpha)
        track.center_z = self.ema(track.center_z, detection.center_z, self.position_alpha)
        track.yaw = self.smooth_angle(track.yaw, detection.yaw, self.yaw_alpha)
        track.confidence = self.ema(track.confidence, detection.confidence, self.confidence_alpha)

        if self.should_switch_class(track, detection):
            track.class_id = detection.class_id
            track.class_name = detection.class_name

        track.length = detection.length
        track.width = detection.width
        track.height = detection.height
        track.corners_xy = self.smoothed_corners(track, detection)

        track.hit_count += 1
        track.missed_count = 0
        track.last_seen_sec = now_sec
        track.last_update_sec = now_sec
        dt = max(now_sec - previous_update_sec, 1e-3)
        measured_velocity_x = (track.center_x - previous_center_x) / dt
        measured_velocity_y = (track.center_y - previous_center_y) / dt
        track.velocity_x = self.ema(track.velocity_x, measured_velocity_x, self.position_alpha)
        track.velocity_y = self.ema(track.velocity_y, measured_velocity_y, self.position_alpha)

        if track.hit_count >= self.confirm_hits or detection.confidence >= self.instant_confirm_confidence:
            track.state = self.STATE_CONFIRMED
        elif track.state == self.STATE_LOST:
            track.state = self.STATE_CONFIRMED

    def update_unmatched_track(self, track: RackTrack, now_sec: float) -> None:
        track.missed_count += 1
        track.last_update_sec = now_sec

        seconds_since_seen = now_sec - track.last_seen_sec
        age_sec = now_sec - track.first_seen_sec

        if track.state == self.STATE_TENTATIVE:
            if age_sec > self.tentative_timeout_sec or track.missed_count > 1:
                track.state = self.STATE_DELETED
            return

        if seconds_since_seen > self.max_missed_sec:
            track.state = self.STATE_DELETED
            return

        track.state = self.STATE_LOST

    def create_track(self, detection: RackDetection, now_sec: float) -> None:
        state = self.STATE_TENTATIVE

        if self.confirm_hits <= 1 or detection.confidence >= self.instant_confirm_confidence:
            state = self.STATE_CONFIRMED

        track = RackTrack(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            center_x=detection.center_x,
            center_y=detection.center_y,
            center_z=detection.center_z,
            yaw=detection.yaw,
            length=detection.length,
            width=detection.width,
            height=detection.height,
            corners_xy=detection.corners_xy,
            first_seen_sec=now_sec,
            last_seen_sec=now_sec,
            last_update_sec=now_sec,
            hit_count=1,
            missed_count=0,
            state=state,
            velocity_x=0.0,
            velocity_y=0.0,
        )

        self.next_track_id += 1
        self.tracks.append(track)

    def remove_deleted_tracks(self, now_sec: float) -> None:
        active_tracks: list[RackTrack] = []
        for track in self.tracks:
            if track.state == self.STATE_DELETED:
                self.recently_deleted_tracks.append(track)
            else:
                active_tracks.append(track)
        self.tracks = active_tracks
        self.prune_recently_deleted_tracks(now_sec)

    def get_publishable_detections(self) -> list[RackDetection]:
        publishable_tracks = [
            track for track in self.tracks
            if track.state in (self.STATE_CONFIRMED, self.STATE_LOST)
        ]

        publishable_tracks.sort(key=lambda track: track.track_id)
        return [track.to_detection() for track in publishable_tracks]

    def get_match_distance_gate(self, track: RackTrack) -> float:
        if track.state == self.STATE_LOST:
            return self.lost_match_distance_m
        if self.track_speed(track) >= self.moving_speed_threshold_mps:
            return self.moving_match_distance_m
        return self.match_distance_m

    def get_yaw_gate_rad(self, track: RackTrack) -> float:
        if track.state == self.STATE_LOST:
            return self.lost_yaw_gate_rad
        if self.track_speed(track) >= self.moving_speed_threshold_mps:
            return self.moving_yaw_gate_rad
        return self.yaw_gate_rad

    def predict_track_xy(self, track: RackTrack) -> tuple[float, float]:
        dt = max(track.last_update_sec - track.last_seen_sec, 0.0)
        if track.state != self.STATE_LOST or dt <= 1e-6:
            return track.center_x, track.center_y
        return (
            track.center_x + track.velocity_x * dt,
            track.center_y + track.velocity_y * dt,
        )

    def track_speed(self, track: RackTrack) -> float:
        return math.hypot(track.velocity_x, track.velocity_y)

    def compute_size_cost(self, track: RackTrack, detection: RackDetection) -> float:
        ratios = []
        for track_size, detection_size in (
            (track.length, detection.length),
            (track.width, detection.width),
            (track.height, detection.height),
        ):
            denominator = max(track_size, detection_size, 1e-6)
            ratios.append(abs(track_size - detection_size) / denominator)
        return float(np.mean(ratios))

    def try_reidentify_track(self, detection: RackDetection, now_sec: float) -> bool:
        best_track: RackTrack | None = None
        best_cost: float | None = None

        for track in self.recently_deleted_tracks:
            reid_cost = self.compute_reid_cost(track, detection, now_sec)
            if reid_cost is None:
                continue
            if best_cost is None or reid_cost < best_cost:
                best_cost = reid_cost
                best_track = track

        if best_track is None:
            return False

        self.recently_deleted_tracks = [
            track for track in self.recently_deleted_tracks
            if track.track_id != best_track.track_id
        ]
        best_track.state = self.STATE_LOST
        best_track.last_update_sec = now_sec
        self.tracks.append(best_track)
        self.update_matched_track(best_track, detection, now_sec)
        return True

    def try_recover_lost_track(self, detection: RackDetection, now_sec: float) -> bool:
        best_track: RackTrack | None = None
        best_cost: float | None = None

        for track in self.tracks:
            if track.state != self.STATE_LOST:
                continue

            recovery_cost = self.compute_lost_recovery_cost(track, detection, now_sec)
            if recovery_cost is None:
                continue

            if best_cost is None or recovery_cost < best_cost:
                best_cost = recovery_cost
                best_track = track

        if best_track is None:
            return False

        self.update_matched_track(best_track, detection, now_sec)
        return True

    def compute_lost_recovery_cost(
        self,
        track: RackTrack,
        detection: RackDetection,
        now_sec: float,
    ) -> Optional[float]:
        if detection.class_id != track.class_id:
            return None

        predicted_center_x, predicted_center_y = self.predict_recent_track_xy(track, now_sec)
        distance = math.hypot(
            detection.center_x - predicted_center_x,
            detection.center_y - predicted_center_y,
        )
        distance_gate = max(self.lost_match_distance_m, self.reid_max_distance_m)
        if distance > distance_gate:
            return None

        yaw_difference = abs(self.normalize_angle(detection.yaw - track.yaw))
        yaw_gate = max(self.lost_yaw_gate_rad, self.reid_max_yaw_gate_rad)
        if yaw_difference > yaw_gate:
            return None

        size_cost = self.compute_size_cost(track, detection)
        if size_cost > self.reid_max_size_ratio_delta:
            return None

        iou = self.oriented_iou_from_corners(track.corners_xy, detection.corners_xy)
        return (
            distance / max(distance_gate, 1e-6)
            + 0.30 * (yaw_difference / max(yaw_gate, 1e-6))
            + 0.25 * size_cost / max(self.reid_max_size_ratio_delta, 1e-6)
            + 0.20 * (1.0 - iou)
        )

    def compute_reid_cost(self, track: RackTrack, detection: RackDetection, now_sec: float) -> Optional[float]:
        if detection.class_id != track.class_id:
            return None

        predicted_center_x, predicted_center_y = self.predict_recent_track_xy(track, now_sec)
        distance = math.hypot(
            detection.center_x - predicted_center_x,
            detection.center_y - predicted_center_y,
        )
        if distance > self.reid_max_distance_m:
            return None

        yaw_difference = abs(self.normalize_angle(detection.yaw - track.yaw))
        if yaw_difference > self.reid_max_yaw_gate_rad:
            return None

        size_cost = self.compute_size_cost(track, detection)
        if size_cost > self.reid_max_size_ratio_delta:
            return None

        iou = self.oriented_iou_from_corners(track.corners_xy, detection.corners_xy)
        return (
            distance / max(self.reid_max_distance_m, 1e-6)
            + 0.35 * (yaw_difference / max(self.reid_max_yaw_gate_rad, 1e-6))
            + 0.35 * size_cost / max(self.reid_max_size_ratio_delta, 1e-6)
            + 0.25 * (1.0 - iou)
        )

    def predict_recent_track_xy(self, track: RackTrack, now_sec: float) -> tuple[float, float]:
        dt = max(now_sec - track.last_update_sec, 0.0)
        return (
            track.center_x + track.velocity_x * dt,
            track.center_y + track.velocity_y * dt,
        )

    def prune_recently_deleted_tracks(self, now_sec: float) -> None:
        self.recently_deleted_tracks = [
            track for track in self.recently_deleted_tracks
            if (now_sec - track.last_update_sec) <= self.max_missed_sec * 2.0
        ]

    def suppress_lost_tracks_shadowed_by_confirmed_tracks(self) -> None:
        confirmed_tracks = [
            track for track in self.tracks
            if track.state == self.STATE_CONFIRMED
        ]

        if not confirmed_tracks:
            return

        filtered_tracks: list[RackTrack] = []
        for track in self.tracks:
            if track.state != self.STATE_LOST:
                filtered_tracks.append(track)
                continue

            if any(
                self.lost_track_is_shadowed_by_confirmed_track(track, confirmed_track)
                for confirmed_track in confirmed_tracks
            ):
                continue

            filtered_tracks.append(track)

        self.tracks = filtered_tracks

    def lost_track_is_shadowed_by_confirmed_track(
        self,
        lost_track: RackTrack,
        confirmed_track: RackTrack,
    ) -> bool:
        if lost_track.class_id != confirmed_track.class_id:
            return False

        predicted_center_x, predicted_center_y = self.predict_track_xy(lost_track)
        center_distance = math.hypot(
            confirmed_track.center_x - predicted_center_x,
            confirmed_track.center_y - predicted_center_y,
        )
        duplicate_distance_gate = min(
            self.lost_match_distance_m,
            max(self.match_distance_m, 0.60),
        )
        if center_distance > duplicate_distance_gate:
            return False

        yaw_difference = abs(self.normalize_angle(confirmed_track.yaw - lost_track.yaw))
        if yaw_difference > max(self.yaw_gate_rad, self.reid_max_yaw_gate_rad):
            return False

        confirmed_detection = confirmed_track.to_detection()
        size_cost = self.compute_size_cost(lost_track, confirmed_detection)
        if size_cost > self.reid_max_size_ratio_delta:
            return False

        iou = self.oriented_iou_from_corners(lost_track.corners_xy, confirmed_track.corners_xy)
        max_horizontal_extent = max(
            lost_track.length,
            lost_track.width,
            confirmed_track.length,
            confirmed_track.width,
            1e-6,
        )
        return iou >= max(self.min_iou * 0.5, 0.10) or center_distance <= 0.35 * max_horizontal_extent

    def should_switch_class(self, track: RackTrack, detection: RackDetection) -> bool:
        if detection.class_id == track.class_id:
            return False

        if track.state != self.STATE_CONFIRMED:
            return True

        return detection.confidence >= max(0.45, track.confidence + 0.10)

    def smoothed_corners(self, track: RackTrack, detection: RackDetection) -> list[list[float]]:
        track_corners = np.asarray(track.corners_xy, dtype=np.float32).reshape(4, 2)
        detection_corners = np.asarray(detection.corners_xy, dtype=np.float32).reshape(4, 2)

        if track_corners.shape != detection_corners.shape:
            return detection.corners_xy

        smoothed = (1.0 - self.position_alpha) * track_corners + self.position_alpha * detection_corners
        return smoothed.astype(float).tolist()

    def oriented_iou_from_corners(self, corners_a: list[list[float]], corners_b: list[list[float]]) -> float:
        points_a = np.asarray(corners_a, dtype=np.float32).reshape(4, 2)
        points_b = np.asarray(corners_b, dtype=np.float32).reshape(4, 2)

        area_a = abs(float(cv2.contourArea(points_a)))
        area_b = abs(float(cv2.contourArea(points_b)))

        if area_a <= 1e-6 or area_b <= 1e-6:
            return 0.0

        rect_a = cv2.minAreaRect(points_a)
        rect_b = cv2.minAreaRect(points_b)

        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        if intersection_type == cv2.INTERSECT_NONE or intersection_points is None:
            return 0.0

        intersection_area = abs(float(cv2.contourArea(intersection_points.astype(np.float32))))
        union_area = area_a + area_b - intersection_area

        if union_area <= 1e-6:
            return 0.0

        return intersection_area / union_area

    def ema(self, old_value: float, new_value: float, alpha: float) -> float:
        return (1.0 - alpha) * old_value + alpha * new_value

    def smooth_angle(self, old_angle: float, new_angle: float, alpha: float) -> float:
        difference = self.normalize_angle(new_angle - old_angle)
        return self.normalize_angle(old_angle + alpha * difference)

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


class TilePlanner:
    def __init__(self, tile_size_px: int, overlap_ratio: float) -> None:
        self.tile_size_px = int(tile_size_px)
        self.overlap_ratio = float(overlap_ratio)

        if self.tile_size_px <= 0:
            raise ValueError("tile_size_px muss größer 0 sein.")

        if self.overlap_ratio < 0.0 or self.overlap_ratio >= 1.0:
            raise ValueError("tile_overlap_ratio muss im Bereich [0.0, 1.0) liegen.")

    def create_windows(self, image_width: int, image_height: int) -> list[TileWindow]:
        step = max(int(round(self.tile_size_px * (1.0 - self.overlap_ratio))), 1)

        x_starts = self.compute_starts(image_width, self.tile_size_px, step)
        y_starts = self.compute_starts(image_height, self.tile_size_px, step)

        return [
            TileWindow(x_min=x_start, y_min=y_start, size=self.tile_size_px)
            for y_start in y_starts
            for x_start in x_starts
        ]

    def compute_starts(self, image_size: int, tile_size: int, step: int) -> list[int]:
        if image_size <= tile_size:
            return [0]

        starts = list(range(0, image_size - tile_size + 1, step))
        final_start = image_size - tile_size

        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        return starts


class ClassThresholdFilter:
    def __init__(
        self,
        confidence_threshold: float,
        side_confidence_threshold: float,
        top_confidence_threshold: float,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.side_confidence_threshold = float(side_confidence_threshold)
        self.top_confidence_threshold = float(top_confidence_threshold)

    def keep(self, candidate: DetectionCandidate) -> bool:
        if candidate.confidence < self.confidence_threshold:
            return False

        if candidate.class_id == 0:
            return candidate.confidence >= self.side_confidence_threshold

        if candidate.class_id == 1:
            return candidate.confidence >= self.top_confidence_threshold

        return False


class OrientedBoxNms:
    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = float(iou_threshold)

    def apply(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        final_candidates: list[DetectionCandidate] = []

        for class_id in sorted(set(candidate.class_id for candidate in candidates)):
            class_candidates = [candidate for candidate in candidates if candidate.class_id == class_id]
            final_candidates.extend(self.apply_single_class(class_candidates))

        return sorted(final_candidates, key=lambda candidate: candidate.confidence, reverse=True)

    def apply_single_class(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        sorted_candidates = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
        kept: list[DetectionCandidate] = []

        while sorted_candidates:
            current = sorted_candidates.pop(0)
            kept.append(current)

            remaining: list[DetectionCandidate] = []

            for candidate in sorted_candidates:
                iou = self.oriented_iou(current.pixel_corners, candidate.pixel_corners)

                if iou < self.iou_threshold:
                    remaining.append(candidate)

            sorted_candidates = remaining

        return kept

    def oriented_iou(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        area_a = abs(float(cv2.contourArea(points_a.astype(np.float32))))
        area_b = abs(float(cv2.contourArea(points_b.astype(np.float32))))

        if area_a <= 1e-6 or area_b <= 1e-6:
            return 0.0

        rect_a = cv2.minAreaRect(points_a.astype(np.float32))
        rect_b = cv2.minAreaRect(points_b.astype(np.float32))

        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        if intersection_type == cv2.INTERSECT_NONE or intersection_points is None:
            return 0.0

        intersection_area = abs(float(cv2.contourArea(intersection_points.astype(np.float32))))
        union_area = area_a + area_b - intersection_area

        if union_area <= 1e-6:
            return 0.0

        return intersection_area / union_area


class PhysicalRackDeduplicator:
    def __init__(
        self,
        enabled: bool,
        geometry: BevGeometry,
        merge_distance_m: float,
        merge_iou_threshold: float,
        prefer_side_visible: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.geometry = geometry
        self.merge_distance_m = float(merge_distance_m)
        self.merge_iou_threshold = float(merge_iou_threshold)
        self.prefer_side_visible = bool(prefer_side_visible)

    def apply(self, candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        if not self.enabled:
            return candidates

        sorted_candidates = sorted(
            candidates,
            key=self.sort_key,
        )

        kept: list[DetectionCandidate] = []

        for candidate in sorted_candidates:
            if self.is_duplicate_of_kept(candidate, kept):
                continue

            kept.append(candidate)

        return sorted(kept, key=lambda candidate: candidate.confidence, reverse=True)

    def sort_key(self, candidate: DetectionCandidate) -> tuple[int, float]:
        if self.prefer_side_visible:
            class_priority = 0 if candidate.class_id == 0 else 1
        else:
            class_priority = 0

        return (class_priority, -candidate.confidence)

    def is_duplicate_of_kept(
        self,
        candidate: DetectionCandidate,
        kept_candidates: list[DetectionCandidate],
    ) -> bool:
        for kept in kept_candidates:
            if self.is_same_physical_rack(candidate, kept):
                return True

        return False

    def is_same_physical_rack(
        self,
        candidate_a: DetectionCandidate,
        candidate_b: DetectionCandidate,
    ) -> bool:
        center_a = self.center(candidate_a.pixel_corners)
        center_b = self.center(candidate_b.pixel_corners)

        distance_px = float(np.linalg.norm(center_a - center_b))
        distance_m = distance_px * self.geometry.resolution_m_per_px

        if distance_m <= self.merge_distance_m:
            return True

        iou = self.oriented_iou(candidate_a.pixel_corners, candidate_b.pixel_corners)
        return iou >= self.merge_iou_threshold

    def center(self, pixel_corners: np.ndarray) -> np.ndarray:
        return np.mean(np.asarray(pixel_corners, dtype=np.float32).reshape(4, 2), axis=0)

    def oriented_iou(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        points_a = np.asarray(points_a, dtype=np.float32).reshape(4, 2)
        points_b = np.asarray(points_b, dtype=np.float32).reshape(4, 2)

        area_a = abs(float(cv2.contourArea(points_a)))
        area_b = abs(float(cv2.contourArea(points_b)))

        if area_a <= 1e-6 or area_b <= 1e-6:
            return 0.0

        rect_a = cv2.minAreaRect(points_a)
        rect_b = cv2.minAreaRect(points_b)

        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        if intersection_type == cv2.INTERSECT_NONE or intersection_points is None:
            return 0.0

        intersection_area = abs(float(cv2.contourArea(intersection_points.astype(np.float32))))
        union_area = area_a + area_b - intersection_area

        if union_area <= 1e-6:
            return 0.0

        return intersection_area / union_area


class RackDetectionConverter:
    CLASS_NAMES = {
        0: "rack_side_visible",
        1: "rack_top_visible",
    }

    def __init__(
        self,
        geometry: BevGeometry,
        marker_size: list[float],
        rack_floor_z: float,
        marker_yaw_mode: str,
        marker_fixed_yaw_rad: float,
        marker_yaw_snap_step_deg: float,
        marker_yaw_snap_offset_deg: float,
    ) -> None:
        self.geometry = geometry
        self.marker_length = float(marker_size[0])
        self.marker_width = float(marker_size[1])
        self.marker_height = float(marker_size[2])
        self.rack_floor_z = float(rack_floor_z)
        self.marker_yaw_mode = str(marker_yaw_mode).strip().lower()
        self.marker_fixed_yaw_rad = float(marker_fixed_yaw_rad)
        self.marker_yaw_snap_step_rad = math.radians(float(marker_yaw_snap_step_deg))
        self.marker_yaw_snap_offset_rad = math.radians(float(marker_yaw_snap_offset_deg))

    def convert(
        self,
        detection_id: int,
        candidate: DetectionCandidate,
    ) -> RackDetection:
        pixel_corners = np.asarray(candidate.pixel_corners, dtype=np.float32).reshape(4, 2)
        world_corners = self.geometry.pixel_to_world_xy(pixel_corners)

        center_xy = np.mean(world_corners, axis=0)
        yaw = self.normalize_marker_yaw(self.compute_yaw(world_corners))

        return RackDetection(
            detection_id=detection_id,
            class_id=candidate.class_id,
            class_name=candidate.class_name,
            confidence=float(candidate.confidence),
            center_x=float(center_xy[0]),
            center_y=float(center_xy[1]),
            center_z=float(self.rack_floor_z + self.marker_height * 0.5),
            yaw=float(yaw),
            length=self.marker_length,
            width=self.marker_width,
            height=self.marker_height,
            corners_xy=world_corners.astype(float).tolist(),
            track_state="raw",
            hit_count=1,
            missed_count=0,
        )

    def compute_yaw(self, world_corners: np.ndarray) -> float:
        if world_corners.shape != (4, 2):
            return 0.0

        edge_01 = world_corners[1] - world_corners[0]
        edge_12 = world_corners[2] - world_corners[1]

        len_01 = float(np.linalg.norm(edge_01))
        len_12 = float(np.linalg.norm(edge_12))

        edge = edge_01 if len_01 >= len_12 else edge_12

        if float(np.linalg.norm(edge)) < 1e-6:
            return 0.0

        return math.atan2(float(edge[1]), float(edge[0]))

    def normalize_marker_yaw(self, yaw: float) -> float:
        if self.marker_yaw_mode == "fixed":
            return self.normalize_angle(self.marker_fixed_yaw_rad)

        if self.marker_yaw_mode == "snap":
            return self.snap_yaw(yaw)

        return self.normalize_angle(yaw)

    def snap_yaw(self, yaw: float) -> float:
        if self.marker_yaw_snap_step_rad <= 1e-6:
            return self.normalize_angle(yaw)

        shifted = yaw - self.marker_yaw_snap_offset_rad
        snapped = round(shifted / self.marker_yaw_snap_step_rad) * self.marker_yaw_snap_step_rad
        return self.normalize_angle(snapped + self.marker_yaw_snap_offset_rad)

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


class MarkerFactory:
    def __init__(
        self,
        output_frame: str,
        marker_alpha: float,
        text_marker_enabled: bool,
        text_marker_include_id: bool,
        marker_lifetime_sec: float,
    ) -> None:
        self.output_frame = output_frame
        self.marker_alpha = float(marker_alpha)
        self.text_marker_enabled = bool(text_marker_enabled)
        self.text_marker_include_id = bool(text_marker_include_id)
        self.marker_lifetime_sec = max(float(marker_lifetime_sec), 0.0)

    def build_marker_array(self, detections: list[RackDetection], stamp) -> MarkerArray:
        marker_array = MarkerArray()
        marker_array.markers.extend(self.build_delete_all_markers(stamp))

        for detection in detections:
            marker_array.markers.append(self.build_cube_marker(detection, stamp))

            if self.should_publish_text_marker(detection):
                marker_array.markers.append(self.build_text_marker(detection, stamp))

        return marker_array

    def build_delete_all_markers(self, stamp) -> list[Marker]:
        return [
            self.build_delete_all_marker("rack_yolo_obb", stamp),
            self.build_delete_all_marker("rack_yolo_obb_text", stamp),
        ]

    def build_delete_all_marker(self, namespace: str, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.output_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = 0
        marker.action = Marker.DELETEALL
        return marker

    def should_publish_text_marker(self, detection: RackDetection) -> bool:
        return self.text_marker_enabled and detection.track_state != "lost"

    def apply_lifetime(self, marker: Marker) -> Marker:
        lifetime_sec = int(self.marker_lifetime_sec)
        lifetime_nanosec = int(round((self.marker_lifetime_sec - lifetime_sec) * 1e9))

        if lifetime_nanosec >= 1_000_000_000:
            lifetime_sec += 1
            lifetime_nanosec -= 1_000_000_000

        marker.lifetime = Duration(sec=lifetime_sec, nanosec=lifetime_nanosec)
        return marker

    def build_cube_marker(self, detection: RackDetection, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.output_frame
        marker.header.stamp = stamp
        marker.ns = "rack_yolo_obb"
        marker.id = detection.detection_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = detection.center_x
        marker.pose.position.y = detection.center_y
        marker.pose.position.z = detection.center_z
        marker.pose.orientation.z = math.sin(detection.yaw * 0.5)
        marker.pose.orientation.w = math.cos(detection.yaw * 0.5)

        marker.scale.x = detection.length
        marker.scale.y = detection.width
        marker.scale.z = detection.height

        if detection.class_id == 0:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.0

        if detection.track_state == "lost":
            marker.color.a = self.marker_alpha * 0.45
        else:
            marker.color.a = self.marker_alpha

        return self.apply_lifetime(marker)

    def build_text_marker(self, detection: RackDetection, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.output_frame
        marker.header.stamp = stamp
        marker.ns = "rack_yolo_obb_text"
        marker.id = 10000 + detection.detection_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = detection.center_x
        marker.pose.position.y = detection.center_y
        marker.pose.position.z = detection.center_z + detection.height * 0.6
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.35
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        text_parts = []
        if self.text_marker_include_id:
            text_parts.append(f"id {detection.detection_id}")
        text_parts.extend(
            [
                detection.class_name,
                detection.track_state,
                f"{detection.confidence:.2f}",
            ]
        )
        marker.text = " ".join(text_parts)
        return self.apply_lifetime(marker)


class DetectionMessageBuilder:
    @staticmethod
    def to_pose_array(detections: list[RackDetection], frame_id: str, stamp) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = stamp

        for detection in detections:
            pose = Pose()
            pose.position.x = detection.center_x
            pose.position.y = detection.center_y
            pose.position.z = detection.center_z
            pose.orientation.z = math.sin(detection.yaw * 0.5)
            pose.orientation.w = math.cos(detection.yaw * 0.5)
            pose_array.poses.append(pose)

        return pose_array

    @staticmethod
    def to_json_string(detections: list[RackDetection], frame_id: str, stamp, stats: dict) -> String:
        message = String()

        payload = {
            "frame_id": frame_id,
            "stamp": {
                "sec": int(stamp.sec),
                "nanosec": int(stamp.nanosec),
            },
            "stats": stats,
            "detections": [
                {
                    "id": detection.detection_id,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "center": {
                        "x": detection.center_x,
                        "y": detection.center_y,
                        "z": detection.center_z,
                    },
                    "yaw": detection.yaw,
                    "size": {
                        "length": detection.length,
                        "width": detection.width,
                        "height": detection.height,
                    },
                    "corners_xy": detection.corners_xy,
                    "track": {
                        "state": detection.track_state,
                        "hit_count": detection.hit_count,
                        "missed_count": detection.missed_count,
                    },
                }
                for detection in detections
            ],
        }

        message.data = json.dumps(payload, ensure_ascii=False)
        return message

    @staticmethod
    def to_stable_tracks_json_string(
        detections: list[RackDetection],
        frame_id: str,
        stamp,
        include_lost_tracks: bool,
    ) -> String:
        message = String()

        tracks = []
        for detection in detections:
            if detection.track_state == TrackStabilizer.STATE_TENTATIVE:
                continue
            if detection.track_state == TrackStabilizer.STATE_LOST and not include_lost_tracks:
                continue
            if detection.track_state not in (TrackStabilizer.STATE_CONFIRMED, TrackStabilizer.STATE_LOST):
                continue

            tracks.append(
                {
                    "track_id": int(detection.detection_id),
                    "class_id": int(detection.class_id),
                    "class_name": detection.class_name,
                    "state": detection.track_state,
                    "confidence": float(detection.confidence),
                    "center_x": float(detection.center_x),
                    "center_y": float(detection.center_y),
                    "center_z": float(detection.center_z),
                    "yaw": float(detection.yaw),
                    "length": float(detection.length),
                    "width": float(detection.width),
                    "height": float(detection.height),
                    "hit_count": int(detection.hit_count),
                    "missed_count": int(detection.missed_count),
                }
            )

        payload = {
            "frame_id": frame_id,
            "stamp": {
                "sec": int(stamp.sec),
                "nanosec": int(stamp.nanosec),
            },
            "tracks": tracks,
        }

        message.data = json.dumps(payload, ensure_ascii=False)
        return message

    @staticmethod
    def to_bev_image_message(bev_image: np.ndarray, frame_id: str, stamp) -> Image:
        rgb_image = np.clip(bev_image * 255.0, 0.0, 255.0).astype(np.uint8)

        message = Image()
        message.header.frame_id = frame_id
        message.header.stamp = stamp
        message.height = int(rgb_image.shape[0])
        message.width = int(rgb_image.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = int(rgb_image.shape[1] * rgb_image.shape[2])
        message.data = rgb_image.tobytes()
        return message


class YoloObbBevDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_obb_bev_detector_node")

        if YOLO is None:
            raise RuntimeError(
                "Ultralytics konnte nicht importiert werden. "
                f"Ursache: {YOLO_IMPORT_ERROR}"
            )

        self.declare_node_parameters()
        self.load_parameters()
        self.initialize_model()

        self.bev_builder = BevImageBuilder(
            geometry=self.geometry,
            density_clip_count=self.density_clip_count,
        )
        self.display_bev_builder = BevImageBuilder(
            geometry=self.display_geometry,
            density_clip_count=self.density_clip_count,
        )
        self.tile_planner = TilePlanner(
            tile_size_px=self.tile_size_px,
            overlap_ratio=self.tile_overlap_ratio,
        )
        self.threshold_filter = ClassThresholdFilter(
            confidence_threshold=self.confidence_threshold,
            side_confidence_threshold=self.side_confidence_threshold,
            top_confidence_threshold=self.top_confidence_threshold,
        )
        self.obb_nms = OrientedBoxNms(
            iou_threshold=self.obb_nms_iou_threshold,
        )
        self.physical_deduplicator = PhysicalRackDeduplicator(
            enabled=self.deduplicate_physical_racks,
            geometry=self.geometry,
            merge_distance_m=self.physical_merge_distance_m,
            merge_iou_threshold=self.physical_merge_iou_threshold,
            prefer_side_visible=self.physical_prefer_side_visible,
        )
        self.detection_converter = RackDetectionConverter(
            geometry=self.geometry,
            marker_size=self.marker_size,
            rack_floor_z=self.rack_floor_z,
            marker_yaw_mode=self.marker_yaw_mode,
            marker_fixed_yaw_rad=self.marker_fixed_yaw_rad,
            marker_yaw_snap_step_deg=self.marker_yaw_snap_step_deg,
            marker_yaw_snap_offset_deg=self.marker_yaw_snap_offset_deg,
        )
        self.track_stabilizer = TrackStabilizer(
            match_distance_m=self.track_match_distance_m,
            yaw_gate_deg=self.track_yaw_gate_deg,
            min_iou=self.track_min_iou,
            max_match_cost=self.track_max_match_cost,
            confirm_hits=self.track_confirm_hits,
            instant_confirm_confidence=self.track_instant_confirm_confidence,
            max_missed_sec=self.track_max_missed_sec,
            tentative_timeout_sec=self.track_tentative_timeout_sec,
            position_alpha=self.track_position_alpha,
            yaw_alpha=self.track_yaw_alpha,
            confidence_alpha=self.track_confidence_alpha,
            moving_match_distance_m=self.track_moving_match_distance_m,
            lost_match_distance_m=self.track_lost_match_distance_m,
            moving_yaw_gate_deg=self.track_moving_yaw_gate_deg,
            lost_yaw_gate_deg=self.track_lost_yaw_gate_deg,
            reid_max_distance_m=self.track_reid_max_distance_m,
            reid_max_size_ratio_delta=self.track_reid_max_size_ratio_delta,
            reid_max_yaw_diff_deg=self.track_reid_max_yaw_diff_deg,
            moving_speed_threshold_mps=self.track_moving_speed_threshold_mps,
        )
        self.marker_factory = MarkerFactory(
            output_frame=self.output_frame,
            marker_alpha=self.marker_alpha,
            text_marker_enabled=self.text_marker_enabled,
            text_marker_include_id=self.text_marker_include_id,
            marker_lifetime_sec=self.marker_lifetime_sec,
        )

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.centroid_pub = self.create_publisher(PoseArray, self.centroid_topic, 10)
        self.json_pub = self.create_publisher(String, self.json_topic, 10)
        self.bev_image_pub = None
        if self.publish_bev_image:
            self.bev_image_pub = self.create_publisher(Image, self.bev_image_topic, 10)
        self.stable_track_pub = None
        if self.publish_stable_tracks:
            self.stable_track_pub = self.create_publisher(String, self.stable_track_topic, 10)

        self.tracking_centroid_pub = None
        if self.publish_tracking_centroids:
            self.tracking_centroid_pub = self.create_publisher(PoseArray, "/tracking/cluster_centroids", 10)

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.handle_cloud,
            qos_profile_sensor_data,
        )

        self.last_inference_time_sec: Optional[float] = None
        self.received_count = 0
        self.inference_count = 0

        self.get_logger().info("yolo_obb_bev_detector_node started.")
        self.get_logger().info(f"Input topic: {self.input_topic}")
        self.get_logger().info(f"Expected frame_id: {self.expected_frame_id}")
        self.get_logger().info(f"Output frame: {self.output_frame}")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"ROI min: {self.geometry.roi_min.tolist()}")
        self.get_logger().info(f"ROI max: {self.geometry.roi_max.tolist()}")
        self.get_logger().info(
            f"BEV image: {self.geometry.width_px}x{self.geometry.height_px} px "
            f"@ {self.geometry.resolution_m_per_px:.4f} m/px"
        )
        self.get_logger().info(
            f"BEV display padding: x={self.bev_padding_m_x:.2f}m, y={self.bev_padding_m_y:.2f}m"
        )
        self.get_logger().info(f"Display ROI min: {self.display_geometry.roi_min.tolist()}")
        self.get_logger().info(f"Display ROI max: {self.display_geometry.roi_max.tolist()}")
        self.get_logger().info(
            f"Display BEV image: {self.display_geometry.width_px}x{self.display_geometry.height_px} px "
            f"@ {self.display_geometry.resolution_m_per_px:.4f} m/px"
        )
        self.get_logger().info(
            f"Sliding window: tile_size_px={self.tile_size_px}, "
            f"tile_overlap_ratio={self.tile_overlap_ratio}, imgsz={self.imgsz}"
        )
        self.get_logger().info(
            f"Thresholds: model_conf={self.model_confidence_threshold}, "
            f"global_conf={self.confidence_threshold}, "
            f"side_conf={self.side_confidence_threshold}, "
            f"top_conf={self.top_confidence_threshold}, "
            f"obb_nms_iou={self.obb_nms_iou_threshold}"
        )
        self.get_logger().info(
            f"Physical dedup: enabled={self.deduplicate_physical_racks}, "
            f"merge_distance={self.physical_merge_distance_m:.2f}m, "
            f"merge_iou={self.physical_merge_iou_threshold:.2f}, "
            f"prefer_side={self.physical_prefer_side_visible}"
        )
        self.get_logger().info(
            f"MOT-Light: enabled={self.track_enabled}, "
            f"match_distance={self.track_match_distance_m:.2f}m, "
            f"moving_match_distance={self.track_moving_match_distance_m:.2f}m, "
            f"lost_match_distance={self.track_lost_match_distance_m:.2f}m, "
            f"yaw_gate={self.track_yaw_gate_deg:.1f}deg, "
            f"confirm_hits={self.track_confirm_hits}, "
            f"max_missed_sec={self.track_max_missed_sec:.1f}"
        )
        self.get_logger().info(
            f"Track prediction/reid: moving_speed_threshold={self.track_moving_speed_threshold_mps:.2f}m/s, "
            f"reid_distance={self.track_reid_max_distance_m:.2f}m, "
            f"reid_size_delta={self.track_reid_max_size_ratio_delta:.2f}, "
            f"reid_yaw_diff_deg={self.track_reid_max_yaw_diff_deg:.1f}"
        )
        self.get_logger().info(
            f"Marker yaw: mode={self.marker_yaw_mode}, "
            f"fixed_yaw_rad={self.marker_fixed_yaw_rad:.4f}, "
            f"snap_step_deg={self.marker_yaw_snap_step_deg:.1f}, "
            f"snap_offset_deg={self.marker_yaw_snap_offset_deg:.1f}"
        )
        self.get_logger().info(
            f"Marker config: alpha={self.marker_alpha:.2f}, "
            f"text_enabled={self.text_marker_enabled}, "
            f"text_include_id={self.text_marker_include_id}, "
            f"lifetime_sec={self.marker_lifetime_sec:.2f}"
        )
        self.get_logger().info(
            f"Stable track output: enabled={self.publish_stable_tracks}, "
            f"topic={self.stable_track_topic}, include_lost={self.stable_track_include_lost}"
        )
        self.get_logger().info(
            f"BEV image output: enabled={self.publish_bev_image}, topic={self.bev_image_topic}"
        )
        self.get_logger().info(f"Marker topic: {self.marker_topic}")
        self.get_logger().info(f"Centroid topic: {self.centroid_topic}")
        self.get_logger().info(f"JSON topic: {self.json_topic}")

    def declare_node_parameters(self) -> None:
        self.declare_parameter("input_topic", "/rslidar_points")
        self.declare_parameter("expected_frame_id", "rslidar")
        self.declare_parameter("output_frame", "rslidar")

        self.declare_parameter(
            "model_path",
            "/home/edv/ros2_ws/bev_dataset/models/rack_bev_obb_yolo26m_v4m_racksv2_best_epoch40.pt",
        )

        self.declare_parameter("roi_min", [-14.5, -15.0, -2.0])
        self.declare_parameter("roi_max", [9.0, 6.0, 3.0])
        self.declare_parameter("resolution_m_per_px", 0.01)
        self.declare_parameter("bev_padding_m", 1.0)
        self.declare_parameter("bev_padding_m_x", 0.0)
        self.declare_parameter("bev_padding_m_y", 0.0)

        self.declare_parameter("density_clip_count", 12)
        self.declare_parameter("min_points_in_roi", 1000)

        self.declare_parameter("inference_interval_sec", 1.0)
        self.declare_parameter("imgsz", 1024)
        self.declare_parameter("model_confidence_threshold", 0.03)
        self.declare_parameter("confidence_threshold", 0.03)
        self.declare_parameter("iou_threshold", 0.50)
        self.declare_parameter("device", "0")

        self.declare_parameter("tile_size_px", 1024)
        self.declare_parameter("tile_overlap_ratio", 0.25)
        self.declare_parameter("side_confidence_threshold", 0.20)
        self.declare_parameter("top_confidence_threshold", 0.05)
        self.declare_parameter("obb_nms_iou_threshold", 0.20)

        self.declare_parameter("deduplicate_physical_racks", True)
        self.declare_parameter("physical_merge_distance_m", 0.45)
        self.declare_parameter("physical_merge_iou_threshold", 0.05)
        self.declare_parameter("physical_prefer_side_visible", True)

        self.declare_parameter("marker_size", [1.05, 1.05, 2.00])
        self.declare_parameter("marker_yaw_mode", "fixed")
        self.declare_parameter("marker_fixed_yaw_rad", 0.0)
        self.declare_parameter("marker_yaw_snap_step_deg", 90.0)
        self.declare_parameter("marker_yaw_snap_offset_deg", 0.0)
        self.declare_parameter("rack_floor_z", -0.95)
        self.declare_parameter("marker_alpha", 0.35)
        self.declare_parameter("text_marker_enabled", True)
        self.declare_parameter("text_marker_include_id", True)
        self.declare_parameter("marker_lifetime_sec", 2.5)

        self.declare_parameter("track_enabled", True)
        self.declare_parameter("track_match_distance_m", 0.50)
        self.declare_parameter("track_yaw_gate_deg", 45.0)
        self.declare_parameter("track_min_iou", 0.02)
        self.declare_parameter("track_max_match_cost", 2.40)
        self.declare_parameter("track_confirm_hits", 2)
        self.declare_parameter("track_instant_confirm_confidence", 0.45)
        self.declare_parameter("track_max_missed_sec", 8.0)
        self.declare_parameter("track_tentative_timeout_sec", 3.0)
        self.declare_parameter("track_position_alpha", 0.25)
        self.declare_parameter("track_yaw_alpha", 0.20)
        self.declare_parameter("track_confidence_alpha", 0.35)
        self.declare_parameter("track_moving_match_distance_m", 0.90)
        self.declare_parameter("track_lost_match_distance_m", 1.30)
        self.declare_parameter("track_moving_yaw_gate_deg", 75.0)
        self.declare_parameter("track_lost_yaw_gate_deg", 100.0)
        self.declare_parameter("track_reid_max_distance_m", 1.50)
        self.declare_parameter("track_reid_max_size_ratio_delta", 0.35)
        self.declare_parameter("track_reid_max_yaw_diff_deg", 100.0)
        self.declare_parameter("track_moving_speed_threshold_mps", 0.15)

        self.declare_parameter("marker_topic", "/detection/rack_obb_markers")
        self.declare_parameter("centroid_topic", "/detection/rack_centroids")
        self.declare_parameter("json_topic", "/detection/rack_detections_json")
        self.declare_parameter("stable_track_topic", "/tracking/stable_tracks")
        self.declare_parameter("publish_stable_tracks", True)
        self.declare_parameter("stable_track_include_lost", True)
        self.declare_parameter("bev_image_topic", "/detection/bev_image")
        self.declare_parameter("publish_bev_image", True)
        self.declare_parameter("publish_tracking_centroids", False)

    def load_parameters(self) -> None:
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.expected_frame_id = str(self.get_parameter("expected_frame_id").value).strip()
        self.output_frame = str(self.get_parameter("output_frame").value).strip()

        self.model_path = str(Path(str(self.get_parameter("model_path").value)).expanduser())

        roi_min = [float(value) for value in self.get_parameter("roi_min").value]
        roi_max = [float(value) for value in self.get_parameter("roi_max").value]
        resolution_m_per_px = float(self.get_parameter("resolution_m_per_px").value)
        self.bev_padding_m = max(float(self.get_parameter("bev_padding_m").value), 0.0)
        bev_padding_m_x = max(float(self.get_parameter("bev_padding_m_x").value), 0.0)
        bev_padding_m_y = max(float(self.get_parameter("bev_padding_m_y").value), 0.0)
        self.bev_padding_m_x = bev_padding_m_x if bev_padding_m_x > 0.0 else self.bev_padding_m
        self.bev_padding_m_y = bev_padding_m_y if bev_padding_m_y > 0.0 else self.bev_padding_m

        self.geometry = BevGeometry.from_values(
            roi_min=roi_min,
            roi_max=roi_max,
            resolution_m_per_px=resolution_m_per_px,
        )
        self.display_geometry = self.geometry.expand_xy(
            padding_x_m=self.bev_padding_m_x,
            padding_y_m=self.bev_padding_m_y,
        )

        self.density_clip_count = int(self.get_parameter("density_clip_count").value)
        self.min_points_in_roi = int(self.get_parameter("min_points_in_roi").value)

        self.inference_interval_sec = float(self.get_parameter("inference_interval_sec").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.model_confidence_threshold = float(self.get_parameter("model_confidence_threshold").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.device = str(self.get_parameter("device").value)

        self.tile_size_px = int(self.get_parameter("tile_size_px").value)
        self.tile_overlap_ratio = float(self.get_parameter("tile_overlap_ratio").value)
        self.side_confidence_threshold = float(self.get_parameter("side_confidence_threshold").value)
        self.top_confidence_threshold = float(self.get_parameter("top_confidence_threshold").value)
        self.obb_nms_iou_threshold = float(self.get_parameter("obb_nms_iou_threshold").value)

        self.deduplicate_physical_racks = bool(self.get_parameter("deduplicate_physical_racks").value)
        self.physical_merge_distance_m = float(self.get_parameter("physical_merge_distance_m").value)
        self.physical_merge_iou_threshold = float(self.get_parameter("physical_merge_iou_threshold").value)
        self.physical_prefer_side_visible = bool(self.get_parameter("physical_prefer_side_visible").value)

        self.marker_size = [float(value) for value in self.get_parameter("marker_size").value]
        self.marker_yaw_mode = str(self.get_parameter("marker_yaw_mode").value)
        self.marker_fixed_yaw_rad = float(self.get_parameter("marker_fixed_yaw_rad").value)
        self.marker_yaw_snap_step_deg = float(self.get_parameter("marker_yaw_snap_step_deg").value)
        self.marker_yaw_snap_offset_deg = float(self.get_parameter("marker_yaw_snap_offset_deg").value)
        self.rack_floor_z = float(self.get_parameter("rack_floor_z").value)
        self.marker_alpha = float(self.get_parameter("marker_alpha").value)
        self.text_marker_enabled = bool(self.get_parameter("text_marker_enabled").value)
        self.text_marker_include_id = bool(self.get_parameter("text_marker_include_id").value)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)

        self.track_enabled = bool(self.get_parameter("track_enabled").value)
        self.track_match_distance_m = float(self.get_parameter("track_match_distance_m").value)
        self.track_yaw_gate_deg = float(self.get_parameter("track_yaw_gate_deg").value)
        self.track_min_iou = float(self.get_parameter("track_min_iou").value)
        self.track_max_match_cost = float(self.get_parameter("track_max_match_cost").value)
        self.track_confirm_hits = int(self.get_parameter("track_confirm_hits").value)
        self.track_instant_confirm_confidence = float(self.get_parameter("track_instant_confirm_confidence").value)
        self.track_max_missed_sec = float(self.get_parameter("track_max_missed_sec").value)
        self.track_tentative_timeout_sec = float(self.get_parameter("track_tentative_timeout_sec").value)
        self.track_position_alpha = float(self.get_parameter("track_position_alpha").value)
        self.track_yaw_alpha = float(self.get_parameter("track_yaw_alpha").value)
        self.track_confidence_alpha = float(self.get_parameter("track_confidence_alpha").value)
        self.track_moving_match_distance_m = float(self.get_parameter("track_moving_match_distance_m").value)
        self.track_lost_match_distance_m = float(self.get_parameter("track_lost_match_distance_m").value)
        self.track_moving_yaw_gate_deg = float(self.get_parameter("track_moving_yaw_gate_deg").value)
        self.track_lost_yaw_gate_deg = float(self.get_parameter("track_lost_yaw_gate_deg").value)
        self.track_reid_max_distance_m = float(self.get_parameter("track_reid_max_distance_m").value)
        self.track_reid_max_size_ratio_delta = float(self.get_parameter("track_reid_max_size_ratio_delta").value)
        self.track_reid_max_yaw_diff_deg = float(self.get_parameter("track_reid_max_yaw_diff_deg").value)
        self.track_moving_speed_threshold_mps = float(self.get_parameter("track_moving_speed_threshold_mps").value)

        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.centroid_topic = str(self.get_parameter("centroid_topic").value)
        self.json_topic = str(self.get_parameter("json_topic").value)
        self.stable_track_topic = str(self.get_parameter("stable_track_topic").value)
        self.publish_stable_tracks = bool(self.get_parameter("publish_stable_tracks").value)
        self.stable_track_include_lost = bool(self.get_parameter("stable_track_include_lost").value)
        self.bev_image_topic = str(self.get_parameter("bev_image_topic").value)
        self.publish_bev_image = bool(self.get_parameter("publish_bev_image").value)
        self.publish_tracking_centroids = bool(self.get_parameter("publish_tracking_centroids").value)

    def initialize_model(self) -> None:
        model_file = Path(self.model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"YOLO-Modell nicht gefunden: {model_file}")

        self.model = YOLO(str(model_file))

    def handle_cloud(self, msg: PointCloud2) -> None:
        self.received_count += 1

        if self.expected_frame_id and msg.header.frame_id != self.expected_frame_id:
            if self.received_count <= 5:
                self.get_logger().warn(
                    f"Skipping cloud with frame_id={msg.header.frame_id!r}; "
                    f"expected {self.expected_frame_id!r}."
                )
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.last_inference_time_sec is not None:
            if now_sec - self.last_inference_time_sec < self.inference_interval_sec:
                return

        self.last_inference_time_sec = now_sec

        points_xyz = PointCloudReader.to_xyz_array(msg)
        bev_image, stats = self.bev_builder.build(points_xyz)
        display_bev_image = None
        display_stats = None
        if self.publish_bev_image:
            display_bev_image, display_stats = self.display_bev_builder.build(points_xyz)
        stats["display_points_in_roi"] = int(display_stats["points_in_roi"]) if display_stats is not None else stats["points_in_roi"]
        stats["display_occupied_pixels"] = (
            int(display_stats["occupied_pixels"]) if display_stats is not None else stats["occupied_pixels"]
        )
        stats["display_image_width_px"] = int(self.display_geometry.width_px)
        stats["display_image_height_px"] = int(self.display_geometry.height_px)

        if stats["points_in_roi"] < self.min_points_in_roi:
            self.get_logger().warn(
                f"Skipping inference: only {stats['points_in_roi']} ROI points "
                f"(min_points_in_roi={self.min_points_in_roi})."
            )
            self.publish_empty(msg.header.stamp, stats, display_bev_image)
            return

        raw_detections = self.run_inference(bev_image)
        self.inference_count += 1

        if self.track_enabled:
            detections = self.track_stabilizer.update(raw_detections, now_sec)
        else:
            detections = raw_detections

        stats["raw_detections"] = len(raw_detections)
        stats["published_detections"] = len(detections)
        stats["active_tracks"] = len(self.track_stabilizer.tracks) if self.track_enabled else 0

        self.publish_detections(detections, msg.header.stamp, stats, display_bev_image if display_bev_image is not None else bev_image)

        side_count = sum(1 for detection in detections if detection.class_id == 0)
        top_count = sum(1 for detection in detections if detection.class_id == 1)
        confirmed_count = sum(1 for detection in detections if detection.track_state == "confirmed")
        lost_count = sum(1 for detection in detections if detection.track_state == "lost")

        self.get_logger().info(
            f"Inference #{self.inference_count}: raw={len(raw_detections)}, "
            f"published={len(detections)}, confirmed={confirmed_count}, lost={lost_count}, "
            f"side={side_count}, top={top_count}, "
            f"points_in_roi={stats['points_in_roi']}, occupied_pixels={stats['occupied_pixels']}"
        )

    def run_inference(self, bev_image: np.ndarray) -> list[RackDetection]:
        image_height, image_width = bev_image.shape[:2]
        windows = self.tile_planner.create_windows(image_width, image_height)

        raw_candidates: list[DetectionCandidate] = []

        for window in windows:
            tile = bev_image[window.y_min:window.y_max, window.x_min:window.x_max]

            if tile.shape[0] != self.tile_size_px or tile.shape[1] != self.tile_size_px:
                continue

            results = self.model.predict(
                source=tile,
                imgsz=self.imgsz,
                conf=self.model_confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            if not results:
                continue

            raw_candidates.extend(self.extract_candidates_from_result(results[0], window))

        threshold_candidates = [
            candidate for candidate in raw_candidates
            if self.threshold_filter.keep(candidate)
        ]

        nms_candidates = self.obb_nms.apply(threshold_candidates)
        final_candidates = self.physical_deduplicator.apply(nms_candidates)

        detections: list[RackDetection] = []

        for index, candidate in enumerate(final_candidates, start=1):
            detections.append(
                self.detection_converter.convert(
                    detection_id=index,
                    candidate=candidate,
                )
            )

        return detections

    def extract_candidates_from_result(self, result, window: TileWindow) -> list[DetectionCandidate]:
        if result.obb is None:
            return []

        if result.obb.xyxyxyxy is None:
            return []

        pixel_boxes = result.obb.xyxyxyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy() if result.obb.conf is not None else np.ones(len(pixel_boxes))
        classes = result.obb.cls.cpu().numpy() if result.obb.cls is not None else np.zeros(len(pixel_boxes))

        candidates: list[DetectionCandidate] = []

        for index, pixel_box in enumerate(pixel_boxes):
            class_id = int(classes[index])

            if class_id not in (0, 1):
                continue

            pixel_corners = np.asarray(pixel_box, dtype=np.float32).reshape(4, 2)
            pixel_corners[:, 0] += window.x_min
            pixel_corners[:, 1] += window.y_min

            class_name = RackDetectionConverter.CLASS_NAMES.get(class_id, f"class_{class_id}")

            candidates.append(
                DetectionCandidate(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[index]),
                    pixel_corners=pixel_corners,
                    source_tile=window,
                )
            )

        return candidates

    def publish_empty(self, stamp, stats: dict, bev_image: np.ndarray | None = None) -> None:
        self.publish_detections([], stamp, stats, bev_image)

    def publish_detections(self, detections: list[RackDetection], stamp, stats: dict, bev_image: np.ndarray | None) -> None:
        marker_array = self.marker_factory.build_marker_array(detections, stamp)
        pose_array = DetectionMessageBuilder.to_pose_array(detections, self.output_frame, stamp)
        json_message = DetectionMessageBuilder.to_json_string(detections, self.output_frame, stamp, stats)

        self.marker_pub.publish(marker_array)
        self.centroid_pub.publish(pose_array)
        self.json_pub.publish(json_message)
        if self.bev_image_pub is not None and bev_image is not None:
            bev_image_message = DetectionMessageBuilder.to_bev_image_message(bev_image, self.output_frame, stamp)
            self.bev_image_pub.publish(bev_image_message)
        if self.stable_track_pub is not None:
            stable_track_message = DetectionMessageBuilder.to_stable_tracks_json_string(
                detections=detections,
                frame_id=self.output_frame,
                stamp=stamp,
                include_lost_tracks=self.stable_track_include_lost,
            )
            self.stable_track_pub.publish(stable_track_message)

        if self.tracking_centroid_pub is not None:
            self.tracking_centroid_pub.publish(pose_array)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)

    node = YoloObbBevDetectorNode()

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
