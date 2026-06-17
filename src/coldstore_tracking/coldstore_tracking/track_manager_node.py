from __future__ import annotations

from dataclasses import dataclass
from builtin_interfaces.msg import Duration
from builtin_interfaces.msg import Time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .event_utils import (
    as_float,
    as_int,
    build_track_states_payload,
    get_payload_dict,
    get_payload_list,
    make_string_message,
    parse_string_message,
)
from .tracking_types import Track


@dataclass(frozen=True)
class SourceTrackObservation:
    source_track_id: int
    state: str
    class_id: int
    class_name: str
    confidence: float
    center: np.ndarray
    yaw: float
    length: float
    width: float
    height: float
    hit_count: int
    missed_count: int
    raw_item: dict


@dataclass(frozen=True)
class RecoveryCandidate:
    track_id: int
    score: float
    distance_m: float
    identity_confidence: float
    strict_match: bool


class MotionStateLogic:
    MOTION_STATIC = 'static'
    MOTION_MOVING = 'moving'
    MOTION_OCCLUDED = 'occluded'
    MOTION_DISAPPEARED = 'disappeared'
    MOTION_NEWLY_APPEARED = 'newly_appeared'

    def __init__(
        self,
        newly_appeared_hold_sec: float,
        motion_static_threshold_m: float,
        motion_moving_threshold_m: float,
        motion_required_updates: int,
        occluded_timeout_sec: float,
    ) -> None:
        self.newly_appeared_hold_sec = float(newly_appeared_hold_sec)
        self.motion_static_threshold_m = float(motion_static_threshold_m)
        self.motion_moving_threshold_m = float(motion_moving_threshold_m)
        self.motion_required_updates = max(int(motion_required_updates), 1)
        self.occluded_timeout_sec = float(occluded_timeout_sec)

    def update_confirmed_motion_state(
        self,
        track: Track,
        previous_motion_state: str,
        displacement_m: float,
        stamp_sec: float,
    ) -> None:
        if displacement_m <= self.motion_static_threshold_m:
            track.static_streak += 1
            track.moving_streak = 0
        elif displacement_m >= self.motion_moving_threshold_m:
            track.moving_streak += 1
            track.static_streak = 0
        else:
            track.static_streak = 0
            track.moving_streak = 0

        is_new_track = track.first_seen_sec <= 0.0 or (track.last_stamp_sec - track.first_seen_sec) <= self.newly_appeared_hold_sec
        if is_new_track and previous_motion_state == self.MOTION_NEWLY_APPEARED:
            track.motion_state = self.MOTION_NEWLY_APPEARED
        elif is_new_track and track.hit_count <= self.motion_required_updates:
            track.motion_state = self.MOTION_NEWLY_APPEARED
        elif track.moving_streak >= self.motion_required_updates:
            track.motion_state = self.MOTION_MOVING
        elif track.static_streak >= self.motion_required_updates:
            track.motion_state = self.MOTION_STATIC
        elif previous_motion_state in (self.MOTION_STATIC, self.MOTION_MOVING):
            track.motion_state = previous_motion_state
        else:
            track.motion_state = self.MOTION_STATIC

        self.apply_motion_state_change(track, previous_motion_state, stamp_sec)

    def update_lost_motion_state(self, track: Track, stamp_sec: float) -> None:
        previous_motion_state = track.motion_state
        time_since_confirmed = max(track.last_stamp_sec - track.last_confirmed_sec, 0.0)
        if time_since_confirmed <= self.occluded_timeout_sec:
            track.motion_state = self.MOTION_OCCLUDED
        else:
            track.motion_state = self.MOTION_DISAPPEARED
        self.apply_motion_state_change(track, previous_motion_state, stamp_sec)

    def apply_motion_state_change(self, track: Track, previous_motion_state: str, stamp_sec: float) -> None:
        if track.motion_state == previous_motion_state:
            return
        if track.motion_state == self.MOTION_OCCLUDED:
            track.occluded_transition_count += 1
        track.last_motion_state_change_sec = stamp_sec


class TrackMarkerFactory:
    def __init__(self, marker_lifetime_sec: float) -> None:
        self.marker_lifetime_sec = max(float(marker_lifetime_sec), 0.0)

    def build_marker_array(self, tracks: Dict[int, Track], stamp, marker_frame: str) -> MarkerArray:
        marker_array = MarkerArray()
        marker_array.markers.extend(self.build_delete_all_markers(stamp, marker_frame))

        for track in sorted(tracks.values(), key=lambda item: item.track_id):
            marker_array.markers.append(self.build_sphere_marker(track, stamp, marker_frame))
            marker_array.markers.append(self.build_text_marker(track, stamp, marker_frame))

        return marker_array

    def build_delete_all_markers(self, stamp, marker_frame: str) -> list[Marker]:
        return [self.build_delete_all_marker(stamp, marker_frame)]

    def build_delete_all_marker(self, stamp, marker_frame: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = marker_frame
        marker.header.stamp = stamp
        marker.ns = 'track_marker_cleanup'
        marker.id = 0
        marker.action = Marker.DELETEALL
        return marker

    def build_sphere_marker(self, track: Track, stamp, marker_frame: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = marker_frame
        marker.header.stamp = stamp
        marker.ns = 'tracks'
        marker.id = int(track.track_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(track.centroid[0])
        marker.pose.position.y = float(track.centroid[1])
        marker.pose.position.z = float(track.centroid[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r, marker.color.g, marker.color.b = self.track_color(track)
        marker.color.a = 0.45 if track.motion_state == MotionStateLogic.MOTION_DISAPPEARED else 1.0
        return self.apply_lifetime(marker)

    def build_text_marker(self, track: Track, stamp, marker_frame: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = marker_frame
        marker.header.stamp = stamp
        marker.ns = 'track_labels'
        marker.id = int(track.track_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(track.centroid[0])
        marker.pose.position.y = float(track.centroid[1])
        marker.pose.position.z = float(track.centroid[2] + 0.35)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.28
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        speed = float(np.linalg.norm(track.velocity))
        barcode = track.barcode_id if track.barcode_id else '-'
        class_label = track.class_name if track.class_name else '-'
        marker.text = (
            f'T{track.track_id} {track.state}/{track.motion_state} {class_label} id:{barcode} '
            f'hit:{track.hit_count} miss:{track.source_missed_count} v:{speed:.2f}m/s'
        )
        return self.apply_lifetime(marker)

    def track_color(self, track: Track) -> tuple[float, float, float]:
        if track.state == 'lost':
            return 1.0, 0.8, 0.0
        if track.motion_state == MotionStateLogic.MOTION_MOVING:
            return 0.0, 0.85, 0.2
        if track.motion_state == MotionStateLogic.MOTION_NEWLY_APPEARED:
            return 0.2, 0.6, 1.0
        if track.motion_state == MotionStateLogic.MOTION_DISAPPEARED:
            return 0.6, 0.6, 0.6
        if track.missed_updates == 0:
            return 1.0, 0.0, 0.0
        return 1.0, 0.8, 0.0

    def apply_lifetime(self, marker: Marker) -> Marker:
        lifetime_sec = int(self.marker_lifetime_sec)
        lifetime_nanosec = int(round((self.marker_lifetime_sec - lifetime_sec) * 1e9))
        if lifetime_nanosec >= 1_000_000_000:
            lifetime_sec += 1
            lifetime_nanosec -= 1_000_000_000
        marker.lifetime = Duration(sec=lifetime_sec, nanosec=lifetime_nanosec)
        return marker


class TrackManagerNode(Node):
    MOTION_STATIC = MotionStateLogic.MOTION_STATIC
    MOTION_MOVING = MotionStateLogic.MOTION_MOVING
    MOTION_OCCLUDED = MotionStateLogic.MOTION_OCCLUDED
    MOTION_DISAPPEARED = MotionStateLogic.MOTION_DISAPPEARED
    MOTION_NEWLY_APPEARED = MotionStateLogic.MOTION_NEWLY_APPEARED
    ACTIVE_MOTION_STATES = {MOTION_STATIC, MOTION_MOVING, MOTION_NEWLY_APPEARED}
    ASSIGNMENT_ALLOWED_IDENTITY_STATES = {'direct', 'new', 'recovered_strict'}
    MARRIAGE_KNOWN_EXISTING = 'known_existing'
    MARRIAGE_UNASSIGNED_NEW = 'unassigned_new'
    MARRIAGE_ASSIGNED = 'assigned'

    def __init__(self) -> None:
        super().__init__('track_manager_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('max_match_distance', 0.80)
        self.declare_parameter('touched_match_distance', 1.80)
        self.declare_parameter('touched_update_alpha', 0.55)
        self.declare_parameter('max_missed_updates', 80)
        self.declare_parameter('create_tracks_from_touched', False)
        self.declare_parameter('use_stable_tracks_input', False)
        self.declare_parameter('stable_track_topic', '/tracking/stable_tracks')
        self.declare_parameter('stable_track_include_lost', True)
        self.declare_parameter('stable_track_preserve_missing_tracks', False)
        self.declare_parameter('stable_track_publish_disappeared_markers', False)
        self.declare_parameter('stable_track_discontinuity_distance_m', 1.50)
        self.declare_parameter('identity_recovery_enabled', True)
        self.declare_parameter('identity_recovery_ttl_sec', 8.0)
        self.declare_parameter('identity_recovery_gate_m', 0.90)
        self.declare_parameter('identity_recovery_strict_gate_m', 0.45)
        self.declare_parameter('identity_recovery_max_ambiguous_score_gap', 0.15)
        self.declare_parameter('identity_recovery_use_velocity_prediction', True)
        self.declare_parameter('identity_recovery_log_events', True)
        self.declare_parameter('identity_preserve_uid_only_on_strict_match', True)
        self.declare_parameter('newly_appeared_hold_sec', 1.5)
        self.declare_parameter('motion_static_threshold_m', 0.03)
        self.declare_parameter('motion_moving_threshold_m', 0.12)
        self.declare_parameter('motion_required_updates', 3)
        self.declare_parameter('occluded_timeout_sec', 3.0)
        self.declare_parameter('disappeared_retention_sec', 4.0)
        self.declare_parameter('track_marker_lifetime_sec', 2.5)
        self.declare_parameter('assignment_initialization_duration_sec', 15.0)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_match_distance = float(self.get_parameter('max_match_distance').value)
        self.touched_match_distance = float(self.get_parameter('touched_match_distance').value)
        self.touched_update_alpha = float(self.get_parameter('touched_update_alpha').value)
        self.max_missed_updates = int(self.get_parameter('max_missed_updates').value)
        self.create_tracks_from_touched = bool(self.get_parameter('create_tracks_from_touched').value)
        self.use_stable_tracks_input = bool(self.get_parameter('use_stable_tracks_input').value)
        self.stable_track_topic = str(self.get_parameter('stable_track_topic').value)
        self.stable_track_include_lost = bool(self.get_parameter('stable_track_include_lost').value)
        self.stable_track_preserve_missing_tracks = bool(self.get_parameter('stable_track_preserve_missing_tracks').value)
        self.stable_track_publish_disappeared_markers = bool(self.get_parameter('stable_track_publish_disappeared_markers').value)
        self.stable_track_discontinuity_distance_m = float(self.get_parameter('stable_track_discontinuity_distance_m').value)
        self.identity_recovery_enabled = bool(self.get_parameter('identity_recovery_enabled').value)
        self.identity_recovery_ttl_sec = float(self.get_parameter('identity_recovery_ttl_sec').value)
        self.identity_recovery_gate_m = float(self.get_parameter('identity_recovery_gate_m').value)
        self.identity_recovery_strict_gate_m = float(self.get_parameter('identity_recovery_strict_gate_m').value)
        self.identity_recovery_max_ambiguous_score_gap = float(self.get_parameter('identity_recovery_max_ambiguous_score_gap').value)
        self.identity_recovery_use_velocity_prediction = bool(self.get_parameter('identity_recovery_use_velocity_prediction').value)
        self.identity_recovery_log_events = bool(self.get_parameter('identity_recovery_log_events').value)
        self.identity_preserve_uid_only_on_strict_match = bool(self.get_parameter('identity_preserve_uid_only_on_strict_match').value)
        self.newly_appeared_hold_sec = float(self.get_parameter('newly_appeared_hold_sec').value)
        self.motion_static_threshold_m = float(self.get_parameter('motion_static_threshold_m').value)
        self.motion_moving_threshold_m = float(self.get_parameter('motion_moving_threshold_m').value)
        self.motion_required_updates = max(int(self.get_parameter('motion_required_updates').value), 1)
        self.occluded_timeout_sec = float(self.get_parameter('occluded_timeout_sec').value)
        self.disappeared_retention_sec = float(self.get_parameter('disappeared_retention_sec').value)
        self.track_marker_lifetime_sec = float(self.get_parameter('track_marker_lifetime_sec').value)
        self.assignment_initialization_duration_sec = max(
            float(self.get_parameter('assignment_initialization_duration_sec').value),
            0.0,
        )

        self.motion_logic = MotionStateLogic(
            newly_appeared_hold_sec=self.newly_appeared_hold_sec,
            motion_static_threshold_m=self.motion_static_threshold_m,
            motion_moving_threshold_m=self.motion_moving_threshold_m,
            motion_required_updates=self.motion_required_updates,
            occluded_timeout_sec=self.occluded_timeout_sec,
        )
        self.track_marker_factory = TrackMarkerFactory(
            marker_lifetime_sec=self.track_marker_lifetime_sec,
        )

        self.tracks: Dict[int, Track] = {}
        self.identity_lost_tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_stamp_sec = 0.0
        self.last_stamp_msg = None
        self.last_source_frame_id = ''
        self.initialization_start_sec: float | None = None
        self.initialization_end_sec: float | None = None

        self.centroid_sub = None
        self.touched_centroid_sub = None
        self.stable_track_sub = None
        if self.use_stable_tracks_input:
            self.stable_track_sub = self.create_subscription(
                String,
                self.stable_track_topic,
                self.stable_track_callback,
                10,
            )
        else:
            self.centroid_sub = self.create_subscription(
                PoseArray,
                '/tracking/cluster_centroids',
                self.centroid_callback,
                10,
            )
            self.touched_centroid_sub = self.create_subscription(
                PoseArray,
                '/tracking/touched_cluster_centroids',
                self.touched_centroid_callback,
                10,
            )
        self.assignment_sub = self.create_subscription(
            String,
            '/tracking/id_assignments',
            self.assignment_callback,
            10,
        )
        self.remove_track_sub = self.create_subscription(
            String,
            '/tracking/remove_track_events',
            self.remove_track_callback,
            10,
        )

        self.track_marker_pub = self.create_publisher(MarkerArray, '/tracking/track_markers', 10)
        self.track_state_pub = self.create_publisher(String, '/tracking/track_states', 10)

        self.clear_tracks_srv = self.create_service(
            Trigger,
            '/tracking/clear_tracks',
            self.clear_tracks,
        )

        self.get_logger().info('track_manager_node started.')
        self.get_logger().info(
            f'Input mode: stable_tracks={self.use_stable_tracks_input}, '
            f'stable_track_topic={self.stable_track_topic}, '
            f'include_lost={self.stable_track_include_lost}, '
            f'preserve_missing_tracks={self.stable_track_preserve_missing_tracks}, '
            f'publish_disappeared_markers={self.stable_track_publish_disappeared_markers}, '
            f'discontinuity_distance_m={self.stable_track_discontinuity_distance_m:.2f}'
        )
        self.get_logger().info(
            f'Identity recovery: enabled={self.identity_recovery_enabled}, '
            f'ttl_sec={self.identity_recovery_ttl_sec:.2f}, '
            f'gate_m={self.identity_recovery_gate_m:.2f}, '
            f'strict_gate_m={self.identity_recovery_strict_gate_m:.2f}, '
            f'ambiguous_gap={self.identity_recovery_max_ambiguous_score_gap:.2f}, '
            f'use_velocity_prediction={self.identity_recovery_use_velocity_prediction}, '
            f'preserve_uid_only_on_strict={self.identity_preserve_uid_only_on_strict_match}'
        )
        self.get_logger().info(
            f'Motion logic: hold_sec={self.newly_appeared_hold_sec:.2f}, '
            f'static_threshold_m={self.motion_static_threshold_m:.3f}, '
            f'moving_threshold_m={self.motion_moving_threshold_m:.3f}, '
            f'required_updates={self.motion_required_updates}, '
            f'occluded_timeout_sec={self.occluded_timeout_sec:.2f}, '
            f'disappeared_retention_sec={self.disappeared_retention_sec:.2f}, '
            f'marker_lifetime_sec={self.track_marker_lifetime_sec:.2f}'
        )
        self.get_logger().info(
            f'Assignment baseline: initialization_duration_sec='
            f'{self.assignment_initialization_duration_sec:.2f}'
        )

    def centroid_callback(self, pose_array: PoseArray) -> None:
        detections = self.pose_array_to_detections(pose_array)
        stamp_sec = self.stamp_to_sec(pose_array)
        self.ensure_initialization_window(stamp_sec)
        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = pose_array.header.stamp

        self.update_tracks_with_normal_detections(detections, stamp_sec)
        self.publish_track_markers(pose_array.header.stamp)
        self.publish_track_states(stamp_sec)

    def touched_centroid_callback(self, pose_array: PoseArray) -> None:
        detections = self.pose_array_to_detections(pose_array)
        if not detections:
            return

        stamp_sec = self.stamp_to_sec(pose_array)
        self.ensure_initialization_window(stamp_sec)
        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = pose_array.header.stamp

        self.update_tracks_with_touched_detections(detections, stamp_sec)
        self.publish_track_markers(pose_array.header.stamp)
        self.publish_track_states(stamp_sec)

    def stable_track_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = [item for item in get_payload_list(payload, 'tracks') if isinstance(item, dict)]
        stamp_data = get_payload_dict(payload, 'stamp')
        stamp_sec = self.stable_stamp_to_sec(stamp_data)
        frame_id = str(payload.get('frame_id', '')).strip()
        self.ensure_initialization_window(stamp_sec)

        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = Time(
            sec=as_int(stamp_data.get('sec', 0)),
            nanosec=as_int(stamp_data.get('nanosec', 0)),
        )
        self.last_source_frame_id = frame_id

        self.sync_tracks_from_stable_tracks(tracks, stamp_sec, frame_id)
        self.publish_track_markers(self.last_stamp_msg)
        self.publish_track_states(stamp_sec)

    def ensure_initialization_window(self, stamp_sec: float) -> None:
        if stamp_sec <= 0.0:
            return
        if self.initialization_start_sec is not None:
            return
        self.initialization_start_sec = stamp_sec
        self.initialization_end_sec = stamp_sec + self.assignment_initialization_duration_sec
        self.get_logger().info(
            f'Assignment initialization window started at {stamp_sec:.3f}s and ends at '
            f'{self.initialization_end_sec:.3f}s'
        )

    def update_tracks_with_normal_detections(self, detections: List[np.ndarray], stamp_sec: float) -> None:
        if not self.tracks:
            for detection in detections:
                self.create_track(detection, stamp_sec)
            return

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))
        candidate_matches: List[Tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track.centroid - detection))
                if distance <= self.max_match_distance:
                    candidate_matches.append((distance, track_id, detection_index))

        candidate_matches.sort(key=lambda item: item[0])

        for _, track_id, detection_index in candidate_matches:
            if track_id not in unmatched_track_ids:
                continue
            if detection_index not in unmatched_detection_indices:
                continue

            self.update_track(track_id, detections[detection_index], stamp_sec, update_alpha=1.0)
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        for track_id in list(unmatched_track_ids):
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track.missed_updates += 1
            if track.missed_updates > self.max_missed_updates:
                del self.tracks[track_id]

        for detection_index in sorted(unmatched_detection_indices):
            self.create_track(detections[detection_index], stamp_sec)

    def update_tracks_with_touched_detections(self, detections: List[np.ndarray], stamp_sec: float) -> None:
        if not self.tracks:
            if self.create_tracks_from_touched:
                for detection in detections:
                    self.create_track(detection, stamp_sec)
            return

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))
        candidate_matches: List[Tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track.centroid - detection))
                if distance <= self.touched_match_distance:
                    candidate_matches.append((distance, track_id, detection_index))

        candidate_matches.sort(key=lambda item: item[0])

        for _, track_id, detection_index in candidate_matches:
            if track_id not in unmatched_track_ids:
                continue
            if detection_index not in unmatched_detection_indices:
                continue

            self.update_track(
                track_id,
                detections[detection_index],
                stamp_sec,
                update_alpha=self.touched_update_alpha,
            )
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        if self.create_tracks_from_touched:
            for detection_index in sorted(unmatched_detection_indices):
                self.create_track(detections[detection_index], stamp_sec)

    def create_track(self, detection: np.ndarray, stamp_sec: float) -> None:
        track = Track(
            track_id=self.next_track_id,
            centroid=detection.copy(),
            velocity=np.zeros(3, dtype=np.float32),
            age=1,
            missed_updates=0,
            last_stamp_sec=stamp_sec,
            barcode_id='',
            hit_count=1,
            frame_id=self.last_source_frame_id or self.target_frame,
            motion_state=self.MOTION_NEWLY_APPEARED,
            first_seen_sec=stamp_sec,
            last_seen_sec=stamp_sec,
            last_confirmed_sec=stamp_sec,
            last_motion_state_change_sec=stamp_sec,
            identity_confidence=1.0,
            identity_state='new',
            last_strict_identity_match=True,
        )
        self.update_track_assignment_state(track)
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def update_track(self, track_id: int, detection: np.ndarray, stamp_sec: float, update_alpha: float) -> None:
        track = self.tracks[track_id]
        previous_centroid = track.centroid.copy()
        previous_motion_state = track.motion_state
        alpha = min(max(update_alpha, 0.0), 1.0)
        blended_centroid = track.centroid * (1.0 - alpha) + detection * alpha
        dt = max(stamp_sec - track.last_stamp_sec, 1e-3)
        track.velocity = (blended_centroid - track.centroid) / dt
        track.centroid = blended_centroid.astype(np.float32)
        track.age += 1
        track.hit_count = max(track.hit_count + 1, track.age)
        track.missed_updates = 0
        track.source_missed_count = 0
        track.last_stamp_sec = stamp_sec
        track.last_seen_sec = stamp_sec
        track.last_confirmed_sec = stamp_sec
        track.frame_id = self.last_source_frame_id or self.target_frame
        track.state = 'confirmed'
        self.motion_logic.update_confirmed_motion_state(
            track,
            previous_motion_state=previous_motion_state,
            displacement_m=float(np.linalg.norm(track.centroid - previous_centroid)),
            stamp_sec=stamp_sec,
        )
        self.update_track_assignment_state(track)

    def sync_tracks_from_stable_tracks(self, items: List[dict], stamp_sec: float, frame_id: str) -> None:
        previous_tracks = self.tracks
        previous_candidates = self.identity_collect_recovery_candidates(previous_tracks, stamp_sec)
        observations = self.normalize_stable_track_items(items)
        accepted_tracks: Dict[int, Track] = {}
        accepted_track_ids = set()
        consumed_source_ids = set()

        source_track_pool = dict(previous_candidates)
        source_track_pool.update(previous_tracks)
        source_to_track_id = {
            track.source_track_id: track_id
            for track_id, track in source_track_pool.items()
            if track.source_track_id > 0
        }

        for observation in observations:
            previous_track_id = source_to_track_id.get(observation.source_track_id)
            previous_track = source_track_pool.get(previous_track_id) if previous_track_id is not None else None
            if previous_track is None:
                continue
            if self.stable_track_identity_continuity_broken(
                previous_track=previous_track,
                new_position=observation.center,
                state=observation.state,
                displacement_m=float(np.linalg.norm(observation.center - previous_track.centroid)),
            ):
                continue

            accepted_tracks[previous_track.track_id] = self.build_track_from_observation(
                observation=observation,
                persistent_track_id=previous_track.track_id,
                stamp_sec=stamp_sec,
                frame_id=frame_id,
                previous_track=previous_track,
                identity_state='direct' if observation.state == 'confirmed' else 'lost',
                identity_confidence=1.0,
                strict_identity_match=True,
                preserve_barcode=True,
            )
            accepted_track_ids.add(previous_track.track_id)
            consumed_source_ids.add(observation.source_track_id)

        remaining_observations = [
            observation
            for observation in observations
            if observation.source_track_id not in consumed_source_ids
        ]

        recovery_matches = self.resolve_identity_recoveries(
            observations=remaining_observations,
            candidates=previous_candidates,
            accepted_track_ids=accepted_track_ids,
            stamp_sec=stamp_sec,
        )

        for observation in remaining_observations:
            match = recovery_matches.get(observation.source_track_id)
            if match is not None:
                previous_track = previous_candidates.get(match.track_id)
                if previous_track is None:
                    continue
                preserve_barcode = self.should_preserve_barcode_for_recovery(
                    previous_track=previous_track,
                    strict_match=match.strict_match,
                    accepted_tracks=accepted_tracks,
                )
                identity_state = 'recovered_strict' if match.strict_match else 'recovered_weak'
                accepted_tracks[previous_track.track_id] = self.build_track_from_observation(
                    observation=observation,
                    persistent_track_id=previous_track.track_id,
                    stamp_sec=stamp_sec,
                    frame_id=frame_id,
                    previous_track=previous_track,
                    identity_state=identity_state,
                    identity_confidence=match.identity_confidence,
                    strict_identity_match=match.strict_match,
                    preserve_barcode=preserve_barcode,
                )
                accepted_track_ids.add(previous_track.track_id)
                consumed_source_ids.add(observation.source_track_id)
                self.log_identity_event(
                    f'Recovered persistent track T{previous_track.track_id} from source '
                    f'T{observation.source_track_id}, distance={match.distance_m:.2f}m, '
                    f'dt={max(stamp_sec - previous_track.last_seen_sec, 0.0):.2f}s'
                )
                continue

            new_track = self.build_track_from_observation(
                observation=observation,
                persistent_track_id=self.next_track_id,
                stamp_sec=stamp_sec,
                frame_id=frame_id,
                previous_track=None,
                identity_state='new',
                identity_confidence=1.0,
                strict_identity_match=True,
                preserve_barcode=False,
            )
            accepted_tracks[new_track.track_id] = new_track
            accepted_track_ids.add(new_track.track_id)
            self.next_track_id += 1

        self.identity_lost_tracks = self.build_identity_lost_tracks(
            previous_tracks=previous_tracks,
            previous_candidates=previous_candidates,
            accepted_track_ids=accepted_track_ids,
            stamp_sec=stamp_sec,
            frame_id=frame_id,
        )

        if self.stable_track_preserve_missing_tracks:
            for track_id, track in self.identity_lost_tracks.items():
                accepted_tracks[track_id] = track

        self.tracks = accepted_tracks
        if self.tracks:
            self.next_track_id = max(self.next_track_id, max(self.tracks.keys()) + 1)

    def normalize_stable_track_items(self, items: List[dict]) -> List[SourceTrackObservation]:
        observations: List[SourceTrackObservation] = []
        seen_source_ids = set()

        for item in items:
            source_track_id = as_int(item.get('track_id', 0))
            if source_track_id <= 0 or source_track_id in seen_source_ids:
                continue

            state = str(item.get('state', '')).strip().lower()
            if state == 'tentative':
                continue
            if state == 'lost' and not self.stable_track_include_lost:
                continue
            if state not in ('confirmed', 'lost'):
                continue

            seen_source_ids.add(source_track_id)
            observations.append(
                SourceTrackObservation(
                    source_track_id=source_track_id,
                    state=state,
                    class_id=as_int(item.get('class_id', -1), default=-1),
                    class_name=str(item.get('class_name', '')),
                    confidence=as_float(item.get('confidence', 0.0)),
                    center=np.array(
                        [
                            as_float(item.get('center_x', 0.0)),
                            as_float(item.get('center_y', 0.0)),
                            as_float(item.get('center_z', 0.0)),
                        ],
                        dtype=np.float32,
                    ),
                    yaw=as_float(item.get('yaw', 0.0)),
                    length=as_float(item.get('length', 0.0)),
                    width=as_float(item.get('width', 0.0)),
                    height=as_float(item.get('height', 0.0)),
                    hit_count=max(as_int(item.get('hit_count', 1), default=1), 1),
                    missed_count=max(as_int(item.get('missed_count', 0), default=0), 0),
                    raw_item=item,
                )
            )

        return observations

    def build_track_from_observation(
        self,
        observation: SourceTrackObservation,
        persistent_track_id: int,
        stamp_sec: float,
        frame_id: str,
        previous_track: Track | None,
        identity_state: str,
        identity_confidence: float,
        strict_identity_match: bool,
        preserve_barcode: bool,
    ) -> Track:
        velocity = np.zeros(3, dtype=np.float32)
        barcode_id = ''
        previous_motion_state = self.MOTION_NEWLY_APPEARED
        first_seen_sec = stamp_sec
        last_confirmed_sec = stamp_sec if observation.state == 'confirmed' else 0.0
        static_streak = 0
        moving_streak = 0
        lost_transition_count = 0
        occluded_transition_count = 0
        reappeared_count = 0
        last_motion_state_change_sec = stamp_sec
        age = max(observation.hit_count, 1)
        hit_count = max(observation.hit_count, 1)
        missed_updates = observation.missed_count if observation.state != 'confirmed' else 0
        source_missed_count = observation.missed_count
        identity_recovered_count = 0
        last_source_track_id = observation.source_track_id

        if previous_track is not None:
            dt = max(stamp_sec - previous_track.last_stamp_sec, 1e-3)
            velocity = ((observation.center - previous_track.centroid) / dt).astype(np.float32)
            barcode_id = previous_track.barcode_id if preserve_barcode else ''
            previous_motion_state = previous_track.motion_state
            first_seen_sec = previous_track.first_seen_sec or stamp_sec
            last_confirmed_sec = previous_track.last_confirmed_sec
            static_streak = previous_track.static_streak
            moving_streak = previous_track.moving_streak
            lost_transition_count = previous_track.lost_transition_count
            occluded_transition_count = previous_track.occluded_transition_count
            reappeared_count = previous_track.reappeared_count
            last_motion_state_change_sec = previous_track.last_motion_state_change_sec
            age = max(previous_track.age + 1, observation.hit_count, 1)
            identity_recovered_count = previous_track.identity_recovered_count
            last_source_track_id = observation.source_track_id
            if observation.state == 'confirmed':
                hit_count = max(previous_track.hit_count + 1, observation.hit_count, 1)
                missed_updates = 0
            else:
                hit_count = max(previous_track.hit_count, observation.hit_count, 1)
                missed_updates = max(previous_track.missed_updates + 1, observation.missed_count)
            if identity_state.startswith('recovered'):
                identity_recovered_count += 1
                if previous_track.source_track_id > 0:
                    last_source_track_id = previous_track.source_track_id

        track = Track(
            track_id=persistent_track_id,
            centroid=observation.center.copy(),
            velocity=velocity,
            age=age,
            missed_updates=missed_updates,
            last_stamp_sec=stamp_sec,
            barcode_id=barcode_id,
            class_id=observation.class_id,
            class_name=observation.class_name,
            state=observation.state,
            confidence=observation.confidence,
            yaw=observation.yaw,
            length=observation.length,
            width=observation.width,
            height=observation.height,
            hit_count=hit_count,
            source_missed_count=source_missed_count,
            frame_id=frame_id,
            motion_state=previous_motion_state,
            static_streak=static_streak,
            moving_streak=moving_streak,
            first_seen_sec=first_seen_sec,
            last_seen_sec=stamp_sec,
            last_confirmed_sec=last_confirmed_sec,
            lost_transition_count=lost_transition_count,
            occluded_transition_count=occluded_transition_count,
            reappeared_count=reappeared_count,
            last_motion_state_change_sec=last_motion_state_change_sec,
            source_track_id=observation.source_track_id,
            last_source_track_id=last_source_track_id,
            identity_recovered_count=identity_recovered_count,
            identity_confidence=max(0.0, min(identity_confidence, 1.0)),
            identity_state=identity_state,
            last_strict_identity_match=bool(strict_identity_match),
        )

        if observation.state == 'confirmed':
            track.last_confirmed_sec = stamp_sec
            displacement_m = 0.0 if previous_track is None else float(np.linalg.norm(observation.center - previous_track.centroid))
            if previous_track is not None and (previous_track.state == 'lost' or identity_state.startswith('recovered')):
                track.reappeared_count += 1
            self.motion_logic.update_confirmed_motion_state(
                track,
                previous_motion_state=previous_motion_state,
                displacement_m=displacement_m,
                stamp_sec=stamp_sec,
            )
        else:
            if previous_track is None or previous_track.state != 'lost':
                track.lost_transition_count += 1
            track.identity_state = 'lost'
            self.motion_logic.update_lost_motion_state(track, stamp_sec=stamp_sec)

        self.update_track_assignment_state(track)
        return track

    def resolve_identity_recoveries(
        self,
        observations: List[SourceTrackObservation],
        candidates: Dict[int, Track],
        accepted_track_ids: set[int],
        stamp_sec: float,
    ) -> Dict[int, RecoveryCandidate]:
        if not self.identity_recovery_enabled or self.identity_recovery_gate_m <= 0.0:
            return {}

        proposals_by_source: Dict[int, List[RecoveryCandidate]] = {}
        proposals_by_track: Dict[int, List[Tuple[int, RecoveryCandidate]]] = {}

        for observation in observations:
            if observation.state != 'confirmed':
                continue

            source_candidates: List[RecoveryCandidate] = []
            for track_id, candidate_track in candidates.items():
                if track_id in accepted_track_ids:
                    continue
                proposal = self.build_recovery_candidate(
                    observation=observation,
                    candidate_track=candidate_track,
                    stamp_sec=stamp_sec,
                )
                if proposal is None:
                    continue
                source_candidates.append(proposal)
                proposals_by_track.setdefault(track_id, []).append((observation.source_track_id, proposal))

            if source_candidates:
                proposals_by_source[observation.source_track_id] = sorted(source_candidates, key=lambda item: item.score)

        unambiguous_sources: Dict[int, RecoveryCandidate] = {}
        for source_track_id, source_candidates in proposals_by_source.items():
            best = source_candidates[0]
            second = source_candidates[1] if len(source_candidates) > 1 else None
            if second is not None and (second.score - best.score) < self.identity_recovery_max_ambiguous_score_gap:
                rejected_ids = f'T{best.track_id}/T{second.track_id}'
                self.log_identity_event(
                    f'Rejected recovery for source T{source_track_id}: ambiguous candidates {rejected_ids}'
                )
                continue
            unambiguous_sources[source_track_id] = best

        blocked_track_ids = set()
        for track_id, candidate_entries in proposals_by_track.items():
            candidate_entries.sort(key=lambda item: item[1].score)
            if len(candidate_entries) < 2:
                continue
            best_source_id, best = candidate_entries[0]
            second_source_id, second = candidate_entries[1]
            if (second.score - best.score) < self.identity_recovery_max_ambiguous_score_gap:
                blocked_track_ids.add(track_id)
                self.log_identity_event(
                    f'Rejected recovery for candidate T{track_id}: ambiguous sources '
                    f'T{best_source_id}/T{second_source_id}'
                )

        assignments: Dict[int, RecoveryCandidate] = {}
        used_track_ids = set()
        for source_track_id, proposal in sorted(
            unambiguous_sources.items(),
            key=lambda item: item[1].score,
        ):
            if proposal.track_id in blocked_track_ids or proposal.track_id in used_track_ids:
                continue
            assignments[source_track_id] = proposal
            used_track_ids.add(proposal.track_id)

        return assignments

    def build_recovery_candidate(
        self,
        observation: SourceTrackObservation,
        candidate_track: Track,
        stamp_sec: float,
    ) -> RecoveryCandidate | None:
        if not self.is_class_compatible(observation.class_name, candidate_track.class_name):
            return None

        dt = max(stamp_sec - candidate_track.last_seen_sec, 0.0)
        if self.identity_recovery_ttl_sec >= 0.0 and dt > self.identity_recovery_ttl_sec:
            return None

        predicted_position = candidate_track.centroid
        if self.identity_recovery_use_velocity_prediction:
            predicted_position = candidate_track.centroid + candidate_track.velocity * dt

        distance_m = float(np.linalg.norm(observation.center - predicted_position))
        if distance_m > self.identity_recovery_gate_m:
            return None

        score = distance_m
        if observation.length > 0.0 and candidate_track.length > 0.0:
            score += abs(observation.length - candidate_track.length) * 0.20
        if observation.width > 0.0 and candidate_track.width > 0.0:
            score += abs(observation.width - candidate_track.width) * 0.20
        if observation.height > 0.0 and candidate_track.height > 0.0:
            score += abs(observation.height - candidate_track.height) * 0.10
        if observation.yaw != 0.0 or candidate_track.yaw != 0.0:
            score += self.angular_difference_rad(observation.yaw, candidate_track.yaw) * 0.10

        identity_confidence = max(0.0, 1.0 - min(distance_m / max(self.identity_recovery_gate_m, 1e-6), 1.0))
        return RecoveryCandidate(
            track_id=candidate_track.track_id,
            score=score,
            distance_m=distance_m,
            identity_confidence=identity_confidence,
            strict_match=distance_m <= self.identity_recovery_strict_gate_m,
        )

    def build_track_from_stable_item(
        self,
        item: dict,
        stamp_sec: float,
        frame_id: str,
        previous_track: Track | None,
    ) -> Track:
        del item, stamp_sec, frame_id, previous_track
        raise NotImplementedError('Use build_track_from_observation() for stable-track sync.')

    def stable_track_identity_continuity_broken(
        self,
        previous_track: Track,
        new_position: np.ndarray,
        state: str,
        displacement_m: float,
    ) -> bool:
        if not self.use_stable_tracks_input:
            return False
        if self.stable_track_discontinuity_distance_m <= 0.0:
            return False
        if state != 'confirmed':
            return False
        if previous_track.state == 'lost':
            return False
        return displacement_m > self.stable_track_discontinuity_distance_m

    def build_disappeared_track(self, previous_track: Track, stamp_sec: float, frame_id: str) -> Track | None:
        if self.disappeared_retention_sec >= 0.0:
            disappeared_age_sec = stamp_sec - previous_track.last_seen_sec
            if disappeared_age_sec > self.disappeared_retention_sec:
                return None

        return Track(
            track_id=previous_track.track_id,
            centroid=previous_track.centroid.copy(),
            velocity=previous_track.velocity.copy(),
            age=previous_track.age,
            missed_updates=previous_track.missed_updates + 1,
            last_stamp_sec=stamp_sec,
            barcode_id=previous_track.barcode_id,
            class_id=previous_track.class_id,
            class_name=previous_track.class_name,
            state='deleted',
            confidence=previous_track.confidence,
            yaw=previous_track.yaw,
            length=previous_track.length,
            width=previous_track.width,
            height=previous_track.height,
            hit_count=previous_track.hit_count,
            source_missed_count=previous_track.source_missed_count + 1,
            frame_id=frame_id or previous_track.frame_id,
            motion_state=self.MOTION_DISAPPEARED,
            static_streak=previous_track.static_streak,
            moving_streak=previous_track.moving_streak,
            first_seen_sec=previous_track.first_seen_sec,
            last_seen_sec=previous_track.last_seen_sec,
            last_confirmed_sec=previous_track.last_confirmed_sec,
            lost_transition_count=previous_track.lost_transition_count,
            occluded_transition_count=previous_track.occluded_transition_count,
            reappeared_count=previous_track.reappeared_count,
            last_motion_state_change_sec=previous_track.last_motion_state_change_sec,
            source_track_id=previous_track.source_track_id,
            last_source_track_id=previous_track.last_source_track_id or previous_track.source_track_id,
            identity_recovered_count=previous_track.identity_recovered_count,
            identity_confidence=max(previous_track.identity_confidence * 0.5, 0.0),
            identity_state='disappeared',
            last_strict_identity_match=previous_track.last_strict_identity_match,
            marriage_state=previous_track.marriage_state,
        )

    def build_identity_lost_tracks(
        self,
        previous_tracks: Dict[int, Track],
        previous_candidates: Dict[int, Track],
        accepted_track_ids: set[int],
        stamp_sec: float,
        frame_id: str,
    ) -> Dict[int, Track]:
        lost_tracks: Dict[int, Track] = {}
        merged_previous = dict(previous_candidates)
        merged_previous.update(previous_tracks)

        for track_id, previous_track in merged_previous.items():
            if track_id in accepted_track_ids:
                continue
            disappeared_track = self.build_disappeared_track(previous_track, stamp_sec, frame_id)
            if disappeared_track is None:
                continue
            if self.identity_recovery_ttl_sec >= 0.0:
                age_sec = stamp_sec - previous_track.last_seen_sec
                if age_sec > self.identity_recovery_ttl_sec:
                    continue
            lost_tracks[track_id] = disappeared_track

        return lost_tracks

    def identity_collect_recovery_candidates(self, previous_tracks: Dict[int, Track], stamp_sec: float) -> Dict[int, Track]:
        candidates = dict(self.identity_lost_tracks)
        for track_id, track in previous_tracks.items():
            if self.identity_recovery_ttl_sec >= 0.0 and (stamp_sec - track.last_seen_sec) > self.identity_recovery_ttl_sec:
                continue
            candidates[track_id] = track
        return candidates

    def should_preserve_barcode_for_recovery(
        self,
        previous_track: Track,
        strict_match: bool,
        accepted_tracks: Dict[int, Track],
    ) -> bool:
        barcode_id = previous_track.barcode_id.strip()
        if not barcode_id:
            return True
        if self.identity_preserve_uid_only_on_strict_match and not strict_match:
            self.log_identity_event(
                f'Cleared barcode for T{previous_track.track_id} because recovery was not strict enough'
            )
            return False
        if self.barcode_conflicts_with_active_track(barcode_id, accepted_tracks, previous_track.track_id):
            self.log_identity_event(
                f'Cleared barcode for T{previous_track.track_id} because barcode "{barcode_id}" is already active'
            )
            return False
        return True

    def barcode_conflicts_with_active_track(
        self,
        barcode_id: str,
        accepted_tracks: Dict[int, Track],
        excluded_track_id: int,
    ) -> bool:
        if not barcode_id:
            return False
        for track_id, track in accepted_tracks.items():
            if track_id == excluded_track_id:
                continue
            if track.barcode_id and track.barcode_id == barcode_id:
                return True
        return False

    @staticmethod
    def rack_group_name(class_name: str) -> str:
        normalized = class_name.strip().lower()
        if normalized in {'rack_side_visible', 'rack_top_visible'}:
            return 'rack'
        if normalized == 'rack_entrance_visible':
            return 'rack_entrance_visible'
        return normalized

    def is_class_compatible(self, first: str, second: str) -> bool:
        first_group = self.rack_group_name(first)
        second_group = self.rack_group_name(second)
        if not first_group or not second_group:
            return first_group == second_group
        if first_group == 'rack_entrance_visible' or second_group == 'rack_entrance_visible':
            return first_group == second_group
        return first_group == second_group

    @staticmethod
    def angular_difference_rad(first: float, second: float) -> float:
        diff = float(first) - float(second)
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return abs(diff)

    def log_identity_event(self, message: str) -> None:
        if self.identity_recovery_log_events:
            self.get_logger().info(message)

    def is_initialization_mode(self, stamp_sec: float) -> bool:
        if self.initialization_end_sec is None:
            return self.assignment_initialization_duration_sec > 0.0
        return stamp_sec <= self.initialization_end_sec

    def compute_marriage_state(self, track: Track) -> str:
        if track.barcode_id.strip():
            return self.MARRIAGE_ASSIGNED

        first_seen_sec = float(track.first_seen_sec)
        initialization_end_sec = self.initialization_end_sec
        if initialization_end_sec is None:
            return self.MARRIAGE_KNOWN_EXISTING
        if first_seen_sec <= initialization_end_sec:
            return self.MARRIAGE_KNOWN_EXISTING
        return self.MARRIAGE_UNASSIGNED_NEW

    def evaluate_marriage_eligibility(
        self,
        track: Track,
        stamp_sec: float,
    ) -> tuple[bool, str, list[str]]:
        blockers: list[str] = []
        marriage_state = self.compute_marriage_state(track)

        if marriage_state == self.MARRIAGE_KNOWN_EXISTING:
            blockers.append('known_existing')
        if marriage_state == self.MARRIAGE_ASSIGNED or track.barcode_id.strip():
            blockers.append('already_assigned')
        if track.state == 'lost':
            blockers.append('track_lost')
        elif track.state != 'confirmed':
            blockers.append('not_confirmed')

        if track.motion_state == self.MOTION_MOVING:
            blockers.append('motion_state_moving')
        elif track.motion_state in (self.MOTION_OCCLUDED, self.MOTION_DISAPPEARED):
            blockers.append('track_not_visible')
        elif track.motion_state != self.MOTION_STATIC:
            blockers.append('motion_state_moving')

        if track.identity_state == 'recovered_weak':
            blockers.append('identity_recovered_weak')
        elif track.identity_state not in self.ASSIGNMENT_ALLOWED_IDENTITY_STATES:
            blockers.append('identity_not_safe')

        if max(stamp_sec - track.last_seen_sec, 0.0) > max(self.occluded_timeout_sec, 0.0) and 'track_not_visible' not in blockers:
            blockers.append('track_not_visible')

        if not blockers:
            return True, 'eligible', []

        unique_blockers = list(dict.fromkeys(blockers))
        if 'already_assigned' in unique_blockers:
            reason = 'already_assigned'
        elif 'known_existing' in unique_blockers:
            reason = 'known_existing'
        elif 'track_lost' in unique_blockers:
            reason = 'track_lost'
        elif 'track_not_visible' in unique_blockers:
            reason = 'track_not_visible'
        elif 'not_confirmed' in unique_blockers:
            reason = 'not_confirmed'
        elif 'motion_state_moving' in unique_blockers:
            reason = 'track_moving'
        elif 'identity_recovered_weak' in unique_blockers:
            reason = 'identity_recovered_weak'
        else:
            reason = 'identity_not_safe'
        return False, reason, unique_blockers

    def update_track_assignment_state(self, track: Track) -> None:
        track.marriage_state = self.compute_marriage_state(track)

    def barcode_in_use(self, barcode_id: str, exclude_track_id: int = 0) -> bool:
        normalized = barcode_id.strip()
        if not normalized:
            return False
        for track_id, track in self.tracks.items():
            if track_id == exclude_track_id:
                continue
            if track.barcode_id.strip() == normalized:
                return True
        return False

    def assignment_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = as_int(payload.get('track_id', 0))
        barcode_id = str(payload.get('barcode_id', ''))

        if track_id <= 0 or not barcode_id:
            return

        track = self.tracks.get(track_id)
        if track is None:
            self.get_logger().warning(
                f'Received assignment for unknown track_id={track_id}.',
                throttle_duration_sec=5.0,
            )
            return

        if self.barcode_in_use(barcode_id, exclude_track_id=track_id):
            self.get_logger().warning(
                f'Rejected assignment for T{track_id}: barcode "{barcode_id}" is already active.',
                throttle_duration_sec=2.0,
            )
            return

        if track.barcode_id == barcode_id:
            return

        if track.barcode_id.strip() and track.barcode_id != barcode_id:
            self.get_logger().warning(
                f'Rejected assignment for T{track_id}: already assigned to "{track.barcode_id}".',
                throttle_duration_sec=2.0,
            )
            return

        stamp_sec = self.current_stamp_sec()
        eligible, reason, blockers = self.evaluate_marriage_eligibility(track, stamp_sec)
        if not eligible:
            self.get_logger().warning(
                f'Rejected assignment for T{track_id}: reason={reason}, blockers={blockers}.',
                throttle_duration_sec=2.0,
            )
            return

        track.barcode_id = barcode_id
        self.update_track_assignment_state(track)
        self.get_logger().info(f'Assigned barcode "{barcode_id}" to track T{track_id}.')
        self.publish_track_states(stamp_sec)
        self.publish_track_markers(self.current_stamp_msg())

    def remove_track_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = as_int(payload.get('track_id', 0))
        barcode_id = str(payload.get('barcode_id', ''))

        if track_id <= 0:
            return

        track = self.tracks.get(track_id)
        if track is None:
            self.get_logger().warning(
                f'Received remove request for unknown track_id={track_id}.',
                throttle_duration_sec=5.0,
            )
            return

        if barcode_id and track.barcode_id and track.barcode_id != barcode_id:
            self.get_logger().warning(
                f'Remove request barcode mismatch for T{track_id}: '
                f'track="{track.barcode_id}" request="{barcode_id}".',
                throttle_duration_sec=5.0,
            )
            return

        del self.tracks[track_id]
        self.identity_lost_tracks.pop(track_id, None)
        self.get_logger().info(f'Removed track T{track_id} with barcode "{barcode_id}".')
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())

    def clear_tracks(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.tracks.clear()
        self.identity_lost_tracks.clear()
        self.next_track_id = 1
        self.initialization_start_sec = None
        self.initialization_end_sec = None
        response.success = True
        response.message = 'All tracks cleared.'
        self.get_logger().info(response.message)
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())
        return response

    def publish_track_states(self, stamp_sec: float) -> None:
        payload = build_track_states_payload(self.tracks, stamp_sec)
        payload['initialization_mode'] = self.is_initialization_mode(stamp_sec)
        if self.initialization_end_sec is not None:
            payload['initialization_end_sec'] = float(self.initialization_end_sec)
        for item in payload.get('tracks', []):
            if not isinstance(item, dict):
                continue
            track_id = as_int(item.get('track_id', 0))
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track.marriage_state = self.compute_marriage_state(track)
            eligible, reason, blockers = self.evaluate_marriage_eligibility(track, stamp_sec)
            item['marriage_state'] = track.marriage_state
            item['is_marriage_eligible'] = bool(eligible)
            item['eligibility_reason'] = reason
            item['eligibility_blockers'] = blockers
        self.track_state_pub.publish(make_string_message(payload))

    def publish_track_markers(self, stamp) -> None:
        marker_frame = self.last_source_frame_id or self.target_frame
        marker_tracks = self.tracks
        if self.use_stable_tracks_input and not self.stable_track_publish_disappeared_markers:
            marker_tracks = {
                track_id: track
                for track_id, track in self.tracks.items()
                if track.state in ('confirmed', 'lost')
            }
        marker_array = self.track_marker_factory.build_marker_array(marker_tracks, stamp, marker_frame)
        self.track_marker_pub.publish(marker_array)

    def pose_array_to_detections(self, pose_array: PoseArray) -> List[np.ndarray]:
        return [
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
            for pose in pose_array.poses
        ]

    @staticmethod
    def stamp_to_sec(pose_array: PoseArray) -> float:
        return float(pose_array.header.stamp.sec) + float(pose_array.header.stamp.nanosec) * 1e-9

    def current_stamp_sec(self) -> float:
        if self.last_stamp_sec > 0.0:
            return self.last_stamp_sec

        now_msg = self.get_clock().now().to_msg()
        return float(now_msg.sec) + float(now_msg.nanosec) * 1e-9

    def current_stamp_msg(self):
        if self.last_stamp_msg is not None:
            return self.last_stamp_msg
        return self.get_clock().now().to_msg()

    @staticmethod
    def stable_stamp_to_sec(stamp_data: dict) -> float:
        return as_float(stamp_data.get('sec', 0)) + as_float(stamp_data.get('nanosec', 0)) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
