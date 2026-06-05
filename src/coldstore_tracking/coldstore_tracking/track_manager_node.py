from __future__ import annotations

from builtin_interfaces.msg import Time
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .event_utils import build_track_states_payload, make_string_message, parse_string_message
from .tracking_types import Track


class TrackManagerNode(Node):
    MOTION_STATIC = 'static'
    MOTION_MOVING = 'moving'
    MOTION_OCCLUDED = 'occluded'
    MOTION_DISAPPEARED = 'disappeared'
    MOTION_NEWLY_APPEARED = 'newly_appeared'
    ACTIVE_MOTION_STATES = {MOTION_STATIC, MOTION_MOVING, MOTION_NEWLY_APPEARED}

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
        self.declare_parameter('newly_appeared_hold_sec', 1.5)
        self.declare_parameter('motion_static_threshold_m', 0.03)
        self.declare_parameter('motion_moving_threshold_m', 0.12)
        self.declare_parameter('motion_required_updates', 3)
        self.declare_parameter('occluded_timeout_sec', 3.0)
        self.declare_parameter('disappeared_retention_sec', 4.0)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_match_distance = float(self.get_parameter('max_match_distance').value)
        self.touched_match_distance = float(self.get_parameter('touched_match_distance').value)
        self.touched_update_alpha = float(self.get_parameter('touched_update_alpha').value)
        self.max_missed_updates = int(self.get_parameter('max_missed_updates').value)
        self.create_tracks_from_touched = bool(self.get_parameter('create_tracks_from_touched').value)
        self.use_stable_tracks_input = bool(self.get_parameter('use_stable_tracks_input').value)
        self.stable_track_topic = str(self.get_parameter('stable_track_topic').value)
        self.stable_track_include_lost = bool(self.get_parameter('stable_track_include_lost').value)
        self.newly_appeared_hold_sec = float(self.get_parameter('newly_appeared_hold_sec').value)
        self.motion_static_threshold_m = float(self.get_parameter('motion_static_threshold_m').value)
        self.motion_moving_threshold_m = float(self.get_parameter('motion_moving_threshold_m').value)
        self.motion_required_updates = max(int(self.get_parameter('motion_required_updates').value), 1)
        self.occluded_timeout_sec = float(self.get_parameter('occluded_timeout_sec').value)
        self.disappeared_retention_sec = float(self.get_parameter('disappeared_retention_sec').value)

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_stamp_sec = 0.0
        self.last_stamp_msg = None
        self.last_source_frame_id = ''

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
            f'include_lost={self.stable_track_include_lost}'
        )
        self.get_logger().info(
            f'Motion logic: hold_sec={self.newly_appeared_hold_sec:.2f}, '
            f'static_threshold_m={self.motion_static_threshold_m:.3f}, '
            f'moving_threshold_m={self.motion_moving_threshold_m:.3f}, '
            f'required_updates={self.motion_required_updates}, '
            f'occluded_timeout_sec={self.occluded_timeout_sec:.2f}, '
            f'disappeared_retention_sec={self.disappeared_retention_sec:.2f}'
        )

    def centroid_callback(self, pose_array: PoseArray) -> None:
        detections = self.pose_array_to_detections(pose_array)
        stamp_sec = self.stamp_to_sec(pose_array)
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
        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = pose_array.header.stamp

        self.update_tracks_with_touched_detections(detections, stamp_sec)
        self.publish_track_markers(pose_array.header.stamp)
        self.publish_track_states(stamp_sec)

    def stable_track_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        tracks = payload.get('tracks', [])
        stamp_data = payload.get('stamp', {})
        stamp_sec = self.stable_stamp_to_sec(stamp_data)
        frame_id = str(payload.get('frame_id', '')).strip()

        self.last_stamp_sec = stamp_sec
        self.last_stamp_msg = Time(
            sec=int(stamp_data.get('sec', 0)),
            nanosec=int(stamp_data.get('nanosec', 0)),
        )
        self.last_source_frame_id = frame_id

        self.sync_tracks_from_stable_tracks(tracks, stamp_sec, frame_id)
        self.publish_track_markers(self.last_stamp_msg)
        self.publish_track_states(stamp_sec)

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
        )
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
        self.update_confirmed_motion_state(
            track,
            previous_motion_state=previous_motion_state,
            displacement_m=float(np.linalg.norm(track.centroid - previous_centroid)),
            stamp_sec=stamp_sec,
        )

    def sync_tracks_from_stable_tracks(self, items: List[dict], stamp_sec: float, frame_id: str) -> None:
        previous_tracks = self.tracks
        accepted_tracks: Dict[int, Track] = {}
        accepted_track_ids = set()

        for item in items:
            track_id = int(item.get('track_id', 0))
            if track_id <= 0:
                continue

            state = str(item.get('state', '')).strip().lower()
            if state == 'tentative':
                continue
            if state == 'lost' and not self.stable_track_include_lost:
                continue
            if state not in ('confirmed', 'lost'):
                continue

            previous_track = previous_tracks.get(track_id)
            accepted_tracks[track_id] = self.build_track_from_stable_item(
                item=item,
                stamp_sec=stamp_sec,
                frame_id=frame_id,
                previous_track=previous_track,
            )
            accepted_track_ids.add(track_id)

        for track_id, previous_track in previous_tracks.items():
            if track_id in accepted_track_ids:
                continue

            disappeared_track = self.build_disappeared_track(previous_track, stamp_sec, frame_id)
            if disappeared_track is not None:
                accepted_tracks[track_id] = disappeared_track

        self.tracks = accepted_tracks
        if self.tracks:
            self.next_track_id = max(self.next_track_id, max(self.tracks.keys()) + 1)

    def build_track_from_stable_item(
        self,
        item: dict,
        stamp_sec: float,
        frame_id: str,
        previous_track: Track | None,
    ) -> Track:
        position = np.array(
            [
                float(item.get('center_x', 0.0)),
                float(item.get('center_y', 0.0)),
                float(item.get('center_z', 0.0)),
            ],
            dtype=np.float32,
        )
        state = str(item.get('state', 'confirmed')).strip().lower()
        velocity = np.zeros(3, dtype=np.float32)
        barcode_id = ''
        previous_motion_state = self.MOTION_NEWLY_APPEARED
        first_seen_sec = stamp_sec
        last_confirmed_sec = stamp_sec if state == 'confirmed' else 0.0
        static_streak = 0
        moving_streak = 0
        lost_transition_count = 0
        occluded_transition_count = 0
        reappeared_count = 0
        last_motion_state_change_sec = stamp_sec

        if previous_track is not None:
            dt = max(stamp_sec - previous_track.last_stamp_sec, 1e-3)
            velocity = ((position - previous_track.centroid) / dt).astype(np.float32)
            barcode_id = previous_track.barcode_id
            previous_motion_state = previous_track.motion_state
            first_seen_sec = previous_track.first_seen_sec or stamp_sec
            last_confirmed_sec = previous_track.last_confirmed_sec
            static_streak = previous_track.static_streak
            moving_streak = previous_track.moving_streak
            lost_transition_count = previous_track.lost_transition_count
            occluded_transition_count = previous_track.occluded_transition_count
            reappeared_count = previous_track.reappeared_count
            last_motion_state_change_sec = previous_track.last_motion_state_change_sec

        track = Track(
            track_id=int(item.get('track_id', 0)),
            centroid=position,
            velocity=velocity,
            age=max(int(item.get('hit_count', 1)), 1),
            missed_updates=int(item.get('missed_count', 0)),
            last_stamp_sec=stamp_sec,
            barcode_id=barcode_id,
            class_id=int(item.get('class_id', -1)),
            class_name=str(item.get('class_name', '')),
            state=state,
            confidence=float(item.get('confidence', 0.0)),
            yaw=float(item.get('yaw', 0.0)),
            length=float(item.get('length', 0.0)),
            width=float(item.get('width', 0.0)),
            height=float(item.get('height', 0.0)),
            hit_count=int(item.get('hit_count', 0)),
            source_missed_count=int(item.get('missed_count', 0)),
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
        )

        if state == 'confirmed':
            track.last_confirmed_sec = stamp_sec
            displacement_m = 0.0
            if previous_track is not None:
                displacement_m = float(np.linalg.norm(position - previous_track.centroid))
            if previous_track is not None and previous_track.state == 'lost':
                track.reappeared_count += 1
            self.update_confirmed_motion_state(
                track,
                previous_motion_state=previous_motion_state,
                displacement_m=displacement_m,
                stamp_sec=stamp_sec,
            )
        else:
            if previous_track is None or previous_track.state != 'lost':
                track.lost_transition_count += 1
            self.update_lost_motion_state(track, stamp_sec=stamp_sec)

        return track

    def build_disappeared_track(self, previous_track: Track, stamp_sec: float, frame_id: str) -> Track | None:
        if self.disappeared_retention_sec >= 0.0:
            disappeared_age_sec = stamp_sec - previous_track.last_seen_sec
            if disappeared_age_sec > self.disappeared_retention_sec:
                return None

        return Track(
            track_id=previous_track.track_id,
            centroid=previous_track.centroid.copy(),
            velocity=np.zeros(3, dtype=np.float32),
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
        )

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

    def assignment_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = int(payload.get('track_id', 0))
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

        if track.barcode_id == barcode_id:
            return

        track.barcode_id = barcode_id
        self.get_logger().info(f'Assigned barcode "{barcode_id}" to track T{track_id}.')
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())

    def remove_track_callback(self, msg: String) -> None:
        payload = parse_string_message(msg)
        track_id = int(payload.get('track_id', 0))
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
        self.get_logger().info(f'Removed track T{track_id} with barcode "{barcode_id}".')
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())

    def clear_tracks(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self.tracks.clear()
        self.next_track_id = 1
        response.success = True
        response.message = 'All tracks cleared.'
        self.get_logger().info(response.message)
        self.publish_track_states(self.current_stamp_sec())
        self.publish_track_markers(self.current_stamp_msg())
        return response

    def publish_track_states(self, stamp_sec: float) -> None:
        payload = build_track_states_payload(self.tracks, stamp_sec)
        self.track_state_pub.publish(make_string_message(payload))

    def publish_track_markers(self, stamp) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_frame = self.last_source_frame_id or self.target_frame
        marker_id = 0
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            sphere = Marker()
            sphere.header.frame_id = marker_frame
            sphere.header.stamp = stamp
            sphere.ns = 'tracks'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(track.centroid[0])
            sphere.pose.position.y = float(track.centroid[1])
            sphere.pose.position.z = float(track.centroid[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25

            if track.state == 'lost':
                sphere.color.r = 1.0
                sphere.color.g = 0.8
                sphere.color.b = 0.0
            elif track.motion_state == self.MOTION_MOVING:
                sphere.color.r = 0.0
                sphere.color.g = 0.85
                sphere.color.b = 0.2
            elif track.motion_state == self.MOTION_NEWLY_APPEARED:
                sphere.color.r = 0.2
                sphere.color.g = 0.6
                sphere.color.b = 1.0
            elif track.motion_state == self.MOTION_DISAPPEARED:
                sphere.color.r = 0.6
                sphere.color.g = 0.6
                sphere.color.b = 0.6
            elif track.missed_updates == 0:
                sphere.color.r = 1.0
                sphere.color.g = 0.0
                sphere.color.b = 0.0
            else:
                sphere.color.r = 1.0
                sphere.color.g = 0.8
                sphere.color.b = 0.0
            sphere.color.a = 0.45 if track.motion_state == self.MOTION_DISAPPEARED else 1.0
            marker_array.markers.append(sphere)
            marker_id += 1

            text = Marker()
            text.header.frame_id = marker_frame
            text.header.stamp = stamp
            text.ns = 'track_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(track.centroid[0])
            text.pose.position.y = float(track.centroid[1])
            text.pose.position.z = float(track.centroid[2] + 0.35)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.28
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 0.0
            text.color.a = 1.0

            speed = float(np.linalg.norm(track.velocity))
            barcode = track.barcode_id if track.barcode_id else '-'
            class_label = track.class_name if track.class_name else '-'
            text.text = (
                f'T{track.track_id} {track.state}/{track.motion_state} {class_label} id:{barcode} '
                f'hit:{track.hit_count} miss:{track.source_missed_count} v:{speed:.2f}m/s'
            )
            marker_array.markers.append(text)
            marker_id += 1

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
        return float(stamp_data.get('sec', 0)) + float(stamp_data.get('nanosec', 0)) * 1e-9


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
