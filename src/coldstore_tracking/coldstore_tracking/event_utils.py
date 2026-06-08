from __future__ import annotations

import json
from typing import Any, Dict, List

from std_msgs.msg import String

from .tracking_types import Track


def make_string_message(payload: Dict[str, Any]) -> String:
    msg = String()
    msg.data = json.dumps(payload, sort_keys=True)
    return msg


def parse_string_message(msg: String) -> Dict[str, Any]:
    if not msg.data:
        return {}
    try:
        payload = json.loads(msg.data)
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def get_payload_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def get_payload_list(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_track_states_payload(tracks: Dict[int, Track], stamp_sec: float) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    frame_id = ''

    for track in sorted(tracks.values(), key=lambda item: item.track_id):
        if not frame_id and track.frame_id:
            frame_id = str(track.frame_id)
        items.append({
            'track_id': int(track.track_id),
            'barcode_id': str(track.barcode_id),
            'class_id': int(track.class_id),
            'class_name': str(track.class_name),
            'state': str(track.state),
            'motion_state': str(track.motion_state),
            'confidence': float(track.confidence),
            'x': float(track.centroid[0]),
            'y': float(track.centroid[1]),
            'z': float(track.centroid[2]),
            'yaw': float(track.yaw),
            'length': float(track.length),
            'width': float(track.width),
            'height': float(track.height),
            'vx': float(track.velocity[0]),
            'vy': float(track.velocity[1]),
            'vz': float(track.velocity[2]),
            'age': int(track.age),
            'hit_count': int(track.hit_count),
            'missed_updates': int(track.missed_updates),
            'source_missed_count': int(track.source_missed_count),
            'first_seen_sec': float(track.first_seen_sec),
            'last_seen_sec': float(track.last_seen_sec),
            'last_confirmed_sec': float(track.last_confirmed_sec),
            'lost_transition_count': int(track.lost_transition_count),
            'occluded_transition_count': int(track.occluded_transition_count),
            'reappeared_count': int(track.reappeared_count),
            'last_motion_state_change_sec': float(track.last_motion_state_change_sec),
            'last_stamp_sec': float(track.last_stamp_sec),
        })

    return {
        'stamp_sec': float(stamp_sec),
        'frame_id': frame_id,
        'tracks': items,
    }


def build_scan_event_payload(
    event_id: str,
    scanner_id: str,
    direction: str,
    barcode_id: str,
    stamp_sec: float,
    position_x: float,
    position_y: float,
    position_z: float,
    track_id: int = 0,
) -> Dict[str, Any]:
    return {
        'event_id': str(event_id),
        'scanner_id': str(scanner_id),
        'direction': str(direction),
        'barcode_id': str(barcode_id),
        'track_id': int(track_id),
        'stamp_sec': float(stamp_sec),
        'position': {
            'x': float(position_x),
            'y': float(position_y),
            'z': float(position_z),
        },
    }

def build_assignment_payload(
    event_id: str,
    scanner_id: str,
    direction: str,
    barcode_id: str,
    track_id: int,
    stamp_sec: float,
) -> Dict[str, Any]:
    return {
        'event_id': str(event_id),
        'scanner_id': str(scanner_id),
        'direction': str(direction),
        'barcode_id': str(barcode_id),
        'track_id': int(track_id),
        'stamp_sec': float(stamp_sec),
    }
