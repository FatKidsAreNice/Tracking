from __future__ import annotations

import json
from typing import Dict, List, Any

from std_msgs.msg import String

from .tracking_types import Track


def make_string_message(payload: Dict[str, Any]) -> String:
    msg = String()
    msg.data = json.dumps(payload, sort_keys=True)
    return msg


def parse_string_message(msg: String) -> Dict[str, Any]:
    if not msg.data:
        return {}
    return json.loads(msg.data)


def build_track_states_payload(tracks: Dict[int, Track], stamp_sec: float) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    for track in sorted(tracks.values(), key=lambda item: item.track_id):
        items.append({
            'track_id': int(track.track_id),
            'barcode_id': str(track.barcode_id),
            'x': float(track.centroid[0]),
            'y': float(track.centroid[1]),
            'z': float(track.centroid[2]),
            'vx': float(track.velocity[0]),
            'vy': float(track.velocity[1]),
            'vz': float(track.velocity[2]),
            'age': int(track.age),
            'missed_updates': int(track.missed_updates),
            'last_stamp_sec': float(track.last_stamp_sec),
        })

    return {
        'stamp_sec': float(stamp_sec),
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
