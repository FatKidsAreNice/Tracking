from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np


def resolve_floorplan_image_path(image_path: str) -> Path | None:
    if not image_path:
        return None

    raw_path = Path(image_path).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        try:
            package_share = Path(get_package_share_directory('coldstore_tracking'))
        except Exception:
            package_share = None

        if package_share is not None:
            candidates.append(package_share / 'config' / raw_path.name)
            candidates.append(package_share / raw_path)
            candidates.append(package_share.parent.parent / 'src' / 'coldstore_tracking' / 'config' / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_floorplan_image(image_path: str, rotation_deg: float) -> np.ndarray | None:
    resolved_path = resolve_floorplan_image_path(image_path)
    if resolved_path is None:
        return None

    image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rotate_floorplan_image(image_rgb, rotation_deg)


def rotate_floorplan_image(image: np.ndarray, rotation_deg: float) -> np.ndarray:
    normalized_deg = int(round(rotation_deg)) % 360
    if normalized_deg == 0:
        return image
    if normalized_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if normalized_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    cos_value = abs(rotation_matrix[0, 0])
    sin_value = abs(rotation_matrix[0, 1])
    bound_width = max(int(math.ceil(height * sin_value + width * cos_value)), 1)
    bound_height = max(int(math.ceil(height * cos_value + width * sin_value)), 1)

    rotation_matrix[0, 2] += bound_width * 0.5 - center[0]
    rotation_matrix[1, 2] += bound_height * 0.5 - center[1]
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def render_floorplan_background(
    floorplan_image: np.ndarray,
    target_size: tuple[int, int],
    scale: float,
    offset_x_ratio: float,
    offset_y_ratio: float,
    fit_mode: str = "contain",
    background_rgb: tuple[int, int, int] = (243, 247, 244),
) -> np.ndarray:
    target_width, target_height = target_size
    image_height, image_width = floorplan_image.shape[:2]
    normalized_fit_mode = str(fit_mode).strip().lower()
    scale_x = target_width / max(image_width, 1)
    scale_y = target_height / max(image_height, 1)
    if normalized_fit_mode == "cover":
        base_scale = max(scale_x, scale_y)
    else:
        base_scale = min(scale_x, scale_y)
    final_scale = max(base_scale * float(scale), 1e-6)
    scaled_width = max(int(round(image_width * final_scale)), 1)
    scaled_height = max(int(round(image_height * final_scale)), 1)
    interpolation = cv2.INTER_AREA if final_scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(floorplan_image, (scaled_width, scaled_height), interpolation=interpolation)

    canvas = np.full((target_height, target_width, 3), background_rgb, dtype=np.uint8)
    x_offset = int(round((target_width - scaled_width) * 0.5 + float(offset_x_ratio) * target_width))
    y_offset = int(round((target_height - scaled_height) * 0.5 + float(offset_y_ratio) * target_height))

    paste_image_onto_canvas(canvas, scaled, x_offset, y_offset)
    return canvas


def paste_image_onto_canvas(canvas: np.ndarray, image: np.ndarray, x_offset: int, y_offset: int) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    image_height, image_width = image.shape[:2]

    x_start = max(x_offset, 0)
    y_start = max(y_offset, 0)
    x_end = min(x_offset + image_width, canvas_width)
    y_end = min(y_offset + image_height, canvas_height)

    if x_start >= x_end or y_start >= y_end:
        return

    source_x_start = x_start - x_offset
    source_y_start = y_start - y_offset
    source_x_end = source_x_start + (x_end - x_start)
    source_y_end = source_y_start + (y_end - y_start)
    canvas[y_start:y_end, x_start:x_end] = image[source_y_start:source_y_end, source_x_start:source_x_end]


def clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)
