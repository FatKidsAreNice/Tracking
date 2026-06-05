#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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
class Detection:
    class_id: int
    class_name: str
    confidence: float
    points: np.ndarray
    source_tile: TileWindow


class TilePlanner:
    def __init__(self, tile_size: int, overlap_ratio: float) -> None:
        self.tile_size = int(tile_size)
        self.overlap_ratio = float(overlap_ratio)

        if self.tile_size <= 0:
            raise ValueError("tile_size muss größer 0 sein.")

        if self.overlap_ratio < 0.0 or self.overlap_ratio >= 1.0:
            raise ValueError("overlap_ratio muss im Bereich [0.0, 1.0) liegen.")

    def create_windows(self, image_width: int, image_height: int) -> list[TileWindow]:
        step = max(int(round(self.tile_size * (1.0 - self.overlap_ratio))), 1)

        x_starts = self.compute_starts(image_width, self.tile_size, step)
        y_starts = self.compute_starts(image_height, self.tile_size, step)

        return [
            TileWindow(x_min=x_start, y_min=y_start, size=self.tile_size)
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


class DetectionConverter:
    def __init__(self, class_names: dict[int, str]) -> None:
        self.class_names = class_names

    def from_tile_result(self, result, tile_window: TileWindow) -> list[Detection]:
        if result.obb is None:
            return []

        if result.obb.xyxyxyxy is None:
            return []

        boxes = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy() if result.obb.conf is not None else np.ones(len(boxes))
        classes = result.obb.cls.cpu().numpy() if result.obb.cls is not None else np.zeros(len(boxes))

        detections: list[Detection] = []

        for index, box in enumerate(boxes):
            class_id = int(classes[index])
            confidence = float(confs[index])

            points = np.asarray(box, dtype=np.float32).reshape(4, 2)
            points[:, 0] += tile_window.x_min
            points[:, 1] += tile_window.y_min

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    points=points,
                    source_tile=tile_window,
                )
            )

        return detections


class ClassThresholdFilter:
    def __init__(self, side_conf: float, top_conf: float) -> None:
        self.side_conf = float(side_conf)
        self.top_conf = float(top_conf)

    def keep(self, detection: Detection) -> bool:
        if detection.class_id == 0:
            return detection.confidence >= self.side_conf

        if detection.class_id == 1:
            return detection.confidence >= self.top_conf

        return False


class OrientedBoxNms:
    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = float(iou_threshold)

    def apply(self, detections: list[Detection]) -> list[Detection]:
        final_detections: list[Detection] = []

        for class_id in sorted(set(det.class_id for det in detections)):
            class_detections = [det for det in detections if det.class_id == class_id]
            final_detections.extend(self.apply_single_class(class_detections))

        return sorted(final_detections, key=lambda det: det.confidence, reverse=True)

    def apply_single_class(self, detections: list[Detection]) -> list[Detection]:
        sorted_detections = sorted(detections, key=lambda det: det.confidence, reverse=True)
        kept: list[Detection] = []

        while sorted_detections:
            current = sorted_detections.pop(0)
            kept.append(current)

            remaining: list[Detection] = []

            for candidate in sorted_detections:
                iou = self.oriented_iou(current.points, candidate.points)

                if iou < self.iou_threshold:
                    remaining.append(candidate)

            sorted_detections = remaining

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


class ResultWriter:
    CLASS_COLORS = {
        0: (0, 255, 0),
        1: (0, 180, 255),
    }

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser()
        self.images_dir = self.output_root / "images"
        self.json_dir = self.output_root / "json"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        image_path: Path,
        image: np.ndarray,
        detections_before_nms: list[Detection],
        detections_after_nms: list[Detection],
        args: argparse.Namespace,
    ) -> None:
        output_image = image.copy()

        for detection in detections_after_nms:
            self.draw_detection(output_image, detection)

        output_image_path = self.images_dir / f"{image_path.stem}_pred.png"
        output_json_path = self.json_dir / f"{image_path.stem}_pred.json"

        cv2.imwrite(str(output_image_path), output_image)

        payload = self.build_payload(
            image_path=image_path,
            image=image,
            detections_before_nms=detections_before_nms,
            detections_after_nms=detections_after_nms,
            args=args,
        )

        output_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Bild gespeichert: {output_image_path}")
        print(f"JSON gespeichert: {output_json_path}")

    def draw_detection(self, image: np.ndarray, detection: Detection) -> None:
        points = detection.points.astype(np.int32)
        color = self.CLASS_COLORS.get(detection.class_id, (255, 255, 255))

        cv2.polylines(image, [points], isClosed=True, color=color, thickness=3)

        center = points.mean(axis=0).astype(int)
        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.putText(
            image,
            label,
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def build_payload(
        self,
        image_path: Path,
        image: np.ndarray,
        detections_before_nms: list[Detection],
        detections_after_nms: list[Detection],
        args: argparse.Namespace,
    ) -> dict:
        height, width = image.shape[:2]

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_image": str(image_path),
            "image_width": width,
            "image_height": height,
            "model": str(args.model),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "imgsz": args.imgsz,
            "model_conf": args.model_conf,
            "side_conf": args.side_conf,
            "top_conf": args.top_conf,
            "nms_iou": args.nms_iou,
            "counts": {
                "before_threshold_and_nms": len(detections_before_nms),
                "after_threshold_and_nms": len(detections_after_nms),
            },
            "detections": [
                self.detection_to_dict(detection)
                for detection in detections_after_nms
            ],
        }

    def detection_to_dict(self, detection: Detection) -> dict:
        center = np.mean(detection.points, axis=0)
        yaw = self.compute_yaw(detection.points)

        return {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "center_px": {
                "x": float(center[0]),
                "y": float(center[1]),
            },
            "yaw_rad_image": yaw,
            "points_px": detection.points.astype(float).tolist(),
            "source_tile": {
                "x_min": detection.source_tile.x_min,
                "y_min": detection.source_tile.y_min,
                "size": detection.source_tile.size,
            },
        }

    def compute_yaw(self, points: np.ndarray) -> float:
        edge_01 = points[1] - points[0]
        edge_12 = points[2] - points[1]

        len_01 = float(np.linalg.norm(edge_01))
        len_12 = float(np.linalg.norm(edge_12))

        edge = edge_01 if len_01 >= len_12 else edge_12

        if float(np.linalg.norm(edge)) < 1e-6:
            return 0.0

        return math.atan2(float(edge[1]), float(edge[0]))


class FullBevSlidingWindowPredictor:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = YOLO(str(args.model))
        self.tile_planner = TilePlanner(
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
        )
        self.converter = DetectionConverter(
            class_names={
                0: "rack_side_visible",
                1: "rack_top_visible",
            }
        )
        self.threshold_filter = ClassThresholdFilter(
            side_conf=args.side_conf,
            top_conf=args.top_conf,
        )
        self.nms = OrientedBoxNms(iou_threshold=args.nms_iou)
        self.writer = ResultWriter(output_root=Path(args.output))

    def run(self) -> None:
        image_paths = self.collect_images(Path(self.args.source))

        print(f"Gefundene Bilder: {len(image_paths)}")

        for image_path in image_paths:
            self.process_image(image_path)

    def collect_images(self, source: Path) -> list[Path]:
        source = source.expanduser()

        if source.is_file():
            return [source]

        if source.is_dir():
            return sorted(
                path for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            )

        raise FileNotFoundError(f"Quelle nicht gefunden: {source}")

    def process_image(self, image_path: Path) -> None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Bild konnte nicht gelesen werden: {image_path}")

        height, width = image.shape[:2]
        windows = self.tile_planner.create_windows(width, height)

        print()
        print(f"Verarbeite: {image_path}")
        print(f"Bildgröße: {width}x{height}")
        print(f"Tiles: {len(windows)}")

        raw_detections: list[Detection] = []

        for tile_index, window in enumerate(windows, start=1):
            tile = image[window.y_min:window.y_max, window.x_min:window.x_max]

            if tile.shape[0] != self.args.tile_size or tile.shape[1] != self.args.tile_size:
                continue

            results = self.model.predict(
                source=tile,
                imgsz=self.args.imgsz,
                conf=self.args.model_conf,
                iou=self.args.model_iou,
                device=self.args.device,
                verbose=False,
            )

            if results:
                raw_detections.extend(self.converter.from_tile_result(results[0], window))

            if tile_index % 10 == 0 or tile_index == len(windows):
                print(f"  Tiles verarbeitet: {tile_index}/{len(windows)}")

        filtered_detections = [
            detection for detection in raw_detections
            if self.threshold_filter.keep(detection)
        ]

        final_detections = self.nms.apply(filtered_detections)

        print(f"Raw detections:      {len(raw_detections)}")
        print(f"Nach Threshold:      {len(filtered_detections)}")
        print(f"Nach OBB-NMS:        {len(final_detections)}")
        print(f"  rack_side_visible: {sum(1 for det in final_detections if det.class_id == 0)}")
        print(f"  rack_top_visible:  {sum(1 for det in final_detections if det.class_id == 1)}")

        self.writer.write(
            image_path=image_path,
            image=image,
            detections_before_nms=raw_detections,
            detections_after_nms=final_detections,
            args=self.args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Full-BEV Sliding-Window Prediction für YOLO-OBB Tile-Modelle."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "ros2_ws/bev_dataset/models/rack_bev_obb_yolo26s_v2_tiled_best_epoch161.pt",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Ein Full-BEV-Bild oder ein Ordner mit Full-BEV-Bildern.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "ros2_ws/bev_dataset/predictions/full_bev_sliding_v2",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--model-conf", type=float, default=0.05)
    parser.add_argument("--model-iou", type=float, default=0.50)

    parser.add_argument("--side-conf", type=float, default=0.25)
    parser.add_argument("--top-conf", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FullBevSlidingWindowPredictor(args)
    predictor.run()


if __name__ == "__main__":
    main()
