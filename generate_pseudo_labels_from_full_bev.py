#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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


@dataclass(frozen=True)
class ImagePredictionResult:
    image_path: Path
    copied_image_path: Path
    label_path: Path
    overlay_path: Path
    json_path: Path
    raw_count: int
    threshold_count: int
    final_count: int
    side_count: int
    top_count: int


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


class BoxGeometry:
    @staticmethod
    def compute_center(points: np.ndarray) -> np.ndarray:
        return np.mean(points.astype(np.float32), axis=0)

    @staticmethod
    def compute_yaw(points: np.ndarray) -> float:
        points = points.astype(np.float32).reshape(4, 2)

        edge_01 = points[1] - points[0]
        edge_12 = points[2] - points[1]

        len_01 = float(np.linalg.norm(edge_01))
        len_12 = float(np.linalg.norm(edge_12))

        edge = edge_01 if len_01 >= len_12 else edge_12

        if float(np.linalg.norm(edge)) < 1e-6:
            return 0.0

        return math.atan2(float(edge[1]), float(edge[0]))

    @staticmethod
    def fixed_box_from_center_yaw(center: np.ndarray, yaw: float, box_size_px: float) -> np.ndarray:
        half = float(box_size_px) * 0.5

        local_points = np.asarray(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32,
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        rotation = np.asarray(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ],
            dtype=np.float32,
        )

        return local_points @ rotation.T + center.astype(np.float32)

    @staticmethod
    def clamp_points_to_image(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        clamped = points.copy().astype(np.float32)
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(image_width - 1))
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(image_height - 1))
        return clamped


class DetectionNormalizer:
    def __init__(self, fixed_box_size_px: float) -> None:
        self.fixed_box_size_px = float(fixed_box_size_px)

    def normalize(self, detection: Detection, image_width: int, image_height: int) -> Detection:
        points = detection.points.astype(np.float32)

        if self.fixed_box_size_px > 0.0:
            center = BoxGeometry.compute_center(points)
            yaw = BoxGeometry.compute_yaw(points)
            points = BoxGeometry.fixed_box_from_center_yaw(center, yaw, self.fixed_box_size_px)

        points = BoxGeometry.clamp_points_to_image(points, image_width, image_height)

        return Detection(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            points=points,
            source_tile=detection.source_tile,
        )


class OrientedBoxNms:
    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = float(iou_threshold)

    def apply(self, detections: list[Detection]) -> list[Detection]:
        final_detections: list[Detection] = []

        for class_id in sorted(set(detection.class_id for detection in detections)):
            class_detections = [detection for detection in detections if detection.class_id == class_id]
            final_detections.extend(self.apply_single_class(class_detections))

        return sorted(final_detections, key=lambda detection: detection.confidence, reverse=True)

    def apply_single_class(self, detections: list[Detection]) -> list[Detection]:
        sorted_detections = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
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


class PseudoLabelWriter:
    CLASS_COLORS = {
        0: (0, 255, 0),
        1: (0, 180, 255),
    }

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.expanduser()

        self.images_dir = self.output_root / "images" / "train"
        self.labels_dir = self.output_root / "labels" / "train"
        self.overlays_dir = self.output_root / "overlays"
        self.json_dir = self.output_root / "json"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def write_data_yaml(self) -> None:
        data_yaml = self.output_root / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.output_root}",
                    "train: images/train",
                    "val: images/train",
                    "test: images/train",
                    "",
                    "names:",
                    "  0: rack_side_visible",
                    "  1: rack_top_visible",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def write_image_result(
        self,
        image_path: Path,
        image: np.ndarray,
        raw_detections: list[Detection],
        threshold_detections: list[Detection],
        final_detections: list[Detection],
        args: argparse.Namespace,
    ) -> ImagePredictionResult:
        target_image_path = self.images_dir / image_path.name
        target_label_path = self.labels_dir / f"{image_path.stem}.txt"
        overlay_path = self.overlays_dir / f"{image_path.stem}_overlay.png"
        json_path = self.json_dir / f"{image_path.stem}.json"

        shutil.copy2(image_path, target_image_path)

        target_label_path.write_text(
            self.build_yolo_obb_label_text(final_detections, image.shape[1], image.shape[0]),
            encoding="utf-8",
        )

        overlay = self.draw_overlay(image, final_detections)
        cv2.imwrite(str(overlay_path), overlay)

        json_path.write_text(
            json.dumps(
                self.build_json_payload(
                    image_path=image_path,
                    image=image,
                    raw_detections=raw_detections,
                    threshold_detections=threshold_detections,
                    final_detections=final_detections,
                    args=args,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        side_count = sum(1 for detection in final_detections if detection.class_id == 0)
        top_count = sum(1 for detection in final_detections if detection.class_id == 1)

        return ImagePredictionResult(
            image_path=image_path,
            copied_image_path=target_image_path,
            label_path=target_label_path,
            overlay_path=overlay_path,
            json_path=json_path,
            raw_count=len(raw_detections),
            threshold_count=len(threshold_detections),
            final_count=len(final_detections),
            side_count=side_count,
            top_count=top_count,
        )

    def build_yolo_obb_label_text(self, detections: list[Detection], image_width: int, image_height: int) -> str:
        lines: list[str] = []

        for detection in detections:
            normalized = detection.points.copy().astype(np.float32)
            normalized[:, 0] /= float(image_width)
            normalized[:, 1] /= float(image_height)
            normalized = np.clip(normalized, 0.0, 1.0)

            values = [str(detection.class_id)] + [f"{value:.6f}" for value in normalized.reshape(-1)]
            lines.append(" ".join(values))

        return "\n".join(lines) + ("\n" if lines else "")

    def draw_overlay(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        overlay = image.copy()

        for detection in detections:
            color = self.CLASS_COLORS.get(detection.class_id, (255, 255, 255))
            points = detection.points.astype(np.int32)
            center = points.mean(axis=0).astype(int)

            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=3)

            label = f"{detection.class_name} {detection.confidence:.2f}"

            cv2.putText(
                overlay,
                label,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        return overlay

    def build_json_payload(
        self,
        image_path: Path,
        image: np.ndarray,
        raw_detections: list[Detection],
        threshold_detections: list[Detection],
        final_detections: list[Detection],
        args: argparse.Namespace,
    ) -> dict:
        image_height, image_width = image.shape[:2]

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_image": str(image_path),
            "image_width": image_width,
            "image_height": image_height,
            "model": str(args.model),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "imgsz": args.imgsz,
            "model_conf": args.model_conf,
            "side_conf": args.side_conf,
            "top_conf": args.top_conf,
            "nms_iou": args.nms_iou,
            "fixed_box_size_px": args.fixed_box_size_px,
            "counts": {
                "raw": len(raw_detections),
                "after_threshold": len(threshold_detections),
                "after_nms": len(final_detections),
                "rack_side_visible": sum(1 for detection in final_detections if detection.class_id == 0),
                "rack_top_visible": sum(1 for detection in final_detections if detection.class_id == 1),
            },
            "detections": [
                self.detection_to_dict(detection)
                for detection in final_detections
            ],
        }

    def detection_to_dict(self, detection: Detection) -> dict:
        center = BoxGeometry.compute_center(detection.points)
        yaw = BoxGeometry.compute_yaw(detection.points)

        return {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "center_px": {
                "x": float(center[0]),
                "y": float(center[1]),
            },
            "yaw_rad_image": float(yaw),
            "points_px": detection.points.astype(float).tolist(),
            "source_tile": {
                "x_min": detection.source_tile.x_min,
                "y_min": detection.source_tile.y_min,
                "size": detection.source_tile.size,
            },
        }

    def write_manifest(self, results: list[ImagePredictionResult]) -> None:
        manifest_path = self.output_root / "review_manifest.csv"

        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source_image",
                    "copied_image",
                    "label",
                    "overlay",
                    "json",
                    "raw_count",
                    "threshold_count",
                    "final_count",
                    "rack_side_visible",
                    "rack_top_visible",
                ]
            )

            for result in results:
                writer.writerow(
                    [
                        str(result.image_path),
                        str(result.copied_image_path),
                        str(result.label_path),
                        str(result.overlay_path),
                        str(result.json_path),
                        result.raw_count,
                        result.threshold_count,
                        result.final_count,
                        result.side_count,
                        result.top_count,
                    ]
                )


class SourceImageCollector:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def collect(self, source: Path, max_images: int) -> list[Path]:
        source = source.expanduser()

        if source.is_file():
            return [source]

        if not source.exists():
            raise FileNotFoundError(f"Quelle nicht gefunden: {source}")

        image_paths = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        if max_images > 0:
            return image_paths[:max_images]

        return image_paths


class PseudoLabelGenerator:
    CLASS_NAMES = {
        0: "rack_side_visible",
        1: "rack_top_visible",
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = YOLO(str(args.model))
        self.tile_planner = TilePlanner(
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
        )
        self.converter = DetectionConverter(self.CLASS_NAMES)
        self.threshold_filter = ClassThresholdFilter(
            side_conf=args.side_conf,
            top_conf=args.top_conf,
        )
        self.normalizer = DetectionNormalizer(
            fixed_box_size_px=args.fixed_box_size_px,
        )
        self.nms = OrientedBoxNms(args.nms_iou)
        self.writer = PseudoLabelWriter(Path(args.output))
        self.collector = SourceImageCollector()

    def run(self) -> None:
        image_paths = self.collector.collect(Path(self.args.source), self.args.max_images)

        if not image_paths:
            raise RuntimeError("Keine Eingabebilder gefunden.")

        self.writer.write_data_yaml()

        print(f"Gefundene Bilder: {len(image_paths)}")
        print(f"Output: {Path(self.args.output).expanduser()}")

        results: list[ImagePredictionResult] = []

        for index, image_path in enumerate(image_paths, start=1):
            print()
            print(f"[{index}/{len(image_paths)}] {image_path}")
            result = self.process_image(image_path)
            results.append(result)

        self.writer.write_manifest(results)

        print()
        print("Pseudo-Label-Erzeugung abgeschlossen.")
        print(f"Review-Ordner: {Path(self.args.output).expanduser()}")
        print(f"Overlays:      {Path(self.args.output).expanduser() / 'overlays'}")
        print(f"Labels:        {Path(self.args.output).expanduser() / 'labels' / 'train'}")
        print(f"Manifest:      {Path(self.args.output).expanduser() / 'review_manifest.csv'}")

    def process_image(self, image_path: Path) -> ImagePredictionResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Bild konnte nicht gelesen werden: {image_path}")

        image_height, image_width = image.shape[:2]
        windows = self.tile_planner.create_windows(image_width, image_height)

        raw_detections: list[Detection] = []

        for tile_index, window in enumerate(windows, start=1):
            tile = image[window.y_min:window.y_max, window.x_min:window.x_max]

            if tile.shape[0] != self.args.tile_size or tile.shape[1] != self.args.tile_size:
                continue

            predictions = self.model.predict(
                source=tile,
                imgsz=self.args.imgsz,
                conf=self.args.model_conf,
                iou=self.args.model_iou,
                device=self.args.device,
                verbose=False,
            )

            if predictions:
                raw_detections.extend(self.converter.from_tile_result(predictions[0], window))

            if tile_index % 10 == 0 or tile_index == len(windows):
                print(f"  Tiles verarbeitet: {tile_index}/{len(windows)}")

        threshold_detections = [
            detection for detection in raw_detections
            if self.threshold_filter.keep(detection)
        ]

        nms_detections = self.nms.apply(threshold_detections)

        final_detections = [
            self.normalizer.normalize(detection, image_width, image_height)
            for detection in nms_detections
        ]

        side_count = sum(1 for detection in final_detections if detection.class_id == 0)
        top_count = sum(1 for detection in final_detections if detection.class_id == 1)

        print(f"  Raw detections:      {len(raw_detections)}")
        print(f"  Nach Threshold:      {len(threshold_detections)}")
        print(f"  Nach OBB-NMS:        {len(final_detections)}")
        print(f"    rack_side_visible: {side_count}")
        print(f"    rack_top_visible:  {top_count}")

        return self.writer.write_image_result(
            image_path=image_path,
            image=image,
            raw_detections=raw_detections,
            threshold_detections=threshold_detections,
            final_detections=final_detections,
            args=self.args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Pseudo-Labels aus Full-BEV-Bildern per YOLO-OBB Sliding Window."
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
        default=Path.home() / "ros2_ws/bev_dataset/pseudo_labels_v3_review",
    )

    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--model-conf", type=float, default=0.03)
    parser.add_argument("--model-iou", type=float, default=0.50)

    parser.add_argument("--side-conf", type=float, default=0.25)
    parser.add_argument("--top-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.20)

    parser.add_argument(
        "--fixed-box-size-px",
        type=float,
        default=130.0,
        help="Feste Schrank-Footprint-Größe in Pixeln. 130px = 1.30m bei 0.01m/px. 0 deaktiviert.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional: maximale Anzahl Bilder aus Ordner. 0 = alle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = PseudoLabelGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
