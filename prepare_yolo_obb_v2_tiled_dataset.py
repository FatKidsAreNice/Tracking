#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    source_root: Path
    full_target_root: Path
    tiled_target_root: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    tile_size_px: int
    tile_overlap_ratio: float
    train_object_jitter_tiles_per_object: int
    train_object_jitter_px: int
    max_negative_tiles_per_image: int
    min_object_area_px2_in_tile: float


@dataclass(frozen=True)
class ObbLabel:
    class_id: int
    points: np.ndarray


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    label_path: Path


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


class SourceClassMapper:
    PERSON_CLASS_ID = 0
    RACK_SIDE_SOURCE_CLASS_ID = 1
    RACK_TOP_SOURCE_CLASS_ID = 2

    RACK_SIDE_TARGET_CLASS_ID = 0
    RACK_TOP_TARGET_CLASS_ID = 1

    @classmethod
    def map_class(cls, source_class_id: int) -> int | None:
        if source_class_id == cls.PERSON_CLASS_ID:
            return None

        if source_class_id == cls.RACK_SIDE_SOURCE_CLASS_ID:
            return cls.RACK_SIDE_TARGET_CLASS_ID

        if source_class_id == cls.RACK_TOP_SOURCE_CLASS_ID:
            return cls.RACK_TOP_TARGET_CLASS_ID

        return None


class YoloObbLabelParser:
    def parse_label_file(self, label_path: Path, image_width: int, image_height: int) -> list[ObbLabel]:
        labels: list[ObbLabel] = []

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            try:
                source_class_id = int(float(parts[0]))
            except ValueError as exc:
                raise RuntimeError(f"{label_path}:{line_number}: ungültige class_id") from exc

            target_class_id = SourceClassMapper.map_class(source_class_id)

            if target_class_id is None:
                continue

            normalized_points = self.convert_parts_to_normalized_points(parts, label_path, line_number)
            pixel_points = normalized_points.copy()
            pixel_points[:, 0] *= image_width
            pixel_points[:, 1] *= image_height

            labels.append(
                ObbLabel(
                    class_id=target_class_id,
                    points=pixel_points.astype(np.float32),
                )
            )

        return labels

    def convert_parts_to_normalized_points(self, parts: list[str], label_path: Path, line_number: int) -> np.ndarray:
        if len(parts) == 5:
            return self.xywh_to_points(parts)

        values = [float(value) for value in parts[1:]]

        if len(parts) == 9:
            points = np.asarray(values, dtype=np.float32).reshape(4, 2)
            self.validate_normalized_points(points, label_path, line_number)
            return points

        if len(values) >= 6 and len(values) % 2 == 0:
            polygon_points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
            self.validate_normalized_points(polygon_points, label_path, line_number)
            return self.polygon_to_obb_points(polygon_points)

        raise RuntimeError(f"{label_path}:{line_number}: unbekanntes Label-Format mit {len(parts)} Werten")

    def xywh_to_points(self, parts: list[str]) -> np.ndarray:
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x_min = x_center - width * 0.5
        x_max = x_center + width * 0.5
        y_min = y_center - height * 0.5
        y_max = y_center + height * 0.5

        return np.asarray(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )

    def polygon_to_obb_points(self, normalized_polygon_points: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(normalized_polygon_points.astype(np.float32))
        box = cv2.boxPoints(rect)
        return self.order_points_clockwise(box)

    def order_points_clockwise(self, points: np.ndarray) -> np.ndarray:
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        ordered = points[sorted_indices]

        start_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start_index, axis=0)

        return ordered.astype(np.float32)

    def validate_normalized_points(self, points: np.ndarray, label_path: Path, line_number: int) -> None:
        if points.ndim != 2 or points.shape[1] != 2:
            raise RuntimeError(f"{label_path}:{line_number}: ungültige Punktform {points.shape}")

        for value in points.reshape(-1):
            if value < -0.001 or value > 1.001:
                raise RuntimeError(f"{label_path}:{line_number}: Koordinate außerhalb 0..1: {value}")


class DatasetCollector:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.image_dir = source_root / "train" / "images"
        self.label_dir = source_root / "train" / "labels"

    def collect(self) -> list[ImageLabelPair]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Bildordner nicht gefunden: {self.image_dir}")

        if not self.label_dir.exists():
            raise FileNotFoundError(f"Labelordner nicht gefunden: {self.label_dir}")

        image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        pairs: list[ImageLabelPair] = []
        missing_labels: list[str] = []

        for image_path in image_paths:
            label_path = self.label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue

            pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path))

        if missing_labels:
            raise RuntimeError(
                "Für folgende Bilder fehlen Label-Dateien:\n"
                + "\n".join(f"  {name}" for name in missing_labels[:50])
            )

        if not pairs:
            raise RuntimeError(f"Keine Bild/Label-Paare gefunden in {self.source_root}")

        return pairs


class DatasetSplitter:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def split(self, pairs: list[ImageLabelPair]) -> dict[str, list[ImageLabelPair]]:
        shuffled = list(pairs)
        random.seed(self.config.seed)
        random.shuffle(shuffled)

        total_count = len(shuffled)
        train_count = int(round(total_count * self.config.train_ratio))
        val_count = int(round(total_count * self.config.val_ratio))

        if train_count + val_count >= total_count:
            train_count = max(total_count - 2, 1)
            val_count = 1

        test_count = total_count - train_count - val_count

        if test_count <= 0:
            test_count = 1
            train_count = max(total_count - val_count - test_count, 1)

        return {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:train_count + val_count],
            "test": shuffled[train_count + val_count:],
        }


class DataYamlWriter:
    @staticmethod
    def write(target_root: Path) -> None:
        data_yaml = target_root / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {target_root}",
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                    "",
                    "names:",
                    "  0: rack_side_visible",
                    "  1: rack_top_visible",
                    "",
                ]
            ),
            encoding="utf-8",
        )


class FullDatasetWriter:
    def __init__(self, target_root: Path, parser: YoloObbLabelParser) -> None:
        self.target_root = target_root
        self.parser = parser

    def write(self, split_pairs: dict[str, list[ImageLabelPair]]) -> None:
        self.prepare_target()

        for split, pairs in split_pairs.items():
            for pair in pairs:
                self.write_pair(split, pair)

        DataYamlWriter.write(self.target_root)

    def prepare_target(self) -> None:
        self.backup_existing_target(self.target_root)

        for split in ("train", "val", "test"):
            (self.target_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def backup_existing_target(self, target_root: Path) -> None:
        if not target_root.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = target_root.with_name(f"{target_root.name}_backup_{timestamp}")
        shutil.move(str(target_root), str(backup_root))
        print(f"Bestehender Ordner gesichert: {backup_root}")

    def write_pair(self, split: str, pair: ImageLabelPair) -> None:
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Bild konnte nicht gelesen werden: {pair.image_path}")

        height, width = image.shape[:2]
        labels = self.parser.parse_label_file(pair.label_path, width, height)

        target_image_path = self.target_root / "images" / split / pair.image_path.name
        target_label_path = self.target_root / "labels" / split / f"{pair.image_path.stem}.txt"

        shutil.copy2(pair.image_path, target_image_path)
        target_label_path.write_text(self.labels_to_yolo_lines(labels, width, height), encoding="utf-8")

    def labels_to_yolo_lines(self, labels: list[ObbLabel], image_width: int, image_height: int) -> str:
        lines: list[str] = []

        for label in labels:
            normalized = label.points.copy()
            normalized[:, 0] /= image_width
            normalized[:, 1] /= image_height
            normalized = np.clip(normalized, 0.0, 1.0)

            values = [str(label.class_id)] + [f"{value:.6f}" for value in normalized.reshape(-1)]
            lines.append(" ".join(values))

        return "\n".join(lines) + ("\n" if lines else "")


class TilePlanner:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def create_windows(self, image_width: int, image_height: int, labels: list[ObbLabel], split: str) -> list[TileWindow]:
        windows = self.create_grid_windows(image_width, image_height)

        if split == "train" and self.config.train_object_jitter_tiles_per_object > 0:
            windows.extend(self.create_object_jitter_windows(image_width, image_height, labels))

        unique: dict[tuple[int, int, int], TileWindow] = {}
        for window in windows:
            unique[(window.x_min, window.y_min, window.size)] = window

        return list(unique.values())

    def create_grid_windows(self, image_width: int, image_height: int) -> list[TileWindow]:
        tile_size = self.config.tile_size_px
        step = max(int(tile_size * (1.0 - self.config.tile_overlap_ratio)), 1)

        x_starts = self.compute_starts(image_width, tile_size, step)
        y_starts = self.compute_starts(image_height, tile_size, step)

        return [
            TileWindow(x_min=x_start, y_min=y_start, size=tile_size)
            for y_start in y_starts
            for x_start in x_starts
        ]

    def create_object_jitter_windows(self, image_width: int, image_height: int, labels: list[ObbLabel]) -> list[TileWindow]:
        tile_size = self.config.tile_size_px
        windows: list[TileWindow] = []

        for label in labels:
            center = np.mean(label.points, axis=0)

            for _ in range(self.config.train_object_jitter_tiles_per_object):
                jitter_x = random.randint(-self.config.train_object_jitter_px, self.config.train_object_jitter_px)
                jitter_y = random.randint(-self.config.train_object_jitter_px, self.config.train_object_jitter_px)

                x_min = int(round(center[0] - tile_size * 0.5 + jitter_x))
                y_min = int(round(center[1] - tile_size * 0.5 + jitter_y))

                x_min = self.clamp_start(x_min, image_width, tile_size)
                y_min = self.clamp_start(y_min, image_height, tile_size)

                windows.append(TileWindow(x_min=x_min, y_min=y_min, size=tile_size))

        return windows

    def compute_starts(self, image_size: int, tile_size: int, step: int) -> list[int]:
        if image_size <= tile_size:
            return [0]

        starts = list(range(0, image_size - tile_size + 1, step))
        final_start = image_size - tile_size

        if starts[-1] != final_start:
            starts.append(final_start)

        return starts

    def clamp_start(self, start: int, image_size: int, tile_size: int) -> int:
        if image_size <= tile_size:
            return 0

        return max(0, min(start, image_size - tile_size))


class TiledDatasetWriter:
    def __init__(self, config: DatasetConfig, parser: YoloObbLabelParser) -> None:
        self.config = config
        self.parser = parser
        self.tile_planner = TilePlanner(config)

    def write(self, split_pairs: dict[str, list[ImageLabelPair]]) -> None:
        self.prepare_target()

        summary: dict[str, dict[str, int]] = {}

        for split, pairs in split_pairs.items():
            summary[split] = self.write_split(split, pairs)

        DataYamlWriter.write(self.config.tiled_target_root)
        self.print_summary(summary)

    def prepare_target(self) -> None:
        self.backup_existing_target(self.config.tiled_target_root)

        for split in ("train", "val", "test"):
            (self.config.tiled_target_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.config.tiled_target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def backup_existing_target(self, target_root: Path) -> None:
        if not target_root.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = target_root.with_name(f"{target_root.name}_backup_{timestamp}")
        shutil.move(str(target_root), str(backup_root))
        print(f"Bestehender Ordner gesichert: {backup_root}")

    def write_split(self, split: str, pairs: list[ImageLabelPair]) -> dict[str, int]:
        written_tiles = 0
        positive_tiles = 0
        negative_tiles = 0
        written_objects = 0

        for pair in pairs:
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise RuntimeError(f"Bild konnte nicht gelesen werden: {pair.image_path}")

            image_height, image_width = image.shape[:2]
            labels = self.parser.parse_label_file(pair.label_path, image_width, image_height)
            windows = self.tile_planner.create_windows(image_width, image_height, labels, split)
            per_image_negative_count = 0

            for tile_index, window in enumerate(windows):
                tile_labels = self.labels_inside_tile(labels, window)

                is_positive = len(tile_labels) > 0

                if not is_positive:
                    if per_image_negative_count >= self.config.max_negative_tiles_per_image:
                        continue
                    per_image_negative_count += 1

                tile_image = image[window.y_min:window.y_max, window.x_min:window.x_max]

                if tile_image.shape[0] != self.config.tile_size_px or tile_image.shape[1] != self.config.tile_size_px:
                    continue

                tile_stem = f"{pair.image_path.stem}_tile_{tile_index:04d}_x{window.x_min}_y{window.y_min}"
                tile_image_path = self.config.tiled_target_root / "images" / split / f"{tile_stem}.png"
                tile_label_path = self.config.tiled_target_root / "labels" / split / f"{tile_stem}.txt"

                cv2.imwrite(str(tile_image_path), tile_image)
                tile_label_path.write_text(self.tile_labels_to_yolo_lines(tile_labels, window), encoding="utf-8")

                written_tiles += 1
                written_objects += len(tile_labels)

                if is_positive:
                    positive_tiles += 1
                else:
                    negative_tiles += 1

        return {
            "tiles": written_tiles,
            "positive_tiles": positive_tiles,
            "negative_tiles": negative_tiles,
            "objects": written_objects,
        }

    def labels_inside_tile(self, labels: list[ObbLabel], window: TileWindow) -> list[ObbLabel]:
        tile_labels: list[ObbLabel] = []

        for label in labels:
            shifted_points = label.points.copy()
            shifted_points[:, 0] -= window.x_min
            shifted_points[:, 1] -= window.y_min

            if not self.is_fully_inside_tile(shifted_points, window.size):
                continue

            area = abs(float(cv2.contourArea(shifted_points.astype(np.float32))))

            if area < self.config.min_object_area_px2_in_tile:
                continue

            tile_labels.append(
                ObbLabel(
                    class_id=label.class_id,
                    points=shifted_points.astype(np.float32),
                )
            )

        return tile_labels

    def is_fully_inside_tile(self, points: np.ndarray, tile_size: int) -> bool:
        return bool(
            np.all(points[:, 0] >= 0.0)
            and np.all(points[:, 0] <= float(tile_size - 1))
            and np.all(points[:, 1] >= 0.0)
            and np.all(points[:, 1] <= float(tile_size - 1))
        )

    def tile_labels_to_yolo_lines(self, labels: list[ObbLabel], window: TileWindow) -> str:
        lines: list[str] = []

        for label in labels:
            normalized = label.points.copy()
            normalized[:, 0] /= window.size
            normalized[:, 1] /= window.size
            normalized = np.clip(normalized, 0.0, 1.0)

            values = [str(label.class_id)] + [f"{value:.6f}" for value in normalized.reshape(-1)]
            lines.append(" ".join(values))

        return "\n".join(lines) + ("\n" if lines else "")

    def print_summary(self, summary: dict[str, dict[str, int]]) -> None:
        print()
        print("Tile-Dataset erzeugt:")
        print(f"Ziel: {self.config.tiled_target_root}")

        for split in ("train", "val", "test"):
            values = summary.get(split, {})
            print(
                f"{split}: "
                f"tiles={values.get('tiles', 0)}, "
                f"positive={values.get('positive_tiles', 0)}, "
                f"negative={values.get('negative_tiles', 0)}, "
                f"objects={values.get('objects', 0)}"
            )


class DatasetValidator:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def validate(self, dataset_root: Path) -> None:
        for split in ("train", "val", "test"):
            image_dir = dataset_root / "images" / split
            label_dir = dataset_root / "labels" / split

            images = [
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            ]
            labels = list(label_dir.glob("*.txt"))

            if len(images) != len(labels):
                raise RuntimeError(f"{dataset_root.name}/{split}: images={len(images)}, labels={len(labels)}")

            for label_path in labels:
                self.validate_label_file(label_path)

    def validate_label_file(self, label_path: Path) -> None:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 9:
                raise RuntimeError(f"{label_path}:{line_number}: erwartet 9 Werte, gefunden {len(parts)}")

            class_id = int(parts[0])

            if class_id not in (0, 1):
                raise RuntimeError(f"{label_path}:{line_number}: ungültige class_id {class_id}")

            coords = [float(value) for value in parts[1:]]

            for value in coords:
                if value < 0.0 or value > 1.0:
                    raise RuntimeError(f"{label_path}:{line_number}: Koordinate außerhalb 0..1: {value}")


def main() -> None:
    config = DatasetConfig(
        source_root=Path.home() / "ros2_ws/bev_dataset/bev_dataset_labeld",
        full_target_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v2_full",
        tiled_target_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v2_tiled",
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        seed=42,
        tile_size_px=1024,
        tile_overlap_ratio=0.25,
        train_object_jitter_tiles_per_object=2,
        train_object_jitter_px=256,
        max_negative_tiles_per_image=2,
        min_object_area_px2_in_tile=100.0,
    )

    print("Quelle:", config.source_root)

    collector = DatasetCollector(config.source_root)
    pairs = collector.collect()
    print(f"Gefundene Bild/Label-Paare: {len(pairs)}")

    splitter = DatasetSplitter(config)
    split_pairs = splitter.split(pairs)

    for split, values in split_pairs.items():
        print(f"{split}: {len(values)} Bilder")

    parser = YoloObbLabelParser()

    full_writer = FullDatasetWriter(config.full_target_root, parser)
    full_writer.write(split_pairs)

    tiled_writer = TiledDatasetWriter(config, parser)
    tiled_writer.write(split_pairs)

    validator = DatasetValidator()
    validator.validate(config.full_target_root)
    validator.validate(config.tiled_target_root)

    print()
    print("Fertig.")
    print(f"Full Dataset:  {config.full_target_root}")
    print(f"Tiled Dataset: {config.tiled_target_root}")
    print()
    print("Trainingsempfehlung:")
    print(f"  data={config.tiled_target_root / 'data.yaml'}")
    print("  imgsz=1024")


if __name__ == "__main__":
    main()
