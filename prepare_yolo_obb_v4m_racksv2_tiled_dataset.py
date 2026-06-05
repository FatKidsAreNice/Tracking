#!/usr/bin/env python3
from __future__ import annotations

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
    max_negative_tiles_per_image: int
    min_object_area_px2_in_tile: float
    fixed_box_size_px: float


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class SplitPair:
    image_path: Path
    label_path: Path
    split: str


@dataclass(frozen=True)
class ObbLabel:
    class_id: int
    points: np.ndarray


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


class DatasetPathService:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    @staticmethod
    def backup_existing(path: Path) -> None:
        if not path.exists():
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}_backup_{stamp}")
        shutil.move(str(path), str(backup_path))
        print(f"Backup erstellt: {backup_path}")

    @staticmethod
    def write_data_yaml(root: Path) -> None:
        (root / "data.yaml").write_text(
            "\n".join(
                [
                    f"path: {root}",
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


class PairCollector:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def collect(self, source_root: Path) -> list[ImageLabelPair]:
        image_root = source_root / "images"
        label_root = source_root / "labels"

        if not image_root.exists():
            raise FileNotFoundError(f"Bildordner fehlt: {image_root}")

        if not label_root.exists():
            raise FileNotFoundError(f"Labelordner fehlt: {label_root}")

        image_paths = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        pairs: list[ImageLabelPair] = []
        missing_labels: list[Path] = []

        for image_path in image_paths:
            label_candidates = list(label_root.rglob(f"{image_path.stem}.txt"))

            if not label_candidates:
                missing_labels.append(image_path)
                continue

            if len(label_candidates) > 1:
                raise RuntimeError(
                    f"Mehrere Labels für {image_path.name}: "
                    + ", ".join(str(path) for path in label_candidates)
                )

            pairs.append(
                ImageLabelPair(
                    image_path=image_path,
                    label_path=label_candidates[0],
                )
            )

        if missing_labels:
            print("Fehlende Labels:")
            for path in missing_labels[:50]:
                print(f"  {path}")
            raise RuntimeError(f"{len(missing_labels)} Bilder ohne Label.")

        if not pairs:
            raise RuntimeError(f"Keine Bild/Label-Paare gefunden in: {source_root}")

        return pairs


class DatasetSplitter:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def split(self, pairs: list[ImageLabelPair]) -> list[SplitPair]:
        shuffled = list(pairs)
        random.seed(self.config.seed)
        random.shuffle(shuffled)

        total_count = len(shuffled)
        train_count = int(round(total_count * self.config.train_ratio))
        val_count = int(round(total_count * self.config.val_ratio))

        if train_count < 1:
            train_count = 1

        if val_count < 1 and total_count >= 3:
            val_count = 1

        if train_count + val_count >= total_count and total_count >= 3:
            train_count = total_count - 2
            val_count = 1

        split_pairs: list[SplitPair] = []

        for pair in shuffled[:train_count]:
            split_pairs.append(SplitPair(pair.image_path, pair.label_path, "train"))

        for pair in shuffled[train_count:train_count + val_count]:
            split_pairs.append(SplitPair(pair.image_path, pair.label_path, "val"))

        for pair in shuffled[train_count + val_count:]:
            split_pairs.append(SplitPair(pair.image_path, pair.label_path, "test"))

        return split_pairs


class ObbGeometry:
    @staticmethod
    def compute_yaw(points: np.ndarray) -> float:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)

        edge_01 = points[1] - points[0]
        edge_12 = points[2] - points[1]

        len_01 = float(np.linalg.norm(edge_01))
        len_12 = float(np.linalg.norm(edge_12))

        edge = edge_01 if len_01 >= len_12 else edge_12

        if float(np.linalg.norm(edge)) < 1e-6:
            return 0.0

        return float(np.arctan2(edge[1], edge[0]))

    @staticmethod
    def fixed_square_from_center_yaw(center: np.ndarray, yaw: float, size_px: float) -> np.ndarray:
        half = float(size_px) * 0.5

        local = np.asarray(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32,
        )

        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        rotation = np.asarray(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ],
            dtype=np.float32,
        )

        return local @ rotation.T + center.astype(np.float32)

    @staticmethod
    def clamp_to_image(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        clamped = points.copy().astype(np.float32)
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, float(image_width - 1))
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, float(image_height - 1))
        return clamped


class YoloObbLabelService:
    def __init__(self, fixed_box_size_px: float) -> None:
        self.fixed_box_size_px = float(fixed_box_size_px)

    def parse_pixel_labels(self, label_path: Path, image_width: int, image_height: int) -> list[ObbLabel]:
        labels: list[ObbLabel] = []

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 9:
                raise RuntimeError(f"{label_path}:{line_number}: erwartet 9 Werte, gefunden {len(parts)}")

            try:
                class_id = int(float(parts[0]))
                coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)
            except ValueError as exc:
                raise RuntimeError(f"{label_path}:{line_number}: ungültiger Zahlenwert") from exc

            if class_id not in (0, 1):
                raise RuntimeError(f"{label_path}:{line_number}: ungültige class_id {class_id}")

            for value in coords.reshape(-1):
                if value < 0.0 or value > 1.0:
                    raise RuntimeError(f"{label_path}:{line_number}: Koordinate außerhalb 0..1: {value}")

            points = coords.copy()
            points[:, 0] *= float(image_width)
            points[:, 1] *= float(image_height)

            if self.fixed_box_size_px > 0.0:
                center = np.mean(points, axis=0)
                yaw = ObbGeometry.compute_yaw(points)
                points = ObbGeometry.fixed_square_from_center_yaw(center, yaw, self.fixed_box_size_px)
                points = ObbGeometry.clamp_to_image(points, image_width, image_height)

            labels.append(
                ObbLabel(
                    class_id=class_id,
                    points=points.astype(np.float32),
                )
            )

        return labels

    def labels_to_yolo_text(self, labels: list[ObbLabel], image_width: int, image_height: int) -> str:
        lines: list[str] = []

        for label in labels:
            normalized = label.points.copy().astype(np.float32)
            normalized[:, 0] /= float(image_width)
            normalized[:, 1] /= float(image_height)
            normalized = np.clip(normalized, 0.0, 1.0)

            values = [str(label.class_id)] + [f"{value:.6f}" for value in normalized.reshape(-1)]
            lines.append(" ".join(values))

        return "\n".join(lines) + ("\n" if lines else "")


class FullDatasetWriter:
    def __init__(self, config: DatasetConfig, label_service: YoloObbLabelService) -> None:
        self.config = config
        self.label_service = label_service

    def write(self, split_pairs: list[SplitPair]) -> None:
        DatasetPathService.backup_existing(self.config.full_target_root)

        for split in ("train", "val", "test"):
            (self.config.full_target_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.config.full_target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for pair in split_pairs:
            self.write_pair(pair)

        DatasetPathService.write_data_yaml(self.config.full_target_root)

    def write_pair(self, pair: SplitPair) -> None:
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Bild konnte nicht gelesen werden: {pair.image_path}")

        image_height, image_width = image.shape[:2]
        labels = self.label_service.parse_pixel_labels(pair.label_path, image_width, image_height)

        target_image = self.config.full_target_root / "images" / pair.split / pair.image_path.name
        target_label = self.config.full_target_root / "labels" / pair.split / f"{pair.image_path.stem}.txt"

        shutil.copy2(pair.image_path, target_image)
        target_label.write_text(
            self.label_service.labels_to_yolo_text(labels, image_width, image_height),
            encoding="utf-8",
        )


class TilePlanner:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def create_windows(self, image_width: int, image_height: int) -> list[TileWindow]:
        tile_size = self.config.tile_size_px
        step = max(int(round(tile_size * (1.0 - self.config.tile_overlap_ratio))), 1)

        x_starts = self.compute_starts(image_width, tile_size, step)
        y_starts = self.compute_starts(image_height, tile_size, step)

        return [
            TileWindow(x_min=x_start, y_min=y_start, size=tile_size)
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


class TiledDatasetWriter:
    def __init__(self, config: DatasetConfig, label_service: YoloObbLabelService) -> None:
        self.config = config
        self.label_service = label_service
        self.tile_planner = TilePlanner(config)

    def write(self) -> None:
        DatasetPathService.backup_existing(self.config.tiled_target_root)

        for split in ("train", "val", "test"):
            (self.config.tiled_target_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.config.tiled_target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        summary: dict[str, dict[str, int]] = {}

        for split in ("train", "val", "test"):
            summary[split] = self.write_split(split)

        DatasetPathService.write_data_yaml(self.config.tiled_target_root)
        self.print_summary(summary)

    def write_split(self, split: str) -> dict[str, int]:
        image_dir = self.config.full_target_root / "images" / split
        label_dir = self.config.full_target_root / "labels" / split

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in DatasetPathService.IMAGE_SUFFIXES
        )

        written_tiles = 0
        positive_tiles = 0
        negative_tiles = 0
        written_objects = 0

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                raise RuntimeError(f"Label fehlt: {label_path}")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise RuntimeError(f"Bild konnte nicht gelesen werden: {image_path}")

            image_height, image_width = image.shape[:2]
            labels = self.label_service.parse_pixel_labels(label_path, image_width, image_height)
            windows = self.tile_planner.create_windows(image_width, image_height)

            negative_count_for_image = 0

            for tile_index, window in enumerate(windows):
                tile_labels = self.labels_inside_tile(labels, window)
                is_positive = len(tile_labels) > 0

                if not is_positive:
                    if negative_count_for_image >= self.config.max_negative_tiles_per_image:
                        continue
                    negative_count_for_image += 1

                tile_image = image[window.y_min:window.y_max, window.x_min:window.x_max]

                if tile_image.shape[0] != self.config.tile_size_px or tile_image.shape[1] != self.config.tile_size_px:
                    continue

                tile_stem = f"{image_path.stem}_tile_{tile_index:04d}_x{window.x_min}_y{window.y_min}"
                target_image = self.config.tiled_target_root / "images" / split / f"{tile_stem}.png"
                target_label = self.config.tiled_target_root / "labels" / split / f"{tile_stem}.txt"

                cv2.imwrite(str(target_image), tile_image)
                target_label.write_text(
                    self.label_service.labels_to_yolo_text(tile_labels, window.size, window.size),
                    encoding="utf-8",
                )

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
            shifted = label.points.copy().astype(np.float32)
            shifted[:, 0] -= window.x_min
            shifted[:, 1] -= window.y_min

            if not self.is_fully_inside_tile(shifted, window.size):
                continue

            area = abs(float(cv2.contourArea(shifted.astype(np.float32))))

            if area < self.config.min_object_area_px2_in_tile:
                continue

            tile_labels.append(
                ObbLabel(
                    class_id=label.class_id,
                    points=shifted,
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

    def print_summary(self, summary: dict[str, dict[str, int]]) -> None:
        print()
        print("V4m RacksV2 Tiled Dataset erstellt.")
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

    def validate(self, root: Path) -> None:
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split

            if not image_dir.exists():
                raise RuntimeError(f"Bildsplit fehlt: {image_dir}")

            if not label_dir.exists():
                raise RuntimeError(f"Labelsplit fehlt: {label_dir}")

            images = [
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            ]

            labels = list(label_dir.glob("*.txt"))

            if len(images) != len(labels):
                raise RuntimeError(f"{root.name}/{split}: images={len(images)}, labels={len(labels)}")

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
        source_root=Path.home() / "ros2_ws/bev_dataset/cvat_exports/racks_v2_corrected_standard",
        full_target_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v4m_racksv2_full",
        tiled_target_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v4m_racksv2_tiled",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        seed=42,
        tile_size_px=1024,
        tile_overlap_ratio=0.25,
        max_negative_tiles_per_image=2,
        min_object_area_px2_in_tile=50.0,
        fixed_box_size_px=105.0,
    )

    print("Quelle:", config.source_root)
    print("Fixed box size px:", config.fixed_box_size_px)

    collector = PairCollector()
    pairs = collector.collect(config.source_root)
    print(f"Gefundene Bild/Label-Paare: {len(pairs)}")

    splitter = DatasetSplitter(config)
    split_pairs = splitter.split(pairs)

    for split in ("train", "val", "test"):
        print(f"{split}: {sum(1 for pair in split_pairs if pair.split == split)} Bilder")

    label_service = YoloObbLabelService(
        fixed_box_size_px=config.fixed_box_size_px,
    )

    full_writer = FullDatasetWriter(config, label_service)
    full_writer.write(split_pairs)

    tiled_writer = TiledDatasetWriter(config, label_service)
    tiled_writer.write()

    validator = DatasetValidator()
    validator.validate(config.full_target_root)
    validator.validate(config.tiled_target_root)

    print()
    print("Fertig.")
    print(f"V4m Full:  {config.full_target_root}")
    print(f"V4m Tiled: {config.tiled_target_root}")
    print()
    print("Training:")
    print(f"data={config.tiled_target_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
