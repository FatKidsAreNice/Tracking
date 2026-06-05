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
    base_full_root: Path
    corrected_root: Path
    v3_full_root: Path
    v3_tiled_root: Path
    seed: int
    tile_size_px: int
    tile_overlap_ratio: float
    train_object_jitter_tiles_per_object: int
    train_object_jitter_px: int
    max_negative_tiles_per_image: int
    min_object_area_px2_in_tile: float


@dataclass(frozen=True)
class ImageLabelPair:
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


class DatasetPaths:
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


class YoloObbLabelValidator:
    def validate_file(self, label_path: Path) -> None:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 9:
                raise RuntimeError(f"{label_path}:{line_number}: erwartet 9 Werte, gefunden {len(parts)}")

            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise RuntimeError(f"{label_path}:{line_number}: ungültiger Zahlenwert") from exc

            if class_id not in (0, 1):
                raise RuntimeError(f"{label_path}:{line_number}: ungültige class_id {class_id}")

            for value in coords:
                if value < 0.0 or value > 1.0:
                    raise RuntimeError(f"{label_path}:{line_number}: Koordinate außerhalb 0..1: {value}")


class YoloObbLabelParser:
    def __init__(self) -> None:
        self.validator = YoloObbLabelValidator()

    def parse_pixel_labels(self, label_path: Path, image_width: int, image_height: int) -> list[ObbLabel]:
        self.validator.validate_file(label_path)

        labels: list[ObbLabel] = []

        for line in label_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()
            class_id = int(parts[0])
            coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)

            coords[:, 0] *= image_width
            coords[:, 1] *= image_height

            labels.append(
                ObbLabel(
                    class_id=class_id,
                    points=coords.astype(np.float32),
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


class PairCollector:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def collect_split_pairs(self, root: Path, split: str) -> list[ImageLabelPair]:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if not image_dir.exists() or not label_dir.exists():
            return []

        pairs: list[ImageLabelPair] = []

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                raise RuntimeError(f"Label fehlt für Bild: {image_path}")

            pairs.append(
                ImageLabelPair(
                    image_path=image_path,
                    label_path=label_path,
                    split=split,
                )
            )

        return pairs

    def collect_all_base_pairs(self, root: Path) -> list[ImageLabelPair]:
        pairs: list[ImageLabelPair] = []

        for split in ("train", "val", "test"):
            pairs.extend(self.collect_split_pairs(root, split))

        if not pairs:
            raise RuntimeError(f"Keine Base-Paare gefunden in: {root}")

        return pairs

    def collect_corrected_pairs(self, root: Path) -> list[ImageLabelPair]:
        pairs = self.collect_split_pairs(root, "train")

        if not pairs:
            raise RuntimeError(f"Keine korrigierten CVAT-Paare gefunden in: {root}")

        return pairs


class V3FullDatasetBuilder:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.collector = PairCollector()
        self.validator = YoloObbLabelValidator()

    def build(self) -> None:
        DatasetPaths.backup_existing(self.config.v3_full_root)

        for split in ("train", "val", "test"):
            (self.config.v3_full_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.config.v3_full_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        base_pairs = self.collector.collect_all_base_pairs(self.config.base_full_root)
        corrected_pairs = self.collector.collect_corrected_pairs(self.config.corrected_root)
        corrected_stems = {pair.image_path.stem for pair in corrected_pairs}

        copied_base = 0
        replaced_base = 0

        for pair in base_pairs:
            if pair.image_path.stem in corrected_stems:
                replaced_base += 1
                continue

            self.copy_pair(pair, pair.split)
            copied_base += 1

        copied_corrected = 0

        for pair in corrected_pairs:
            self.copy_pair(pair, "train")
            copied_corrected += 1

        DatasetPaths.write_data_yaml(self.config.v3_full_root)

        print()
        print("V3 Full Dataset erstellt.")
        print(f"Base übernommen:     {copied_base}")
        print(f"Base ersetzt:        {replaced_base}")
        print(f"CVAT korrigiert:     {copied_corrected}")
        print(f"Ziel:                {self.config.v3_full_root}")

    def copy_pair(self, pair: ImageLabelPair, target_split: str) -> None:
        self.validator.validate_file(pair.label_path)

        target_image = self.config.v3_full_root / "images" / target_split / pair.image_path.name
        target_label = self.config.v3_full_root / "labels" / target_split / pair.label_path.name

        shutil.copy2(pair.image_path, target_image)
        shutil.copy2(pair.label_path, target_label)


class TilePlanner:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def create_windows(self, image_width: int, image_height: int, labels: list[ObbLabel], split: str) -> list[TileWindow]:
        windows = self.create_grid_windows(image_width, image_height)

        if split == "train":
            windows.extend(self.create_object_jitter_windows(image_width, image_height, labels))

        unique: dict[tuple[int, int], TileWindow] = {}

        for window in windows:
            unique[(window.x_min, window.y_min)] = window

        return list(unique.values())

    def create_grid_windows(self, image_width: int, image_height: int) -> list[TileWindow]:
        tile_size = self.config.tile_size_px
        step = max(int(round(tile_size * (1.0 - self.config.tile_overlap_ratio))), 1)

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

        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        return starts

    def clamp_start(self, start: int, image_size: int, tile_size: int) -> int:
        if image_size <= tile_size:
            return 0

        return max(0, min(start, image_size - tile_size))


class V3TiledDatasetBuilder:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.collector = PairCollector()
        self.parser = YoloObbLabelParser()
        self.tile_planner = TilePlanner(config)

    def build(self) -> None:
        random.seed(self.config.seed)
        DatasetPaths.backup_existing(self.config.v3_tiled_root)

        for split in ("train", "val", "test"):
            (self.config.v3_tiled_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.config.v3_tiled_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        summary: dict[str, dict[str, int]] = {}

        for split in ("train", "val", "test"):
            pairs = self.collector.collect_split_pairs(self.config.v3_full_root, split)
            summary[split] = self.write_split(split, pairs)

        DatasetPaths.write_data_yaml(self.config.v3_tiled_root)
        self.print_summary(summary)

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
            labels = self.parser.parse_pixel_labels(pair.label_path, image_width, image_height)
            windows = self.tile_planner.create_windows(image_width, image_height, labels, split)

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

                tile_stem = f"{pair.image_path.stem}_tile_{tile_index:04d}_x{window.x_min}_y{window.y_min}"
                tile_image_path = self.config.v3_tiled_root / "images" / split / f"{tile_stem}.png"
                tile_label_path = self.config.v3_tiled_root / "labels" / split / f"{tile_stem}.txt"

                cv2.imwrite(str(tile_image_path), tile_image)
                tile_label_path.write_text(
                    self.parser.labels_to_yolo_text(tile_labels, window.size, window.size),
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
        print("V3 Tiled Dataset erstellt.")
        print(f"Ziel: {self.config.v3_tiled_root}")

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

            if not image_dir.exists() or not label_dir.exists():
                raise RuntimeError(f"Split fehlt in {root}: {split}")

            images = [
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            ]
            labels = list(label_dir.glob("*.txt"))

            if len(images) != len(labels):
                raise RuntimeError(f"{root.name}/{split}: images={len(images)}, labels={len(labels)}")

            validator = YoloObbLabelValidator()

            for label_path in labels:
                validator.validate_file(label_path)


def main() -> None:
    config = DatasetConfig(
        base_full_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v2_full",
        corrected_root=Path.home() / "ros2_ws/bev_dataset/cvat_exports/rack_bev_v3_review_corrected",
        v3_full_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v3_full",
        v3_tiled_root=Path.home() / "ros2_ws/bev_dataset/yolo_obb_v3_tiled",
        seed=42,
        tile_size_px=1024,
        tile_overlap_ratio=0.25,
        train_object_jitter_tiles_per_object=2,
        train_object_jitter_px=256,
        max_negative_tiles_per_image=2,
        min_object_area_px2_in_tile=100.0,
    )

    full_builder = V3FullDatasetBuilder(config)
    full_builder.build()

    tiled_builder = V3TiledDatasetBuilder(config)
    tiled_builder.build()

    validator = DatasetValidator()
    validator.validate(config.v3_full_root)
    validator.validate(config.v3_tiled_root)

    print()
    print("Fertig.")
    print(f"V3 Full:  {config.v3_full_root}")
    print(f"V3 Tiled: {config.v3_tiled_root}")
    print()
    print("Training:")
    print(f"data={config.v3_tiled_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
