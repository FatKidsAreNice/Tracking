#!/usr/bin/env python3
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CleanImportConfig:
    source_root: Path
    target_root: Path


class ObbPointOrderer:
    @staticmethod
    def normalize_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)

        if abs(float(cv2.contourArea(points))) < 1e-6:
            raise ValueError("OBB-Fläche ist zu klein oder degeneriert.")

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.float32)

        return ObbPointOrderer.order_as_tl_tr_br_bl(box)

    @staticmethod
    def order_as_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)

        sorted_by_y = points[np.argsort(points[:, 1])]
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]

        top_sorted = top_points[np.argsort(top_points[:, 0])]
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]

        top_left = top_sorted[0]
        top_right = top_sorted[1]
        bottom_left = bottom_sorted[0]
        bottom_right = bottom_sorted[1]

        return np.asarray(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def clamp_normalized(points: np.ndarray) -> np.ndarray:
        clamped = points.copy().astype(np.float32)
        clamped[:, 0] = np.clip(clamped[:, 0], 0.0, 1.0)
        clamped[:, 1] = np.clip(clamped[:, 1], 0.0, 1.0)
        return clamped


class LabelCleaner:
    def clean_label_file(self, source_label: Path, target_label: Path) -> tuple[int, int]:
        output_lines: list[str] = []
        dropped_count = 0

        for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 9:
                dropped_count += 1
                print(f"DROP {source_label.name}:{line_number}: erwartet 9 Werte, gefunden {len(parts)}")
                continue

            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                dropped_count += 1
                print(f"DROP {source_label.name}:{line_number}: ungültiger Zahlenwert")
                continue

            if class_id not in (0, 1):
                dropped_count += 1
                print(f"DROP {source_label.name}:{line_number}: unbekannte class_id {class_id}")
                continue

            points = np.asarray(coords, dtype=np.float32).reshape(4, 2)

            try:
                normalized_points = ObbPointOrderer.normalize_points(points)
                normalized_points = ObbPointOrderer.clamp_normalized(normalized_points)
            except ValueError as exc:
                dropped_count += 1
                print(f"DROP {source_label.name}:{line_number}: {exc}")
                continue

            output_line = " ".join(
                [str(class_id)]
                + [f"{value:.6f}" for value in normalized_points.reshape(-1)]
            )
            output_lines.append(output_line)

        target_label.write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8",
        )

        return len(output_lines), dropped_count


class CleanPseudoLabelImportBuilder:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, config: CleanImportConfig) -> None:
        self.config = config
        self.source_image_dir = self.config.source_root / "images" / "train"
        self.source_label_dir = self.config.source_root / "labels" / "train"
        self.target_image_dir = self.config.target_root / "images" / "train"
        self.target_label_dir = self.config.target_root / "labels" / "train"
        self.label_cleaner = LabelCleaner()

    def run(self) -> None:
        self.validate_source()
        self.prepare_target()
        copied_images, copied_labels, kept_objects, dropped_objects = self.copy_and_clean()
        self.write_data_yaml()
        self.print_summary(copied_images, copied_labels, kept_objects, dropped_objects)

    def validate_source(self) -> None:
        if not self.source_image_dir.exists():
            raise FileNotFoundError(f"Bildquelle fehlt: {self.source_image_dir}")

        if not self.source_label_dir.exists():
            raise FileNotFoundError(f"Labelquelle fehlt: {self.source_label_dir}")

    def prepare_target(self) -> None:
        if self.config.target_root.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = self.config.target_root.with_name(
                f"{self.config.target_root.name}_backup_{timestamp}"
            )
            shutil.move(str(self.config.target_root), str(backup_root))
            print(f"Bestehender Zielordner gesichert: {backup_root}")

        self.target_image_dir.mkdir(parents=True, exist_ok=True)
        self.target_label_dir.mkdir(parents=True, exist_ok=True)

    def copy_and_clean(self) -> tuple[int, int, int, int]:
        copied_images = 0
        copied_labels = 0
        kept_objects = 0
        dropped_objects = 0

        image_paths = sorted(
            path for path in self.source_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        for image_path in image_paths:
            source_label = self.source_label_dir / f"{image_path.stem}.txt"

            if not source_label.exists():
                print(f"SKIP {image_path.name}: Label fehlt")
                continue

            target_image = self.target_image_dir / image_path.name
            target_label = self.target_label_dir / source_label.name

            shutil.copy2(image_path, target_image)
            kept_count, dropped_count = self.label_cleaner.clean_label_file(source_label, target_label)

            copied_images += 1
            copied_labels += 1
            kept_objects += kept_count
            dropped_objects += dropped_count

        return copied_images, copied_labels, kept_objects, dropped_objects

    def write_data_yaml(self) -> None:
        data_yaml = self.config.target_root / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.config.target_root}",
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

    def print_summary(
        self,
        copied_images: int,
        copied_labels: int,
        kept_objects: int,
        dropped_objects: int,
    ) -> None:
        print()
        print("Clean Import erstellt.")
        print(f"Quelle: {self.config.source_root}")
        print(f"Ziel:   {self.config.target_root}")
        print(f"Bilder: {copied_images}")
        print(f"Labels: {copied_labels}")
        print(f"Objekte behalten: {kept_objects}")
        print(f"Objekte verworfen: {dropped_objects}")
        print()
        print("Importiere im Labeltool nur diesen Ordner:")
        print(f"  {self.config.target_root}")
        print()
        print("Nicht importieren:")
        print(f"  {self.config.source_root / 'overlays'}")
        print(f"  {self.config.source_root / 'json'}")


def main() -> None:
    config = CleanImportConfig(
        source_root=Path.home() / "ros2_ws/bev_dataset/pseudo_labels_v3_review",
        target_root=Path.home() / "ros2_ws/bev_dataset/pseudo_labels_v3_review_import_clean",
    )

    builder = CleanPseudoLabelImportBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()
