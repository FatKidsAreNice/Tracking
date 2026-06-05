#!/usr/bin/env python3
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class NormalizeConfig:
    source_root: Path
    target_root: Path


class RacksV2ExportNormalizer:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, config: NormalizeConfig) -> None:
        self.config = config

    def run(self) -> None:
        self.prepare_target()
        image_pairs = self.collect_image_label_pairs()

        copied_images = 0
        copied_labels = 0
        converted_objects = 0

        for image_path, label_path in image_pairs:
            target_image = self.config.target_root / "images" / "train" / image_path.name
            target_label = self.config.target_root / "labels" / "train" / label_path.name

            shutil.copy2(image_path, target_image)
            object_count = self.convert_label_file(label_path, target_label)

            copied_images += 1
            copied_labels += 1
            converted_objects += object_count

        self.write_data_yaml()
        self.write_train_txt()

        print()
        print("RacksV2 Export normalisiert.")
        print(f"Quelle: {self.config.source_root}")
        print(f"Ziel:   {self.config.target_root}")
        print(f"Bilder: {copied_images}")
        print(f"Labels: {copied_labels}")
        print(f"Objekte konvertiert: {converted_objects}")
        print()
        print("Neues Klassenmapping:")
        print("  0 = rack_side_visible")
        print("  1 = rack_top_visible")

    def prepare_target(self) -> None:
        if self.config.target_root.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = self.config.target_root.with_name(
                f"{self.config.target_root.name}_backup_{stamp}"
            )
            shutil.move(str(self.config.target_root), str(backup_root))
            print(f"Bestehender Zielordner gesichert: {backup_root}")

        (self.config.target_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.config.target_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    def collect_image_label_pairs(self) -> list[tuple[Path, Path]]:
        image_root = self.config.source_root / "images"
        label_root = self.config.source_root / "labels"

        if not image_root.exists():
            raise FileNotFoundError(f"Bildordner fehlt: {image_root}")

        if not label_root.exists():
            raise FileNotFoundError(f"Labelordner fehlt: {label_root}")

        image_paths = sorted(
            path for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        pairs: list[tuple[Path, Path]] = []
        missing_labels: list[Path] = []

        for image_path in image_paths:
            label_candidates = list(label_root.rglob(f"{image_path.stem}.txt"))

            if not label_candidates:
                missing_labels.append(image_path)
                continue

            if len(label_candidates) > 1:
                raise RuntimeError(
                    f"Mehrere Label-Dateien für {image_path.name}: "
                    + ", ".join(str(path) for path in label_candidates)
                )

            pairs.append((image_path, label_candidates[0]))

        if missing_labels:
            print("Fehlende Labels:")
            for image_path in missing_labels[:50]:
                print(f"  {image_path}")
            raise RuntimeError(f"{len(missing_labels)} Bilder ohne Label-Datei.")

        if not pairs:
            raise RuntimeError(f"Keine Bild/Label-Paare gefunden in {self.config.source_root}")

        return pairs

    def convert_label_file(self, source_label: Path, target_label: Path) -> int:
        output_lines: list[str] = []
        object_count = 0

        for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 9:
                raise RuntimeError(
                    f"{source_label}:{line_number}: YOLO-OBB erwartet 9 Werte, gefunden {len(parts)}"
                )

            old_class_id = int(float(parts[0]))
            new_class_id = self.convert_class_id(old_class_id, source_label, line_number)

            coords = [float(value) for value in parts[1:]]

            for value in coords:
                if value < 0.0 or value > 1.0:
                    raise RuntimeError(
                        f"{source_label}:{line_number}: Koordinate außerhalb 0..1: {value}"
                    )

            output_lines.append(
                " ".join([str(new_class_id)] + [f"{value:.6f}" for value in coords])
            )
            object_count += 1

        target_label.write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8",
        )

        return object_count

    def convert_class_id(self, old_class_id: int, source_label: Path, line_number: int) -> int:
        if old_class_id == 0:
            return 1

        if old_class_id == 1:
            return 0

        raise RuntimeError(f"{source_label}:{line_number}: unbekannte class_id {old_class_id}")

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

    def write_train_txt(self) -> None:
        image_dir = self.config.target_root / "images" / "train"
        train_txt = self.config.target_root / "train.txt"

        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

        train_txt.write_text(
            "\n".join(f"images/train/{path.name}" for path in image_paths) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    config = NormalizeConfig(
        source_root=Path.home() / "ros2_ws/bev_dataset/cvat_exports/racks_v2_corrected",
        target_root=Path.home() / "ros2_ws/bev_dataset/cvat_exports/racks_v2_corrected_standard",
    )

    normalizer = RacksV2ExportNormalizer(config)
    normalizer.run()


if __name__ == "__main__":
    main()
