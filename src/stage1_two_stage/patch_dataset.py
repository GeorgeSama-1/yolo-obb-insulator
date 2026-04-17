from __future__ import annotations

from pathlib import Path
import math
import shutil

from PIL import Image
import yaml

from src.data_tools.augment_yolo_obb import parse_yolo_obb_line
from src.stage1_two_stage.classifier_train import normalize_label
from src.stage1_two_stage.crops import padded_crop_bounds


def _class_names(dataset_root: Path) -> dict[int, str]:
    data = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8")) or {}
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        return {index: name for index, name in enumerate(raw_names)}
    return {int(key): value for key, value in raw_names.items()}


def _image_paths_for_split(dataset_root: Path, split: str) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.is_file())


def _annotation_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _obb_to_bounds(line: str, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    annotation = parse_yolo_obb_line(line)
    width, height = image_size
    xs = [point[0] * width for point in annotation.points]
    ys = [point[1] * height for point in annotation.points]
    left = max(0, math.floor(min(xs)))
    top = max(0, math.floor(min(ys)))
    right = min(width, math.ceil(max(xs)))
    bottom = min(height, math.ceil(max(ys)))
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    return padded_crop_bounds((left, top, right, bottom), image_size=image_size, padding=padding)


def _save_patch(image: Image.Image, bounds: tuple[int, int, int, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(bounds).save(output_path)


def export_stage1_patch_classifier_dataset(
    input_root: str | Path,
    output_root: str | Path,
    padding: int = 0,
    train_normal_to_abnormal_ratio: float = 2.0,
) -> None:
    if train_normal_to_abnormal_ratio <= 0:
        raise ValueError("train_normal_to_abnormal_ratio must be positive")

    input_root = Path(input_root)
    output_root = Path(output_root)
    class_names = _class_names(input_root)

    train_records: dict[str, list[Path]] = {"normal": [], "abnormal": []}

    for split in sorted(path.name for path in (input_root / "labels").iterdir() if path.is_dir()):
        for image_path in _image_paths_for_split(input_root, split):
            image = Image.open(image_path).convert("RGB")
            image_size = image.size
            label_path = input_root / "labels" / split / f"{image_path.stem}.txt"

            for index, line in enumerate(_annotation_lines(label_path)):
                annotation = parse_yolo_obb_line(line)
                label_name = normalize_label(class_names[annotation.class_id])
                patch_name = f"{image_path.stem}_{index:03d}.jpg"
                output_path = output_root / split / label_name / patch_name
                bounds = _obb_to_bounds(line, image_size=image_size, padding=padding)
                _save_patch(image, bounds, output_path)

                if split == "train":
                    train_records[label_name].append(output_path)

    normal_count = len(train_records["normal"])
    abnormal_count = len(train_records["abnormal"])
    if abnormal_count == 0:
        return

    target_abnormal_count = math.ceil(normal_count / train_normal_to_abnormal_ratio)
    extra_needed = max(0, target_abnormal_count - abnormal_count)

    for copy_index in range(extra_needed):
        source_patch = train_records["abnormal"][copy_index % abnormal_count]
        dest_patch = source_patch.with_name(f"{source_patch.stem}_os_{copy_index + 1:02d}{source_patch.suffix}")
        shutil.copy2(source_patch, dest_patch)
