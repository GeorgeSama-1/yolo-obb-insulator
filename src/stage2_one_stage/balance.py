from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import random
import shutil

from PIL import Image, ImageEnhance
import yaml

from src.data_tools.augment_yolo_obb import (
    YoloObbAnnotation,
    format_yolo_obb_line,
    parse_yolo_obb_line,
)


def _load_dataset_yaml(dataset_root: str | Path) -> dict:
    dataset_root = Path(dataset_root)
    return yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8")) or {}


def _class_names(dataset_root: str | Path) -> dict[int, str]:
    data = _load_dataset_yaml(dataset_root)
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        return {index: name for index, name in enumerate(raw_names)}
    return {int(key): value for key, value in raw_names.items()}


def _copy_dataset_yaml(input_root: Path, output_root: Path) -> None:
    data = _load_dataset_yaml(input_root)
    data["path"] = str(output_root.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _load_annotations(label_path: Path) -> list[YoloObbAnnotation]:
    if not label_path.exists():
        return []
    return [
        parse_yolo_obb_line(line)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _save_annotations(annotations: list[YoloObbAnnotation], label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(format_yolo_obb_line(annotation) for annotation in annotations)
    label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _copy_split(input_root: Path, output_root: Path, split: str) -> None:
    input_images = input_root / "images" / split
    input_labels = input_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(path for path in input_images.iterdir() if path.is_file()):
        shutil.copy2(image_path, output_images / image_path.name)
        label_path = input_labels / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, output_labels / label_path.name)


def _images_for_split(dataset_root: Path, split: str) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.is_file())


def _image_contains_class(annotations: list[YoloObbAnnotation], class_id: int) -> bool:
    return any(annotation.class_id == class_id for annotation in annotations)


def build_stage2_class_balance_report(dataset_root: str | Path) -> dict:
    dataset_root = Path(dataset_root)
    names = _class_names(dataset_root)
    instance_counts = Counter({name: 0 for name in names.values()})
    image_counts = Counter({name: 0 for name in names.values()})

    labels_root = dataset_root / "labels"
    for split_dir in sorted(path for path in labels_root.iterdir() if path.is_dir()):
        for label_path in sorted(path for path in split_dir.iterdir() if path.is_file() and path.suffix == ".txt"):
            annotations = _load_annotations(label_path)
            seen_in_image: set[int] = set()
            for annotation in annotations:
                label_name = names[annotation.class_id]
                instance_counts[label_name] += 1
                seen_in_image.add(annotation.class_id)
            for class_id in seen_in_image:
                image_counts[names[class_id]] += 1

    normal_instances = instance_counts.get("normal", 0)
    abnormal_instances = instance_counts.get("abnormal", 0)
    ratio = None
    if normal_instances:
        ratio = abnormal_instances / normal_instances

    return {
        "instance_counts": dict(instance_counts),
        "image_counts": dict(image_counts),
        "instance_ratio": {"abnormal_over_normal": ratio},
    }


def write_stage2_class_balance_reports(
    dataset_root: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict:
    report = build_stage2_class_balance_report(dataset_root)

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for label in ("normal", "abnormal"):
        rows.append(
            f"| {label} | {report['instance_counts'].get(label, 0)} | {report['image_counts'].get(label, 0)} |"
        )
    markdown = "\n".join(
        [
            "# Stage2 Class Balance",
            "",
            "| class | instances | images |",
            "| --- | ---: | ---: |",
            *rows,
            "",
            f"- abnormal/normal instance ratio: {report['instance_ratio']['abnormal_over_normal']}",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return report


def _transform_points_light(points: list[tuple[float, float]], angle_deg: float, scale: float) -> list[tuple[float, float]]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad) * scale
    sin_a = math.sin(angle_rad) * scale
    transformed = []
    for x, y in points:
        dx = x - 0.5
        dy = y - 0.5
        new_x = 0.5 + dx * cos_a - dy * sin_a
        new_y = 0.5 + dx * sin_a + dy * cos_a
        transformed.append((_clamp(new_x), _clamp(new_y)))
    return transformed


def _scale_image_about_center(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    if abs(scale - 1.0) < 1e-6:
        return image.copy()

    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.Resampling.BICUBIC,
    )
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))

    if scale >= 1.0:
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))

    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _light_augment_image(image: Image.Image, rng: random.Random) -> tuple[Image.Image, float, float]:
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.95, 1.05)
    brightness = rng.uniform(0.92, 1.08)

    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    scaled = _scale_image_about_center(rotated, scale)
    brightened = ImageEnhance.Brightness(scaled).enhance(brightness)
    return brightened, angle, scale


def generate_stage2_abnormal_boost_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    abnormal_target_per_image: int = 3,
    seed: int = 42,
) -> None:
    del seed
    if abnormal_target_per_image < 1:
        raise ValueError("abnormal_target_per_image must be at least 1")

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    _copy_dataset_yaml(dataset_root, output_root)
    names = _class_names(dataset_root)
    abnormal_class_id = next(class_id for class_id, name in names.items() if name == "abnormal")

    for split_dir in sorted((dataset_root / "labels").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split != "train":
            _copy_split(dataset_root, output_root, split)
            continue

        output_images = output_root / "images" / split
        output_labels = output_root / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image_path in _images_for_split(dataset_root, split):
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            annotations = _load_annotations(label_path)

            shutil.copy2(image_path, output_images / image_path.name)
            _save_annotations(annotations, output_labels / f"{image_path.stem}.txt")

            if not _image_contains_class(annotations, abnormal_class_id):
                continue

            for copy_index in range(1, abnormal_target_per_image):
                copy_name = f"{image_path.stem}_abn_boost_{copy_index:02d}{image_path.suffix.lower()}"
                copy_label = f"{image_path.stem}_abn_boost_{copy_index:02d}.txt"
                shutil.copy2(image_path, output_images / copy_name)
                _save_annotations(annotations, output_labels / copy_label)


def generate_stage2_abnormal_light_aug_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    abnormal_target_per_image: int = 3,
    seed: int = 42,
) -> None:
    if abnormal_target_per_image < 1:
        raise ValueError("abnormal_target_per_image must be at least 1")

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    _copy_dataset_yaml(dataset_root, output_root)
    names = _class_names(dataset_root)
    abnormal_class_id = next(class_id for class_id, name in names.items() if name == "abnormal")
    rng = random.Random(seed)

    for split_dir in sorted((dataset_root / "labels").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split != "train":
            _copy_split(dataset_root, output_root, split)
            continue

        output_images = output_root / "images" / split
        output_labels = output_root / "labels" / split
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image_path in _images_for_split(dataset_root, split):
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            annotations = _load_annotations(label_path)

            original_image = Image.open(image_path).convert("RGB")
            original_image.save(output_images / image_path.name)
            _save_annotations(annotations, output_labels / f"{image_path.stem}.txt")

            if not _image_contains_class(annotations, abnormal_class_id):
                continue

            for copy_index in range(1, abnormal_target_per_image):
                aug_image, angle_deg, scale = _light_augment_image(original_image, rng)
                aug_annotations = [
                    YoloObbAnnotation(
                        class_id=annotation.class_id,
                        points=_transform_points_light(annotation.points, angle_deg, scale),
                    )
                    for annotation in annotations
                ]
                image_name = f"{image_path.stem}_abn_light_{copy_index:02d}{image_path.suffix.lower()}"
                label_name = f"{image_path.stem}_abn_light_{copy_index:02d}.txt"
                aug_image.save(output_images / image_name)
                _save_annotations(aug_annotations, output_labels / label_name)
