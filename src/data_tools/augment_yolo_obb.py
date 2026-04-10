from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from PIL import Image, ImageEnhance, ImageOps
import yaml

from src.stage0_obb.visualize import visualize_yolo_obb_split


Point = tuple[float, float]


@dataclass(frozen=True)
class YoloObbAnnotation:
    class_id: int
    points: list[Point]


def parse_yolo_obb_line(line: str) -> YoloObbAnnotation:
    values = line.strip().split()
    if len(values) != 9:
        raise ValueError(f"Expected 9 values per YOLO OBB line, got {len(values)}: {line!r}")
    class_id = int(values[0])
    coords = [float(value) for value in values[1:]]
    points = [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
    return YoloObbAnnotation(class_id=class_id, points=points)


def format_yolo_obb_line(annotation: YoloObbAnnotation) -> str:
    flattened = [f"{value:.6f}" for point in annotation.points for value in point]
    return " ".join([str(annotation.class_id), *flattened])


def _clamp_coord(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_points(points: list[Point]) -> list[Point]:
    return [(_clamp_coord(x), _clamp_coord(y)) for x, y in points]


def apply_horizontal_flip_to_points(points: list[Point]) -> list[Point]:
    return [(1.0 - x, y) for x, y in points]


def apply_vertical_flip_to_points(points: list[Point]) -> list[Point]:
    return [(x, 1.0 - y) for x, y in points]


def apply_rotate_90_to_points(points: list[Point]) -> list[Point]:
    return [(1.0 - y, x) for x, y in points]


def apply_rotate_180_to_points(points: list[Point]) -> list[Point]:
    return [(1.0 - x, 1.0 - y) for x, y in points]


def apply_rotate_270_to_points(points: list[Point]) -> list[Point]:
    return [(y, 1.0 - x) for x, y in points]


def _load_annotations(label_path: Path) -> list[YoloObbAnnotation]:
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    return [parse_yolo_obb_line(line) for line in lines if line.strip()]


def _save_annotations(annotations: list[YoloObbAnnotation], label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(format_yolo_obb_line(annotation) for annotation in annotations)
    label_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _transform_annotations(
    annotations: list[YoloObbAnnotation],
    transform_name: str,
) -> list[YoloObbAnnotation]:
    point_transform = {
        "identity": lambda pts: pts,
        "hflip": apply_horizontal_flip_to_points,
        "vflip": apply_vertical_flip_to_points,
        "rot90": apply_rotate_90_to_points,
        "rot180": apply_rotate_180_to_points,
        "rot270": apply_rotate_270_to_points,
    }[transform_name]
    return [
        YoloObbAnnotation(
            class_id=annotation.class_id,
            points=_clamp_points(point_transform(annotation.points)),
        )
        for annotation in annotations
    ]


def _transform_image(image: Image.Image, transform_name: str) -> Image.Image:
    if transform_name == "identity":
        return image.copy()
    if transform_name == "hflip":
        return ImageOps.mirror(image)
    if transform_name == "vflip":
        return ImageOps.flip(image)
    if transform_name == "rot90":
        return image.transpose(Image.Transpose.ROTATE_270)
    if transform_name == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if transform_name == "rot270":
        return image.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported transform: {transform_name}")


def _apply_photometric_jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    brightness = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    contrast = ImageEnhance.Contrast(brightness).enhance(rng.uniform(0.8, 1.2))
    color = ImageEnhance.Color(contrast).enhance(rng.uniform(0.85, 1.15))
    return color


def _copy_dataset_yaml(input_root: Path, output_root: Path) -> None:
    input_yaml = input_root / "dataset.yaml"
    if not input_yaml.exists():
        return
    output_root.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(input_yaml.read_text(encoding="utf-8")) or {}
    data["path"] = str(output_root.resolve())
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def augment_dataset_split(
    input_root: str | Path,
    output_root: str | Path,
    split: str,
    target_per_image: int,
    seed: int = 42,
) -> list[Path]:
    if target_per_image < 1:
        raise ValueError("target_per_image must be at least 1")

    input_root = Path(input_root)
    output_root = Path(output_root)
    images_dir = input_root / "images" / split
    labels_dir = input_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    transform_pool = ["hflip", "vflip", "rot90", "rot180", "rot270"]
    written: list[Path] = []

    for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
        label_path = labels_dir / f"{image_path.stem}.txt"
        annotations = _load_annotations(label_path)

        original_image = Image.open(image_path).convert("RGB")
        original_output = output_images / image_path.name
        shutil.copy2(image_path, original_output)
        _save_annotations(
            [
                YoloObbAnnotation(class_id=annotation.class_id, points=_clamp_points(annotation.points))
                for annotation in annotations
            ],
            output_labels / f"{image_path.stem}.txt",
        )
        written.append(original_output)

        for copy_index in range(1, target_per_image):
            transform_name = rng.choice(transform_pool)
            transformed_image = _transform_image(original_image, transform_name)
            transformed_image = _apply_photometric_jitter(transformed_image, rng)
            transformed_annotations = _transform_annotations(annotations, transform_name)

            output_image_path = output_images / f"{image_path.stem}_aug_{copy_index:02d}{image_path.suffix.lower()}"
            output_label_path = output_labels / f"{image_path.stem}_aug_{copy_index:02d}.txt"
            transformed_image.save(output_image_path)
            _save_annotations(transformed_annotations, output_label_path)
            written.append(output_image_path)

    return written


def copy_unaugmented_split(input_root: str | Path, output_root: str | Path, split: str) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
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


def augment_yolo_obb_dataset(
    input_root: str | Path,
    output_root: str | Path,
    target_per_image: int,
    augment_splits: tuple[str, ...] = ("train",),
    seed: int = 42,
    preview_output_dir: str | Path | None = None,
    preview_splits: tuple[str, ...] = (),
    preview_limit_per_split: int | None = None,
) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
    _copy_dataset_yaml(input_root, output_root)

    labels_root = input_root / "labels"
    for split_dir in sorted(path for path in labels_root.iterdir() if path.is_dir()):
        split = split_dir.name
        if split in augment_splits:
            augment_dataset_split(
                input_root=input_root,
                output_root=output_root,
                split=split,
                target_per_image=target_per_image,
                seed=seed,
            )
        else:
            copy_unaugmented_split(input_root, output_root, split)

    if preview_output_dir is not None:
        dataset_yaml = output_root / "dataset.yaml"
        data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        raw_names = data.get("names", {})
        class_names = {int(key): value for key, value in raw_names.items()}
        preview_root = Path(preview_output_dir)
        for split in preview_splits:
            visualize_yolo_obb_split(
                dataset_root=output_root,
                split=split,
                output_dir=preview_root / split,
                class_names=class_names,
                limit=preview_limit_per_split,
            )
