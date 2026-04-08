from pathlib import Path

from PIL import Image, ImageDraw


def color_for_class(class_name: str) -> tuple[int, int, int]:
    palette = {
        "insulator": (0, 180, 255),
        "normal": (0, 200, 0),
        "abnormal": (220, 30, 30),
    }
    return palette.get(class_name, (255, 180, 0))


def draw_obb_overlay(
    image_path: str | Path,
    detections: list[dict],
    output_path: str | Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for detection in detections:
        points = [tuple(point) for point in detection["points"]]
        draw.polygon(points, outline=color_for_class(detection["label"]), width=3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def load_yolo_obb_detections(
    lines: list[str],
    class_names: dict[int, str],
    image_size: tuple[int, int],
) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []
    for line in lines:
        values = line.strip().split()
        if not values:
            continue
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
        points = []
        for index in range(0, len(coords), 2):
            x = coords[index] * width
            y = coords[index + 1] * height
            points.append((x, y))
        detections.append({"label": class_names[class_id], "points": points})
    return detections


def visualize_yolo_obb_split(
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path,
    class_names: dict[int, str],
    limit: int | None = None,
) -> list[Path]:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    if limit is not None:
        image_paths = image_paths[:limit]

    written: list[Path] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        image = Image.open(image_path)
        detections = load_yolo_obb_detections(
            lines,
            class_names=class_names,
            image_size=image.size,
        )
        output_path = output_root / f"{image_path.stem}_overlay.jpg"
        written.append(draw_obb_overlay(image_path, detections, output_path))
    return written


def _to_list(value):
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_ultralytics_obb_detections(result, class_names: dict[int, str]) -> list[dict]:
    if getattr(result, "obb", None) is None:
        return []

    polygons = _to_list(result.obb.xyxyxyxy)
    classes = _to_list(result.obb.cls)
    detections: list[dict] = []
    for polygon, class_id in zip(polygons, classes):
        points = [tuple(point) for point in polygon]
        detections.append({"label": class_names[int(class_id)], "points": points})
    return detections


def visualize_prediction_results(
    results: list,
    output_dir: str | Path,
    class_names: dict[int, str],
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for result in results:
        image_path = Path(result.path)
        detections = load_ultralytics_obb_detections(result, class_names=class_names)
        output_path = output_root / f"{image_path.stem}_pred.jpg"
        written.append(draw_obb_overlay(image_path, detections, output_path))
    return written
