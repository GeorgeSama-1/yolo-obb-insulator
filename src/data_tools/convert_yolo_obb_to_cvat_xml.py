from __future__ import annotations

from datetime import datetime, UTC
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

from src.data_tools.ingest import YoloObbPair, find_yolo_obb_pairs
from src.data_tools.prepare_yolo_obb import reorder_points_clockwise


def _load_yolo_obb_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _line_to_rotated_box(line: str, image_size: tuple[int, int]) -> tuple[int, dict[str, str]]:
    width, height = image_size
    parts = line.split()
    class_id = int(parts[0])
    coords = [float(value) for value in parts[1:9]]
    points = []
    for index in range(0, len(coords), 2):
        x = coords[index] * width
        y = coords[index + 1] * height
        points.append((x, y))

    ordered = reorder_points_clockwise(points)
    corners = _order_box_corners(ordered)
    top_left, top_right, bottom_right, bottom_left = corners
    center_x = sum(x for x, _ in corners) / 4
    center_y = sum(y for _, y in corners) / 4
    width_px = math.dist(top_left, top_right)
    height_px = math.dist(top_right, bottom_right)
    angle_radians = math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])

    rotation_degrees = math.degrees(angle_radians) % 360

    return class_id, {
        "xtl": f"{center_x - width_px / 2:.2f}",
        "ytl": f"{center_y - height_px / 2:.2f}",
        "xbr": f"{center_x + width_px / 2:.2f}",
        "ybr": f"{center_y + height_px / 2:.2f}",
        "rotation": f"{rotation_degrees:.2f}",
    }


def _order_box_corners(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    sums = [x + y for x, y in points]
    diffs = [y - x for x, y in points]
    top_left = points[sums.index(min(sums))]
    bottom_right = points[sums.index(max(sums))]
    top_right = points[diffs.index(min(diffs))]
    bottom_left = points[diffs.index(max(diffs))]
    return top_left, top_right, bottom_right, bottom_left


def export_cvat_xml_from_yolo_obb(
    dataset_root: str | Path,
    output_path: str | Path,
    class_names: dict[int, str],
    split: str = "train",
) -> Path:
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    pairs = [
        YoloObbPair(
            stem=image_path.stem,
            image_path=image_path,
            label_path=labels_dir / f"{image_path.stem}.txt",
        )
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file())
    ]
    return export_cvat_xml_from_yolo_obb_pairs(
        pairs=pairs,
        output_path=output_path,
        class_names=class_names,
        task_name=dataset_root.name,
    )


def export_cvat_xml_from_flat_yolo_obb(
    source_dir: str | Path,
    output_path: str | Path,
    class_names: dict[int, str],
) -> Path:
    source_dir = Path(source_dir)
    pairs = find_yolo_obb_pairs(source_dir)
    return export_cvat_xml_from_yolo_obb_pairs(
        pairs=pairs,
        output_path=output_path,
        class_names=class_names,
        task_name=source_dir.name,
    )


def export_cvat_xml_from_split_root(
    source_dir: str | Path,
    output_path: str | Path,
    class_names: dict[int, str],
    split: str = "train",
) -> Path:
    source_dir = Path(source_dir)
    images_dir = source_dir / "images" / split
    labels_dir = source_dir / "labels" / split
    pairs = [
        YoloObbPair(
            stem=image_path.stem,
            image_path=image_path,
            label_path=labels_dir / f"{image_path.stem}.txt",
        )
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file())
    ]
    return export_cvat_xml_from_yolo_obb_pairs(
        pairs=pairs,
        output_path=output_path,
        class_names=class_names,
        task_name=source_dir.name,
    )


def export_cvat_xml_from_yolo_obb_pairs(
    pairs: list[YoloObbPair],
    output_path: str | Path,
    class_names: dict[int, str],
    task_name: str,
) -> Path:

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "created").text = datetime.now(UTC).isoformat()
    labels = ET.SubElement(task, "labels")
    for _, class_name in sorted(class_names.items()):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = class_name

    for image_id, pair in enumerate(sorted(pairs, key=lambda item: item.image_path.name)):
        with Image.open(pair.image_path) as image:
            width, height = image.size

        image_element = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": pair.image_path.name,
                "width": str(width),
                "height": str(height),
            },
        )

        for line in _load_yolo_obb_lines(pair.label_path):
            class_id, box = _line_to_rotated_box(line, image_size=(width, height))
            ET.SubElement(
                image_element,
                "box",
                {
                    "label": class_names[class_id],
                    "occluded": "0",
                    "source": "manual",
                    "z_order": "0",
                    **box,
                },
            )

    tree = ET.ElementTree(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
