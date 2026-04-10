from pathlib import Path
import math
import shutil

import yaml


def reorder_points_clockwise(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) != 4:
        raise ValueError("OBB points must contain exactly 4 vertices")

    center_x = sum(x for x, _ in points) / len(points)
    center_y = sum(y for _, y in points) / len(points)
    ordered = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))

    start_index = min(range(len(ordered)), key=lambda idx: (ordered[idx][1], ordered[idx][0]))
    return ordered[start_index:] + ordered[:start_index]


def reorder_yolo_obb_line_clockwise(line: str) -> str:
    parts = line.split()
    if len(parts) < 9:
        raise ValueError(f"Expected at least 9 values, got {len(parts)}: {line!r}")

    class_id = parts[0]
    coords = [float(value) for value in parts[1:9]]
    points = [(coords[index], coords[index + 1]) for index in range(0, len(coords), 2)]
    ordered = reorder_points_clockwise(points)
    return " ".join([class_id, *[f"{value:.6f}" for point in ordered for value in point]])


def export_existing_yolo_obb_dataset(
    pairs: list,
    output_dir: str | Path,
    split: dict[str, list[str]],
    class_names: list[str],
) -> Path:
    output_root = Path(output_dir)

    for split_name, stems in split.items():
        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for pair in pairs:
            if pair.stem not in stems:
                continue
            shutil.copy2(pair.image_path, images_dir / pair.image_path.name)
            shutil.copy2(pair.label_path, labels_dir / pair.label_path.name)

    data_yaml = output_root / "dataset.yaml"
    yaml.safe_dump(
        {
            "path": str(output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(class_names)},
        },
        data_yaml.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=False,
    )
    return data_yaml
