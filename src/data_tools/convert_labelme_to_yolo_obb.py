from pathlib import Path
import json
import shutil

import yaml


def _normalize_point(point: list[float], image_width: int, image_height: int) -> tuple[float, float]:
    return point[0] / image_width, point[1] / image_height


def convert_shape_to_yolo_obb_line(
    shape: dict,
    image_width: int,
    image_height: int,
    class_to_id: dict[str, int],
) -> str:
    class_id = class_to_id[shape["label"]]
    normalized = [
        _normalize_point(point, image_width=image_width, image_height=image_height)
        for point in shape["points"]
    ]
    flattened = [f"{value:.6f}" for point in normalized for value in point]
    return " ".join([str(class_id), *flattened])


def convert_annotation_to_yolo_obb(
    annotation_path: str | Path,
    label_path: str | Path,
    class_to_id: dict[str, int],
) -> None:
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    image_width = int(data["imageWidth"])
    image_height = int(data["imageHeight"])
    lines = [
        convert_shape_to_yolo_obb_line(
            shape,
            image_width=image_width,
            image_height=image_height,
            class_to_id=class_to_id,
        )
        for shape in data.get("shapes", [])
    ]
    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_obb_dataset(
    pairs: list,
    output_dir: str | Path,
    split: dict[str, list[str]],
    class_names: list[str],
) -> Path:
    output_root = Path(output_dir)
    class_to_id = {name: index for index, name in enumerate(class_names)}

    for split_name, stems in split.items():
        images_dir = output_root / "images" / split_name
        labels_dir = output_root / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for pair in pairs:
            if pair.stem not in stems:
                continue
            shutil.copy2(pair.image_path, images_dir / pair.image_path.name)
            convert_annotation_to_yolo_obb(
                annotation_path=pair.annotation_path,
                label_path=labels_dir / f"{pair.stem}.txt",
                class_to_id=class_to_id,
            )

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
