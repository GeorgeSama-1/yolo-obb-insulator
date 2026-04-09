from pathlib import Path
import shutil

import yaml


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
