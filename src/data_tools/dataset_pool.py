from __future__ import annotations

from pathlib import Path
import shutil

from src.data_tools.ingest import YoloObbPair, find_yolo_obb_pairs


def merge_yolo_obb_into_pool(
    source_dir: str | Path,
    pool_dir: str | Path,
    split: str = "train",
    overwrite: bool = False,
) -> int:
    pool_root = Path(pool_dir)
    pool_root.mkdir(parents=True, exist_ok=True)
    pairs = _discover_yolo_obb_pairs(source_dir=source_dir, split=split)

    for pair in pairs:
        image_target = pool_root / pair.image_path.name
        label_target = pool_root / pair.label_path.name
        if not overwrite and (image_target.exists() or label_target.exists()):
            raise FileExistsError(f"Duplicate sample already exists in pool: {pair.stem}")
        shutil.copy2(pair.image_path, image_target)
        shutil.copy2(pair.label_path, label_target)

    return len(pairs)


def _discover_yolo_obb_pairs(source_dir: str | Path, split: str) -> list[YoloObbPair]:
    source_root = Path(source_dir)
    split_images_dir = source_root / "images" / split
    split_labels_dir = source_root / "labels" / split

    if split_images_dir.exists() and split_labels_dir.exists():
        return [
            YoloObbPair(
                stem=image_path.stem,
                image_path=image_path,
                label_path=split_labels_dir / f"{image_path.stem}.txt",
            )
            for image_path in sorted(path for path in split_images_dir.iterdir() if path.is_file())
        ]

    return find_yolo_obb_pairs(source_root)
