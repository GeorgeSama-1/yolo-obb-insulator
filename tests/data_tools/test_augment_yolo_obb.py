from pathlib import Path

from PIL import Image

from src.data_tools.augment_yolo_obb import (
    augment_dataset_split,
    augment_yolo_obb_dataset,
    apply_horizontal_flip_to_points,
)


def test_apply_horizontal_flip_to_points_mirrors_x_coordinates():
    points = [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)]
    flipped = apply_horizontal_flip_to_points(points)
    assert flipped == [(0.9, 0.2), (0.6, 0.2), (0.6, 0.6), (0.9, 0.6)]


def test_augment_dataset_split_expands_each_image_to_target_count(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    (input_root / "images" / "train").mkdir(parents=True)
    (input_root / "labels" / "train").mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(input_root / "images" / "train" / "sample.jpg")
    (input_root / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    written = augment_dataset_split(
        input_root=input_root,
        output_root=output_root,
        split="train",
        target_per_image=3,
        seed=7,
    )

    assert len(written) == 3
    assert len(list((output_root / "images" / "train").glob("*.jpg"))) == 3
    assert len(list((output_root / "labels" / "train").glob("*.txt"))) == 3


def test_augment_yolo_obb_dataset_creates_output_root_and_dataset_yaml(tmp_path):
    input_root = tmp_path / "input"
    (input_root / "images" / "train").mkdir(parents=True)
    (input_root / "labels" / "train").mkdir(parents=True)
    (input_root / "images" / "val").mkdir(parents=True)
    (input_root / "labels" / "val").mkdir(parents=True)

    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(input_root / "images" / "train" / "sample.jpg")
    Image.new("RGB", (16, 16), color=(40, 50, 60)).save(input_root / "images" / "val" / "holdout.jpg")
    (input_root / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )
    (input_root / "labels" / "val" / "holdout.txt").write_text(
        "0 0.200000 0.200000 0.500000 0.200000 0.500000 0.500000 0.200000 0.500000\n",
        encoding="utf-8",
    )
    (input_root / "dataset.yaml").write_text(
        "path: /tmp/input\ntrain: images/train\nval: images/val\nnames:\n  0: insulator\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "augmented"
    augment_yolo_obb_dataset(input_root=input_root, output_root=output_root, target_per_image=2, seed=3)

    assert (output_root / "dataset.yaml").exists()
    assert len(list((output_root / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((output_root / "images" / "val").glob("*.jpg"))) == 1


def test_augment_yolo_obb_dataset_can_write_preview_overlays(tmp_path):
    input_root = tmp_path / "input"
    (input_root / "images" / "train").mkdir(parents=True)
    (input_root / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(input_root / "images" / "train" / "sample.jpg")
    (input_root / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )
    (input_root / "dataset.yaml").write_text(
        "path: /tmp/input\ntrain: images/train\nval: images/val\nnames:\n  0: insulator\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "augmented"
    augment_yolo_obb_dataset(
        input_root=input_root,
        output_root=output_root,
        target_per_image=2,
        seed=3,
        preview_output_dir=output_root / "preview",
        preview_splits=("train",),
        preview_limit_per_split=1,
    )

    assert len(list((output_root / "preview" / "train").glob("*_overlay.jpg"))) == 1
