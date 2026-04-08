from pathlib import Path

from PIL import Image

from src.stage0_obb.visualize import (
    color_for_class,
    load_yolo_obb_detections,
    visualize_yolo_obb_split,
)


def test_color_for_class_returns_rgb_triplet():
    color = color_for_class("insulator")
    assert len(color) == 3


def test_load_yolo_obb_detections_reads_label_file():
    detections = load_yolo_obb_detections(
        [
            "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000",
        ],
        class_names={0: "insulator"},
        image_size=(100, 100),
    )
    assert detections[0]["label"] == "insulator"
    assert detections[0]["points"][0] == (10.0, 10.0)


def test_visualize_yolo_obb_split_writes_overlay_images(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(dataset_root / "images" / "train" / "sample.jpg")
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    written = visualize_yolo_obb_split(
        dataset_root=dataset_root,
        split="train",
        output_dir=output_dir,
        class_names={0: "insulator"},
        limit=1,
    )

    assert len(written) == 1
    assert Path(written[0]).exists()
