from pathlib import Path

from PIL import Image

from src.data_tools.validate_yolo_obb import validate_yolo_obb_dataset


def test_validate_yolo_obb_dataset_reports_no_errors_for_valid_sample(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(dataset_root / "images" / "train" / "sample.jpg")
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    errors = validate_yolo_obb_dataset(dataset_root, class_ids={0})
    assert errors == []


def test_validate_yolo_obb_dataset_reports_out_of_range_coordinate(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(dataset_root / "images" / "train" / "sample.jpg")
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        "0 1.200000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    errors = validate_yolo_obb_dataset(dataset_root, class_ids={0})
    assert any("outside [0, 1]" in error for error in errors)
