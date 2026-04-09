from pathlib import Path

from PIL import Image
import yaml

from src.data_tools.ingest import find_yolo_obb_pairs
from src.data_tools.prepare_yolo_obb import export_existing_yolo_obb_dataset


def test_export_existing_yolo_obb_dataset_writes_split_structure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(source / "sample.jpg")
    (source / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    pairs = find_yolo_obb_pairs(source)
    data_yaml = export_existing_yolo_obb_dataset(
        pairs=pairs,
        output_dir=tmp_path / "output",
        split={"train": ["sample"], "val": []},
        class_names=["insulator"],
    )

    assert (tmp_path / "output" / "images" / "train" / "sample.jpg").exists()
    assert (tmp_path / "output" / "labels" / "train" / "sample.txt").exists()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert data["names"][0] == "insulator"
