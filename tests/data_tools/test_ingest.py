from pathlib import Path

from PIL import Image

from src.data_tools.ingest import find_labelme_pairs, find_yolo_obb_pairs


def test_find_labelme_pairs_returns_image_annotation_pairs(tmp_path):
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(tmp_path / "sample.jpg")
    (tmp_path / "sample.json").write_text('{"shapes": []}', encoding="utf-8")
    pairs = find_labelme_pairs(tmp_path)
    assert len(pairs) == 1


def test_find_yolo_obb_pairs_returns_image_label_pairs():
    pairs = find_yolo_obb_pairs("datasets")
    assert len(pairs) == 6
