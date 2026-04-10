from pathlib import Path

from PIL import Image
import pytest

from src.data_tools.dataset_pool import merge_yolo_obb_into_pool


def test_merge_yolo_obb_into_pool_copies_flat_source(tmp_path):
    source = tmp_path / "source"
    pool = tmp_path / "datasets_pool"
    source.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(source / "sample.jpg")
    (source / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    copied = merge_yolo_obb_into_pool(source_dir=source, pool_dir=pool)

    assert copied == 1
    assert (pool / "sample.jpg").exists()
    assert (pool / "sample.txt").exists()


def test_merge_yolo_obb_into_pool_copies_split_source(tmp_path):
    source = tmp_path / "source"
    pool = tmp_path / "datasets_pool"
    (source / "images" / "train").mkdir(parents=True)
    (source / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(source / "images" / "train" / "sample.jpg")
    (source / "labels" / "train" / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )

    copied = merge_yolo_obb_into_pool(source_dir=source, pool_dir=pool, split="train")

    assert copied == 1
    assert (pool / "sample.jpg").exists()
    assert (pool / "sample.txt").exists()


def test_merge_yolo_obb_into_pool_rejects_duplicate_without_overwrite(tmp_path):
    source = tmp_path / "source"
    pool = tmp_path / "datasets_pool"
    source.mkdir()
    pool.mkdir()
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(source / "sample.jpg")
    (source / "sample.txt").write_text(
        "0 0.100000 0.100000 0.400000 0.100000 0.400000 0.400000 0.100000 0.400000\n",
        encoding="utf-8",
    )
    Image.new("RGB", (32, 32), color=(10, 10, 10)).save(pool / "sample.jpg")
    (pool / "sample.txt").write_text(
        "0 0.200000 0.200000 0.500000 0.200000 0.500000 0.500000 0.200000 0.500000\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="sample"):
        merge_yolo_obb_into_pool(source_dir=source, pool_dir=pool)
