from pathlib import Path

from PIL import Image
import yaml

from src.stage1_two_stage.patch_dataset import export_stage1_patch_classifier_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_stage2_source(root: Path) -> Path:
    _write_image(root / "images" / "train" / "sample_a.jpg", (120, 120, 120))
    _write_image(root / "images" / "train" / "sample_b.jpg", (140, 140, 140))
    _write_image(root / "images" / "val" / "sample_val.jpg", (160, 160, 160))

    _write_label(
        root / "labels" / "train" / "sample_a.txt",
        [
            "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30",
            "0 0.40 0.10 0.60 0.10 0.60 0.30 0.40 0.30",
            "1 0.10 0.40 0.30 0.40 0.30 0.60 0.10 0.60",
        ],
    )
    _write_label(
        root / "labels" / "train" / "sample_b.txt",
        [
            "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30",
        ],
    )
    _write_label(
        root / "labels" / "val" / "sample_val.txt",
        [
            "1 0.20 0.20 0.45 0.20 0.45 0.45 0.20 0.45",
        ],
    )

    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "normal", 1: "abnormal"},
            },
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return root


def test_export_stage1_patch_classifier_dataset_crops_stage2_labels_into_class_dirs(tmp_path):
    source = _make_stage2_source(tmp_path / "stage2")
    output = tmp_path / "stage1"

    export_stage1_patch_classifier_dataset(source, output, padding=0, train_normal_to_abnormal_ratio=2.0)

    assert sorted(path.name for path in (output / "train" / "normal").iterdir()) == [
        "sample_a_000.jpg",
        "sample_a_001.jpg",
        "sample_b_000.jpg",
    ]
    assert sorted(path.name for path in (output / "train" / "abnormal").iterdir()) == [
        "sample_a_002.jpg",
        "sample_a_002_os_01.jpg",
    ]
    assert sorted(path.name for path in (output / "val" / "abnormal").iterdir()) == [
        "sample_val_000.jpg",
    ]


def test_export_stage1_patch_classifier_dataset_keeps_all_normal_and_only_oversamples_abnormal(tmp_path):
    source = _make_stage2_source(tmp_path / "stage2")
    output = tmp_path / "stage1"

    export_stage1_patch_classifier_dataset(source, output, padding=0, train_normal_to_abnormal_ratio=2.0)

    train_normal = sorted(path.name for path in (output / "train" / "normal").iterdir())
    train_abnormal = sorted(path.name for path in (output / "train" / "abnormal").iterdir())

    assert len(train_normal) == 3
    assert len(train_abnormal) == 2
    assert all("_os_" not in name for name in train_normal)
    assert any("_os_" in name for name in train_abnormal)
