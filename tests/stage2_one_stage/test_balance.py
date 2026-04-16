from pathlib import Path

from PIL import Image
import yaml

from src.stage2_one_stage.balance import (
    build_stage2_class_balance_report,
    generate_stage2_abnormal_boost_dataset,
    generate_stage2_abnormal_light_aug_dataset,
    write_stage2_class_balance_reports,
)


def _write_sample(image_path: Path, label_path: Path, class_id: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)
    label_path.write_text(
        f"{class_id} 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n",
        encoding="utf-8",
    )


def _make_stage2_dataset(root: Path) -> Path:
    _write_sample(root / "images" / "train" / "normal.jpg", root / "labels" / "train" / "normal.txt", 0)
    _write_sample(
        root / "images" / "train" / "abnormal.jpg",
        root / "labels" / "train" / "abnormal.txt",
        1,
    )
    _write_sample(root / "images" / "val" / "val_abnormal.jpg", root / "labels" / "val" / "val_abnormal.txt", 1)
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


def test_build_stage2_class_balance_report_counts_instances_and_images(tmp_path):
    dataset_root = _make_stage2_dataset(tmp_path / "dataset")

    report = build_stage2_class_balance_report(dataset_root)

    assert report["instance_counts"] == {"normal": 1, "abnormal": 2}
    assert report["image_counts"] == {"normal": 1, "abnormal": 2}
    assert report["instance_ratio"]["abnormal_over_normal"] == 2.0


def test_generate_stage2_abnormal_boost_dataset_only_expands_abnormal_train_images(tmp_path):
    dataset_root = _make_stage2_dataset(tmp_path / "dataset")
    output_root = tmp_path / "boosted"

    generate_stage2_abnormal_boost_dataset(dataset_root, output_root, abnormal_target_per_image=3, seed=7)

    train_images = sorted(path.name for path in (output_root / "images" / "train").iterdir())
    val_images = sorted(path.name for path in (output_root / "images" / "val").iterdir())

    assert train_images == [
        "abnormal.jpg",
        "abnormal_abn_boost_01.jpg",
        "abnormal_abn_boost_02.jpg",
        "normal.jpg",
    ]
    assert val_images == ["val_abnormal.jpg"]


def test_generate_stage2_abnormal_light_aug_only_creates_augmented_copies_for_abnormal_train_images(tmp_path):
    dataset_root = _make_stage2_dataset(tmp_path / "dataset")
    output_root = tmp_path / "light_aug"

    generate_stage2_abnormal_light_aug_dataset(dataset_root, output_root, abnormal_target_per_image=3, seed=11)

    train_images = sorted(path.name for path in (output_root / "images" / "train").iterdir())
    train_labels = sorted(path.name for path in (output_root / "labels" / "train").iterdir())
    dataset_yaml = yaml.safe_load((output_root / "dataset.yaml").read_text(encoding="utf-8")) or {}

    assert train_images == [
        "abnormal.jpg",
        "abnormal_abn_light_01.jpg",
        "abnormal_abn_light_02.jpg",
        "normal.jpg",
    ]
    assert train_labels == [
        "abnormal.txt",
        "abnormal_abn_light_01.txt",
        "abnormal_abn_light_02.txt",
        "normal.txt",
    ]
    assert dataset_yaml["path"] == str(output_root.resolve())


def test_write_stage2_class_balance_reports_creates_json_and_markdown(tmp_path):
    dataset_root = _make_stage2_dataset(tmp_path / "dataset")
    json_path = tmp_path / "reports" / "metrics" / "stage2_class_balance.json"
    md_path = tmp_path / "reports" / "tables" / "stage2_class_balance.md"

    write_stage2_class_balance_reports(dataset_root, json_path=json_path, markdown_path=md_path)

    json_report = yaml.safe_load(json_path.read_text(encoding="utf-8"))
    markdown_report = md_path.read_text(encoding="utf-8")

    assert json_report["instance_counts"] == {"normal": 1, "abnormal": 2}
    assert "| abnormal | 2 | 2 |" in markdown_report
