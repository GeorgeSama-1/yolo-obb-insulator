from src.stage1_two_stage.classifier_train import build_stage1_train_args, normalize_label


def test_normalize_label_accepts_normal_and_abnormal():
    assert normalize_label("normal") == "normal"


def test_build_stage1_train_args_supports_yolo_cls_defaults_and_extra_kwargs():
    config = {
        "model": "yolo11m-cls.pt",
        "data": "data/processed/stage1_patch_classifier",
        "epochs": 80,
        "imgsz": 384,
        "batch": 16,
        "project": "runs/stage1_two_stage",
        "name": "stage1_patch_classifier",
        "device": "0",
        "workers": 8,
        "patience": 20,
    }

    assert build_stage1_train_args(config) == {
        "data": "data/processed/stage1_patch_classifier",
        "epochs": 80,
        "imgsz": 384,
        "batch": 16,
        "project": "runs/stage1_two_stage",
        "name": "stage1_patch_classifier",
        "device": "0",
        "workers": 8,
        "patience": 20,
    }
