from src.stage0_obb.train import build_stage0_train_args


def test_build_stage0_train_args_passes_through_extra_train_kwargs():
    config = {
        "model": "yolo11n-obb.pt",
        "data": "data.yaml",
        "epochs": 150,
        "imgsz": 1600,
        "batch": 8,
        "project": "runs/stage0_obb",
        "name": "stage0_custom",
        "device": "0,1",
        "workers": 12,
        "patience": 30,
        "optimizer": "AdamW",
        "close_mosaic": 5,
    }

    assert build_stage0_train_args(config) == {
        "data": "data.yaml",
        "epochs": 150,
        "imgsz": 1600,
        "batch": 8,
        "project": "runs/stage0_obb",
        "name": "stage0_custom",
        "device": "0,1",
        "workers": 12,
        "patience": 30,
        "optimizer": "AdamW",
        "close_mosaic": 5,
    }
