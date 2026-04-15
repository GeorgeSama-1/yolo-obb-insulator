from src.stage2_one_stage.train import build_stage2_train_args


def test_build_stage2_train_args_passes_through_extra_train_kwargs():
    config = {
        "model": "yolo11n-obb.pt",
        "data": "data.yaml",
        "epochs": 150,
        "imgsz": 1280,
        "batch": 8,
        "project": "runs/stage2_one_stage",
        "name": "stage2_custom",
        "device": "0",
        "workers": 8,
        "patience": 40,
        "optimizer": "AdamW",
        "degrees": 10.0,
    }

    assert build_stage2_train_args(config) == {
        "data": "data.yaml",
        "epochs": 150,
        "imgsz": 1280,
        "batch": 8,
        "project": "runs/stage2_one_stage",
        "name": "stage2_custom",
        "device": "0",
        "workers": 8,
        "patience": 40,
        "optimizer": "AdamW",
        "degrees": 10.0,
    }
