from pathlib import Path

from src.stage0_obb.config import load_stage0_config


def stage2_model_name(config: dict) -> str:
    return config["model"]


def train_stage2(config_path: str | Path):
    from ultralytics import YOLO

    config = load_stage0_config(config_path)
    model = YOLO(stage2_model_name(config))
    return model.train(
        data=config["data"],
        epochs=config.get("epochs", 100),
        imgsz=config.get("imgsz", 1024),
        batch=config.get("batch", 4),
        project=config.get("project", "runs/stage2_one_stage"),
        name=config.get("name", "defect_obb"),
        device=config.get("device", "cpu"),
    )
