from pathlib import Path

from src.stage0_obb.config import load_stage0_config


def build_stage0_train_args(config: dict) -> dict:
    train_args = {
        "data": config["data"],
        "epochs": config.get("epochs", 100),
        "imgsz": config.get("imgsz", 1024),
        "batch": config.get("batch", 4),
        "project": config.get("project", "runs/stage0_obb"),
        "name": config.get("name", "insulator"),
        "device": config.get("device", "cpu"),
    }
    excluded_keys = {"model", *train_args.keys()}
    extra_args = {key: value for key, value in config.items() if key not in excluded_keys}
    train_args.update(extra_args)
    return train_args


def train_stage0(config_path: str | Path):
    from ultralytics import YOLO

    config = load_stage0_config(config_path)
    model = YOLO(config["model"])
    return model.train(**build_stage0_train_args(config))
