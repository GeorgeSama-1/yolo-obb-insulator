from pathlib import Path

from src.stage0_obb.config import load_stage0_config
from src.stage0_obb.train import build_stage0_train_args


def stage2_model_name(config: dict) -> str:
    return config["model"]


def build_stage2_train_args(config: dict) -> dict:
    train_args = build_stage0_train_args(config)
    train_args.setdefault("project", "runs/stage2_one_stage")
    train_args.setdefault("name", "defect_obb")
    return train_args


def train_stage2(config_path: str | Path):
    from ultralytics import YOLO

    config = load_stage0_config(config_path)
    model = YOLO(stage2_model_name(config))
    return model.train(**build_stage2_train_args(config))
