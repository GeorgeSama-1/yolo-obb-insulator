from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_two_stage.classifier_train import build_stage1_train_args, stage1_model_name


def main() -> None:
    parser = ArgumentParser(description="Train the Stage 1 YOLO-cls patch classifier.")
    parser.add_argument("--config", default="configs/stage1_two_stage/classifier.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    from ultralytics import YOLO

    model = YOLO(stage1_model_name(config))
    model.train(**build_stage1_train_args(config))


if __name__ == "__main__":
    main()
