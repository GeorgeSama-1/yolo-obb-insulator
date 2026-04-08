from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_one_stage.train import train_stage2


def main() -> None:
    parser = ArgumentParser(description="Train the Stage 2 defect YOLO11 OBB model.")
    parser.add_argument("--config", default="configs/stage2_one_stage/train_defect_obb.yaml")
    args = parser.parse_args()
    train_stage2(args.config)


if __name__ == "__main__":
    main()
