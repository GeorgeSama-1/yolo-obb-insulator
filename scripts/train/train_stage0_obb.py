from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_obb.train import train_stage0


def main() -> None:
    parser = ArgumentParser(description="Train the Stage 0 insulator YOLO11 OBB detector.")
    parser.add_argument("--config", default="configs/stage0_obb/train_insulator.yaml")
    args = parser.parse_args()
    train_stage0(args.config)


if __name__ == "__main__":
    main()
