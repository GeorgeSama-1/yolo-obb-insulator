from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.defaults import DEFAULT_STAGE1_PATCH_DATA_DIR, DEFAULT_STAGE1_PATCH_SOURCE_DIR
from src.stage1_two_stage.patch_dataset import export_stage1_patch_classifier_dataset


def main() -> None:
    parser = ArgumentParser(description="Generate Stage 1 patch classifier dataset from a Stage 2 YOLO OBB dataset.")
    parser.add_argument("--input", default=DEFAULT_STAGE1_PATCH_SOURCE_DIR, help="Source Stage 2 YOLO OBB dataset root")
    parser.add_argument("--output", default=DEFAULT_STAGE1_PATCH_DATA_DIR, help="Output Stage 1 classifier dataset root")
    parser.add_argument("--padding", type=int, default=8, help="Extra padding in pixels around each OBB crop")
    parser.add_argument(
        "--train-normal-to-abnormal-ratio",
        type=float,
        default=2.0,
        help="Target train split ratio by oversampling abnormal only",
    )
    args = parser.parse_args()

    export_stage1_patch_classifier_dataset(
        input_root=args.input,
        output_root=args.output,
        padding=args.padding,
        train_normal_to_abnormal_ratio=args.train_normal_to_abnormal_ratio,
    )
    print(f"Stage 1 patch classifier dataset written to {args.output}")


if __name__ == "__main__":
    main()
