from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.augment_yolo_obb import augment_yolo_obb_dataset


def main() -> None:
    parser = ArgumentParser(description="Offline augmentation for YOLO OBB datasets.")
    parser.add_argument("--input", required=True, help="Path to the source YOLO OBB dataset root")
    parser.add_argument("--output", required=True, help="Path to the augmented YOLO OBB dataset root")
    parser.add_argument(
        "--target-per-image",
        type=int,
        default=20,
        help="Total number of images to keep per original image in augmented splits, including the original",
    )
    parser.add_argument(
        "--augment-splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to augment; non-listed splits are copied as-is",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_yolo_obb_dataset(
        input_root=args.input,
        output_root=args.output,
        target_per_image=args.target_per_image,
        augment_splits=tuple(args.augment_splits),
        seed=args.seed,
    )
    print(
        f"Augmented dataset written to {args.output} "
        f"with target_per_image={args.target_per_image} for splits={args.augment_splits}"
    )


if __name__ == "__main__":
    main()
