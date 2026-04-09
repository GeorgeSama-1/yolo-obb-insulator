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
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also export overlay preview images after augmentation for manual inspection",
    )
    parser.add_argument(
        "--preview-output",
        default=None,
        help="Directory for overlay previews; default is <output>/preview",
    )
    parser.add_argument(
        "--preview-splits",
        nargs="+",
        default=None,
        help="Dataset splits to render preview overlays for; default follows augment splits",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="Maximum number of preview overlays to write per split",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preview_output = None
    preview_splits: tuple[str, ...] = ()
    if args.preview:
        preview_output = args.preview_output or str(Path(args.output) / "preview")
        preview_splits = tuple(args.preview_splits or args.augment_splits)

    augment_yolo_obb_dataset(
        input_root=args.input,
        output_root=args.output,
        target_per_image=args.target_per_image,
        augment_splits=tuple(args.augment_splits),
        seed=args.seed,
        preview_output_dir=preview_output,
        preview_splits=preview_splits,
        preview_limit_per_split=args.preview_limit if args.preview else None,
    )
    print(
        f"Augmented dataset written to {args.output} "
        f"with target_per_image={args.target_per_image} for splits={args.augment_splits}"
    )
    if args.preview:
        print(f"Preview overlays written to {preview_output}")


if __name__ == "__main__":
    main()
