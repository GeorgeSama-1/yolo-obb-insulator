from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.dataset_pool import merge_yolo_obb_into_pool


def main() -> None:
    parser = ArgumentParser(
        description="Merge corrected YOLO OBB samples into a flat datasets pool for the next training round."
    )
    parser.add_argument("--source", required=True, help="Flat source dir or images/<split> + labels/<split> root")
    parser.add_argument("--pool", default="datasets_pool", help="Flat pool directory that stores all confirmed samples")
    parser.add_argument("--split", default="train", help="Split name to read when source uses images/<split> layout")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image/txt pairs with the same stem")
    args = parser.parse_args()

    copied = merge_yolo_obb_into_pool(
        source_dir=args.source,
        pool_dir=args.pool,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(f"Merged {copied} samples into {Path(args.pool).resolve()}")


if __name__ == "__main__":
    main()
