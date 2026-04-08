from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_obb.visualize import visualize_yolo_obb_split


def main() -> None:
    parser = ArgumentParser(description="Visualize YOLO OBB labels as overlay images for manual inspection.")
    parser.add_argument("--dataset", required=True, help="YOLO OBB dataset root containing dataset.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Directory for exported overlay images")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    data = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8")) or {}
    raw_names = data.get("names", {})
    class_names = {int(key): value for key, value in raw_names.items()}

    written = visualize_yolo_obb_split(
        dataset_root=dataset_root,
        split=args.split,
        output_dir=args.output,
        class_names=class_names,
        limit=args.limit,
    )
    print(f"Wrote {len(written)} overlay image(s) to {args.output}")


if __name__ == "__main__":
    main()
