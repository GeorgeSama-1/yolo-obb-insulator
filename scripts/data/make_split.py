from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.ingest import find_labelme_pairs, find_yolo_obb_pairs
from src.data_tools.splits import make_debug_split, write_split_manifest


def _detect_source_format(source: str | Path) -> str:
    root = Path(source)
    if any(root.glob("*.json")):
        return "labelme"
    if any(root.glob("*.txt")):
        return "yolo_obb"
    raise FileNotFoundError(f"No supported annotation files found in {root}")


def main() -> None:
    parser = ArgumentParser(description="Create a reproducible train/val split manifest.")
    parser.add_argument("--source", default="datasets")
    parser.add_argument("--output", default="data/splits/debug_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    args = parser.parse_args()

    source_format = _detect_source_format(args.source)
    if source_format == "labelme":
        pairs = find_labelme_pairs(args.source)
    else:
        pairs = find_yolo_obb_pairs(args.source)
    split = make_debug_split([pair.stem for pair in pairs], seed=args.seed, val_ratio=args.val_ratio)
    path = write_split_manifest(split, args.output)
    print(f"Wrote split manifest to {path}")


if __name__ == "__main__":
    main()
