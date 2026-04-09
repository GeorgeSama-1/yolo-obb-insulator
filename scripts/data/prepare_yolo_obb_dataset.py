from argparse import ArgumentParser
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.ingest import find_yolo_obb_pairs
from src.data_tools.prepare_yolo_obb import export_existing_yolo_obb_dataset


def main() -> None:
    parser = ArgumentParser(description="Organize flat YOLO OBB txt + image files into train/val dataset structure.")
    parser.add_argument("--source", default="datasets")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", default="data/processed/yolo_obb_insulator")
    parser.add_argument("--class-name", action="append", dest="class_names")
    args = parser.parse_args()

    pairs = find_yolo_obb_pairs(args.source)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    class_names = args.class_names or ["insulator"]
    data_yaml = export_existing_yolo_obb_dataset(pairs, args.output, split, class_names)
    print(f"Wrote dataset config to {data_yaml}")


if __name__ == "__main__":
    main()
