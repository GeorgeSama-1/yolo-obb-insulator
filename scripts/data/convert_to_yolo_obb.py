from argparse import ArgumentParser
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.convert_labelme_to_yolo_obb import export_yolo_obb_dataset
from src.data_tools.ingest import find_labelme_pairs


def main() -> None:
    parser = ArgumentParser(description="Convert LabelMe polygons to YOLO OBB labels.")
    parser.add_argument("--source", default="datasets")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", default="data/processed/yolo_obb_insulator")
    parser.add_argument("--class-name", action="append", dest="class_names")
    args = parser.parse_args()

    pairs = find_labelme_pairs(args.source)
    split = json.loads(open(args.split_json, "r", encoding="utf-8").read())
    class_names = args.class_names or ["insulator"]
    data_yaml = export_yolo_obb_dataset(pairs, args.output, split, class_names)
    print(f"Wrote dataset config to {data_yaml}")


if __name__ == "__main__":
    main()
