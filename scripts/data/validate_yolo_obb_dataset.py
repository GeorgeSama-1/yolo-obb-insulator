from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.validate_yolo_obb import validate_yolo_obb_dataset


def main() -> None:
    parser = ArgumentParser(description="Validate YOLO OBB dataset labels and image-label pairing.")
    parser.add_argument("--dataset", required=True, help="YOLO OBB dataset root")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    dataset_yaml = dataset_root / "dataset.yaml"
    class_ids = None
    if dataset_yaml.exists():
        data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        class_ids = {int(key) for key in names.keys()}

    errors = validate_yolo_obb_dataset(dataset_root, class_ids=class_ids)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("YOLO OBB dataset validation passed")


if __name__ == "__main__":
    main()
