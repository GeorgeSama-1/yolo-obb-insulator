from argparse import ArgumentParser
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.ingest import find_labelme_pairs
from src.data_tools.validate import validate_labelme_annotation


def main() -> None:
    parser = ArgumentParser(description="Validate LabelMe OBB annotations.")
    parser.add_argument("--source", default="datasets")
    parser.add_argument("--label", action="append", dest="labels")
    args = parser.parse_args()

    allowed_labels = set(args.labels) if args.labels else None
    errors: list[str] = []
    for pair in find_labelme_pairs(args.source):
        data = json.loads(pair.annotation_path.read_text(encoding="utf-8"))
        file_errors = validate_labelme_annotation(data, allowed_labels=allowed_labels)
        errors.extend([f"{pair.annotation_path.name}: {message}" for message in file_errors])

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
