from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.convert_yolo_obb_to_cvat_xml import (
    export_cvat_xml_from_flat_yolo_obb,
    export_cvat_xml_from_split_root,
    export_cvat_xml_from_yolo_obb,
)


def main() -> None:
    parser = ArgumentParser(description="Convert YOLO OBB labels into CVAT XML annotations.")
    parser.add_argument("--dataset", help="Prepared YOLO OBB dataset root with images/<split> and labels/<split>")
    parser.add_argument("--source", help="Flat source directory containing image + txt pairs")
    parser.add_argument("--output", required=True, help="Output XML path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--class-name", action="append", dest="class_names")
    args = parser.parse_args()

    if bool(args.dataset) == bool(args.source):
        parser.error("Provide exactly one of --dataset or --source")

    if args.dataset:
        dataset_root = Path(args.dataset)
        data = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8")) or {}
        raw_names = data.get("names", {})
        class_names = {int(key): value for key, value in raw_names.items()}
        xml_path = export_cvat_xml_from_yolo_obb(
            dataset_root=dataset_root,
            output_path=args.output,
            class_names=class_names,
            split=args.split,
        )
    else:
        class_names = {index: name for index, name in enumerate(args.class_names or ["insulator"])}
        source_root = Path(args.source)
        if (source_root / "images" / args.split).exists() and (source_root / "labels" / args.split).exists():
            xml_path = export_cvat_xml_from_split_root(
                source_dir=source_root,
                output_path=args.output,
                class_names=class_names,
                split=args.split,
            )
        else:
            xml_path = export_cvat_xml_from_flat_yolo_obb(
                source_dir=source_root,
                output_path=args.output,
                class_names=class_names,
            )

    print(f"Wrote CVAT XML to {xml_path}")


if __name__ == "__main__":
    main()
