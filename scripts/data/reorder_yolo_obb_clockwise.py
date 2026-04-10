from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.prepare_yolo_obb import reorder_yolo_obb_line_clockwise


def main() -> None:
    parser = ArgumentParser(description="Reorder YOLO OBB vertices clockwise starting from the top-left point.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing YOLO OBB txt labels")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        reordered = [reorder_yolo_obb_line_clockwise(line) for line in lines]
        label_path.write_text("\n".join(reordered) + ("\n" if reordered else ""), encoding="utf-8")

    print(f"Reordered labels in {labels_dir}")


if __name__ == "__main__":
    main()
