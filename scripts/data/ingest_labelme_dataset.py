from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.ingest import stage_labelme_dataset


def main() -> None:
    parser = ArgumentParser(description="Copy LabelMe images and annotations into the raw data area.")
    parser.add_argument("--source", default="datasets")
    parser.add_argument("--target", default="data/raw/labelme_obb")
    args = parser.parse_args()
    pairs = stage_labelme_dataset(args.source, args.target)
    print(f"Staged {len(pairs)} image/annotation pairs into {args.target}")


if __name__ == "__main__":
    main()
