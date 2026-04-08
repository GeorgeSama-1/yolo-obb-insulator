from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_one_stage.dataset import normalize_stage2_label


def main() -> None:
    parser = ArgumentParser(description="Normalize a Stage 2 label.")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    print(normalize_stage2_label(args.label))


if __name__ == "__main__":
    main()
