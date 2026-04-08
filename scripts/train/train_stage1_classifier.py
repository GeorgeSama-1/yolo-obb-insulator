from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = ArgumentParser(description="Load Stage 1 classifier config.")
    parser.add_argument("--config", default="configs/stage1_two_stage/classifier.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    print(config.get("model", "unknown"))


if __name__ == "__main__":
    main()
