from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_obb.predict import predict_stage0


def main() -> None:
    parser = ArgumentParser(description="Run Stage 0 OBB inference.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    results = predict_stage0(args.weights, args.source)
    print(f"Predicted {len(results)} batch item(s) from {Path(args.source)}")


if __name__ == "__main__":
    main()
