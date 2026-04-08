from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_obb.metrics import compute_count_metrics, write_metrics_report


def main() -> None:
    parser = ArgumentParser(description="Evaluate Stage 0 count predictions from integer lists.")
    parser.add_argument("--ground-truth", nargs="+", type=int, required=True)
    parser.add_argument("--predicted", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="reports/metrics/stage0_count_metrics.json")
    args = parser.parse_args()

    metrics = compute_count_metrics(args.ground_truth, args.predicted)
    write_metrics_report(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
