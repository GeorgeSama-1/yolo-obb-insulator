from argparse import ArgumentParser
from pathlib import Path
import sys

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage0_obb.predict import predict_stage0
from src.stage0_obb.visualize import visualize_prediction_results


def main() -> None:
    parser = ArgumentParser(description="Run Stage 0 OBB prediction and export overlay images.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image file, directory, or glob source for inference")
    parser.add_argument("--output", required=True, help="Directory for prediction overlays")
    parser.add_argument("--dataset-yaml", default=None, help="Optional dataset yaml to resolve class names")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    class_names = {0: "insulator"}
    if args.dataset_yaml:
        data = yaml.safe_load(Path(args.dataset_yaml).read_text(encoding="utf-8")) or {}
        raw_names = data.get("names", {})
        class_names = {int(key): value for key, value in raw_names.items()}

    results = predict_stage0(args.weights, args.source, imgsz=args.imgsz, conf=args.conf)
    written = visualize_prediction_results(results, output_dir=args.output, class_names=class_names)
    print(f"Wrote {len(written)} prediction overlay image(s) to {args.output}")


if __name__ == "__main__":
    main()
