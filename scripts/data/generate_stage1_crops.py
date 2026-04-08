from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_two_stage.crops import padded_crop_bounds


def main() -> None:
    parser = ArgumentParser(description="Generate Stage 1 crop bounds.")
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--bottom", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    print(
        padded_crop_bounds(
            (args.left, args.top, args.right, args.bottom),
            image_size=(args.width, args.height),
            padding=args.padding,
        )
    )


if __name__ == "__main__":
    main()
