from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.defaults import DEFAULT_STAGE2_ABNORMAL_BOOST_X6_DIR, DEFAULT_STAGE2_AUGMENTED_DIR
from src.stage2_one_stage.balance import generate_stage2_abnormal_boost_dataset


def main() -> None:
    parser = ArgumentParser(description="Generate the Stage 2 abnormal boost x6 dataset.")
    parser.add_argument("--input", default=DEFAULT_STAGE2_AUGMENTED_DIR, help="Prepared Stage 2 YOLO OBB dataset root")
    parser.add_argument("--output", default=DEFAULT_STAGE2_ABNORMAL_BOOST_X6_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    print(f"[stage2-abn-boost-x6] reading dataset: {args.input}")
    print("[stage2-abn-boost-x6] generating abnormal boost x6 dataset...")
    generate_stage2_abnormal_boost_dataset(
        args.input,
        args.output,
        abnormal_target_per_image=6,
        seed=args.seed,
        progress_callback=print,
        progress_every=args.progress_every,
    )
    print(f"[stage2-abn-boost-x6] dataset written to {args.output}")


if __name__ == "__main__":
    main()
