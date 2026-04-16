from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.defaults import (
    DEFAULT_STAGE2_ABNORMAL_BOOST_DIR,
    DEFAULT_STAGE2_ABNORMAL_LIGHT_AUG_DIR,
    DEFAULT_STAGE2_AUGMENTED_DIR,
    DEFAULT_STAGE2_CLASS_BALANCE_JSON,
    DEFAULT_STAGE2_CLASS_BALANCE_MD,
)
from src.stage2_one_stage.balance import (
    generate_stage2_abnormal_boost_dataset,
    generate_stage2_abnormal_light_aug_dataset,
    write_stage2_class_balance_reports,
)


def main() -> None:
    parser = ArgumentParser(description="Prepare Stage 2 class-balance reports and abnormal-focused dataset variants.")
    parser.add_argument("--input", default=DEFAULT_STAGE2_AUGMENTED_DIR, help="Prepared Stage 2 YOLO OBB dataset root")
    parser.add_argument("--report-json", default=DEFAULT_STAGE2_CLASS_BALANCE_JSON)
    parser.add_argument("--report-md", default=DEFAULT_STAGE2_CLASS_BALANCE_MD)
    parser.add_argument("--abnormal-boost-output", default=DEFAULT_STAGE2_ABNORMAL_BOOST_DIR)
    parser.add_argument("--abnormal-light-output", default=DEFAULT_STAGE2_ABNORMAL_LIGHT_AUG_DIR)
    parser.add_argument("--abnormal-target-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = write_stage2_class_balance_reports(
        args.input,
        json_path=args.report_json,
        markdown_path=args.report_md,
    )
    generate_stage2_abnormal_boost_dataset(
        args.input,
        args.abnormal_boost_output,
        abnormal_target_per_image=args.abnormal_target_per_image,
        seed=args.seed,
    )
    generate_stage2_abnormal_light_aug_dataset(
        args.input,
        args.abnormal_light_output,
        abnormal_target_per_image=args.abnormal_target_per_image,
        seed=args.seed,
    )

    print(f"Stage2 class balance written to {args.report_json} and {args.report_md}")
    print(
        f"Generated abnormal-focused datasets at {args.abnormal_boost_output} "
        f"and {args.abnormal_light_output}"
    )
    print(
        "Instance counts: "
        f"normal={report['instance_counts'].get('normal', 0)}, "
        f"abnormal={report['instance_counts'].get('abnormal', 0)}"
    )


if __name__ == "__main__":
    main()
