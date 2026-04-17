from src.common.defaults import (
    DEFAULT_STAGE0_POOL_NAME,
    DEFAULT_STAGE0_PROCESSED_DIR,
    DEFAULT_STAGE0_SPLIT_PATH,
    DEFAULT_STAGE1_PATCH_DATA_DIR,
    DEFAULT_STAGE1_PATCH_SOURCE_DIR,
    DEFAULT_STAGE2_PROCESSED_DIR,
)


def test_default_stage_paths_follow_canonical_naming():
    assert DEFAULT_STAGE0_POOL_NAME == "datasets_pool_stage0_insulator_obb"
    assert DEFAULT_STAGE0_PROCESSED_DIR == "data/processed/stage0_insulator_obb"
    assert DEFAULT_STAGE0_SPLIT_PATH == "data/splits/stage0_insulator_obb_split.json"
    assert DEFAULT_STAGE1_PATCH_DATA_DIR == "data/processed/stage1_patch_classifier"
    assert DEFAULT_STAGE1_PATCH_SOURCE_DIR == "data/processed/stage2_defect_obb_abn_boost"
    assert DEFAULT_STAGE2_PROCESSED_DIR == "data/processed/stage2_defect_obb"
