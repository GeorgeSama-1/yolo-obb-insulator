from src.common.paths import (
    cvat_exports_dir,
    data_dir,
    dataset_pool_dir,
    incoming_batches_dir,
    prelabel_batches_dir,
    processed_stage_dir,
    project_root,
)


def test_paths_resolve():
    assert project_root().exists()
    assert data_dir().name == "data"


def test_canonical_experiment_paths_are_named_for_reuse():
    assert incoming_batches_dir().name == "incoming_batches"
    assert prelabel_batches_dir().name == "prelabel_batches"
    assert cvat_exports_dir().name == "cvat_exports"
    assert dataset_pool_dir("stage0_insulator_obb").name == "datasets_pool_stage0_insulator_obb"
    assert processed_stage_dir("stage2_defect_obb").name == "stage2_defect_obb"
