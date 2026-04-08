from src.stage2_one_stage.train import stage2_model_name


def test_stage2_model_name_reads_from_config_dict():
    assert stage2_model_name({"model": "yolo11n-obb.pt"}) == "yolo11n-obb.pt"
