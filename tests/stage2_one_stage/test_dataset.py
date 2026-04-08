from src.stage2_one_stage.dataset import normalize_stage2_label


def test_normalize_stage2_label_accepts_expected_classes():
    assert normalize_stage2_label("abnormal") == "abnormal"
