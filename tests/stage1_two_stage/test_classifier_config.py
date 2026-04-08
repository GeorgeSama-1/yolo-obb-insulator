from src.stage1_two_stage.classifier_train import normalize_label


def test_normalize_label_accepts_normal_and_abnormal():
    assert normalize_label("normal") == "normal"
