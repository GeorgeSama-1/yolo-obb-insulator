from src.stage1_two_stage.classifier_train import normalize_label


def normalize_prediction(label: str) -> str:
    return normalize_label(label)
