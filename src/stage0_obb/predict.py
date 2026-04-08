from pathlib import Path


def predict_stage0(weights: str | Path, source: str | Path, **kwargs):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.predict(source=str(source), save=False, **kwargs)
