VALID_STAGE1_LABELS = {"normal", "abnormal"}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in VALID_STAGE1_LABELS:
        raise ValueError(f"unsupported stage1 label: {label}")
    return normalized


def build_stage1_train_args(config: dict) -> dict:
    train_args = {
        "data": config["data"],
        "epochs": config.get("epochs", 50),
        "imgsz": config.get("imgsz", 384),
        "batch": config.get("batch", 16),
        "project": config.get("project", "runs/stage1_two_stage"),
        "name": config.get("name", "stage1_patch_classifier"),
        "device": config.get("device", "cpu"),
    }
    excluded_keys = {"model", *train_args.keys()}
    extra_args = {key: value for key, value in config.items() if key not in excluded_keys}
    train_args.update(extra_args)
    return train_args


def stage1_model_name(config: dict) -> str:
    return config["model"]
