VALID_STAGE1_LABELS = {"normal", "abnormal"}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in VALID_STAGE1_LABELS:
        raise ValueError(f"unsupported stage1 label: {label}")
    return normalized
