VALID_STAGE2_LABELS = {"normal", "abnormal"}


def normalize_stage2_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in VALID_STAGE2_LABELS:
        raise ValueError(f"unsupported stage2 label: {label}")
    return normalized
