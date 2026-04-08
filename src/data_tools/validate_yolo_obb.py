from pathlib import Path


def validate_yolo_obb_label_file(label_path: Path, class_ids: set[int] | None = None) -> list[str]:
    errors: list[str] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 9:
            errors.append(f"{label_path}: line {line_number} must contain 9 values")
            continue

        try:
            class_id = int(values[0])
        except ValueError:
            errors.append(f"{label_path}: line {line_number} has invalid class id")
            continue

        if class_ids is not None and class_id not in class_ids:
            errors.append(f"{label_path}: line {line_number} has unknown class id {class_id}")

        try:
            coords = [float(value) for value in values[1:]]
        except ValueError:
            errors.append(f"{label_path}: line {line_number} contains non-numeric coordinates")
            continue

        for coord in coords:
            if coord < 0.0 or coord > 1.0:
                errors.append(f"{label_path}: line {line_number} coordinate {coord} is outside [0, 1]")
                break
    return errors


def validate_yolo_obb_dataset(dataset_root: str | Path, class_ids: set[int] | None = None) -> list[str]:
    dataset_root = Path(dataset_root)
    errors: list[str] = []

    for split_dir in sorted((dataset_root / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        labels_dir = dataset_root / "labels" / split
        image_files = sorted(path for path in split_dir.iterdir() if path.is_file())
        label_files = sorted(path for path in labels_dir.glob("*.txt")) if labels_dir.exists() else []

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        for stem in sorted(image_stems - label_stems):
            errors.append(f"split={split}: missing label for image {stem}")
        for stem in sorted(label_stems - image_stems):
            errors.append(f"split={split}: missing image for label {stem}")

        for label_path in label_files:
            errors.extend(validate_yolo_obb_label_file(label_path, class_ids=class_ids))
    return errors
