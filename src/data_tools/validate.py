from typing import Any


def validate_labelme_shape(shape: dict[str, Any], allowed_labels: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if shape.get("shape_type") != "polygon":
        errors.append("shape_type must be polygon")

    points = shape.get("points", [])
    if len(points) != 4:
        errors.append("polygon must contain exactly four points")

    if allowed_labels is not None and shape.get("label") not in allowed_labels:
        errors.append(f"unknown label: {shape.get('label')}")
    return errors


def validate_labelme_annotation(data: dict[str, Any], allowed_labels: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not data.get("imagePath"):
        errors.append("imagePath is required")
    if not data.get("imageWidth"):
        errors.append("imageWidth is required")
    if not data.get("imageHeight"):
        errors.append("imageHeight is required")

    shapes = data.get("shapes", [])
    if not shapes:
        errors.append("at least one shape is required")

    for index, shape in enumerate(shapes):
        shape_errors = validate_labelme_shape(shape, allowed_labels=allowed_labels)
        errors.extend([f"shape[{index}]: {message}" for message in shape_errors])
    return errors
