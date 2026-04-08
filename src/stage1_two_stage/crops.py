from pathlib import Path
import json

from PIL import Image


def padded_crop_bounds(
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def save_crop_manifest(records: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def crop_and_save(image_path: str | Path, bounds: tuple[int, int, int, int], output_path: str | Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    crop = image.crop(bounds)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return output
