from dataclasses import dataclass
from pathlib import Path
import shutil


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LabelMePair:
    stem: str
    image_path: Path
    annotation_path: Path


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def find_labelme_pairs(source_dir: str | Path) -> list[LabelMePair]:
    root = Path(source_dir)
    annotations = {path.stem: path for path in sorted(root.glob("*.json"))}
    pairs: list[LabelMePair] = []

    for image_path in sorted(root.iterdir()):
        if not image_path.is_file() or not _is_image(image_path):
            continue
        annotation_path = annotations.get(image_path.stem)
        if annotation_path is None:
            raise FileNotFoundError(f"Missing annotation for image: {image_path.name}")
        pairs.append(
            LabelMePair(
                stem=image_path.stem,
                image_path=image_path,
                annotation_path=annotation_path,
            )
        )

    image_stems = {pair.stem for pair in pairs}
    extra_annotations = sorted(set(annotations) - image_stems)
    if extra_annotations:
        missing = ", ".join(extra_annotations)
        raise FileNotFoundError(f"Missing image for annotations: {missing}")
    return pairs


def stage_labelme_dataset(source_dir: str | Path, target_dir: str | Path) -> list[LabelMePair]:
    pairs = find_labelme_pairs(source_dir)
    target_root = Path(target_dir)
    images_dir = target_root / "images"
    annotations_dir = target_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        shutil.copy2(pair.image_path, images_dir / pair.image_path.name)
        shutil.copy2(pair.annotation_path, annotations_dir / pair.annotation_path.name)
    return pairs
