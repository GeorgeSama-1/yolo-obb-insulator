from pathlib import Path
import json
import random


def make_debug_split(items: list[str], seed: int = 42, val_ratio: float = 0.25) -> dict[str, list[str]]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    val_count = max(1, int(round(len(ordered) * val_ratio))) if ordered else 0
    val_items = sorted(ordered[:val_count])
    train_items = sorted(ordered[val_count:])
    return {"train": train_items, "val": val_items}


def write_split_manifest(split: dict[str, list[str]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
