from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def reports_dir() -> Path:
    return project_root() / "reports"


def runs_dir() -> Path:
    return project_root() / "runs"


def incoming_batches_dir() -> Path:
    return project_root() / "incoming_batches"


def prelabel_batches_dir() -> Path:
    return project_root() / "prelabel_batches"


def cvat_exports_dir() -> Path:
    return project_root() / "cvat_exports"


def dataset_pool_dir(task_name: str) -> Path:
    return project_root() / f"datasets_pool_{task_name}"


def processed_stage_dir(stage_name: str) -> Path:
    return data_dir() / "processed" / stage_name
