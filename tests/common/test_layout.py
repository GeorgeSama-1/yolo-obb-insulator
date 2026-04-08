from pathlib import Path


def test_project_layout_exists():
    assert Path("src/common").exists()
    assert Path("src/stage0_obb").exists()
