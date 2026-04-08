from src.common.paths import data_dir, project_root


def test_paths_resolve():
    assert project_root().exists()
    assert data_dir().name == "data"
