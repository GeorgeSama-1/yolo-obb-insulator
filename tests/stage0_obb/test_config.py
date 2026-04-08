from src.stage0_obb.config import load_stage0_config


def test_load_stage0_config_reads_model_name(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: yolo11n-obb.pt\n", encoding="utf-8")
    data = load_stage0_config(cfg)
    assert data["model"] == "yolo11n-obb.pt"
