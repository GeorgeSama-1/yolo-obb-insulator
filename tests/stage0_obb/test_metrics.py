from src.stage0_obb.metrics import compute_count_metrics


def test_compute_count_metrics_reports_mae_and_exact_accuracy():
    metrics = compute_count_metrics([3, 4, 5], [3, 5, 4])
    assert "mae" in metrics
    assert "exact_accuracy" in metrics
