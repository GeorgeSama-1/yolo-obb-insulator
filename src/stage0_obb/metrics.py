from pathlib import Path
import json


def compute_count_metrics(ground_truth_counts: list[int], predicted_counts: list[int]) -> dict[str, float]:
    if len(ground_truth_counts) != len(predicted_counts):
        raise ValueError("ground truth and prediction counts must have the same length")
    if not ground_truth_counts:
        return {"mae": 0.0, "exact_accuracy": 0.0}

    absolute_errors = [abs(gt - pred) for gt, pred in zip(ground_truth_counts, predicted_counts)]
    exact = [1 for gt, pred in zip(ground_truth_counts, predicted_counts) if gt == pred]
    return {
        "mae": sum(absolute_errors) / len(absolute_errors),
        "exact_accuracy": sum(exact) / len(ground_truth_counts),
    }


def write_metrics_report(metrics: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
