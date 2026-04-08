from pathlib import Path

from PIL import Image

from src.stage0_obb.visualize import (
    load_ultralytics_obb_detections,
    visualize_prediction_results,
)


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def tolist(self):
        return self._value


class FakeObb:
    def __init__(self, polygons, classes):
        self.xyxyxyxy = FakeTensor(polygons)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, path, polygons, classes):
        self.path = str(path)
        self.obb = FakeObb(polygons, classes)


def test_load_ultralytics_obb_detections_extracts_points_and_labels(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(50, 50, 50)).save(image_path)
    result = FakeResult(image_path, [[[1, 1], [10, 1], [10, 10], [1, 10]]], [0])

    detections = load_ultralytics_obb_detections(result, class_names={0: "insulator"})
    assert detections[0]["label"] == "insulator"
    assert detections[0]["points"][0] == (1, 1)


def test_visualize_prediction_results_writes_overlay_files(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(50, 50, 50)).save(image_path)
    result = FakeResult(image_path, [[[1, 1], [10, 1], [10, 10], [1, 10]]], [0])

    written = visualize_prediction_results(
        [result],
        output_dir=tmp_path / "out",
        class_names={0: "insulator"},
    )
    assert len(written) == 1
    assert Path(written[0]).exists()
