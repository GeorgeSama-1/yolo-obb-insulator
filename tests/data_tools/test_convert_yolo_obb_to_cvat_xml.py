from pathlib import Path
import math
import xml.etree.ElementTree as ET

from PIL import Image

from src.data_tools.convert_yolo_obb_to_cvat_xml import (
    export_cvat_xml_from_flat_yolo_obb,
    export_cvat_xml_from_split_root,
    export_cvat_xml_from_yolo_obb,
)


def _rotated_rect_points(
    center: tuple[float, float],
    size: tuple[float, float],
    angle_degrees: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    width, height = size
    radians = math.radians(angle_degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    half_width = width / 2
    half_height = height / 2

    corners = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]

    points = []
    for x_offset, y_offset in corners:
        x = cx + (x_offset * cos_theta - y_offset * sin_theta)
        y = cy + (x_offset * sin_theta + y_offset * cos_theta)
        points.append((x, y))
    return points


def test_export_cvat_xml_from_yolo_obb_writes_rotated_box_annotations(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)

    image_size = (100, 50)
    Image.new("RGB", image_size, color=(20, 20, 20)).save(dataset_root / "images" / "train" / "sample.jpg")
    points = _rotated_rect_points(center=(50.0, 25.0), size=(40.0, 20.0), angle_degrees=30.0)
    normalized = [f"{x / image_size[0]:.6f} {y / image_size[1]:.6f}" for x, y in points]
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        f"0 {' '.join(normalized)}\n",
        encoding="utf-8",
    )

    xml_path = export_cvat_xml_from_yolo_obb(
        dataset_root=dataset_root,
        output_path=tmp_path / "annotations.xml",
        class_names={0: "insulator"},
        split="train",
    )

    root = ET.parse(xml_path).getroot()
    images = root.findall(".//image")
    boxes = root.findall(".//box")
    polygons = root.findall(".//polygon")

    assert len(images) == 1
    assert len(boxes) == 1
    assert len(polygons) == 0
    assert boxes[0].attrib["label"] == "insulator"
    assert boxes[0].attrib["xtl"] == "30.00"
    assert boxes[0].attrib["ytl"] == "15.00"
    assert boxes[0].attrib["xbr"] == "70.00"
    assert boxes[0].attrib["ybr"] == "35.00"
    assert float(boxes[0].attrib["rotation"]) == 30.0


def test_export_cvat_xml_from_yolo_obb_keeps_top_edge_as_rotation_reference(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)

    image_size = (100, 100)
    Image.new("RGB", image_size, color=(20, 20, 20)).save(dataset_root / "images" / "train" / "sample.jpg")
    points = _rotated_rect_points(center=(50.0, 50.0), size=(20.0, 40.0), angle_degrees=13.0)
    normalized = [f"{x / image_size[0]:.6f} {y / image_size[1]:.6f}" for x, y in points]
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        f"0 {' '.join(normalized)}\n",
        encoding="utf-8",
    )

    xml_path = export_cvat_xml_from_yolo_obb(
        dataset_root=dataset_root,
        output_path=tmp_path / "annotations.xml",
        class_names={0: "insulator"},
        split="train",
    )

    box = ET.parse(xml_path).getroot().findall(".//box")[0]

    assert box.attrib["xtl"] == "40.00"
    assert box.attrib["ytl"] == "30.00"
    assert box.attrib["xbr"] == "60.00"
    assert box.attrib["ybr"] == "70.00"
    assert float(box.attrib["rotation"]) == 13.0


def test_export_cvat_xml_from_yolo_obb_normalizes_negative_rotation(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)

    image_size = (100, 100)
    Image.new("RGB", image_size, color=(20, 20, 20)).save(dataset_root / "images" / "train" / "sample.jpg")
    points = _rotated_rect_points(center=(50.0, 50.0), size=(40.0, 20.0), angle_degrees=-17.0)
    normalized = [f"{x / image_size[0]:.6f} {y / image_size[1]:.6f}" for x, y in points]
    (dataset_root / "labels" / "train" / "sample.txt").write_text(
        f"0 {' '.join(normalized)}\n",
        encoding="utf-8",
    )

    xml_path = export_cvat_xml_from_yolo_obb(
        dataset_root=dataset_root,
        output_path=tmp_path / "annotations.xml",
        class_names={0: "insulator"},
        split="train",
    )

    box = ET.parse(xml_path).getroot().findall(".//box")[0]

    assert float(box.attrib["rotation"]) == 343.0


def test_export_cvat_xml_from_flat_yolo_obb_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (120, 80), color=(20, 20, 20)).save(source / "sample.jpg")
    (source / "sample.txt").write_text(
        "0 0.250000 0.250000 0.500000 0.250000 0.500000 0.500000 0.250000 0.500000\n",
        encoding="utf-8",
    )

    xml_path = export_cvat_xml_from_flat_yolo_obb(
        source_dir=source,
        output_path=tmp_path / "annotations.xml",
        class_names={0: "insulator"},
    )

    box = ET.parse(xml_path).getroot().findall(".//box")[0]

    assert box.attrib["label"] == "insulator"
    assert box.attrib["xtl"] == "30.00"
    assert box.attrib["ytl"] == "20.00"
    assert box.attrib["xbr"] == "60.00"
    assert box.attrib["ybr"] == "40.00"


def test_export_cvat_xml_from_split_root_source(tmp_path):
    source = tmp_path / "source"
    (source / "images" / "train").mkdir(parents=True)
    (source / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (120, 80), color=(20, 20, 20)).save(source / "images" / "train" / "sample.jpg")
    (source / "labels" / "train" / "sample.txt").write_text(
        "0 0.250000 0.250000 0.500000 0.250000 0.500000 0.500000 0.250000 0.500000\n",
        encoding="utf-8",
    )

    xml_path = export_cvat_xml_from_split_root(
        source_dir=source,
        output_path=tmp_path / "annotations.xml",
        class_names={0: "insulator"},
        split="train",
    )

    box = ET.parse(xml_path).getroot().findall(".//box")[0]

    assert box.attrib["label"] == "insulator"
    assert box.attrib["xtl"] == "30.00"
    assert box.attrib["ytl"] == "20.00"
    assert box.attrib["xbr"] == "60.00"
    assert box.attrib["ybr"] == "40.00"
