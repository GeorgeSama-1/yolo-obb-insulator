from src.data_tools.convert_labelme_to_yolo_obb import convert_shape_to_yolo_obb_line


def test_convert_shape_to_yolo_obb_line_emits_class_and_eight_coordinates():
    shape = {"label": "insulator", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    line = convert_shape_to_yolo_obb_line(
        shape,
        image_width=20,
        image_height=20,
        class_to_id={"insulator": 0},
    )
    assert len(line.split()) == 9
