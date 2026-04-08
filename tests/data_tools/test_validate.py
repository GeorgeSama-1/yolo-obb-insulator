from src.data_tools.validate import validate_labelme_shape


def test_validate_labelme_shape_accepts_four_point_polygon():
    shape = {
        "label": "insulator",
        "shape_type": "polygon",
        "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }
    assert validate_labelme_shape(shape) == []
