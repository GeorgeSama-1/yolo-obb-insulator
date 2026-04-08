from src.stage1_two_stage.crops import padded_crop_bounds


def test_padded_crop_bounds_expands_box_within_image():
    bounds = padded_crop_bounds((10, 10, 20, 20), image_size=(100, 100), padding=4)
    assert bounds[0] >= 0
    assert bounds[1] >= 0
