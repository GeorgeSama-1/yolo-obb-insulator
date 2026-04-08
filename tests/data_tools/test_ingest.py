from src.data_tools.ingest import find_labelme_pairs


def test_find_labelme_pairs_returns_image_annotation_pairs():
    pairs = find_labelme_pairs("datasets")
    assert len(pairs) == 6
