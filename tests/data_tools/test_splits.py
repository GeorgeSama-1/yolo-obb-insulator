from src.data_tools.splits import make_debug_split


def test_make_debug_split_contains_train_and_val_keys():
    split = make_debug_split(["a", "b", "c", "d"])
    assert "train" in split and "val" in split
