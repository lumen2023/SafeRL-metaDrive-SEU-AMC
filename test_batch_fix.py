#!/usr/bin/env python3
"""Small repro for the FastCollector partial-reset info merge fix."""

import numpy as np
from tianshou.data import Batch

def _split_info_items(info, expected_count):
    if isinstance(info, Batch):
        try:
            return [info[i] for i in range(expected_count)]
        except TypeError:
            if expected_count == 1:
                return [info]
            raise

    if isinstance(info, dict):
        if expected_count == 1:
            return [Batch(info)]
        return [
            Batch({key: value[i] for key, value in info.items()})
            for i in range(expected_count)
        ]

    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.tolist()

    if isinstance(info, (list, tuple)):
        return [item if isinstance(item, Batch) else Batch(item) for item in info]

    if expected_count == 1:
        return [Batch(info)]

    raise TypeError(f"Unsupported info type: {type(info)}")


def _merge_reset_info(existing_info, local_ids, info, active_count):
    current_items = _split_info_items(existing_info, active_count)
    incoming_items = _split_info_items(info, len(local_ids))
    for offset, local_id in enumerate(np.asarray(local_ids, dtype=int).tolist()):
        current_items[local_id] = incoming_items[offset]
    return Batch.stack(current_items)


def test_batch_info_assignment():
    """Test that the partial-reset merge tolerates new keys and list/dict info."""
    print("Testing Batch info merge...")

    active_count = 3
    local_ids = np.array([0, 2])

    initial_info = {"cost": np.array([0.0, 0.1, 0.2]), "other": np.array([1, 2, 3])}
    print(f"Initial info: {initial_info}")

    dict_info = {"cost": np.array([0.0, 0.0]), "other": np.array([10, 30])}
    merged_dict = _merge_reset_info(initial_info, local_ids, dict_info, active_count)
    print(f"Merged dict-style info: {merged_dict}")

    list_info = [
        {"cost": 0.0, "new_key": 7},
        {"cost": 0.0, "new_key": 9},
    ]
    merged_list = _merge_reset_info(initial_info, local_ids, list_info, active_count)
    print(f"Merged list-style info with new key: {merged_list}")

    assert np.allclose(merged_dict.cost, np.array([0.0, 0.1, 0.0]))
    assert np.array_equal(merged_dict.other, np.array([10, 2, 30]))
    assert np.allclose(merged_list.cost, np.array([0.0, 0.1, 0.0]))
    assert np.array_equal(merged_list.new_key, np.array([7, 0, 9]))

    active_info = Batch(
        cost=np.arange(19, dtype=np.float32),
        other=np.arange(100, 119, dtype=np.int64),
    )
    active_local_ids = np.array([0, 18])
    active_reset_info = {"cost": np.array([0.0, 0.0]), "other": np.array([500, 900])}
    merged_active = _merge_reset_info(active_info, active_local_ids, active_reset_info, active_count=19)
    print(f"Merged active-view info: {merged_active}")

    assert np.allclose(merged_active.cost[[0, 18]], np.array([0.0, 0.0]))
    assert np.array_equal(merged_active.other[[0, 18]], np.array([500, 900]))
    return True


if __name__ == "__main__":
    success = test_batch_info_assignment()
    if success:
        print("\n✓ Test passed! The fix works correctly.")
    else:
        print("\n✗ Test failed!")
