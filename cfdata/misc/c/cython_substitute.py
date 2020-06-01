import math
import numpy as np
from collections import Counter

from ..toolkit import is_numeric


def naive_transform_data_with_dicts(data, list_of_idx, list_of_dict):
    assert len(list_of_idx) == len(list_of_dict)
    if len(list_of_idx) == 0:
        return data
    new_data_samples = []
    for sample in data:
        for d, idx in zip(list_of_dict, list_of_idx):
            elem = float(sample[idx])
            if math.isnan(elem):
                sample[idx] = d.get("nan", 0)
            else:
                sample[idx] = d.get(elem, 0)
        new_data_samples.append(sample)
    return np.vstack(new_data_samples)


def naive_get_counter_from_arr(arr):
    return Counter(arr)


def naive_is_all_numeric(arr):
    return all(map(is_numeric, arr))


def naive_flat_str_to_float32(arr):
    return np.asarray(arr, np.float32)


__all__ = [
    "naive_transform_data_with_dicts", "naive_get_counter_from_arr",
    "naive_is_all_numeric", "naive_flat_str_to_float32"
]
