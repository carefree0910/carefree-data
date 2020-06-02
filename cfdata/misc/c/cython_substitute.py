import math
import numpy as np
from collections import Counter

from ..toolkit import is_numeric


def naive_transform_flat_data_with_dict(flat_data, transform_dict):
    for i, elem in enumerate(flat_data):
        elem = float(elem)
        if math.isnan(elem):
            flat_data[i] = transform_dict.get("nan", 0)
        else:
            flat_data[i] = transform_dict.get(elem, 0)
    return flat_data


def naive_is_all_numeric(arr):
    return all(map(is_numeric, arr))


def naive_flat_arr_to_float32(arr):
    return np.asarray(arr, np.float32)


__all__ = [
    "naive_transform_flat_data_with_dict",
    "naive_is_all_numeric", "naive_flat_arr_to_float32"
]
