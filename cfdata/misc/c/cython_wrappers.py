from collections import Counter

try:
    from .cython_utils import *
except ImportError:
    raise


def _to_list(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return arr


def c_transform_data_with_dicts(data, list_of_idx, list_of_dict):
    return transform_data_with_dicts(data, list_of_idx, list_of_dict)


def c_get_counter_from_arr(arr):
    return Counter(counter(_to_list(arr)))


def c_is_all_numeric(arr):
    return is_all_numeric(_to_list(arr))


def c_flat_str_to_float32(arr):
    return flat_str_to_float32(_to_list(arr))


__all__ = [
    "c_transform_data_with_dicts", "c_get_counter_from_arr",
    "c_is_all_numeric", "c_flat_str_to_float32"
]
