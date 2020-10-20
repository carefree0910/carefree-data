try:
    from .cython_utils import *
except ImportError:
    raise


def _to_list(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return arr


def c_transform_flat_data_with_dict(flat_data, transform_dict):
    return transform_flat_data_with_dict(flat_data, transform_dict)


def c_is_all_numeric(arr):
    return is_all_numeric(_to_list(arr))


def c_flat_arr_to_float32(arr):
    return flat_arr_to_float32(_to_list(arr))


__all__ = [
    "c_transform_flat_data_with_dict",
    "c_is_all_numeric",
    "c_flat_arr_to_float32",
]
