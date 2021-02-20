import math
import numpy as np

from typing import Any
from typing import Dict
from cftool.misc import is_numeric


def naive_transform_flat_data_with_dict(
    flat_data: np.ndarray,
    transform_dict: Dict[Any, Any],
    need_truncate: bool,
) -> np.ndarray:
    nan_value = transform_dict.get("nan", 0)
    oob_value = nan_value if need_truncate else 0
    for i, elem in enumerate(flat_data):
        elem = float(elem)
        if math.isnan(elem):
            flat_data[i] = nan_value
        else:
            flat_data[i] = transform_dict.get(elem, oob_value)
    return flat_data


def naive_is_all_numeric(array: np.ndarray) -> bool:
    return all(map(is_numeric, array))


def naive_flat_arr_to_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, np.float32)


__all__ = [
    "naive_transform_flat_data_with_dict",
    "naive_is_all_numeric",
    "naive_flat_arr_to_float32",
]
